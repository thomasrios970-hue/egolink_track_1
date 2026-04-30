"""
提取字幕文本的 MacBERT 特征。

执行逻辑：
1. 读取 manifest CSV，其中每一行需要包含 Vid、sub_id 和 subtitle_path 字段。
2. 按 subtitle_path 读取 WebVTT 字幕，过滤 WEBVTT 标记、时间轴和序号行。
3. 使用本地 AutoTokenizer 对字幕文本分词、截断和 padding。
4. 使用本地 AutoModel 提取文本表示。
5. 取 [CLS] 位置的向量，并保存为 {Vid}_{sub_id}.npy。

运行示例：
python code/extract_features/extract_macbert_features.py \
  --cuda-visible-devices 4 \
  --manifest /data/wzw/egolink_race/data/manifest/manifest.csv \
  --model-dir /data/wzw/egolink_race/model/chinese-macbert-large \
  --save-root /data/wzw/egolink_race/feature/macbert_large \
  --max-length 256
"""

import argparse
import os
import re
from pathlib import Path

DEFAULT_MANIFEST = Path("/data/wzw/egolink_race/data/manifest/manifest.csv")
DEFAULT_MODEL_DIR = Path("/data/wzw/egolink_race/model/chinese-macbert-large")
DEFAULT_SAVE_ROOT = Path("/data/wzw/egolink_race/feature/macbert_large")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="从 manifest 中的字幕文件提取 MacBERT 文本特征并保存为 .npy 文件。",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="设置 CUDA_VISIBLE_DEVICES，例如 4；不传则使用当前环境变量。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest CSV 路径，需包含 Vid、sub_id、subtitle_path 字段。",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="本地 MacBERT 模型目录。",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=DEFAULT_SAVE_ROOT,
        help="特征 .npy 文件输出目录。",
    )
    parser.add_argument(
        "--max-length",
        type=positive_int,
        default=256,
        help="字幕文本 tokenizer 的最大长度。",
    )
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=500,
        help="每处理多少条 manifest 记录打印一次进度。",
    )
    return parser


def read_vtt(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "WEBVTT":
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        lines.append(line)
    return " ".join(lines)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    args.save_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    df = pd.read_csv(args.manifest)

    success = 0
    skipped = 0
    failed = 0

    for idx, row in df.iterrows():
        vid = str(row["Vid"])
        sub_id = str(row["sub_id"])
        subtitle_path = Path(row["subtitle_path"])
        save_path = args.save_root / f"{vid}_{sub_id}.npy"

        if save_path.exists():
            skipped += 1
            continue

        try:
            text = read_vtt(subtitle_path)
            if not text:
                text = "[PAD]"

            inputs = tokenizer(
                text,
                max_length=args.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            np.save(save_path, feat)

            success += 1

        except Exception as e:
            failed += 1
            print("failed:", subtitle_path, e)

        if (idx + 1) % args.progress_interval == 0:
            print(f"[{idx+1}/{len(df)}] success={success}, skipped={skipped}, failed={failed}")

    print("Done")
    print("success:", success)
    print("skipped:", skipped)
    print("failed:", failed)


if __name__ == "__main__":
    main()
