"""
提取音频的 HuBERT 特征。

执行逻辑：
1. 读取 manifest CSV，其中每一行需要包含 Vid、sub_id 和 audio_path 字段。
2. 按 audio_path 读取音频；如果音频是多声道，则先转为单声道。
3. 使用本地 Wav2Vec2FeatureExtractor 对音频做 HuBERT 输入预处理。
4. 使用本地 HubertModel 提取音频帧级特征。
5. 对时间维度取均值，并保存为 {Vid}_{sub_id}.npy。

运行示例：
python code/extract_features/extract_hubert_features.py \
  --cuda-visible-devices 4 \
  --manifest /data/wzw/egolink_race/data/manifest/manifest.csv \
  --model-dir /data/wzw/egolink_race/model/chinese-hubert-large \
  --save-root /data/wzw/egolink_race/feature/hubert_large
"""

import argparse
import os
from pathlib import Path

DEFAULT_MANIFEST = Path("/data/wzw/egolink_race/data/manifest/manifest.csv")
DEFAULT_MODEL_DIR = Path("/data/wzw/egolink_race/model/chinese-hubert-large")
DEFAULT_SAVE_ROOT = Path("/data/wzw/egolink_race/feature/hubert_large")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="从 manifest 中的音频文件提取 HuBERT 特征并保存为 .npy 文件。",
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
        help="manifest CSV 路径，需包含 Vid、sub_id、audio_path 字段。",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="本地 HubertModel 模型目录。",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=DEFAULT_SAVE_ROOT,
        help="特征 .npy 文件输出目录。",
    )
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=200,
        help="每处理多少条 manifest 记录打印一次进度。",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    model = HubertModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    df = pd.read_csv(args.manifest)

    success = 0
    skipped = 0
    failed = 0

    args.save_root.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        vid = str(row["Vid"])
        sub_id = str(row["sub_id"])
        audio_path = Path(row["audio_path"])
        save_path = args.save_root / f"{vid}_{sub_id}.npy"

        if save_path.exists():
            skipped += 1
            continue

        try:
            audio, sr = sf.read(audio_path)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            inputs = processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            feat = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            np.save(save_path, feat)

            success += 1

        except Exception as e:
            failed += 1
            print("failed:", audio_path, e)

        if (idx + 1) % args.progress_interval == 0:
            print(f"[{idx+1}/{len(df)}] success={success}, skipped={skipped}, failed={failed}")

    print("Done")
    print("success:", success)
    print("skipped:", skipped)
    print("failed:", failed)


if __name__ == "__main__":
    main()
