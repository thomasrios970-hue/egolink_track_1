"""
提取视频的 CLIP 视觉特征。

执行逻辑：
1. 读取 manifest CSV，其中每一行需要包含 Vid、sub_id 和 video_path 字段。
2. 按 video_path 打开视频，并在整段视频中均匀抽取指定数量的帧。
3. 将抽到的帧 resize 到指定尺寸，按 CLIP 的均值和方差做归一化预处理。
4. 使用本地 CLIPVisionModel 对每帧提取视觉特征。
5. 对同一个视频的多帧特征取均值，并保存为 {Vid}_{sub_id}.npy。

运行示例：
python code/extract_features/extract_clip_features.py \
  --cuda-visible-devices 4 \
  --manifest /data/wzw/egolink_race/data/manifest/manifest.csv \
  --model-dir /data/wzw/egolink_race/model/clip-vit-large-patch14 \
  --save-root /data/wzw/egolink_race/feature/clip_large \
  --num-frames 8 \
  --image-size 224
"""

import argparse
import os
from pathlib import Path

DEFAULT_MANIFEST = Path("/data/wzw/egolink_race/data/manifest/manifest.csv")
DEFAULT_MODEL_DIR = Path("/data/wzw/egolink_race/model/clip-vit-large-patch14")
DEFAULT_SAVE_ROOT = Path("/data/wzw/egolink_race/feature/clip_large")

CLIP_MEAN_VALUES = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD_VALUES = (0.26862954, 0.26130258, 0.27577711)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="从 manifest 中的视频文件提取 CLIP 视觉特征并保存为 .npy 文件。",
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
        help="manifest CSV 路径，需包含 Vid、sub_id、video_path 字段。",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="本地 CLIPVisionModel 模型目录。",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=DEFAULT_SAVE_ROOT,
        help="特征 .npy 文件输出目录。",
    )
    parser.add_argument(
        "--num-frames",
        type=positive_int,
        default=8,
        help="每个视频均匀抽取的帧数。",
    )
    parser.add_argument(
        "--image-size",
        type=positive_int,
        default=224,
        help="送入 CLIP 前的正方形图像尺寸。",
    )
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=200,
        help="每处理多少条 manifest 记录打印一次进度。",
    )
    return parser


def sample_frames(video_path, num_frames, image_size, cv2, np, torch, clip_mean, clip_std):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError("cannot read video frames")

    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - clip_mean) / clip_std
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("no valid frames")

    frames = np.stack(frames, axis=0)
    frames = np.transpose(frames, (0, 3, 1, 2))
    return torch.from_numpy(frames)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from transformers import CLIPVisionModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    clip_mean = np.array(CLIP_MEAN_VALUES, dtype=np.float32)
    clip_std = np.array(CLIP_STD_VALUES, dtype=np.float32)

    args.save_root.mkdir(parents=True, exist_ok=True)

    model = CLIPVisionModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    df = pd.read_csv(args.manifest)

    success = 0
    skipped = 0
    failed = 0

    for idx, row in df.iterrows():
        vid = str(row["Vid"])
        sub_id = str(row["sub_id"])
        video_path = Path(row["video_path"])
        save_path = args.save_root / f"{vid}_{sub_id}.npy"

        if save_path.exists():
            skipped += 1
            continue

        try:
            pixel_values = sample_frames(
                video_path,
                args.num_frames,
                args.image_size,
                cv2,
                np,
                torch,
                clip_mean,
                clip_std,
            ).to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            feat = outputs.pooler_output.mean(dim=0).cpu().numpy()
            np.save(save_path, feat)

            success += 1

        except Exception as e:
            failed += 1
            print("failed:", video_path, e)

        if (idx + 1) % args.progress_interval == 0:
            print(f"[{idx+1}/{len(df)}] success={success}, skipped={skipped}, failed={failed}")

    print("Done")
    print("success:", success)
    print("skipped:", skipped)
    print("failed:", failed)


if __name__ == "__main__":
    main()
