"""
从视频中均匀抽取图片帧并保存到 data/processed。

执行逻辑：
1. 从 config.PATH_TO_VIDEO_MANIFEST 读取 manifest，要求包含 Vid、sub_id 和 video_path 字段。
2. 对每个视频按照 --num-frames 在整段视频中均匀抽取固定数量的帧；--num-frames all 会依次处理 8、16、32 帧。
3. 默认将图片保存到 Path(config.PATH_TO_PROCESSED_DIR) / f"frames_{num_frames}"。
4. 每个视频保存到 frames_{num_frames}/{Vid}/{sub_id}/ 目录下，并写入 metadata.json 记录原视频帧号。
5. 如果目标目录中已有足够数量的帧图片，则自动跳过。

运行示例：
python code/proceess_data/process_video.py \
  --num-frames all \
  --image-format jpg \
  --progress-interval 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

FRAME_COUNTS = [8, 16, 32]
FRAME_COUNT_CHOICES = [str(num_frames) for num_frames in FRAME_COUNTS] + ["all"]
IMAGE_FORMAT = "jpg"
JPEG_QUALITY = 100
PROGRESS_INTERVAL = 10


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="从 manifest 中的视频文件均匀抽取图片帧并保存到 data/processed。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(config.PATH_TO_VIDEO_MANIFEST),
        help="manifest CSV 路径，需包含 Vid、sub_id 和 video_path 字段。",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(config.PATH_TO_PROCESSED_DIR),
        help="processed 输出根目录；最终会拼接 frames_<num_frames>。",
    )
    parser.add_argument(
        "--num-frames",
        choices=FRAME_COUNT_CHOICES,
        default="8",
        help="每个视频均匀抽取的图片帧数量；可选 8、16、32 或 all。",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default=IMAGE_FORMAT,
        help="保存图片格式。",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=positive_int,
        default=JPEG_QUALITY,
        help="jpg 保存质量，image-format=jpg 时生效。",
    )
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=PROGRESS_INTERVAL,
        help="每处理多少条 manifest 记录打印一次进度。",
    )
    return parser


def get_target_frame_counts(num_frames):
    return FRAME_COUNTS if num_frames == "all" else [int(num_frames)]


def get_output_root(processed_root, num_frames):
    return processed_root / f"frames_{num_frames}"


def get_sample_dir(output_root, vid, sub_id):
    return output_root / str(vid) / str(sub_id)


def has_existing_frames(sample_dir, num_frames, image_format):
    if not sample_dir.exists():
        return False
    return len(list(sample_dir.glob(f"*.{image_format}"))) >= num_frames


def sample_indices(total_frames, num_frames, np):
    if total_frames <= 0:
        raise ValueError("cannot read video frame count")
    return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()


def write_metadata(sample_dir, row, video_path, total_frames, indices, num_frames, image_format):
    metadata = {
        "Vid": str(row["Vid"]),
        "sub_id": str(row["sub_id"]),
        "video_path": str(video_path),
        "total_frames": int(total_frames),
        "num_frames": int(num_frames),
        "sampled_indices": [int(index) for index in indices],
        "image_format": image_format,
    }
    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_video_frames(video_path, sample_dir, num_frames, image_format, jpeg_quality, cv2, np):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("cannot read video frames")

    indices = sample_indices(total_frames, num_frames, np)
    saved = 0

    for frame_order, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()

        if not ok:
            continue

        image_path = sample_dir / f"frame_{frame_order:03d}.{image_format}"
        if image_format == "jpg":
            ok = cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
        else:
            ok = cv2.imwrite(str(image_path), frame)

        if ok:
            saved += 1

    cap.release()

    if saved == 0:
        raise ValueError("no valid frames saved")

    return total_frames, indices, saved


def process_frame_count(df, args, num_frames, cv2, np):
    output_root = get_output_root(args.processed_root, num_frames)
    output_root.mkdir(parents=True, exist_ok=True)
    print("output_root:", output_root)

    missing = 0
    skipped = 0
    success = 0
    failed = 0

    for idx, row in df.iterrows():
        vid = str(row["Vid"])
        sub_id = str(row["sub_id"])
        video_path = Path(row["video_path"])
        sample_dir = get_sample_dir(output_root, vid, sub_id)

        if not video_path.exists():
            missing += 1
            continue

        if has_existing_frames(sample_dir, num_frames, args.image_format):
            skipped += 1
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            total_frames, indices, saved = save_video_frames(
                video_path,
                sample_dir,
                num_frames,
                args.image_format,
                args.jpeg_quality,
                cv2,
                np,
            )
            if saved < num_frames:
                failed += 1
                continue
            write_metadata(
                sample_dir,
                row,
                video_path,
                total_frames,
                indices,
                num_frames,
                args.image_format,
            )
            success += 1
        except Exception as e:
            failed += 1
            print(f"failed: {video_path}, {e}")

        if (idx + 1) % args.progress_interval == 0:
            print(
                f"frames={num_frames} [{idx + 1}/{len(df)}] "
                f"success={success}, skipped={skipped}, "
                f"missing={missing}, failed={failed}"
            )

    print(
        f"frames={num_frames} done: "
        f"success={success}, skipped={skipped}, "
        f"missing={missing}, failed={failed}"
    )
    return {
        "total": len(df),
        "success": success,
        "skipped": skipped,
        "missing": missing,
        "failed": failed,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    import cv2
    import numpy as np
    import pandas as pd

    df = pd.read_csv(args.manifest)
    required_columns = {"Vid", "sub_id", "video_path"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"manifest missing columns: {sorted(missing_columns)}")

    target_frame_counts = get_target_frame_counts(args.num_frames)
    print("manifest:", args.manifest)
    print("processed_root:", args.processed_root)
    print("target_frame_counts:", target_frame_counts)

    summaries = [
        process_frame_count(df, args, num_frames, cv2, np)
        for num_frames in target_frame_counts
    ]

    print("Done")
    print(f"total={sum(summary['total'] for summary in summaries)}")
    print(f"success={sum(summary['success'] for summary in summaries)}")
    print(f"skipped={sum(summary['skipped'] for summary in summaries)}")
    print(f"missing={sum(summary['missing'] for summary in summaries)}")
    print(f"failed={sum(summary['failed'] for summary in summaries)}")


if __name__ == "__main__":
    main()
