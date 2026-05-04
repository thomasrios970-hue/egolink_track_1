"""
使用 openai-whisper 自动生成 WebVTT 字幕文件。

执行逻辑：
1. 读取 config.PATH_TO_AUDIO_MANIFEST，manifest 需要包含 Vid、sub_id 和 audio_path 字段。
2. 使用 openai-whisper 对每条音频做自动语音识别，得到带 start/end/text 的 segments。
3. 将结果保存为 Path(config.PATH_TO_PROCESSED_DIR) / "whisper" / model_name / Vid / f"{sub_id}.vtt"。
4. 输出格式保持和 data/subtext 下的人工字幕一致，以 WEBVTT 开头，并用空行分隔每个字幕段。
5. 如果目标 .vtt 已存在则自动跳过。

运行示例：
python code/proceess_data/process_whisper.py \
  --model-name small \
  --gpu 4

如果没有安装 openai-whisper，请先运行：
python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -U openai-whisper
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

DEFAULT_MODEL_NAME = "small"
MODEL_NAMES = ["tiny", "base", "small", "medium", "large"]
SUPPORTED_GPUS = [4, 5]
DEFAULT_GPU = 4
DEFAULT_DOWNLOAD_ROOT = Path(config.PATH_TO_HUGGINGFACE_MODEL) / "whisper"
PROGRESS_INTERVAL = 10


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="使用 openai-whisper 从音频生成 WebVTT 字幕。",
    )
    parser.add_argument(
        "--model-name",
        choices=MODEL_NAMES,
        default=DEFAULT_MODEL_NAME,
        help="Whisper 模型名称，例如 tiny/base/small/medium/large。",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        choices=SUPPORTED_GPUS,
        default=DEFAULT_GPU,
        help="使用的 GPU 卡号，只允许 4 或 5。",
    )
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=PROGRESS_INTERVAL,
        help="每处理多少条 manifest 记录打印一次进度。",
    )
    return parser


def format_vtt_time(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def clean_segment_text(text):
    return " ".join(str(text).split())


def segments_to_vtt(segments):
    lines = ["WEBVTT", ""]
    for segment in segments:
        text = clean_segment_text(segment.get("text", ""))
        if not text:
            continue

        start = format_vtt_time(segment.get("start", 0.0))
        end = format_vtt_time(segment.get("end", 0.0))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def load_whisper_model(model_name, gpu):
    try:
        import whisper
    except ImportError as e:
        raise RuntimeError(
            "openai-whisper is not installed. Please run: pip install openai-whisper"
        ) from e

    DEFAULT_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    return whisper.load_model(
        model_name,
        device=f"cuda:{gpu}",
        download_root=str(DEFAULT_DOWNLOAD_ROOT),
    )


def transcribe_audio(model, audio_path):
    return model.transcribe(
        str(audio_path),
        verbose=False,
        fp16=False,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    import pandas as pd

    manifest_path = Path(config.PATH_TO_AUDIO_MANIFEST)
    whisper_root = Path(config.PATH_TO_PROCESSED_DIR) / "whisper" / args.model_name

    df = pd.read_csv(manifest_path)
    required_columns = {"Vid", "sub_id", "audio_path"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"manifest missing columns: {sorted(missing_columns)}")

    whisper_root.mkdir(parents=True, exist_ok=True)

    print("manifest:", manifest_path)
    print("whisper_root:", whisper_root)
    print("model_name:", args.model_name)
    print("gpu:", args.gpu)
    print("download_root:", DEFAULT_DOWNLOAD_ROOT)

    model = load_whisper_model(args.model_name, args.gpu)

    total = len(df)
    missing = 0
    skipped = 0
    success = 0
    failed = 0

    for idx, row in df.iterrows():
        vid = str(row["Vid"])
        sub_id = str(row["sub_id"])
        audio_path = Path(row["audio_path"])
        save_path = whisper_root / vid / f"{sub_id}.vtt"

        if not audio_path.exists():
            missing += 1
            continue

        if save_path.exists():
            skipped += 1
            continue

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            result = transcribe_audio(model, audio_path)
            vtt_text = segments_to_vtt(result.get("segments", []))
            save_path.write_text(vtt_text, encoding="utf-8")
            success += 1
        except Exception as e:
            failed += 1
            print(f"failed: {audio_path}, {e}")

        if (idx + 1) % args.progress_interval == 0:
            print(
                f"[{idx + 1}/{total}] "
                f"success={success}, skipped={skipped}, "
                f"missing={missing}, failed={failed}"
            )

    print("Done")
    print(f"total={total}")
    print(f"success={success}")
    print(f"skipped={skipped}")
    print(f"missing={missing}")
    print(f"failed={failed}")


if __name__ == "__main__":
    main()
