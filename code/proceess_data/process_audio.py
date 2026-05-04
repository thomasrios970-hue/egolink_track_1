"""
从音频、字幕和 data2 情绪标注中生成每个视频的 audio_json 预处理文件。

执行逻辑：
1. 读取 config.PATH_TO_AUDIO_MANIFEST，manifest 需要包含 Vid、sub_id、audio_path 和 subtitle_path 字段。
2. 优先从 subtitle_path 解析 WebVTT 字幕；如果字幕不存在或解析为空，再用 openai-whisper 从 audio_path 生成 segments。
3. 从 audio_path 读取 wav 音频，计算整段音量、语速、静音比例、高音和笑声等启发式特征。
4. 从同目录的 emotion_sort_manifest.csv 读取 data2 情绪标注，并按 Vid/sub_id 写入 tone_hint 相关字段。
5. 在顶层 JSON 中新增 clips：有 data2 时按 data2 的 start_time/end_time 切小段；没有 data2 时按字幕/ASR segments 切小段。
6. 将每个视频保存为 Path(config.PATH_TO_PROCESSED_DIR) / "audio_json" / Vid / f"{sub_id}.json"。
7. 如果目标 json 已存在且字段完整则自动跳过；旧格式缺 clips 时会自动补生成。

运行示例：
python code/proceess_data/process_audio.py

如果需要 Whisper fallback 但未安装 openai-whisper，请先运行：
pip install openai-whisper
"""

import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

PROGRESS_INTERVAL = 10
WHISPER_MODEL_NAME = "small"
WHISPER_GPU = 4
WHISPER_DOWNLOAD_ROOT = Path(config.PATH_TO_HUGGINGFACE_MODEL) / "whisper"
LAUGHTER_KEYWORDS = (
    "laugh",
    "laughs",
    "laughed",
    "laughing",
    "laughter",
    "haha",
    "hehe",
    "lol",
    "giggle",
    "giggles",
    "chuckle",
    "chuckles",
    "哈哈",
    "呵呵",
    "笑",
)

# AUDIO_JSON_FIELDS 是输出字段注册表。后续要增删 json 字段时，优先改这里。
# speech_text: 整段字幕文本；segments: 带 start/end/text 的字幕片段。
# volume_mean: 平均音量；speech_rate: 每秒词数；laughter: 是否检测到笑声线索。
# silence_ratio: 静音比例；high_pitch: 是否偏高音；tone_hint: data2 主情绪。
# tone_hints: data2 去重情绪列表；emotion_annotations: data2 原始情绪标注摘要。
# clips: 更小时间段列表，用于后续按时间片做多模态对齐。
AUDIO_JSON_FIELDS = [
    "speech_text",
    "segments",
    "volume_mean",
    "speech_rate",
    "laughter",
    "silence_ratio",
    "high_pitch",
    "tone_hint",
    "tone_hints",
    "emotion_annotations",
    "clips",
]

# CLIP_JSON_FIELDS 是每个 clips[i] 的字段注册表，后续 clip 字段增删只改这里。
# video_id/sub_id/clip_id: clip 身份信息；start_time/end_time: clip 时间边界。
# person: data2 标注人物，没有标注时为 unknown。
# speech_text/segments: 与该 clip 时间有交集的字幕或 Whisper ASR 文本。
# tone_hint/tone_hints/emotion_annotations: 该 clip 对应的 data2 情绪信息。
# volume_mean/speech_rate/laughter/silence_ratio/high_pitch: 该 clip 的音频统计特征。
CLIP_JSON_FIELDS = [
    "video_id",
    "sub_id",
    "clip_id",
    "start_time",
    "end_time",
    "person",
    "speech_text",
    "segments",
    "tone_hint",
    "tone_hints",
    "emotion_annotations",
    "volume_mean",
    "speech_rate",
    "laughter",
    "silence_ratio",
    "high_pitch",
]

TIME_LINE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}(?::\d{2})?\.\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}(?::\d{2})?\.\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]")
_WHISPER_MODEL = None


def parse_vtt_time(value):
    parts = value.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"unsupported vtt timestamp: {value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_subtitle_text(text):
    text = html.unescape(text)
    text = TAG_RE.sub("", text)
    return " ".join(text.split())


def read_vtt_segments(path):
    if path is None or is_missing_value(path):
        return []

    path = Path(str(path))
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    segments = []
    current_start = None
    current_end = None
    current_text = []

    def flush_segment():
        if current_start is None:
            return
        text = clean_subtitle_text(" ".join(current_text))
        if not text:
            return
        segments.append({
            "start": round(float(current_start), 3),
            "end": round(float(current_end), 3),
            "text": text,
        })

    for raw_line in lines:
        line = raw_line.strip()
        match = TIME_LINE_RE.match(line)
        if match:
            flush_segment()
            current_start = parse_vtt_time(match.group("start"))
            current_end = parse_vtt_time(match.group("end"))
            current_text = []
            continue

        if not line:
            flush_segment()
            current_start = None
            current_end = None
            current_text = []
            continue

        if line == "WEBVTT" or line.startswith(("NOTE", "STYLE", "REGION")):
            continue

        if current_start is not None:
            current_text.append(line)

    flush_segment()
    return segments


def is_missing_value(value):
    if value is None:
        return True
    try:
        if value != value:
            return True
    except Exception:
        pass
    return not str(value).strip()


def get_whisper_device():
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{WHISPER_GPU}"
    except Exception:
        pass
    return "cpu"


def load_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    try:
        import whisper
    except ImportError as e:
        raise RuntimeError(
            "字幕不存在或解析为空，需要使用 openai-whisper fallback。"
            "请先运行: pip install openai-whisper"
        ) from e

    WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    device = get_whisper_device()
    print(
        "loading whisper fallback:",
        f"model={WHISPER_MODEL_NAME}",
        f"device={device}",
        f"download_root={WHISPER_DOWNLOAD_ROOT}",
    )
    _WHISPER_MODEL = whisper.load_model(
        WHISPER_MODEL_NAME,
        device=device,
        download_root=str(WHISPER_DOWNLOAD_ROOT),
    )
    return _WHISPER_MODEL


def transcribe_audio_segments(audio_path):
    if audio_path is None or is_missing_value(audio_path):
        return []

    audio_path = Path(str(audio_path))
    if not audio_path.exists():
        return []

    model = load_whisper_model()
    result = model.transcribe(str(audio_path), verbose=False, fp16=False)
    segments = []
    for item in result.get("segments", []):
        text = clean_subtitle_text(item.get("text", ""))
        if not text:
            continue
        segments.append({
            "start": round(float(item.get("start", 0.0)), 3),
            "end": round(float(item.get("end", 0.0)), 3),
            "text": text,
        })
    return segments


def read_transcript_segments(row):
    segments = read_vtt_segments(row.get("subtitle_path", ""))
    if segments:
        return segments
    return transcribe_audio_segments(row.get("audio_path", ""))


def count_words(text):
    return len(WORD_RE.findall(text))


def compute_speech_rate(segments):
    word_count = sum(count_words(segment["text"]) for segment in segments)
    speech_duration = sum(
        max(0.0, float(segment["end"]) - float(segment["start"]))
        for segment in segments
    )
    if speech_duration <= 0:
        return 0.0
    return round(word_count / speech_duration, 4)


def frame_rms(sample, sr, np):
    if sample.size == 0:
        return np.asarray([], dtype=np.float32)

    frame_length = max(1, int(sr * 0.025))
    hop_length = max(1, int(sr * 0.010))
    if sample.size < frame_length:
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[: sample.size] = sample
        sample = padded

    starts = range(0, sample.size - frame_length + 1, hop_length)
    values = [
        float(np.sqrt(np.mean(sample[start:start + frame_length] ** 2)))
        for start in starts
    ]
    return np.asarray(values, dtype=np.float32)


def compute_silence_ratio(sample, sr, np):
    rms = frame_rms(sample, sr, np)
    if rms.size == 0:
        return 1.0
    threshold = max(0.005, float(np.percentile(rms, 95)) * 0.1)
    return round(float(np.mean(rms < threshold)), 4)


def compute_energy_laughter_hint(sample, sr, np):
    rms = frame_rms(sample, sr, np)
    if rms.size < 3:
        return False
    high_threshold = max(0.02, float(np.percentile(rms, 90)))
    high_energy_ratio = float(np.mean(rms > high_threshold))
    duration = sample.size / sr if sr > 0 else 0.0
    return duration > 0 and high_energy_ratio > 0.08


def compute_high_pitch(sample, sr):
    if sample.size == 0 or sr <= 0:
        return False

    try:
        import torch
        from torchaudio.functional import detect_pitch_frequency

        waveform = torch.from_numpy(sample.astype("float32")).unsqueeze(0)
        pitch = detect_pitch_frequency(waveform, sr).squeeze(0)
        pitch = pitch[pitch > 0]
        if pitch.numel() == 0:
            return False
        return bool(torch.median(pitch).item() >= 220.0)
    except Exception:
        return False


def read_audio_sample(audio_path, np, sf):
    if audio_path is None or is_missing_value(audio_path):
        return None, 0

    audio_path = Path(str(audio_path))
    if not audio_path.exists():
        return None, 0

    sample, sr = sf.read(audio_path)
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim > 1:
        sample = sample.mean(axis=1)
    return sample, sr


def compute_audio_stats(sample, sr, segments, np):
    if sample is None:
        return {
            "volume_mean": 0.0,
            "speech_rate": compute_speech_rate(segments),
            "laughter": detect_laughter(segments, None, 0, np),
            "silence_ratio": 1.0,
            "high_pitch": False,
        }

    volume_mean = round(float(np.mean(np.abs(sample))) if sample.size else 0.0, 6)
    speech_rate = compute_speech_rate(segments)
    silence_ratio = compute_silence_ratio(sample, sr, np)
    high_pitch = compute_high_pitch(sample, sr)
    laughter = detect_laughter(segments, sample, sr, np)

    return {
        "volume_mean": volume_mean,
        "speech_rate": speech_rate,
        "laughter": laughter,
        "silence_ratio": silence_ratio,
        "high_pitch": high_pitch,
    }


def read_audio_stats(audio_path, segments, np, sf):
    sample, sr = read_audio_sample(audio_path, np, sf)
    return compute_audio_stats(sample, sr, segments, np)


def slice_audio_sample(sample, sr, start_time, end_time, np):
    if sample is None or sr <= 0:
        return None

    start_index = max(0, int(round(float(start_time) * sr)))
    end_index = min(sample.size, int(round(float(end_time) * sr)))
    if end_index <= start_index:
        return np.asarray([], dtype=np.float32)
    return sample[start_index:end_index]


def detect_laughter(segments, sample, sr, np):
    text = " ".join(segment["text"].lower() for segment in segments)
    if any(keyword in text for keyword in LAUGHTER_KEYWORDS):
        return True
    if sample is None or sr <= 0:
        return False
    return compute_energy_laughter_hint(sample, sr, np)


def load_emotion_annotations(path, pd):
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    required_columns = {"Vid", "sub_id", "emotion"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"emotion manifest missing columns: {sorted(missing_columns)}")

    annotations = {}
    detail_columns = [
        "person",
        "emotion",
        "degree",
        "start_time",
        "end_time",
        "reason",
    ]

    for _, row in df.iterrows():
        key = (str(row["Vid"]), str(row["sub_id"]))
        item = {}
        for column in detail_columns:
            value = row[column] if column in row else ""
            if pd.isna(value):
                value = ""
            item[column] = value.item() if hasattr(value, "item") else value
        item["emotion"] = str(item["emotion"]).strip().lower()
        for time_column in ("start_time", "end_time"):
            time_value = parse_optional_float(item.get(time_column, ""))
            item[time_column] = round(time_value, 3) if time_value is not None else ""
        annotations.setdefault(key, []).append(item)

    return annotations


def parse_optional_float(value):
    if is_missing_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_tone_fields(annotations):
    if not annotations:
        return {
            "tone_hint": "unknown",
            "tone_hints": [],
            "emotion_annotations": [],
        }

    emotions = [
        str(item.get("emotion", "")).strip().lower()
        for item in annotations
        if str(item.get("emotion", "")).strip()
    ]
    if not emotions:
        tone_hint = "unknown"
        tone_hints = []
    else:
        counter = Counter(emotions)
        tone_hint = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        tone_hints = sorted(set(emotions))

    return {
        "tone_hint": tone_hint,
        "tone_hints": tone_hints,
        "emotion_annotations": annotations,
    }


def segments_overlap(segment, start_time, end_time):
    segment_start = float(segment.get("start", 0.0))
    segment_end = float(segment.get("end", 0.0))
    return segment_end > start_time and segment_start < end_time


def filter_segments_for_clip(segments, start_time, end_time):
    return [
        segment
        for segment in segments
        if segments_overlap(segment, start_time, end_time)
    ]


def get_annotation_time_key(annotation):
    start_time = parse_optional_float(annotation.get("start_time", ""))
    end_time = parse_optional_float(annotation.get("end_time", ""))
    if start_time is None or end_time is None or end_time <= start_time:
        return None
    return (round(start_time, 3), round(end_time, 3))


def group_annotations_by_time(annotations):
    groups = {}
    for annotation in annotations:
        key = get_annotation_time_key(annotation)
        if key is None:
            continue
        groups.setdefault(key, []).append(annotation)
    return groups


def unique_non_empty(values):
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def build_clip_json(
    row,
    clip_index,
    start_time,
    end_time,
    person,
    clip_segments,
    emotion_annotations,
    sample,
    sr,
    np,
):
    clip_sample = slice_audio_sample(sample, sr, start_time, end_time, np)
    audio_stats = compute_audio_stats(clip_sample, sr, clip_segments, np)
    tone_fields = build_tone_fields(emotion_annotations)
    values = {
        "video_id": str(row["Vid"]),
        "sub_id": str(row["sub_id"]),
        "clip_id": f"clip_{clip_index:03d}",
        "start_time": round(float(start_time), 3),
        "end_time": round(float(end_time), 3),
        "person": person or "unknown",
        "speech_text": " ".join(segment["text"] for segment in clip_segments),
        "segments": clip_segments,
        **tone_fields,
        **audio_stats,
    }
    return {field: values[field] for field in CLIP_JSON_FIELDS}


def build_data2_clips(row, annotations, segments, sample, sr, np):
    groups = group_annotations_by_time(annotations)
    clips = []
    for clip_index, ((start_time, end_time), group) in enumerate(sorted(groups.items())):
        people = unique_non_empty(item.get("person", "") for item in group)
        person = " / ".join(people) if people else "unknown"
        clip_segments = filter_segments_for_clip(segments, start_time, end_time)
        clips.append(build_clip_json(
            row,
            clip_index,
            start_time,
            end_time,
            person,
            clip_segments,
            group,
            sample,
            sr,
            np,
        ))
    return clips


def build_segment_clips(row, segments, sample, sr, np):
    clips = []
    for clip_index, segment in enumerate(segments):
        start_time = float(segment.get("start", 0.0))
        end_time = float(segment.get("end", 0.0))
        if end_time <= start_time:
            continue
        clips.append(build_clip_json(
            row,
            clip_index,
            start_time,
            end_time,
            "unknown",
            [segment],
            [],
            sample,
            sr,
            np,
        ))
    return clips


def build_clips(row, annotations, segments, sample, sr, np):
    data2_clips = build_data2_clips(row, annotations, segments, sample, sr, np)
    if data2_clips:
        return data2_clips
    return build_segment_clips(row, segments, sample, sr, np)


def build_audio_json(row, emotion_annotations, np, sf):
    segments = read_transcript_segments(row)
    speech_text = " ".join(segment["text"] for segment in segments)
    sample, sr = read_audio_sample(row["audio_path"], np, sf)
    audio_stats = compute_audio_stats(sample, sr, segments, np)
    tone_fields = build_tone_fields(emotion_annotations)
    clips = build_clips(row, emotion_annotations, segments, sample, sr, np)
    values = {
        "speech_text": speech_text,
        "segments": segments,
        **audio_stats,
        **tone_fields,
        "clips": clips,
    }
    return {field: values[field] for field in AUDIO_JSON_FIELDS}


def existing_json_is_complete(path):
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(field in payload for field in AUDIO_JSON_FIELDS)


def main():
    import numpy as np
    import pandas as pd
    import soundfile as sf

    manifest_path = Path(config.PATH_TO_AUDIO_MANIFEST)
    emotion_manifest_path = manifest_path.parent / "emotion_sort_manifest.csv"
    audio_json_root = Path(config.PATH_TO_PROCESSED_DIR) / "audio_json"

    df = pd.read_csv(manifest_path)
    required_columns = {"Vid", "sub_id", "audio_path", "subtitle_path"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"manifest missing columns: {sorted(missing_columns)}")

    emotion_map = load_emotion_annotations(emotion_manifest_path, pd)
    audio_json_root.mkdir(parents=True, exist_ok=True)

    print("manifest:", manifest_path)
    print("emotion_manifest:", emotion_manifest_path)
    print("audio_json_root:", audio_json_root)

    total = len(df)
    skipped = 0
    success = 0
    failed = 0

    for idx, row in df.iterrows():
        vid = str(row["Vid"])
        sub_id = str(row["sub_id"])
        save_path = audio_json_root / vid / f"{sub_id}.json"

        if existing_json_is_complete(save_path):
            skipped += 1
            continue

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = build_audio_json(
                row,
                emotion_map.get((vid, sub_id), []),
                np,
                sf,
            )
            save_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            success += 1
        except Exception as e:
            failed += 1
            print(f"failed: Vid={vid}, sub_id={sub_id}, {e}")

        if (idx + 1) % PROGRESS_INTERVAL == 0:
            print(
                f"[{idx + 1}/{total}] "
                f"success={success}, skipped={skipped}, failed={failed}"
            )

    print("Done")
    print(f"total={total}")
    print(f"success={success}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")


if __name__ == "__main__":
    main()
