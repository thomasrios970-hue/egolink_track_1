"""
零训练 Qwen2.5-Omni baseline v1。

脚本作用：
1. 不训练、不微调、不更新任何模型参数，只调用现成 Qwen2.5-Omni 做推理。
2. 对每条 MCP 样本读取 16 帧图片、字幕文本和 audio_json 音频摘要。
3. 将视频帧、字幕、音频摘要、问题和选项拼成 prompt，让 Qwen2.5-Omni 输出选项字母和一句证据。
4. 只使用 Qwen2.5-Omni 的 Thinker 文本输出，不生成语音，降低显存和推理开销。
5. 结果逐行写入 JSONL，便于中断后继续跑；已有 sample_id 会自动跳过。

执行逻辑：
- 如果 manifest 中已有 question 和 A/B/C/D 字段，就按 MCP 四选一任务推理。
- 如果暂时只有当前情绪 manifest，可以用 --task emotion_sort 或 --task sentiment
  自动构造一个选择题，用来先跑通整个多模态流程。
- 图片优先读取 data/processed/frames_16/{Vid}/{sub_id}/；如果没有，会从 video_path
  临时均匀抽取 16 帧，不要求提前保存。

运行示例：
python code/baseline/qwen_omni_baseline_v1.py \
  --task mcp \
  --manifest data/question/mcp_data2_500.csv \
  --model-path /data/wzw/egolink_race/model/Qwen2.5-Omni-7B \
  --gpu 5 \
  --limit 5

依赖缺失时按提示安装：
pip install -U "transformers>=4.52.0" accelerate pillow opencv-python soundfile
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

DEFAULT_MODEL_PATH = Path(config.PATH_TO_HUGGINGFACE_MODEL) / "Qwen2.5-Omni-7B"
DEFAULT_MANIFESTS = {
    "mcp": Path(config.PATH_TO_PROCESSED_DIR).parent / "question" / "mcp_data2_500.csv",
    "qa": Path(config.PATH_TO_AUDIO_MANIFEST),
    "sentiment": Path(config.PATH_TO_AUDIO_MANIFEST),
    "emotion_sort": Path(config.PATH_TO_AUDIO_MANIFEST).parent / "emotion_sort_manifest.csv",
}
DEFAULT_OUTPUT_ROOT = Path(config.PATH_TO_PROCESSED_DIR) / "baseline_v1" / "qwen_omni"
DEFAULT_FRAMES_ROOT = Path(config.PATH_TO_PROCESSED_DIR) / "frames_16"
DEFAULT_AUDIO_JSON_ROOT = Path(config.PATH_TO_PROCESSED_DIR) / "audio_json"
SUPPORTED_GPUS = [4, 5]
DEFAULT_GPU = 5
DEFAULT_NUM_FRAMES = 16
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_DEVICE_MAP = "single"
DEFAULT_NUM_VOTES = 1
DEFAULT_VOTE_SEED = 2026

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text responses."
)

EMOTION_OPTIONS = [
    "angry",
    "disgusted",
    "happy",
    "sad",
    "sarcastic",
    "scared",
    "shy",
    "surprised",
]
SENTIMENT_OPTIONS = [
    "very negative (-3)",
    "negative (-2)",
    "slightly negative (-1)",
    "slightly positive (1)",
    "positive (2)",
    "very positive (3)",
]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="零训练调用 Qwen2.5-Omni，使用 16 帧、字幕和音频摘要生成 baseline v1 结果。",
    )
    parser.add_argument(
        "--task",
        choices=["mcp", "qa", "emotion_sort", "sentiment"],
        default="mcp",
        help="mcp/qa 读取 question/A/B/C/D；emotion_sort/sentiment 仅用于流程调试。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="输入 manifest。mcp/qa 任务需包含 question 和选项列；不传则按 task 使用默认 manifest。",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Qwen2.5-Omni 模型本地路径或 HuggingFace/ModelScope ID。",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=DEFAULT_FRAMES_ROOT,
        help="16 帧图片根目录，默认 data/processed/frames_16。",
    )
    parser.add_argument(
        "--audio-json-root",
        type=Path,
        default=DEFAULT_AUDIO_JSON_ROOT,
        help="audio_json 根目录，默认 data/processed/audio_json。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="结果 JSONL 路径；不传则写入 data/processed/baseline_v1/qwen_omni/<task>_qwen_omni_baseline_v1.jsonl。",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="只跑指定 split，例如 test/val/train；manifest 没有 split 列时忽略。",
    )
    parser.add_argument(
        "--limit",
        type=non_negative_int,
        default=0,
        help="最多处理多少条，0 表示不限制。",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        choices=SUPPORTED_GPUS,
        default=DEFAULT_GPU,
        help="使用的物理 GPU 卡号，只允许 4 或 5；默认 5。",
    )
    parser.add_argument(
        "--num-frames",
        type=positive_int,
        default=DEFAULT_NUM_FRAMES,
        help="每条视频输入的均匀抽帧数量，默认 16。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Qwen2.5-Omni 最多生成 token 数。",
    )
    parser.add_argument(
        "--num-votes",
        type=positive_int,
        default=DEFAULT_NUM_VOTES,
        help="每题推理投票次数；大于 1 时会打乱选项顺序并按选项文本投票，默认 1。",
    )
    parser.add_argument(
        "--vote-seed",
        type=int,
        default=DEFAULT_VOTE_SEED,
        help="选项打乱投票的随机种子，默认 2026。",
    )
    parser.add_argument(
        "--device-map",
        choices=["single", "auto"],
        default=DEFAULT_DEVICE_MAP,
        help="模型加载方式；single 会完整放到所选 GPU，auto 交给 accelerate 自动切分。",
    )
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=1,
        help="每处理多少条打印一次进度。",
    )
    parser.add_argument(
        "--allow-leakage-audio-fields",
        action="store_true",
        help="允许在 prompt 中使用 tone_hint/tone_hints 等标注衍生字段；正式评分不要开启。",
    )
    return parser


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
    raise ValueError(f"unsupported manifest format: {path}")


def normalize_row(row):
    return {str(k): v for k, v in dict(row).items()}


def row_value(row, *names, default=""):
    lowered = {str(k).lower(): k for k in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            value = row.get(key, default)
            if value is None:
                return default
            return value
    return default


def get_manifest_path(args):
    return args.manifest if args.manifest is not None else DEFAULT_MANIFESTS[args.task]


def get_output_path(args):
    if args.output_path is not None:
        return args.output_path
    return DEFAULT_OUTPUT_ROOT / f"{args.task}_qwen_omni_baseline_v1.jsonl"


def filter_rows(rows, split):
    if not split:
        return rows
    if not rows or "split" not in {key.lower() for key in rows[0]}:
        return rows
    return [
        row for row in rows
        if str(row_value(row, "split")).strip().lower() == split.lower()
    ]


def make_sample_id(row, index):
    explicit_id = row_value(row, "id", "qid", "question_id", "sample_id", default="")
    if str(explicit_id).strip():
        return str(explicit_id).strip()
    vid = str(row_value(row, "Vid", "video_id", default="")).strip()
    sub_id = str(row_value(row, "sub_id", "clip_id", default="")).strip()
    if vid and sub_id:
        person = str(row_value(row, "person", default="")).strip()
        start_time = str(row_value(row, "start_time", default="")).strip()
        end_time = str(row_value(row, "end_time", default="")).strip()
        parts = [vid, sub_id]
        if person:
            parts.append(person)
        if start_time or end_time:
            parts.extend([start_time, end_time])
        return "_".join(parts)
    return f"row_{index:06d}"


def make_row_id(row, index):
    row_id = str(row_value(row, "row_id", default="")).strip()
    return row_id or f"row_{index:06d}"


def option_columns_for_qa(row):
    candidates = [
        ("A", "B", "C", "D"),
        ("option_a", "option_b", "option_c", "option_d"),
        ("choice_a", "choice_b", "choice_c", "choice_d"),
    ]
    for names in candidates:
        values = [str(row_value(row, name, default="")).strip() for name in names]
        if all(values):
            return values
    options = row_value(row, "options", "choices", default="")
    if isinstance(options, dict):
        values = [str(options.get(letter, "")).strip() for letter in "ABCD"]
        if all(values):
            return values
    if isinstance(options, list) and len(options) >= 4:
        return [str(item).strip() for item in options[:4]]
    raise ValueError("qa manifest requires question and A/B/C/D option columns")


def build_question_and_options(row, task):
    if task in {"mcp", "qa"}:
        question = str(row_value(row, "question", "query", "Question", default="")).strip()
        if not question:
            raise ValueError("mcp/qa manifest requires a question column")
        return question, option_columns_for_qa(row)

    if task == "emotion_sort":
        person = str(row_value(row, "person", default="the target person")).strip()
        if not person:
            person = "the target person"
        question = (
            "What is the best emotion category for the target person in this egocentric video? "
            f"Target person: {person}."
        )
        return question, EMOTION_OPTIONS

    if task == "sentiment":
        question = (
            "What is the overall sentiment degree expressed in this egocentric video clip?"
        )
        return question, SENTIMENT_OPTIONS

    raise ValueError(f"unsupported task: {task}")


def get_gold_label(row, task, options):
    if task in {"mcp", "qa"}:
        raw = str(row_value(row, "answer", "label", "gold", default="")).strip()
        return raw or None
    if task == "emotion_sort":
        emotion = str(row_value(row, "emotion", default="")).strip().lower()
        return option_to_letter(emotion, options)
    if task == "sentiment":
        label = str(row_value(row, "label", default="")).strip()
        label_map = {
            "-3": "very negative (-3)",
            "-2": "negative (-2)",
            "-1": "slightly negative (-1)",
            "1": "slightly positive (1)",
            "2": "positive (2)",
            "3": "very positive (3)",
        }
        return option_to_letter(label_map.get(label, ""), options)
    return None


def option_to_letter(value, options):
    value = str(value).strip().lower()
    for index, option in enumerate(options):
        if str(option).strip().lower() == value:
            return LETTERS[index]
    return None


def load_audio_json(row, audio_json_root):
    vid = str(row_value(row, "Vid", "video_id", default="")).strip()
    sub_id = str(row_value(row, "sub_id", "clip_id", default="")).strip()
    if not vid or not sub_id:
        return {}
    path = audio_json_root / vid / f"{sub_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compact_text(text, max_chars):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def summarize_audio(audio_json, row, allow_leakage_audio_fields=False):
    if not audio_json:
        return "Audio summary is unavailable."

    summary = {
        "volume_mean": audio_json.get("volume_mean", 0.0),
        "speech_rate": audio_json.get("speech_rate", 0.0),
        "laughter": audio_json.get("laughter", False),
        "silence_ratio": audio_json.get("silence_ratio", 0.0),
        "high_pitch": audio_json.get("high_pitch", False),
    }
    if allow_leakage_audio_fields:
        summary["tone_hint"] = audio_json.get("tone_hint", "unknown")
        summary["tone_hints"] = audio_json.get("tone_hints", [])

    clip = find_matching_clip(audio_json.get("clips", []), row)
    if clip:
        summary["matched_clip"] = {
            "start_time": clip.get("start_time"),
            "end_time": clip.get("end_time"),
            "person": clip.get("person"),
            "speech_text": compact_text(clip.get("speech_text", ""), 500),
        }
        if allow_leakage_audio_fields:
            summary["matched_clip"]["tone_hint"] = clip.get("tone_hint")
    return json.dumps(summary, ensure_ascii=False)


def find_matching_clip(clips, row):
    start = to_float(row_value(row, "start_time", default=""))
    end = to_float(row_value(row, "end_time", default=""))
    if start is None or end is None:
        return None
    for clip in clips:
        clip_start = to_float(clip.get("start_time"))
        clip_end = to_float(clip.get("end_time"))
        if clip_start is None or clip_end is None:
            continue
        if abs(clip_start - start) < 0.05 and abs(clip_end - end) < 0.05:
            return clip
    return None


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_vtt_text(path):
    path = Path(str(path))
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
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


def get_subtitle_text(row, audio_json):
    if audio_json:
        clip = find_matching_clip(audio_json.get("clips", []), row)
        if clip:
            clip_text = str(clip.get("speech_text", "")).strip()
            if clip_text:
                return clip_text
    subtitle = str(audio_json.get("speech_text", "")).strip() if audio_json else ""
    if subtitle:
        return subtitle
    return read_vtt_text(row_value(row, "subtitle_path", default=""))


def build_prompt(question, options, subtitle, audio_summary):
    option_lines = "\n".join(
        f"{LETTERS[index]}. {option}"
        for index, option in enumerate(options)
    )
    valid_letters = "/".join(LETTERS[: len(options)])
    return f"""You are given an egocentric video, subtitle, and audio summary.
This is a forced-choice classification task. Choose exactly one option letter.
Do not continue the conversation. Do not repeat the prompt. Do not ask questions.
Use the visual context, the target time span, subtitle, and audio cues together.

Subtitle:
{subtitle}

Audio summary:
{audio_summary}

Question:
{question}

Options:
{option_lines}

Choose the best answer from the options above. If uncertain, still choose the most likely option.

Return exactly these three lines and nothing else:
Answer: {valid_letters}
Confidence: 0.0-1.0
Evidence: one short sentence."""


def get_saved_frame_paths(row, frames_root, num_frames):
    vid = str(row_value(row, "Vid", "video_id", default="")).strip()
    sub_id = str(row_value(row, "sub_id", "clip_id", default="")).strip()
    if not vid or not sub_id:
        return []
    sample_dir = frames_root / vid / sub_id
    paths = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.png"))
    if len(paths) <= num_frames:
        return paths
    indices = uniform_indices(len(paths), num_frames)
    return [paths[index] for index in indices]


def uniform_indices(total, count):
    if count <= 1:
        return [0]
    if total <= count:
        return list(range(total))
    return [
        round(index * (total - 1) / (count - 1))
        for index in range(count)
    ]


def frame_indices_for_row(row, total_frames, fps, num_frames):
    start = to_float(row_value(row, "start_time", default=""))
    end = to_float(row_value(row, "end_time", default=""))
    if fps <= 0 or start is None or end is None or end <= start:
        return uniform_indices(total_frames, num_frames)

    start_frame = max(0, min(total_frames - 1, int(start * fps)))
    end_frame = max(start_frame, min(total_frames - 1, int(end * fps)))
    window_total = end_frame - start_frame + 1
    return [
        start_frame + offset
        for offset in uniform_indices(window_total, num_frames)
    ]


def has_time_window(row):
    start = to_float(row_value(row, "start_time", default=""))
    end = to_float(row_value(row, "end_time", default=""))
    return start is not None and end is not None and end > start


def extract_temp_frames(row, temp_dir, num_frames):
    video_path = Path(str(row_value(row, "video_path", default="")))
    if not video_path.exists():
        return []

    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "缺少 opencv-python，无法从 video_path 临时抽帧。请先运行: pip install opencv-python"
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        cap.release()
        return []

    paths = []
    for order, frame_index in enumerate(frame_indices_for_row(row, total_frames, fps, num_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        path = Path(temp_dir) / f"frame_{order:03d}.jpg"
        if cv2.imwrite(str(path), frame):
            paths.append(path)
    cap.release()
    return paths


def get_frame_paths(row, frames_root, num_frames, temp_dir):
    if has_time_window(row):
        temp_paths = extract_temp_frames(row, temp_dir, num_frames)
        if temp_paths:
            return temp_paths
    paths = get_saved_frame_paths(row, frames_root, num_frames)
    if len(paths) >= num_frames:
        return paths[:num_frames]
    temp_paths = extract_temp_frames(row, temp_dir, num_frames)
    return temp_paths or paths


def get_device_map_arg(device_map):
    if device_map == "single":
        return {"": "cuda:0"}
    return "auto"


def load_model_and_processor(model_path, device_map):
    try:
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
    except Exception as e:
        raise RuntimeError(
            "当前环境缺少 Qwen2.5-Omni 推理依赖。请先运行: "
            'pip install -U "transformers>=4.52.0" accelerate pillow soundfile'
        ) from e

    device_map_arg = get_device_map_arg(device_map)
    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device_map_arg,
            trust_remote_code=True,
        )
    except TypeError:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map_arg,
            trust_remote_code=True,
        )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def get_model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device

    for module_name in ("thinker", "model"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        device = getattr(module, "device", None)
        if device is not None and str(device) != "meta":
            return device
        try:
            device = next(module.parameters()).device
            if str(device) != "meta":
                return device
        except StopIteration:
            continue

    hf_device_map = getattr(model, "hf_device_map", {})
    for device in hf_device_map.values():
        if str(device) not in {"meta", "disk", "cpu"}:
            return device

    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RuntimeError(
        "模型仍在 meta device 上，请改用 --device-map single 或检查显存/accelerate 加载状态"
    )


def build_messages(frame_paths, prompt):
    content = [
        {"type": "image", "path": str(Path(path).resolve())}
        for path in frame_paths
    ]
    content.append({"type": "text", "text": prompt})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def generate_answer(model, processor, frame_paths, prompt, max_new_tokens):
    messages = build_messages(frame_paths, prompt)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(get_model_input_device(model))
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def first_assistant_chunk(raw_output):
    text = str(raw_output or "").strip()
    for marker in ("\nHuman:", "\nUser:", "\nAssistant:", "\nSystem:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def parse_answer(raw_output, option_count, options=None):
    valid = LETTERS[:option_count]
    text = first_assistant_chunk(raw_output)
    match = re.search(
        r"Answer\s*:\s*([A-Z])\b(?!\s*/)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        letter = match.group(1).upper()
        return letter if letter in valid else None
    match = re.search(r"^\s*([A-Z])\s*[\.\),，、:：]?", text, flags=re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        return letter if letter in valid else None
    match = re.search(r"\b(?:option|choice)\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        return letter if letter in valid else None
    if options:
        lowered = text.lower()
        hits = []
        for index, option in enumerate(options):
            option_text = str(option).strip().lower()
            if option_text and re.search(rf"\b{re.escape(option_text)}\b", lowered):
                hits.append(LETTERS[index])
        if len(set(hits)) == 1:
            return hits[0]
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        letter = match.group(1).upper()
        return letter if letter in valid else None
    return None


def parse_evidence(raw_output):
    text = first_assistant_chunk(raw_output)
    match = re.search(r"Evidence\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group(1).strip().split())


def parse_confidence(raw_output):
    text = first_assistant_chunk(raw_output)
    match = re.search(
        r"Confidence\s*:\s*([01](?:\.\d+)?|\.\d+|100\s*%|\d{1,2}(?:\.\d+)?\s*%)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(r"^\s*[A-Z]\s*[\.\),，、:：]?\s*([01](?:\.\d+)?)\b", text)
    if not match:
        return None
    text = match.group(1).strip().replace(" ", "")
    try:
        if text.endswith("%"):
            value = float(text[:-1]) / 100.0
        else:
            value = float(text)
    except ValueError:
        return None
    return max(0.0, min(1.0, value))


def load_done_ids(output_path):
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            sample_id = item.get("sample_id")
            if sample_id and not item.get("error"):
                done.add(str(sample_id))
    return done


def prediction_text(answer, options):
    if not isinstance(answer, str):
        return None
    answer = answer.strip().upper()
    if answer not in LETTERS[:len(options)]:
        return None
    return options[LETTERS.index(answer)]


def vote_options(options, sample_id, vote_index, seed):
    options = list(options)
    if vote_index == 0:
        return options
    rng = random.Random(f"{seed}:{sample_id}:{vote_index}")
    shuffled = list(options)
    rng.shuffle(shuffled)
    return shuffled


def run_one_vote(
    model,
    processor,
    frame_paths,
    question,
    original_options,
    subtitle,
    audio_summary,
    max_new_tokens,
    sample_id,
    vote_index,
    vote_seed,
):
    current_options = vote_options(original_options, sample_id, vote_index, vote_seed)
    prompt = build_prompt(question, current_options, subtitle, audio_summary)
    raw_output = generate_answer(
        model,
        processor,
        frame_paths,
        prompt,
        max_new_tokens,
    )
    raw_answer = parse_answer(raw_output, len(current_options), current_options)
    prediction = prediction_text(raw_answer, current_options)
    mapped_answer = option_to_letter(prediction, original_options)
    return {
        "vote_index": vote_index,
        "options": {
            LETTERS[i]: option
            for i, option in enumerate(current_options)
        },
        "raw_answer": raw_answer,
        "mapped_answer": mapped_answer,
        "prediction": prediction,
        "confidence": parse_confidence(raw_output),
        "evidence": parse_evidence(raw_output),
        "raw_output": raw_output,
    }


def aggregate_votes(votes):
    valid_votes = [
        vote for vote in votes
        if vote.get("mapped_answer") in LETTERS
    ]
    if not valid_votes:
        return None, None, "", ""

    counts = Counter(vote["mapped_answer"] for vote in valid_votes)
    confidence_sums = Counter()
    first_index = {}
    for order, vote in enumerate(valid_votes):
        answer = vote["mapped_answer"]
        confidence = vote.get("confidence")
        confidence_sums[answer] += confidence if confidence is not None else 0.0
        first_index.setdefault(answer, order)

    best_answer = max(
        counts,
        key=lambda answer: (
            counts[answer],
            confidence_sums[answer],
            -first_index[answer],
        ),
    )
    chosen_vote = next(
        vote for vote in valid_votes
        if vote["mapped_answer"] == best_answer
    )
    confidence_values = [
        vote.get("confidence")
        for vote in valid_votes
        if vote["mapped_answer"] == best_answer and vote.get("confidence") is not None
    ]
    confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else None
    )
    return (
        best_answer,
        confidence,
        chosen_vote.get("evidence", ""),
        chosen_vote.get("raw_output", ""),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    manifest_path = get_manifest_path(args)
    output_path = get_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(row) for row in read_table(manifest_path)]
    for original_index, row in enumerate(rows):
        if not str(row_value(row, "row_id", default="")).strip():
            row["row_id"] = f"row_{original_index:06d}"
    rows = filter_rows(rows, args.split)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no rows to process")

    done_ids = load_done_ids(output_path)

    print("manifest:", manifest_path)
    print("output_path:", output_path)
    print("model_path:", args.model_path)
    print("frames_root:", args.frames_root)
    print("audio_json_root:", args.audio_json_root)
    print("task:", args.task)
    print("split:", args.split)
    print("rows:", len(rows))
    print("done:", len(done_ids))
    print("physical_gpu:", args.gpu)
    print("visible_cuda_device:", "cuda:0")
    print("device_map:", args.device_map)
    print("num_votes:", args.num_votes)
    print("allow_leakage_audio_fields:", args.allow_leakage_audio_fields)

    model, processor = load_model_and_processor(args.model_path, args.device_map)

    success = 0
    skipped = 0
    failed = 0

    with output_path.open("a", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            sample_id = make_sample_id(row, index)
            if sample_id in done_ids:
                skipped += 1
                continue

            try:
                question, options = build_question_and_options(row, args.task)
                audio_json = load_audio_json(row, args.audio_json_root)
                subtitle = compact_text(get_subtitle_text(row, audio_json), 2200)
                audio_summary = summarize_audio(
                    audio_json,
                    row,
                    allow_leakage_audio_fields=args.allow_leakage_audio_fields,
                )

                with tempfile.TemporaryDirectory(prefix="qwen_omni_frames_") as temp_dir:
                    frame_paths = get_frame_paths(row, args.frames_root, args.num_frames, temp_dir)
                    if not frame_paths:
                        raise ValueError("no video frames found or extracted")
                    votes = [
                        run_one_vote(
                            model,
                            processor,
                            frame_paths,
                            question,
                            options,
                            subtitle,
                            audio_summary,
                            args.max_new_tokens,
                            sample_id,
                            vote_index,
                            args.vote_seed,
                        )
                        for vote_index in range(args.num_votes)
                    ]

                answer, confidence, evidence, raw_output = aggregate_votes(votes)
                gold = get_gold_label(row, args.task, options)
                result = {
                    "row_id": make_row_id(row, index),
                    "sample_id": sample_id,
                    "Vid": row_value(row, "Vid", "video_id", default=""),
                    "sub_id": row_value(row, "sub_id", "clip_id", default=""),
                    "person": row_value(row, "person", default=""),
                    "split": row_value(row, "split", default=""),
                    "question": question,
                    "options": {
                        LETTERS[i]: option
                        for i, option in enumerate(options)
                    },
                    "answer": answer,
                    "prediction": prediction_text(answer, options),
                    "confidence": confidence,
                    "evidence": evidence,
                    "raw_output": raw_output,
                    "num_votes": args.num_votes,
                    "votes": votes if args.num_votes > 1 else [],
                    "gold": gold,
                    "correct": (answer == gold) if gold else None,
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                done_ids.add(sample_id)
                success += 1
            except Exception as e:
                failed += 1
                result = {
                    "row_id": make_row_id(row, index),
                    "sample_id": sample_id,
                    "Vid": row_value(row, "Vid", "video_id", default=""),
                    "sub_id": row_value(row, "sub_id", "clip_id", default=""),
                    "error": str(e),
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

            if (index + 1) % args.progress_interval == 0:
                print(
                    f"[{index + 1}/{len(rows)}] "
                    f"success={success}, skipped={skipped}, failed={failed}"
                )

    print("Done")
    print(f"success={success}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
