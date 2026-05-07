"""
批量生成改进版本地 MCQ 测试集

执行逻辑：
1. 从 data/annotation/data2.xlsx 读取 EgoLink / E3 标注。
2. 按 Vid 划分训练集和测试集，保证训练 Vid 与测试 Vid 不重叠。
3. emotion 和 reason 只使用标注文件生成。
4. predict 使用当前片段 frames_16、当前字幕、当前音频转写文本，以及之后子视频的 frames_8、字幕、音频转写文本。
5. ego_summary 使用当前片段 frames_16、当前字幕、当前音频转写文本，以及之前子视频的 frames_8、字幕、音频转写文本。
6. 写出 questions、answer_key、review jsonl 和 review csv。

运行示例：
python code/track_1/improve_produce_question_test.py \
  --num_per_type 50 \
  --seed 42
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib import request

import pandas as pd
import whisper

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


subtitle_dir = Path(config.PATH_TO_DATA_DIR) / "subtext"
video_dir = Path(config.PATH_TO_DATA_DIR) / "E3" / "E3"
audio_dir = Path(config.PATH_TO_DATA_DIR) / "audio"
frame_root = Path(config.PATH_TO_PROCESSED_DIR) / "frames_16"
audio_text_root = Path(config.PATH_TO_PROCESSED_DIR) / "audio_to_text"
data_path = Path(config.PATH_TO_DATA_DIR) / "annotation" / "data2.xlsx"
output_dir = Path(config.PATH_TO_QUESTION_DIR) / "test_question"
TARGET_VID = 8745
TARGET_SUB_ID = 3

QUESTIONS_PATH = output_dir / "local_test_questions.jsonl"
ANSWER_KEY_PATH = output_dir / "local_test_answer_key.jsonl"
REVIEW_PATH = output_dir / "local_test_review.jsonl"
REVIEW_CSV_PATH = output_dir / "local_test_review.csv"
MCQ_TRAIN_PATH = Path(config.PATH_TO_QUESTION_DIR) / "mcq_train.jsonl"

MODEL_NAME = "gpt-5.5"
QWEN_API_URL = os.environ.get(
    "QWEN_API_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
API_SLEEP_MIN = 1.0
API_SLEEP_MAX = 3.0
API_MAX_RETRY = 3
MAX_CONTEXT_SUBVIDEOS = 3
whisper_model = None

REQUIRED_COLUMNS = {
    "Vid",
    "sub_id",
    "person",
    "emotion",
    "degree",
    "start_time",
    "end_time",
    "reason",
    "set",
}
QUESTION_TYPES = ["emotion", "reason", "predict", "ego_summary"]
OPTION_LETTERS = ["A", "B", "C", "D"]


"""参数解析器统一在这里设置，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--num_per_type", type=int, default=10, help="每类题目生成数量")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--test_vid_ratio", type=float, default=0.2, help="没有 mcq_train.jsonl 时的测试 Vid 比例")
parser.add_argument("--whisper_model", default="small", help="Whisper 模型名称")
parser.add_argument("--gpu", type=int, default=4, help="Whisper 使用的 GPU，-1 表示 CPU")
args = parser.parse_args()
rng = random.Random(args.seed)


def read_subtitle_text(Vid, sub_id, start_time: float, end_time: float) -> str:
    """
    作用：根据 Vid、sub_id 和时间范围读取对应 vtt 字幕片段。
    输入：Vid、sub_id、start_time、end_time。
    输出：该时间段附近的字幕文本 str。
    """
    def parse_time(value):
        value = value.strip().replace(",", ".")
        parts = value.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)

    path = subtitle_dir / str(Vid) / f"{sub_id}.vtt"
    if not path.exists():
        return ""
    start_bound = max(0.0, float(start_time) - 2.0)
    end_bound = float(end_time) + 2.0
    kept, cur_start, cur_end, cur_text = [], None, None, []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text == "WEBVTT" or text.isdigit():
            continue
        if "-->" in text:
            if cur_start is not None and cur_text and cur_start <= end_bound and cur_end >= start_bound:
                kept.append(" ".join(cur_text))
            left, right = text.split("-->", 1)
            cur_start = parse_time(left.strip().split()[0])
            cur_end = parse_time(right.strip().split()[0])
            cur_text = []
        else:
            cur_text.append(text)
    if cur_start is not None and cur_text and cur_start <= end_bound and cur_end >= start_bound:
        kept.append(" ".join(cur_text))
    return " ".join(" ".join(kept).split())[:2000]


def get_or_extract_frame_paths(Vid, sub_id, num_frames: int) -> list:
    """
    作用：读取指定帧数的抽帧图片；如果不存在，就从视频抽取并保存到 frames_{num_frames}/{Vid}/{sub_id}。
    输入：Vid、sub_id、num_frames。
    输出：抽帧图片路径列表 list。
    """
    save_dir = Path(config.PATH_TO_PROCESSED_DIR) / f"frames_{num_frames}" / str(Vid) / str(sub_id)
    frame_paths = sorted(save_dir.glob("*.jpg")) + sorted(save_dir.glob("*.png"))
    if len(frame_paths) >= num_frames:
        return frame_paths

    video_path = video_dir / str(Vid) / f"{sub_id}.mp4"
    if not video_path.exists():
        return frame_paths

    import cv2
    import numpy as np

    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frame_paths

    for frame_order, frame_index in enumerate(np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()):
        image_path = save_dir / f"frame_{frame_order:03d}.jpg"
        if image_path.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cap.release()
    return sorted(save_dir.glob("*.jpg")) + sorted(save_dir.glob("*.png"))


def transcribe_audio(audio_path: Path, Vid, sub_id) -> str:
    """
    作用：用 Whisper 将 wav 音频转写成文本，并缓存到 processed/audio_to_text。
    输入：audio_path 音频路径、Vid、sub_id。
    输出：音频转写文本 str。
    """
    save_path = audio_text_root / str(Vid) / f"{sub_id}.txt"
    if save_path.exists():
        return save_path.read_text(encoding="utf-8").strip()
    if not audio_path.exists():
        return ""
    global whisper_model
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"
    if whisper_model is None:
        whisper_model = whisper.load_model(
            name=args.whisper_model,
            device=device,
            download_root=str(Path(config.PATH_TO_MODEL_DIR) / "huggingface_model" / "whisper"),
        )
    result = whisper_model.transcribe(str(audio_path), verbose=False, fp16=False)
    audio_text = " ".join(segment.get("text", "") for segment in result.get("segments", [])).strip()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(audio_text, encoding="utf-8")
    return audio_text


def read_full_subtitle_text(Vid, sub_id) -> str:
    """
    作用：读取某个子视频完整 vtt 字幕文本。
    输入：Vid、sub_id。
    输出：去掉时间戳后的字幕文本 str。
    """
    path = subtitle_dir / str(Vid) / f"{sub_id}.vtt"
    if not path.exists():
        return ""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if text and text != "WEBVTT" and "-->" not in text and not text.isdigit():
            lines.append(text)
    return " ".join(" ".join(lines).split())[:2000]


def get_context_frame_paths(Vid, sub_id, direction: str) -> dict:
    """
    作用：读取当前子视频的 16 帧，以及同一 Vid 中当前 sub_id 之前或之后所有子视频的 8 帧、字幕和音频转写文本。
    输入：Vid、sub_id、direction；direction 为 before 或 after。
    输出：包含 current_frame_paths、context_frame_paths、context_audio_text、context_subtitle_text 的 dict。
    """
    current_frame_paths = get_or_extract_frame_paths(Vid, sub_id, 16)  # 输入：Vid、sub_id、16；输出：当前子视频 frames_16 图片路径。

    current_sub_id = int(float(sub_id))
    context_frame_paths = []
    context_audio_texts = []
    context_subtitle_texts = []
    video_root = video_dir / str(Vid)
    sub_ids = sorted(int(path.stem) for path in video_root.glob("*.mp4") if path.stem.replace(".", "", 1).isdigit())
    if direction == "before":
        sub_ids = [item for item in sub_ids if item < current_sub_id][-MAX_CONTEXT_SUBVIDEOS:]
    else:
        sub_ids = [item for item in sub_ids if item > current_sub_id][:MAX_CONTEXT_SUBVIDEOS]
    for sub_dir_id in sub_ids:
        context_frame_paths.extend(get_or_extract_frame_paths(Vid, sub_dir_id, 8))  # 输入：Vid、上下文sub_id、8；输出：上下文 frames_8 图片路径。
        audio_path = audio_dir / str(Vid) / f"{sub_dir_id}.wav"
        if audio_path.exists():
            audio_text = transcribe_audio(audio_path, Vid, sub_dir_id)  # 输入：上下文音频路径、Vid、sub_id；输出：Whisper 转写文本。
            if audio_text:
                context_audio_texts.append(f"sub_id {sub_dir_id}: {audio_text}")
        subtitle_text = read_full_subtitle_text(Vid, sub_dir_id)  # 输入：Vid、上下文sub_id；输出：上下文字幕文本。
        if subtitle_text:
            context_subtitle_texts.append(f"sub_id {sub_dir_id}: {subtitle_text}")
    return {
        "current_frame_paths": current_frame_paths,
        "context_frame_paths": context_frame_paths,
        "context_audio_text": "\n".join(context_audio_texts),
        "context_subtitle_text": "\n".join(context_subtitle_texts),
    }


def build_prompt(sample: dict, question_type: str, context_frame_paths: list = None, audio_text: str = "", context_audio_text: str = "", subtitle_text: str = "", context_subtitle_text: str = "") -> str:
    """
    作用：为一个样本和问题类型构建 Qwen3 输入提示词。
    输入：sample 字典、question_type 字符串、context_frame_paths 上下文 8 帧路径列表、audio_text 当前音频转写文本、context_audio_text 上下文音频转写文本、subtitle_text 当前字幕、context_subtitle_text 上下文字幕。
    输出：prompt 字符串。
    """
    if question_type == "emotion":
        base = {
            "Vid": sample["Vid"],
            "sub_id": sample["sub_id"],
            "person": sample["person"],
            "emotion": sample["emotion"],
            "start_time": sample["start_time"],
            "end_time": sample["end_time"],
        }
        task = (
            "Create an MCQ asking what emotion the person has in this localized segment. "
            "The correct_answer_text must be exactly the annotated emotion. "
            "Generate three plausible but wrong emotion words. "
            "Use only the annotation fields shown below; do not use subtitles, audio, or video."
        )
    elif question_type == "reason":
        base = {
            "Vid": sample["Vid"],
            "sub_id": sample["sub_id"],
            "person": sample["person"],
            "emotion": sample["emotion"],
            "reason": sample["reason"],
            "start_time": sample["start_time"],
            "end_time": sample["end_time"],
        }
        task = (
            "Create an MCQ asking why the person has this emotion in this localized segment. "
            "The correct_answer_text must be exactly the annotated reason. "
            "Use only the annotation fields shown below to write a natural question and three plausible but wrong reasons. "
            "Do not use subtitles, audio, or video."
        )
    elif question_type == "predict":
        base = {
            "Vid": sample["Vid"],
            "sub_id": sample["sub_id"],
            "person": sample["person"],
            "emotion": sample["emotion"],
            "reason": sample["reason"],
            "start_time": sample["start_time"],
            "end_time": sample["end_time"],
            "video_path": str(video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"),
            "audio_path": str(audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"),
            "frame_dir": str(frame_root / str(sample["Vid"]) / str(sample["sub_id"])),
            "future_8frame_context_count": len(context_frame_paths or []),
            "future_subtitle_text": context_subtitle_text,
            "future_audio_transcript": context_audio_text,
            "subtitle_text": subtitle_text,
            "audio_transcript": audio_text,
        }
        task = (
            "Create an MCQ asking what will most likely happen right after the current sub_id segment, "
            "or what the main participant will most likely intend to do next. "
            "The current sub_id segment is the question anchor. Use the future sub_id frames, subtitles, and audio transcripts as auxiliary evidence "
            "for what actually happens afterward, so the correct answer is as accurate as possible. "
            "If subtitles and Whisper transcripts conflict, prefer the evidence that is clearer, more segment-relevant, and more consistent with the frames. "
            "Do not make the future clips themselves the question target. "
            "All options must describe next-step behavioral intentions or immediate next events."
        )
    else:
        base = {
            "Vid": sample["Vid"],
            "sub_id": sample["sub_id"],
            "person": sample["person"],
            "emotion": sample["emotion"],
            "reason": sample["reason"],
            "start_time": sample["start_time"],
            "end_time": sample["end_time"],
            "video_path": str(video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"),
            "audio_path": str(audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"),
            "frame_dir": str(frame_root / str(sample["Vid"]) / str(sample["sub_id"])),
            "previous_8frame_context_count": len(context_frame_paths or []),
            "previous_subtitle_text": context_subtitle_text,
            "previous_audio_transcript": context_audio_text,
            "subtitle_text": subtitle_text,
            "audio_transcript": audio_text,
        }
        task = (
            "Create an MCQ asking for my high-level ego-centric summary of the current sub_id segment. "
            "The correct option should describe what I, as the camera wearer, understand, feel, or experience in the current segment. "
            "The correct option must not be only an emotion word, only a reason rewrite, or only a next action. "
            "Use the previous sub_id frames, subtitles, and audio transcripts only as background clues for understanding the current segment. "
            "If subtitles and Whisper transcripts conflict, prefer the evidence that is clearer, more segment-relevant, and more consistent with the frames. "
            "Do not summarize the previous clips themselves; the answer must summarize the current sub_id segment."
        )

    return f"""
You are generating one multiple-choice question for an EgoLink / E3 localized video segment.
Return strict JSON only. Do not include markdown, comments, or extra text.

Required JSON schema:
{{
  "question": "...",
  "correct_answer_text": "...",
  "incorrect_options": ["...", "...", "..."]
}}

Question type: {question_type}
Task: {task}

Rules:
- Options should be concise and rely on the real segment content.
- For emotion and reason, use annotation fields only.
- For predict, future sub_id frames/subtitles/audio transcripts are auxiliary evidence for the next event after the current segment; the question still anchors on the current sub_id.
- For ego_summary, previous sub_id frames/subtitles/audio transcripts are background clues; the answer must summarize the current sub_id from my first-person perspective.
- If subtitle text and Whisper transcript conflict, prefer the evidence that best matches the frames and current localized segment.
- For ego_summary, the subject is always the camera wearer / I.
- Incorrect options should be plausible misunderstandings.
- Incorrect options must not repeat or be semantically equivalent to the correct answer.
- All four options should have similar topic, length, and grammar.
- Keep only key information and avoid unrelated details.

Segment information:
{json.dumps(base, ensure_ascii=False)}
""".strip()


def encode_media(path: Path) -> str:
    """
    作用：把本地音频或视频文件编码成 base64 字符串，供多模态 API 使用。
    输入：path 文件路径。
    输出：base64 字符串；如果文件不存在则输出空字符串。
    """
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def call_qwen3(prompt: str, frame_paths: list = None, context_frame_paths: list = None, context_label: str = "") -> dict:
    """
    作用：使用 API 模式调用 Qwen3，并解析模型返回的严格 JSON。
    输入：prompt 字符串；可选 frame_paths 当前抽帧图片路径列表、context_frame_paths 上下文 8 帧路径列表、context_label 上下文说明。
    输出：包含 question、correct_answer_text、incorrect_options 的字典。
    """
    content = [{"type": "text", "text": prompt}]
    if frame_paths:
        content.append({"type": "text", "text": "Current localized segment frames:"})
        for frame_path in frame_paths:
            image_base64 = encode_media(frame_path)  # 输入：抽帧图片路径；输出：图片 base64 字符串。
            if image_base64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
    if context_frame_paths:
        content.append({"type": "text", "text": context_label})
        for frame_path in context_frame_paths:
            image_base64 = encode_media(frame_path)  # 输入：上下文抽帧图片路径；输出：图片 base64 字符串。
            if image_base64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You must return exactly one non-empty JSON object. "
                    "Do not return an empty message. Do not include markdown."
                ),
            },
            {"role": "user", "content": content if len(content) > 1 else prompt},
        ],
        "temperature": 0.2,
        "max_completion_tokens": 1024,
    }
    if len(content) == 1:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if QWEN_API_KEY:
        headers["Authorization"] = f"Bearer {QWEN_API_KEY}"
    result = {}
    content = None
    data = None
    for retry_idx in range(API_MAX_RETRY):
        time.sleep(random.uniform(API_SLEEP_MIN, API_SLEEP_MAX))
        req = request.Request(
            QWEN_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            error_text = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API request failed: HTTP {e.code}, {error_text}") from e
        message = result.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", item)) for item in content)
        if content is None or not str(content).strip():
            print(f"warning: empty API content, retry {retry_idx + 1}/{API_MAX_RETRY}", flush=True)
            content = None
            continue
        content = str(content).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"warning: invalid JSON, retry {retry_idx + 1}/{API_MAX_RETRY}", flush=True)
            data = None
            continue
        if len(data.get("incorrect_options", [])) == 3:
            break
        print(f"warning: incorrect_options not 3, retry {retry_idx + 1}/{API_MAX_RETRY}", flush=True)
        data = None
    if data is None:
        raise RuntimeError(
            "API response has invalid JSON schema after retries. Last content: "
            + str(content)
        )
    return data


def make_options(correct_answer_text: str, incorrect_options: list[str], rng: random.Random) -> tuple[dict, str]:
    """
    作用：把正确答案和 3 个错误选项随机打乱成 A/B/C/D。
    输入：correct_answer_text 字符串、incorrect_options 列表、rng 随机数对象。
    输出：options 字典和正确答案字母。
    """
    items = [(" ".join(str(correct_answer_text).split()), True)]
    for option in incorrect_options:
        text = " ".join(str(option).split())
        if text and text not in [item[0] for item in items]:
            items.append((text, False))
    if len(items) != 4:
        raise ValueError("options must contain 1 correct answer and 3 unique incorrect options")
    rng.shuffle(items)
    options = {letter: items[idx][0] for idx, letter in enumerate(OPTION_LETTERS)}
    answer = next(letter for idx, letter in enumerate(OPTION_LETTERS) if items[idx][1])
    return options, answer


"""代码块部分：读取数据、划分 Vid、生成题目、写出文件"""
df = pd.read_excel(data_path)
missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
    raise ValueError(f"missing columns: {sorted(missing)}")

samples = []
skipped = 0
for _, row in df.iterrows():
    if any(pd.isna(row[col]) or str(row[col]).strip() == "" for col in ["person", "emotion", "reason", "start_time", "end_time"]):
        skipped += 1
        continue

    start_time = float(row["start_time"])
    end_time = float(row["end_time"])
    if start_time >= end_time:
        skipped += 1
        continue

    Vid = row["Vid"].item() if hasattr(row["Vid"], "item") else row["Vid"]
    sub_id = row["sub_id"].item() if hasattr(row["sub_id"], "item") else row["sub_id"]
    Vid = int(Vid) if isinstance(Vid, float) and Vid.is_integer() else Vid
    sub_id = int(sub_id) if isinstance(sub_id, float) and sub_id.is_integer() else sub_id
    if not (video_dir / str(Vid) / f"{sub_id}.mp4").exists():
        skipped += 1
        continue

    samples.append(
        {
            "Vid": Vid,
            "sub_id": sub_id,
            "person": " ".join(str(row["person"]).split()),
            "emotion": " ".join(str(row["emotion"]).split()),
            "degree": row["degree"].item() if hasattr(row["degree"], "item") else row["degree"],
            "start_time": int(start_time) if start_time.is_integer() else start_time,
            "end_time": int(end_time) if end_time.is_integer() else end_time,
            "reason": " ".join(str(row["reason"]).split()),
            "set": " ".join(str(row["set"]).split()),
        }
    )

all_vids = sorted({str(sample["Vid"]) for sample in samples})
train_vids = {str(sample["Vid"]) for sample in samples if sample["set"] in ["train", "val"]}
test_vids = {str(sample["Vid"]) for sample in samples if sample["set"] == "test"} - train_vids

if MCQ_TRAIN_PATH.exists():
    with MCQ_TRAIN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if "Vid" in item:
                    train_vids.add(str(item["Vid"]))
    test_vids = test_vids - train_vids

if not test_vids:
    rng.shuffle(all_vids)
    test_count = max(1, int(len(all_vids) * args.test_vid_ratio))
    test_vids = set(all_vids[:test_count])
    train_vids = set(all_vids[test_count:])

candidates = [
    sample
    for sample in samples
    if int(sample["Vid"]) == TARGET_VID and int(sample["sub_id"]) == TARGET_SUB_ID
]
rng.shuffle(candidates)

questions = []
answer_keys = []
review_rows = []
used_qids = set()
counts = {question_type: 0 for question_type in QUESTION_TYPES}

for question_type in QUESTION_TYPES:
    print(f"generating {question_type}...", flush=True)
    for sample in candidates:
        if counts[question_type] >= args.num_per_type:
            break

        video_path = video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"
        audio_path = audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"
        frame_paths = []
        context_frame_paths = []
        audio_text = ""
        context_audio_text = ""
        subtitle_text = ""
        context_subtitle_text = ""
        context_label = ""
        if question_type == "predict":
            frame_data = get_context_frame_paths(  # 输入：Vid、sub_id、after；输出：当前子视频 frames_16 和之后子视频 frames_8/audio_text/subtitle_text。
                sample["Vid"],
                sample["sub_id"],
                "after",
            )
            frame_paths = frame_data["current_frame_paths"]
            context_frame_paths = frame_data["context_frame_paths"]
            audio_text = transcribe_audio(audio_path, sample["Vid"], sample["sub_id"])  # 输入：当前音频路径、Vid、sub_id；输出：Whisper 转写文本。
            context_audio_text = frame_data["context_audio_text"]
            subtitle_text = read_subtitle_text(sample["Vid"], sample["sub_id"], sample["start_time"], sample["end_time"])  # 输入：Vid、sub_id、start/end时间；输出：当前字幕文本。
            context_subtitle_text = frame_data["context_subtitle_text"]
            context_label = "Future sub_id 8-frame context from the same Vid, ordered by time:"
        elif question_type == "ego_summary":
            frame_data = get_context_frame_paths(  # 输入：Vid、sub_id、before；输出：当前子视频 frames_16 和之前子视频 frames_8/audio_text/subtitle_text。
                sample["Vid"],
                sample["sub_id"],
                "before",
            )
            frame_paths = frame_data["current_frame_paths"]
            context_frame_paths = frame_data["context_frame_paths"]
            audio_text = transcribe_audio(audio_path, sample["Vid"], sample["sub_id"])  # 输入：当前音频路径、Vid、sub_id；输出：Whisper 转写文本。
            context_audio_text = frame_data["context_audio_text"]
            subtitle_text = read_subtitle_text(sample["Vid"], sample["sub_id"], sample["start_time"], sample["end_time"])  # 输入：Vid、sub_id、start/end时间；输出：当前字幕文本。
            context_subtitle_text = frame_data["context_subtitle_text"]
            context_label = "Previous sub_id 8-frame context from the same Vid, ordered by time:"
        prompt = build_prompt(  # 输入：样本、题型、上下文8帧路径、当前/上下文音频转写文本、当前/上下文字幕；输出：Qwen3 prompt 字符串。
            sample,
            question_type,
            context_frame_paths,
            audio_text,
            context_audio_text,
            subtitle_text,
            context_subtitle_text,
        )
        if question_type in ["predict", "ego_summary"]:
            # 输入：prompt、当前抽帧图片路径列表、上下文抽帧图片路径列表、上下文说明；输出：多模态生成的题目 JSON 字典。
            result = call_qwen3(prompt, frame_paths, context_frame_paths, context_label)
        else:
            result = call_qwen3(prompt)  # 输入：prompt 字符串；输出：文本生成的题目 JSON 字典。
        if question_type == "emotion":
            correct_answer_text = sample["emotion"]
        elif question_type == "reason":
            correct_answer_text = sample["reason"]
        else:
            correct_answer_text = result["correct_answer_text"]
        # 输入：正确答案、错误选项、随机对象；输出：A/B/C/D 选项和答案字母。
        options, answer = make_options(correct_answer_text, result["incorrect_options"], rng)

        safe_person = re.sub(r"\W+", "_", sample["person"]).strip("_") or "person"
        qid_base = f"{sample['Vid']}_{sample['sub_id']}_{safe_person}_{question_type}"
        qid = qid_base
        idx = 2
        while qid in used_qids:
            qid = f"{qid_base}_{idx}"
            idx += 1
        used_qids.add(qid)

        questions.append(
            {
                "qid": qid,
                "Vid": sample["Vid"],
                "sub_id": sample["sub_id"],
                "person": sample["person"],
                "question_type": question_type,
                "question": " ".join(str(result["question"]).split()),
                "options": options,
                "start_time": sample["start_time"],
                "end_time": sample["end_time"],
            }
        )
        answer_keys.append({"qid": qid, "answer": answer})
        review_rows.append(
            {
                "qid": qid,
                "Vid": sample["Vid"],
                "sub_id": sample["sub_id"],
                "person": sample["person"],
                "question_type": question_type,
                "question": " ".join(str(result["question"]).split()),
                "A": options.get("A", ""),
                "B": options.get("B", ""),
                "C": options.get("C", ""),
                "D": options.get("D", ""),
                "answer": answer,
                "answer_text": options[answer],
                "emotion": sample["emotion"],
                "degree": sample["degree"],
                "reason": sample["reason"],
                "start_time": sample["start_time"],
                "end_time": sample["end_time"],
                "subtitle_text": subtitle_text,
                "audio_transcript": audio_text,
            }
        )
        counts[question_type] += 1
        print(f"{question_type}: {counts[question_type]}/{args.num_per_type}", flush=True)

    if counts[question_type] < args.num_per_type:
        print(f"warning: {question_type} only generated {counts[question_type]} questions", flush=True)

output_dir.mkdir(parents=True, exist_ok=True)
with QUESTIONS_PATH.open("w", encoding="utf-8") as f:
    for row in questions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with ANSWER_KEY_PATH.open("w", encoding="utf-8") as f:
    for row in answer_keys:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with REVIEW_PATH.open("w", encoding="utf-8") as f:
    for row in review_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
pd.DataFrame(review_rows).to_csv(REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")

print(f"total_questions: {len(questions)}", flush=True)
print(f"counts_by_type: {counts}", flush=True)
print(f"questions_path: {QUESTIONS_PATH}", flush=True)
print(f"answer_key_path: {ANSWER_KEY_PATH}", flush=True)
print(f"review_path: {REVIEW_PATH}", flush=True)
print(f"review_csv_path: {REVIEW_CSV_PATH}", flush=True)
print(f"skipped_samples: {skipped}", flush=True)
print(f"train_test_vid_overlap: {bool(train_vids & test_vids)}", flush=True)
