"""
只针对 8745_3 子视频生成改进版本地 MCQ 测试集

执行逻辑：
1. 从 data/annotation/data2.xlsx 读取 EgoLink / E3 标注。
2. 只保留 Vid=8745 且 sub_id=3 的定位片段。
3. emotion 和 reason 只使用标注文件生成。
4. predict 使用当前片段之后子视频的 frames_8，ego_summary 使用当前片段之前子视频的 frames_8。
5. 写出 questions、answer_key、review jsonl 和 review csv。

运行示例：
python code/track_1/improve_produce_question_test.py \
  --num_per_type 1 \
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

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


subtitle_dir = Path(config.PATH_TO_DATA_DIR) / "subtext"
video_dir = Path(config.PATH_TO_DATA_DIR) / "E3" / "E3"
audio_dir = Path(config.PATH_TO_DATA_DIR) / "audio"
frame_root = Path(config.PATH_TO_PROCESSED_DIR) / "frames_16"
context_frame_root = Path(config.PATH_TO_PROCESSED_DIR) / "frames_8"
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
parser.add_argument("--num_per_type", type=int, default=1, help="每类题目生成数量")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--test_vid_ratio", type=float, default=0.2, help="没有 mcq_train.jsonl 时的测试 Vid 比例")
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


def get_context_frame_paths(Vid, sub_id, direction: str) -> list:
    """
    作用：读取同一 Vid 中当前 sub_id 之前或之后所有子视频的 8 帧图片路径。
    输入：Vid、sub_id、direction；direction 为 before 或 after。
    输出：上下文子视频的 frames_8 图片路径列表 list。
    """
    root = context_frame_root / str(Vid)
    if not root.exists():
        return []
    current_sub_id = int(float(sub_id))
    frame_paths = []
    for sub_dir in sorted(root.iterdir(), key=lambda path: int(float(path.name))):
        if not sub_dir.is_dir():
            continue
        sub_dir_id = int(float(sub_dir.name))
        if direction == "before" and sub_dir_id >= current_sub_id:
            continue
        if direction == "after" and sub_dir_id <= current_sub_id:
            continue
        frame_paths.extend(sorted(sub_dir.glob("*.jpg")))
        frame_paths.extend(sorted(sub_dir.glob("*.png")))
    return frame_paths


def build_prompt(sample: dict, question_type: str, subtitle_text: str, context_frame_paths: list = None) -> str:
    """
    作用：为一个样本和问题类型构建 Qwen3 输入提示词。
    输入：sample 字典、question_type 字符串、subtitle_text 字符串、context_frame_paths 上下文 8 帧路径列表。
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
            "subtitle_text": subtitle_text,
        }
        task = (
            "Create an MCQ asking what will most likely happen right after the current sub_id segment, "
            "or what the main participant will most likely intend to do next. "
            "The current sub_id segment is the question anchor. Use the future sub_id 8-frame context only as evidence "
            "for what actually happens afterward, so the correct answer is as accurate as possible. "
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
            "subtitle_text": subtitle_text,
        }
        task = (
            "Create an MCQ asking for my high-level ego-centric summary of the current sub_id segment. "
            "The correct option should describe what I, as the camera wearer, understand, feel, or experience in the current segment. "
            "The correct option must not be only an emotion word, only a reason rewrite, or only a next action. "
            "Use the previous sub_id 8-frame context only as background clues for understanding the current segment. "
            "Do not summarize the previous clips; the answer must summarize the current sub_id segment."
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
- For predict, future sub_id frames are evidence for the next event after the current segment; the question still anchors on the current sub_id.
- For ego_summary, previous sub_id frames are background clues; the answer must summarize the current sub_id from my first-person perspective.
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


def call_qwen3(prompt: str, frame_paths: list = None, audio_path: Path = None, context_frame_paths: list = None, context_label: str = "") -> dict:
    """
    作用：使用 API 模式调用 Qwen3，并解析模型返回的严格 JSON。
    输入：prompt 字符串；可选 frame_paths 当前抽帧图片路径列表、audio_path 音频路径、context_frame_paths 上下文 8 帧路径列表、context_label 上下文说明。
    输出：包含 question、correct_answer_text、incorrect_options 的字典。
    """
    content = [{"type": "text", "text": prompt}]
    if audio_path is not None:
        audio_base64 = encode_media(audio_path)  # 输入：音频路径；输出：音频 base64 字符串。
        if audio_base64:
            content.append({"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}})
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
        "temperature": 0.5,
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

        subtitle_text = ""
        if question_type in ["predict", "ego_summary"]:
            subtitle_text = read_subtitle_text(  # 输入：Vid、sub_id、start/end 时间；输出：字幕文本 str。
                sample["Vid"],
                sample["sub_id"],
                sample["start_time"],
                sample["end_time"],
            )
        video_path = video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"
        audio_path = audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"
        frame_paths = sorted((frame_root / str(sample["Vid"]) / str(sample["sub_id"])).glob("*.jpg"))
        context_frame_paths = []
        context_label = ""
        if question_type == "predict":
            context_frame_paths = get_context_frame_paths(  # 输入：Vid、sub_id、after；输出：之后子视频 frames_8 图片路径列表。
                sample["Vid"],
                sample["sub_id"],
                "after",
            )
            context_label = "Future sub_id 8-frame context from the same Vid, ordered by time:"
        elif question_type == "ego_summary":
            context_frame_paths = get_context_frame_paths(  # 输入：Vid、sub_id、before；输出：之前子视频 frames_8 图片路径列表。
                sample["Vid"],
                sample["sub_id"],
                "before",
            )
            context_label = "Previous sub_id 8-frame context from the same Vid, ordered by time:"
        prompt = build_prompt(  # 输入：样本、题型、字幕、上下文 8 帧路径；输出：Qwen3 prompt 字符串。
            sample,
            question_type,
            subtitle_text,
            context_frame_paths,
        )
        if question_type in ["predict", "ego_summary"]:
            # 输入：prompt、当前抽帧图片路径列表、音频路径、上下文 8 帧路径列表、上下文说明；输出：多模态生成的题目 JSON 字典。
            result = call_qwen3(prompt, frame_paths, audio_path, context_frame_paths, context_label)
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
        generation_basis = {
            "note": "正式题目文件不包含这些依据。本实验只生成 8745_3；predict 使用之后子视频 frames_8，ego_summary 使用之前子视频 frames_8。",
            "multimodal_used": question_type in ["predict", "ego_summary"],
            "start_time": sample["start_time"],
            "end_time": sample["end_time"],
        }
        if question_type in ["emotion", "reason"]:
            generation_basis["person"] = sample["person"]
            generation_basis["emotion"] = sample["emotion"]
            if question_type == "reason":
                generation_basis["reason"] = sample["reason"]
            generation_basis["annotation_only"] = True
        else:
            generation_basis["annotation_fields_used"] = ["person", "emotion", "reason", "start_time", "end_time"]
            generation_basis["video_path"] = str(video_path)
            generation_basis["audio_path"] = str(audio_path)
            generation_basis["frame_paths"] = [str(path) for path in frame_paths]
            if question_type == "predict":
                generation_basis["future_8frame_paths"] = [str(path) for path in context_frame_paths]
            if question_type == "ego_summary":
                generation_basis["previous_8frame_paths"] = [str(path) for path in context_frame_paths]
            generation_basis["subtitle_text"] = subtitle_text

        review_rows.append(
            {
                "qid": qid,
                "Vid": sample["Vid"],
                "sub_id": sample["sub_id"],
                "person": sample["person"],
                "question_type": question_type,
                "question": " ".join(str(result["question"]).split()),
                "options": options,
                "answer": answer,
                "answer_text": options[answer],
                "emotion": sample["emotion"],
                "degree": sample["degree"],
                "reason": sample["reason"],
                "start_time": sample["start_time"],
                "end_time": sample["end_time"],
                "generation_basis": generation_basis,
                "video_path": str(video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"),
                "audio_path": str(audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"),
                "subtitle_path": str(subtitle_dir / str(sample["Vid"]) / f"{sample['sub_id']}.vtt"),
                "frame_dir": str(frame_root / str(sample["Vid"]) / str(sample["sub_id"])),
                "subtitle_text": subtitle_text,
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
review_csv_rows = []
for row in review_rows:
    review_csv_rows.append(
        {
            "qid": row["qid"],
            "Vid": row["Vid"],
            "sub_id": row["sub_id"],
            "person": row["person"],
            "question_type": row["question_type"],
            "question": row["question"],
            "A": row["options"].get("A", ""),
            "B": row["options"].get("B", ""),
            "C": row["options"].get("C", ""),
            "D": row["options"].get("D", ""),
            "answer": row["answer"],
            "answer_text": row["answer_text"],
            "emotion": row["emotion"],
            "degree": row["degree"],
            "reason": row["reason"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "subtitle_text": row["subtitle_text"],
        }
    )
pd.DataFrame(review_csv_rows).to_csv(REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")

print(f"total_questions: {len(questions)}", flush=True)
print(f"counts_by_type: {counts}", flush=True)
print(f"questions_path: {QUESTIONS_PATH}", flush=True)
print(f"answer_key_path: {ANSWER_KEY_PATH}", flush=True)
print(f"review_path: {REVIEW_PATH}", flush=True)
print(f"review_csv_path: {REVIEW_CSV_PATH}", flush=True)
print(f"skipped_samples: {skipped}", flush=True)
print(f"train_test_vid_overlap: {bool(train_vids & test_vids)}", flush=True)
