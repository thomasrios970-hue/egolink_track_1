"""
根据 EgoLink / E3 标注、字幕、音频和视频帧生成本地 MCQ 测试集

执行逻辑：
1. 从 data/annotation/data2.xlsx 读取 EgoLink / E3 标注。
2. 过滤有效定位片段，并按 Vid 隔离训练集和本地测试集候选。
3. emotion 和 reason 只使用标注文件生成，保证测试题干干净。
4. predict 和 ego_summary 只使用标注中的 start_time/end_time，并结合字幕文本、音频和抽帧图片生成。
5. 写出题目文件和答案文件两个 JSONL。

运行示例：
python code/track_1/produce_question_test.py \
  --num_per_type 5 \
  --seed 44
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
data_path = Path(config.PATH_TO_DATA_DIR) / "annotation" / "data2.xlsx"
output_dir = Path(config.PATH_TO_QUESTION_DIR) / "test_question"

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
parser.add_argument("--num_per_type", type=int, default=2, help="每类题目生成数量")
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


def build_prompt(sample: dict, question_type: str, subtitle_text: str) -> str:
    """
    作用：为一个样本和问题类型构建 Qwen3 输入提示词。
    输入：sample 字典、question_type 字符串、subtitle_text 字符串。
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
            "start_time": sample["start_time"],
            "end_time": sample["end_time"],
            "video_path": str(video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"),
            "audio_path": str(audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"),
            "frame_dir": str(frame_root / str(sample["Vid"]) / str(sample["sub_id"])),
            "subtitle_text": subtitle_text,
        }
        task = (
            "Create an MCQ asking what will most likely happen next or what the main participant will most likely intend to do next. "
            "Use only the start_time/end_time from the annotation plus the attached video frames, audio, and subtitle text. "
            "Do not use annotated person, emotion, or reason as generation evidence. "
            "All options must describe next-step behavioral intentions."
        )
    else:
        base = {
            "Vid": sample["Vid"],
            "sub_id": sample["sub_id"],
            "start_time": sample["start_time"],
            "end_time": sample["end_time"],
            "video_path": str(video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"),
            "audio_path": str(audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"),
            "frame_dir": str(frame_root / str(sample["Vid"]) / str(sample["sub_id"])),
            "subtitle_text": subtitle_text,
        }
        task = (
            "Create an MCQ asking for my high-level ego-centric summary of the social event. "
            "The correct option should describe what I, as the camera wearer, understand, feel, or experience in the segment. "
            "The correct option must not be only an emotion word, only a reason rewrite, or only a next action. "
            "Use only the start_time/end_time from the annotation plus the attached video frames, audio, and subtitle text. "
            "Do not use annotated person, emotion, or reason as generation evidence."
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
- For predict and ego_summary, use only start_time/end_time from annotation plus attached video frames, audio, and subtitle text.
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


def call_qwen3(prompt: str, frame_paths: list = None, audio_path: Path = None) -> dict:
    """
    作用：使用 API 模式调用 Qwen3，并解析模型返回的严格 JSON。
    输入：prompt 字符串；可选 frame_paths 抽帧图片路径列表、audio_path 音频路径。
    输出：包含 question、correct_answer_text、incorrect_options 的字典。
    """
    content = [{"type": "text", "text": prompt}]
    if audio_path is not None:
        audio_base64 = encode_media(audio_path)  # 输入：音频路径；输出：音频 base64 字符串。
        if audio_base64:
            content.append({"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}})
    if frame_paths is not None:
        for frame_path in frame_paths:
            image_base64 = encode_media(frame_path)  # 输入：抽帧图片路径；输出：图片 base64 字符串。
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
        if content is not None and str(content).strip():
            content = str(content).strip()
            break
        print(f"warning: empty API content, retry {retry_idx + 1}/{API_MAX_RETRY}", flush=True)
        content = None
    if content is None:
        raise RuntimeError(
            "API response has no message.content after retries. Full response: "
            + json.dumps(result, ensure_ascii=False)
        )
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    if len(data.get("incorrect_options", [])) != 3:
        raise ValueError("incorrect_options must have exactly 3 items")
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

candidates = [sample for sample in samples if str(sample["Vid"]) in test_vids]
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
        prompt = build_prompt(sample, question_type, subtitle_text)  # 输入：样本、题型、字幕；输出：Qwen3 prompt 字符串。
        video_path = video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"
        audio_path = audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"
        frame_paths = sorted((frame_root / str(sample["Vid"]) / str(sample["sub_id"])).glob("*.jpg"))
        if question_type in ["predict", "ego_summary"]:
            # 输入：prompt、抽帧图片路径列表、音频路径；输出：多模态生成的题目 JSON 字典。
            result = call_qwen3(prompt, frame_paths, audio_path)
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
            "note": "正式题目文件不包含这些依据。emotion/reason 只用标注文件；predict/ego_summary 只用标注 start_time/end_time 加文本、音频和视频帧。",
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
            generation_basis["annotation_fields_used"] = ["start_time", "end_time"]
            generation_basis["annotation_person_emotion_reason_used"] = False
            generation_basis["video_path"] = str(video_path)
            generation_basis["audio_path"] = str(audio_path)
            generation_basis["frame_paths"] = [str(path) for path in frame_paths]
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
pd.DataFrame(review_rows).to_csv(REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")

print(f"total_questions: {len(questions)}", flush=True)
print(f"counts_by_type: {counts}", flush=True)
print(f"questions_path: {QUESTIONS_PATH}", flush=True)
print(f"answer_key_path: {ANSWER_KEY_PATH}", flush=True)
print(f"review_path: {REVIEW_PATH}", flush=True)
print(f"review_csv_path: {REVIEW_CSV_PATH}", flush=True)
print(f"skipped_samples: {skipped}", flush=True)
print(f"train_test_vid_overlap: {bool(train_vids & test_vids)}", flush=True)
