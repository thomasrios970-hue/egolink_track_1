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
  --num_per_type 1 \
  --seed 202657
"""

import argparse
import base64
import json
import os
import random
import re
import socket
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
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

QUESTIONS_PATH = output_dir / "local_test_questions.jsonl"
ANSWER_KEY_PATH = output_dir / "local_test_answer_key.jsonl"
REVIEW_PATH = output_dir / "local_test_review.jsonl"
REVIEW_CSV_PATH = output_dir / "local_test_review.csv"
MCQ_TRAIN_PATH = Path(config.PATH_TO_QUESTION_DIR) / "mcq_train.jsonl"

MODEL_NAME = "qwen3.6-plus"
QWEN_API_URL = os.environ.get(
    "QWEN_API_URL"
)
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
API_SLEEP_MIN = 1.0
API_SLEEP_MAX = 3.0
API_MAX_RETRY = 3
API_TIMEOUT = 120
MAX_COMPLETION_TOKENS = 4096
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
    作用：读取当前子视频的 16 帧，以及同一 Vid 中当前 sub_id 之前或之后最多 3 个子视频的 8 帧、字幕和音频转写文本。
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
            "Create an MCQ asking what the relevant person in the current segment most likely intends to do next. "
            "This is an intention prediction question, not an outcome prediction question. "
            "The current sub_id segment is the question anchor. Use the future sub_id frames, subtitles, and audio transcripts only as auxiliary evidence "
            "for inferring the person's intention in the current segment. "
            "If subtitles and Whisper transcripts conflict, prefer the evidence that is clearer, more segment-relevant, and more consistent with the frames. "
            "Do not make the future clips themselves the question target. "
            "All options must describe possible intentions, plans, or purposes, not later outcomes or completed events. "
            "Use a natural subject from the visible scene or annotation, such as I, the man in black, the woman in white, or another clearly identified person. "
            "Do not use vague phrases like 'the camera wearer' or 'the person' when a clearer subject is available."
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
            "The correct option should describe what I understand, feel, or experience in the current segment. "
            "The correct option must not be only an emotion word, only a reason rewrite, or only a next action. "
            "Use the previous sub_id frames, subtitles, and audio transcripts only as background clues for understanding the current segment. "
            "If subtitles and Whisper transcripts conflict, prefer the evidence that is clearer, more segment-relevant, and more consistent with the frames. "
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
- For predict, future sub_id frames/subtitles/audio transcripts are only auxiliary evidence for inferring intention; do not ask for or answer with later outcomes.
- For predict, use a clear natural subject from the current segment; it can be I or a visible person such as the man in black. Do not force first person if another visible person is the target.
- For ego_summary, previous sub_id frames/subtitles/audio transcripts are background clues; the answer must summarize the current sub_id from my first-person perspective.
- If subtitle text and Whisper transcript conflict, prefer the evidence that best matches the frames and current localized segment.
- For ego_summary, the subject is always I; use I/me/my and never use 'the camera wearer' in the question or options.
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
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
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
            with request.urlopen(req, timeout=API_TIMEOUT) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            error_text = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API request failed: HTTP {e.code}, {error_text}") from e
        except (TimeoutError, socket.timeout) as e:
            raise RuntimeError("API timeout, skip this attempt without inner retry") from e
        except URLError as e:
            if isinstance(e.reason, TimeoutError) or isinstance(e.reason, socket.timeout):
                raise RuntimeError("API timeout, skip this attempt without inner retry") from e
            raise
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


def read_jsonl(path: Path) -> list:
    """
    作用：读取 jsonl 文件。
    输入：path 文件路径。
    输出：字典列表 list。
    """
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list):
    """
    作用：覆盖写出 jsonl 文件。
    输入：path 文件路径、rows 字典列表。
    输出：无输出。
    """
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict):
    """
    作用：追加写出一行 jsonl。
    输入：path 文件路径、row 字典。
    输出：无输出。
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: dict):
    """
    作用：追加写出一行 csv。
    输入：path 文件路径、row 字典。
    输出：无输出。
    """
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def load_resume_rows() -> tuple[list, list, list, set]:
    """
    作用：读取已有输出，只保留 questions、answer_key、review 都完整存在的 qid。
    输入：无输入。
    输出：questions、answer_keys、review_rows、ready_qids。
    """
    q_rows = read_jsonl(QUESTIONS_PATH)  # 输入：questions 路径；输出：已有题目列表。
    a_rows = read_jsonl(ANSWER_KEY_PATH)  # 输入：answer_key 路径；输出：已有答案列表。
    r_rows = read_jsonl(REVIEW_PATH)  # 输入：review 路径；输出：已有复查列表。
    qids = {row["qid"] for row in q_rows} & {row["qid"] for row in a_rows} & {row["qid"] for row in r_rows}
    q_rows = [row for row in q_rows if row["qid"] in qids]
    a_rows = [row for row in a_rows if row["qid"] in qids]
    r_rows = [row for row in r_rows if row["qid"] in qids]
    return q_rows, a_rows, r_rows, qids


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
    if str(sample["Vid"]) in test_vids
]
rng.shuffle(candidates)

output_dir.mkdir(parents=True, exist_ok=True)
questions, answer_keys, review_rows, used_qids = load_resume_rows()  # 输入：无输入；输出：已有完整输出和可续跑 qid。
write_jsonl(QUESTIONS_PATH, questions)  # 输入：questions 路径和已有完整题目；输出：清理后的 questions jsonl。
write_jsonl(ANSWER_KEY_PATH, answer_keys)  # 输入：answer_key 路径和已有完整答案；输出：清理后的 answer_key jsonl。
write_jsonl(REVIEW_PATH, review_rows)  # 输入：review 路径和已有完整复查数据；输出：清理后的 review jsonl。
if review_rows:
    pd.DataFrame(review_rows).to_csv(REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")
elif REVIEW_CSV_PATH.exists():
    REVIEW_CSV_PATH.unlink()
counts = {question_type: sum(row["question_type"] == question_type for row in questions) for question_type in QUESTION_TYPES}
print(f"resume_existing: {counts}", flush=True)

def generate_one_question(sample: dict, question_type: str) -> bool:
    """
    作用：为一个样本生成一道题，并立刻追加写入四个输出文件。
    输入：sample 样本字典、question_type 题型字符串。
    输出：生成成功返回 True，已有 qid 也返回 True；API 或选项失败返回 False。
    """
    safe_person = re.sub(r"\W+", "_", sample["person"]).strip("_") or "person"
    qid_base = f"{sample['Vid']}_{sample['sub_id']}_{safe_person}_{question_type}"
    if qid_base in used_qids:
        return True
    qid = qid_base
    idx = 2
    while qid in used_qids:
        qid = f"{qid_base}_{idx}"
        idx += 1

    audio_path = audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"
    frame_paths = []
    context_frame_paths = []
    audio_text = ""
    context_audio_text = ""
    subtitle_text = ""
    context_subtitle_text = ""
    context_label = ""
    if question_type == "predict":
        frame_data = get_context_frame_paths(sample["Vid"], sample["sub_id"], "after")  # 输入：Vid、sub_id、after；输出：之后上下文帧/音频文本/字幕。
        frame_paths = frame_data["current_frame_paths"]
        context_frame_paths = frame_data["context_frame_paths"]
        audio_text = transcribe_audio(audio_path, sample["Vid"], sample["sub_id"])  # 输入：当前音频路径、Vid、sub_id；输出：Whisper 转写文本。
        context_audio_text = frame_data["context_audio_text"]
        subtitle_text = read_subtitle_text(sample["Vid"], sample["sub_id"], sample["start_time"], sample["end_time"])  # 输入：Vid、sub_id、start/end时间；输出：当前字幕文本。
        context_subtitle_text = frame_data["context_subtitle_text"]
        context_label = "Future sub_id 8-frame context from the same Vid, ordered by time:"
    elif question_type == "ego_summary":
        frame_data = get_context_frame_paths(sample["Vid"], sample["sub_id"], "before")  # 输入：Vid、sub_id、before；输出：之前上下文帧/音频文本/字幕。
        frame_paths = frame_data["current_frame_paths"]
        context_frame_paths = frame_data["context_frame_paths"]
        audio_text = transcribe_audio(audio_path, sample["Vid"], sample["sub_id"])  # 输入：当前音频路径、Vid、sub_id；输出：Whisper 转写文本。
        context_audio_text = frame_data["context_audio_text"]
        subtitle_text = read_subtitle_text(sample["Vid"], sample["sub_id"], sample["start_time"], sample["end_time"])  # 输入：Vid、sub_id、start/end时间；输出：当前字幕文本。
        context_subtitle_text = frame_data["context_subtitle_text"]
        context_label = "Previous sub_id 8-frame context from the same Vid, ordered by time:"

    prompt = build_prompt(sample, question_type, context_frame_paths, audio_text, context_audio_text, subtitle_text, context_subtitle_text)  # 输入：样本、题型、多模态文本线索；输出：prompt 字符串。
    try:
        result = call_qwen3(prompt, frame_paths, context_frame_paths, context_label) if question_type in ["predict", "ego_summary"] else call_qwen3(prompt)  # 输入：prompt 和可选图片；输出：题目 JSON。
        correct_answer_text = sample["emotion"] if question_type == "emotion" else sample["reason"] if question_type == "reason" else result["correct_answer_text"]
        options, answer = make_options(correct_answer_text, result["incorrect_options"], rng)  # 输入：正确答案、错误选项、随机对象；输出：选项和答案字母。
    except Exception as e:
        print(f"warning: failed {qid_base} because {e}", flush=True)
        return False

    question_row = {
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
    answer_row = {"qid": qid, "answer": answer}
    review_row = {
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
    append_jsonl(QUESTIONS_PATH, question_row)  # 输入：questions 路径和题目行；输出：追加写入 questions。
    append_jsonl(ANSWER_KEY_PATH, answer_row)  # 输入：answer_key 路径和答案行；输出：追加写入 answer_key。
    append_jsonl(REVIEW_PATH, review_row)  # 输入：review 路径和复查行；输出：追加写入 review jsonl。
    append_csv(REVIEW_CSV_PATH, review_row)  # 输入：review_csv 路径和复查行；输出：追加写入 review csv。
    questions.append(question_row)
    answer_keys.append(answer_row)
    review_rows.append(review_row)
    used_qids.add(qid)
    counts[question_type] += 1
    print(f"{question_type}: {counts[question_type]}/{args.num_per_type}", flush=True)
    return True


for question_type in QUESTION_TYPES:
    print(f"generating {question_type}...", flush=True)
    failed_samples = []
    for sample in candidates:
        if counts[question_type] >= args.num_per_type:
            break
        ok = generate_one_question(sample, question_type)  # 输入：样本、题型；输出：是否生成成功。
        if not ok:
            failed_samples.append(sample)

    if counts[question_type] < args.num_per_type and failed_samples:
        print(f"retry_failed_{question_type}: {len(failed_samples)}", flush=True)
        for sample in failed_samples:
            if counts[question_type] >= args.num_per_type:
                break
            generate_one_question(sample, question_type)  # 输入：失败样本、题型；输出：补试是否成功。

    if counts[question_type] < args.num_per_type:
        print(f"warning: {question_type} only generated {counts[question_type]} questions", flush=True)

print(f"total_questions: {len(questions)}", flush=True)
print(f"counts_by_type: {counts}", flush=True)
print(f"questions_path: {QUESTIONS_PATH}", flush=True)
print(f"answer_key_path: {ANSWER_KEY_PATH}", flush=True)
print(f"review_path: {REVIEW_PATH}", flush=True)
print(f"review_csv_path: {REVIEW_CSV_PATH}", flush=True)
print(f"skipped_samples: {skipped}", flush=True)
print(f"train_test_vid_overlap: {bool(train_vids & test_vids)}", flush=True)
