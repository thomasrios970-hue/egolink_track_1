"""
脚本作用：
使用 EgoLink / E3 的 train 标注、本地 Qwen2.5-Omni、视频、音频和字幕生成 MCQ 训练集草稿。

执行逻辑：
1. 从 data/annotation/data2.xlsx 读取标注；正式批量只使用 set=train 的样本。
2. emotion 和 reason 只根据标注字段生成，正确答案强制来自标注。
3. predict 使用当前视频、当前音频、当前字幕，以及之后最多 3 个子视频的视频、音频和字幕作为辅助线索。
4. ego_summary 使用当前视频、当前音频、当前字幕，以及之前最多 3 个子视频的视频、音频和字幕作为辅助线索。
5. 正式批量时输出到 question/train_question；如需小样本调试，可临时改顶部 TEST_VID/TEST_SUB_ID 和 output_dir。
6. 输出三个 jsonl 和一个 csv：questions、answer_key、review jsonl、review csv。

运行示例：
python code/track_1/produce_question/produce_question_train.py \
  --num_per_type 6000 \
  --seed 2100 \
  --gpu 2,3,4,5
"""

"""需要的库统一在这里导入"""
import argparse
import contextlib
import json
import logging
import os
import random
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

"""所有输入输出路径都从 config 已有路径拼接出来"""
subtitle_dir = Path(config.PATH_TO_DATA_DIR) / "subtext"
video_dir = Path(config.PATH_TO_DATA_DIR) / "E3" / "E3"
audio_dir = Path(config.PATH_TO_DATA_DIR) / "audio"
data_path = Path(config.PATH_TO_DATA_DIR) / "annotation" / "data2.xlsx"
model_path = Path(config.PATH_TO_MODEL_DIR) / "modelscope_model" / "Qwen2.5-Omni-7B"
output_dir = Path(config.PATH_TO_QUESTION_DIR) / "train_question"

QUESTIONS_PATH = output_dir / "local_train_questions.jsonl"
ANSWER_KEY_PATH = output_dir / "local_train_answer_key.jsonl"
REVIEW_PATH = output_dir / "local_train_review.jsonl"
REVIEW_CSV_PATH = output_dir / "local_train_review.csv"

TEST_VID = None
TEST_SUB_ID = None
MODEL_NAME = "Qwen2.5-Omni-7B"
# 之后如果换 Qwen3-Omni，只需要替换 call_qwen_omni 函数内部的模型加载和输入格式。
MAX_CONTEXT_SUBVIDEOS = 3
OMNI_MAX_RETRY = 3
REQUIRED_COLUMNS = {"Vid", "sub_id", "person", "emotion", "degree", "start_time", "end_time", "reason", "set"}
QUESTION_TYPES = ["emotion", "reason", "predict", "ego_summary"]
OPTION_LETTERS = ["A", "B", "C", "D"]
processor = None
model = None

"""参数解析器统一在这里设置，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--num_per_type", type=int, default=10, help="每类题目生成数量")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--gpu", default="4,5", help="使用哪些 GPU，如 4 或 4,5；为空则不设置")
parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
parser.add_argument("--temperature", type=float, default=0.2, help="本地 Qwen2.5-Omni 生成温度")
args = parser.parse_args()
if args.gpu.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
rng = random.Random(args.seed)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.utils import logging as transformers_logging
from qwen_omni_utils import process_mm_info

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


"""读取指定子视频完整 vtt 字幕文本，输入：Vid、sub_id -> 输出：字幕文本 str"""
def read_full_subtitle_text(Vid, sub_id) -> str:
    path = subtitle_dir / str(Vid) / f"{sub_id}.vtt"
    if not path.exists():
        return ""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if text and text != "WEBVTT" and "-->" not in text and not text.isdigit():
            lines.append(text)
    return " ".join(" ".join(lines).split())[:2000]


"""读取前后上下文媒体，输入：Vid、sub_id、before/after -> 输出：视频、音频、字幕字典"""
def get_context_media(Vid, sub_id, direction: str) -> dict:
    current_sub_id = int(float(sub_id))
    video_root = video_dir / str(Vid)
    sub_ids = sorted(int(path.stem) for path in video_root.glob("*.mp4") if path.stem.isdigit())
    sub_ids = [item for item in sub_ids if item < current_sub_id][-MAX_CONTEXT_SUBVIDEOS:] if direction == "before" else [item for item in sub_ids if item > current_sub_id][:MAX_CONTEXT_SUBVIDEOS]
    video_paths = [video_dir / str(Vid) / f"{item}.mp4" for item in sub_ids]
    audio_paths = [audio_dir / str(Vid) / f"{item}.wav" for item in sub_ids]
    subtitle_text = "\n".join(f"sub_id {item}: {read_full_subtitle_text(Vid, item)}" for item in sub_ids)
    return {"video_paths": [p for p in video_paths if p.exists()], "audio_paths": [p for p in audio_paths if p.exists()], "subtitle_text": subtitle_text}


"""构建提示词，输入：样本字典、题型、上下文字幕、当前字幕 -> 输出：prompt 字符串"""
def build_prompt(sample: dict, question_type: str, context_subtitle_text: str = "", subtitle_text: str = "") -> str:
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
            "future_subtitle_text": context_subtitle_text,
            "subtitle_text": subtitle_text,
        }
        task = (
            "Create an MCQ asking what intention, goal, or purpose the relevant person in the current segment most likely has next. "
            "This is an intention prediction question, not an outcome prediction question. "
            "The question must explicitly ask about intention, goal, or purpose; do not ask only what action the person will do next. "
            "The current sub_id segment is the question anchor. Use the future sub_id videos, subtitles, and audio only as auxiliary evidence "
            "for inferring the person's intention in the current segment. "
            "If subtitles and audio conflict, prefer the evidence that is clearer, more segment-relevant, and more consistent with the video. "
            "Do not make the future clips themselves the question target. "
            "All options must describe possible intentions, plans, or purposes, not later outcomes or completed events. "
            "Use a natural subject from the visible scene or annotation, such as I, the man in black, the woman in white, or another clearly identified person. "
            "Do not use vague phrases like 'the camera wearer' or 'the person' when a clearer subject is available. "
            "Each option must state whose intention it is and what goal they are trying to achieve, not just a bare action. "
            "For example, write 'the girl tries to turn on the TV to continue watching' instead of 'turn on the TV'."
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
            "previous_subtitle_text": context_subtitle_text,
            "subtitle_text": subtitle_text,
        }
        task = (
            "Create an MCQ asking for my high-level ego-centric summary of the current sub_id segment. "
            "The correct option should describe what I understand, feel, or experience in the current segment. "
            "The correct option must not be only an emotion word, only a reason rewrite, or only a next action. "
            "Use the previous sub_id videos, subtitles, and audio only as background clues for understanding the current segment. "
            "If subtitles and audio conflict, prefer the evidence that is clearer, more segment-relevant, and more consistent with the video. "
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
The incorrect_options array must contain exactly 3 wrong options. The correct_answer_text is stored separately.
Do not put 4 items in incorrect_options. Do not generate A/B/C/D labels yourself.

Question type: {question_type}
Task: {task}

Rules:
- Options should be concise and rely on the real segment content.
- For emotion and reason, use annotation fields only.
- For predict, future sub_id videos/subtitles/audio are only auxiliary evidence for inferring intention; do not ask for or answer with later outcomes.
- For predict, the question must ask about intention, goal, or purpose, not only the next visible action.
- For predict, use a clear natural subject from the current segment; it can be I or a visible person such as the man in black. Do not force first person if another visible person is the target.
- For predict, every answer option must include a subject and an intention or goal; do not output bare actions like "turn on the TV".
- For ego_summary, previous sub_id videos/subtitles/audio are background clues; the answer must summarize the current sub_id from my first-person perspective.
- If subtitle text and audio conflict, prefer the evidence that best matches the video and current localized segment.
- For ego_summary, the subject is always I; use I/me/my and never use 'the camera wearer' in the question or options.
- Incorrect options should be plausible misunderstandings.
- incorrect_options must contain exactly 3 items.
- Do not repeat correct_answer_text inside incorrect_options.
- Incorrect options must not repeat or be semantically equivalent to the correct answer.
- correct_answer_text and the three incorrect_options should have similar topic, length, and grammar.
- Keep only key information and avoid unrelated details.

Segment information:
{json.dumps(base, ensure_ascii=False)}
""".strip()


"""调用本地 Qwen2.5-Omni，输入：prompt、视频路径列表、音频路径列表 -> 输出：题目 JSON 字典"""
def call_qwen_omni(prompt: str, video_paths: list = None, audio_paths: list = None) -> dict:
    global processor, model
    if processor is None or model is None:
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path), local_files_only=True)
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
                enable_audio_output=False,
            )
    content = [{"type": "text", "text": prompt}]
    for path in video_paths or []:
        content.append({"type": "video", "video": str(path)})
    for path in audio_paths or []:
        content.append({"type": "audio", "audio": str(path)})
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You generate strict JSON multiple-choice questions."}]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)
    last_content = ""
    for retry_idx in range(OMNI_MAX_RETRY):
        output_ids = model.generate(
            **inputs,
            use_audio_in_video=False,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
        )
        last_content = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        content = re.sub(r"^```(?:json)?\s*", "", last_content)
        content = re.sub(r"\s*```$", "", content)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"warning: invalid JSON from local omni, retry {retry_idx + 1}/{OMNI_MAX_RETRY}", flush=True)
            continue
        if len(data.get("incorrect_options", [])) == 3:
            return data
        print(f"warning: incorrect_options not 3, retry {retry_idx + 1}/{OMNI_MAX_RETRY}", flush=True)
    raise ValueError("incorrect_options must have exactly 3 items, last model output: " + last_content[:500])


"""打乱选项，输入：正确答案、错误选项列表、随机对象 -> 输出：options 字典和答案字母"""
def make_options(correct_answer_text: str, incorrect_options: list, rng: random.Random) -> tuple[dict, str]:
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


"""读取 jsonl，输入：文件路径 -> 输出：字典列表 list"""
def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


"""覆盖写出 jsonl，输入：文件路径、字典列表 -> 输出：无输出"""
def write_jsonl(path: Path, rows: list):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


"""追加 jsonl，输入：文件路径、字典 -> 输出：无输出"""
def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


"""追加 csv，输入：文件路径、字典 -> 输出：无输出"""
def append_csv(path: Path, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


"""读取并清理断点输出，输入：无输入 -> 输出：questions、answer_keys、review_rows、used_qids"""
def load_resume_rows() -> tuple[list, list, list, set]:
    def unique_rows(rows):
        kept, seen = [], set()
        for row in rows:
            qid = row.get("qid")
            if qid and qid not in seen:
                kept.append(row)
                seen.add(qid)
        return kept

    q_rows = unique_rows(read_jsonl(QUESTIONS_PATH))
    a_rows = unique_rows(read_jsonl(ANSWER_KEY_PATH))
    r_rows = unique_rows(read_jsonl(REVIEW_PATH))
    qids = {row["qid"] for row in q_rows} & {row["qid"] for row in a_rows} & {row["qid"] for row in r_rows}
    r_rows = [
        {
            "qid": row["qid"],
            "Vid": row["Vid"],
            "sub_id": row["sub_id"],
            "person": row["person"],
            "question_type": row["question_type"],
            "question": row["question"],
            "A": row.get("A", ""),
            "B": row.get("B", ""),
            "C": row.get("C", ""),
            "D": row.get("D", ""),
            "answer": row["answer"],
            "answer_text": row["answer_text"],
            "emotion": row.get("emotion", ""),
            "degree": row.get("degree", ""),
            "reason": row.get("reason", ""),
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "subtitle_text": row.get("subtitle_text", ""),
            "audio_transcript": row.get("audio_transcript", ""),
        }
        for row in r_rows
        if row["qid"] in qids
    ]
    return [row for row in q_rows if row["qid"] in qids], [row for row in a_rows if row["qid"] in qids], r_rows, qids


"""代码块部分：读取训练样本，生成题目并追加写出"""
df = pd.read_excel(data_path)
missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
    raise ValueError(f"missing columns: {sorted(missing)}")

samples, skipped = [], 0
for _, row in df.iterrows():
    if TEST_VID is None and str(row["set"]).strip() != "train":
        continue
    if any(pd.isna(row[col]) or str(row[col]).strip() == "" for col in ["person", "emotion", "reason", "start_time", "end_time"]):
        skipped += 1
        continue
    start_time, end_time = float(row["start_time"]), float(row["end_time"])
    if start_time >= end_time:
        skipped += 1
        continue
    Vid = int(row["Vid"]) if float(row["Vid"]).is_integer() else row["Vid"]
    sub_id = int(row["sub_id"]) if float(row["sub_id"]).is_integer() else row["sub_id"]
    if TEST_VID is not None and (Vid != TEST_VID or sub_id != TEST_SUB_ID):
        continue
    if not (video_dir / str(Vid) / f"{sub_id}.mp4").exists():
        skipped += 1
        continue
    samples.append({
        "Vid": Vid,
        "sub_id": sub_id,
        "person": " ".join(str(row["person"]).split()),
        "emotion": " ".join(str(row["emotion"]).split()),
        "degree": row["degree"].item() if hasattr(row["degree"], "item") else row["degree"],
        "start_time": int(start_time) if start_time.is_integer() else start_time,
        "end_time": int(end_time) if end_time.is_integer() else end_time,
        "reason": " ".join(str(row["reason"]).split()),
    })

rng.shuffle(samples)
output_dir.mkdir(parents=True, exist_ok=True)
questions, answer_keys, review_rows, used_qids = load_resume_rows()  # 输入：无输入；输出：清理后的断点数据。
write_jsonl(QUESTIONS_PATH, questions)  # 输入：questions 路径和清理后题目；输出：覆盖写回。
write_jsonl(ANSWER_KEY_PATH, answer_keys)  # 输入：answer_key 路径和清理后答案；输出：覆盖写回。
write_jsonl(REVIEW_PATH, review_rows)  # 输入：review 路径和清理后复查行；输出：覆盖写回。
if review_rows:
    pd.DataFrame(review_rows).to_csv(REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")
elif REVIEW_CSV_PATH.exists():
    REVIEW_CSV_PATH.unlink()
counts = {question_type: sum(row.get("question_type") == question_type for row in questions) for question_type in QUESTION_TYPES}
print(f"resume_existing: {counts}", flush=True)

for question_type in QUESTION_TYPES:
    print(f"generating {question_type}...", flush=True)
    failed_samples = []
    for sample in samples:
        if counts[question_type] >= args.num_per_type:
            break
        safe_person = re.sub(r"\W+", "_", sample["person"]).strip("_") or "person"
        qid = f"{sample['Vid']}_{sample['sub_id']}_{safe_person}_{question_type}"
        if qid in used_qids:
            continue
        video_paths, audio_paths, context_subtitle_text = [], [], ""
        subtitle_text = read_full_subtitle_text(sample["Vid"], sample["sub_id"])  # 输入：Vid、sub_id；输出：当前字幕文本。
        if question_type == "predict":
            context = get_context_media(sample["Vid"], sample["sub_id"], "after")  # 输入：Vid、sub_id、after；输出：之后上下文媒体和字幕。
            video_paths = [video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"] + context["video_paths"]
            audio_paths = [audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"] + context["audio_paths"]
            context_subtitle_text = context["subtitle_text"]
        elif question_type == "ego_summary":
            context = get_context_media(sample["Vid"], sample["sub_id"], "before")  # 输入：Vid、sub_id、before；输出：之前上下文媒体和字幕。
            video_paths = [video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"] + context["video_paths"]
            audio_paths = [audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"] + context["audio_paths"]
            context_subtitle_text = context["subtitle_text"]
        try:
            prompt = build_prompt(sample, question_type, context_subtitle_text, subtitle_text)  # 输入：样本、题型、字幕；输出：prompt 字符串。
            result = call_qwen_omni(prompt, video_paths, audio_paths)  # 输入：prompt、视频路径、音频路径；输出：题目 JSON。
            correct = sample["emotion"] if question_type == "emotion" else sample["reason"] if question_type == "reason" else result["correct_answer_text"]
            options, answer = make_options(correct, result["incorrect_options"], rng)  # 输入：正确答案、错误选项、随机对象；输出：选项和答案字母。
        except Exception as e:
            print(f"warning: failed {qid} because {e}", flush=True)
            failed_samples.append(sample)
            continue
        question_row = {
            "qid": qid,
            "Vid": sample["Vid"],
            "sub_id": sample["sub_id"],
            "person": sample["person"],
            "question_type": question_type,
            "question": " ".join(str(result["question"]).split()),
            "options": options,
            "answer": answer,
            "answer_text": options[answer],
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
            "question": question_row["question"],
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
            "audio_transcript": "",
        }
        append_jsonl(QUESTIONS_PATH, question_row)  # 输入：questions 路径和训练题行；输出：追加写入 questions。
        append_jsonl(ANSWER_KEY_PATH, answer_row)  # 输入：answer_key 路径和答案行；输出：追加写入 answer_key。
        append_jsonl(REVIEW_PATH, review_row)  # 输入：review 路径和复查行；输出：追加写入 review jsonl。
        append_csv(REVIEW_CSV_PATH, review_row)  # 输入：review_csv 路径和复查行；输出：追加写入 review csv。
        used_qids.add(qid)
        questions.append(question_row)
        answer_keys.append(answer_row)
        review_rows.append(review_row)
        counts[question_type] += 1
        print(f"{question_type}: {counts[question_type]}/{args.num_per_type}", flush=True)

    if counts[question_type] < args.num_per_type and failed_samples:
        print(f"retry_failed_{question_type}: {len(failed_samples)}", flush=True)
        for sample in failed_samples:
            if counts[question_type] >= args.num_per_type:
                break
            safe_person = re.sub(r"\W+", "_", sample["person"]).strip("_") or "person"
            qid = f"{sample['Vid']}_{sample['sub_id']}_{safe_person}_{question_type}"
            if qid in used_qids:
                continue
            video_paths, audio_paths, context_subtitle_text = [], [], ""
            subtitle_text = read_full_subtitle_text(sample["Vid"], sample["sub_id"])  # 输入：Vid、sub_id；输出：当前字幕文本。
            if question_type == "predict":
                context = get_context_media(sample["Vid"], sample["sub_id"], "after")  # 输入：Vid、sub_id、after；输出：之后上下文媒体和字幕。
                video_paths = [video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"] + context["video_paths"]
                audio_paths = [audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"] + context["audio_paths"]
                context_subtitle_text = context["subtitle_text"]
            elif question_type == "ego_summary":
                context = get_context_media(sample["Vid"], sample["sub_id"], "before")  # 输入：Vid、sub_id、before；输出：之前上下文媒体和字幕。
                video_paths = [video_dir / str(sample["Vid"]) / f"{sample['sub_id']}.mp4"] + context["video_paths"]
                audio_paths = [audio_dir / str(sample["Vid"]) / f"{sample['sub_id']}.wav"] + context["audio_paths"]
                context_subtitle_text = context["subtitle_text"]
            try:
                prompt = build_prompt(sample, question_type, context_subtitle_text, subtitle_text)  # 输入：样本、题型、字幕；输出：prompt 字符串。
                result = call_qwen_omni(prompt, video_paths, audio_paths)  # 输入：prompt、视频路径、音频路径；输出：题目 JSON。
                correct = sample["emotion"] if question_type == "emotion" else sample["reason"] if question_type == "reason" else result["correct_answer_text"]
                options, answer = make_options(correct, result["incorrect_options"], rng)  # 输入：正确答案、错误选项、随机对象；输出：选项和答案字母。
            except Exception as e:
                print(f"warning: retry failed {qid} because {e}", flush=True)
                continue
            question_row = {
                "qid": qid,
                "Vid": sample["Vid"],
                "sub_id": sample["sub_id"],
                "person": sample["person"],
                "question_type": question_type,
                "question": " ".join(str(result["question"]).split()),
                "options": options,
                "answer": answer,
                "answer_text": options[answer],
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
                "question": question_row["question"],
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
                "audio_transcript": "",
            }
            append_jsonl(QUESTIONS_PATH, question_row)  # 输入：questions 路径和训练题行；输出：追加写入 questions。
            append_jsonl(ANSWER_KEY_PATH, answer_row)  # 输入：answer_key 路径和答案行；输出：追加写入 answer_key。
            append_jsonl(REVIEW_PATH, review_row)  # 输入：review 路径和复查行；输出：追加写入 review jsonl。
            append_csv(REVIEW_CSV_PATH, review_row)  # 输入：review_csv 路径和复查行；输出：追加写入 review csv。
            used_qids.add(qid)
            questions.append(question_row)
            answer_keys.append(answer_row)
            review_rows.append(review_row)
            counts[question_type] += 1
            print(f"{question_type}: {counts[question_type]}/{args.num_per_type}", flush=True)

    if counts[question_type] < args.num_per_type:
        print(f"warning: {question_type} only generated {counts[question_type]} questions", flush=True)

print(f"total_questions: {sum(counts.values())}", flush=True)
print(f"counts_by_type: {counts}", flush=True)
print(f"questions_path: {QUESTIONS_PATH}", flush=True)
print(f"answer_key_path: {ANSWER_KEY_PATH}", flush=True)
print(f"review_path: {REVIEW_PATH}", flush=True)
print(f"review_csv_path: {REVIEW_CSV_PATH}", flush=True)
print(f"skipped_samples: {skipped}", flush=True)
