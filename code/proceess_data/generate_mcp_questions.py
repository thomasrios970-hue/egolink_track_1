"""
根据 data2 情绪标注生成 MCP 四选一题目。

脚本作用：
1. 读取现有 data2 派生 manifest，也就是 data/manifest/emotion_sort_manifest.csv。
2. 从 person、emotion、start_time、end_time 等标注字段构造多选题。
3. 每道题保留 Vid/sub_id/video_path/audio_path/subtitle_path，后续 Qwen Omni baseline
   可以直接读取视频帧、字幕和 audio_json。
4. 默认生成 500 道题，输出到 data/question/mcp_data2_500.csv。

执行逻辑：
- 正确答案来自 data2 的 emotion 字段。
- 干扰项从其它 emotion 类别中随机抽取。
- 采样时尽量按 emotion 类别均衡，避免题目全部偏向 happy/surprised。
- 输出字段使用 question、A、B、C、D、answer，和 baseline 的 MCP/QA 输入格式保持一致。
- --mode fine 生成 8 类细粒度情绪题；--mode coarse 生成 4 类粗粒度情绪题。

运行示例：
python code/proceess_data/generate_mcp_questions.py
python code/proceess_data/generate_mcp_questions.py \
  --mode coarse \
  --output data/question/mcp_data2_500_coarse.csv
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


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
LETTERS = "ABCD"
DEFAULT_MANIFEST = Path(config.PATH_TO_AUDIO_MANIFEST).parent / "emotion_sort_manifest.csv"
DEFAULT_OUTPUT = Path(config.PATH_TO_PROCESSED_DIR).parent / "question" / "mcp_data2_500.csv"
DEFAULT_NUM_QUESTIONS = 500
DEFAULT_SEED = 2026
MODES = ["fine", "coarse"]

COARSE_OPTIONS = [
    "positive / pleased",
    "negative / upset",
    "shy, nervous, or socially tense",
    "surprised / unexpected reaction",
]
COARSE_EMOTION_MAP = {
    "happy": "positive / pleased",
    "angry": "negative / upset",
    "disgusted": "negative / upset",
    "sad": "negative / upset",
    "sarcastic": "shy, nervous, or socially tense",
    "scared": "shy, nervous, or socially tense",
    "shy": "shy, nervous, or socially tense",
    "surprised": "surprised / unexpected reaction",
}


QUESTION_TEMPLATES = [
    (
        "In this egocentric video clip, what emotion does the target person express? "
        "Target person: {person}. Focus on {time_hint}."
    ),
    (
        "Considering the video frames, subtitle, and audio summary, which emotion best "
        "describes the target person? Target person: {person}. Relevant time: {time_hint}."
    ),
    (
        "What is the most likely emotional state of {person} in this clip? "
        "Use the visual context, speech, and audio cues around {time_hint}."
    ),
]
COARSE_QUESTION_TEMPLATES = [
    (
        "What is the coarse emotional state of the target person in this egocentric "
        "video clip? Target person: {person}. Focus on {time_hint}."
    ),
    (
        "Considering the video frames, subtitle, and audio summary, which broad "
        "emotion group best describes {person} around {time_hint}?"
    ),
    (
        "Which coarse emotion category best matches the target person's reaction? "
        "Target person: {person}. Relevant time: {time_hint}."
    ),
]


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="根据 data2 情绪标注生成 500 道 MCP 四选一题。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="data2 情绪 manifest，默认 data/manifest/emotion_sort_manifest.csv。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 MCP 题目 CSV，默认 data/question/mcp_data2_500.csv。",
    )
    parser.add_argument(
        "--num-questions",
        type=positive_int,
        default=DEFAULT_NUM_QUESTIONS,
        help="生成题目数量，默认 500。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="随机种子，默认 2026。",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="fine",
        help="fine 生成 8 类细粒度情绪题；coarse 生成 4 类粗粒度情绪题。",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="只从指定 split 采样，例如 train/val/test；all 表示不限制。",
    )
    return parser


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean_text(value, default="unknown"):
    text = " ".join(str(value or "").split())
    return text if text else default


def normalize_emotion(value):
    emotion = clean_text(value, "").lower()
    return emotion if emotion in EMOTION_OPTIONS else ""


def time_hint(row):
    start = clean_text(row.get("start_time"), "")
    end = clean_text(row.get("end_time"), "")
    if start and end:
        return f"{start}s to {end}s"
    if start:
        return f"after {start}s"
    if end:
        return f"before {end}s"
    return "the annotated moment"


def valid_rows(rows, split):
    filtered = []
    for row in rows:
        emotion = normalize_emotion(row.get("emotion"))
        if not emotion:
            continue
        if split != "all" and clean_text(row.get("split"), "").lower() != split.lower():
            continue
        if not clean_text(row.get("Vid"), "") or not clean_text(row.get("sub_id"), ""):
            continue
        filtered.append(row)
    return filtered


def balanced_sample(rows, total, rng):
    by_emotion = defaultdict(list)
    for row in rows:
        by_emotion[normalize_emotion(row.get("emotion"))].append(row)
    for items in by_emotion.values():
        rng.shuffle(items)

    emotions = [emotion for emotion in EMOTION_OPTIONS if by_emotion.get(emotion)]
    if not emotions:
        return []

    base = total // len(emotions)
    remainder = total % len(emotions)
    selected = []
    leftovers = []
    for index, emotion in enumerate(emotions):
        quota = base + (1 if index < remainder else 0)
        items = by_emotion[emotion]
        selected.extend(items[:quota])
        leftovers.extend(items[quota:])

    if len(selected) < total:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])

    rng.shuffle(selected)
    return selected[:total]


def balanced_sample_by_label(rows, total, rng, label_fn, labels):
    by_label = defaultdict(list)
    for row in rows:
        label = label_fn(row)
        if label:
            by_label[label].append(row)
    for items in by_label.values():
        rng.shuffle(items)

    available_labels = [label for label in labels if by_label.get(label)]
    if not available_labels:
        return []

    base = total // len(available_labels)
    remainder = total % len(available_labels)
    selected = []
    leftovers = []
    for index, label in enumerate(available_labels):
        quota = base + (1 if index < remainder else 0)
        items = by_label[label]
        selected.extend(items[:quota])
        leftovers.extend(items[quota:])

    if len(selected) < total:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])

    rng.shuffle(selected)
    return selected[:total]


def build_options(correct_emotion, rng):
    distractors = [emotion for emotion in EMOTION_OPTIONS if emotion != correct_emotion]
    chosen = rng.sample(distractors, 3) + [correct_emotion]
    rng.shuffle(chosen)
    answer = LETTERS[chosen.index(correct_emotion)]
    return chosen, answer


def build_coarse_options(correct_label, rng):
    chosen = list(COARSE_OPTIONS)
    rng.shuffle(chosen)
    answer = LETTERS[chosen.index(correct_label)]
    return chosen, answer


def build_question(row, rng):
    person = clean_text(row.get("person"), "the target person")
    template = rng.choice(QUESTION_TEMPLATES)
    return template.format(person=person, time_hint=time_hint(row))


def build_coarse_question(row, rng):
    person = clean_text(row.get("person"), "the target person")
    template = rng.choice(COARSE_QUESTION_TEMPLATES)
    return template.format(person=person, time_hint=time_hint(row))


def coarse_label_for_row(row):
    return COARSE_EMOTION_MAP.get(normalize_emotion(row.get("emotion")), "")


def build_output_row(row, index, rng, mode):
    correct_emotion = normalize_emotion(row.get("emotion"))
    if mode == "coarse":
        answer_text = coarse_label_for_row(row)
        options, answer = build_coarse_options(answer_text, rng)
        question = build_coarse_question(row, rng)
        task_type = "coarse_emotion_mcp"
        question_id = f"mcp_data2_coarse_{index:04d}"
    else:
        answer_text = correct_emotion
        options, answer = build_options(correct_emotion, rng)
        question = build_question(row, rng)
        task_type = "emotion_mcp"
        question_id = f"mcp_data2_{index:04d}"
    return {
        "question_id": question_id,
        "task_type": task_type,
        "Vid": clean_text(row.get("Vid"), ""),
        "sub_id": clean_text(row.get("sub_id"), ""),
        "video_path": clean_text(row.get("video_path"), ""),
        "audio_path": clean_text(row.get("audio_path"), ""),
        "subtitle_path": clean_text(row.get("subtitle_path"), ""),
        "split": clean_text(row.get("split"), ""),
        "person": clean_text(row.get("person"), ""),
        "start_time": clean_text(row.get("start_time"), ""),
        "end_time": clean_text(row.get("end_time"), ""),
        "question": question,
        "A": options[0],
        "B": options[1],
        "C": options[2],
        "D": options[3],
        "answer": answer,
        "answer_text": answer_text,
        "source_emotion": correct_emotion,
        "source_coarse_emotion": coarse_label_for_row(row),
        "source_degree": clean_text(row.get("degree"), ""),
        "source_reason": clean_text(row.get("reason"), ""),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "task_type",
        "Vid",
        "sub_id",
        "video_path",
        "audio_path",
        "subtitle_path",
        "split",
        "person",
        "start_time",
        "end_time",
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "answer_text",
        "source_emotion",
        "source_coarse_emotion",
        "source_degree",
        "source_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    rows = valid_rows(read_rows(args.manifest), args.split)
    if len(rows) < args.num_questions:
        raise ValueError(
            f"not enough valid rows: requested {args.num_questions}, got {len(rows)}"
        )

    if args.mode == "coarse":
        sampled = balanced_sample_by_label(
            rows,
            args.num_questions,
            rng,
            coarse_label_for_row,
            COARSE_OPTIONS,
        )
    else:
        sampled = balanced_sample(rows, args.num_questions, rng)
    output_rows = [
        build_output_row(row, index, rng, args.mode)
        for index, row in enumerate(sampled)
    ]
    write_csv(args.output, output_rows)

    counts = defaultdict(int)
    for row in output_rows:
        counts[row["answer_text"]] += 1
    print(f"wrote {len(output_rows)} questions to {args.output}")
    labels = COARSE_OPTIONS if args.mode == "coarse" else EMOTION_OPTIONS
    print("label_distribution=" + ", ".join(
        f"{label}:{counts[label]}" for label in labels
    ))


if __name__ == "__main__":
    main()
