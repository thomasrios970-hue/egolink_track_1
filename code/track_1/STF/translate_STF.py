"""
脚本作用：
将已经生成好的 train_question 和 val_question 转成 ms-swift 多模态 SFT 对话格式。

执行逻辑：
1. 读取 data/question/train_question 和 data/question/val_question 中的 questions 与 answer_key。
2. 将每道 MCQ 转成 ms-swift 标准 messages 对话格式：system + user + assistant。
3. user 中使用 <video><audio> 作为多模态占位，顶层 videos/audios 保存真实路径；assistant 只输出正确答案字母。
4. 分别导出到 data/question/train_question_SFT 和 data/question/val_question_SFT。

运行示例：
python code/track_1/translate_STF.py
"""

"""需要的库统一在这里导入"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

"""所有输入输出路径都从 config 已有路径拼接出来"""
question_root = Path(config.PATH_TO_QUESTION_DIR)
video_dir = Path(config.PATH_TO_DATA_DIR) / "E3" / "E3"
audio_dir = Path(config.PATH_TO_DATA_DIR) / "audio"

DATASETS = {
    "train": {
        "input_dir": question_root / "train_question",
        "output_dir": question_root / "train_question_SFT",
        "questions_name": "local_train_questions.jsonl",
        "answer_name": "local_train_answer_key.jsonl",
        "output_name": "local_train_sft.jsonl",
    },
    "val": {
        "input_dir": question_root / "val_question",
        "output_dir": question_root / "val_question_SFT",
        "questions_name": "local_val_questions.jsonl",
        "answer_name": "local_val_answer_key.jsonl",
        "output_name": "local_val_sft.jsonl",
    },
}

"""参数解析器统一在这里设置，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "val", "all"], default="all", help="要转换的数据划分")
args = parser.parse_args()


"""读取 jsonl 文件，输入：jsonl 路径 -> 输出：字典列表"""
def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


"""写出 jsonl 文件，输入：jsonl 路径、字典列表 -> 输出：无输出"""
def write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


"""构建用户问题文本，输入：题目字典 -> 输出：SFT user 文本"""
def build_user_content(question: dict) -> str:
    options = question.get("options", {})
    option_text = "\n".join(f"{letter}. {options.get(letter, '')}" for letter in ["A", "B", "C", "D"])
    return f"""
<video><audio>
Please answer the multiple-choice question based on the given video and audio.
Return only one option letter: A, B, C, or D.

Target segment: {question["start_time"]}s to {question["end_time"]}s

Question:
{question["question"]}

Options:
{option_text}
""".strip()


"""转换一个数据划分，输入：split 名称、配置字典 -> 输出：转换条数 int"""
def convert_split(split: str, info: dict) -> int:
    questions = read_jsonl(info["input_dir"] / info["questions_name"])  # 输入：questions 文件；输出：题目列表。
    answer_map = {row["qid"]: row["answer"] for row in read_jsonl(info["input_dir"] / info["answer_name"])}  # 输入：answer_key 文件；输出：qid 到答案字典。
    sft_rows = []
    for question in questions:
        qid = question["qid"]
        answer = answer_map.get(qid, question.get("answer", ""))
        if answer not in ["A", "B", "C", "D"]:
            continue
        video_path = video_dir / str(question["Vid"]) / f"{question['sub_id']}.mp4"
        audio_path = audio_dir / str(question["Vid"]) / f"{question['sub_id']}.wav"
        sft_rows.append({
            "id": qid,
            "qid": qid,
            "Vid": question["Vid"],
            "sub_id": question["sub_id"],
            "person": question.get("person", ""),
            "question_type": question.get("question_type", ""),
            "start_time": question["start_time"],
            "end_time": question["end_time"],
            "answer": answer,
            "answer_text": question.get("answer_text", question.get("options", {}).get(answer, "")),
            "videos": [str(video_path)],
            "audios": [str(audio_path)],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for multimodal multiple-choice question answering."},
                {"role": "user", "content": build_user_content(question)},
                {"role": "assistant", "content": answer},
            ],
        })
    write_jsonl(info["output_dir"] / info["output_name"], sft_rows)  # 输入：输出路径和 SFT 列表；输出：写出 jsonl。
    return len(sft_rows)


"""代码块部分：转换 train/val 并打印必要信息"""
splits = ["train", "val"] if args.split == "all" else [args.split]
for split in splits:
    count = convert_split(split, DATASETS[split])  # 输入：划分名和路径配置；输出：转换条数。
    print(f"{split}_sft_count: {count}", flush=True)
    print(f"{split}_sft_path: {DATASETS[split]['output_dir'] / DATASETS[split]['output_name']}", flush=True)
