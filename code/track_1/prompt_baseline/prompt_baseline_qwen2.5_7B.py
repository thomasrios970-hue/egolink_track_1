"""
脚本作用：
使用本地 Qwen2.5-Omni-7B 对已经生成好的本地测试集做 prompt baseline 测试。

执行逻辑：
1. 读取 question/test_question 下的 local_test_questions.jsonl 和 local_test_answer_key.jsonl。
2. 通过 --mode 控制输入证据：audio 只给音频，video 只给视频，text 只给字幕，multimodal 三者全给，all 依次跑四种。
3. 要求模型只输出答案字母，统计四类题准确率和整体准确率。
4. 将结果写入 log/track_1/prompt_baseline 对应子目录，日志文件名包含整体准确率和当前时间。

运行示例：
python code/track_1/prompt_baseline/prompt_baseline_qwen2.5_7B.py \
  --gpu 4,5 \
  --mode all \
  --temperature 0.2
"""

"""需要的库统一在这里导入"""
import argparse
import contextlib
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

"""所有输入输出路径都从 config 已有路径拼接出来"""
question_dir = Path(config.PATH_TO_QUESTION_DIR) / "test_question"
questions_path = question_dir / "local_test_questions.jsonl"
answer_key_path = question_dir / "local_test_answer_key.jsonl"
subtitle_dir = Path(config.PATH_TO_DATA_DIR) / "subtext"
video_dir = Path(config.PATH_TO_DATA_DIR) / "E3" / "E3"
audio_dir = Path(config.PATH_TO_DATA_DIR) / "audio"
model_path = Path(config.PATH_TO_MODEL_DIR) / "modelscope_model" / "Qwen2.5-Omni-7B"
base_log_dir = Path("log") / "track_1" / "prompt_baseline"

QUESTION_TYPES = ["emotion", "reason", "predict", "ego_summary"]
EVAL_MODES = ["audio", "video", "text", "multimodal"]
MODE_DIR = {"audio": "single_audio", "video": "single_video", "text": "single_text", "multimodal": "all"}
processor = None
model = None
BEIJING_TZ = timezone(timedelta(hours=8))

"""参数解析器统一在这里设置，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="4,5", help="使用哪些 GPU，如 4 或 4,5；为空则不设置")
parser.add_argument("--max_new_tokens", type=int, default=16, help="最大生成 token 数")
parser.add_argument("--temperature", type=float, default=0.2, help="本地 Qwen2.5-Omni 生成温度")
parser.add_argument("--limit", type=int, default=0, help="只测试前 N 道题，0 表示全部")
parser.add_argument("--mode", choices=["audio", "video", "text", "multimodal", "all"], default="all", help="输入证据类型；all 表示依次跑四种")
args = parser.parse_args()

if args.gpu.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.utils import logging as transformers_logging
from qwen_omni_utils import process_mm_info

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


"""读取 jsonl 文件，输入：jsonl 路径 -> 输出：字典列表"""
def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


"""构建测试提示词，输入：题目字典、字幕文本、输入模式 -> 输出：prompt 字符串"""
def build_prompt(question: dict, subtitle_text: str, mode: str) -> str:
    options = "\n".join(f"{key}. {value}" for key, value in question["options"].items())
    evidence = {
        "audio": "Use only the current audio as external evidence. Do not use video or subtitle text.",
        "video": "Use only the current video as external evidence. Do not use audio or subtitle text.",
        "text": "Use only the current subtitle text as external evidence. Do not use video or audio.",
        "multimodal": "Use the current video, current audio, and current subtitle text as external evidence.",
    }[mode]
    subtitle_part = f"subtitle_text: {subtitle_text}" if mode in ["text", "multimodal"] else "subtitle_text: not provided in this mode"
    return f"""
You are answering one multiple-choice question about the current localized video segment.
{evidence}
The target segment is from {question["start_time"]}s to {question["end_time"]}s in the current subvideo.
Return only one letter: A, B, C, or D. Do not output any explanation.

question_type: {question["question_type"]}
person: {question["person"]}
{subtitle_part}

Question:
{question["question"]}

Options:
{options}
""".strip()


"""调用本地 Qwen2.5-Omni，输入：prompt、视频路径、音频路径、输入模式 -> 输出：答案字母 str"""
def call_qwen_omni(prompt: str, video_path: Path, audio_path: Path, mode: str) -> str:
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
    if mode in ["video", "multimodal"]:
        content.append({"type": "video", "video": str(video_path)})
    if mode in ["audio", "multimodal"]:
        content.append({"type": "audio", "audio": str(audio_path)})
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "Answer MCQ questions. Return only A, B, C, or D."}]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)
    output_ids = model.generate(
        **inputs,
        use_audio_in_video=False,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
    )
    output_text = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    match = re.search(r"\b([ABCD])\b", output_text.upper())
    return match.group(1) if match else ""


"""运行一种输入模式的测试并写日志，输入：mode 字符串、题目列表、答案字典 -> 输出：整体准确率 float"""
def run_mode(mode: str, questions: list, answer_map: dict) -> float:
    stats = {question_type: {"correct": 0, "total": 0} for question_type in QUESTION_TYPES}
    skipped = 0

    for idx, question in enumerate(questions, 1):
        qid = question["qid"]
        answer = answer_map.get(qid, "")
        question_type = question["question_type"]
        video_path = video_dir / str(question["Vid"]) / f"{question['sub_id']}.mp4"
        audio_path = audio_dir / str(question["Vid"]) / f"{question['sub_id']}.wav"
        need_video = mode in ["video", "multimodal"]
        need_audio = mode in ["audio", "multimodal"]
        if not answer or (need_video and not video_path.exists()) or (need_audio and not audio_path.exists()):
            skipped += 1
            continue
        subtitle_text = read_full_subtitle_text(question["Vid"], question["sub_id"]) if mode in ["text", "multimodal"] else ""  # 输入：Vid、sub_id；输出：当前字幕文本。
        prompt = build_prompt(question, subtitle_text, mode)  # 输入：题目、字幕、模式；输出：测试 prompt。
        try:
            pred = call_qwen_omni(prompt, video_path, audio_path, mode)  # 输入：prompt、视频、音频、模式；输出：答案字母。
        except Exception as e:
            skipped += 1
            print(f"warning: failed mode={mode} qid={qid} video_path={video_path} audio_path={audio_path} because {e}", flush=True)
            continue
        stats[question_type]["total"] += 1
        stats[question_type]["correct"] += int(pred == answer)
        print(f"{mode} {idx}/{len(questions)} {qid} pred={pred or 'NA'} answer={answer}", flush=True)

    total = sum(item["total"] for item in stats.values())
    correct = sum(item["correct"] for item in stats.values())
    overall_acc = correct / total if total else 0.0
    now_time = datetime.now(BEIJING_TZ)
    now = now_time.strftime("%Y%m%d_%H%M%S")
    log_dir = base_log_dir / MODE_DIR[mode]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"prompt_baseline_qwen2.5_7B__{now}__ACC-{overall_acc:.4f}.txt"

    lines = [
        f"time: {now_time.strftime('%Y-%m-%d %H:%M:%S')} Beijing Time",
        f"model: Qwen2.5-Omni-7B",
        f"mode: {mode}",
        f"overall_accuracy: {overall_acc:.4f} ({correct}/{total})",
    ]
    for question_type in QUESTION_TYPES:
        item = stats[question_type]
        acc = item["correct"] / item["total"] if item["total"] else 0.0
        lines.append(f"{question_type}_accuracy: {acc:.4f} ({item['correct']}/{item['total']})")
    lines.append(f"skipped: {skipped}")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{mode}_accuracy: {overall_acc:.4f} ({correct}/{total})", flush=True)
    print(f"log_path: {log_path}", flush=True)
    return overall_acc


"""代码块部分：读取测试集，按选择的模式测试并写日志"""
questions = read_jsonl(questions_path)  # 输入：测试题路径；输出：测试题列表。
answer_map = {row["qid"]: row["answer"] for row in read_jsonl(answer_key_path)}  # 输入：答案路径；输出：qid 到答案的字典。
if args.limit > 0:
    questions = questions[:args.limit]

if args.mode == "all":
    summary = {mode: run_mode(mode, questions, answer_map) for mode in EVAL_MODES}  # 输入：四种模式；输出：各模式整体准确率。
    print(f"all_mode_accuracy: {summary}", flush=True)
else:
    run_mode(args.mode, questions, answer_map)  # 输入：当前选择模式；输出：该模式整体准确率。
