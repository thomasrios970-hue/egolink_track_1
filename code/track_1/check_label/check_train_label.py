"""
脚本作用：
使用本地 Qwen2.5-Omni-7B 或 Qwen3-Omni-30B-A3B-Thinking 检查 data/track_1/difference.csv 中样本的标注质量。

执行逻辑：
1. 读取 difference.csv，每条样本根据 Vid/sub_id 拼接对应音频路径。
2. 只把当前样本的视频和音频输入本地 Omni 模型，不使用字幕。
3. 判断视频音频质量、reason 是否匹配、emotion 是否匹配、是否和 track2 宽松相关。
4. 结果逐条追加写入 data/track_1/check_label/{模型名}/difference_train_label.csv，支持断点续跑。

运行示例：
python code/track_1/check_label/check_train_label.py --gpu 4,5,6,7
python code/track_1/check_label/check_train_label.py --gpu 4,5,6,7 --model_name qwen2.5 --limit 5
python code/track_1/check_label/check_train_label.py --gpu 4,5,6,7 --model_name all --limit 5
"""

"""需要的库统一在这里导入"""
import argparse
import contextlib
import csv
import gc
import json
import os
import re
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

"""参数解析器统一在这里设置和添加，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="4,5,6,7", help="使用哪些 GPU，如 4,5,6,7")
parser.add_argument("--model_name", choices=["qwen3", "qwen2.5", "all"], default="qwen3", help="使用哪个本地模型")
parser.add_argument("--temperature", type=float, default=0.1, help="生成温度")
parser.add_argument("--limit", type=int, default=0, help="调试时限制处理数量，0 表示全量")
parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="是否跳过已有结果")
parser.add_argument("--order", choices=["tail", "head"], default="tail", help="检查顺序，tail 表示从表尾开始，head 表示从表头开始")
args = parser.parse_args()

if args.gpu.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import pandas as pd
import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

"""所有输入输出路径都从 config 已有路径拼接出来"""
data_path = Path(config.PATH_TO_DATA_DIR) / "track_1" / "difference.csv"
output_root = Path(config.PATH_TO_DATA_DIR) / "track_1" / "check_label"
audio_root = Path(config.PATH_TO_DATA_DIR) / "audio"
model_root = Path(config.PATH_TO_MODEL_DIR) / "modelscope_model"

MODEL_CONFIG = {
    "qwen3": {
        "display_name": "Qwen3-Omni-30B-A3B-Thinking",
        "processor_cls": Qwen3OmniMoeProcessor,
        "model_cls": Qwen3OmniMoeForConditionalGeneration,
        "max_new_tokens": 512,
    },
    "qwen2.5": {
        "display_name": "Qwen2.5-Omni-7B",
        "processor_cls": Qwen2_5OmniProcessor,
        "model_cls": Qwen2_5OmniForConditionalGeneration,
        "max_new_tokens": 256,
    },
}

processor = None
model = None
current_model_name = None
OMNI_MAX_RETRY = 2

OUTPUT_COLUMNS = [
    "video_path",
    "audio_path",
    "Vid",
    "sub_id",
    "set",
    "person",
    "emotion",
    "degree",
    "start_time",
    "end_time",
    "reason",
    "is_video_file_exists",
    "is_audio_file_exists",
    "is_good_quality",
    "quality_reason",
    "reason_match",
    "reason_match_reason",
    "emotion_match",
    "emotion_match_reason",
    "track2_related",
    "track2_reason",
    "suggestion",
    "model_raw_output",
]


"""生成断点续跑键，输入：样本字典 -> 输出：唯一键字符串"""
def make_resume_key(row: dict) -> str:
    return "|".join(
        [
            str(row.get("video_path", "")),
            str(row.get("person", "")),
            str(row.get("start_time", "")),
            str(row.get("end_time", "")),
            str(row.get("reason", "")),
        ]
    )


"""读取已有结果键，输入：输出 CSV 路径 -> 输出：已完成样本键集合 set"""
def read_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return {make_resume_key(row) for row in df.to_dict("records")}


"""构建质量检查 prompt，输入：样本字典 -> 输出：prompt 字符串"""
def build_prompt(row: dict) -> str:
    sample = {
        "Vid": row["Vid"],
        "sub_id": row["sub_id"],
        "person": row["person"],
        "emotion": row["emotion"],
        "degree": row["degree"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "reason": row["reason"],
        "set": row.get("set", ""),
    }
    return f"""
You are checking one EgoLink / E3 localized annotation sample.
Use only the attached video and audio. Do not assume subtitles.
Focus on the segment from start_time to end_time inside this sub-video.

Annotation sample:
{json.dumps(sample, ensure_ascii=False)}

Judge these items:
1. is_good_quality: whether the video/audio is clear enough and temporally relevant enough to check the annotation.
2. reason_match: whether the annotated reason matches the visible/audible event in the target segment.
3. emotion_match: whether the annotated emotion matches the target person's expression, tone, action, or social context.
4. track2_related: use a broad standard. Shopping, ordering food, finding objects, navigation, price/menu/screen reading, task execution, tool use, multi-step decisions, and first-person actionable task clues are related.

Return strict JSON only. No markdown, no extra text.
Required schema:
{{
  "is_good_quality": true,
  "quality_reason": "short reason",
  "reason_match": true,
  "reason_match_reason": "short reason",
  "emotion_match": true,
  "emotion_match_reason": "short reason",
  "track2_related": false,
  "track2_reason": "short reason",
  "suggestion": "short manual review suggestion"
}}
""".strip()


"""解析模型 JSON，输入：模型原始输出字符串 -> 输出：解析后的字典"""
def parse_model_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("model output has no JSON object")
    data = json.loads(match.group(0))
    for key in ["is_good_quality", "reason_match", "emotion_match", "track2_related"]:
        if key not in data:
            raise ValueError(f"missing key: {key}")
        if isinstance(data[key], str):
            data[key] = data[key].strip().lower() == "true"
        else:
            data[key] = bool(data[key])
    return data


"""释放模型显存，输入：无输入 -> 输出：无输出"""
def release_model():
    global processor, model, current_model_name
    processor = None
    model = None
    current_model_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


"""调用本地 Omni，输入：模型配置、prompt、视频路径、音频路径 -> 输出：模型判断字典和原始输出字符串"""
def call_omni(model_key: str, prompt: str, video_path: Path, audio_path: Path) -> tuple[dict, str]:
    global processor, model, current_model_name
    model_info = MODEL_CONFIG[model_key]
    model_path = model_root / model_info["display_name"]
    if processor is None or model is None or current_model_name != model_key:
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        release_model()
        print(f"loading model: {model_path}", flush=True)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            processor = model_info["processor_cls"].from_pretrained(str(model_path), local_files_only=True)
            model = model_info["model_cls"].from_pretrained(
                str(model_path),
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
                enable_audio_output=False,
            )
        current_model_name = model_key
        print("model loaded", flush=True)

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a strict annotation quality checker. Return JSON only."}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt},
            ],
        },
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
            max_new_tokens=model_info["max_new_tokens"],
            do_sample=args.temperature > 0,
            temperature=args.temperature,
        )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        last_content = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        try:
            return parse_model_json(last_content), last_content
        except Exception:
            if retry_idx + 1 == OMNI_MAX_RETRY:
                raise
    raise ValueError("invalid model JSON")


"""构建缺失媒体结果，输入：样本字典、音频路径、缺失原因 -> 输出：结果字典"""
def make_missing_result(row: dict, audio_path: Path, reason: str) -> dict:
    return {
        "video_path": row["video_path"],
        "audio_path": str(audio_path),
        "Vid": row["Vid"],
        "sub_id": row["sub_id"],
        "set": row.get("set", ""),
        "person": row["person"],
        "emotion": row["emotion"],
        "degree": row["degree"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "reason": row["reason"],
        "is_video_file_exists": Path(str(row["video_path"])).exists(),
        "is_audio_file_exists": audio_path.exists(),
        "is_good_quality": False,
        "quality_reason": reason,
        "reason_match": False,
        "reason_match_reason": "cannot check because media is missing",
        "emotion_match": False,
        "emotion_match_reason": "cannot check because media is missing",
        "track2_related": False,
        "track2_reason": "cannot determine because media is missing",
        "suggestion": "check missing media path",
        "model_raw_output": "",
    }


"""追加写 CSV，输入：输出路径、结果字典 -> 输出：无输出"""
def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if need_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


"""统计并打印结果，输入：输出 CSV 路径 -> 输出：终端打印统计信息"""
def print_stats(path: Path):
    df = pd.read_csv(path)
    total = len(df)
    bad = int((df["is_good_quality"].astype(str).str.lower() == "false").sum())
    reason_bad = int((df["reason_match"].astype(str).str.lower() == "false").sum())
    emotion_bad = int((df["emotion_match"].astype(str).str.lower() == "false").sum())
    track2 = int((df["track2_related"].astype(str).str.lower() == "true").sum())

    def ratio(value: int) -> str:
        return f"{value / total:.4f}" if total else "0.0000"

    print(f"total_processed: {total}")
    print(f"bad_quality: {bad} ({ratio(bad)})")
    print(f"reason_mismatch: {reason_bad} ({ratio(reason_bad)})")
    print(f"emotion_mismatch: {emotion_bad} ({ratio(emotion_bad)})")
    print(f"track2_related: {track2} ({ratio(track2)})")
    print(f"output_path: {path}")


"""运行一个模型的检查，输入：模型 key、样本列表 -> 输出：写出该模型结果 CSV"""
def run_one_model(model_key: str, rows: list):
    model_name = MODEL_CONFIG[model_key]["display_name"]
    output_path = output_root / model_name / "difference_train_label.csv"
    done_keys = read_done_keys(output_path) if args.resume else set()  # 输入：输出 CSV；输出：已完成样本键集合。
    todo_count = sum(1 for row in rows if make_resume_key(row) not in done_keys)
    print(f"model_name: {model_name}", flush=True)
    print(f"total_input: {len(rows)}", flush=True)
    print(f"resume_skipped: {len(rows) - todo_count}", flush=True)
    print(f"todo_count: {todo_count}", flush=True)

    done_idx = 0
    for row in rows:
        key = make_resume_key(row)  # 输入：样本字典；输出：断点续跑键。
        if key in done_keys:
            continue
        done_idx += 1
        print(f"checking {done_idx}/{todo_count}: Vid={row['Vid']} sub_id={row['sub_id']} person={row['person']}", flush=True)
        video_path = Path(str(row["video_path"]))
        audio_path = audio_root / str(row["Vid"]) / f"{row['sub_id']}.wav"

        if not video_path.exists() or not audio_path.exists():
            miss_reason = "missing_video" if not video_path.exists() else "missing_audio"
            result = make_missing_result(row, audio_path, miss_reason)  # 输入：样本、音频路径、缺失原因；输出：缺失媒体结果。
        else:
            prompt = build_prompt(row)  # 输入：样本字典；输出：质量检查 prompt。
            try:
                model_result, raw = call_omni(model_key, prompt, video_path, audio_path)  # 输入：模型、prompt、视频、音频；输出：模型判断和原始输出。
                result = {
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "Vid": row["Vid"],
                    "sub_id": row["sub_id"],
                    "set": row.get("set", ""),
                    "person": row["person"],
                    "emotion": row["emotion"],
                    "degree": row["degree"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "reason": row["reason"],
                    "is_video_file_exists": True,
                    "is_audio_file_exists": True,
                    "is_good_quality": model_result.get("is_good_quality", False),
                    "quality_reason": model_result.get("quality_reason", ""),
                    "reason_match": model_result.get("reason_match", False),
                    "reason_match_reason": model_result.get("reason_match_reason", ""),
                    "emotion_match": model_result.get("emotion_match", False),
                    "emotion_match_reason": model_result.get("emotion_match_reason", ""),
                    "track2_related": model_result.get("track2_related", False),
                    "track2_reason": model_result.get("track2_reason", ""),
                    "suggestion": model_result.get("suggestion", ""),
                    "model_raw_output": raw,
                }
            except Exception as e:
                result = make_missing_result(row, audio_path, "model_check_failed")  # 输入：样本、音频路径、失败原因；输出：失败结果。
                result["model_raw_output"] = str(e)
                result["quality_reason"] = f"model_check_failed: {e}"

        append_csv(output_path, result)  # 输入：输出路径、结果字典；输出：追加一行 CSV。
        print(f"finished {done_idx}/{todo_count}: good={result['is_good_quality']} reason_match={result['reason_match']} emotion_match={result['emotion_match']} track2={result['track2_related']}", flush=True)

    print_stats(output_path)  # 输入：输出 CSV；输出：终端统计。
    release_model()  # 输入：无输入；输出：释放当前模型。


"""代码块部分：读取 difference.csv，按指定模型逐条检查并追加保存"""
if not data_path.exists():
    raise FileNotFoundError(f"input not found: {data_path}")

model_keys = ["qwen3", "qwen2.5"] if args.model_name == "all" else [args.model_name]
for key in model_keys:
    path = model_root / MODEL_CONFIG[key]["display_name"]
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")

df = pd.read_csv(data_path)
rows = df.to_dict("records")
if args.order == "tail":
    rows = list(reversed(rows))
if args.limit > 0:
    rows = rows[: args.limit]

for model_key in model_keys:
    run_one_model(model_key, rows)  # 输入：模型 key、样本列表；输出：该模型检查结果 CSV。
