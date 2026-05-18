"""
脚本作用：
使用 ms-swift 对 Qwen2.5-Omni-7B 做 LoRA SFT，并在验证集上自动推理统计准确率。

执行逻辑：
1. 读取 train_question_SFT 和 val_question_SFT 中的 jsonl 数据。
2. 调用 swift sft 进行 LoRA 微调，增量权重保存到 checkpoints/track_1/STF/LoRA_STF。
3. 训练结束后选择 best checkpoint 或最新 checkpoint，调用 swift infer 在验证集上推理。
4. 解析推理结果，统计 emotion/reason/predict/ego_summary 和整体准确率。
5. 用模型名、本地北京时间和 ACC 重命名 checkpoint 文件夹，并写出同名日志。

运行示例：
/opt/conda/envs/egolink/bin/python code/track_1/STF/LoRA_STF.py \
  --gpu 2,3,4,5 \
  --save_steps 200 \
  --eval_steps 200
"""

"""需要的库统一在这里导入"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

"""所有输入输出路径都从 config 已有路径拼接出来"""
model_path = Path(config.PATH_TO_MODEL_DIR) / "modelscope_model" / "Qwen2.5-Omni-7B"
train_path = Path(config.PATH_TO_QUESTION_DIR) / "train_question_SFT" / "local_train_sft.jsonl"
val_path = Path(config.PATH_TO_QUESTION_DIR) / "val_question_SFT" / "local_val_sft.jsonl"
test_path = Path(config.PATH_TO_QUESTION_DIR) / "test_question_SFT" / "local_test_sft.jsonl"
checkpoint_root = Path("checkpoints") / "track_1" / "STF" / "LoRA_STF"
log_root = Path("log") / "track_1" / "STF" / "LoRA_STF"
swift_compat_dir = log_root / "swift_compat"

MODEL_NAME = "Qwen2.5-Omni-7B"
QUESTION_TYPES = ["emotion", "reason", "predict", "ego_summary"]
BEIJING_TZ = timezone(timedelta(hours=8))
SWIFT_CMD = shutil.which("swift") or str(Path(sys.executable).with_name("swift"))

"""参数解析器统一在这里设置，参数尽量少"""
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="2,3,4,5", help="使用哪些 GPU，如 2,3,4,5；为空则不设置")
parser.add_argument("--infer_gpu", default="", help="验证推理使用哪些 GPU，默认与 --gpu 一致")
parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--learning_rate", type=str, default="1e-4", help="学习率")
parser.add_argument("--num_train_epochs", type=str, default="1", help="训练 epoch 数")
parser.add_argument("--max_steps", type=int, default=0, help="最大训练步数，0 表示不传该参数")
parser.add_argument("--max_length", type=int, default=2048, help="最大长度")
parser.add_argument("--save_steps", type=int, default=100, help="保存间隔")
parser.add_argument("--eval_steps", type=int, default=100, help="验证间隔")
parser.add_argument("--logging_steps", type=int, default=5, help="日志间隔")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
parser.add_argument("--save_total_limit", type=int, default=2, help="最多保存 checkpoint 数")
parser.add_argument("--max_new_tokens", type=int, default=8, help="验证推理最大生成 token 数")
parser.add_argument("--dataloader_num_workers", type=int, default=0, help="数据加载进程数，视频解码不稳时建议为 0")
parser.add_argument("--no_filter_video_decode", action="store_true", help="不预先过滤当前环境无法解码的视频")
parser.add_argument("--progress_interval", type=int, default=100, help="过滤视频时每隔多少条打印一次进度")
parser.add_argument("--video_fps", type=str, default="0.2", help="训练时视频采样 fps")
parser.add_argument("--fps_max_frames", type=str, default="8", help="训练时每个视频最多采样帧数")
parser.add_argument("--video_max_pixels", type=str, default="200704", help="训练时视频单帧最大像素")
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


"""检查视频能否被当前 decord 解码，输入：视频路径、缓存字典 -> 输出：bool"""
def can_decode_video(video_path: str, cache: dict) -> bool:
    if video_path in cache:
        return cache[video_path]
    try:
        from decord import VideoReader, cpu
        cache[video_path] = len(VideoReader(video_path, ctx=cpu(0))) >= 2
        if not cache[video_path]:
            print(f"warning: skip bad video: {video_path} because decoded frames < 2", flush=True)
    except Exception as e:
        cache[video_path] = False
        print(f"warning: skip bad video: {video_path} because {e}", flush=True)
    return cache[video_path]


"""生成过滤后的 SFT 文件，输入：原始路径、输出路径 -> 输出：保留数量、跳过数量"""
def make_filtered_sft(input_path: Path, output_path: Path) -> tuple[int, int]:
    rows = read_jsonl(input_path)
    if args.no_filter_video_decode:
        write_jsonl(output_path, rows)
        return len(rows), 0
    cache, kept, skipped = {}, [], 0
    print(f"filter start: {input_path} total={len(rows)}", flush=True)
    for idx, row in enumerate(rows, 1):
        videos = row.get("videos", [])
        if all(can_decode_video(video, cache) for video in videos):
            kept.append(row)
        else:
            skipped += 1
        if idx % args.progress_interval == 0 or idx == len(rows):
            print(f"filter progress: {input_path.name} {idx}/{len(rows)} kept={len(kept)} skipped={skipped}", flush=True)
    write_jsonl(output_path, kept)
    return len(kept), skipped


"""生成推理用精简 SFT 文件，输入：完整验证集路径、输出路径 -> 输出：无输出"""
def make_infer_sft(input_path: Path, output_path: Path):
    keep_keys = ["messages", "videos", "audios"]
    rows = [{key: row[key] for key in keep_keys if key in row} for row in read_jsonl(input_path)]
    write_jsonl(output_path, rows)


"""运行命令并写入日志，输入：命令列表、环境变量、日志路径 -> 输出：无输出"""
def run_cmd(cmd: list, env: dict, run_log_path: Path):
    with run_log_path.open("a", encoding="utf-8") as f:
        f.write("\n\n" + "=" * 80 + "\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        print(" ".join(cmd), flush=True)
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


"""检查必要环境，输入：无输入 -> 输出：无输出"""
def check_required():
    if not Path(SWIFT_CMD).exists() and shutil.which(SWIFT_CMD) is None:
        raise RuntimeError("swift command not found，请先安装 ms-swift 后再运行本脚本。")
    for path in [model_path, train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"required path not found: {path}")


"""写 swift 兼容补丁，输入：补丁目录 -> 输出：sitecustomize.py 路径"""
def write_swift_compat(compat_dir: Path) -> Path:
    compat_dir.mkdir(parents=True, exist_ok=True)
    patch_path = compat_dir / "sitecustomize.py"
    patch_path.write_text(
        "try:\n"
        "    import transformers.integrations.tensor_parallel as tp\n"
        "    if not hasattr(tp, 'EmbeddingParallel') and hasattr(tp, 'ColwiseParallel'):\n"
        "        tp.EmbeddingParallel = tp.ColwiseParallel\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8"
    )
    return patch_path


"""选择 checkpoint，输入：训练输出目录 -> 输出：checkpoint 路径"""
def select_checkpoint(output_dir: Path) -> Path:
    state_path = output_dir / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            return Path(best)
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found in {output_dir}")
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))[-1]


"""从文本中抽取选项字母，输入：模型输出文本 -> 输出：A/B/C/D 或空字符串"""
def extract_answer(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", str(text).upper())
    return match.group(1) if match else ""


"""从推理结果行中取模型输出，输入：结果字典 -> 输出：输出文本"""
def get_prediction_text(row: dict) -> str:
    for key in ["response", "prediction", "predict", "output", "generated_text"]:
        if key in row:
            return row[key]
    messages = row.get("messages", [])
    if messages and isinstance(messages, list):
        return messages[-1].get("content", "")
    return ""


"""统计验证准确率，输入：验证集路径、推理结果路径 -> 输出：统计字典"""
def compute_accuracy(val_path: Path, result_path: Path) -> dict:
    val_rows = read_jsonl(val_path)
    result_rows = read_jsonl(result_path) if result_path.exists() else []
    stats = {question_type: {"correct": 0, "total": 0} for question_type in QUESTION_TYPES}
    unparsed = 0

    for idx, val_row in enumerate(val_rows):
        result_row = result_rows[idx] if idx < len(result_rows) else {}
        gold = val_row.get("answer") or val_row.get("messages", [{}])[-1].get("content", "")
        pred = extract_answer(get_prediction_text(result_row))
        question_type = val_row.get("question_type", "")
        if question_type not in stats:
            continue
        if not pred:
            unparsed += 1
        stats[question_type]["total"] += 1
        stats[question_type]["correct"] += int(pred == gold)

    total = sum(item["total"] for item in stats.values())
    correct = sum(item["correct"] for item in stats.values())
    stats["overall"] = {"correct": correct, "total": total}
    stats["unparsed"] = unparsed
    return stats


"""格式化准确率，输入：统计项 -> 输出：准确率字符串"""
def format_acc(item: dict) -> str:
    return f"{item['correct'] / item['total']:.4f}" if item["total"] else "0.0000"


"""用 swift infer 评测一个数据集，输入：名称、SFT路径、checkpoint、环境、stem -> 输出：统计字典和相关路径"""
def eval_split(split: str, sft_path: Path, checkpoint: Path, env: dict, stem: str) -> tuple[dict, Path, Path, Path, int, int]:
    filtered_path = log_root / f"{stem}_{split}_filtered.jsonl"
    infer_path = log_root / f"{stem}_{split}_infer.jsonl"
    prediction_path = log_root / f"{stem}_{split}_predictions.jsonl"
    kept_count, skipped_count = make_filtered_sft(sft_path, filtered_path)
    make_infer_sft(filtered_path, infer_path)
    infer_cmd = [
        SWIFT_CMD, "infer",
        "--adapters", str(checkpoint),
        "--val_dataset", str(infer_path),
        "--result_path", str(prediction_path),
        "--temperature", "0",
        "--max_new_tokens", str(args.max_new_tokens),
        "--remove_unused_columns", "true",
        "--truncation_strategy", "left",
    ]
    run_cmd(infer_cmd, env, run_log_path)
    return compute_accuracy(filtered_path, prediction_path), filtered_path, infer_path, prediction_path, kept_count, skipped_count


"""代码块部分：训练、验证、重命名 checkpoint、写日志"""
check_required()  # 输入：无输入；输出：检查必要环境。

now_time = datetime.now(BEIJING_TZ)
time_name = now_time.strftime("%Y%m%d_%H%M")
pending_stem = f"{MODEL_NAME}__{time_name}__ACC-pending"
pending_output_dir = checkpoint_root / pending_stem
run_log_path = log_root / f"{pending_stem}_run.txt"
prediction_path = log_root / f"{pending_stem}_predictions.jsonl"
filtered_train_path = log_root / f"{pending_stem}_train_filtered.jsonl"
filtered_val_path = log_root / f"{pending_stem}_val_filtered.jsonl"
infer_val_path = log_root / f"{pending_stem}_val_infer.jsonl"

checkpoint_root.mkdir(parents=True, exist_ok=True)
log_root.mkdir(parents=True, exist_ok=True)
compat_patch_path = write_swift_compat(swift_compat_dir)  # 输入：兼容补丁目录；输出：sitecustomize.py 路径。

env = os.environ.copy()
env["PYTHONPATH"] = str(swift_compat_dir.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
env.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
env["FPS"] = str(args.video_fps)
env["FPS_MAX_FRAMES"] = str(args.fps_max_frames)
env["VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
gpu_list = [gpu.strip() for gpu in args.gpu.split(",") if gpu.strip()]
if gpu_list:
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    env["NPROC_PER_NODE"] = str(len(gpu_list))

infer_env = env.copy()
infer_gpu_list = [gpu.strip() for gpu in (args.infer_gpu.strip() or args.gpu).split(",") if gpu.strip()]
if infer_gpu_list:
    infer_env["CUDA_VISIBLE_DEVICES"] = ",".join(infer_gpu_list)
    infer_env["NPROC_PER_NODE"] = str(len(infer_gpu_list))

train_count, train_skipped = make_filtered_sft(train_path, filtered_train_path)  # 输入：训练 SFT；输出：过滤后训练 SFT。
val_count, val_skipped = make_filtered_sft(val_path, filtered_val_path)  # 输入：验证 SFT；输出：过滤后验证 SFT。
make_infer_sft(filtered_val_path, infer_val_path)  # 输入：完整验证 SFT；输出：推理精简 SFT。

train_cmd = [
    SWIFT_CMD, "sft",
    "--model", str(model_path),
    "--dataset", str(filtered_train_path),
    "--val_dataset", str(filtered_val_path),
    "--tuner_type", "lora",
    "--lora_rank", str(args.lora_rank),
    "--lora_alpha", str(args.lora_alpha),
    "--target_modules", "all-linear",
    "--torch_dtype", "bfloat16",
    "--num_train_epochs", str(args.num_train_epochs),
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
    "--learning_rate", str(args.learning_rate),
    "--max_length", str(args.max_length),
    "--truncation_strategy", "left",
    "--eval_steps", str(args.eval_steps),
    "--save_steps", str(args.save_steps),
    "--save_total_limit", str(args.save_total_limit),
    "--logging_steps", str(args.logging_steps),
    "--output_dir", str(pending_output_dir),
    "--add_version", "false",
    "--warmup_ratio", "0.05",
    "--dataloader_num_workers", str(args.dataloader_num_workers),
]
if args.max_steps > 0:
    train_cmd.extend(["--max_steps", str(args.max_steps)])

run_cmd(train_cmd, env, run_log_path)  # 输入：训练命令、环境变量、运行日志；输出：完成 LoRA 训练。
selected_checkpoint = select_checkpoint(pending_output_dir)  # 输入：训练输出目录；输出：待评测 checkpoint。

infer_cmd = [
    SWIFT_CMD, "infer",
    "--adapters", str(selected_checkpoint),
    "--val_dataset", str(infer_val_path),
    "--result_path", str(prediction_path),
    "--temperature", "0",
    "--max_new_tokens", str(args.max_new_tokens),
    "--remove_unused_columns", "true",
    "--truncation_strategy", "left",
]
run_cmd(infer_cmd, infer_env, run_log_path)  # 输入：推理命令、推理环境变量、运行日志；输出：验证集预测文件。

stats = compute_accuracy(filtered_val_path, prediction_path)  # 输入：验证集和预测文件；输出：准确率统计。
test_stats, test_filtered_path, test_infer_path, test_prediction_path, test_count, test_skipped = eval_split(
    "test", test_path, selected_checkpoint, infer_env, pending_stem
)  # 输入：test SFT 和 checkpoint；输出：test 准确率统计和中间文件路径。
overall_acc = format_acc(stats["overall"])
test_overall_acc = format_acc(test_stats["overall"])
final_stem = f"{MODEL_NAME}__{time_name}__ACC-{test_overall_acc}"
final_output_dir = checkpoint_root / final_stem
final_log_path = log_root / f"{final_stem}.txt"
final_prediction_path = log_root / f"{final_stem}_predictions.jsonl"
final_filtered_train_path = log_root / f"{final_stem}_train_filtered.jsonl"
final_filtered_val_path = log_root / f"{final_stem}_val_filtered.jsonl"
final_infer_val_path = log_root / f"{final_stem}_val_infer.jsonl"
final_test_prediction_path = log_root / f"{final_stem}_test_predictions.jsonl"
final_filtered_test_path = log_root / f"{final_stem}_test_filtered.jsonl"
final_infer_test_path = log_root / f"{final_stem}_test_infer.jsonl"

if final_output_dir.exists():
    raise FileExistsError(f"final checkpoint dir already exists: {final_output_dir}")
pending_output_dir.rename(final_output_dir)
if prediction_path.exists():
    prediction_path.rename(final_prediction_path)
if filtered_train_path.exists():
    filtered_train_path.rename(final_filtered_train_path)
if filtered_val_path.exists():
    filtered_val_path.rename(final_filtered_val_path)
if infer_val_path.exists():
    infer_val_path.rename(final_infer_val_path)
if test_prediction_path.exists():
    test_prediction_path.rename(final_test_prediction_path)
if test_filtered_path.exists():
    test_filtered_path.rename(final_filtered_test_path)
if test_infer_path.exists():
    test_infer_path.rename(final_infer_test_path)
run_log_path.rename(log_root / f"{final_stem}_run.txt")

lines = [
    f"time: {now_time.strftime('%Y-%m-%d %H:%M')} Beijing Time",
    f"model: {MODEL_NAME}",
    f"model_path: {model_path}",
    f"train_path: {train_path}",
    f"val_path: {val_path}",
    f"test_path: {test_path}",
    f"filtered_train_path: {final_filtered_train_path}",
    f"filtered_val_path: {final_filtered_val_path}",
    f"infer_val_path: {final_infer_val_path}",
    f"filtered_test_path: {final_filtered_test_path}",
    f"infer_test_path: {final_infer_test_path}",
    f"filtered_train_count: {train_count}",
    f"filtered_train_skipped: {train_skipped}",
    f"filtered_val_count: {val_count}",
    f"filtered_val_skipped: {val_skipped}",
    f"filtered_test_count: {test_count}",
    f"filtered_test_skipped: {test_skipped}",
    f"gpu_arg: {args.gpu}",
    f"train_gpu: {','.join(gpu_list)}",
    f"infer_gpu: {','.join(infer_gpu_list)}",
    f"checkpoint_dir: {final_output_dir}",
    f"selected_checkpoint: {selected_checkpoint}",
    f"val_prediction_path: {final_prediction_path}",
    f"test_prediction_path: {final_test_prediction_path}",
    "",
    "LoRA parameters:",
    f"tuner_type: lora",
    f"lora_rank: {args.lora_rank}",
    f"lora_alpha: {args.lora_alpha}",
    f"target_modules: all-linear",
    f"learning_rate: {args.learning_rate}",
    f"num_train_epochs: {args.num_train_epochs}",
    f"max_steps: {args.max_steps}",
    f"max_length: {args.max_length}",
    f"gradient_accumulation_steps: {args.gradient_accumulation_steps}",
    f"dataloader_num_workers: {args.dataloader_num_workers}",
    f"video_fps: {args.video_fps}",
    f"fps_max_frames: {args.fps_max_frames}",
    f"video_max_pixels: {args.video_max_pixels}",
    f"swift_compat_patch: {compat_patch_path}",
    f"save_steps: {args.save_steps}",
    f"eval_steps: {args.eval_steps}",
    f"logging_steps: {args.logging_steps}",
    f"save_total_limit: {args.save_total_limit}",
    "",
    "Validation Accuracy:",
]
for question_type in QUESTION_TYPES:
    item = stats[question_type]
    lines.append(f"{question_type}_accuracy: {format_acc(item)} ({item['correct']}/{item['total']})")
lines.extend([
    f"overall_accuracy: {overall_acc} ({stats['overall']['correct']}/{stats['overall']['total']})",
    f"unparsed_predictions: {stats['unparsed']}",
    "",
    "Test Accuracy:",
])
for question_type in QUESTION_TYPES:
    item = test_stats[question_type]
    lines.append(f"{question_type}_accuracy: {format_acc(item)} ({item['correct']}/{item['total']})")
lines.extend([
    f"overall_accuracy: {test_overall_acc} ({test_stats['overall']['correct']}/{test_stats['overall']['total']})",
    f"unparsed_predictions: {test_stats['unparsed']}",
])
final_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"final_checkpoint_dir: {final_output_dir}", flush=True)
print(f"final_log_path: {final_log_path}", flush=True)
print(f"val_overall_accuracy: {overall_acc}", flush=True)
print(f"test_overall_accuracy: {test_overall_acc}", flush=True)
