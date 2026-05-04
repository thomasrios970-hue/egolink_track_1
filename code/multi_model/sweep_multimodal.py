"""
多模态融合参数自动搜索脚本。

脚本会围绕当前最有效的 HuBERT+CLIP attention 配置自动生成若干组参数，
依次调用 train_multimodal.py 训练。每个配置训练结束后读取对应 result log，
按指定指标排序，并把完整排行榜和最佳配置保存到 log 目录。

运行示例：
python code/multi_model/sweep_multimodal.py \
  --task sentiment \
  --modalities hubert,clip \
  --cuda-visible-devices 4 \
  --trials 12 \
  --epochs 80 \
  --metric test_macro_f1
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BEIJING_TZ = timezone(timedelta(hours=8))
PROJECT_ROOT = Path("/data/wzw/egolink_race")
DEFAULT_LOG_DIR = PROJECT_ROOT / "log"
TRAIN_SCRIPT = PROJECT_ROOT / "code/multi_model/train_multimodal.py"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="自动搜索多模态融合训练参数，并按结果指标保存排行榜。",
    )
    parser.add_argument("--task", choices=["sentiment", "emotion_sort"], default="sentiment")
    parser.add_argument("--modalities", default="hubert,clip")
    parser.add_argument("--fusion", choices=["attention", "concat"], default="attention")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--trials", type=positive_int, default=12)
    parser.add_argument("--epochs", type=positive_int, default=80)
    parser.add_argument("--early-stop-patience", type=non_negative_int, default=10)
    parser.add_argument("--num-workers", type=non_negative_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric",
        choices=["test_macro_f1", "test_acc", "test_weighted_f1"],
        default="test_macro_f1",
        help="用于选择最佳参数的指标。",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要运行的命令，不真正训练。",
    )
    return parser


def candidate_pool():
    # 第一组是当前 emotion_sort 已知表现最稳的 hubert-large+clip-p14 配置，
    # 后面围绕正则、融合头深度和不平衡策略做随机搜索。
    anchor = {
        "batch_size": 128,
        "lr": 1e-4,
        "weight_decay": 5e-3,
        "hidden_dim": 256,
        "dropout": 0.5,
        "label_smoothing": 0.1,
        "grad_clip": 1.0,
        "modality_dropout": 0.15,
        "num_layers": 1,
        "loss": "ce",
        "focal_gamma": 2.0,
        "class_weight": "none",
        "sampler": "none",
    }

    grid = {
        "batch_size": [96, 128, 192],
        "lr": [6e-5, 1e-4, 2e-4, 3e-4],
        "weight_decay": [1e-3, 3e-3, 5e-3, 8e-3],
        "hidden_dim": [256, 384, 512],
        "dropout": [0.35, 0.45, 0.5, 0.55],
        "label_smoothing": [0.03, 0.05, 0.08, 0.1],
        "grad_clip": [1.0],
        "modality_dropout": [0.0, 0.1, 0.15, 0.25],
        "num_layers": [1, 2],
    }
    imbalance_settings = [
        {"loss": "ce", "focal_gamma": 2.0, "class_weight": "none", "sampler": "none"},
        {"loss": "ce", "focal_gamma": 2.0, "class_weight": "sqrt_inv", "sampler": "none"},
        {"loss": "ce", "focal_gamma": 2.0, "class_weight": "effective", "sampler": "none"},
        {"loss": "ce", "focal_gamma": 2.0, "class_weight": "none", "sampler": "sqrt_inv"},
        {"loss": "focal", "focal_gamma": 1.5, "class_weight": "sqrt_inv", "sampler": "none"},
        {"loss": "focal", "focal_gamma": 2.0, "class_weight": "sqrt_inv", "sampler": "sqrt_inv"},
    ]

    keys = list(grid)
    candidates = []
    for values in itertools.product(*[grid[key] for key in keys]):
        base_candidate = dict(zip(keys, values))
        for imbalance in imbalance_settings:
            candidate = dict(base_candidate)
            candidate.update(imbalance)
            candidates.append(candidate)

    return anchor, candidates


def pick_candidates(trials, seed):
    anchor, candidates = candidate_pool()
    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected = [anchor]
    seen = {tuple(sorted(anchor.items()))}

    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        if key in seen:
            continue
        selected.append(candidate)
        seen.add(key)
        if len(selected) >= trials:
            break

    return selected[:trials]


def build_command(args, candidate, trial_index):
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--task",
        args.task,
        "--modalities",
        args.modalities,
        "--fusion",
        args.fusion,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(candidate["batch_size"]),
        "--lr",
        f"{candidate['lr']:g}",
        "--weight-decay",
        f"{candidate['weight_decay']:g}",
        "--hidden-dim",
        str(candidate["hidden_dim"]),
        "--dropout",
        f"{candidate['dropout']:g}",
        "--label-smoothing",
        f"{candidate['label_smoothing']:g}",
        "--grad-clip",
        f"{candidate['grad_clip']:g}",
        "--modality-dropout",
        f"{candidate['modality_dropout']:g}",
        "--num-layers",
        str(candidate["num_layers"]),
        "--loss",
        candidate["loss"],
        "--focal-gamma",
        f"{candidate['focal_gamma']:g}",
        "--class-weight",
        candidate["class_weight"],
        "--sampler",
        candidate["sampler"],
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--seed",
        str(args.seed + trial_index),
        "--num-workers",
        str(args.num_workers),
        "--log-dir",
        str(args.log_dir),
    ]
    if args.cuda_visible_devices is not None:
        command.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    return command


def command_to_text(command):
    return " ".join(
        f'"{item}"' if " " in item else item
        for item in command
    )


def run_command(command):
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    result_log = None
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
        if line.startswith("Result log:"):
            result_log = Path(line.split(":", 1)[1].strip())

    return_code = process.wait()
    return return_code, result_log, output_lines


def read_result(result_log):
    data = json.loads(result_log.read_text(encoding="utf-8"))
    return {
        "result_log": str(result_log),
        "test_acc": data["test_acc"],
        "test_macro_f1": data["test_macro_f1"],
        "test_weighted_f1": data["test_weighted_f1"],
        "test_attention": data.get("test_attention"),
        "best_epoch": data.get("best_epoch"),
        "best_val_macro_f1": data.get("best_val_macro_f1"),
        "checkpoint_paths": data.get("checkpoint_paths", []),
    }


def save_summary(args, started_at, records):
    args.log_dir.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda item: item.get(args.metric, -1.0), reverse=True)
    best = sorted_records[0] if sorted_records else None

    best_metric = f"{best[args.metric]:.4f}" if best else "none"
    stem = (
        f"{started_at}__sweep-multimodal__task-{args.task}"
        f"__mods-{args.modalities.replace(',', '+')}"
        f"__fusion-{args.fusion}__trials-{args.trials}"
        f"__metric-{args.metric}__best-{best_metric}"
    )
    summary_path = args.log_dir / f"{stem}.json"
    txt_path = args.log_dir / f"{stem}.txt"

    summary = {
        "metric": args.metric,
        "best": best,
        "records": sorted_records,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [f"metric: {args.metric}", ""]
    if best:
        lines.extend([
            "best command:",
            best["command"],
            "",
            f"best {args.metric}: {best[args.metric]:.6f}",
            f"test_acc: {best['test_acc']:.6f}",
            f"test_macro_f1: {best['test_macro_f1']:.6f}",
            f"test_weighted_f1: {best['test_weighted_f1']:.6f}",
            f"test_attention: {best.get('test_attention')}",
            "",
            "ranking:",
        ])
        for rank, record in enumerate(sorted_records, start=1):
            lines.append(
                f"{rank:02d}. {args.metric}={record[args.metric]:.6f} "
                f"acc={record['test_acc']:.6f} "
                f"macro={record['test_macro_f1']:.6f} "
                f"weighted={record['test_weighted_f1']:.6f} "
                f"trial={record['trial']}"
            )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, txt_path, best


def main():
    args = build_parser().parse_args()
    started_at = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")
    candidates = pick_candidates(args.trials, args.seed)

    records = []
    for trial_index, candidate in enumerate(candidates, start=1):
        command = build_command(args, candidate, trial_index)
        command_text = command_to_text(command)
        print("=" * 80)
        print(f"trial {trial_index}/{len(candidates)}")
        print(command_text)

        if args.dry_run:
            records.append({
                "trial": trial_index,
                "candidate": candidate,
                "command": command_text,
            })
            continue

        return_code, result_log, _ = run_command(command)
        record = {
            "trial": trial_index,
            "candidate": candidate,
            "command": command_text,
            "return_code": return_code,
        }

        if return_code != 0:
            record["error"] = f"command failed with return code {return_code}"
            records.append(record)
            continue
        if result_log is None or not result_log.exists():
            record["error"] = "result log was not found in command output"
            records.append(record)
            continue

        record.update(read_result(result_log))
        records.append(record)

        print(
            f"trial {trial_index} done: "
            f"macro={record['test_macro_f1']:.4f} "
            f"acc={record['test_acc']:.4f} "
            f"weighted={record['test_weighted_f1']:.4f}"
        )

    if args.dry_run:
        return

    valid_records = [record for record in records if args.metric in record]
    summary_path, txt_path, best = save_summary(args, started_at, valid_records)
    print("=" * 80)
    print("Sweep summary:", summary_path)
    print("Sweep txt:", txt_path)
    if best:
        print(f"Best {args.metric}: {best[args.metric]:.6f}")
        print("Best command:", best["command"])


if __name__ == "__main__":
    main()
