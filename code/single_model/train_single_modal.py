"""
统一训练单模态特征分类 baseline。

支持 HuBERT、MacBERT 和 CLIP 三种已经提取好的 .npy 特征。脚本会读取
manifest.csv，按 Vid 和 sub_id 拼接特征路径，过滤缺失特征，然后训练一个
MLP 分类器，并按验证集 macro F1 保存最优模型。

运行示例：
python code/single_model/train_single_modal.py \
  --modality clip \
  --cuda-visible-devices 4 \
  --epochs 30 \
  --batch-size 128
"""

import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

DEFAULT_TASK_MANIFESTS = {
    "sentiment": Path("/data/wzw/egolink_race/data/manifest/emotion_degree_audio_manifest.csv"),
    "emotion_sort": Path("/data/wzw/egolink_race/data/manifest/emotion_sort_manifest.csv"),
}
DEFAULT_FEATURE_ROOTS = {
    "hubert": Path("/data/wzw/egolink_race/feature/hubert_large"),
    "macbert": Path("/data/wzw/egolink_race/feature/macbert_large"),
    "clip": Path("/data/wzw/egolink_race/feature/clip_large"),
}
DEFAULT_SAVE_ROOT = Path("/data/wzw/egolink_race/checkpoints/single_model")
DEFAULT_LOG_DIR = Path("/data/wzw/egolink_race/log")
BEIJING_TZ = timezone(timedelta(hours=8))
TASK_LABEL_MAPS = {
    "sentiment": {-3: 0, -2: 1, -1: 2, 1: 3, 2: 4, 3: 5},
    "emotion_sort": {
        "angry": 0,
        "disgusted": 1,
        "happy": 2,
        "sad": 3,
        "sarcastic": 4,
        "scared": 5,
        "shy": 6,
        "surprised": 7,
    },
}


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


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


def dropout_float(value):
    value = float(value)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return value


def non_negative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return value

"""
# 1. 创建解析器
parser = argparse.ArgumentParser(description="训练脚本")

# 2. 添加参数
parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
parser.add_argument("--lr", type=float, default=0.001, help="学习率")

# 3. 解析命令行参数
args = parser.parse_args()

# 4. 使用参数
print(f"训练轮数: {args.epochs}, 学习率: {args.lr}")=
"""
def build_parser():
    parser = argparse.ArgumentParser(
        description="训练 HuBERT、MacBERT 或 CLIP 单模态 .npy 特征的 MLP baseline。",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_LABEL_MAPS),
        default="sentiment",
        help="训练任务；sentiment 是 -3 到 3 六分类，emotion_sort 是 happy 等八分类。",
    )
    parser.add_argument(
        "--modality",
        choices=sorted(DEFAULT_FEATURE_ROOTS),
        required=True,
        help="选择训练的单模态特征。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest CSV 路径；不传则按 task 使用默认 manifest。",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="特征 .npy 目录；不传则按 modality 使用默认目录。",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="模型和结果保存目录；不传则保存到 checkpoints/single_model/<task>/<modality>。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="实验摘要文件保存目录；文件名会记录时间、参数、种子和最终指标。",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="设置 CUDA_VISIBLE_DEVICES，例如 4；不传则使用当前环境变量。",
    )
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--model-type",
        choices=["mlp", "deep_mlp", "residual_mlp"],
        default="mlp",
        help="分类头结构；mlp 是旧 baseline，residual_mlp 更适合冲更高分。",
    )
    parser.add_argument("--hidden-dim", type=positive_int, default=256)
    parser.add_argument(
        "--num-layers",
        type=positive_int,
        default=2,
        help="deep_mlp/residual_mlp 使用的隐藏层数量。",
    )
    parser.add_argument("--dropout", type=dropout_float, default=0.3)
    parser.add_argument(
        "--label-smoothing",
        type=dropout_float,
        default=0.0,
        help="CrossEntropyLoss 的 label smoothing，默认关闭。",
    )
    parser.add_argument(
        "--grad-clip",
        type=non_negative_float,
        default=0.0,
        help="梯度裁剪阈值；0 表示关闭。",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=non_negative_int,
        default=0,
        help="验证集 macro F1 连续多少轮不提升就停止；0 表示关闭。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=non_negative_int, default=4)
    return parser


def set_seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(args):
    feature_root = args.feature_root
    if feature_root is None:
        feature_root = DEFAULT_FEATURE_ROOTS[args.modality]

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = DEFAULT_SAVE_ROOT / args.task / args.modality

    manifest = args.manifest
    if manifest is None:
        manifest = DEFAULT_TASK_MANIFESTS[args.task]

    return manifest, feature_root, save_dir


def normalize_label(raw_label, task):
    if task == "sentiment":
        return int(raw_label)
    raise ValueError(f"unsupported task: {task}")


def get_label(row, task):
    if task == "emotion_sort" and "emotion" in row:
        return str(row["emotion"]).strip().lower()
    return normalize_label(row["label"], task)


def build_dataframe(manifest_path, feature_root, task, label_map):
    df = pd.read_csv(manifest_path)
    required_columns = {"Vid", "sub_id", "label", "split"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"manifest missing columns: {sorted(missing_columns)}")

    rows = []
    split_total = df["split"].value_counts().to_dict()
    split_usable = {}

    for _, row in df.iterrows():
        label = get_label(row, task)
        if label not in label_map:
            raise ValueError(f"unsupported label: {label}")

        feature_path = feature_root / f"{row['Vid']}_{row['sub_id']}.npy"
        if not feature_path.exists():
            feature_path = feature_root / f"{row['Vid']}" / f"{row['sub_id']}.npy"
        if not feature_path.exists():
            continue

        item = row.to_dict()
        item["feature_path"] = str(feature_path)
        item["label_id"] = label_map[label]
        rows.append(item)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError(f"no usable samples found in feature root: {feature_root}")

    split_usable = out_df["split"].value_counts().to_dict()
    for split in ["train", "val", "test"]:
        total = int(split_total.get(split, 0))
        usable = int(split_usable.get(split, 0))
        print(f"{split}: usable={usable}, missing={total - usable}, total={total}")

    return out_df


class FeatureDataset:
    def __init__(self, df, torch):
        self.paths = df["feature_path"].tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.torch = torch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx]).astype(np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        y = self.labels[idx]
        return self.torch.from_numpy(x), self.torch.tensor(y, dtype=self.torch.long)


def infer_input_dim(feature_path):
    feature = np.load(feature_path)
    if feature.ndim != 1:
        feature = feature.reshape(-1)
    return int(feature.shape[0])


def make_model(input_dim, hidden_dim, dropout, num_classes, nn, model_type, num_layers):
    if model_type == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    if model_type == "deep_mlp":
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, num_classes))
        return nn.Sequential(*layers)

    if model_type == "residual_mlp":
        return ResidualMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            num_layers=num_layers,
            nn=nn,
        )

    raise ValueError(f"unsupported model type: {model_type}")


class ResidualMLP:
    def __init__(self, input_dim, hidden_dim, dropout, num_classes, num_layers, nn):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output(x)

    def train(self, mode=True):
        self.input.train(mode)
        self.blocks.train(mode)
        self.output.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self.input.to(*args, **kwargs)
        self.blocks.to(*args, **kwargs)
        self.output.to(*args, **kwargs)
        return self

    def parameters(self):
        yield from self.input.parameters()
        yield from self.blocks.parameters()
        yield from self.output.parameters()

    def state_dict(self):
        return {
            "input": self.input.state_dict(),
            "blocks": self.blocks.state_dict(),
            "output": self.output.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.input.load_state_dict(state_dict["input"])
        self.blocks.load_state_dict(state_dict["blocks"])
        self.output.load_state_dict(state_dict["output"])


def make_old_model(input_dim, hidden_dim, dropout, num_classes, nn):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def compute_metrics(labels, preds, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    acc = float((labels == preds).mean()) if len(labels) else 0.0

    f1_scores = []
    supports = []
    for cls in range(num_classes):
        true_positive = int(((preds == cls) & (labels == cls)).sum())
        false_positive = int(((preds == cls) & (labels != cls)).sum())
        false_negative = int(((preds != cls) & (labels == cls)).sum())
        support = int((labels == cls).sum())

        precision_den = true_positive + false_positive
        recall_den = true_positive + false_negative
        precision = true_positive / precision_den if precision_den else 0.0
        recall = true_positive / recall_den if recall_den else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        f1_scores.append(f1)
        supports.append(support)

    macro_f1 = float(np.mean(f1_scores))
    total_support = int(np.sum(supports))
    weighted_f1 = (
        float(np.average(f1_scores, weights=supports))
        if total_support
        else 0.0
    )

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def evaluate(model, loader, device, torch, num_classes):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()

            preds.extend(pred.tolist())
            labels.extend(y.numpy().tolist())

    return compute_metrics(labels, preds, num_classes)


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def format_metric(value):
    return f"{value:.4f}"


def build_run_stem(args, started_at, best_epoch, test_metrics):
    parts = [
        started_at,
        f"task-{args.task}",
        f"mod-{args.modality}",
        f"seed-{args.seed}",
        f"ep-{args.epochs}",
        f"bs-{args.batch_size}",
        f"lr-{args.lr:g}",
        f"wd-{args.weight_decay:g}",
        f"model-{args.model_type}",
        f"hid-{args.hidden_dim}",
        f"layers-{args.num_layers}",
        f"drop-{args.dropout:g}",
        f"ls-{args.label_smoothing:g}",
        f"bestep-{best_epoch}",
        f"acc-{format_metric(test_metrics['acc'])}",
        f"macro-{format_metric(test_metrics['macro_f1'])}",
        f"weighted-{format_metric(test_metrics['weighted_f1'])}",
    ]
    return "__".join(parts)


def save_result_log(args, run_stem, best_epoch, best_f1, test_metrics, save_dir, checkpoint_path):
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{run_stem}.txt"
    content = {
        "save_dir": str(save_dir),
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    }
    save_json(log_path, content)
    return log_path


def main():
    parser = build_parser()
    args = parser.parse_args()
    started_at = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, torch)

    label_map = TASK_LABEL_MAPS[args.task]
    num_classes = len(label_map)

    manifest, feature_root, save_dir = resolve_paths(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(manifest, feature_root, args.task, label_map)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train, val and test splits must all have at least one usable sample")

    input_dim = infer_input_dim(train_df["feature_path"].iloc[0])
    print("input_dim:", input_dim)
    print("device:", device)

    train_loader = DataLoader(
        FeatureDataset(train_df, torch),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FeatureDataset(val_df, torch),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        FeatureDataset(test_df, torch),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = make_model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=num_classes,
        nn=nn,
        model_type=args.model_type,
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()
    config.update(
        {
            "manifest": str(manifest),
            "feature_root": str(feature_root),
            "save_dir": str(save_dir),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "label_map": {str(key): value for key, value in label_map.items()},
            "device": device,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        }
    )
    best_f1 = -1.0
    best_epoch = 0
    best_path = save_dir / f"_tmp_{started_at}_{os.getpid()}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_df)
        train_metrics = evaluate(model, train_loader, device, torch, num_classes)
        val_metrics = evaluate(model, val_loader, device, torch, num_classes)
        test_metrics = evaluate(model, test_loader, device, torch, num_classes)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "test_acc": test_metrics["acc"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        }
        history.append(record)

        print(
            f"epoch={epoch:02d} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"saved current best: epoch={best_epoch} val_macro_f1={best_f1:.4f}")
        elif args.early_stop_patience > 0 and epoch - best_epoch >= args.early_stop_patience:
            print(
                f"early stopped: no val_macro_f1 improvement for "
                f"{args.early_stop_patience} epochs"
            )
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, torch, num_classes)
    run_stem = build_run_stem(args, started_at, best_epoch, test_metrics)
    checkpoint_path = save_dir / f"{run_stem}.pt"
    config_path = save_dir / f"{run_stem}.config.json"
    metrics_path = save_dir / f"{run_stem}.metrics.json"

    torch.save(model.state_dict(), checkpoint_path)
    best_path.unlink(missing_ok=True)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "history": history,
    }
    config.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "metrics_path": str(metrics_path),
            "run_stem": run_stem,
        }
    )
    save_json(config_path, json_safe(config))
    save_json(metrics_path, metrics)
    log_path = save_result_log(
        args,
        run_stem,
        best_epoch,
        best_f1,
        test_metrics,
        save_dir,
        checkpoint_path,
    )

    print("Best epoch:", best_epoch)
    print("Best val macro_f1:", best_f1)
    print("Test acc:", test_metrics["acc"])
    print("Test macro_f1:", test_metrics["macro_f1"])
    print("Test weighted_f1:", test_metrics["weighted_f1"])
    print("Checkpoint:", checkpoint_path)
    print("Config:", config_path)
    print("Metrics:", metrics_path)
    print("Result log:", log_path)


if __name__ == "__main__":
    main()
