"""
统一训练单模态特征分类 baseline。

支持 feature 目录下已经提取好的单模态 UTTERANCE/UTT 级别 .npy 特征。
脚本会读取 manifest.csv，按 Vid 和 sub_id 拼接特征路径，过滤缺失特征，
然后用训练集统计量做 z-score 特征归一化，训练一个 MLP 分类器，并按
指定验证集指标保存最优模型。

运行示例（emotion_sort 当前推荐：低容量 + 强正则，仍按 val_weighted_f1 保存 best）：
python code/single_model/train_single_modal.py \
  --task emotion_sort \
  --modality all \
  --cuda-visible-devices 4 \
  --epochs 80 \
  --batch-size 128 \
  --lr 1e-4 \
  --weight-decay 5e-3 \
  --hidden-dim 256 \
  --num-layers 1 \
  --dropout 0.5 \
  --label-smoothing 0.1 \
  --grad-clip 1.0 \
  --early-stop-patience 8 \
  --selection-metric val_weighted_f1

快速 smoke test：
python code/single_model/train_single_modal.py \
  --task emotion_sort \
  --modality all \
  --cuda-visible-devices 4 \
  --epochs 1 \
  --batch-size 512 \
  --num-workers 0
"""

import argparse
import gc
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
FEATURE_ROOT = Path("/data/wzw/egolink_race/feature")
FEATURE_GROUP_DIRS = [
    FEATURE_ROOT / "audio_features",
    FEATURE_ROOT / "txt_features",
    FEATURE_ROOT / "visual_features",
]
FEATURE_GROUP_NAMES = {path.name for path in FEATURE_GROUP_DIRS}
FEATURE_LEVEL_DIRS = {
    "utterance": ["UTTERANCE", "UTT"],
}
FEATURE_ALIASES = {
    "hubert": Path("/data/wzw/egolink_race/feature/audio_features/chinese-hubert-large"),
    "macbert": Path("/data/wzw/egolink_race/feature/txt_features/chinese-macbert-large"),
    "clip": Path("/data/wzw/egolink_race/feature/visual_features/clip-vit-large-patch14"),
}
DEFAULT_CHECKPOINT_ROOT = Path("/data/wzw/egolink_race/checkpoints")
DEFAULT_LOG_ROOT = Path("/data/wzw/egolink_race/log")
OUTPUT_TASK_DIRS = {
    "sentiment": "emotion_sentiment",
    "emotion_sort": "emotion_sort",
}
FEATURE_LOG_GROUPS = {
    "audio_features": "audio",
    "txt_features": "txt",
    "visual_features": "visual",
    "custom_features": "custom",
}
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


def discover_feature_roots():
    roots = {}
    for group_dir in FEATURE_GROUP_DIRS:
        if not group_dir.exists():
            continue
        for child in sorted(group_dir.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name in roots:
                name = f"{group_dir.name}-{name}"
            roots[name] = child
    return roots


def get_feature_roots():
    roots = {}
    roots.update(FEATURE_ALIASES)
    roots.update(discover_feature_roots())
    return roots


def infer_feature_group(feature_root):
    for part in Path(feature_root).parts:
        if part in FEATURE_GROUP_NAMES:
            return part
    return "custom_features"


def infer_log_group(feature_group):
    return FEATURE_LOG_GROUPS.get(feature_group, feature_group)


def get_output_task_dir(task):
    return OUTPUT_TASK_DIRS.get(task, task)

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
    feature_roots = get_feature_roots()
    parser = argparse.ArgumentParser(
        description="训练 feature 目录下单模态 .npy 特征的 MLP baseline。",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_LABEL_MAPS),
        default="sentiment",
        help="训练任务；sentiment 是 -3 到 3 六分类，emotion_sort 是 happy 等八分类。",
    )
    parser.add_argument(
        "--modality",
        choices=sorted(feature_roots) + ["all"],
        required=True,
        help="选择训练的单模态特征；all 表示扫描并训练 feature 下现有全部特征目录。",
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
        help="特征 .npy 目录；不传则按 modality 使用默认目录。modality=all 时不能传该参数。",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="模型和结果保存目录；不传则保存到 checkpoints/<task>/single_model/<feature_group>/<modality>。modality=all 时会在该目录下再分 group/modality 子目录。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="实验摘要文件保存目录；不传则保存到 log/<task>/single_model/audio、txt 或 visual。modality=all 且传入该参数时会在该目录下再分 audio/txt/visual。",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="设置 CUDA_VISIBLE_DEVICES，例如 4；不传则使用当前环境变量。",
    )
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--epochs", type=positive_int, default=80)
    parser.add_argument("--lr", type=positive_float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument(
        "--model-type",
        choices=["mlp", "deep_mlp", "residual_mlp"],
        default="residual_mlp",
        help="分类头结构；mlp 是旧 baseline，residual_mlp 更适合冲更高分。",
    )
    parser.add_argument("--hidden-dim", type=positive_int, default=256)
    parser.add_argument(
        "--num-layers",
        type=positive_int,
        default=1,
        help="deep_mlp/residual_mlp 使用的隐藏层数量。",
    )
    parser.add_argument(
        "--feature-level",
        choices=["utterance"],
        default="utterance",
        help="使用哪个级别的特征；当前脚本只使用 UTTERANCE/UTT 定长特征。",
    )
    parser.add_argument(
        "--feature-normalization",
        choices=["none", "zscore"],
        default="zscore",
        help="特征归一化方式；zscore 使用训练集 mean/std 并应用到 train/val/test。",
    )
    parser.add_argument(
        "--zscore-eps",
        type=positive_float,
        default=1e-6,
        help="z-score 标准差下限，避免除以 0。",
    )
    parser.add_argument("--dropout", type=dropout_float, default=0.5)
    parser.add_argument(
        "--label-smoothing",
        type=dropout_float,
        default=0.1,
        help="CrossEntropyLoss 的 label smoothing。",
    )
    parser.add_argument(
        "--grad-clip",
        type=non_negative_float,
        default=1.0,
        help="梯度裁剪阈值；0 表示关闭。",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=non_negative_int,
        default=8,
        help="验证集选择指标连续多少轮不提升就停止；0 表示关闭。",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["val_macro_f1", "val_weighted_f1", "val_acc"],
        default="val_weighted_f1",
        help="用于保存 best checkpoint 的验证集指标；冲 weighted F1/Accuracy 时建议 val_weighted_f1 或 val_acc。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=non_negative_int, default=4)
    return parser


def get_target_modalities(modality):
    if modality == "all":
        return list(discover_feature_roots())
    return [modality]


def set_seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(args, all_mode=False):
    feature_root = args.feature_root
    if feature_root is None:
        feature_roots = get_feature_roots()
        if args.modality not in feature_roots:
            raise ValueError(f"unsupported modality: {args.modality}")
        feature_root = feature_roots[args.modality]
    feature_group = infer_feature_group(feature_root)
    output_task_dir = get_output_task_dir(args.task)

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = (
            DEFAULT_CHECKPOINT_ROOT
            / output_task_dir
            / "single_model"
            / feature_group
            / args.modality
        )
    elif all_mode:
        save_dir = save_dir / feature_group / args.modality

    manifest = args.manifest
    if manifest is None:
        manifest = DEFAULT_TASK_MANIFESTS[args.task]

    return manifest, feature_root, save_dir, feature_group, output_task_dir


def normalize_label(raw_label, task):
    if task == "sentiment":
        return int(raw_label)
    raise ValueError(f"unsupported task: {task}")


def get_label(row, task):
    if task == "emotion_sort" and "emotion" in row:
        return str(row["emotion"]).strip().lower()
    return normalize_label(row["label"], task)


def get_level_dirs(feature_level):
    if feature_level not in FEATURE_LEVEL_DIRS:
        raise ValueError(f"unsupported feature level: {feature_level}")
    return FEATURE_LEVEL_DIRS[feature_level]


def find_feature_path(feature_root, vid, sub_id, feature_level):
    candidates = []
    for level_dir in get_level_dirs(feature_level):
        candidates.extend([
            feature_root / level_dir / str(vid) / f"{sub_id}.npy",
            feature_root / level_dir / f"{vid}_{sub_id}.npy",
        ])
    candidates.extend([
        feature_root / f"{vid}_{sub_id}.npy",
        feature_root / str(vid) / f"{sub_id}.npy",
    ])

    for path in candidates:
        if path.exists():
            return path
    return None


def infer_feature_level(path):
    path_parts = {part.upper() for part in Path(path).parts}
    if "UTTERANCE" in path_parts or "UTT" in path_parts:
        return "utterance"
    return "direct"


def build_dataframe(manifest_path, feature_root, task, label_map, feature_level="utterance"):
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

        vid = str(row["Vid"])
        sub_id = str(row["sub_id"])
        feature_path = find_feature_path(feature_root, vid, sub_id, feature_level)
        if feature_path is None:
            continue

        item = row.to_dict()
        item["feature_path"] = str(feature_path)
        item["feature_level"] = infer_feature_level(feature_path)
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
    print("feature_level:", out_df["feature_level"].value_counts().to_dict())

    return out_df


class FeatureDataset:
    def __init__(self, df, torch, feature_mean=None, feature_std=None):
        self.paths = df["feature_path"].tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.torch = torch
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = load_feature_vector(self.paths[idx])
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std
            x = np.ascontiguousarray(x)
        y = self.labels[idx]
        return (
            self.torch.tensor(x, dtype=self.torch.float32),
            self.torch.tensor(y, dtype=self.torch.long),
        )


def load_feature_vector(feature_path):
    feature = np.load(feature_path).astype(np.float32)
    feature = np.squeeze(feature)
    if feature.ndim != 1:
        raise ValueError(
            f"expected 1D UTTERANCE/UTT feature, got shape {feature.shape}: "
            f"{feature_path}"
        )
    return np.ascontiguousarray(feature)


def infer_input_dim(feature_path):
    feature = load_feature_vector(feature_path)
    return int(feature.shape[0])


def compute_zscore_stats(df, eps):
    paths = df["feature_path"].tolist()
    if not paths:
        raise ValueError("cannot compute z-score stats from an empty dataframe")

    input_dim = infer_input_dim(paths[0])
    feature_sum = np.zeros(input_dim, dtype=np.float64)
    feature_sq_sum = np.zeros(input_dim, dtype=np.float64)

    for path in paths:
        feature = load_feature_vector(path)
        if feature.shape[0] != input_dim:
            raise ValueError(
                f"inconsistent feature dim: expected {input_dim}, "
                f"got {feature.shape[0]} from {path}"
            )
        feature64 = feature.astype(np.float64, copy=False)
        feature_sum += feature64
        feature_sq_sum += feature64 * feature64

    count = float(len(paths))
    mean = feature_sum / count
    variance = np.maximum(feature_sq_sum / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    std[std < eps] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


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
        class ResidualMLP(nn.Module):
            def __init__(self):
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

            def forward(self, x):
                x = self.input(x)
                for block in self.blocks:
                    x = x + block(x)
                return self.output(x)

        return ResidualMLP()

    raise ValueError(f"unsupported model type: {model_type}")


def make_old_model(input_dim, hidden_dim, dropout, num_classes, nn):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def build_label_names(label_map):
    label_names = [None] * len(label_map)
    for label, label_id in label_map.items():
        label_names[label_id] = str(label)
    return [
        label_name if label_name is not None else str(label_id)
        for label_id, label_name in enumerate(label_names)
    ]


def build_label_distribution(df, label_names):
    counts = df["label_id"].value_counts().to_dict()
    return {
        label_names[label_id]: int(counts.get(label_id, 0))
        for label_id in range(len(label_names))
    }


def build_split_label_distribution(train_df, val_df, test_df, label_names):
    return {
        "train": build_label_distribution(train_df, label_names),
        "val": build_label_distribution(val_df, label_names),
        "test": build_label_distribution(test_df, label_names),
    }


def compute_metrics(labels, preds, num_classes, label_names=None, focus_labels=None):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    acc = float((labels == preds).mean()) if len(labels) else 0.0
    if label_names is None:
        label_names = [str(label_id) for label_id in range(num_classes)]
    if focus_labels is None:
        focus_labels = []

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels.tolist(), preds.tolist()):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            confusion[label, pred] += 1

    f1_scores = []
    supports = []
    per_class_metrics = {}
    for cls in range(num_classes):
        true_positive = int(confusion[cls, cls])
        false_positive = int(confusion[:, cls].sum() - true_positive)
        false_negative = int(confusion[cls, :].sum() - true_positive)
        support = int(confusion[cls, :].sum())

        precision_den = true_positive + false_positive
        recall_den = true_positive + false_negative
        precision = true_positive / precision_den if precision_den else 0.0
        recall = true_positive / recall_den if recall_den else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        f1_scores.append(f1)
        supports.append(support)
        per_class_metrics[label_names[cls]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    macro_f1 = float(np.mean(f1_scores))
    total_support = int(np.sum(supports))
    weighted_f1 = (
        float(np.average(f1_scores, weights=supports))
        if total_support
        else 0.0
    )
    focus_class_metrics = {
        label: per_class_metrics[label]
        for label in focus_labels
        if label in per_class_metrics
    }

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_metrics": per_class_metrics,
        "focus_class_metrics": focus_class_metrics,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": label_names,
    }


def evaluate(model, loader, device, torch, num_classes, label_names=None, focus_labels=None):
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

    return compute_metrics(
        labels,
        preds,
        num_classes,
        label_names=label_names,
        focus_labels=focus_labels,
    )


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_metric(value):
    return f"{value:.4f}"


def build_run_stem(args, started_at, test_metrics):
    parts = [
        f"mod-{args.modality}",
        started_at,
        f"WAF-{format_metric(test_metrics['weighted_f1'])}",
        f"ACC-{format_metric(test_metrics['acc'])}",
    ]
    return "__".join(parts)


def save_result_log(args, run_stem, config, metrics, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_stem}.txt"
    content = {
        "run_stem": run_stem,
        "task": args.task,
        "modality": args.modality,
        "selection_metric": args.selection_metric,
        "best_epoch": metrics["best_epoch"],
        "best_selection_score": metrics["best_selection_score"],
        "test_acc": metrics["test_acc"],
        "test_macro_f1": metrics["test_macro_f1"],
        "test_weighted_f1": metrics["test_weighted_f1"],
        "per_class_metrics": metrics["per_class_metrics"],
        "happy_sad_metrics": metrics["happy_sad_metrics"],
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_labels": metrics["confusion_matrix_labels"],
        "split_label_distribution": metrics["split_label_distribution"],
        "args": json_safe(vars(args)),
        "config": json_safe(config),
        "metrics": metrics,
    }
    save_json(log_path, content)
    return log_path


def resolve_log_dir(args, feature_group, all_mode, output_task_dir):
    log_group = infer_log_group(feature_group)
    if args.log_dir is None:
        return DEFAULT_LOG_ROOT / output_task_dir / "single_model" / log_group, log_group
    if all_mode:
        return args.log_dir / log_group, log_group
    return args.log_dir, log_group


def run_single_modality(args, started_at, all_mode, torch, nn, DataLoader):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, torch)

    label_map = TASK_LABEL_MAPS[args.task]
    label_names = build_label_names(label_map)
    focus_labels = ["happy", "sad"] if args.task == "emotion_sort" else []
    num_classes = len(label_map)

    manifest, feature_root, save_dir, feature_group, output_task_dir = resolve_paths(
        args,
        all_mode=all_mode,
    )
    log_dir, log_group = resolve_log_dir(
        args,
        feature_group,
        all_mode,
        output_task_dir,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("modality:", args.modality)
    print("output_task_dir:", output_task_dir)
    print("manifest:", manifest)
    print("feature_root:", feature_root)
    print("feature_group:", feature_group)
    print("log_group:", log_group)
    print("save_dir:", save_dir)
    print("log_dir:", log_dir)

    df = build_dataframe(
        manifest,
        feature_root,
        args.task,
        label_map,
        feature_level=args.feature_level,
    )
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train, val and test splits must all have at least one usable sample")

    split_label_distribution = build_split_label_distribution(
        train_df,
        val_df,
        test_df,
        label_names,
    )
    input_dim = infer_input_dim(train_df["feature_path"].iloc[0])
    feature_mean = None
    feature_std = None
    if args.feature_normalization == "zscore":
        print("computing train z-score stats...")
        feature_mean, feature_std = compute_zscore_stats(train_df, args.zscore_eps)
        print("feature_normalization: zscore")
    else:
        print("feature_normalization: none")
    print("input_dim:", input_dim)
    print("device:", device)

    train_loader = DataLoader(
        FeatureDataset(train_df, torch, feature_mean, feature_std),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FeatureDataset(val_df, torch, feature_mean, feature_std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        FeatureDataset(test_df, torch, feature_mean, feature_std),
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
            "started_at": started_at,
            "output_task_dir": output_task_dir,
            "manifest": str(manifest),
            "feature_root": str(feature_root),
            "feature_group": feature_group,
            "log_group": log_group,
            "save_dir": str(save_dir),
            "log_dir": str(log_dir),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "feature_level": args.feature_level,
            "feature_normalization": args.feature_normalization,
            "zscore_eps": args.zscore_eps,
            "used_feature_levels": df["feature_level"].value_counts().to_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "label_names": label_names,
            "label_map": {str(key): value for key, value in label_map.items()},
            "split_label_distribution": split_label_distribution,
            "device": device,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "normalization_stats": (
                {
                    "mean": feature_mean.tolist(),
                    "std": feature_std.tolist(),
                    "eps": args.zscore_eps,
                }
                if args.feature_normalization == "zscore"
                else None
            ),
        }
    )
    best_score = -1.0
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
        train_metrics = evaluate(
            model,
            train_loader,
            device,
            torch,
            num_classes,
            label_names=label_names,
            focus_labels=focus_labels,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            torch,
            num_classes,
            label_names=label_names,
            focus_labels=focus_labels,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            torch,
            num_classes,
            label_names=label_names,
            focus_labels=focus_labels,
        )
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
            f"train_acc={train_metrics['acc']:.4f} "
            f"train_weighted_f1={train_metrics['weighted_f1']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"test_weighted_f1={test_metrics['weighted_f1']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}"
        )

        current_score = record[args.selection_metric]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(
                f"saved current best: epoch={best_epoch} "
                f"{args.selection_metric}={best_score:.4f}"
            )
        elif args.early_stop_patience > 0 and epoch - best_epoch >= args.early_stop_patience:
            print(
                f"early stopped: no {args.selection_metric} improvement for "
                f"{args.early_stop_patience} epochs"
            )
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    train_metrics = evaluate(
        model,
        train_loader,
        device,
        torch,
        num_classes,
        label_names=label_names,
        focus_labels=focus_labels,
    )
    val_metrics = evaluate(
        model,
        val_loader,
        device,
        torch,
        num_classes,
        label_names=label_names,
        focus_labels=focus_labels,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        torch,
        num_classes,
        label_names=label_names,
        focus_labels=focus_labels,
    )
    run_stem = build_run_stem(args, started_at, test_metrics)
    checkpoint_path = save_dir / f"{run_stem}.pt"
    config_path = save_dir / f"{run_stem}.config.json"
    metrics_path = save_dir / f"{run_stem}.metrics.json"

    torch.save(model.state_dict(), checkpoint_path)
    best_path.unlink(missing_ok=True)

    metrics = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "per_class_metrics": test_metrics["per_class_metrics"],
        "focus_class_metrics": test_metrics["focus_class_metrics"],
        "happy_sad_metrics": test_metrics["focus_class_metrics"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "confusion_matrix_labels": test_metrics["confusion_matrix_labels"],
        "split_label_distribution": split_label_distribution,
        "feature_normalization": args.feature_normalization,
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
    log_path = save_result_log(args, run_stem, config, metrics, log_dir)

    print("Best epoch:", best_epoch)
    print("Selection metric:", args.selection_metric)
    print("Best selection score:", best_score)
    print("Test acc:", test_metrics["acc"])
    print("Test macro_f1:", test_metrics["macro_f1"])
    print("Test weighted_f1:", test_metrics["weighted_f1"])
    print("Checkpoint:", checkpoint_path)
    print("Config:", config_path)
    print("Metrics:", metrics_path)
    print("Result log:", log_path)

    return {
        "modality": args.modality,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "test_acc": test_metrics["acc"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.requested_modality = args.modality
    started_at = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    target_modalities = get_target_modalities(args.modality)
    all_mode = args.modality == "all"
    if all_mode and args.feature_root is not None:
        raise ValueError("--feature-root is only supported when --modality is a single modality")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    summaries = []
    failed = 0

    for index, modality in enumerate(target_modalities, start=1):
        run_args = argparse.Namespace(**vars(args))
        run_args.modality = modality
        print(f"[{index}/{len(target_modalities)}] start modality={modality}")

        try:
            summaries.append(
                run_single_modality(
                    run_args,
                    started_at,
                    all_mode,
                    torch,
                    nn,
                    DataLoader,
                )
            )
        except Exception as e:
            if not all_mode:
                raise
            failed += 1
            print(f"failed modality={modality}: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Done")
    print(f"total={len(target_modalities)}")
    print(f"success={len(summaries)}")
    print(f"failed={failed}")
    for summary in summaries:
        print(
            f"{summary['modality']}: "
            f"ACC={summary['test_acc']:.4f}, "
            f"WAF={summary['test_weighted_f1']:.4f}, "
            f"log={summary['log_path']}"
        )


if __name__ == "__main__":
    main()
