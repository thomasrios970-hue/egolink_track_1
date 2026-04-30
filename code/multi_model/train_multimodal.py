"""
MER-style 多模态特征融合训练入口。

脚本读取 sentiment 或 emotion_sort 的 manifest CSV，根据每条样本的
Vid 和 sub_id 在 HuBERT、MacBERT、CLIP 等特征目录中查找对应 .npy
文件。只有所选模态特征全部存在的样本才会进入训练。模型会先把每个
模态独立投影到同一维度，再使用 attention 学习每个样本的模态权重，
或使用 concat 直接拼接模态表示，最后完成分类。attention 模式会在
每轮评估和最终结果中保存平均模态权重，方便观察模型更依赖哪个模态。

运行示例：
python code/multi_model/train_multimodal.py \
  --task sentiment \
  --modalities hubert,macbert,clip \
  --fusion attention \
  --cuda-visible-devices 4 \
  --epochs 80 \
  --batch-size 128 \
  --lr 3e-4 \
  --weight-decay 1e-3 \
  --hidden-dim 512 \
  --dropout 0.4 \
  --label-smoothing 0.05 \
  --grad-clip 1.0 \
  --early-stop-patience 10
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TASK_MANIFESTS = {
    "sentiment": Path("/data/wzw/egolink_race/data/manifest/emotion_degree_manifest.csv"),
    "emotion_sort": Path("/data/wzw/egolink_race/data/manifest/emotion_sort_manifest.csv"),
}
DEFAULT_FEATURE_ROOTS = {
    "hubert": Path("/data/wzw/egolink_race/feature/hubert_large"),
    "macbert": Path("/data/wzw/egolink_race/feature/macbert_large"),
    "clip": Path("/data/wzw/egolink_race/feature/clip_large"),
}
DEFAULT_SAVE_ROOT = Path("/data/wzw/egolink_race/checkpoints/multi_model")
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


def non_negative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return value


def probability_float(value):
    value = float(value)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="训练 HuBERT、MacBERT、CLIP 多模态 .npy 特征的 MER-style 融合分类模型。",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_LABEL_MAPS),
        default="sentiment",
        help="训练任务；sentiment 是 -3 到 3 六分类，emotion_sort 是 happy 等八分类。",
    )
    parser.add_argument(
        "--modalities",
        default="hubert,macbert,clip",
        help="用逗号分隔的模态列表，例如 hubert,macbert,clip 或 macbert,clip。",
    )
    parser.add_argument(
        "--fusion",
        choices=["attention", "concat"],
        default="attention",
        help="多模态融合方式；attention 会学习每个样本的模态权重。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest CSV 路径；不传则按 task 使用默认 manifest。",
    )
    parser.add_argument("--hubert-root", type=Path, default=DEFAULT_FEATURE_ROOTS["hubert"])
    parser.add_argument("--macbert-root", type=Path, default=DEFAULT_FEATURE_ROOTS["macbert"])
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_FEATURE_ROOTS["clip"])
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="模型和结果保存目录；不传则保存到 checkpoints/multi_model/<task>/<fusion>_<modalities>。",
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
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=non_negative_float, default=1e-4)
    parser.add_argument("--hidden-dim", type=positive_int, default=256)
    parser.add_argument("--dropout", type=probability_float, default=0.3)
    parser.add_argument(
        "--label-smoothing",
        type=probability_float,
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
    parser.add_argument(
        "--cv-folds",
        type=positive_int,
        default=1,
        help="MER-style 交叉验证折数；1 表示使用原始 train/val，>1 表示合并 train+val 做 K 折，并平均各折 test 概率。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=non_negative_int, default=4)
    return parser


def parse_modalities(modalities_text):
    modalities = []
    seen = set()
    for item in modalities_text.split(","):
        modality = item.strip().lower()
        if not modality:
            continue
        if modality not in DEFAULT_FEATURE_ROOTS:
            raise ValueError(
                f"unsupported modality: {modality}; "
                f"choices are {sorted(DEFAULT_FEATURE_ROOTS)}"
            )
        if modality not in seen:
            modalities.append(modality)
            seen.add(modality)

    if len(modalities) < 2:
        raise ValueError("multimodal training requires at least two modalities")
    return modalities


def set_seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(args, modalities):
    manifest = args.manifest
    if manifest is None:
        manifest = DEFAULT_TASK_MANIFESTS[args.task]

    feature_roots = {
        "hubert": args.hubert_root,
        "macbert": args.macbert_root,
        "clip": args.clip_root,
    }
    feature_roots = {modality: feature_roots[modality] for modality in modalities}

    save_dir = args.save_dir
    if save_dir is None:
        modality_tag = "+".join(modalities)
        save_dir = DEFAULT_SAVE_ROOT / args.task / f"{args.fusion}_{modality_tag}"

    return manifest, feature_roots, save_dir


def label_id_from_row(row, task, label_map):
    if task == "sentiment":
        label = int(float(row["label"]))
        if label not in label_map:
            raise ValueError(f"unsupported sentiment label: {label}")
        return label_map[label]

    if task == "emotion_sort":
        if "emotion" in row and not pd.isna(row["emotion"]):
            emotion = str(row["emotion"]).strip().lower()
            if emotion not in label_map:
                raise ValueError(f"unsupported emotion label: {emotion}")
            return label_map[emotion]

        label = int(float(row["label"]))
        if not 0 <= label < len(label_map):
            raise ValueError(f"unsupported emotion_sort label: {label}")
        return label

    raise ValueError(f"unsupported task: {task}")


def build_dataframe(manifest_path, feature_roots, modalities, task, label_map):
    df = pd.read_csv(manifest_path)
    required_columns = {"Vid", "sub_id", "label", "split"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"manifest missing columns: {sorted(missing_columns)}")

    rows = []
    split_total = df["split"].value_counts().to_dict()

    for _, row in df.iterrows():
        feature_paths = {}
        has_all_features = True
        for modality in modalities:
            feature_path = feature_roots[modality] / f"{row['Vid']}_{row['sub_id']}.npy"
            if not feature_path.exists():
                has_all_features = False
                break
            feature_paths[f"{modality}_path"] = str(feature_path)

        if not has_all_features:
            continue

        item = row.to_dict()
        item.update(feature_paths)
        item["label_id"] = label_id_from_row(row, task, label_map)
        rows.append(item)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        roots = {modality: str(feature_roots[modality]) for modality in modalities}
        raise ValueError(f"no usable samples found for modalities {modalities}: {roots}")

    split_usable = out_df["split"].value_counts().to_dict()
    split_stats = {}
    for split in ["train", "val", "test"]:
        total = int(split_total.get(split, 0))
        usable = int(split_usable.get(split, 0))
        missing = total - usable
        split_stats[split] = {"usable": usable, "missing": missing, "total": total}
        print(f"{split}: usable={usable}, missing={missing}, total={total}")

    return out_df, split_stats


class MultimodalFeatureDataset:
    def __init__(self, df, modalities, torch):
        self.modalities = modalities
        self.paths = {
            modality: df[f"{modality}_path"].tolist()
            for modality in modalities
        }
        self.labels = df["label_id"].astype(int).tolist()
        self.torch = torch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = {}
        for modality in self.modalities:
            x = np.load(self.paths[modality][idx]).astype(np.float32)
            if x.ndim != 1:
                x = x.reshape(-1)
            features[modality] = self.torch.from_numpy(x)

        y = self.torch.tensor(self.labels[idx], dtype=self.torch.long)
        return features, y


def infer_feature_dims(df, modalities):
    feature_dims = {}
    for modality in modalities:
        feature = np.load(df[f"{modality}_path"].iloc[0])
        if feature.ndim != 1:
            feature = feature.reshape(-1)
        feature_dims[modality] = int(feature.shape[0])
    return feature_dims


def make_model(feature_dims, modalities, hidden_dim, dropout, num_classes, fusion, nn, torch):
    class ModalityEncoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class MultimodalFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.modalities = list(modalities)
            self.fusion = fusion
            self.encoders = nn.ModuleDict({
                modality: ModalityEncoder(feature_dims[modality])
                for modality in self.modalities
            })

            if self.fusion == "attention":
                attention_hidden = max(hidden_dim // 2, 32)
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim, attention_hidden),
                    nn.Tanh(),
                    nn.Linear(attention_hidden, 1),
                )
                classifier_input_dim = hidden_dim
            elif self.fusion == "concat":
                self.attention = None
                classifier_input_dim = hidden_dim * len(self.modalities)
            else:
                raise ValueError(f"unsupported fusion: {self.fusion}")

            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, batch):
            embeddings = [self.encoders[modality](batch[modality]) for modality in self.modalities]

            if self.fusion == "attention":
                stacked = torch.stack(embeddings, dim=1)
                scores = self.attention(stacked).squeeze(-1)
                weights = torch.softmax(scores, dim=1)
                fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
            else:
                weights = None
                fused = torch.cat(embeddings, dim=1)

            logits = self.classifier(fused)
            return logits, weights

    return MultimodalFusionModel()


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
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if total_support else 0.0

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def move_batch_to_device(batch, device):
    return {modality: feature.to(device) for modality, feature in batch.items()}


def average_attention_dict(weight_sum, weight_count, modalities):
    if weight_count == 0:
        return None
    weights = weight_sum / weight_count
    return {
        modality: float(weights[idx])
        for idx, modality in enumerate(modalities)
    }


def evaluate(model, loader, device, torch, num_classes, modalities):
    model.eval()
    labels, probs, attention = predict(model, loader, device, torch, modalities)
    preds = probs.argmax(axis=1)
    metrics = compute_metrics(labels, preds, num_classes)
    return metrics, attention


def predict(model, loader, device, torch, modalities):
    model.eval()
    probs = []
    labels = []
    weight_sum = np.zeros(len(modalities), dtype=np.float64)
    weight_count = 0

    with torch.no_grad():
        for batch, y in loader:
            batch = move_batch_to_device(batch, device)
            logits, weights = model(batch)
            prob = torch.softmax(logits, dim=1).cpu().numpy()

            probs.append(prob)
            labels.extend(y.numpy().tolist())

            if weights is not None:
                weight_sum += weights.sum(dim=0).detach().cpu().numpy()
                weight_count += int(weights.shape[0])

    return (
        np.asarray(labels, dtype=np.int64),
        np.concatenate(probs, axis=0),
        average_attention_dict(weight_sum, weight_count, modalities),
    )


def average_attention_list(attention_list, modalities):
    usable = [attention for attention in attention_list if attention]
    if not usable:
        return None
    return {
        modality: float(np.mean([attention[modality] for attention in usable]))
        for modality in modalities
    }


def build_stratified_folds(labels, cv_folds, seed):
    labels = np.asarray(labels, dtype=np.int64)
    if cv_folds > len(labels):
        raise ValueError(f"cv-folds={cv_folds} is larger than usable dev samples={len(labels)}")

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(cv_folds)]

    for label in sorted(np.unique(labels).tolist()):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        for offset, index in enumerate(indices.tolist()):
            folds[offset % cv_folds].append(index)

    for fold_index, fold in enumerate(folds, start=1):
        if not fold:
            raise ValueError(f"fold {fold_index} is empty; reduce --cv-folds")
        fold.sort()

    return folds


def make_loader(df, modalities, torch, DataLoader, args, shuffle):
    return DataLoader(
        MultimodalFeatureDataset(df, modalities, torch),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def train_one_fold(
    fold_name,
    train_df,
    val_df,
    test_df,
    feature_dims,
    args,
    modalities,
    num_classes,
    device,
    torch,
    nn,
    DataLoader,
    save_dir,
    started_at,
):
    train_loader = make_loader(train_df, modalities, torch, DataLoader, args, shuffle=True)
    val_loader = make_loader(val_df, modalities, torch, DataLoader, args, shuffle=False)
    test_loader = make_loader(test_df, modalities, torch, DataLoader, args, shuffle=False)

    model = make_model(
        feature_dims=feature_dims,
        modalities=modalities,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=num_classes,
        fusion=args.fusion,
        nn=nn,
        torch=torch,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_path = save_dir / f"_tmp_{started_at}_{os.getpid()}_{fold_name}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch, y in train_loader:
            batch = move_batch_to_device(batch, device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        train_loss = total_loss / len(train_df)
        train_metrics, train_attention = evaluate(model, train_loader, device, torch, num_classes, modalities)
        val_metrics, val_attention = evaluate(model, val_loader, device, torch, num_classes, modalities)
        test_metrics, test_attention = evaluate(model, test_loader, device, torch, num_classes, modalities)

        record = {
            "fold": fold_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_attention": train_attention,
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_attention": val_attention,
            "test_acc": test_metrics["acc"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_attention": test_attention,
        }
        history.append(record)

        print(
            f"{fold_name} epoch={epoch:02d} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f} "
            f"test_attention: {format_attention(test_attention)}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"{fold_name} saved current best: epoch={best_epoch} val_macro_f1={best_f1:.4f}")
        elif args.early_stop_patience > 0 and epoch - best_epoch >= args.early_stop_patience:
            print(
                f"{fold_name} early stopped: no val_macro_f1 improvement for "
                f"{args.early_stop_patience} epochs"
            )
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_labels, test_probs, test_attention = predict(model, test_loader, device, torch, modalities)
    test_metrics = compute_metrics(test_labels, test_probs.argmax(axis=1), num_classes)

    return {
        "fold": fold_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_path": best_path,
        "history": history,
        "test_labels": test_labels,
        "test_probs": test_probs,
        "test_metrics": test_metrics,
        "test_attention": test_attention,
    }


def format_metric(value):
    return f"{value:.4f}"


def format_attention(attention):
    if not attention:
        return "none"
    return " ".join(
        f"{modality}={weight:.4f}"
        for modality, weight in attention.items()
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
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_run_stem(args, started_at, modalities, best_epoch, test_metrics):
    parts = [
        started_at,
        f"task-{args.task}",
        f"mods-{'+'.join(modalities)}",
        f"fusion-{args.fusion}",
    ]
    if args.cv_folds > 1:
        parts.append(f"cv-{args.cv_folds}")
    parts.extend([
        f"seed-{args.seed}",
        f"ep-{args.epochs}",
        f"bs-{args.batch_size}",
        f"lr-{args.lr:g}",
        f"wd-{args.weight_decay:g}",
        f"hid-{args.hidden_dim}",
        f"drop-{args.dropout:g}",
        f"ls-{args.label_smoothing:g}",
        f"bestep-{best_epoch}",
        f"acc-{format_metric(test_metrics['acc'])}",
        f"macro-{format_metric(test_metrics['macro_f1'])}",
        f"weighted-{format_metric(test_metrics['weighted_f1'])}",
    ])
    return "__".join(parts)


def save_result_log(args, run_stem, best_epoch, best_f1, test_metrics, test_attention, save_dir, checkpoint_paths):
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{run_stem}.txt"
    content = {
        "save_dir": str(save_dir),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "cv_folds": args.cv_folds,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_attention": test_attention,
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

    modalities = parse_modalities(args.modalities)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, torch)

    label_map = TASK_LABEL_MAPS[args.task]
    num_classes = len(label_map)

    manifest, feature_roots, save_dir = resolve_paths(args, modalities)
    save_dir.mkdir(parents=True, exist_ok=True)

    df, split_stats = build_dataframe(
        manifest_path=manifest,
        feature_roots=feature_roots,
        modalities=modalities,
        task=args.task,
        label_map=label_map,
    )
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train, val and test splits must all have at least one usable sample")

    feature_dims = infer_feature_dims(train_df, modalities)
    print("modalities:", ",".join(modalities))
    print("fusion:", args.fusion)
    print("feature_dims:", feature_dims)
    print("device:", device)
    print("cv_folds:", args.cv_folds)

    config = vars(args).copy()
    config.update(
        {
            "manifest": str(manifest),
            "feature_roots": {
                modality: str(feature_roots[modality])
                for modality in modalities
            },
            "save_dir": str(save_dir),
            "modalities": modalities,
            "feature_dims": feature_dims,
            "num_classes": num_classes,
            "label_map": {str(key): value for key, value in label_map.items()},
            "device": device,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "split_stats": split_stats,
        }
    )

    if args.cv_folds == 1:
        fold_results = [
            train_one_fold(
                fold_name="run",
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_dims=feature_dims,
                args=args,
                modalities=modalities,
                num_classes=num_classes,
                device=device,
                torch=torch,
                nn=nn,
                DataLoader=DataLoader,
                save_dir=save_dir,
                started_at=started_at,
            )
        ]
        best_epoch = fold_results[0]["best_epoch"]
        best_f1 = fold_results[0]["best_val_macro_f1"]
        test_metrics = fold_results[0]["test_metrics"]
        test_attention = fold_results[0]["test_attention"]
    else:
        dev_df = pd.concat([train_df, val_df], ignore_index=True)
        folds = build_stratified_folds(dev_df["label_id"].tolist(), args.cv_folds, args.seed)
        all_indices = np.arange(len(dev_df))
        fold_results = []

        for fold_idx, val_indices in enumerate(folds, start=1):
            val_index_set = set(val_indices)
            train_indices = [int(index) for index in all_indices if int(index) not in val_index_set]
            fold_train_df = dev_df.iloc[train_indices].reset_index(drop=True)
            fold_val_df = dev_df.iloc[val_indices].reset_index(drop=True)

            print(
                f"===== CV fold {fold_idx}/{args.cv_folds}: "
                f"train={len(fold_train_df)} val={len(fold_val_df)} test={len(test_df)} ====="
            )
            fold_results.append(
                train_one_fold(
                    fold_name=f"fold{fold_idx}",
                    train_df=fold_train_df,
                    val_df=fold_val_df,
                    test_df=test_df,
                    feature_dims=feature_dims,
                    args=args,
                    modalities=modalities,
                    num_classes=num_classes,
                    device=device,
                    torch=torch,
                    nn=nn,
                    DataLoader=DataLoader,
                    save_dir=save_dir,
                    started_at=started_at,
                )
            )

        test_labels = fold_results[0]["test_labels"]
        for result in fold_results[1:]:
            if not np.array_equal(test_labels, result["test_labels"]):
                raise ValueError("test labels differ across folds; cannot average probabilities")

        mean_probs = np.mean([result["test_probs"] for result in fold_results], axis=0)
        test_metrics = compute_metrics(test_labels, mean_probs.argmax(axis=1), num_classes)
        test_attention = average_attention_list(
            [result["test_attention"] for result in fold_results],
            modalities,
        )
        best_epoch = "cv"
        best_f1 = float(np.mean([result["best_val_macro_f1"] for result in fold_results]))

    run_stem = build_run_stem(args, started_at, modalities, best_epoch, test_metrics)
    config_path = save_dir / f"{run_stem}.config.json"
    metrics_path = save_dir / f"{run_stem}.metrics.json"

    checkpoint_paths = []
    for result in fold_results:
        suffix = ".pt" if args.cv_folds == 1 else f".{result['fold']}.pt"
        checkpoint_path = save_dir / f"{run_stem}{suffix}"
        result["best_path"].replace(checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_attention": test_attention,
        "folds": [
            {
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "best_val_macro_f1": result["best_val_macro_f1"],
                "test_metrics": result["test_metrics"],
                "test_attention": result["test_attention"],
            }
            for result in fold_results
        ],
        "history": [result["history"] for result in fold_results],
    }
    config.update(
        {
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "config_path": str(config_path),
            "metrics_path": str(metrics_path),
            "run_stem": run_stem,
        }
    )
    save_json(config_path, json_safe(config))
    save_json(metrics_path, metrics)
    log_path = save_result_log(
        args=args,
        run_stem=run_stem,
        best_epoch=best_epoch,
        best_f1=best_f1,
        test_metrics=test_metrics,
        test_attention=test_attention,
        save_dir=save_dir,
        checkpoint_paths=checkpoint_paths,
    )

    print("Best epoch:", best_epoch)
    print("Best val macro_f1:", best_f1)
    print("Test acc:", test_metrics["acc"])
    print("Test macro_f1:", test_metrics["macro_f1"])
    print("Test weighted_f1:", test_metrics["weighted_f1"])
    print("test_attention:", format_attention(test_attention))
    print("Checkpoints:", ", ".join(str(path) for path in checkpoint_paths))
    print("Config:", config_path)
    print("Metrics:", metrics_path)
    print("Result log:", log_path)


if __name__ == "__main__":
    main()
