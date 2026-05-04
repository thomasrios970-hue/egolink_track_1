"""
MER-style 多模态特征融合训练入口。

改进点（相比旧版）：
1. 默认使用 chinese-hubert-large + xlm-roberta-xl + clip-vit-large-patch14 三模态
2. 默认启用 hierarchical 训练模式（先粗分再细分）
3. 默认启用 cosine LR scheduler + warmup
4. 默认启用 person embedding + person prior
5. 默认启用 ensemble_top_k=5
6. 默认启用 sqrt_inv 类别权重 + 采样器
7. 默认 selection_metric=val_acc，tie_breaker=val_weighted_f1
8. log 格式与内容与原版完全兼容

推荐运行（三模态 hierarchical，完整配置）：
python /data/wzw/egolink_race/code/multi_model/train_multimodal.py --cuda-visible-devices 4

只用音视频两模态对比：
python train_multimodal.py \
  --modalities hubert,clip \
  --cuda-visible-devices 4

smoke test：
python /data/wzw/egolink_race/code/multi_model/train_multimodal.py \
  --cuda-visible-devices 4 \
  --epochs 2 \
  --batch-size 256 \
  --num-workers 0
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TASK_MANIFESTS = {
    "sentiment": Path("/data/wzw/egolink_race/data/manifest/emotion_degree_audio_manifest.csv"),
    "emotion_sort": Path("/data/wzw/egolink_race/data/manifest/emotion_sort_manifest.csv"),
}
DEFAULT_FEATURE_ROOTS = {
    "hubert": Path("/data/wzw/egolink_race/feature/audio_features/chinese-hubert-large"),
    "macbert": Path("/data/wzw/egolink_race/feature/txt_features/xlm-roberta-xl"),
    "target_text": Path("/data/wzw/egolink_race/feature/target_txt_features/xlm-roberta-xl"),
    "clip": Path("/data/wzw/egolink_race/feature/visual_features/clip-vit-large-patch14"),
}
FEATURE_GROUP_ROOTS = {
    "hubert": Path("/data/wzw/egolink_race/feature/audio_features"),
    "macbert": Path("/data/wzw/egolink_race/feature/txt_features"),
    "target_text": Path("/data/wzw/egolink_race/feature/target_txt_features"),
    "clip": Path("/data/wzw/egolink_race/feature/visual_features"),
}
DEFAULT_FEATURE_NAMES = {
    "hubert": "chinese-hubert-large",
    "macbert": "xlm-roberta-xl",
    "target_text": "xlm-roberta-xl",
    "clip": "clip-vit-large-patch14",
}
FEATURE_ARG_NAMES = {
    "hubert": "audio_feature",
    "macbert": "txt_feature",
    "target_text": "target_text_feature",
    "clip": "visual_feature",
}
DEFAULT_CHECKPOINT_ROOT = Path("/data/wzw/egolink_race/checkpoints")
DEFAULT_LOG_ROOT = Path("/data/wzw/egolink_race/log")
OUTPUT_TASK_DIRS = {
    "sentiment": "emotion_sentiment",
    "emotion_sort": "emotion_sort",
}
FEATURE_LEVEL_DIRS = ["UTTERANCE", "UTT"]
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
HIERARCHICAL_COARSE_LABELS = ["angry", "happy", "surprised", "other"]
HIERARCHICAL_EXPERT_LABELS = ["disgusted", "sad", "sarcastic", "scared", "shy"]
PRESETS = {
    "none": {},
    "emotion_sort_hier_acc": {
        "task": "emotion_sort",
        "modalities": "hubert,macbert,clip",
        "audio_feature": "chinese-hubert-large",
        "txt_feature": "xlm-roberta-xl",
        "visual_feature": "clip-vit-large-patch14",
        "fusion": "attention",
        "training_mode": "hierarchical",
        "epochs": 90,
        "batch_size": 128,
        "lr": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 3,
        "weight_decay": 1e-2,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.5,
        "modality_dropout": 0.15,
        "label_smoothing": 0.08,
        "use_person_embedding": True,
        "person_embedding_dim": 32,
        "use_person_prior": True,
        "person_prior_smoothing": 1.0,
        "class_weight": "sqrt_inv",
        "class_weight_max": 4.0,
        "sampler": "sqrt_inv",
        "selection_metric": "val_accselect",
        "selection_tie_breaker": "val_weighted_f1",
        "ensemble_top_k": 5,
        "ensemble_metric": "val_acc",
        "early_stop_patience": 12,
        "prediction_dir": Path("/data/wzw/egolink_race/data/processed/model_predictions/emotion_sort"),
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


def discover_feature_names(modality):
    group_root = FEATURE_GROUP_ROOTS[modality]
    if not group_root.exists():
        return []
    return sorted(
        child.name
        for child in group_root.iterdir()
        if child.is_dir()
    )


def get_output_task_dir(task):
    return OUTPUT_TASK_DIRS.get(task, task)


def build_parser():
    audio_features = discover_feature_names("hubert")
    txt_features = discover_feature_names("macbert")
    target_text_features = discover_feature_names("target_text")
    visual_features = discover_feature_names("clip")
    parser = argparse.ArgumentParser(
        description="训练 HuBERT、MacBERT、CLIP 多模态 .npy 特征的 MER-style 融合分类模型。",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="emotion_sort_hier_acc",
        help=(
            "实验预设；默认 emotion_sort_hier_acc 使用三模态 large 特征 + hierarchical "
            "+ cosine LR + person embedding + sqrt_inv 类别权重 + ensemble top-5。"
            "传 none 可完全使用显式参数和脚本原始默认值。"
        ),
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_LABEL_MAPS),
        default="emotion_sort",
    )
    parser.add_argument(
        "--modalities",
        default="hubert,macbert,clip",
        help="用逗号分隔的模态列表，例如 hubert,macbert,clip 或 hubert,clip。",
    )
    parser.add_argument(
        "--fusion",
        choices=["attention", "concat"],
        default="attention",
    )
    parser.add_argument(
        "--training-mode",
        choices=["flat", "hierarchical"],
        default="flat",
        help=(
            "flat 是原始 8 分类；hierarchical 先分 angry/happy/surprised/other，"
            "再用 other expert 细分 disgusted/sad/sarcastic/scared/shy。"
        ),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--feature-combo",
        choices=["single", "all"],
        default="single",
    )
    parser.add_argument(
        "--audio-feature",
        choices=audio_features + ["all"] if audio_features else ["all"],
        default=DEFAULT_FEATURE_NAMES["hubert"],
    )
    parser.add_argument(
        "--txt-feature",
        choices=txt_features + ["all"] if txt_features else ["all"],
        default=DEFAULT_FEATURE_NAMES["macbert"],
    )
    parser.add_argument(
        "--target-text-feature",
        choices=sorted(set(target_text_features + [DEFAULT_FEATURE_NAMES["target_text"], "all"])),
        default=DEFAULT_FEATURE_NAMES["target_text"],
    )
    parser.add_argument(
        "--visual-feature",
        choices=visual_features + ["all"] if visual_features else ["all"],
        default=DEFAULT_FEATURE_NAMES["clip"],
    )
    parser.add_argument("--hubert-root", type=Path, default=None)
    parser.add_argument("--macbert-root", type=Path, default=None)
    parser.add_argument("--target-text-root", type=Path, default=None)
    parser.add_argument("--clip-root", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--epochs", type=positive_int, default=90)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--lr", type=positive_float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        default="cosine",
    )
    parser.add_argument("--warmup-epochs", type=non_negative_int, default=3)
    parser.add_argument("--min-lr-ratio", type=positive_float, default=0.1)
    parser.add_argument("--weight-decay", type=non_negative_float, default=1e-2)
    parser.add_argument("--hidden-dim", type=positive_int, default=256)
    parser.add_argument("--num-layers", type=positive_int, default=2)
    parser.add_argument(
        "--use-person-embedding",
        action="store_true",
        default=False,
    )
    parser.add_argument("--person-embedding-dim", type=positive_int, default=32)
    parser.add_argument("--use-person-prior", action="store_true", default=False)
    parser.add_argument("--person-prior-smoothing", type=non_negative_float, default=1.0)
    parser.add_argument("--dropout", type=probability_float, default=0.5)
    parser.add_argument("--modality-dropout", type=probability_float, default=0.15)
    parser.add_argument(
        "--feature-normalization",
        choices=["none", "zscore"],
        default="zscore",
    )
    parser.add_argument("--zscore-eps", type=positive_float, default=1e-6)
    parser.add_argument("--label-smoothing", type=probability_float, default=0.08)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=non_negative_float, default=2.0)
    parser.add_argument(
        "--class-weight",
        choices=["none", "inverse", "sqrt_inv", "effective"],
        default="none",
    )
    parser.add_argument("--class-weight-beta", type=probability_float, default=0.999)
    parser.add_argument("--class-weight-max", type=non_negative_float, default=5.0)
    parser.add_argument(
        "--sampler",
        choices=["none", "inverse", "sqrt_inv", "effective"],
        default="none",
    )
    parser.add_argument("--grad-clip", type=non_negative_float, default=1.0)
    parser.add_argument("--early-stop-patience", type=non_negative_int, default=12)
    parser.add_argument(
        "--selection-metric",
        choices=[
            "val_macro_f1",
            "val_weighted_f1",
            "val_acc",
            "val_acc_weighted_f1",
            "val_weighted_macro_f1",
            "val_acc_weighted_macro_f1",
        ],
        default="val_acc",
    )
    parser.add_argument("--selection-tolerance", type=non_negative_float, default=0.0)
    parser.add_argument(
        "--selection-tie-breaker",
        choices=["none", "later", "val_acc", "val_weighted_f1", "val_macro_f1"],
        default="val_weighted_f1",
    )
    parser.add_argument("--selection-min-val-acc", type=non_negative_float, default=0.0)
    parser.add_argument("--selection-min-val-weighted-f1", type=non_negative_float, default=0.0)
    parser.add_argument("--ensemble-top-k", type=positive_int, default=5)
    parser.add_argument(
        "--ensemble-metric",
        choices=[
            "val_acc",
            "val_weighted_f1",
            "val_acc_weighted_f1",
            "val_macro_f1",
            "val_weighted_macro_f1",
            "val_acc_weighted_macro_f1",
        ],
        default="val_acc",
    )
    parser.add_argument("--cv-folds", type=positive_int, default=1)
    parser.add_argument("--prediction-dir", type=Path, default=None)
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


def get_explicit_arg_dests(parser, argv):
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit_dests = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit_dests.add(dest)
    return explicit_dests


def apply_preset(args, explicit_dests):
    preset = PRESETS.get(args.preset, {})
    if not preset:
        return args
    for dest, value in preset.items():
        if dest in explicit_dests:
            continue
        setattr(args, dest, value)
    return args


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
    feature_roots = {}
    feature_names = {}
    custom_roots = {
        "hubert": args.hubert_root,
        "macbert": args.macbert_root,
        "target_text": args.target_text_root,
        "clip": args.clip_root,
    }
    for modality in modalities:
        if custom_roots[modality] is not None:
            feature_roots[modality] = custom_roots[modality]
            feature_names[modality] = custom_roots[modality].name
            continue
        feature_name = getattr(args, FEATURE_ARG_NAMES[modality])
        if feature_name == "all":
            raise ValueError("resolve_paths only supports a concrete feature setting")
        feature_roots[modality] = FEATURE_GROUP_ROOTS[modality] / feature_name
        feature_names[modality] = feature_name
    save_dir = args.save_dir
    if save_dir is None:
        modality_tag = "+".join(feature_names[modality] for modality in modalities)
        save_dir = (
            DEFAULT_CHECKPOINT_ROOT
            / get_output_task_dir(args.task)
            / "multi_model"
            / f"{args.fusion}_{modality_tag}"
        )
    return manifest, feature_roots, feature_names, save_dir


def resolve_log_dir(args):
    if args.log_dir is not None:
        return args.log_dir
    return DEFAULT_LOG_ROOT / get_output_task_dir(args.task) / "muti_model"


def build_feature_runs(args, modalities):
    choices = []
    custom_roots = {
        "hubert": args.hubert_root,
        "macbert": args.macbert_root,
        "target_text": args.target_text_root,
        "clip": args.clip_root,
    }
    for modality in modalities:
        if custom_roots[modality] is not None:
            choices.append([custom_roots[modality].name])
            continue
        feature_name = getattr(args, FEATURE_ARG_NAMES[modality])
        if args.feature_combo == "all":
            feature_name = "all"
        if feature_name == "all":
            discovered = discover_feature_names(modality)
            if not discovered:
                raise ValueError(f"no features found under {FEATURE_GROUP_ROOTS[modality]}")
            choices.append(discovered)
        else:
            choices.append([feature_name])
    multi_run = any(len(choice) > 1 for choice in choices)
    runs = []
    for combo in iter_product(choices):
        run_args = argparse.Namespace(**vars(args))
        feature_roots = {}
        feature_names = {}
        for modality, feature_name in zip(modalities, combo):
            custom_root = custom_roots[modality]
            if custom_root is not None:
                feature_roots[modality] = custom_root
                feature_names[modality] = custom_root.name
            else:
                feature_roots[modality] = FEATURE_GROUP_ROOTS[modality] / feature_name
                feature_names[modality] = feature_name
                setattr(run_args, FEATURE_ARG_NAMES[modality], feature_name)
        save_dir = run_args.save_dir
        if save_dir is None:
            modality_tag = "+".join(feature_names[modality] for modality in modalities)
            save_dir = (
                DEFAULT_CHECKPOINT_ROOT
                / get_output_task_dir(run_args.task)
                / "multi_model"
                / f"{run_args.fusion}_{modality_tag}"
            )
        elif multi_run:
            modality_tag = "+".join(feature_names[modality] for modality in modalities)
            save_dir = save_dir / f"{run_args.fusion}_{modality_tag}"
        run_args.log_dir = resolve_log_dir(run_args)
        runs.append((run_args, feature_roots, feature_names, save_dir))
    return runs


def iter_product(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for value in first:
        for suffix in iter_product(rest):
            yield [value] + suffix


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
    for row_index, row in df.iterrows():
        row_id = get_row_id(row, row_index)
        feature_paths = {}
        has_all_features = True
        for modality in modalities:
            if modality == "target_text":
                feature_path = find_target_text_feature_path(
                    feature_roots[modality], row_id,
                )
            else:
                feature_path = find_feature_path(
                    feature_roots[modality],
                    str(row["Vid"]),
                    str(row["sub_id"]),
                )
            if feature_path is None:
                has_all_features = False
                break
            feature_paths[f"{modality}_path"] = str(feature_path)
        if not has_all_features:
            continue
        item = row.to_dict()
        item["row_id"] = row_id
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


def get_row_id(row, row_index):
    if "row_id" in row and not pd.isna(row["row_id"]):
        text = str(row["row_id"]).strip()
        if text:
            return text
    return f"row_{int(row_index):06d}"


def find_feature_path(feature_root, vid, sub_id):
    candidates = []
    for level_dir in FEATURE_LEVEL_DIRS:
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


def find_target_text_feature_path(feature_root, row_id):
    candidates = []
    for level_dir in FEATURE_LEVEL_DIRS:
        candidates.append(feature_root / level_dir / f"{row_id}.npy")
    candidates.append(feature_root / f"{row_id}.npy")
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_person_name(value):
    text = str(value).strip().lower()
    return text if text and text != "nan" else "<unknown>"


def build_person_vocab(train_df):
    if "person" not in train_df.columns:
        raise ValueError("--use-person-embedding requires a person column in the manifest")
    persons = sorted({normalize_person_name(value) for value in train_df["person"].tolist()})
    vocab = {"<unk>": 0}
    for person in persons:
        if person not in vocab:
            vocab[person] = len(vocab)
    return vocab


def build_person_priors(train_df, person_vocab, num_classes, smoothing):
    priors = np.full(
        (len(person_vocab), num_classes),
        float(smoothing),
        dtype=np.float32,
    )
    for _, row in train_df.iterrows():
        person_id = person_vocab.get(normalize_person_name(row["person"]), 0)
        label_id = int(row["label_id"])
        priors[person_id, label_id] += 1.0
    row_sums = priors.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    priors = priors / row_sums
    priors[0, :] = 1.0 / num_classes
    return priors.astype(np.float32)


class MultimodalFeatureDataset:
    def __init__(
        self,
        df,
        modalities,
        torch,
        feature_means=None,
        feature_stds=None,
        person_vocab=None,
        person_priors=None,
    ):
        self.modalities = modalities
        self.paths = {
            modality: df[f"{modality}_path"].tolist()
            for modality in modalities
        }
        self.labels = df["label_id"].astype(int).tolist()
        self.torch = torch
        self.feature_means = feature_means or {}
        self.feature_stds = feature_stds or {}
        self.person_vocab = person_vocab
        self.person_priors = person_priors
        if person_vocab is not None:
            self.person_ids = [
                person_vocab.get(normalize_person_name(value), 0)
                for value in df["person"].tolist()
            ]
        else:
            self.person_ids = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = {}
        for modality in self.modalities:
            x = load_feature_vector(self.paths[modality][idx])
            if modality in self.feature_means and modality in self.feature_stds:
                x = (x - self.feature_means[modality]) / self.feature_stds[modality]
                x = np.ascontiguousarray(x)
            features[modality] = self.torch.tensor(x, dtype=self.torch.float32)
        if self.person_ids is not None:
            person_id = self.person_ids[idx]
            features["_person_id"] = self.torch.tensor(person_id, dtype=self.torch.long)
            if self.person_priors is not None:
                features["_person_prior"] = self.torch.tensor(
                    self.person_priors[person_id],
                    dtype=self.torch.float32,
                )
        y = self.torch.tensor(self.labels[idx], dtype=self.torch.long)
        return features, y


def load_feature_vector(feature_path):
    feature = np.load(feature_path).astype(np.float32)
    feature = np.squeeze(feature)
    if feature.ndim != 1:
        raise ValueError(
            f"expected 1D UTTERANCE/UTT feature, got shape {feature.shape}: "
            f"{feature_path}"
        )
    return np.ascontiguousarray(feature)


def infer_feature_dims(df, modalities):
    feature_dims = {}
    for modality in modalities:
        feature = load_feature_vector(df[f"{modality}_path"].iloc[0])
        feature_dims[modality] = int(feature.shape[0])
    return feature_dims


def compute_zscore_stats(df, modalities, eps):
    feature_means = {}
    feature_stds = {}
    for modality in modalities:
        paths = df[f"{modality}_path"].tolist()
        if not paths:
            raise ValueError(f"cannot compute z-score stats from empty {modality} paths")
        input_dim = int(load_feature_vector(paths[0]).shape[0])
        feature_sum = np.zeros(input_dim, dtype=np.float64)
        feature_sq_sum = np.zeros(input_dim, dtype=np.float64)
        for path in paths:
            feature = load_feature_vector(path)
            if feature.shape[0] != input_dim:
                raise ValueError(
                    f"inconsistent {modality} feature dim: expected {input_dim}, "
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
        feature_means[modality] = mean.astype(np.float32)
        feature_stds[modality] = std.astype(np.float32)
    return feature_means, feature_stds


def make_model(
    feature_dims,
    modalities,
    hidden_dim,
    num_layers,
    dropout,
    modality_dropout,
    num_classes,
    fusion,
    person_vocab_size,
    person_embedding_dim,
    use_person_prior,
    nn,
    torch,
):
    residual_layers = max(num_layers - 1, 0)

    def make_hidden_block():
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    class ModalityEncoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList([make_hidden_block() for _ in range(residual_layers)])

        def forward(self, x):
            x = self.input(x)
            for block in self.blocks:
                x = x + block(x)
            return x

    class ResidualClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList([make_hidden_block() for _ in range(residual_layers)])
            self.output = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            x = self.input(x)
            for block in self.blocks:
                x = x + block(x)
            return self.output(x)

    class MultimodalFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.modalities = list(modalities)
            self.fusion = fusion
            self.modality_dropout = modality_dropout
            self.person_embedding = None
            if person_vocab_size > 0:
                self.person_embedding = nn.Sequential(
                    nn.Embedding(person_vocab_size, person_embedding_dim),
                    nn.LayerNorm(person_embedding_dim),
                    nn.Dropout(dropout),
                )
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
            if self.person_embedding is not None:
                classifier_input_dim += person_embedding_dim
            if use_person_prior:
                classifier_input_dim += num_classes
            self.classifier = ResidualClassifier(classifier_input_dim)

        def build_modality_keep_mask(self, batch_size, device):
            if (
                not self.training
                or self.modality_dropout <= 0
                or len(self.modalities) <= 1
            ):
                return None
            keep = torch.rand(batch_size, len(self.modalities), device=device) >= self.modality_dropout
            empty_rows = ~keep.any(dim=1)
            if empty_rows.any():
                row_indices = empty_rows.nonzero(as_tuple=True)[0]
                fallback = torch.randint(0, len(self.modalities), (int(row_indices.numel()),), device=device)
                keep[empty_rows] = False
                keep[row_indices, fallback] = True
            return keep

        def forward(self, batch):
            embeddings = [
                self.encoders[modality](batch[modality])
                for modality in self.modalities
            ]
            keep = self.build_modality_keep_mask(
                batch_size=embeddings[0].shape[0],
                device=embeddings[0].device,
            )
            if self.fusion == "attention":
                stacked = torch.stack(embeddings, dim=1)
                scores = self.attention(stacked).squeeze(-1)
                if keep is not None:
                    scores = scores.masked_fill(~keep, -1e4)
                weights = torch.softmax(scores, dim=1)
                fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
            else:
                weights = None
                if keep is not None:
                    stacked = torch.stack(embeddings, dim=1)
                    stacked = stacked * keep.unsqueeze(-1)
                    embeddings = [stacked[:, idx, :] for idx in range(len(self.modalities))]
                fused = torch.cat(embeddings, dim=1)
            if self.person_embedding is not None:
                person_id = batch.get("_person_id")
                if person_id is None:
                    raise ValueError("person embedding is enabled but _person_id is missing")
                person_emb = self.person_embedding(person_id)
                fused = torch.cat([fused, person_emb], dim=1)
            if use_person_prior:
                person_prior = batch.get("_person_prior")
                if person_prior is None:
                    raise ValueError("person prior is enabled but _person_prior is missing")
                fused = torch.cat([fused, person_prior], dim=1)
            logits = self.classifier(fused)
            return logits, weights

    return MultimodalFusionModel()


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


def compute_class_counts(df, num_classes):
    return np.bincount(
        df["label_id"].astype(int).to_numpy(),
        minlength=num_classes,
    ).astype(np.float64)


def compute_balancing_weights(counts, strategy, beta=0.999, max_weight=5.0):
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.ones_like(counts, dtype=np.float64)
    valid = counts > 0
    if strategy == "none":
        return weights.astype(np.float32)
    if strategy == "inverse":
        weights[valid] = 1.0 / counts[valid]
    elif strategy == "sqrt_inv":
        weights[valid] = 1.0 / np.sqrt(counts[valid])
    elif strategy == "effective":
        effective_num = 1.0 - np.power(beta, counts[valid])
        weights[valid] = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    else:
        raise ValueError(f"unsupported balancing strategy: {strategy}")
    if valid.any():
        weights[valid] = weights[valid] / weights[valid].mean()
    weights[~valid] = 0.0
    if max_weight > 0:
        weights = np.minimum(weights, max_weight)
    return weights.astype(np.float32)


def build_class_weight_tensor(args, train_df, num_classes, torch, device):
    if args.class_weight == "none":
        return None, None
    counts = compute_class_counts(train_df, num_classes)
    weights = compute_balancing_weights(
        counts, args.class_weight,
        beta=args.class_weight_beta, max_weight=args.class_weight_max,
    )
    return torch.tensor(weights, dtype=torch.float32, device=device), weights


def build_sample_weights(args, train_df, num_classes):
    if args.sampler == "none":
        return None, None
    counts = compute_class_counts(train_df, num_classes)
    class_weights = compute_balancing_weights(
        counts, args.sampler,
        beta=args.class_weight_beta, max_weight=args.class_weight_max,
    )
    labels = train_df["label_id"].astype(int).to_numpy()
    sample_weights = class_weights[labels].astype(np.float64)
    return sample_weights, class_weights


def build_split_label_distribution(train_df, val_df, test_df, label_names):
    return {
        "train": build_label_distribution(train_df, label_names),
        "val": build_label_distribution(val_df, label_names),
        "test": build_label_distribution(test_df, label_names),
    }


def label_name_to_id(label_names):
    return {label_name: label_id for label_id, label_name in enumerate(label_names)}


def map_coarse_label(label_id, label_names):
    label_name = label_names[int(label_id)]
    if label_name == "angry":
        return 0
    if label_name == "happy":
        return 1
    if label_name == "surprised":
        return 2
    return 3


def make_coarse_df(df, label_names):
    out_df = df.copy()
    out_df["flat_label_id"] = out_df["label_id"].astype(int)
    out_df["label_id"] = out_df["flat_label_id"].map(
        lambda label_id: map_coarse_label(label_id, label_names)
    )
    return out_df


def make_expert_df(df, label_names, include_all=False):
    label_to_id = label_name_to_id(label_names)
    expert_original_ids = {
        label_to_id[label_name]
        for label_name in HIERARCHICAL_EXPERT_LABELS
    }
    expert_label_to_id = {
        label_name: label_id
        for label_id, label_name in enumerate(HIERARCHICAL_EXPERT_LABELS)
    }
    if include_all:
        out_df = df.copy()
    else:
        out_df = df[df["label_id"].astype(int).isin(expert_original_ids)].copy()
    out_df["flat_label_id"] = out_df["label_id"].astype(int)

    def map_expert_label(flat_label_id):
        label_name = label_names[int(flat_label_id)]
        return expert_label_to_id.get(label_name, 0)

    out_df["label_id"] = out_df["flat_label_id"].map(map_expert_label)
    return out_df.reset_index(drop=True)


def combine_hierarchical_probs(coarse_probs, expert_probs, label_names):
    coarse_probs = np.asarray(coarse_probs, dtype=np.float64)
    expert_probs = np.asarray(expert_probs, dtype=np.float64)
    if len(coarse_probs) != len(expert_probs):
        raise ValueError(
            f"coarse/expert prediction length mismatch: "
            f"{len(coarse_probs)} vs {len(expert_probs)}"
        )
    label_to_id = label_name_to_id(label_names)
    final_probs = np.zeros((len(coarse_probs), len(label_names)), dtype=np.float64)
    final_probs[:, label_to_id["angry"]] = coarse_probs[:, 0]
    final_probs[:, label_to_id["happy"]] = coarse_probs[:, 1]
    final_probs[:, label_to_id["surprised"]] = coarse_probs[:, 2]
    other_prob = coarse_probs[:, 3]
    for expert_id, label_name in enumerate(HIERARCHICAL_EXPERT_LABELS):
        final_probs[:, label_to_id[label_name]] = other_prob * expert_probs[:, expert_id]
    row_sums = final_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return final_probs / row_sums


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
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if total_support else 0.0
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


def move_batch_to_device(batch, device):
    return {modality: feature.to(device) for modality, feature in batch.items()}


def average_attention_dict(weight_sum, weight_count, modalities):
    if weight_count == 0:
        return None
    weights = weight_sum / weight_count
    return {modality: float(weights[idx]) for idx, modality in enumerate(modalities)}


def evaluate(model, loader, device, torch, num_classes, modalities, label_names=None, focus_labels=None):
    model.eval()
    labels, probs, attention = predict(model, loader, device, torch, modalities)
    preds = probs.argmax(axis=1)
    metrics = compute_metrics(labels, preds, num_classes, label_names=label_names, focus_labels=focus_labels)
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
    usable = [a for a in attention_list if a]
    if not usable:
        return None
    return {
        modality: float(np.mean([a[modality] for a in usable]))
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


def make_loader(
    df, modalities, torch, DataLoader, args, shuffle,
    feature_means=None, feature_stds=None, sample_weights=None,
    person_vocab=None, person_priors=None,
):
    sampler = None
    if sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        MultimodalFeatureDataset(
            df, modalities, torch,
            feature_means=feature_means,
            feature_stds=feature_stds,
            person_vocab=person_vocab,
            person_priors=person_priors,
        ),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
    )


def make_criterion(args, class_weight_tensor, nn):
    if args.loss == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weight_tensor,
            label_smoothing=args.label_smoothing,
        )
    if args.loss == "focal":
        class FocalLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.gamma = args.focal_gamma
                self.weight = class_weight_tensor
                self.label_smoothing = args.label_smoothing

            def forward(self, logits, targets):
                ce = nn.functional.cross_entropy(
                    logits, targets,
                    weight=self.weight,
                    label_smoothing=self.label_smoothing,
                    reduction="none",
                )
                probs = nn.functional.softmax(logits, dim=1)
                target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                focal_factor = (1.0 - target_probs).clamp_min(1e-8).pow(self.gamma)
                return (focal_factor * ce).mean()
        return FocalLoss()
    raise ValueError(f"unsupported loss: {args.loss}")


def get_epoch_lr(args, epoch):
    if args.lr_scheduler == "none":
        return args.lr
    if args.lr_scheduler == "cosine":
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            return args.lr * epoch / args.warmup_epochs
        decay_epochs = max(args.epochs - args.warmup_epochs, 1)
        decay_epoch = min(max(epoch - args.warmup_epochs, 0), decay_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_epoch / decay_epochs))
        min_lr = args.lr * args.min_lr_ratio
        return min_lr + (args.lr - min_lr) * cosine
    raise ValueError(f"unsupported lr_scheduler: {args.lr_scheduler}")


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def is_better_checkpoint(record, best_score, best_record, args):
    if record["val_acc"] < args.selection_min_val_acc:
        return False
    if record["val_weighted_f1"] < args.selection_min_val_weighted_f1:
        return False
    current_score = record[args.selection_metric]
    if current_score > best_score:
        return True
    if best_record is None or args.selection_tolerance <= 0:
        return False
    if current_score + args.selection_tolerance < best_score:
        return False
    if args.selection_tie_breaker == "later":
        return record["epoch"] > best_record["epoch"]
    if args.selection_tie_breaker in {"val_acc", "val_weighted_f1", "val_macro_f1"}:
        return record[args.selection_tie_breaker] > best_record[args.selection_tie_breaker]
    return False


def update_ensemble_candidates(candidates, record, model, save_dir, started_at, fold_name, args, torch):
    if args.ensemble_top_k <= 1:
        return candidates
    score = record[args.ensemble_metric]
    path = (
        save_dir
        / f"_tmp_{started_at}_{os.getpid()}_{fold_name}_ensemble_ep{record['epoch']:03d}.pt"
    )
    torch.save(model.state_dict(), path)
    candidates.append({
        "score": float(score),
        "epoch": int(record["epoch"]),
        "path": path,
        "record": record,
    })
    candidates.sort(key=lambda item: (item["score"], item["epoch"]), reverse=True)
    while len(candidates) > args.ensemble_top_k:
        removed = candidates.pop()
        try:
            removed["path"].unlink()
        except FileNotFoundError:
            pass
    return candidates


def train_one_fold(
    fold_name, train_df, val_df, test_df, feature_dims, args, modalities,
    num_classes, device, torch, nn, DataLoader, save_dir, started_at,
    label_names, focus_labels, extra_eval_dfs=None,
):
    feature_means = {}
    feature_stds = {}
    normalization_stats = None
    if args.feature_normalization == "zscore":
        print(f"{fold_name} computing per-modality train z-score stats...")
        feature_means, feature_stds = compute_zscore_stats(train_df, modalities, args.zscore_eps)
        normalization_stats = {
            modality: {
                "mean": feature_means[modality].tolist(),
                "std": feature_stds[modality].tolist(),
                "eps": args.zscore_eps,
            }
            for modality in modalities
        }
    else:
        print(f"{fold_name} feature_normalization: none")

    class_weight_tensor, class_weights = build_class_weight_tensor(args, train_df, num_classes, torch, device)
    sample_weights, sampler_class_weights = build_sample_weights(args, train_df, num_classes)
    if class_weights is not None:
        print(f"{fold_name} class_weight={args.class_weight}: {class_weights.tolist()}")
    if sampler_class_weights is not None:
        print(f"{fold_name} sampler={args.sampler}: {sampler_class_weights.tolist()}")

    person_vocab = None
    person_priors = None
    if args.use_person_embedding or args.use_person_prior:
        person_vocab = build_person_vocab(train_df)
        print(f"{fold_name} person_vocab_size={len(person_vocab)}")
    if args.use_person_prior:
        person_priors = build_person_priors(
            train_df, person_vocab, num_classes, args.person_prior_smoothing,
        )
        print(f"{fold_name} person_prior_smoothing={args.person_prior_smoothing}")

    train_loader = make_loader(
        train_df, modalities, torch, DataLoader, args, shuffle=True,
        feature_means=feature_means, feature_stds=feature_stds,
        sample_weights=sample_weights, person_vocab=person_vocab, person_priors=person_priors,
    )
    val_loader = make_loader(
        val_df, modalities, torch, DataLoader, args, shuffle=False,
        feature_means=feature_means, feature_stds=feature_stds,
        person_vocab=person_vocab, person_priors=person_priors,
    )
    test_loader = make_loader(
        test_df, modalities, torch, DataLoader, args, shuffle=False,
        feature_means=feature_means, feature_stds=feature_stds,
        person_vocab=person_vocab, person_priors=person_priors,
    )

    model = make_model(
        feature_dims=feature_dims,
        modalities=modalities,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
        num_classes=num_classes,
        fusion=args.fusion,
        person_vocab_size=len(person_vocab) if person_vocab is not None else 0,
        person_embedding_dim=args.person_embedding_dim,
        use_person_prior=args.use_person_prior,
        nn=nn,
        torch=torch,
    ).to(device)

    criterion = make_criterion(args, class_weight_tensor, nn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_epoch = 0
    best_record = None
    best_path = save_dir / f"_tmp_{started_at}_{os.getpid()}_{fold_name}_best.pt"
    fallback_score = -1.0
    fallback_epoch = 0
    fallback_record = None
    fallback_path = save_dir / f"_tmp_{started_at}_{os.getpid()}_{fold_name}_fallback_best.pt"
    ensemble_candidates = []
    history = []

    for epoch in range(1, args.epochs + 1):
        current_lr = get_epoch_lr(args, epoch)
        set_optimizer_lr(optimizer, current_lr)
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
        train_metrics, train_attention = evaluate(
            model, train_loader, device, torch, num_classes, modalities,
            label_names=label_names, focus_labels=focus_labels,
        )
        val_metrics, val_attention = evaluate(
            model, val_loader, device, torch, num_classes, modalities,
            label_names=label_names, focus_labels=focus_labels,
        )
        test_metrics, test_attention = evaluate(
            model, test_loader, device, torch, num_classes, modalities,
            label_names=label_names, focus_labels=focus_labels,
        )

        record = {
            "fold": fold_name,
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_attention": train_attention,
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_acc_weighted_f1": (
                0.7 * val_metrics["acc"] + 0.3 * val_metrics["weighted_f1"]
            ),
            "val_weighted_macro_f1": (
                val_metrics["weighted_f1"] + val_metrics["macro_f1"]
            ),
            "val_acc_weighted_macro_f1": (
                val_metrics["acc"] + val_metrics["weighted_f1"] + val_metrics["macro_f1"]
            ),
            "val_attention": val_attention,
            "test_acc": test_metrics["acc"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_attention": test_attention,
        }
        history.append(record)
        ensemble_candidates = update_ensemble_candidates(
            ensemble_candidates, record, model, save_dir, started_at, fold_name, args, torch,
        )

        print(
            f"{fold_name} epoch={epoch:02d} "
            f"lr={current_lr:.2e} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"train_weighted_f1={train_metrics['weighted_f1']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"test_weighted_f1={test_metrics['weighted_f1']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f} "
            f"test_attention: {format_attention(test_attention)}"
        )

        current_score = record[args.selection_metric]
        if current_score > fallback_score:
            fallback_score = current_score
            fallback_epoch = epoch
            fallback_record = record
            torch.save(model.state_dict(), fallback_path)

        if is_better_checkpoint(record, best_score, best_record, args):
            best_score = max(best_score, current_score)
            best_epoch = epoch
            best_record = record
            torch.save(model.state_dict(), best_path)
            print(
                f"{fold_name} saved current best: epoch={best_epoch} "
                f"{args.selection_metric}={current_score:.4f} "
                f"best_{args.selection_metric}={best_score:.4f}"
            )
        elif (
            args.early_stop_patience > 0
            and epoch - max(best_epoch, fallback_epoch) >= args.early_stop_patience
        ):
            print(
                f"{fold_name} early stopped: no {args.selection_metric} improvement for "
                f"{args.early_stop_patience} epochs"
            )
            break

    if best_record is None:
        if fallback_record is None:
            raise RuntimeError(f"{fold_name} did not produce any checkpoint candidate")
        print(
            f"{fold_name} no checkpoint met selection gates; "
            f"falling back to epoch={fallback_epoch} "
            f"{args.selection_metric}={fallback_score:.4f}"
        )
        best_score = fallback_score
        best_epoch = fallback_epoch
        best_record = fallback_record
        best_path = fallback_path

    extra_eval_dfs = extra_eval_dfs or {}
    extra_predictions = {}

    if args.ensemble_top_k > 1 and ensemble_candidates:
        val_ensemble_probs = []
        val_ensemble_attention = []
        test_ensemble_probs = []
        test_ensemble_attention = []
        extra_ensemble_probs = {name: [] for name in extra_eval_dfs}
        extra_ensemble_attention = {name: [] for name in extra_eval_dfs}
        extra_labels = {}
        val_labels = None
        test_labels = None
        for candidate in ensemble_candidates:
            model.load_state_dict(torch.load(candidate["path"], map_location=device))
            labels, probs, attention = predict(model, val_loader, device, torch, modalities)
            if val_labels is None:
                val_labels = labels
            elif not np.array_equal(val_labels, labels):
                raise ValueError("val labels differ across ensemble checkpoints")
            val_ensemble_probs.append(probs)
            val_ensemble_attention.append(attention)
            labels, probs, attention = predict(model, test_loader, device, torch, modalities)
            if test_labels is None:
                test_labels = labels
            elif not np.array_equal(test_labels, labels):
                raise ValueError("test labels differ across ensemble checkpoints")
            test_ensemble_probs.append(probs)
            test_ensemble_attention.append(attention)
            for name, extra_df in extra_eval_dfs.items():
                extra_loader = make_loader(
                    extra_df, modalities, torch, DataLoader, args, shuffle=False,
                    feature_means=feature_means, feature_stds=feature_stds,
                    person_vocab=person_vocab, person_priors=person_priors,
                )
                labels, probs, attention = predict(model, extra_loader, device, torch, modalities)
                if name not in extra_labels:
                    extra_labels[name] = labels
                elif not np.array_equal(extra_labels[name], labels):
                    raise ValueError(f"{name} labels differ across ensemble checkpoints")
                extra_ensemble_probs[name].append(probs)
                extra_ensemble_attention[name].append(attention)
        val_probs = np.mean(val_ensemble_probs, axis=0)
        val_attention = average_attention_list(val_ensemble_attention, modalities)
        test_probs = np.mean(test_ensemble_probs, axis=0)
        test_attention = average_attention_list(test_ensemble_attention, modalities)
        for name in extra_eval_dfs:
            extra_predictions[name] = {
                "labels": extra_labels[name],
                "probs": np.mean(extra_ensemble_probs[name], axis=0),
                "attention": average_attention_list(extra_ensemble_attention[name], modalities),
            }
    else:
        model.load_state_dict(torch.load(best_path, map_location=device))
        val_labels, val_probs, val_attention = predict(model, val_loader, device, torch, modalities)
        test_labels, test_probs, test_attention = predict(model, test_loader, device, torch, modalities)
        for name, extra_df in extra_eval_dfs.items():
            extra_loader = make_loader(
                extra_df, modalities, torch, DataLoader, args, shuffle=False,
                feature_means=feature_means, feature_stds=feature_stds,
                person_vocab=person_vocab, person_priors=person_priors,
            )
            labels, probs, attention = predict(model, extra_loader, device, torch, modalities)
            extra_predictions[name] = {"labels": labels, "probs": probs, "attention": attention}

    val_metrics = compute_metrics(
        val_labels, val_probs.argmax(axis=1), num_classes,
        label_names=label_names, focus_labels=focus_labels,
    )
    test_metrics = compute_metrics(
        test_labels, test_probs.argmax(axis=1), num_classes,
        label_names=label_names, focus_labels=focus_labels,
    )

    return {
        "fold": fold_name,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "normalization_stats": normalization_stats,
        "best_path": best_path,
        "history": history,
        "class_weights": class_weights,
        "sampler_class_weights": sampler_class_weights,
        "person_vocab": person_vocab,
        "person_priors": person_priors,
        "val_labels": val_labels,
        "val_probs": val_probs,
        "val_metrics": val_metrics,
        "val_attention": val_attention,
        "test_labels": test_labels,
        "test_probs": test_probs,
        "test_metrics": test_metrics,
        "test_attention": test_attention,
        "extra_predictions": extra_predictions,
        "ensemble_candidates": [
            {
                "epoch": c["epoch"],
                "score": c["score"],
                "path": c["path"],
                "record": c["record"],
            }
            for c in ensemble_candidates
        ],
    }


def format_metric(value):
    return f"{value:.4f}"


def format_attention(attention):
    if not attention:
        return "none"
    return " ".join(f"{modality}={weight:.4f}" for modality, weight in attention.items())


def short_feature_name(name):
    replacements = [
        ("chinese-", ""),
        ("clip-vit-large-patch14-336", "clip-p14-336"),
        ("clip-vit-large-patch14", "clip-p14"),
        ("xlm-roberta-", "xlm-r-"),
        ("wav2vec2-", "w2v2-"),
    ]
    short_name = name
    for old, new in replacements:
        short_name = short_name.replace(old, new)
    return short_name[:40]


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_probability_jsonl(path, df, labels, probs, label_names, split_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if len(df) != len(labels) or len(df) != len(probs):
        raise ValueError(
            f"prediction length mismatch for {split_name}: "
            f"df={len(df)} labels={len(labels)} probs={len(probs)}"
        )
    with path.open("w", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(df.iterrows()):
            pred_id = int(np.argmax(probs[idx]))
            item = {
                "row_id": str(row.get("row_id", f"row_{idx:06d}")),
                "Vid": str(row.get("Vid", "")),
                "sub_id": str(row.get("sub_id", "")),
                "person": str(row.get("person", "")),
                "split": str(row.get("split", split_name)),
                "label_id": int(labels[idx]),
                "label_name": label_names[int(labels[idx])],
                "pred_id": pred_id,
                "pred_label": label_names[pred_id],
                "probabilities": {
                    label_names[cls]: float(probs[idx, cls])
                    for cls in range(len(label_names))
                },
                "probabilities_list": [
                    float(probs[idx, cls]) for cls in range(len(label_names))
                ],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def build_feature_tag(modalities, feature_names):
    return "+".join(short_feature_name(feature_names[modality]) for modality in modalities)


def build_run_stem(args, started_at, modalities, feature_names, test_metrics):
    parts = [
        started_at,
        f"mods-{build_feature_tag(modalities, feature_names)}",
        f"fusion-{args.fusion}",
        f"mode-{args.training_mode}",
        f"WAF-{format_metric(test_metrics['weighted_f1'])}",
        f"ACC-{format_metric(test_metrics['acc'])}",
    ]
    if args.cv_folds > 1:
        parts.append(f"cv-{args.cv_folds}")
    return "__".join(parts)


def save_result_log(args, run_stem, config, metrics):
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{run_stem}.txt"
    content = {
        "run_stem": run_stem,
        "task": args.task,
        "modalities": config["modalities"],
        "feature_names": config["feature_names"],
        "fusion": args.fusion,
        "cv_folds": metrics["cv_folds"],
        "best_epoch": metrics["best_epoch"],
        "selection_metric": metrics["selection_metric"],
        "best_selection_score": metrics["best_selection_score"],
        "test_acc": metrics["test_acc"],
        "test_macro_f1": metrics["test_macro_f1"],
        "test_weighted_f1": metrics["test_weighted_f1"],
        "test_attention": metrics["test_attention"],
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


def score_from_metrics(metrics, selection_metric):
    if selection_metric == "val_acc":
        return metrics["acc"]
    if selection_metric == "val_weighted_f1":
        return metrics["weighted_f1"]
    if selection_metric == "val_macro_f1":
        return metrics["macro_f1"]
    if selection_metric == "val_acc_weighted_f1":
        return 0.7 * metrics["acc"] + 0.3 * metrics["weighted_f1"]
    if selection_metric == "val_weighted_macro_f1":
        return metrics["weighted_f1"] + metrics["macro_f1"]
    if selection_metric == "val_acc_weighted_macro_f1":
        return metrics["acc"] + metrics["weighted_f1"] + metrics["macro_f1"]
    raise ValueError(f"unsupported selection metric: {selection_metric}")


def run_hierarchical_setting(
    args, started_at, modalities, feature_names, save_dir, device, torch, nn, DataLoader,
    train_df, val_df, test_df, feature_dims, label_names, label_map,
    split_stats, split_label_distribution, config,
):
    if args.task != "emotion_sort":
        raise ValueError("--training-mode hierarchical currently supports emotion_sort only")
    if args.cv_folds != 1:
        raise ValueError("--training-mode hierarchical currently requires --cv-folds 1")

    focus_labels = ["happy", "sad"]
    num_classes = len(label_names)
    coarse_train_df = make_coarse_df(train_df, label_names).reset_index(drop=True)
    coarse_val_df = make_coarse_df(val_df, label_names).reset_index(drop=True)
    coarse_test_df = make_coarse_df(test_df, label_names).reset_index(drop=True)
    expert_train_df = make_expert_df(train_df, label_names, include_all=False)
    expert_val_df = make_expert_df(val_df, label_names, include_all=False)
    expert_test_df = make_expert_df(test_df, label_names, include_all=False)
    expert_val_all_df = make_expert_df(val_df, label_names, include_all=True)
    expert_test_all_df = make_expert_df(test_df, label_names, include_all=True)

    if expert_train_df.empty or expert_val_df.empty or expert_test_df.empty:
        raise ValueError("hierarchical expert split is empty; cannot train other expert")

    print("hierarchical mode: stage1=angry/happy/surprised/other")
    print("hierarchical mode: stage2=disgusted/sad/sarcastic/scared/shy")
    print("hierarchical sizes:", {
        "coarse_train": len(coarse_train_df), "coarse_val": len(coarse_val_df),
        "coarse_test": len(coarse_test_df), "expert_train": len(expert_train_df),
        "expert_val": len(expert_val_df), "expert_test": len(expert_test_df),
    })

    coarse_result = train_one_fold(
        fold_name="coarse",
        train_df=coarse_train_df, val_df=coarse_val_df, test_df=coarse_test_df,
        feature_dims=feature_dims, args=args, modalities=modalities,
        num_classes=len(HIERARCHICAL_COARSE_LABELS),
        device=device, torch=torch, nn=nn, DataLoader=DataLoader,
        save_dir=save_dir, started_at=started_at,
        label_names=HIERARCHICAL_COARSE_LABELS, focus_labels=[],
    )
    expert_result = train_one_fold(
        fold_name="expert_other",
        train_df=expert_train_df, val_df=expert_val_df, test_df=expert_test_df,
        feature_dims=feature_dims, args=args, modalities=modalities,
        num_classes=len(HIERARCHICAL_EXPERT_LABELS),
        device=device, torch=torch, nn=nn, DataLoader=DataLoader,
        save_dir=save_dir, started_at=started_at,
        label_names=HIERARCHICAL_EXPERT_LABELS, focus_labels=[],
        extra_eval_dfs={"val_all": expert_val_all_df, "test_all": expert_test_all_df},
    )

    val_labels = val_df["label_id"].astype(int).to_numpy()
    test_labels = test_df["label_id"].astype(int).to_numpy()
    val_probs = combine_hierarchical_probs(
        coarse_result["val_probs"],
        expert_result["extra_predictions"]["val_all"]["probs"],
        label_names,
    )
    test_probs = combine_hierarchical_probs(
        coarse_result["test_probs"],
        expert_result["extra_predictions"]["test_all"]["probs"],
        label_names,
    )
    val_metrics = compute_metrics(
        val_labels, val_probs.argmax(axis=1), num_classes,
        label_names=label_names, focus_labels=focus_labels,
    )
    test_metrics = compute_metrics(
        test_labels, test_probs.argmax(axis=1), num_classes,
        label_names=label_names, focus_labels=focus_labels,
    )
    best_score = score_from_metrics(val_metrics, args.selection_metric)

    run_stem = build_run_stem(args, started_at, modalities, feature_names, test_metrics)
    config_path = save_dir / f"{run_stem}.config.json"
    metrics_path = save_dir / f"{run_stem}.metrics.json"
    checkpoint_paths = []
    for stage_name, result in [("coarse", coarse_result), ("expert_other", expert_result)]:
        checkpoint_path = save_dir / f"{run_stem}.{stage_name}.pt"
        result["best_path"].replace(checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

    normalization_stats = {
        "coarse": coarse_result["normalization_stats"],
        "expert_other": expert_result["normalization_stats"],
    }
    metrics = {
        "best_epoch": {
            "coarse": coarse_result["best_epoch"],
            "expert_other": expert_result["best_epoch"],
        },
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "cv_folds": args.cv_folds,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_attention": {
            "coarse": coarse_result["test_attention"],
            "expert_other": expert_result["test_attention"],
        },
        "per_class_metrics": test_metrics["per_class_metrics"],
        "focus_class_metrics": test_metrics["focus_class_metrics"],
        "happy_sad_metrics": test_metrics["focus_class_metrics"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "confusion_matrix_labels": test_metrics["confusion_matrix_labels"],
        "split_label_distribution": split_label_distribution,
        "feature_normalization": args.feature_normalization,
        "modality_dropout": args.modality_dropout,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "class_weight": args.class_weight,
        "sampler": args.sampler,
        "hierarchical": {
            "coarse_labels": HIERARCHICAL_COARSE_LABELS,
            "expert_labels": HIERARCHICAL_EXPERT_LABELS,
            "final_rule": (
                "p(angry/happy/surprised)=coarse direct; "
                "p(other-class)=p(coarse_other)*p(expert_class)"
            ),
            "val_metrics": val_metrics,
            "coarse_test_metrics": coarse_result["test_metrics"],
            "expert_other_test_metrics": expert_result["test_metrics"],
            "stage_sizes": {
                "coarse_train": len(coarse_train_df), "coarse_val": len(coarse_val_df),
                "coarse_test": len(coarse_test_df), "expert_train": len(expert_train_df),
                "expert_val": len(expert_val_df), "expert_test": len(expert_test_df),
            },
        },
        "folds": [
            {
                "fold": "coarse",
                "best_epoch": coarse_result["best_epoch"],
                "selection_metric": coarse_result["selection_metric"],
                "best_selection_score": coarse_result["best_selection_score"],
                "test_metrics": coarse_result["test_metrics"],
                "ensemble_candidates": [
                    {
                        "epoch": c["epoch"], "score": c["score"], "path": str(c["path"]),
                        "val_acc": c["record"]["val_acc"],
                        "val_weighted_f1": c["record"]["val_weighted_f1"],
                        "val_macro_f1": c["record"]["val_macro_f1"],
                    }
                    for c in coarse_result["ensemble_candidates"]
                ],
            },
            {
                "fold": "expert_other",
                "best_epoch": expert_result["best_epoch"],
                "selection_metric": expert_result["selection_metric"],
                "best_selection_score": expert_result["best_selection_score"],
                "test_metrics": expert_result["test_metrics"],
                "ensemble_candidates": [
                    {
                        "epoch": c["epoch"], "score": c["score"], "path": str(c["path"]),
                        "val_acc": c["record"]["val_acc"],
                        "val_weighted_f1": c["record"]["val_weighted_f1"],
                        "val_macro_f1": c["record"]["val_macro_f1"],
                    }
                    for c in expert_result["ensemble_candidates"]
                ],
            },
        ],
        "history": {
            "coarse": coarse_result["history"],
            "expert_other": expert_result["history"],
        },
    }

    prediction_paths = {}
    if args.prediction_dir is not None:
        val_prediction_path = args.prediction_dir / f"{run_stem}__val_probs.jsonl"
        test_prediction_path = args.prediction_dir / f"{run_stem}__test_probs.jsonl"
        save_probability_jsonl(val_prediction_path, val_df, val_labels, val_probs, label_names, "val")
        save_probability_jsonl(test_prediction_path, test_df, test_labels, test_probs, label_names, "test")
        prediction_paths = {"val": str(val_prediction_path), "test": str(test_prediction_path)}
        metrics["prediction_paths"] = prediction_paths
        print("Prediction probabilities:", prediction_paths)

    config.update({
        "training_mode": args.training_mode,
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "run_stem": run_stem,
        "normalization_stats": normalization_stats,
        "prediction_paths": prediction_paths,
        "hierarchical": metrics["hierarchical"],
    })
    save_json(config_path, json_safe(config))
    save_json(metrics_path, metrics)
    log_path = save_result_log(args=args, run_stem=run_stem, config=config, metrics=metrics)

    print("Hierarchical val acc:", val_metrics["acc"])
    print("Hierarchical val weighted_f1:", val_metrics["weighted_f1"])
    print("Hierarchical test acc:", test_metrics["acc"])
    print("Hierarchical test macro_f1:", test_metrics["macro_f1"])
    print("Hierarchical test weighted_f1:", test_metrics["weighted_f1"])
    print("Checkpoints:", ", ".join(str(p) for p in checkpoint_paths))
    print("Config:", config_path)
    print("Metrics:", metrics_path)
    print("Result log:", log_path)

    return {
        "feature_names": feature_names,
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    }


def run_one_setting(
    args, started_at, modalities, feature_roots, feature_names, save_dir,
    device, torch, nn, DataLoader,
):
    set_seed(args.seed, torch)

    label_map = TASK_LABEL_MAPS[args.task]
    label_names = build_label_names(label_map)
    focus_labels = ["happy", "sad"] if args.task == "emotion_sort" else []
    num_classes = len(label_map)
    manifest = args.manifest
    if manifest is None:
        manifest = DEFAULT_TASK_MANIFESTS[args.task]

    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("modalities:", ",".join(modalities))
    print("output_task_dir:", get_output_task_dir(args.task))
    print("feature_names:", {modality: feature_names[modality] for modality in modalities})
    print("feature_roots:", {modality: str(feature_roots[modality]) for modality in modalities})
    print("fusion:", args.fusion)
    print("training_mode:", args.training_mode)
    print("save_dir:", save_dir)
    print("log_dir:", args.log_dir)

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

    split_label_distribution = build_split_label_distribution(train_df, val_df, test_df, label_names)
    feature_dims = infer_feature_dims(train_df, modalities)
    print("feature_dims:", feature_dims)
    print("device:", device)
    print("cv_folds:", args.cv_folds)
    print("feature_normalization:", args.feature_normalization)
    print("modality_dropout:", args.modality_dropout)
    print("imbalance:", {
        "loss": args.loss, "class_weight": args.class_weight,
        "sampler": args.sampler, "class_weight_beta": args.class_weight_beta,
        "class_weight_max": args.class_weight_max,
    })

    config = vars(args).copy()
    config.update({
        "output_task_dir": get_output_task_dir(args.task),
        "manifest": str(manifest),
        "feature_roots": {modality: str(feature_roots[modality]) for modality in modalities},
        "feature_names": feature_names,
        "save_dir": str(save_dir),
        "log_dir": str(args.log_dir),
        "modalities": modalities,
        "feature_dims": feature_dims,
        "num_classes": num_classes,
        "label_names": label_names,
        "label_map": {str(key): value for key, value in label_map.items()},
        "split_label_distribution": split_label_distribution,
        "feature_normalization": args.feature_normalization,
        "zscore_eps": args.zscore_eps,
        "modality_dropout": args.modality_dropout,
        "lr_scheduler": args.lr_scheduler,
        "warmup_epochs": args.warmup_epochs,
        "min_lr_ratio": args.min_lr_ratio,
        "selection_tolerance": args.selection_tolerance,
        "selection_tie_breaker": args.selection_tie_breaker,
        "selection_min_val_acc": args.selection_min_val_acc,
        "selection_min_val_weighted_f1": args.selection_min_val_weighted_f1,
        "ensemble_top_k": args.ensemble_top_k,
        "ensemble_metric": args.ensemble_metric,
        "use_person_embedding": args.use_person_embedding,
        "person_embedding_dim": args.person_embedding_dim,
        "use_person_prior": args.use_person_prior,
        "person_prior_smoothing": args.person_prior_smoothing,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "class_weight": args.class_weight,
        "class_weight_beta": args.class_weight_beta,
        "class_weight_max": args.class_weight_max,
        "sampler": args.sampler,
        "device": device,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "split_stats": split_stats,
    })

    if args.training_mode == "hierarchical":
        return run_hierarchical_setting(
            args=args, started_at=started_at, modalities=modalities,
            feature_names=feature_names, save_dir=save_dir,
            device=device, torch=torch, nn=nn, DataLoader=DataLoader,
            train_df=train_df, val_df=val_df, test_df=test_df,
            feature_dims=feature_dims, label_names=label_names, label_map=label_map,
            split_stats=split_stats, split_label_distribution=split_label_distribution,
            config=config,
        )

    if args.cv_folds == 1:
        fold_results = [
            train_one_fold(
                fold_name="run",
                train_df=train_df, val_df=val_df, test_df=test_df,
                feature_dims=feature_dims, args=args, modalities=modalities,
                num_classes=num_classes, device=device, torch=torch, nn=nn,
                DataLoader=DataLoader, save_dir=save_dir, started_at=started_at,
                label_names=label_names, focus_labels=focus_labels,
            )
        ]
        best_epoch = fold_results[0]["best_epoch"]
        best_score = fold_results[0]["best_selection_score"]
        test_metrics = fold_results[0]["test_metrics"]
        test_attention = fold_results[0]["test_attention"]
    else:
        dev_df = pd.concat([train_df, val_df], ignore_index=True)
        folds = build_stratified_folds(dev_df["label_id"].tolist(), args.cv_folds, args.seed)
        all_indices = np.arange(len(dev_df))
        fold_results = []
        for fold_idx, val_indices in enumerate(folds, start=1):
            val_index_set = set(val_indices)
            train_indices = [int(i) for i in all_indices if int(i) not in val_index_set]
            fold_train_df = dev_df.iloc[train_indices].reset_index(drop=True)
            fold_val_df = dev_df.iloc[val_indices].reset_index(drop=True)
            print(
                f"===== CV fold {fold_idx}/{args.cv_folds}: "
                f"train={len(fold_train_df)} val={len(fold_val_df)} test={len(test_df)} ====="
            )
            fold_results.append(
                train_one_fold(
                    fold_name=f"fold{fold_idx}",
                    train_df=fold_train_df, val_df=fold_val_df, test_df=test_df,
                    feature_dims=feature_dims, args=args, modalities=modalities,
                    num_classes=num_classes, device=device, torch=torch, nn=nn,
                    DataLoader=DataLoader, save_dir=save_dir, started_at=started_at,
                    label_names=label_names, focus_labels=focus_labels,
                )
            )
        test_labels = fold_results[0]["test_labels"]
        for result in fold_results[1:]:
            if not np.array_equal(test_labels, result["test_labels"]):
                raise ValueError("test labels differ across folds")
        mean_probs = np.mean([result["test_probs"] for result in fold_results], axis=0)
        test_metrics = compute_metrics(
            test_labels, mean_probs.argmax(axis=1), num_classes,
            label_names=label_names, focus_labels=focus_labels,
        )
        test_attention = average_attention_list(
            [result["test_attention"] for result in fold_results], modalities,
        )
        best_epoch = "cv"
        best_score = float(np.mean([result["best_selection_score"] for result in fold_results]))

    run_stem = build_run_stem(args, started_at, modalities, feature_names, test_metrics)
    config_path = save_dir / f"{run_stem}.config.json"
    metrics_path = save_dir / f"{run_stem}.metrics.json"
    checkpoint_paths = []
    for result in fold_results:
        suffix = ".pt" if args.cv_folds == 1 else f".{result['fold']}.pt"
        checkpoint_path = save_dir / f"{run_stem}{suffix}"
        result["best_path"].replace(checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

    normalization_stats = (
        fold_results[0]["normalization_stats"]
        if args.cv_folds == 1
        else {result["fold"]: result["normalization_stats"] for result in fold_results}
    )
    metrics = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "cv_folds": args.cv_folds,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_attention": test_attention,
        "per_class_metrics": test_metrics["per_class_metrics"],
        "focus_class_metrics": test_metrics["focus_class_metrics"],
        "happy_sad_metrics": test_metrics["focus_class_metrics"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "confusion_matrix_labels": test_metrics["confusion_matrix_labels"],
        "split_label_distribution": split_label_distribution,
        "feature_normalization": args.feature_normalization,
        "modality_dropout": args.modality_dropout,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "class_weight": args.class_weight,
        "sampler": args.sampler,
        "folds": [
            {
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "selection_metric": result["selection_metric"],
                "best_selection_score": result["best_selection_score"],
                "test_metrics": result["test_metrics"],
                "test_attention": result["test_attention"],
                "normalization_stats": result["normalization_stats"],
                "class_weights": (
                    result["class_weights"].tolist()
                    if result["class_weights"] is not None else None
                ),
                "sampler_class_weights": (
                    result["sampler_class_weights"].tolist()
                    if result["sampler_class_weights"] is not None else None
                ),
                "person_vocab_size": (
                    len(result["person_vocab"]) if result["person_vocab"] is not None else 0
                ),
                "use_person_prior": args.use_person_prior,
                "ensemble_candidates": [
                    {
                        "epoch": c["epoch"], "score": c["score"], "path": str(c["path"]),
                        "val_acc": c["record"]["val_acc"],
                        "val_weighted_f1": c["record"]["val_weighted_f1"],
                        "val_macro_f1": c["record"]["val_macro_f1"],
                    }
                    for c in result["ensemble_candidates"]
                ],
            }
            for result in fold_results
        ],
        "history": [result["history"] for result in fold_results],
    }

    prediction_paths = {}
    if args.prediction_dir is not None:
        if args.cv_folds != 1:
            print("prediction export skipped: --prediction-dir currently supports cv_folds=1")
        else:
            val_prediction_path = args.prediction_dir / f"{run_stem}__val_probs.jsonl"
            test_prediction_path = args.prediction_dir / f"{run_stem}__test_probs.jsonl"
            save_probability_jsonl(
                val_prediction_path, val_df,
                fold_results[0]["val_labels"], fold_results[0]["val_probs"],
                label_names, "val",
            )
            save_probability_jsonl(
                test_prediction_path, test_df,
                fold_results[0]["test_labels"], fold_results[0]["test_probs"],
                label_names, "test",
            )
            prediction_paths = {"val": str(val_prediction_path), "test": str(test_prediction_path)}
            metrics["prediction_paths"] = prediction_paths
            print("Prediction probabilities:", prediction_paths)

    config.update({
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "run_stem": run_stem,
        "normalization_stats": normalization_stats,
        "prediction_paths": prediction_paths,
    })
    save_json(config_path, json_safe(config))
    save_json(metrics_path, metrics)
    log_path = save_result_log(args=args, run_stem=run_stem, config=config, metrics=metrics)

    print("Best epoch:", best_epoch)
    print("Selection metric:", args.selection_metric)
    print("Best selection score:", best_score)
    print("Test acc:", test_metrics["acc"])
    print("Test macro_f1:", test_metrics["macro_f1"])
    print("Test weighted_f1:", test_metrics["weighted_f1"])
    print("test_attention:", format_attention(test_attention))
    print("Checkpoints:", ", ".join(str(p) for p in checkpoint_paths))
    print("Config:", config_path)
    print("Metrics:", metrics_path)
    print("Result log:", log_path)

    return {
        "feature_names": feature_names,
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    }


def main():
    parser = build_parser()
    explicit_dests = get_explicit_arg_dests(parser, sys.argv[1:])
    args = parser.parse_args()
    args = apply_preset(args, explicit_dests)
    started_at = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    modalities = parse_modalities(args.modalities)
    args.log_dir = resolve_log_dir(args)
    feature_runs = build_feature_runs(args, modalities)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    summaries = []
    failed = 0
    for index, (run_args, feature_roots, feature_names, save_dir) in enumerate(feature_runs, start=1):
        print(f"[{index}/{len(feature_runs)}] start features={feature_names}")
        try:
            summaries.append(
                run_one_setting(
                    args=run_args,
                    started_at=started_at,
                    modalities=modalities,
                    feature_roots=feature_roots,
                    feature_names=feature_names,
                    save_dir=save_dir,
                    device=device,
                    torch=torch,
                    nn=nn,
                    DataLoader=DataLoader,
                )
            )
        except Exception as e:
            if len(feature_runs) == 1:
                raise
            failed += 1
            print(f"failed features={feature_names}: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Done")
    print(f"total={len(feature_runs)}")
    print(f"success={len(summaries)}")
    print(f"failed={failed}")
    for summary in summaries:
        print(
            f"{summary['feature_names']}: "
            f"ACC={summary['test_acc']:.4f}, "
            f"macro={summary['test_macro_f1']:.4f}, "
            f"weighted={summary['test_weighted_f1']:.4f}, "
            f"log={summary['log_path']}"
        )


if __name__ == "__main__":
    main()