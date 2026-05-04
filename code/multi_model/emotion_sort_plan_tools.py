"""
Utilities for the emotion_sort v5 sprint plan.

Subcommands:
  assert-target-text   Validate row-aware target_text feature alignment.
  conflict-metrics    Compute overall/conflict/non-conflict metrics for predictions.
  diversity           Compute pairwise Cohen's kappa and error overlap.
  ensemble            Greedy val-only cross-model probability ensemble.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_NAMES = [
    "angry",
    "disgusted",
    "happy",
    "sad",
    "sarcastic",
    "scared",
    "shy",
    "surprised",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
DEFAULT_MANIFEST = Path("/data/wzw/egolink_race/data/manifest/emotion_sort_manifest.csv")
DEFAULT_TARGET_TEXT_ROOT = Path(
    "/data/wzw/egolink_race/feature/target_txt_features/xlm-roberta-xl"
)


def build_parser():
    parser = argparse.ArgumentParser(description="emotion_sort sprint plan utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_manifest_arg(
        subparsers.add_parser(
            "assert-target-text",
            help="Validate target_text row_id feature alignment.",
        )
    )
    subparsers.choices["assert-target-text"].add_argument(
        "--target-root", type=Path, default=DEFAULT_TARGET_TEXT_ROOT
    )
    subparsers.choices["assert-target-text"].add_argument(
        "--output-path", type=Path, default=None
    )

    conflict = subparsers.add_parser(
        "conflict-metrics",
        help="Compute conflict/non-conflict metrics for prediction JSONL files.",
    )
    add_manifest_arg(conflict)
    conflict.add_argument("--prediction", type=Path, action="append", required=True)
    conflict.add_argument("--output-path", type=Path, required=True)

    diversity = subparsers.add_parser(
        "diversity",
        help="Compute pairwise kappa and error overlap on aligned predictions.",
    )
    add_manifest_arg(diversity)
    diversity.add_argument("--prediction", type=Path, action="append", required=True)
    diversity.add_argument("--split", choices=["val", "test"], default="val")
    diversity.add_argument("--output-path", type=Path, required=True)

    ensemble = subparsers.add_parser(
        "ensemble",
        help="Greedy cross-model probability ensemble selected only on val.",
    )
    add_manifest_arg(ensemble)
    ensemble.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as name:val_path:test_path",
    )
    ensemble.add_argument("--output-path", type=Path, required=True)
    ensemble.add_argument("--test-prediction-path", type=Path, default=None)
    ensemble.add_argument("--max-models", type=int, default=5)
    ensemble.add_argument("--min-gain", type=float, default=0.003)
    ensemble.add_argument("--kappa-threshold", type=float, default=0.85)
    ensemble.add_argument("--alpha-step", type=float, default=0.05)
    ensemble.add_argument(
        "--temperatures",
        default="0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated per-candidate probability temperatures searched on val.",
    )
    return parser


def add_manifest_arg(parser):
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def read_manifest(path):
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    if "row_id" not in df.columns:
        df["row_id"] = [f"row_{idx:06d}" for idx in range(len(df))]
    else:
        df["row_id"] = [
            str(value).strip() if str(value).strip() and str(value) != "nan" else f"row_{idx:06d}"
            for idx, value in enumerate(df["row_id"].tolist())
        ]
    df["label_id"] = [
        LABEL_TO_ID[str(emotion).strip().lower()]
        for emotion in df["emotion"].tolist()
    ]
    return df


def conflict_row_ids(df):
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        grouped[(str(row["Vid"]), str(row["sub_id"]))].add(str(row["emotion"]).strip().lower())
    conflict_keys = {key for key, labels in grouped.items() if len(labels) > 1}
    return {
        str(row["row_id"])
        for _, row in df.iterrows()
        if (str(row["Vid"]), str(row["sub_id"])) in conflict_keys
    }


def assert_target_text(args):
    df = read_manifest(args.manifest)
    feature_dir = args.target_root / "UTTERANCE"
    expected_count = len(df)
    errors = []

    if expected_count != 16307:
        errors.append(f"manifest row count expected 16307, got {expected_count}")

    for idx, row in df.iterrows():
        row_id = str(row["row_id"])
        expected_row_id = f"row_{idx:06d}"
        if row_id != expected_row_id:
            errors.append(f"row {idx}: row_id={row_id}, expected {expected_row_id}")
            if len(errors) >= 20:
                break
        path = feature_dir / f"{row_id}.npy"
        if not path.exists():
            errors.append(f"missing feature: {path}")
            if len(errors) >= 20:
                break
        elif path.stem != row_id:
            errors.append(f"misaligned feature stem for row {idx}: {path}")
            if len(errors) >= 20:
                break

    actual_count = sum(1 for _ in feature_dir.glob("row_*.npy")) if feature_dir.exists() else 0
    if actual_count != expected_count:
        errors.append(f"feature count expected {expected_count}, got {actual_count}")

    summary = {
        "manifest": str(args.manifest),
        "target_root": str(args.target_root),
        "feature_dir": str(feature_dir),
        "expected_rows": expected_count,
        "feature_count": actual_count,
        "ok": not errors,
        "errors": errors,
    }
    write_summary(args.output_path, summary)
    if errors:
        raise SystemExit("target_text assertion failed: " + "; ".join(errors[:3]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row):
    for key in ["row_id", "sample_id", "question_id"]:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError(f"prediction row has no row key: {row}")


def probabilities_from_row(row):
    if isinstance(row.get("probabilities_list"), list):
        probs = np.asarray(row["probabilities_list"], dtype=np.float64)
        return normalize_probs(probs)
    if isinstance(row.get("probabilities"), dict):
        probs = np.asarray([row["probabilities"][label] for label in LABEL_NAMES], dtype=np.float64)
        return normalize_probs(probs)
    if row.get("answer") and isinstance(row.get("options"), dict):
        probs = np.full(len(LABEL_NAMES), 1e-6, dtype=np.float64)
        label = str(row["options"].get(str(row["answer"]).strip().upper(), "")).strip().lower()
        pred_id = LABEL_TO_ID.get(label)
        if pred_id is None:
            return np.full(len(LABEL_NAMES), 1.0 / len(LABEL_NAMES), dtype=np.float64)
        confidence = row.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))
        probs[:] = (1.0 - confidence) / (len(LABEL_NAMES) - 1)
        probs[pred_id] = confidence
        return normalize_probs(probs)
    pred_label = str(row.get("pred_label") or row.get("prediction") or "").strip().lower()
    if pred_label in LABEL_TO_ID:
        probs = np.full(len(LABEL_NAMES), 1e-6, dtype=np.float64)
        probs[LABEL_TO_ID[pred_label]] = 1.0
        return normalize_probs(probs)
    raise ValueError(f"cannot infer probabilities for row {row_key(row)}")


def normalize_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if total <= 0:
        return np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
    return probs / total


def label_from_row(row, manifest_labels):
    if "label_id" in row:
        return int(row["label_id"])
    key = row_key(row)
    if key in manifest_labels:
        return int(manifest_labels[key])
    if row.get("gold") and isinstance(row.get("options"), dict):
        label = str(row["options"].get(str(row["gold"]).strip().upper(), "")).strip().lower()
        if label in LABEL_TO_ID:
            return LABEL_TO_ID[label]
    raise ValueError(f"cannot infer label for row {key}")


def load_prediction_table(path, manifest_df=None):
    manifest_labels = {}
    if manifest_df is not None:
        manifest_labels = {
            str(row["row_id"]): int(row["label_id"])
            for _, row in manifest_df.iterrows()
    }
    rows = []
    for row in read_jsonl(path):
        if row.get("error") and not any(row.get(key) for key in ("probabilities", "probabilities_list", "pred_label", "prediction", "answer")):
            continue
        probs = probabilities_from_row(row)
        key = row_key(row)
        rows.append(
            {
                "row_id": key,
                "split": str(row.get("split", "")).strip().lower(),
                "label_id": label_from_row(row, manifest_labels),
                "pred_id": int(np.argmax(probs)),
                "probs": probs,
            }
        )
    return rows


def compute_metrics(labels, preds):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    num_classes = len(LABEL_NAMES)
    acc = float((labels == preds).mean()) if len(labels) else 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels.tolist(), preds.tolist()):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            confusion[label, pred] += 1

    f1_scores = []
    supports = []
    per_class = {}
    for cls, label_name in enumerate(LABEL_NAMES):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum() - tp)
        fn = int(confusion[cls, :].sum() - tp)
        support = int(confusion[cls, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        supports.append(support)
        per_class[label_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if sum(supports) else 0.0
    return {
        "acc": acc,
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": weighted_f1,
        "objective": 0.5 * acc + 0.5 * weighted_f1,
        "support": int(len(labels)),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
    }


def subset_metrics(rows, conflict_ids):
    groups = {
        "overall": rows,
        "conflict": [row for row in rows if row["row_id"] in conflict_ids],
        "non_conflict": [row for row in rows if row["row_id"] not in conflict_ids],
    }
    out = {}
    for name, group_rows in groups.items():
        out[name] = compute_metrics(
            [row["label_id"] for row in group_rows],
            [row["pred_id"] for row in group_rows],
        )
    return out


def conflict_metrics(args):
    manifest_df = read_manifest(args.manifest)
    conflict_ids = conflict_row_ids(manifest_df)
    results = {
        "manifest": str(args.manifest),
        "conflict_row_count": len(conflict_ids),
        "predictions": {},
    }
    for path in args.prediction:
        rows = load_prediction_table(path, manifest_df)
        by_split = defaultdict(list)
        for row in rows:
            split = row["split"] or "unknown"
            by_split[split].append(row)
        results["predictions"][str(path)] = {
            split: subset_metrics(split_rows, conflict_ids)
            for split, split_rows in sorted(by_split.items())
        }
    write_summary(args.output_path, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cohen_kappa(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if len(a) != len(b):
        raise ValueError("kappa inputs must have equal length")
    if len(a) == 0:
        return 0.0
    observed = float((a == b).mean())
    counts_a = np.bincount(a, minlength=len(LABEL_NAMES)) / len(a)
    counts_b = np.bincount(b, minlength=len(LABEL_NAMES)) / len(b)
    expected = float(np.dot(counts_a, counts_b))
    if math.isclose(1.0, expected):
        return 1.0
    return (observed - expected) / (1.0 - expected)


def error_overlap(labels, preds_a, preds_b):
    labels = np.asarray(labels, dtype=np.int64)
    err_a = np.asarray(preds_a, dtype=np.int64) != labels
    err_b = np.asarray(preds_b, dtype=np.int64) != labels
    denom = int(np.logical_or(err_a, err_b).sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(err_a, err_b).sum() / denom)


def aligned_prediction_arrays(paths, manifest_df, split=None):
    tables = []
    for path in paths:
        rows = load_prediction_table(path, manifest_df)
        if split is not None:
            rows = [row for row in rows if row["split"] in {"", split}]
        tables.append({row["row_id"]: row for row in rows})
    common = sorted(set.intersection(*(set(table) for table in tables)))
    labels = np.asarray([tables[0][key]["label_id"] for key in common], dtype=np.int64)
    preds = []
    probs = []
    for table in tables:
        preds.append(np.asarray([table[key]["pred_id"] for key in common], dtype=np.int64))
        probs.append(np.asarray([table[key]["probs"] for key in common], dtype=np.float64))
    return common, labels, preds, probs


def diversity(args):
    manifest_df = read_manifest(args.manifest)
    common, labels, preds, _ = aligned_prediction_arrays(
        args.prediction,
        manifest_df,
        split=args.split,
    )
    names = [path.stem for path in args.prediction]
    pairs = []
    for i in range(len(args.prediction)):
        for j in range(i + 1, len(args.prediction)):
            pairs.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "kappa": float(cohen_kappa(preds[i], preds[j])),
                    "error_overlap": error_overlap(labels, preds[i], preds[j]),
                }
            )
    summary = {
        "split": args.split,
        "aligned_rows": len(common),
        "predictions": [str(path) for path in args.prediction],
        "single_metrics": {
            names[idx]: compute_metrics(labels, preds[idx])
            for idx in range(len(names))
        },
        "pairs": pairs,
    }
    write_summary(args.output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_candidate(text):
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"candidate must be name:val_path:test_path, got {text}")
    return parts[0], Path(parts[1]), Path(parts[2])


def apply_temperature(probs, temperature):
    probs = np.asarray(probs, dtype=np.float64)
    if math.isclose(temperature, 1.0):
        return probs
    logits = np.log(np.maximum(probs, 1e-12)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def best_temperature(labels, probs, temperatures):
    best_temp = temperatures[0]
    best_probs = apply_temperature(probs, best_temp)
    best_metrics = compute_metrics(labels, best_probs.argmax(axis=1))
    for temp in temperatures[1:]:
        temp_probs = apply_temperature(probs, temp)
        metrics = compute_metrics(labels, temp_probs.argmax(axis=1))
        if (metrics["objective"], metrics["acc"], metrics["weighted_f1"]) > (
            best_metrics["objective"], best_metrics["acc"], best_metrics["weighted_f1"]
        ):
            best_temp = temp
            best_probs = temp_probs
            best_metrics = metrics
    return best_temp, best_probs, best_metrics


def ensemble(args):
    manifest_df = read_manifest(args.manifest)
    candidates = [parse_candidate(text) for text in args.candidate]
    candidate_count = len(candidates)
    max_models = min(args.max_models, candidate_count)
    temperatures = [float(item) for item in args.temperatures.split(",") if item.strip()]

    names = [item[0] for item in candidates]
    val_paths = [item[1] for item in candidates]
    test_paths = [item[2] for item in candidates]
    common_val, labels_val, preds_val, probs_val_raw = aligned_prediction_arrays(val_paths, manifest_df, split="val")
    common_test, labels_test, _, probs_test_raw = aligned_prediction_arrays(test_paths, manifest_df, split="test")

    calibrated = []
    for idx, name in enumerate(names):
        temp, val_probs, val_metrics = best_temperature(labels_val, probs_val_raw[idx], temperatures)
        test_probs = apply_temperature(probs_test_raw[idx], temp)
        calibrated.append(
            {
                "name": name,
                "temperature": temp,
                "val_probs": val_probs,
                "test_probs": test_probs,
                "val_metrics": val_metrics,
                "val_preds": val_probs.argmax(axis=1),
            }
        )

    selected = []
    remaining = list(range(candidate_count))
    best_idx = max(
        remaining,
        key=lambda idx: (
            calibrated[idx]["val_metrics"]["objective"],
            calibrated[idx]["val_metrics"]["acc"],
            calibrated[idx]["val_metrics"]["weighted_f1"],
        ),
    )
    selected.append({"idx": best_idx, "weight": 1.0})
    remaining.remove(best_idx)
    current_val = calibrated[best_idx]["val_probs"]
    current_test = calibrated[best_idx]["test_probs"]
    current_metrics = compute_metrics(labels_val, current_val.argmax(axis=1))
    steps = [
        {
            "action": "start",
            "model": names[best_idx],
            "metrics": current_metrics,
        }
    ]

    while remaining and len(selected) < max_models:
        best_add = None
        for idx in list(remaining):
            if any(
                cohen_kappa(calibrated[idx]["val_preds"], calibrated[item["idx"]]["val_preds"])
                >= args.kappa_threshold
                for item in selected
            ):
                continue
            alphas = np.arange(args.alpha_step, 1.0, args.alpha_step)
            for alpha in alphas:
                mixed = alpha * current_val + (1.0 - alpha) * calibrated[idx]["val_probs"]
                metrics = compute_metrics(labels_val, mixed.argmax(axis=1))
                gain = metrics["objective"] - current_metrics["objective"]
                candidate_add = {
                    "idx": idx,
                    "alpha_current": float(alpha),
                    "metrics": metrics,
                    "gain": float(gain),
                    "val_probs": mixed,
                    "test_probs": alpha * current_test + (1.0 - alpha) * calibrated[idx]["test_probs"],
                }
                if best_add is None or (
                    candidate_add["gain"],
                    metrics["acc"],
                    metrics["weighted_f1"],
                ) > (
                    best_add["gain"],
                    best_add["metrics"]["acc"],
                    best_add["metrics"]["weighted_f1"],
                ):
                    best_add = candidate_add
        if best_add is None or best_add["gain"] < args.min_gain:
            break
        decay = best_add["alpha_current"]
        for item in selected:
            item["weight"] *= decay
        selected.append({"idx": best_add["idx"], "weight": 1.0 - decay})
        remaining.remove(best_add["idx"])
        current_val = best_add["val_probs"]
        current_test = best_add["test_probs"]
        current_metrics = best_add["metrics"]
        steps.append(
            {
                "action": "add",
                "model": names[best_add["idx"]],
                "alpha_current": decay,
                "gain": best_add["gain"],
                "metrics": current_metrics,
            }
        )

    test_metrics = compute_metrics(labels_test, current_test.argmax(axis=1))
    selected_models = [
        {
            "name": names[item["idx"]],
            "weight": float(item["weight"]),
            "temperature": float(calibrated[item["idx"]]["temperature"]),
            "val_path": str(val_paths[item["idx"]]),
            "test_path": str(test_paths[item["idx"]]),
            "single_val_metrics": calibrated[item["idx"]]["val_metrics"],
        }
        for item in selected
    ]
    summary = {
        "decision_split": "val",
        "aligned_val_rows": len(common_val),
        "aligned_test_rows": len(common_test),
        "cross_model_top_k": max_models,
        "min_gain": args.min_gain,
        "kappa_threshold": args.kappa_threshold,
        "selected_models": selected_models,
        "steps": steps,
        "val_metrics": current_metrics,
        "test_metrics": test_metrics,
        "clean_val_below_065": (
            current_metrics["acc"] < 0.65 or current_metrics["weighted_f1"] < 0.65
        ),
        "upper_bound_trigger_rule": "val clean ACC < 0.65 or val clean WAF < 0.65",
    }

    test_prediction_path = args.test_prediction_path or args.output_path.with_suffix(".test_predictions.jsonl")
    save_ensemble_predictions(test_prediction_path, common_test, labels_test, current_test)
    summary["test_prediction_path"] = str(test_prediction_path)
    write_summary(args.output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def save_ensemble_predictions(path, row_ids, labels, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    with path.open("w", encoding="utf-8") as f:
        for idx, row_id in enumerate(row_ids):
            pred_id = int(np.argmax(probs[idx]))
            item = {
                "row_id": row_id,
                "label_id": int(labels[idx]),
                "label_name": LABEL_NAMES[int(labels[idx])],
                "pred_id": pred_id,
                "pred_label": LABEL_NAMES[pred_id],
                "probabilities": {
                    LABEL_NAMES[cls]: float(probs[idx, cls])
                    for cls in range(len(LABEL_NAMES))
                },
                "probabilities_list": [
                    float(probs[idx, cls]) for cls in range(len(LABEL_NAMES))
                ],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_summary(path, summary):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = build_parser().parse_args()
    if args.command == "assert-target-text":
        assert_target_text(args)
    elif args.command == "conflict-metrics":
        conflict_metrics(args)
    elif args.command == "diversity":
        diversity(args)
    elif args.command == "ensemble":
        ensemble(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
