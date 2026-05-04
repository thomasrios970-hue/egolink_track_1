"""
Fuse multimodal model probabilities with clean Qwen val/test predictions.

Alpha is selected on val with ACC first by default:
final_prob = alpha * model_prob + (1 - alpha) * qwen_prob
"""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_LABELS = [
    "angry",
    "disgusted",
    "happy",
    "sad",
    "sarcastic",
    "scared",
    "shy",
    "surprised",
]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_parser():
    parser = argparse.ArgumentParser(description="Calibrate model+Qwen probability fusion.")
    parser.add_argument("--model-val", type=Path, default=None)
    parser.add_argument("--model-test", type=Path, default=None)
    parser.add_argument(
        "--model-predictions-dir",
        type=Path,
        default=None,
        help="Optional directory containing one matching *__val_probs.jsonl and *__test_probs.jsonl pair.",
    )
    parser.add_argument("--qwen-val", type=Path, required=True)
    parser.add_argument("--qwen-test", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument(
        "--metric",
        choices=["val_acc", "val_weighted_f1", "val_acc_weighted_f1"],
        default="val_acc",
    )
    return parser


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_model_prediction_paths(args):
    if args.model_val is not None and args.model_test is not None:
        return args.model_val, args.model_test
    if args.model_predictions_dir is None:
        raise ValueError("provide --model-val/--model-test or --model-predictions-dir")

    val_paths = sorted(
        Path(args.model_predictions_dir).glob("*__val_probs.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not val_paths:
        raise ValueError(f"no *__val_probs.jsonl under {args.model_predictions_dir}")
    val_path = val_paths[0]
    run_stem = val_path.name[: -len("__val_probs.jsonl")]
    test_path = Path(args.model_predictions_dir) / f"{run_stem}__test_probs.jsonl"
    if not test_path.exists():
        raise ValueError(f"matching test prediction file not found: {test_path}")
    return val_path, test_path


def get_label_names(model_rows):
    if not model_rows:
        return DEFAULT_LABELS
    probs = model_rows[0].get("probabilities")
    if isinstance(probs, dict) and probs:
        return list(probs.keys())
    values = model_rows[0].get("probabilities_list")
    if isinstance(values, list) and len(values) == len(DEFAULT_LABELS):
        return DEFAULT_LABELS
    raise ValueError("cannot infer label names from model predictions")


def model_prob(row, label_names):
    if isinstance(row.get("probabilities"), dict):
        return np.asarray([float(row["probabilities"][label]) for label in label_names], dtype=np.float64)
    if isinstance(row.get("probabilities_list"), list):
        return np.asarray(row["probabilities_list"], dtype=np.float64)
    raise ValueError(f"model row has no probabilities: {row.get('row_id')}")


def row_key(row):
    row_id = str(row.get("row_id", "")).strip()
    if row_id:
        return row_id
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    raise ValueError("row has neither row_id nor sample_id")


def qwen_prob(row, label_names):
    probs = np.full(len(label_names), 1.0 / len(label_names), dtype=np.float64)
    if row.get("error"):
        return probs

    answer = str(row.get("answer") or "").strip().upper()
    options = row.get("options") or {}
    if not answer or answer not in LETTERS:
        return probs

    option_text = options.get(answer)
    if option_text is None:
        return probs
    label = str(option_text).strip().lower()
    label_to_id = {name.lower(): idx for idx, name in enumerate(label_names)}
    pred_id = label_to_id.get(label)
    if pred_id is None:
        return probs

    confidence = row.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    if len(label_names) == 1:
        return np.ones(1, dtype=np.float64)
    probs = np.full(
        len(label_names),
        (1.0 - confidence) / (len(label_names) - 1),
        dtype=np.float64,
    )
    probs[pred_id] = confidence
    return probs


def compute_metrics(labels, preds, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    acc = float((labels == preds).mean()) if len(labels) else 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels.tolist(), preds.tolist()):
        confusion[label, pred] += 1

    f1_scores = []
    supports = []
    for cls in range(num_classes):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum() - tp)
        fn = int(confusion[cls, :].sum() - tp)
        support = int(confusion[cls, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        supports.append(support)

    macro_f1 = float(np.mean(f1_scores))
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if sum(supports) else 0.0
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "acc_weighted_f1": 0.7 * acc + 0.3 * weighted_f1,
        "confusion_matrix": confusion.tolist(),
    }


def align_rows(model_rows, qwen_rows, label_names):
    qwen_by_key = {row_key(row): row for row in qwen_rows}
    aligned = []
    missing = []
    for model_row in model_rows:
        key = row_key(model_row)
        qwen_row = qwen_by_key.get(key)
        if qwen_row is None:
            missing.append(key)
            continue
        aligned.append(
            {
                "key": key,
                "model_row": model_row,
                "qwen_row": qwen_row,
                "label_id": int(model_row["label_id"]),
                "model_prob": model_prob(model_row, label_names),
                "qwen_prob": qwen_prob(qwen_row, label_names),
            }
        )
    return aligned, missing


def evaluate_alpha(rows, alpha, num_classes):
    labels = []
    preds = []
    for row in rows:
        final_prob = alpha * row["model_prob"] + (1.0 - alpha) * row["qwen_prob"]
        labels.append(row["label_id"])
        preds.append(int(np.argmax(final_prob)))
    return compute_metrics(labels, preds, num_classes)


def save_test_predictions(path, rows, alpha, label_names):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            final_prob = alpha * row["model_prob"] + (1.0 - alpha) * row["qwen_prob"]
            pred_id = int(np.argmax(final_prob))
            item = {
                "row_id": row["key"],
                "label_id": row["label_id"],
                "label_name": label_names[row["label_id"]],
                "pred_id": pred_id,
                "pred_label": label_names[pred_id],
                "alpha": float(alpha),
                "probabilities": {
                    label_names[idx]: float(final_prob[idx])
                    for idx in range(len(label_names))
                },
            }
            for field in ["Vid", "sub_id", "person", "split"]:
                if field in row["model_row"]:
                    item[field] = row["model_row"][field]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = build_parser().parse_args()
    model_val_path, model_test_path = resolve_model_prediction_paths(args)
    model_val_rows = read_jsonl(model_val_path)
    model_test_rows = read_jsonl(model_test_path)
    qwen_val_rows = read_jsonl(args.qwen_val)
    qwen_test_rows = read_jsonl(args.qwen_test)
    label_names = get_label_names(model_val_rows)
    num_classes = len(label_names)

    val_rows, missing_val = align_rows(model_val_rows, qwen_val_rows, label_names)
    test_rows, missing_test = align_rows(model_test_rows, qwen_test_rows, label_names)
    if not val_rows:
        raise ValueError("no aligned val rows")
    if not test_rows:
        raise ValueError("no aligned test rows")

    alphas = np.arange(0.0, 1.0 + args.alpha_step / 2, args.alpha_step)
    records = []
    for alpha in alphas:
        metrics = evaluate_alpha(val_rows, float(alpha), num_classes)
        records.append({"alpha": float(alpha), "metrics": metrics})

    metric_key = args.metric.replace("val_", "")
    best = max(
        records,
        key=lambda item: (
            item["metrics"][metric_key],
            item["metrics"]["acc"],
            item["metrics"]["weighted_f1"],
            item["alpha"],
        ),
    )
    best_alpha = float(best["alpha"])
    val_metrics = best["metrics"]
    test_metrics = evaluate_alpha(test_rows, best_alpha, num_classes)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    test_prediction_path = args.output_path.with_suffix(".test_predictions.jsonl")
    save_test_predictions(test_prediction_path, test_rows, best_alpha, label_names)

    summary = {
        "model_val": str(model_val_path),
        "model_test": str(model_test_path),
        "qwen_val": str(args.qwen_val),
        "qwen_test": str(args.qwen_test),
        "label_names": label_names,
        "metric": args.metric,
        "alpha_step": args.alpha_step,
        "best_alpha": best_alpha,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "aligned_val_rows": len(val_rows),
        "aligned_test_rows": len(test_rows),
        "missing_val_rows": missing_val[:50],
        "missing_test_rows": missing_test[:50],
        "test_prediction_path": str(test_prediction_path),
        "alpha_records": records,
    }
    args.output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("model_val:", model_val_path)
    print("model_test:", model_test_path)
    print("qwen_val:", args.qwen_val)
    print("qwen_test:", args.qwen_test)
    print("best_alpha:", f"{best_alpha:.2f}")
    print(
        "val:",
        f"ACC={val_metrics['acc']:.4f}",
        f"WAF={val_metrics['weighted_f1']:.4f}",
        f"macro={val_metrics['macro_f1']:.4f}",
    )
    print(
        "test:",
        f"ACC={test_metrics['acc']:.4f}",
        f"WAF={test_metrics['weighted_f1']:.4f}",
        f"macro={test_metrics['macro_f1']:.4f}",
    )
    print("summary:", args.output_path)
    print("test_predictions:", test_prediction_path)


if __name__ == "__main__":
    main()
