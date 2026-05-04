"""
评估 Qwen baseline 输出的 JSONL 结果。

脚本作用：
1. 读取 baseline 生成的 JSONL。
2. 根据 answer/gold 或 correct 字段统计准确率。
3. 统计成功、失败、跳过、答案分布和各类别正确率。

运行示例：
python code/baseline/evaluate_qwen_results.py \
  data/processed/baseline_v1/qwen_omni/mcp_qwen_omni_baseline_v1.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="评估 Qwen baseline JSONL 的准确率。",
    )
    parser.add_argument("result_path", type=Path, help="baseline 输出 JSONL 路径。")
    return parser


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_no}: {exc}") from exc
    return rows


def dedupe_by_sample_id(rows):
    latest = {}
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or row.get("question_id") or "").strip()
        key = sample_id or f"__row_{index:06d}"
        latest[key] = row
    return list(latest.values())


def infer_correct(row):
    correct = row.get("correct")
    if isinstance(correct, bool):
        return correct
    answer = str(row.get("answer", "")).strip().upper()
    gold = str(row.get("gold", "")).strip().upper()
    if answer and gold:
        return answer == gold
    return None


def main():
    args = build_parser().parse_args()
    raw_rows = read_jsonl(args.result_path)
    rows = dedupe_by_sample_id(raw_rows)
    total = len(rows)
    failed = sum(1 for row in rows if row.get("error"))
    scored_rows = []
    for row in rows:
        correct = infer_correct(row)
        if correct is not None:
            scored_rows.append((row, correct))

    correct_count = sum(1 for _, correct in scored_rows if correct)
    scored_count = len(scored_rows)
    accuracy = correct_count / scored_count if scored_count else 0.0

    answer_counts = Counter(str(row.get("answer", "")).strip().upper() for row, _ in scored_rows)
    gold_counts = Counter(str(row.get("gold", "")).strip().upper() for row, _ in scored_rows)

    by_gold = defaultdict(lambda: [0, 0])
    for row, correct in scored_rows:
        gold = str(row.get("gold", "")).strip().upper()
        by_gold[gold][1] += 1
        if correct:
            by_gold[gold][0] += 1

    print(f"result_path={args.result_path}")
    print(f"raw_rows={len(raw_rows)}")
    print(f"total_rows={total}")
    print(f"failed_rows={failed}")
    print(f"scored_rows={scored_count}")
    print(f"correct={correct_count}")
    print(f"accuracy={accuracy:.4f}")
    print("answer_distribution=" + dict_to_text(answer_counts))
    print("gold_distribution=" + dict_to_text(gold_counts))
    print("per_gold_accuracy:")
    for gold in sorted(by_gold):
        hit, count = by_gold[gold]
        acc = hit / count if count else 0.0
        print(f"  {gold}: {hit}/{count} = {acc:.4f}")


def dict_to_text(counter):
    if not counter:
        return "{}"
    items = [f"{key}:{counter[key]}" for key in sorted(counter)]
    return "{" + ", ".join(items) + "}"


if __name__ == "__main__":
    main()
