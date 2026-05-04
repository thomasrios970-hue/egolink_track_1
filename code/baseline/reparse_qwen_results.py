"""
重新解析 Qwen baseline 已有 JSONL 的 raw_output。

脚本作用：
1. 不重新跑大模型，只读取已有 raw_output。
2. 使用当前更稳的解析规则重算 answer/prediction/confidence/evidence/correct。
3. 输出新的 JSONL，便于比较 prompt/解析修复后分数。

运行示例：
python code/baseline/reparse_qwen_results.py \
  data/processed/baseline_v1/qwen_omni/mcp_qwen_omni_baseline_v1.jsonl \
  --output data/processed/baseline_v1/qwen_omni/mcp_qwen_omni_baseline_v1_reparsed.jsonl
"""

import argparse
import json
from pathlib import Path

from qwen_omni_baseline_v1 import (
    LETTERS,
    parse_answer,
    parse_confidence,
    parse_evidence,
    prediction_text,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="用当前解析规则重新解析已有 Qwen JSONL raw_output。",
    )
    parser.add_argument("result_path", type=Path, help="已有 baseline JSONL。")
    parser.add_argument("--output", type=Path, default=None, help="输出 JSONL。")
    return parser


def default_output_path(path):
    path = Path(path)
    return path.with_name(f"{path.stem}_reparsed{path.suffix}")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def option_list(options):
    if isinstance(options, dict):
        return [
            str(options.get(letter, "")).strip()
            for letter in LETTERS
            if str(options.get(letter, "")).strip()
        ]
    if isinstance(options, list):
        return [str(item).strip() for item in options if str(item).strip()]
    return []


def main():
    args = build_parser().parse_args()
    output_path = args.output or default_output_path(args.result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in read_jsonl(args.result_path):
            total += 1
            if row.get("raw_output") and row.get("options"):
                options = option_list(row.get("options"))
                answer = parse_answer(row.get("raw_output", ""), len(options), options)
                if answer != row.get("answer"):
                    changed += 1
                row["answer"] = answer
                row["prediction"] = prediction_text(answer, options)
                row["confidence"] = parse_confidence(row.get("raw_output", ""))
                row["evidence"] = parse_evidence(row.get("raw_output", ""))
                gold = row.get("gold")
                row["correct"] = (answer == gold) if gold else None
                row.pop("error", None)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"input_path={args.result_path}")
    print(f"output_path={output_path}")
    print(f"rows={total}")
    print(f"changed_answers={changed}")


if __name__ == "__main__":
    main()
