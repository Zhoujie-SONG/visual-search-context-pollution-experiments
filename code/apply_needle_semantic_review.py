#!/usr/bin/env python3
"""Apply documented human-review corrections to VisualNeedle semantic scores."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


MANUAL_OVERRIDES = {
    "TPROMPT3cc2cc870796": {
        "verdict": "CORRECT",
        "extracted_answer": "Carrot",
        "reason": "问题询问图案，候选明确识别为 carrot；修饰语不改变实体答案。",
    },
    "TPROMPT5013d9f0a3b0": {
        "verdict": "CORRECT",
        "extracted_answer": "A black handbag",
        "reason": "handbag 是 bag 的更具体表达，与参考答案语义一致。",
    },
    "TPROMPTaf2003853447": {
        "verdict": "CORRECT",
        "extracted_answer": "A strawberry-shaped decoration",
        "reason": "问题询问装饰物所呈现的实体，strawberry-shaped 与 Strawberry 语义一致。",
    },
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.semantic_dir / "semantic_rescore.csv"
    rows = read_csv(source_path)
    reviewed = []
    seen_overrides = set()
    for row in rows:
        result = dict(row)
        result["auto_verdict"] = row["verdict"]
        result["auto_reason"] = row["reason"]
        if row["evaluation_source"] == "agent":
            result["review_source"] = "manual_full_review"
            override = MANUAL_OVERRIDES.get(row["sample_id"])
            if row["original_status"] == "invalid":
                result.update(
                    {
                        "verdict": "NO_ANSWER",
                        "semantic_correct": "False",
                        "extracted_answer": "",
                        "reason": "输出停在未完成的搜索/grounding，没有明确最终答案。",
                        "review_source": "manual_no_answer",
                    }
                )
            elif override:
                result.update(override)
                result["semantic_correct"] = "True"
                result["review_source"] = "manual_override"
                seen_overrides.add(row["sample_id"])
        else:
            result["review_source"] = "automatic_judge_only"
        reviewed.append(result)
    if seen_overrides != set(MANUAL_OVERRIDES):
        raise RuntimeError(f"missing override rows: {set(MANUAL_OVERRIDES) - seen_overrides}")

    fields = list(rows[0]) + ["auto_verdict", "auto_reason", "review_source"]
    write_csv(args.semantic_dir / "semantic_rescore_reviewed.csv", reviewed, fields)

    summary = []
    for source in ["agent", "plain_vqa"]:
        source_rows = [row for row in reviewed if row["evaluation_source"] == source]
        categories = sorted({row["category"] for row in source_rows})
        for category in ["ALL", *categories]:
            group = source_rows if category == "ALL" else [row for row in source_rows if row["category"] == category]
            counts = Counter(row["verdict"] for row in group)
            summary.append(
                {
                    "evaluation_source": source,
                    "category": category,
                    "num_samples": len(group),
                    "semantic_correct": counts["CORRECT"],
                    "semantic_accuracy": counts["CORRECT"] / len(group) if group else 0.0,
                    "incorrect": counts["INCORRECT"],
                    "no_answer": counts["NO_ANSWER"],
                    "manual_overrides": sum(row["review_source"] == "manual_override" for row in group),
                    "review_scope": "all agent rows" if source == "agent" else "automatic judge only",
                }
            )
    write_csv(args.semantic_dir / "semantic_summary_reviewed.csv", summary, list(summary[0]))

    agent = next(row for row in summary if row["evaluation_source"] == "agent" and row["category"] == "ALL")
    report = [
        "# VisualNeedle 100-sample Reviewed Semantic Accuracy",
        "",
        f"- Agent semantic accuracy: {agent['semantic_accuracy']:.2%} "
        f"({agent['semantic_correct']}/{agent['num_samples']})",
        f"- Manual corrections to automatic judge: {agent['manual_overrides']}",
        f"- No-answer trajectories: {agent['no_answer']}",
        "- All 100 agent candidate outputs were manually reviewed.",
        "- Missing <answer> tags were ignored when an output still contained one clear final answer.",
        "- Outputs ending in an unresolved grounding/search action were not counted as answers.",
        "",
        "## Manual overrides",
        "",
    ]
    for sample_id, override in MANUAL_OVERRIDES.items():
        report.append(f"- {sample_id}: {override['reason']}")
    (args.semantic_dir / "semantic_review_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"agent semantic accuracy: {agent['semantic_correct']}/{agent['num_samples']} = {agent['semantic_accuracy']:.4f}")


if __name__ == "__main__":
    main()
