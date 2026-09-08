import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from context_pollution_utils import LETTERS, assert_can_write, parse_answer_letter, write_csv


def as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def category_from_row(row: dict) -> str:
    sample_id = row.get("sample_id", "")
    if "/" in sample_id:
        return sample_id.split("/", 1)[0]
    image_path = row.get("image_path", "")
    parts = Path(image_path).parts
    return parts[-2] if len(parts) >= 2 else ""


def valid_letters_for_row(row: dict) -> str:
    try:
        options = json.loads(row.get("options_shuffled") or "[]")
        if isinstance(options, list) and options:
            return LETTERS[: len(options)]
    except json.JSONDecodeError:
        pass
    return "ABCD"


def condition_sort_key(condition: str, rows_by_condition: dict) -> tuple:
    order = {"original_only": -1, "gt_crop_only": 0}
    first_row = rows_by_condition[condition][0]
    return order.get(condition, 999), int(first_row.get("k_irrelevant") or 0), condition


def summarize_rows(rows, prefix: dict | None = None):
    prefix = prefix or {}
    by_condition = defaultdict(list)
    by_sample_condition = {}
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_sample_condition[(row["sample_id"], row["condition"])] = row

    baseline_condition = "gt_crop_only"
    baseline_accuracy = None
    if baseline_condition in by_condition:
        base_rows = by_condition[baseline_condition]
        baseline_accuracy = sum(as_bool(row["correct"]) for row in base_rows) / max(1, len(base_rows))

    summary_rows = []
    for condition in sorted(by_condition, key=lambda c: condition_sort_key(c, by_condition)):
        condition_rows = by_condition[condition]
        total = len(condition_rows)
        correct = sum(as_bool(row["correct"]) for row in condition_rows)
        failed = sum(1 for row in condition_rows if row.get("error") or not row.get("pred_answer_letter"))
        accuracy = correct / total if total else 0.0

        comparable = 0
        flips = 0
        if condition != baseline_condition:
            for row in condition_rows:
                base = by_sample_condition.get((row["sample_id"], baseline_condition))
                if not base:
                    continue
                if base.get("pred_answer_letter") and row.get("pred_answer_letter"):
                    comparable += 1
                    flips += int(base["pred_answer_letter"] != row["pred_answer_letter"])

        times = [float(row["inference_time_sec"]) for row in condition_rows if row.get("inference_time_sec")]
        summary_row = {
            **prefix,
            "condition": condition,
            "k_irrelevant": condition_rows[0].get("k_irrelevant", ""),
            "num_samples": total,
            "accuracy": f"{accuracy:.6f}",
            "accuracy_drop_vs_gt_crop_only": (
                "" if baseline_accuracy is None else f"{baseline_accuracy - accuracy:.6f}"
            ),
            "answer_flip_rate_vs_gt_crop_only": (
                "" if condition == baseline_condition or comparable == 0 else f"{flips / comparable:.6f}"
            ),
            "failed_or_incomplete_generations": failed,
            "avg_inference_time_sec": f"{sum(times) / len(times):.4f}" if times else "",
        }
        summary_rows.append(summary_row)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score context pollution results.")
    parser.add_argument("--input_csv", default="outputs/results_context_pollution.csv")
    parser.add_argument("--summary_csv", default="outputs/summary_context_pollution.csv")
    parser.add_argument("--by_category_csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    summary_csv = Path(args.summary_csv)
    by_category_csv = (
        Path(args.by_category_csv)
        if args.by_category_csv
        else summary_csv.with_name(f"{summary_csv.stem}_by_category{summary_csv.suffix}")
    )
    assert_can_write(summary_csv, args.overwrite)
    assert_can_write(by_category_csv, args.overwrite)

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if not row.get("pred_answer_letter"):
            row["pred_answer_letter"] = parse_answer_letter(
                row.get("model_raw_output", ""), valid_letters_for_row(row)
            )
        row["correct"] = str(row.get("pred_answer_letter") == row.get("gt_answer_letter"))
        row["category"] = category_from_row(row)

    summary_rows = summarize_rows(rows)

    fieldnames = [
        "condition",
        "k_irrelevant",
        "num_samples",
        "accuracy",
        "accuracy_drop_vs_gt_crop_only",
        "answer_flip_rate_vs_gt_crop_only",
        "failed_or_incomplete_generations",
        "avg_inference_time_sec",
    ]
    write_csv(summary_csv, summary_rows, fieldnames)

    by_category_rows = []
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    for category in sorted(by_category):
        for row in summarize_rows(by_category[category], {"category": category}):
            by_category_rows.append(
                {
                    "category": row["category"],
                    "condition": row["condition"],
                    "k_irrelevant": row["k_irrelevant"],
                    "num_samples": row["num_samples"],
                    "accuracy": row["accuracy"],
                    "drop_vs_gt_crop_only": row["accuracy_drop_vs_gt_crop_only"],
                    "flip_rate_vs_gt_crop_only": row["answer_flip_rate_vs_gt_crop_only"],
                    "failed": row["failed_or_incomplete_generations"],
                }
            )
    write_csv(
        by_category_csv,
        by_category_rows,
        [
            "category",
            "condition",
            "k_irrelevant",
            "num_samples",
            "accuracy",
            "drop_vs_gt_crop_only",
            "flip_rate_vs_gt_crop_only",
            "failed",
        ],
    )

    print(f"wrote summary: {summary_csv}")
    print(f"wrote category summary: {by_category_csv}")
    for row in summary_rows:
        print(
            f"{row['condition']:>22}  k={row['k_irrelevant']}  "
            f"acc={row['accuracy']}  failed={row['failed_or_incomplete_generations']}"
        )


if __name__ == "__main__":
    main()
