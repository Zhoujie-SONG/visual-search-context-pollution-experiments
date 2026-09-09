#!/usr/bin/env python3
"""Strict and lenient paired scoring for the direct-answer diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import score_visualneedle_full300 as official_score
import run_visualneedle_minio3_smoke20 as baseline
from run_round6_direct_answer import ARMS, OUTPUT_ROOT, RESULT_FIELDS, SEED, read_jsonl, stage_dir, write_csv


PAIRS = (
    ("O_oracle_top2_direct", "F_full_direct"),
    ("O_oracle_top2_direct", "R_recent_top2_direct"),
    ("O_oracle_top2_direct", "X_random_top2_direct"),
)
MODES = ("strict", "lenient")
BOOTSTRAP_REPLICATES = 10_000


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def load_judges(output: Path) -> Dict[Tuple[str, str], bool]:
    path = output / "official_judge_results.csv"
    if not path.exists():
        return {}
    judges: Dict[Tuple[str, str], bool] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("judge_parse_success") == "True" and row.get("judge_result") in {"True", "False"}:
                judges[(row["sample_id"], row["condition"])] = row["judge_result"] == "True"
    return judges


def attach_scores(rows: Sequence[Dict[str, Any]], judges: Dict[Tuple[str, str], bool]) -> None:
    for row in rows:
        prediction = str(row.get("parsed_answer") or "").strip()
        gold = str(row.get("gt_answer") or "").strip()
        question = str(row.get("question") or "")
        exact = bool(prediction) and prediction == gold
        normalized = bool(prediction) and official_score.answers_match(prediction, gold, question=question)
        judge = judges.get((str(row["sample_id"]), str(row["condition"])))
        needs = bool(row.get("lenient_valid_answer")) and not normalized
        pending = needs and judge is None
        for mode in MODES:
            valid = bool(row.get(f"{mode}_valid_answer"))
            row[f"{mode}_correct_exact"] = valid and exact
            row[f"{mode}_correct_normalized"] = valid and normalized
            if not valid:
                final: Optional[bool] = False
            elif normalized:
                final = True
            elif judge is not None:
                final = judge
            else:
                final = None
            row[f"{mode}_final_correct"] = final
        row["official_judge_result"] = judge
        row["official_judge_pending"] = pending


def summary(rows: Sequence[Dict[str, Any]], arm: str, mode: str, label: str = "all") -> Dict[str, Any]:
    selected = [row for row in rows if row["condition"] == arm]
    valid = [row for row in selected if row[f"{mode}_valid_answer"]]
    scored = [row for row in selected if row[f"{mode}_final_correct"] is not None]
    answered_scored = [row for row in valid if row[f"{mode}_final_correct"] is not None]
    correct = [row for row in scored if row[f"{mode}_final_correct"] is True]
    correct_answered = [row for row in answered_scored if row[f"{mode}_final_correct"] is True]
    n = len(selected)
    return {
        "subgroup": label,
        "condition": arm,
        "mode": mode,
        "num_samples": n,
        "valid_answers": len(valid),
        "answer_rate": len(valid) / n if n else None,
        "final_correct": len(correct),
        "final_scored": len(scored),
        "final_accuracy": len(correct) / len(scored) if scored else None,
        "accuracy_among_answered": len(correct_answered) / len(answered_scored) if answered_scored else None,
        "grounding_present": sum(bool(row.get("grounding_present")) for row in selected),
        "protocol_violation_rate": sum(bool(row.get("protocol_violation")) for row in selected) / n if n else None,
        "judge_pending": sum(bool(row.get("official_judge_pending")) for row in selected),
        "oom": sum(bool(row.get("oom")) for row in selected),
        "error": sum(bool(row.get("error")) for row in selected),
        "mean_input_tokens": safe_mean(float(row.get("input_tokens") or 0) for row in selected),
        "mean_visual_tokens": safe_mean(float(row.get("visual_tokens") or 0) for row in selected),
        "mean_output_tokens": safe_mean(float(row.get("output_tokens") or 0) for row in selected),
        "mean_runtime_seconds": safe_mean(float(row.get("runtime_seconds") or 0) for row in selected),
    }


def bootstrap(differences: Sequence[float], seed: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not differences:
        return None, None, None
    rng = random.Random(seed)
    draws = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(BOOTSTRAP_REPLICATES)]
    return statistics.mean(differences), percentile(draws, 0.025), percentile(draws, 0.975)


def mcnemar(first: Sequence[int], second: Sequence[int]) -> Dict[str, Any]:
    first_only = sum(a == 1 and b == 0 for a, b in zip(first, second))
    second_only = sum(a == 0 and b == 1 for a, b in zip(first, second))
    discordant = first_only + second_only
    if not discordant:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1)) / (2 ** discordant)
        p = min(1.0, 2 * tail)
    return {"first_only": first_only, "second_only": second_only, "discordant": discordant, "p_value": p}


def paired_outputs(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["sample_id"])][str(row["condition"])] = row
    wide: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    for sample_id, arms in sorted(grouped.items()):
        anchor = next(iter(arms.values()))
        item: Dict[str, Any] = {"sample_id": sample_id, "category": anchor["category"], "max_prefix_gt_coverage": anchor["max_prefix_gt_coverage"]}
        for arm in ARMS:
            row = arms[arm]
            for mode in MODES:
                item[f"{arm}__{mode}_valid_answer"] = row[f"{mode}_valid_answer"]
                item[f"{arm}__{mode}_final_correct"] = row[f"{mode}_final_correct"]
            item[f"{arm}__protocol_violation"] = row["protocol_violation"]
            item[f"{arm}__parsed_answer"] = row["parsed_answer"]
        wide.append(item)
        for first, second in PAIRS:
            for mode in MODES:
                a, b = arms[first], arms[second]
                transitions.append({
                    "sample_id": sample_id,
                    "category": anchor["category"],
                    "mode": mode,
                    "first_condition": first,
                    "second_condition": second,
                    "second_wrong_or_no_answer_to_first_correct": a[f"{mode}_final_correct"] is True and b[f"{mode}_final_correct"] is False,
                    "second_correct_to_first_wrong": b[f"{mode}_final_correct"] is True and a[f"{mode}_final_correct"] is False,
                    "second_protocol_violation_to_first_valid_answer": bool(b["protocol_violation"]) and bool(a[f"{mode}_valid_answer"]),
                    "second_no_answer_to_first_answer": bool(a[f"{mode}_valid_answer"]) and not bool(b[f"{mode}_valid_answer"]),
                })

    bootstrap_rows: List[Dict[str, Any]] = []
    mcnemar_rows: List[Dict[str, Any]] = []
    for pair_index, (first, second) in enumerate(PAIRS):
        paired = [arms for arms in grouped.values() if first in arms and second in arms]
        for mode in MODES:
            for metric in ("valid_answer", "final_correct"):
                field = f"{mode}_{metric}"
                values = [
                    (int(bool(arms[first][field])), int(bool(arms[second][field])))
                    for arms in paired
                    if arms[first][field] is not None and arms[second][field] is not None
                ]
                mean, low, high = bootstrap([float(a - b) for a, b in values], SEED + pair_index * 100 + len(bootstrap_rows))
                bootstrap_rows.append({
                    "first_condition": first, "second_condition": second, "mode": mode,
                    "metric": metric, "num_pairs": len(values), "mean_difference_first_minus_second": mean,
                    "ci95_low": low, "ci95_high": high, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                })
                mcnemar_rows.append({
                    "first_condition": first, "second_condition": second, "mode": mode,
                    "metric": metric, "num_pairs": len(values),
                    **mcnemar([a for a, _ in values], [b for _, b in values]),
                })
    return wide, transitions, bootstrap_rows, mcnemar_rows


def format_pct(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.2%}"


def write_report(output: Path, summaries: Sequence[Dict[str, Any]], bootstrap_rows: Sequence[Dict[str, Any]], mcnemar_rows: Sequence[Dict[str, Any]], transitions: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> None:
    strict = {row["condition"]: row for row in summaries if row["mode"] == "strict"}
    lenient = {row["condition"]: row for row in summaries if row["mode"] == "lenient"}
    table = [
        "| Arm | Strict answer | Lenient answer | Strict final | Lenient final | Strict acc/answered | Lenient acc/answered | Grounding violation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        table.append(
            f"| {arm} | {format_pct(strict[arm]['answer_rate'])} | {format_pct(lenient[arm]['answer_rate'])} | "
            f"{format_pct(strict[arm]['final_accuracy'])} | {format_pct(lenient[arm]['final_accuracy'])} | "
            f"{format_pct(strict[arm]['accuracy_among_answered'])} | {format_pct(lenient[arm]['accuracy_among_answered'])} | "
            f"{format_pct(strict[arm]['protocol_violation_rate'])} |"
        )
    pair_lines = []
    for first, second in PAIRS:
        for mode in MODES:
            boot = next(row for row in bootstrap_rows if row["first_condition"] == first and row["second_condition"] == second and row["mode"] == mode and row["metric"] == "final_correct")
            test = next(row for row in mcnemar_rows if row["first_condition"] == first and row["second_condition"] == second and row["mode"] == mode and row["metric"] == "final_correct")
            rescue = sum(row["first_condition"] == first and row["second_condition"] == second and row["mode"] == mode and row["second_wrong_or_no_answer_to_first_correct"] for row in transitions)
            pair_lines.append(
                f"- {first} - {second}, {mode}: delta={format_pct(boot['mean_difference_first_minus_second'])}, "
                f"95% CI [{format_pct(boot['ci95_low'])}, {format_pct(boot['ci95_high'])}], "
                f"McNemar p={float(test['p_value']):.6g}, accuracy rescue={rescue}."
            )
    required = sum(bool(row.get("official_judge_pending")) or (row.get("official_judge_result") is not None) for row in rows)
    pending = sum(bool(row.get("official_judge_pending")) for row in rows)
    report = [
        "# Round-6 Direct-Answer Memory Diagnostic", "",
        f"- Primary N: {len({row['sample_id'] for row in rows})}",
        f"- Generations: {len(rows)}", "",
        "## Main Results", "", *table, "", "## Paired Comparisons", "", *pair_lines, "",
        "## Judge", "",
        f"- Required or completed: {required}",
        f"- Pending: {pending}",
        "- Model: `gemini-3.8-flash`", "",
        "## Interpretation", "",
        "Final interpretation is valid only when judge pending is zero. Use strict results as primary and lenient results as a diagnostic.",
    ]
    (output / "direct_answer_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sanity", "full"), default="full")
    args = parser.parse_args()
    output = stage_dir(args.stage)
    rows = read_jsonl(output / "merged_results.jsonl")
    judges = load_judges(output)
    attach_scores(rows, judges)
    baseline_rows = [summary(rows, arm, mode) for arm in ARMS for mode in MODES]
    write_csv(output / "strict_summary.csv", [row for row in baseline_rows if row["mode"] == "strict"], list(baseline_rows[0].keys()))
    write_csv(output / "lenient_summary.csv", [row for row in baseline_rows if row["mode"] == "lenient"], list(baseline_rows[0].keys()))

    strong_rows = []
    for label, threshold in (("max_prefix_gt_coverage_ge_0.50", 0.50), ("max_prefix_gt_coverage_ge_0.999999", 0.999999)):
        subset = [row for row in rows if float(row.get("max_prefix_gt_coverage") or 0) >= threshold]
        strong_rows.extend(summary(subset, arm, mode, label) for arm in ARMS for mode in MODES)
    write_csv(output / "strong_evidence_summary.csv", strong_rows, list(strong_rows[0].keys()))

    category_rows = []
    for category in sorted({str(row["category"]) for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        category_rows.extend(summary(subset, arm, mode, category) for arm in ARMS for mode in MODES)
    write_csv(output / "category_summary.csv", category_rows, list(category_rows[0].keys()))

    wide, transitions, bootstrap_rows, mcnemar_rows = paired_outputs(rows)
    write_csv(output / "paired_results.csv", wide, list(wide[0].keys()))
    write_csv(output / "sample_transitions.csv", transitions, list(transitions[0].keys()))
    write_csv(output / "bootstrap_pairwise.csv", bootstrap_rows, list(bootstrap_rows[0].keys()))
    write_csv(output / "mcnemar_pairwise.csv", mcnemar_rows, list(mcnemar_rows[0].keys()))
    baseline.write_jsonl(output / "merged_results_scored.jsonl", rows)
    write_csv(output / "merged_results.csv", rows, RESULT_FIELDS)
    write_report(output, baseline_rows, bootstrap_rows, mcnemar_rows, transitions, rows)
    print(json.dumps({
        "stage": args.stage,
        "records": len(rows),
        "judge_required_or_completed": sum(bool(row.get("official_judge_pending")) or row.get("official_judge_result") is not None for row in rows),
        "judge_pending": sum(bool(row.get("official_judge_pending")) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
