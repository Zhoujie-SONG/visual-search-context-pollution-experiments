#!/usr/bin/env python3
"""Offline scoring and paired analysis for round-6 visual-memory surgery."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import score_visualneedle_full300 as official_score
from run_round6_memory_surgery import ARMS, EPISODE_FIELDS, OUTPUT_ROOT, SEED, stage_directory


PAIRWISE = (
    ("B_oracle_top2", "A_full"),
    ("B_oracle_top2", "C_recent_top2"),
    ("B_oracle_top2", "E_random_top2"),
    ("B_oracle_top2", "D_force_answer_r6"),
    ("C_recent_top2", "E_random_top2"),
)
BINARY_METRICS = ("answer", "correct")
CONTINUOUS_METRICS = ("num_additional_crops", "final_round", "runtime_seconds")
BOOTSTRAP_REPLICATES = 10_000
APPROXIMATELY_EQUAL_TOLERANCE = 0.03


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def load_judges(stage_dir: Path) -> Dict[Tuple[str, str], bool]:
    path = stage_dir / "official_judge_results.csv"
    if not path.exists():
        return {}
    results = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("judge_parse_success") == "True" and row.get("judge_result") in {"True", "False"}:
                results[(row["sample_id"], row["condition"])] = row["judge_result"] == "True"
    return results


def attach_scores(episodes: Sequence[Dict[str, Any]], judges: Dict[Tuple[str, str], bool]) -> None:
    for episode in episodes:
        prediction = str(episode.get("final_answer") or "").strip()
        gold = str(episode.get("gt_answer") or "").strip()
        question = str(episode.get("question") or "")
        exact = bool(prediction) and prediction == gold
        normalized = official_score.answers_match(prediction or None, gold, question=question)
        judge = judges.get((str(episode["sample_id"]), str(episode["condition"])))
        pending = bool(prediction) and not normalized and judge is None
        final_correct: Optional[bool]
        if normalized:
            final_correct = True
        elif not prediction:
            final_correct = False
        elif judge is not None:
            final_correct = judge
        else:
            final_correct = None
        episode.update({
            "raw_exact_match": exact,
            "official_normalized_match": normalized,
            "post_intervention_correct_normalized": normalized,
            "official_judge_result": judge,
            "official_judge_pending": pending,
            "official_final_correct": final_correct,
            "post_intervention_final_correct": final_correct,
        })


def summary_row(cohort: str, arm: str, episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    answered = [episode for episode in episodes if bool(episode.get("post_intervention_answer_present"))]
    scored = [episode for episode in episodes if episode.get("official_final_correct") is not None]
    correct = [episode for episode in scored if episode["official_final_correct"] is True]
    answered_scored = [episode for episode in answered if episode.get("official_final_correct") is not None]
    answered_correct = [episode for episode in answered_scored if episode["official_final_correct"] is True]
    natural = [episode for episode in episodes if bool(episode.get("natural_stop"))]
    additional = [float(episode.get("num_additional_crops") or 0) for episode in episodes]
    rounds = [float(episode.get("final_round") or 0) for episode in episodes]
    visual = [float(value) for value in (episode.get("visual_tokens_first_post_intervention") for episode in episodes) if value is not None]
    return {
        "cohort": cohort,
        "condition": arm,
        "num_samples": total,
        "answered": len(answered),
        "answer_rate": len(answered) / total if total else None,
        "final_correct": len(correct),
        "final_scored": len(scored),
        "judge_pending": total - len(scored),
        "final_accuracy": len(correct) / len(scored) if scored else None,
        "accuracy_among_answered": len(answered_correct) / len(answered_scored) if answered_scored else None,
        "natural_stops": len(natural),
        "natural_termination_rate": len(natural) / total if total else None,
        "mean_additional_crops": safe_mean(additional),
        "median_additional_crops": statistics.median(additional) if additional else None,
        "mean_stop_round": safe_mean(rounds),
        "max_round_rate": sum(bool(ep.get("reached_max_rounds")) for ep in episodes) / total if total else None,
        "max_image_rate": sum(bool(ep.get("reached_max_images")) for ep in episodes) / total if total else None,
        "mean_first_post_visual_tokens": safe_mean(visual),
        "mean_runtime_seconds": safe_mean(float(ep.get("runtime_seconds") or 0) for ep in episodes),
        "oom": sum(bool(ep.get("oom")) for ep in episodes),
        "invalid": sum(bool(ep.get("invalid")) for ep in episodes),
    }


def binary_value(episode: Dict[str, Any], metric: str) -> Optional[int]:
    if metric == "answer":
        return int(bool(episode.get("post_intervention_answer_present")))
    value = episode.get("official_final_correct")
    return None if value is None else int(bool(value))


def bootstrap_ci(differences: Sequence[float], seed: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not differences:
        return None, None, None
    observed = statistics.mean(differences)
    rng = random.Random(seed)
    draws = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(BOOTSTRAP_REPLICATES)]
    return observed, percentile(draws, 0.025), percentile(draws, 0.975)


def mcnemar_exact(first: Sequence[int], second: Sequence[int]) -> Dict[str, Any]:
    first_only = sum(a == 1 and b == 0 for a, b in zip(first, second))
    second_only = sum(a == 0 and b == 1 for a, b in zip(first, second))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(first_only, second_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {"first_only": first_only, "second_only": second_only, "discordant": discordant, "p_value": p_value}


def paired_analysis(episodes: Sequence[Dict[str, Any]], cohort: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected = [episode for episode in episodes if episode["cohort"] == cohort]
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for episode in selected:
        by_sample[str(episode["sample_id"])][str(episode["condition"])] = episode
    bootstrap_rows: List[Dict[str, Any]] = []
    mcnemar_rows: List[Dict[str, Any]] = []
    for pair_index, (first_arm, second_arm) in enumerate(PAIRWISE):
        paired = [arms for arms in by_sample.values() if first_arm in arms and second_arm in arms]
        for metric in BINARY_METRICS:
            values = []
            for arms in paired:
                first = binary_value(arms[first_arm], metric)
                second = binary_value(arms[second_arm], metric)
                if first is not None and second is not None:
                    values.append((first, second))
            differences = [float(first - second) for first, second in values]
            mean, low, high = bootstrap_ci(differences, SEED + pair_index * 100 + len(bootstrap_rows))
            bootstrap_rows.append({
                "cohort": cohort, "first_condition": first_arm, "second_condition": second_arm,
                "metric": metric, "num_pairs": len(values), "mean_difference_first_minus_second": mean,
                "ci95_low": low, "ci95_high": high, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            })
            test = mcnemar_exact([value[0] for value in values], [value[1] for value in values])
            mcnemar_rows.append({
                "cohort": cohort, "first_condition": first_arm, "second_condition": second_arm,
                "metric": metric, "num_pairs": len(values), **test,
            })
        for metric in CONTINUOUS_METRICS:
            differences = [
                float(arms[first_arm].get(metric) or 0) - float(arms[second_arm].get(metric) or 0)
                for arms in paired
            ]
            mean, low, high = bootstrap_ci(differences, SEED + pair_index * 1000 + len(bootstrap_rows))
            bootstrap_rows.append({
                "cohort": cohort, "first_condition": first_arm, "second_condition": second_arm,
                "metric": metric, "num_pairs": len(differences), "mean_difference_first_minus_second": mean,
                "median_difference_first_minus_second": statistics.median(differences) if differences else None,
                "ci95_low": low, "ci95_high": high, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            })
    return bootstrap_rows, mcnemar_rows


def cohort_map() -> Dict[str, str]:
    path = OUTPUT_ROOT / "cohort_selection.csv"
    with path.open(encoding="utf-8") as handle:
        return {row["sample_id"]: row["cohort"] for row in csv.DictReader(handle)}


def paired_wide_rows(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        grouped[str(episode["sample_id"])][str(episode["condition"])] = episode
    rows = []
    for sample_id, arms in sorted(grouped.items()):
        first = next(iter(arms.values()))
        row: Dict[str, Any] = {"sample_id": sample_id, "cohort": first["cohort"], "category": first["category"]}
        for arm in ARMS:
            episode = arms.get(arm)
            for metric in ("post_intervention_answer_present", "official_final_correct", "natural_stop", "num_additional_crops", "final_round", "runtime_seconds", "oom", "invalid"):
                row[f"{arm}__{metric}"] = "" if episode is None or episode.get(metric) is None else episode.get(metric)
        rows.append(row)
    return rows


def subgroup_rows(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    primary = [episode for episode in episodes if episode["cohort"] == "primary_prefix_hit"]
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for episode in primary:
        by_sample[str(episode["sample_id"])][str(episode["condition"])] = episode
    rows = []
    for sample_id, arms in sorted(by_sample.items()):
        anchor = arms.get("A_full") or next(iter(arms.values()))
        prefix = anchor.get("prefix_crops", [])
        coverages = [float(crop.get("gt_coverage") or 0) for crop in prefix]
        hit = [index + 1 for index, value in enumerate(coverages) if value >= 0.10]
        oracle = arms.get("B_oracle_top2", {})
        full = arms.get("A_full", {})
        rows.append({
            "sample_id": sample_id,
            "category": anchor.get("category"),
            "first_gt_hit_crop_index": hit[0] if hit else "",
            "max_prefix_gt_coverage": max(coverages, default=0),
            "irrelevant_prefix_crops": sum(value < 0.10 for value in coverages),
            "repeated_prefix_crops": sum(bool(crop.get("high_overlap_previous")) for crop in prefix),
            "oracle_answer_rescue": bool(oracle.get("post_intervention_answer_present")) and not bool(full.get("post_intervention_answer_present")),
            "oracle_accuracy_rescue": oracle.get("official_final_correct") is True and full.get("official_final_correct") is False,
            "oracle_minus_full_additional_crops": float(oracle.get("num_additional_crops") or 0) - float(full.get("num_additional_crops") or 0),
        })
    return rows


def subgroup_summary_rows(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    primary = [episode for episode in episodes if episode["cohort"] == "primary_prefix_hit"]
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for episode in primary:
        by_sample[str(episode["sample_id"])].append(episode)
    labels: Dict[str, Dict[str, str]] = {}
    for sample_id, sample_episodes in by_sample.items():
        anchor = next((episode for episode in sample_episodes if episode["condition"] == "A_full"), sample_episodes[0])
        prefix = anchor.get("prefix_crops", [])
        coverages = [float(crop.get("gt_coverage") or 0) for crop in prefix]
        first_hit = min((index + 1 for index, value in enumerate(coverages) if value >= 0.10), default=6)
        maximum = max(coverages, default=0)
        irrelevant = sum(value < 0.10 for value in coverages)
        repeated = sum(bool(crop.get("high_overlap_previous")) for crop in prefix)
        labels[sample_id] = {
            "first_gt_hit": "early_1_2" if first_hit <= 2 else "middle_3_4" if first_hit <= 4 else "late_5_6",
            "max_gt_coverage": "10_49pct" if maximum < 0.50 else "50_99pct" if maximum < 1.0 else "100pct",
            "irrelevant_prefix_crops": "0_2" if irrelevant <= 2 else "3_4" if irrelevant <= 4 else "5",
            "repeated_prefix_crops": "none" if repeated == 0 else "one_or_more",
        }
    rows = []
    for dimension in ("first_gt_hit", "max_gt_coverage", "irrelevant_prefix_crops", "repeated_prefix_crops"):
        for value in sorted({label[dimension] for label in labels.values()}):
            sample_ids = {sample_id for sample_id, label in labels.items() if label[dimension] == value}
            for arm in ARMS:
                selected = [ep for ep in primary if ep["sample_id"] in sample_ids and ep["condition"] == arm]
                row = summary_row("primary_prefix_hit", arm, selected)
                row.update({"subgroup_dimension": dimension, "subgroup_value": value})
                rows.append(row)
    return rows


def decision_case(summaries: Sequence[Dict[str, Any]], pending: int) -> str:
    if pending:
        return "DEFERRED: official image-aware judge still has pending rows."
    rows = {row["condition"]: row for row in summaries if row["cohort"] == "primary_prefix_hit"}
    if set(ARMS) - set(rows):
        return "NOT_APPLICABLE: all five arms are required."
    accuracy = {arm: float(rows[arm]["final_accuracy"] or 0) for arm in ARMS}
    oracle, recent = accuracy["B_oracle_top2"], accuracy["C_recent_top2"]
    random_accuracy, full = accuracy["E_random_top2"], accuracy["A_full"]
    force = accuracy["D_force_answer_r6"]
    oracle_earlier = float(rows["B_oracle_top2"]["mean_stop_round"] or 0) < float(rows["A_full"]["mean_stop_round"] or 0)
    close = lambda first, second: abs(first - second) <= APPROXIMATELY_EQUAL_TOLERANCE
    if oracle > recent > full and oracle > force and oracle_earlier:
        return "BEST_CASE: selective evidence management improves accuracy and earlier natural termination beyond force-answer."
    if oracle > recent and close(random_accuracy, full):
        return "CASE_1: evidence selection matters (Oracle > Recent and Random approximately Full)."
    if close(oracle, recent) and recent > full:
        return "CASE_2: context burden matters; bounded/recent memory may be sufficient."
    if close(oracle, full) and force > full:
        return "CASE_3: stopping policy is the more likely bottleneck."
    if oracle > full and force >= oracle:
        return "CASE_4: memory pollution exists, but stopping remains the stronger bottleneck."
    return "MIXED_OR_INCONCLUSIVE: observed ordering does not match a preregistered case cleanly."


def render_report(stage: str, summaries: Sequence[Dict[str, Any]], pending: int, episodes: Sequence[Dict[str, Any]]) -> str:
    title = "Smoke Report" if stage == "smoke" else "Formal Report" if stage == "formal" else "Devcheck Report"
    lines = [f"# Round-6 Visual Memory Surgery {title}", ""]
    lines.append(f"- Episodes: {len(episodes)}")
    lines.append(f"- Unique samples: {len({ep['sample_id'] for ep in episodes})}")
    lines.append(f"- Judge pending: {pending}")
    lines.append("- Pending judge rows are excluded from final-accuracy denominators, never counted as wrong.")
    lines.extend(["", "## Primary Prefix-Hit", "", "| Arm | N | Answer rate | Final accuracy | Accuracy among answered | Natural stop | Mean additional crops |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in summaries:
        if row["cohort"] != "primary_prefix_hit":
            continue
        def pct(value: Any) -> str:
            return "NA" if value is None else f"{100 * float(value):.2f}%"
        lines.append(
            f"| {row['condition']} | {row['num_samples']} | {pct(row['answer_rate'])} | "
            f"{pct(row['final_accuracy'])} | {pct(row['accuracy_among_answered'])} | "
            f"{pct(row['natural_termination_rate'])} | {float(row['mean_additional_crops'] or 0):.2f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if pending:
        lines.append("Decision case is deferred until the official image-aware judge has no pending rows.")
    elif stage == "devcheck":
        lines.append("This stage checks reconstruction only; no causal interpretation is made.")
    else:
        lines.append(decision_case(summaries, pending))
        lines.append(f"Approximate-equality tolerance used by the descriptive decision rule: {APPROXIMATELY_EQUAL_TOLERANCE:.2f} absolute accuracy.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("devcheck", "smoke", "formal"), required=True)
    args = parser.parse_args()
    stage_dir = stage_directory(args.stage)
    episodes = read_jsonl(stage_dir / "merged_trajectories.jsonl")
    cohorts = cohort_map()
    for episode in episodes:
        episode["cohort"] = cohorts[str(episode["sample_id"])]
    attach_scores(episodes, load_judges(stage_dir))
    scored_fields = EPISODE_FIELDS + [
        "raw_exact_match", "official_normalized_match", "official_judge_result",
        "official_judge_pending", "official_final_correct", "post_intervention_final_correct",
    ]
    write_csv(stage_dir / "merged_episode_results.csv", [{
        field: json.dumps(ep.get(field), ensure_ascii=False)
        if isinstance(ep.get(field), (list, dict)) else ep.get(field, "")
        for field in scored_fields
    } for ep in episodes], scored_fields)
    baseline_path = stage_dir / "merged_trajectories_scored.jsonl"
    with baseline_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    summaries = [
        summary_row(cohort, arm, [ep for ep in episodes if ep["cohort"] == cohort and ep["condition"] == arm])
        for cohort in ("primary_prefix_hit", "secondary_prefix_no_hit")
        for arm in (["A_full"] if args.stage == "devcheck" else ARMS)
    ]
    write_csv(stage_dir / "primary_prefix_hit_summary.csv", [row for row in summaries if row["cohort"] == "primary_prefix_hit"])
    write_csv(stage_dir / "secondary_prefix_no_hit_summary.csv", [row for row in summaries if row["cohort"] == "secondary_prefix_no_hit"])
    category_rows = []
    for cohort in ("primary_prefix_hit", "secondary_prefix_no_hit"):
        for category in official_score.CATEGORIES:
            for arm in (["A_full"] if args.stage == "devcheck" else ARMS):
                row = summary_row(
                    cohort, arm,
                    [ep for ep in episodes if ep["cohort"] == cohort and ep["category"] == category and ep["condition"] == arm],
                )
                row["category"] = category
                category_rows.append(row)
    write_csv(stage_dir / "summary_by_category.csv", category_rows)
    write_csv(stage_dir / "paired_results.csv", paired_wide_rows(episodes))
    write_csv(stage_dir / "subgroup_prefix_hit.csv", subgroup_rows(episodes))
    write_csv(stage_dir / "subgroup_prefix_hit_summary.csv", subgroup_summary_rows(episodes))
    bootstrap_rows, mcnemar_rows = paired_analysis(episodes, "primary_prefix_hit")
    secondary_bootstrap, secondary_mcnemar = paired_analysis(episodes, "secondary_prefix_no_hit")
    write_csv(stage_dir / "bootstrap_pairwise.csv", bootstrap_rows + secondary_bootstrap)
    write_csv(stage_dir / "mcnemar_pairwise.csv", mcnemar_rows + secondary_mcnemar)

    runtime_rows = []
    for arm in (["A_full"] if args.stage == "devcheck" else ARMS):
        arm_episodes = [episode for episode in episodes if episode["condition"] == arm]
        runtime_rows.append({
            "condition": arm, "num_episodes": len(arm_episodes),
            "sum_runtime_seconds": sum(float(ep.get("runtime_seconds") or 0) for ep in arm_episodes),
            "mean_runtime_seconds": safe_mean(float(ep.get("runtime_seconds") or 0) for ep in arm_episodes),
            "median_runtime_seconds": statistics.median(float(ep.get("runtime_seconds") or 0) for ep in arm_episodes) if arm_episodes else None,
            "max_peak_gpu_memory_mb": max((float(ep.get("peak_gpu_memory_mb") or 0) for ep in arm_episodes), default=None),
        })
    write_csv(stage_dir / "runtime_summary.csv", runtime_rows)
    write_csv(stage_dir / "oom_log.csv", [
        {"sample_id": ep["sample_id"], "condition": ep["condition"], "error_message": ep.get("error_message", "")}
        for ep in episodes if bool(ep.get("oom"))
    ], ("sample_id", "condition", "error_message"))
    pending = sum(bool(ep.get("official_judge_pending")) for ep in episodes)
    report_name = "smoke_report.md" if args.stage == "smoke" else "round6_memory_surgery_report.md" if args.stage == "formal" else "devcheck_report.md"
    (stage_dir / report_name).write_text(render_report(args.stage, summaries, pending, episodes), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "episodes": len(episodes), "judge_pending": pending}, indent=2))


if __name__ == "__main__":
    main()
