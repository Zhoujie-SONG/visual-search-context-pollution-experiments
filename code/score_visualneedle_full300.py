#!/usr/bin/env python3
"""Score and summarize the VisualNeedle-300 Mini-o3 baseline offline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


OFFICIAL_REPO = Path("/home/songzhoujie/cvpr27/VisualNeedle-official")
MATCHING_PATH = OFFICIAL_REPO / "visualneedle_eval" / "matching.py"
spec = importlib.util.spec_from_file_location("visualneedle_official_matching", MATCHING_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load official matching implementation: {MATCHING_PATH}")
official_matching = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official_matching)
answers_match = official_matching.answers_match


CATEGORIES = [
    "Color Recognition",
    "Entity Recognition",
    "OCR Recognition",
    "Occluded Object Recognition",
    "Spatial Relationship",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def score_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    pred = str(episode.get("final_answer") or "").strip()
    gold = str(episode.get("gt_answer") or "").strip()
    question = str(episode.get("question") or "")
    raw_exact = bool(pred) and pred == gold
    official_normalized = answers_match(pred or None, gold, question=question)
    # Official VisualNeedle invokes a configured VLM judge only for unmatched rows.
    # No judge result is synthesized here: unmatched rows remain explicitly pending.
    judge_result = episode.get("official_judge_result")
    judge_available = isinstance(judge_result, bool)
    official_correct = official_normalized or (bool(judge_result) if judge_available else False)
    pending = bool(pred) and not official_normalized and not judge_available
    return {
        "raw_exact_match": raw_exact,
        "official_normalized_match": official_normalized,
        "official_judge_used": not official_normalized,
        "official_judge_result": judge_result if judge_available else "",
        "official_judge_pending": pending,
        "official_final_correct": official_correct,
        "official_final_is_lower_bound": pending,
    }


def metric_row(label: str, episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    exact = sum(bool(ep["scores"]["raw_exact_match"]) for ep in episodes)
    normalized = sum(bool(ep["scores"]["official_normalized_match"]) for ep in episodes)
    final = sum(bool(ep["scores"]["official_final_correct"]) for ep in episodes)
    pending = sum(bool(ep["scores"]["official_judge_pending"]) for ep in episodes)
    return {
        "category": label,
        "num_samples": total,
        "exact_correct": exact,
        "exact_accuracy": rate(exact, total),
        "normalized_correct": normalized,
        "normalized_accuracy": rate(normalized, total),
        "official_final_correct": final,
        "official_final_accuracy": rate(final, total),
        "official_judge_pending": pending,
        "official_final_complete": pending == 0,
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def behavior_row(label: str, episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    crops = [crop for ep in episodes for crop in ep.get("crops", [])]
    crop_counts = [len(ep.get("crops", [])) for ep in episodes]
    coverage_hit = [ep for ep in episodes if any(float(c.get("gt_coverage", 0)) >= 0.10 for c in ep.get("crops", []))]
    center_hit = [ep for ep in episodes if any(bool(c.get("contains_gt_center")) for c in ep.get("crops", []))]
    high_overlap = [c for c in crops if bool(c.get("high_overlap_previous"))]
    irrelevant = [c for c in crops if float(c.get("gt_coverage", 0)) < 0.10]
    correct = [ep for ep in episodes if bool(ep["scores"]["official_final_correct"])]
    hit_wrong = [ep for ep in coverage_hit if not bool(ep["scores"]["official_final_correct"])]
    correct_no_hit = [ep for ep in correct if ep not in coverage_hit]
    return {
        "category": label,
        "num_episodes": len(episodes),
        "total_crops": len(crops),
        "mean_crops_per_episode": safe_mean(crop_counts),
        "median_crops_per_episode": statistics.median(crop_counts) if crop_counts else 0.0,
        "gt_center_hit_episodes": len(center_hit),
        "gt_center_hit_rate": rate(len(center_hit), len(episodes)),
        "gt_coverage_10_hit_episodes": len(coverage_hit),
        "gt_coverage_10_hit_rate": rate(len(coverage_hit), len(episodes)),
        "geometrically_irrelevant_crops": len(irrelevant),
        "geometrically_irrelevant_crop_ratio": rate(len(irrelevant), len(crops)),
        "repeated_high_overlap_crops": len(high_overlap),
        "repeated_high_overlap_crop_ratio": rate(len(high_overlap), len(crops)),
        "hit_gt_but_wrong_episodes": len(hit_wrong),
        "hit_gt_but_wrong_rate_among_hits": rate(len(hit_wrong), len(coverage_hit)),
        "correct_without_gt_hit_episodes": len(correct_no_hit),
        "reached_max_rounds": sum(bool(ep.get("reached_turn_cap")) for ep in episodes),
        "reached_max_images": sum(bool(ep.get("reached_image_cap")) for ep in episodes),
        "invalid": sum(ep.get("status") == "invalid" for ep in episodes),
        "invalid_rate": rate(sum(ep.get("status") == "invalid" for ep in episodes), len(episodes)),
        "oom": sum(ep.get("status") == "oom" for ep in episodes),
        "oom_rate": rate(sum(ep.get("status") == "oom" for ep in episodes), len(episodes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    episodes = read_jsonl(out / "trajectories.jsonl")
    if len(episodes) != 300 or len({ep["sample_id"] for ep in episodes}) != 300:
        raise ValueError("Refusing to score: trajectories.jsonl is not 300 unique episodes")

    judge_path = out / "official_judge_results.csv"
    judge_results: Dict[str, bool] = {}
    judge_models = set()
    if judge_path.exists():
        for row in csv.DictReader(judge_path.open(encoding="utf-8")):
            if row.get("model"):
                judge_models.add(row["model"])
            if row.get("judge_parse_success") == "True" and row.get("judge_result") in {"True", "False"}:
                judge_results[row["sample_id"]] = row["judge_result"] == "True"
    for episode in episodes:
        if episode["sample_id"] in judge_results:
            episode["official_judge_result"] = judge_results[episode["sample_id"]]
        episode["scores"] = score_episode(episode)

    overall = metric_row("OVERALL", episodes)
    by_category = [metric_row(category, [ep for ep in episodes if ep["category"] == category]) for category in CATEGORIES]
    write_csv(out / "accuracy_overall.csv", [overall])
    write_csv(out / "accuracy_by_category.csv", by_category)

    behavior_overall = behavior_row("OVERALL", episodes)
    behavior_categories = [behavior_row(category, [ep for ep in episodes if ep["category"] == category]) for category in CATEGORIES]
    write_csv(out / "search_behavior_by_category.csv", behavior_categories)

    crop_distribution = Counter(len(ep.get("crops", [])) for ep in episodes)
    first_hit_distribution = Counter(
        min(c["turn_index"] for c in ep.get("crops", []) if float(c.get("gt_coverage", 0)) >= 0.10)
        for ep in episodes
        if any(float(c.get("gt_coverage", 0)) >= 0.10 for c in ep.get("crops", []))
    )
    behavior_json = {
        **behavior_overall,
        "crop_count_distribution": dict(sorted(crop_distribution.items())),
        "first_gt_coverage_10_hit_turn_distribution": dict(sorted(first_hit_distribution.items())),
        "gt_hit_definition": "at least one crop with GT coverage >= 10%",
        "geometrically_irrelevant_definition": "crop GT coverage < 10%",
        "repeated_high_overlap_definition": "IoU with any previous crop >= 0.7",
    }
    (out / "search_behavior_summary.json").write_text(json.dumps(behavior_json, ensure_ascii=False, indent=2), encoding="utf-8")

    correct_ids = [ep["sample_id"] for ep in episodes if ep["scores"]["official_final_correct"]]
    (out / "baseline_correct_sample_ids.txt").write_text("\n".join(correct_ids) + ("\n" if correct_ids else ""), encoding="utf-8")

    runtimes = [float(ep.get("runtime_seconds", 0)) for ep in episodes]
    runtime_rows = [{
        "num_samples": len(episodes),
        "sum_episode_runtime_seconds": sum(runtimes),
        "mean_episode_runtime_seconds": safe_mean(runtimes),
        "median_episode_runtime_seconds": statistics.median(runtimes),
        "p90_episode_runtime_seconds": percentile(runtimes, 0.90),
        "min_episode_runtime_seconds": min(runtimes),
        "max_episode_runtime_seconds": max(runtimes),
        "mean_total_output_tokens": safe_mean(float(ep.get("total_output_tokens", 0)) for ep in episodes),
        "mean_peak_gpu_memory_mb": safe_mean(float(ep.get("peak_gpu_memory_mb", 0)) for ep in episodes),
        "max_peak_gpu_memory_mb": max(float(ep.get("peak_gpu_memory_mb", 0)) for ep in episodes),
    }]
    write_csv(out / "runtime_summary.csv", runtime_rows)

    # Add official scoring fields to the episode-level table while retaining all base columns.
    summaries = []
    for ep in episodes:
        crops = ep.get("crops", [])
        hit_turns = [c["turn_index"] for c in crops if float(c.get("gt_coverage", 0)) >= 0.10]
        summaries.append({
            "sample_id": ep["sample_id"], "question": ep["question"], "category": ep["category"],
            "gt_answer": ep["gt_answer"], "final_answer": ep.get("final_answer", ""),
            "status": ep.get("status", ""), "stop_reason": ep.get("stop_reason", ""),
            "num_rounds": ep.get("num_rounds", 0), "num_crops": len(crops),
            "runtime_seconds": ep.get("runtime_seconds", 0), "total_output_tokens": ep.get("total_output_tokens", 0),
            "peak_gpu_memory_mb": ep.get("peak_gpu_memory_mb", 0),
            "first_gt_coverage_10_turn": min(hit_turns) if hit_turns else "",
            "ever_gt_coverage_10_hit": bool(hit_turns), **ep["scores"],
        })
    write_csv(out / "episode_summary.csv", summaries)

    status_counts = Counter(str(ep.get("status")) for ep in episodes)
    final_complete = bool(overall["official_final_complete"])
    category_lines = [
        "| Category | N | Exact | Normalized | Final |",
        "|---|---:|---:|---:|---:|",
    ]
    category_lines.extend(
        f"| {row['category']} | {row['num_samples']} | {row['exact_accuracy']:.2%} | "
        f"{row['normalized_accuracy']:.2%} | {row['official_final_accuracy']:.2%} |"
        for row in by_category
    )
    report = [
        "# VisualNeedle-300 Mini-o3 Baseline Report", "",
        "## Accuracy", "",
        f"- Raw exact match: {overall['exact_accuracy']:.2%} ({overall['exact_correct']}/300)",
        f"- Official normalized match: {overall['normalized_accuracy']:.2%} ({overall['normalized_correct']}/300)",
        f"- Official final accuracy: {overall['official_final_accuracy']:.2%} ({overall['official_final_correct']}/300)",
        f"- Judge pending: {overall['official_judge_pending']} (final accuracy complete: {final_complete})", "",
        f"- Judge model: {', '.join(sorted(judge_models)) if judge_models else 'not available'}",
        f"- Judge protocol: official VisualNeedle image-aware YES/NO prompt; user-authorized Flash substitution.", "",
        "The official evaluator first applies `answers_match`, then sends unmatched answers and the original image to its configured VLM judge. If pending rows exist, they are not silently counted as judged and the displayed final value is only a lower bound.", "",
        "## Accuracy by Category", "",
        *category_lines, "",
        "## Search Behavior", "",
        f"- Mean / median crops: {behavior_overall['mean_crops_per_episode']:.3f} / {behavior_overall['median_crops_per_episode']:.3f}",
        f"- GT coverage >=10% hit: {behavior_overall['gt_coverage_10_hit_rate']:.2%} ({behavior_overall['gt_coverage_10_hit_episodes']}/300)",
        f"- GT center hit: {behavior_overall['gt_center_hit_rate']:.2%} ({behavior_overall['gt_center_hit_episodes']}/300)",
        f"- Geometrically irrelevant crops: {behavior_overall['geometrically_irrelevant_crop_ratio']:.2%} ({behavior_overall['geometrically_irrelevant_crops']}/{behavior_overall['total_crops']})",
        f"- Repeated/high-overlap crops: {behavior_overall['repeated_high_overlap_crop_ratio']:.2%} ({behavior_overall['repeated_high_overlap_crops']}/{behavior_overall['total_crops']})",
        f"- Hit GT but wrong: {behavior_overall['hit_gt_but_wrong_rate_among_hits']:.2%} ({behavior_overall['hit_gt_but_wrong_episodes']}/{behavior_overall['gt_coverage_10_hit_episodes']})",
        f"- Correct without GT hit: {behavior_overall['correct_without_gt_hit_episodes']}", "",
        "## Reliability", "",
        f"- Status counts: `{dict(status_counts)}`",
        f"- Max-round episodes: {behavior_overall['reached_max_rounds']}",
        f"- Max-image episodes: {behavior_overall['reached_max_images']}",
        f"- Mean episode runtime: {runtime_rows[0]['mean_episode_runtime_seconds']:.2f}s",
        f"- Summed episode runtime: {runtime_rows[0]['sum_episode_runtime_seconds'] / 3600:.2f} GPU-hours",
        f"- Final baseline-correct IDs: {len(correct_ids)}",
        "", "## Next-stage Readiness", "",
        "The run is technically complete and suitable for a paired misleading-crop pilot. However, only 29 baseline-correct samples are eligible for correct-to-wrong analysis, while 6 OOM, 10 invalid, and 137 max-round episodes limit statistical power. Preserve these limitations in downstream claims.",
    ]
    (out / "full300_baseline_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    integrity_path = out / "integrity_report.md"
    integrity = integrity_path.read_text(encoding="utf-8") if integrity_path.exists() else "# Integrity Report\n"
    integrity = integrity.split("\n## Final Scoring Integrity", 1)[0].rstrip()
    scoring_integrity = [
        "", "## Final Scoring Integrity", "",
        f"- Judge model(s): `{sorted(judge_models)}`",
        f"- Unique parsed judge records: {len(judge_results)}",
        f"- Judge pending: {overall['official_judge_pending']}",
        f"- Final correct sample IDs: {len(correct_ids)}",
        f"- Episode/judge completeness: **{'PASS' if final_complete and len(judge_results) == 138 else 'FAIL'}**",
    ]
    integrity_path.write_text(integrity + "\n" + "\n".join(scoring_integrity) + "\n", encoding="utf-8")

    print(json.dumps({"accuracy": overall, "behavior": behavior_overall, "runtime": runtime_rows[0], "status": status_counts}, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
