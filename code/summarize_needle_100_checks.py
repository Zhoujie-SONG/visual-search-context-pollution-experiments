#!/usr/bin/env python3
"""Combine geometric crop checks, content labels, and semantic rescoring."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CLASS_TYPES = ["MISLEADING", "EXPLORATION", "REDUNDANT", "NEAR_MISS", "PARSE_INVALID"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def bbox_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.10)
    args = parser.parse_args()

    trajectories = read_jsonl(args.experiment_dir / "trajectories.jsonl")
    classifications = read_jsonl(args.classification_dir / "irrelevant_crop_classifications.jsonl")
    reviewed_semantic_path = args.semantic_dir / "semantic_rescore_reviewed.csv"
    semantic = read_csv(
        reviewed_semantic_path
        if reviewed_semantic_path.exists()
        else args.semantic_dir / "semantic_rescore.csv"
    )
    class_by_key = {
        (row["sample_id"], int(row["turn_idx"])): row for row in classifications
    }
    semantic_agent = {
        row["sample_id"]: row for row in semantic if row["evaluation_source"] == "agent"
    }

    per_crop = []
    per_episode = []
    for trajectory in trajectories:
        episode = trajectory["episode"]
        gt = [float(value) for value in trajectory["gt_bbox_xyxy"]]
        gt_area = bbox_area(gt)
        irrelevant_count = 0
        for crop in trajectory.get("crops", []):
            turn_idx = int(crop["turn_id"]) + 1
            bbox = [float(value) for value in crop["bbox_original"]]
            coverage = intersection_area(bbox, gt) / gt_area if gt_area else 0.0
            irrelevant = coverage < args.coverage_threshold
            label_row = class_by_key.get((episode["sample_id"], turn_idx), {})
            label = label_row.get("type", "") or ("PARSE_INVALID" if irrelevant else "")
            irrelevant_count += int(irrelevant)
            per_crop.append(
                {
                    "sample_id": episode["sample_id"],
                    "category": episode["category"],
                    "turn_idx": turn_idx,
                    "crop_bbox": json.dumps(bbox, ensure_ascii=False),
                    "gt_bbox": json.dumps(gt, ensure_ascii=False),
                    "target_coverage": coverage,
                    "geometrically_irrelevant": irrelevant,
                    "crop_gt_iou": crop.get("iou_with_gt", ""),
                    "content_type": label,
                    "could_yield_wrong_answer": label_row.get("could_yield_wrong_answer", ""),
                    "classification_reason": label_row.get("reason", ""),
                    "crop_path": crop.get("crop_path", ""),
                }
            )
        score = semantic_agent.get(episode["sample_id"], {})
        crop_count = len(trajectory.get("crops", []))
        per_episode.append(
            {
                "sample_id": episode["sample_id"],
                "category": episode["category"],
                "total_crops": crop_count,
                "irrelevant_crops": irrelevant_count,
                "irrelevant_fraction": irrelevant_count / crop_count if crop_count else 0.0,
                "semantic_verdict": score.get("verdict", ""),
                "semantic_correct": score.get("semantic_correct", ""),
                "gt_answer": episode["gt_answer"],
                "semantic_extracted_answer": score.get("extracted_answer", ""),
            }
        )

    if len(trajectories) != 100 or len(per_episode) != 100:
        raise RuntimeError(f"expected 100 episodes, found {len(per_episode)}")
    irrelevant_rows = [row for row in per_crop if row["geometrically_irrelevant"]]
    if len(classifications) != len(irrelevant_rows):
        raise RuntimeError(
            f"classification coverage mismatch: labels={len(classifications)} irrelevant={len(irrelevant_rows)}"
        )

    crop_fields = list(per_crop[0])
    episode_fields = list(per_episode[0])
    write_csv(args.experiment_dir / "crop_check_100_per_turn.csv", per_crop, crop_fields)
    write_csv(args.experiment_dir / "crop_check_100_per_episode.csv", per_episode, episode_fields)

    summary = []
    categories = sorted({row["category"] for row in per_episode})
    for category in ["ALL", *categories]:
        episodes = per_episode if category == "ALL" else [row for row in per_episode if row["category"] == category]
        episode_ids = {row["sample_id"] for row in episodes}
        crops = [row for row in per_crop if row["sample_id"] in episode_ids]
        irrelevant = [row for row in crops if row["geometrically_irrelevant"]]
        counts = Counter(row["content_type"] or "PARSE_INVALID" for row in irrelevant)
        semantic_counts = Counter(row["semantic_verdict"] for row in episodes)
        row = {
            "category": category,
            "num_episodes": len(episodes),
            "total_crops": len(crops),
            "irrelevant_crops": len(irrelevant),
            "irrelevant_crop_rate": len(irrelevant) / len(crops) if crops else 0.0,
            "episodes_with_irrelevant": sum(row["irrelevant_crops"] > 0 for row in episodes),
            "episode_with_irrelevant_rate": sum(row["irrelevant_crops"] > 0 for row in episodes) / len(episodes) if episodes else 0.0,
            "mean_irrelevant_crops": sum(row["irrelevant_crops"] for row in episodes) / len(episodes) if episodes else 0.0,
            "semantic_correct": semantic_counts["CORRECT"],
            "semantic_accuracy": semantic_counts["CORRECT"] / len(episodes) if episodes else 0.0,
            "semantic_incorrect": semantic_counts["INCORRECT"],
            "semantic_no_answer": semantic_counts["NO_ANSWER"],
        }
        for label in CLASS_TYPES:
            row[label.lower()] = counts[label]
            row[f"{label.lower()}_fraction_of_irrelevant"] = counts[label] / len(irrelevant) if irrelevant else 0.0
        summary.append(row)
    summary_fields = list(summary[0])
    write_csv(args.experiment_dir / "crop_and_accuracy_summary_100.csv", summary, summary_fields)

    episode_ids_by_type: Dict[str, set[str]] = defaultdict(set)
    episode_ids_by_type["ANY_IRRELEVANT"] = {
        row["sample_id"] for row in per_episode if row["irrelevant_crops"] > 0
    }
    for row in irrelevant_rows:
        episode_ids_by_type[row["content_type"] or "PARSE_INVALID"].add(row["sample_id"])
    association = []
    for label in ["ANY_IRRELEVANT", *CLASS_TYPES]:
        ids = episode_ids_by_type[label]
        inside = [row for row in per_episode if row["sample_id"] in ids]
        outside = [row for row in per_episode if row["sample_id"] not in ids]
        association.append(
            {
                "crop_group": label,
                "episodes_with_group": len(inside),
                "accuracy_with_group": sum(row["semantic_verdict"] == "CORRECT" for row in inside) / len(inside) if inside else 0.0,
                "episodes_without_group": len(outside),
                "accuracy_without_group": sum(row["semantic_verdict"] == "CORRECT" for row in outside) / len(outside) if outside else 0.0,
            }
        )
    write_csv(
        args.experiment_dir / "crop_type_accuracy_association_100.csv",
        association,
        list(association[0]),
    )

    outcome_burden = []
    for verdict in ["CORRECT", "INCORRECT", "NO_ANSWER"]:
        rows = [row for row in per_episode if row["semantic_verdict"] == verdict]
        outcome_burden.append(
            {
                "semantic_verdict": verdict,
                "num_episodes": len(rows),
                "mean_total_crops": sum(row["total_crops"] for row in rows) / len(rows) if rows else 0.0,
                "mean_irrelevant_crops": sum(row["irrelevant_crops"] for row in rows) / len(rows) if rows else 0.0,
            }
        )
    write_csv(
        args.experiment_dir / "semantic_outcome_crop_burden_100.csv",
        outcome_burden,
        list(outcome_burden[0]),
    )

    all_row = summary[0]
    report = [
        "# VisualNeedle 100-sample Crop and Semantic Accuracy Check",
        "",
        f"- Episodes: {all_row['num_episodes']}",
        f"- Total crops: {all_row['total_crops']}",
        f"- Geometrically irrelevant crops (target coverage < {args.coverage_threshold:.0%}): "
        f"{all_row['irrelevant_crops']} ({all_row['irrelevant_crop_rate']:.2%})",
        f"- Episodes containing at least one irrelevant crop: {all_row['episodes_with_irrelevant']} "
        f"({all_row['episode_with_irrelevant_rate']:.2%})",
        f"- Semantic agent accuracy: {all_row['semantic_accuracy']:.2%} "
        f"({all_row['semantic_correct']}/{all_row['num_episodes']})",
        f"- Semantic incorrect: {all_row['semantic_incorrect']}",
        f"- No answer: {all_row['semantic_no_answer']}",
        "",
        "## Irrelevant-crop content labels",
        "",
    ]
    for label in CLASS_TYPES:
        report.append(
            f"- {label}: {all_row[label.lower()]} "
            f"({all_row[f'{label.lower()}_fraction_of_irrelevant']:.2%})"
        )
    any_irrelevant = association[0]
    report.extend(
        [
            "",
            "## Association with semantic accuracy",
            "",
            f"- With at least one geometrically irrelevant crop: "
            f"{any_irrelevant['accuracy_with_group']:.2%} "
            f"({any_irrelevant['episodes_with_group']} episodes)",
            f"- Without a geometrically irrelevant crop: "
            f"{any_irrelevant['accuracy_without_group']:.2%} "
            f"({any_irrelevant['episodes_without_group']} episodes)",
            "- This is an observational association, not a causal estimate: harder examples can both trigger more crops and reduce accuracy.",
            "",
            "Geometric irrelevance alone does not imply a bad search step. EXPLORATION may be useful for localization, "
            "while MISLEADING and some NEAR_MISS crops are more likely to induce wrong answers.",
        ]
    )
    (args.experiment_dir / "crop_and_accuracy_report_100.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(all_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
