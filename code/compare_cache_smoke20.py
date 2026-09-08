#!/usr/bin/env python3
"""Strictly compare the frozen no-cache and cache-enabled smoke20 trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT = Path("/home/songzhoujie/cvpr27/vstar")
PARENT = PROJECT / "outputs" / "visualneedle_minio3"
OLD_DIR = PARENT / "smoke20"
NEW_DIR = PARENT / "smoke20_cache"
OUTPUT_DIR = PARENT / "cache_validation"
IDS_PATH = PARENT / "smoke20_sample_ids.txt"
SPECIAL_SAMPLE = "TPROMPT65ab072c3b73"
NUM_SHARDS = 8


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").strip()
    return re.sub(r"\s+", " ", value)


def normalize_answer(value: str) -> str:
    value = normalize_text(value).lower()
    return value.strip(" .,!?:;\"'`()[]{}")


def json_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_by_turn(episode: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(crop["turn_index"]): crop for crop in episode.get("crops", [])}


ACTION_FIELDS = [
    "action_type",
    "source",
    "predicted_bbox_raw",
    "predicted_bbox_clipped",
    "bbox_valid",
    "crop_executed",
    "final_answer",
]
CROP_FIELDS = [
    "source",
    "predicted_bbox_raw",
    "predicted_bbox_clipped",
    "original_crop_width",
    "original_crop_height",
    "processed_width",
    "processed_height",
]


def turn_differences(
    old_turn: Optional[Dict[str, Any]],
    new_turn: Optional[Dict[str, Any]],
    old_crop: Optional[Dict[str, Any]],
    new_crop: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    if old_turn is None or new_turn is None:
        return ["turn_missing"], ["turn_missing"]
    exact: List[str] = []
    semantic: List[str] = []
    if old_turn.get("raw_model_output", "") != new_turn.get("raw_model_output", ""):
        exact.append("raw_model_output")
        if normalize_text(old_turn.get("raw_model_output", "")) != normalize_text(new_turn.get("raw_model_output", "")):
            semantic.append("raw_model_output_content")
    for field in ACTION_FIELDS:
        if old_turn.get(field) != new_turn.get(field):
            exact.append(field)
            if field == "final_answer":
                if normalize_answer(str(old_turn.get(field, ""))) != normalize_answer(str(new_turn.get(field, ""))):
                    semantic.append(field)
            else:
                semantic.append(field)
    if (old_crop is None) != (new_crop is None):
        exact.append("crop_presence")
        semantic.append("crop_presence")
    elif old_crop is not None and new_crop is not None:
        for field in CROP_FIELDS:
            if old_crop.get(field) != new_crop.get(field):
                exact.append(f"crop.{field}")
                semantic.append(f"crop.{field}")
    return sorted(set(exact)), sorted(set(semantic))


def compare_episode(old: Dict[str, Any], new: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    old_turns = old.get("turns", [])
    new_turns = new.get("turns", [])
    old_crops = crop_by_turn(old)
    new_crops = crop_by_turn(new)
    per_turn: List[Dict[str, Any]] = []
    exact_differences: List[str] = []
    semantic_differences: List[str] = []
    first_divergence_turn: Any = ""
    first_divergence_fields: List[str] = []
    maximum_turns = max(len(old_turns), len(new_turns))

    for index in range(maximum_turns):
        old_turn = old_turns[index] if index < len(old_turns) else None
        new_turn = new_turns[index] if index < len(new_turns) else None
        turn_index = index + 1
        old_crop = old_crops.get(turn_index)
        new_crop = new_crops.get(turn_index)
        exact, semantic = turn_differences(old_turn, new_turn, old_crop, new_crop)
        if exact:
            exact_differences.extend(f"turn_{turn_index}.{field}" for field in exact)
        if semantic:
            semantic_differences.extend(f"turn_{turn_index}.{field}" for field in semantic)
            if first_divergence_turn == "":
                first_divergence_turn = turn_index
                first_divergence_fields = semantic

        row: Dict[str, Any] = {
            "sample_id": old["sample_id"],
            "category": old["category"],
            "turn_index": turn_index,
            "turn_present_old": old_turn is not None,
            "turn_present_new": new_turn is not None,
            "raw_output_exact": bool(old_turn and new_turn) and old_turn.get("raw_model_output", "") == new_turn.get("raw_model_output", ""),
            "raw_output_semantic_equal": bool(old_turn and new_turn) and normalize_text(old_turn.get("raw_model_output", "")) == normalize_text(new_turn.get("raw_model_output", "")),
            "difference_fields": json_cell(exact),
            "semantic_difference_fields": json_cell(semantic),
        }
        for field in (
            "action_type", "raw_model_output", "final_answer", "source", "predicted_bbox_raw",
            "predicted_bbox_clipped", "bbox_valid", "crop_executed", "finish_reason", "output_tokens",
            "input_tokens", "runtime_seconds", "tokens_per_second", "use_cache",
        ):
            row[f"old_{field}"] = json_cell(old_turn.get(field)) if old_turn else ""
            row[f"new_{field}"] = json_cell(new_turn.get(field)) if new_turn else ""
        for field in CROP_FIELDS:
            row[f"old_crop_{field}"] = json_cell(old_crop.get(field)) if old_crop else ""
            row[f"new_crop_{field}"] = json_cell(new_crop.get(field)) if new_crop else ""
        per_turn.append(row)

    episode_exact_fields = {
        "num_rounds": old.get("num_rounds") == new.get("num_rounds"),
        "num_crops": old.get("num_crops") == new.get("num_crops"),
        "final_answer": old.get("final_answer", "") == new.get("final_answer", ""),
        "produced_final_answer": old.get("produced_final_answer") == new.get("produced_final_answer"),
        "stop_reason": old.get("stop_reason") == new.get("stop_reason"),
        "status": old.get("status") == new.get("status"),
    }
    for field, matches in episode_exact_fields.items():
        if not matches:
            exact_differences.append(f"episode.{field}")
            if field == "final_answer":
                if normalize_answer(old.get("final_answer", "")) != normalize_answer(new.get("final_answer", "")):
                    semantic_differences.append(f"episode.{field}")
            else:
                semantic_differences.append(f"episode.{field}")
    if semantic_differences and first_divergence_turn == "":
        first_divergence_turn = min(len(old_turns), len(new_turns)) + 1
        first_divergence_fields = [item for item in semantic_differences if item.startswith("episode.")]

    if not exact_differences:
        level = "EXACT_TRAJECTORY_MATCH"
    elif not semantic_differences:
        level = "SEMANTIC_TRAJECTORY_MATCH"
    else:
        level = "TRAJECTORY_DIVERGENCE"

    old_runtime = float(old.get("runtime_seconds", 0.0))
    new_runtime = float(new.get("runtime_seconds", 0.0))
    summary = {
        "sample_id": old["sample_id"],
        "category": old["category"],
        "consistency_level": level,
        "first_divergence_turn": first_divergence_turn,
        "first_divergence_fields": json_cell(first_divergence_fields),
        "all_difference_fields": json_cell(sorted(set(exact_differences))),
        "old_final_answer": old.get("final_answer", ""),
        "new_final_answer": new.get("final_answer", ""),
        "final_answer_exact_match_between_runs": old.get("final_answer", "") == new.get("final_answer", ""),
        "final_answer_normalized_match_between_runs": normalize_answer(old.get("final_answer", "")) == normalize_answer(new.get("final_answer", "")),
        "old_normalized_final_answer": normalize_answer(old.get("final_answer", "")),
        "new_normalized_final_answer": normalize_answer(new.get("final_answer", "")),
        "old_accuracy": bool(old.get("normalized_string_match")),
        "new_accuracy": bool(new.get("normalized_string_match")),
        "old_produced_final_answer": old.get("produced_final_answer"),
        "new_produced_final_answer": new.get("produced_final_answer"),
        "old_num_rounds": old.get("num_rounds"),
        "new_num_rounds": new.get("num_rounds"),
        "old_num_crops": old.get("num_crops"),
        "new_num_crops": new.get("num_crops"),
        "old_stop_reason": old.get("stop_reason"),
        "new_stop_reason": new.get("stop_reason"),
        "old_status": old.get("status"),
        "new_status": new.get("status"),
        "old_runtime_seconds": old_runtime,
        "new_runtime_seconds": new_runtime,
        "runtime_speedup": old_runtime / new_runtime if new_runtime else None,
        "old_peak_gpu_memory_mb": float(old.get("peak_gpu_memory_mb", 0.0)),
        "new_peak_gpu_memory_mb": float(new.get("peak_gpu_memory_mb", 0.0)),
        "peak_gpu_memory_delta_mb": float(new.get("peak_gpu_memory_mb", 0.0)) - float(old.get("peak_gpu_memory_mb", 0.0)),
    }
    return summary, per_turn


def distribution(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
    }


def runtime_summary(episodes: Sequence[Dict[str, Any]], ids: Sequence[str]) -> Dict[str, Any]:
    runtimes = [float(episode.get("runtime_seconds", 0.0)) for episode in episodes]
    turn_runtimes = [float(turn.get("runtime_seconds", 0.0)) for episode in episodes for turn in episode.get("turns", [])]
    shard_totals = []
    by_id = {episode["sample_id"]: episode for episode in episodes}
    for shard in range(NUM_SHARDS):
        shard_totals.append(sum(float(by_id[sample_id].get("runtime_seconds", 0.0)) for index, sample_id in enumerate(ids) if index % NUM_SHARDS == shard))
    return {
        "episode_runtime_seconds": distribution(runtimes),
        "mean_runtime_per_turn_seconds": statistics.mean(turn_runtimes) if turn_runtimes else 0.0,
        "median_runtime_per_turn_seconds": statistics.median(turn_runtimes) if turn_runtimes else 0.0,
        "approximate_8_shard_wall_clock_seconds_excluding_load_and_merge": max(shard_totals, default=0.0),
        "shard_runtime_totals_seconds": shard_totals,
        "peak_gpu_memory_mb": distribution([float(episode.get("peak_gpu_memory_mb", 0.0)) for episode in episodes]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, default=OLD_DIR)
    parser.add_argument("--new-dir", type=Path, default=NEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = [line.strip() for line in IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_rows = read_jsonl(args.old_dir / "trajectories.jsonl")
    new_rows = read_jsonl(args.new_dir / "trajectories.jsonl")
    old_by_id = {row["sample_id"]: row for row in old_rows}
    new_by_id = {row["sample_id"]: row for row in new_rows}
    if len(ids) != 20 or set(old_by_id) != set(ids) or set(new_by_id) != set(ids):
        raise ValueError("OLD, NEW, and frozen sample IDs must contain the same 20 samples")

    sample_rows: List[Dict[str, Any]] = []
    turn_rows: List[Dict[str, Any]] = []
    for sample_id in ids:
        sample, turns = compare_episode(old_by_id[sample_id], new_by_id[sample_id])
        sample_rows.append(sample)
        turn_rows.extend(turns)

    levels = Counter(row["consistency_level"] for row in sample_rows)
    old_runtime = runtime_summary(old_rows, ids)
    new_runtime = runtime_summary(new_rows, ids)
    speedups = [float(row["runtime_speedup"]) for row in sample_rows if row["runtime_speedup"] is not None]
    special = next(row for row in sample_rows if row["sample_id"] == SPECIAL_SAMPLE)
    final_exact = sum(bool(row["final_answer_exact_match_between_runs"]) for row in sample_rows)
    final_normalized = sum(bool(row["final_answer_normalized_match_between_runs"]) for row in sample_rows)
    old_accuracy = sum(bool(row["old_accuracy"]) for row in sample_rows) / len(sample_rows)
    new_accuracy = sum(bool(row["new_accuracy"]) for row in sample_rows) / len(sample_rows)
    memory_deltas = [float(row["peak_gpu_memory_delta_mb"]) for row in sample_rows]
    summary = {
        "total_samples": len(sample_rows),
        "exact_trajectory_match": levels["EXACT_TRAJECTORY_MATCH"],
        "semantic_trajectory_match": levels["SEMANTIC_TRAJECTORY_MATCH"],
        "trajectory_divergence": levels["TRAJECTORY_DIVERGENCE"],
        "final_answer_exact_consistency": final_exact / len(sample_rows),
        "final_answer_normalized_consistency": final_normalized / len(sample_rows),
        "old_normalized_accuracy": old_accuracy,
        "new_normalized_accuracy": new_accuracy,
        "accuracy_identical": old_accuracy == new_accuracy,
        "old_runtime": old_runtime,
        "new_runtime": new_runtime,
        "runtime_speedup_distribution": distribution(speedups),
        "aggregate_runtime_speedup": sum(float(row["old_runtime_seconds"]) for row in sample_rows) / sum(float(row["new_runtime_seconds"]) for row in sample_rows),
        "mean_peak_gpu_memory_delta_mb": statistics.mean(memory_deltas),
        "median_peak_gpu_memory_delta_mb": statistics.median(memory_deltas),
        "special_sample": special,
        "baseline_file_hashes": {
            name: sha256(args.old_dir / name)
            for name in ("trajectories.jsonl", "crop_records.csv", "episode_summary.csv")
        },
    }

    sample_fields = list(sample_rows[0].keys())
    turn_fields = list(turn_rows[0].keys())
    write_csv(output_dir / "cache_validation_per_sample.csv", sample_rows, sample_fields)
    write_csv(output_dir / "cache_validation_per_turn.csv", turn_rows, turn_fields)
    runtime_fields = [
        "sample_id", "category", "old_runtime_seconds", "new_runtime_seconds", "runtime_speedup",
        "old_peak_gpu_memory_mb", "new_peak_gpu_memory_mb", "peak_gpu_memory_delta_mb",
        "old_num_rounds", "new_num_rounds", "old_num_crops", "new_num_crops",
    ]
    write_csv(output_dir / "cache_runtime_comparison.csv", sample_rows, runtime_fields)
    write_json(output_dir / "cache_validation_summary.json", summary)

    divergence_rows = [row for row in sample_rows if row["consistency_level"] == "TRAJECTORY_DIVERGENCE"]
    safe = not divergence_rows and final_normalized == len(sample_rows) and old_accuracy == new_accuracy
    lines = [
        "# Mini-o3 VisualNeedle KV-cache Multi-turn Validation",
        "",
        "## 一致性",
        "",
        f"- EXACT_TRAJECTORY_MATCH: **{levels['EXACT_TRAJECTORY_MATCH']}/20**",
        f"- SEMANTIC_TRAJECTORY_MATCH: **{levels['SEMANTIC_TRAJECTORY_MATCH']}/20**",
        f"- TRAJECTORY_DIVERGENCE: **{levels['TRAJECTORY_DIVERGENCE']}/20**",
        f"- final answer exact consistency: **{final_exact}/20 ({final_exact/20:.1%})**",
        f"- final answer normalized consistency: **{final_normalized}/20 ({final_normalized/20:.1%})**",
        f"- normalized accuracy OLD / NEW: **{old_accuracy:.1%} / {new_accuracy:.1%}**",
        "",
        "## 首次分叉",
        "",
        "| sample | first turn | fields | OLD status | NEW status |",
        "|---|---:|---|---|---|",
    ]
    if divergence_rows:
        lines.extend(
            f"| {row['sample_id']} | {row['first_divergence_turn']} | `{row['first_divergence_fields']}` | {row['old_status']} | {row['new_status']} |"
            for row in divergence_rows
        )
    else:
        lines.append("| None | - | - | - | - |")
    lines.extend(
        [
            "",
            "## 性能",
            "",
            f"- mean episode runtime OLD / NEW: **{old_runtime['episode_runtime_seconds']['mean']:.2f}s / {new_runtime['episode_runtime_seconds']['mean']:.2f}s**",
            f"- median episode runtime OLD / NEW: **{old_runtime['episode_runtime_seconds']['median']:.2f}s / {new_runtime['episode_runtime_seconds']['median']:.2f}s**",
            f"- max episode runtime OLD / NEW: **{old_runtime['episode_runtime_seconds']['max']:.2f}s / {new_runtime['episode_runtime_seconds']['max']:.2f}s**",
            f"- approximate 8-shard wall time OLD / NEW: **{old_runtime['approximate_8_shard_wall_clock_seconds_excluding_load_and_merge']:.2f}s / {new_runtime['approximate_8_shard_wall_clock_seconds_excluding_load_and_merge']:.2f}s**",
            f"- aggregate runtime speedup: **{summary['aggregate_runtime_speedup']:.2f}x**",
            f"- median per-sample speedup: **{summary['runtime_speedup_distribution']['median']:.2f}x**",
            f"- mean peak-memory increase: **{summary['mean_peak_gpu_memory_delta_mb']:.1f} MB**",
            f"- `{SPECIAL_SAMPLE}` OLD / NEW: **{special['old_runtime_seconds']:.2f}s / {special['new_runtime_seconds']:.2f}s ({special['runtime_speedup']:.2f}x)**",
            "",
            "## 结论",
            "",
            f"- 是否可安全固化 `use_cache=True`: **{'是' if safe else '尚不能直接判定'}**",
            "- 本轮只改变标准的 within-generate KV cache；未使用跨轮 cache，也未改变 prompt、processor、crop 或 termination policy。",
        ]
    )
    if divergence_rows:
        lines.append("- 分叉样本需结合逐轮 CSV 判断：若 OLD 在 `max_time=600` 被截断而 NEW 正常完成，该差异属于修复既有时间截断，而不是 cache 改变了截断前的模型决策。")
    (output_dir / "cache_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
