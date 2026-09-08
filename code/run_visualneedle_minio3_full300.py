#!/usr/bin/env python3
"""Run the validated Mini-o3 VisualNeedle protocol on all 300 samples."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import run_visualneedle_minio3_smoke20 as baseline
from run_visualneedle_minio3_smoke20_cache import CacheEnabledHFBackend


OUTPUT_DIR = baseline.OUTPUT_PARENT / "full300_baseline"
ORIGINAL_WRITE_CONFIG = baseline.write_config
ORIGINAL_EPISODE_SUMMARY = baseline.episode_summary
ORIGINAL_INTEGRITY_CHECKS = baseline.integrity_checks


def prepare_full_selection() -> List[str]:
    rows = baseline.load_annotation_rows()
    ids = [str(row["id"]) for row in rows]
    if len(ids) != 300 or len(ids) != len(set(ids)):
        raise ValueError(f"Expected 300 unique official samples, found {len(ids)} rows")
    counts = Counter(str(row["category"]) for row in rows)
    expected = Counter({
        "Color Recognition": 73,
        "OCR Recognition": 71,
        "Entity Recognition": 64,
        "Spatial Relationship": 58,
        "Occluded Object Recognition": 34,
    })
    if counts != expected:
        raise ValueError(f"Unexpected official category distribution: {dict(counts)}")
    return ids


def full_episode_summary(episode: Dict[str, Any]) -> Dict[str, Any]:
    row = ORIGINAL_EPISODE_SUMMARY(episode)
    row["total_output_tokens"] = int(episode.get("total_output_tokens", 0))
    row["peak_gpu_memory_mb"] = float(episode.get("peak_gpu_memory_mb", 0.0))
    return row


def full_integrity_checks(
    ids: Sequence[str], episodes: Sequence[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    _, problems = ORIGINAL_INTEGRITY_CHECKS(ids, episodes)
    problems = [problem for problem in problems if not problem.startswith("Expected 20 episode records")]
    if len(episodes) != len(ids):
        problems.append(f"Expected {len(ids)} episode records, found {len(episodes)}")
    if len(ids) != 300:
        problems.append(f"Expected the full 300-sample selection, found {len(ids)}")
    return not problems, problems


def write_full_config(ids: Sequence[str], command: str) -> None:
    ORIGINAL_WRITE_CONFIG(ids, command)
    path = OUTPUT_DIR / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update({
        "experiment": "VisualNeedle-300 Mini-o3 greedy active visual search baseline",
        "dataset_split": "official 300 English samples",
        "selection": "all annotation rows in official file order; no sampling or replacement",
        "selection_seed": None,
        "samples_per_category": None,
        "num_samples": 300,
        "num_shards": 8,
        "use_cache": True,
        "model_config_text_config_use_cache": True,
        "model_generation_config_use_cache": True,
        "model_generate_explicit_use_cache": True,
        "cross_round_kv_reuse": False,
        "output_directory": str(OUTPUT_DIR),
        "protected_output_directories": [
            str(baseline.OUTPUT_PARENT / "smoke20"),
            str(baseline.OUTPUT_PARENT / "smoke20_cache"),
            str(baseline.OUTPUT_PARENT / "cache_validation"),
        ],
    })
    baseline.atomic_write_json(path, config)


def main() -> None:
    baseline.OUTPUT_DIR = OUTPUT_DIR
    baseline.prepare_selection = prepare_full_selection
    baseline.HFBackend = CacheEnabledHFBackend
    baseline.episode_summary = full_episode_summary
    baseline.integrity_checks = full_integrity_checks
    baseline.write_config = write_full_config
    for field in ("use_cache", "tokens_per_second"):
        if field not in baseline.TURN_FIELDS:
            baseline.TURN_FIELDS.append(field)
    for field in ("total_output_tokens", "peak_gpu_memory_mb"):
        if field not in baseline.SUMMARY_FIELDS:
            baseline.SUMMARY_FIELDS.append(field)
    baseline.main()
    if "--merge" in sys.argv:
        report_path = OUTPUT_DIR / "integrity_report.md"
        text = report_path.read_text(encoding="utf-8")
        text = text.replace("# VisualNeedle Mini-o3 Smoke20 Integrity Report", "# VisualNeedle Mini-o3 Full300 Integrity Report")
        text = text.replace("Frozen sample records", "Official sample records")
        report_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
