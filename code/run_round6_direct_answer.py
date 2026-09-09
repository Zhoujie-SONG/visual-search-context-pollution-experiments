#!/usr/bin/env python3
"""One-shot direct-answer diagnostic over frozen Round-6 prefixes."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import run_round6_memory_surgery as surgery
import run_visualneedle_minio3_smoke20 as baseline


PROJECT_ROOT = Path("/home/songzhoujie/cvpr27/vstar")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "visualneedle_minio3" / "round6_direct_answer_v1"
BASELINE_DIR = surgery.BASELINE_DIR
MODEL_PATH = surgery.MODEL_PATH
PREFIX_CROP_COUNT = surgery.PREFIX_CROP_COUNT
SEED = surgery.SEED
ANSWER_MAX_NEW_TOKENS = 128
NUM_SHARDS = 8
ARMS = (
    "F_full_direct",
    "O_oracle_top2_direct",
    "R_recent_top2_direct",
    "X_random_top2_direct",
)
STAGES = ("sanity", "full")
DIRECT_ANSWER_INSTRUCTION = (
    "Visual search is now finished.\n"
    "Do not perform any further grounding or visual tool calls.\n"
    "Based only on the visual evidence already available,\n"
    "answer the original question now.\n"
    "Put only the final answer inside <answer> and </answer>."
)


@dataclass(frozen=True)
class DirectPlan:
    sample_id: str
    condition: str
    retained_crop_indices: List[int]
    evicted_crop_indices: List[int]
    prefix_snapshot_sha256: str
    direct_instruction_sha256: str


RESULT_FIELDS = [
    "sample_id", "condition", "cohort", "category", "question", "gt_answer",
    "original_image_path", "status", "max_prefix_gt_coverage",
    "first_gt_hit_crop_index", "retained_crop_indices", "evicted_crop_indices",
    "retained_gt_coverages", "retained_prefix_image_count", "evicted_prefix_image_count",
    "available_prefix_source_names", "cumulative_acquired_images_at_intervention", "input_image_count",
    "input_tokens", "visual_tokens", "visual_tokens_per_image", "raw_model_output",
    "parsed_answer", "grounding_present", "answer_present", "protocol_violation",
    "strict_valid_answer", "lenient_valid_answer", "strict_correct_exact",
    "strict_correct_normalized", "strict_final_correct", "lenient_correct_exact",
    "lenient_correct_normalized", "lenient_final_correct", "official_judge_result",
    "official_judge_pending", "finish_reason", "output_tokens", "runtime_seconds",
    "peak_gpu_memory_mb", "generation_count", "tool_execution_count", "oom", "error",
    "error_type", "error_message", "prefix_snapshot_sha256", "original_image_sha256",
    "prefix_crop_sha256", "prefix_raw_crop_sha256", "assistant_text_sha256",
    "direct_instruction_sha256", "input_role_sequence",
]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_dir(stage: str) -> Path:
    return OUTPUT_ROOT / stage


def retained_indices(arm: str, sample_id: str, episode: Dict[str, Any]) -> List[int]:
    mapping = {
        "F_full_direct": "A_full",
        "O_oracle_top2_direct": "B_oracle_top2",
        "R_recent_top2_direct": "C_recent_top2",
        "X_random_top2_direct": "E_random_top2",
    }
    return surgery.retained_indices(mapping[arm], sample_id, episode)


def selected_ids(stage: str, cohort_rows: Sequence[Dict[str, Any]]) -> List[str]:
    primary = sorted(str(row["sample_id"]) for row in cohort_rows if row["cohort"] == "primary_prefix_hit")
    return primary[:2] if stage == "sanity" else primary


def baseline_fingerprints() -> Dict[str, str]:
    return {
        name: surgery.sha256_file(BASELINE_DIR / name)
        for name in ("config.json", "trajectories.jsonl", "episode_summary.csv", "official_judge_results.csv")
    }


def prepare(stage: str) -> None:
    surgery.load_frozen_config()
    cohort_rows, bundles, episodes = surgery.build_cohort()
    ids = selected_ids(stage, cohort_rows)
    output = stage_dir(stage)
    prefix_dir = OUTPUT_ROOT / "frozen_prefixes"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    instruction_hash = surgery.sha256_json(DIRECT_ANSWER_INSTRUCTION)
    for sample_id in ids:
        payload = asdict(bundles[sample_id])
        baseline.atomic_write_json(prefix_dir / f"{sample_id}.json", payload)
        prefix_hash = surgery.sha256_json(payload)
        for arm in ARMS:
            retained = retained_indices(arm, sample_id, episodes[sample_id])
            plan = DirectPlan(
                sample_id=sample_id,
                condition=arm,
                retained_crop_indices=retained,
                evicted_crop_indices=[index for index in range(1, PREFIX_CROP_COUNT + 1) if index not in retained],
                prefix_snapshot_sha256=prefix_hash,
                direct_instruction_sha256=instruction_hash,
            )
            baseline.atomic_write_json(output / "inference_plans" / arm / f"{sample_id}.json", asdict(plan))
    config = {
        "experiment": "Round-6 Direct-Answer Memory Diagnostic",
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "baseline_directory": str(BASELINE_DIR),
        "baseline_fingerprints": baseline_fingerprints(),
        "model_path": str(MODEL_PATH),
        "python": "/mnt/data2/szj/envs/venv_e2m/bin/python",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "use_cache": True,
        "do_sample": False,
        "min_pixels": baseline.MIN_PIXELS,
        "max_pixels": baseline.MAX_PIXELS,
        "answer_max_new_tokens": ANSWER_MAX_NEW_TOKENS,
        "prefix_crop_count": PREFIX_CROP_COUNT,
        "intervention_point": "after sixth successful crop observation and before one answer-only generation",
        "direct_answer_instruction": DIRECT_ANSWER_INSTRUCTION,
        "direct_answer_instruction_sha256": instruction_hash,
        "tool_execution_enabled": False,
        "post_intervention_generation_count_per_arm": 1,
        "random_seed": SEED,
        "selected_sample_ids": ids,
        "arms": list(ARMS),
        "ground_truth_isolation": (
            "GPU workers read sanitized frozen prefixes and index-only retention plans; "
            "GT metadata is attached only after all inference records are written"
        ),
    }
    baseline.atomic_write_json(output / "config.json", config)
    (output / "sample_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "samples": len(ids), "arms": list(ARMS)}, indent=2))


def load_prefix(sample_id: str) -> surgery.PrefixBundle:
    payload = json.loads((OUTPUT_ROOT / "frozen_prefixes" / f"{sample_id}.json").read_text(encoding="utf-8"))
    return surgery.PrefixBundle(**payload)


def load_plan(stage: str, arm: str, sample_id: str) -> DirectPlan:
    payload = json.loads((stage_dir(stage) / "inference_plans" / arm / f"{sample_id}.json").read_text(encoding="utf-8"))
    return DirectPlan(**payload)


def append_direct_instruction(messages: List[Dict[str, Any]]) -> None:
    if not messages or messages[-1].get("role") != "user":
        raise RuntimeError("Direct-answer intervention expected a final user observation message")
    content = messages[-1].get("content")
    if not isinstance(content, list):
        raise RuntimeError("Direct-answer intervention expected list-form user content")
    content.append({"type": "text", "text": "\n" + DIRECT_ANSWER_INSTRUCTION})


def parse_direct_output(raw: str) -> Dict[str, Any]:
    grounding_present = bool(baseline.parse_grounding(raw).get("present"))
    answer = baseline.parse_answer(raw)
    answer_present = bool(answer)
    return {
        "parsed_answer": answer,
        "grounding_present": grounding_present,
        "answer_present": answer_present,
        "protocol_violation": grounding_present,
        "strict_valid_answer": answer_present and not grounding_present,
        "lenient_valid_answer": answer_present,
    }


def role_sequence(messages: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(message.get("role") or "") for message in messages]


def run_one(backend: surgery.SurgeryBackend, prefix: surgery.PrefixBundle, plan: DirectPlan) -> Dict[str, Any]:
    surgery_plan = surgery.InterventionPlan(
        sample_id=plan.sample_id,
        condition=plan.condition,
        retained_crop_indices=plan.retained_crop_indices,
        evicted_crop_indices=plan.evicted_crop_indices,
        forced_answer=False,
        prefix_snapshot_sha256=plan.prefix_snapshot_sha256,
    )
    _, available_sources, messages, cumulative_images = surgery.reconstruct_messages(prefix, surgery_plan)
    append_direct_instruction(messages)
    if surgery.sha256_json(DIRECT_ANSWER_INSTRUCTION) != plan.direct_instruction_sha256:
        raise RuntimeError("Direct-answer instruction hash differs from plan")
    backend.torch.cuda.reset_peak_memory_stats()
    generation = backend.generate(messages, ANSWER_MAX_NEW_TOKENS)
    parsed = parse_direct_output(str(generation["raw_model_output"]))
    peak = backend.torch.cuda.max_memory_allocated() / (1024 * 1024)
    return {
        "sample_id": prefix.sample_id,
        "condition": plan.condition,
        "cohort": "primary_prefix_hit",
        "question": prefix.question,
        "original_image_path": prefix.original_image_path,
        "episode_finished": True,
        "status": "completed",
        "retained_crop_indices": plan.retained_crop_indices,
        "evicted_crop_indices": plan.evicted_crop_indices,
        "retained_prefix_image_count": len(plan.retained_crop_indices),
        "evicted_prefix_image_count": len(plan.evicted_crop_indices),
        "available_prefix_source_names": sorted(available_sources),
        "cumulative_acquired_images_at_intervention": cumulative_images,
        "raw_model_output": generation["raw_model_output"],
        **parsed,
        "input_image_count": generation["input_image_count"],
        "input_tokens": generation["input_tokens"],
        "visual_tokens": generation.get("visual_tokens"),
        "visual_tokens_per_image": generation.get("visual_tokens_per_image"),
        "image_grid_thw": generation.get("image_grid_thw"),
        "finish_reason": generation["finish_reason"],
        "output_tokens": generation["output_tokens"],
        "runtime_seconds": generation["runtime_seconds"],
        "peak_gpu_memory_mb": peak,
        "generation_count": 1,
        "tool_execution_count": 0,
        "oom": False,
        "error": False,
        "error_type": "",
        "error_message": "",
        "prefix_snapshot_sha256": plan.prefix_snapshot_sha256,
        "original_image_sha256": prefix.original_image_sha256,
        "prefix_crop_sha256": prefix.crop_image_sha256,
        "prefix_raw_crop_sha256": prefix.raw_crop_image_sha256,
        "assistant_text_sha256": prefix.assistant_text_sha256,
        "direct_instruction_sha256": plan.direct_instruction_sha256,
        "input_role_sequence": role_sequence(messages),
    }


def failed_result(prefix: surgery.PrefixBundle, plan: DirectPlan, exc: Exception) -> Dict[str, Any]:
    message = str(exc)
    oom = "out of memory" in message.lower()
    return {
        "sample_id": prefix.sample_id,
        "condition": plan.condition,
        "cohort": "primary_prefix_hit",
        "question": prefix.question,
        "original_image_path": prefix.original_image_path,
        "episode_finished": True,
        "status": "oom" if oom else "error",
        "retained_crop_indices": plan.retained_crop_indices,
        "evicted_crop_indices": plan.evicted_crop_indices,
        "raw_model_output": "",
        **parse_direct_output(""),
        "generation_count": 1,
        "tool_execution_count": 0,
        "oom": oom,
        "error": not oom,
        "error_type": "oom" if oom else type(exc).__name__,
        "error_message": message,
        "traceback": traceback.format_exc(),
        "prefix_snapshot_sha256": plan.prefix_snapshot_sha256,
        "original_image_sha256": prefix.original_image_sha256,
        "prefix_crop_sha256": prefix.crop_image_sha256,
        "prefix_raw_crop_sha256": prefix.raw_crop_image_sha256,
        "assistant_text_sha256": prefix.assistant_text_sha256,
        "direct_instruction_sha256": plan.direct_instruction_sha256,
    }


def complete_record(path: Path, sample_id: str, arm: str, prefix_hash: str) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = {
        "episode_finished", "sample_id", "condition", "status", "generation_count",
        "tool_execution_count", "raw_model_output", "parsed_answer", "prefix_snapshot_sha256",
    }
    return (
        required.issubset(row)
        and row["episode_finished"] is True
        and row["status"] in {"completed", "oom"}
        and row["sample_id"] == sample_id
        and row["condition"] == arm
        and row["prefix_snapshot_sha256"] == prefix_hash
        and row["generation_count"] == 1
        and row["tool_execution_count"] == 0
    )


def run_shard(stage: str, shard_id: int, num_shards: int) -> None:
    output = stage_dir(stage)
    ids = [line for line in (output / "sample_ids.txt").read_text(encoding="utf-8").splitlines() if line]
    assigned = [sample_id for index, sample_id in enumerate(ids) if index % num_shards == shard_id]
    backend = surgery.SurgeryBackend(MODEL_PATH)
    log = [f"stage={stage}", f"shard={shard_id}/{num_shards}", f"gpu={backend.device_name}"]
    for position, sample_id in enumerate(assigned, start=1):
        prefix = load_prefix(sample_id)
        for arm in ARMS:
            plan = load_plan(stage, arm, sample_id)
            raw_dir = output / arm / "shards" / f"shard_{shard_id}" / "raw_results"
            raw_dir.mkdir(parents=True, exist_ok=True)
            path = raw_dir / f"{sample_id}.json"
            if path.exists() and complete_record(path, sample_id, arm, plan.prefix_snapshot_sha256):
                log.append(f"resume_skip {position}/{len(assigned)} {sample_id} {arm}")
                continue
            log.append(f"start {position}/{len(assigned)} {sample_id} {arm} {time.strftime('%F %T')}")
            try:
                result = run_one(backend, prefix, plan)
            except Exception as exc:
                result = failed_result(prefix, plan, exc)
                backend.torch.cuda.empty_cache()
            baseline.atomic_write_json(path, result)
            log.append(
                f"done {sample_id} {arm} status={result['status']} "
                f"answer={result.get('answer_present')} grounding={result.get('grounding_present')}"
            )
            log_path = output / "shards" / f"shard_{shard_id}" / "run_log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "shard": shard_id, "samples": len(assigned)}, indent=2))


def first_hit(episode: Dict[str, Any]) -> Any:
    hits = [
        index + 1 for index, crop in enumerate(list(episode.get("crops") or [])[:PREFIX_CROP_COUNT])
        if float(crop.get("gt_coverage") or 0) >= surgery.GT_HIT_THRESHOLD
    ]
    return hits[0] if hits else ""


def recursively_find_banned_keys(value: Any, banned: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in banned:
                found.add(key)
            found.update(recursively_find_banned_keys(child, banned))
    elif isinstance(value, list):
        for child in value:
            found.update(recursively_find_banned_keys(child, banned))
    return found


def merge(stage: str, num_shards: int) -> None:
    output = stage_dir(stage)
    ids = [line for line in (output / "sample_ids.txt").read_text(encoding="utf-8").splitlines() if line]
    raw: List[Dict[str, Any]] = []
    for index, sample_id in enumerate(ids):
        for arm in ARMS:
            plan = load_plan(stage, arm, sample_id)
            path = output / arm / "shards" / f"shard_{index % num_shards}" / "raw_results" / f"{sample_id}.json"
            if not path.exists() or not complete_record(path, sample_id, arm, plan.prefix_snapshot_sha256):
                raise RuntimeError(f"Missing or incomplete result: {path}")
            raw.append(json.loads(path.read_text(encoding="utf-8")))

    metadata = baseline.load_evaluation_metadata()
    baseline_episodes = {str(ep["sample_id"]): ep for ep in surgery.load_baseline_episodes()}
    for row in raw:
        sample_id = str(row["sample_id"])
        meta = metadata[sample_id]
        prefix_crops = list(baseline_episodes[sample_id].get("crops") or [])[:PREFIX_CROP_COUNT]
        coverages = [float(crop.get("gt_coverage") or 0) for crop in prefix_crops]
        row.update({
            "category": meta.category,
            "gt_answer": meta.gt_answer,
            "gt_bbox": list(meta.gt_bbox),
            "max_prefix_gt_coverage": max(coverages, default=0.0),
            "first_gt_hit_crop_index": first_hit(baseline_episodes[sample_id]),
            "retained_gt_coverages": [coverages[index - 1] for index in row["retained_crop_indices"]],
        })

    problems: List[str] = []
    expected_total = len(ids) * len(ARMS)
    if len(raw) != expected_total or len({(row["sample_id"], row["condition"]) for row in raw}) != expected_total:
        problems.append(f"Expected {expected_total} unique records, found {len(raw)}")
    by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    baseline_rows = baseline_episodes
    banned = {"category", "gt_answer", "gt_bbox", "gt_coverage", "iou_with_gt", "contains_gt_center"}
    for sample_id, rows in by_sample.items():
        if len(rows) != len(ARMS):
            problems.append(f"{sample_id}: expected four arms, found {len(rows)}")
            continue
        for field in ("prefix_snapshot_sha256", "original_image_sha256", "assistant_text_sha256"):
            if len({row.get(field) for row in rows}) != 1:
                problems.append(f"{sample_id}: {field} differs across arms")
        for field in ("prefix_crop_sha256", "prefix_raw_crop_sha256", "input_role_sequence"):
            if len({json.dumps(row.get(field), sort_keys=True) for row in rows}) != 1:
                problems.append(f"{sample_id}: {field} differs across arms")
        arms = {row["condition"]: row for row in rows}
        if arms["F_full_direct"]["retained_crop_indices"] != list(range(1, 7)):
            problems.append(f"{sample_id}: F retention is not all six crops")
        if arms["R_recent_top2_direct"]["retained_crop_indices"] != [5, 6]:
            problems.append(f"{sample_id}: R retention is not [5, 6]")
        for arm in ("O_oracle_top2_direct", "R_recent_top2_direct", "X_random_top2_direct"):
            if len(arms[arm]["retained_crop_indices"]) != 2:
                problems.append(f"{sample_id}/{arm}: retained count is not two")
            if arms[arm].get("input_image_count") != 3:
                problems.append(f"{sample_id}/{arm}: input image count is not three")
        if arms["F_full_direct"].get("input_image_count") != 7:
            problems.append(f"{sample_id}: F input image count is not seven")
        expected_oracle = retained_indices("O_oracle_top2_direct", sample_id, baseline_rows[sample_id])
        expected_random = retained_indices("X_random_top2_direct", sample_id, baseline_rows[sample_id])
        if arms["O_oracle_top2_direct"]["retained_crop_indices"] != expected_oracle:
            problems.append(f"{sample_id}: O retention differs from controller oracle")
        if arms["X_random_top2_direct"]["retained_crop_indices"] != expected_random:
            problems.append(f"{sample_id}: X retention differs from seed-42 plan")
        for row in rows:
            if row.get("generation_count") != 1 or row.get("tool_execution_count") != 0:
                problems.append(f"{sample_id}/{row['condition']}: not exactly one generation with zero tools")
            if row.get("direct_instruction_sha256") != surgery.sha256_json(DIRECT_ANSWER_INSTRUCTION):
                problems.append(f"{sample_id}/{row['condition']}: direct instruction hash mismatch")

        prefix_payload = json.loads((OUTPUT_ROOT / "frozen_prefixes" / f"{sample_id}.json").read_text(encoding="utf-8"))
        leaked = recursively_find_banned_keys(prefix_payload, banned)
        if leaked:
            problems.append(f"{sample_id}: sanitized prefix contains GT keys {sorted(leaked)}")
        for arm in ARMS:
            plan_payload = json.loads((output / "inference_plans" / arm / f"{sample_id}.json").read_text(encoding="utf-8"))
            leaked = recursively_find_banned_keys(plan_payload, banned)
            if leaked:
                problems.append(f"{sample_id}/{arm}: inference plan contains GT keys {sorted(leaked)}")

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    current = baseline_fingerprints()
    if config.get("baseline_fingerprints") != current:
        problems.append("Frozen baseline fingerprints changed")
    if any(row.get("status") == "error" for row in raw):
        problems.append("One or more direct-answer generations ended with an unexpected error")

    baseline.write_jsonl(output / "merged_results.jsonl", raw)
    write_csv(output / "merged_results.csv", raw, RESULT_FIELDS)
    write_csv(
        output / "oom_log.csv",
        [row for row in raw if row.get("oom")],
        ["sample_id", "condition", "error_type", "error_message"],
    )
    report = [
        "# Round-6 Direct-Answer Integrity Report", "",
        f"- Overall: **{'PASS' if not problems else 'FAIL'}**",
        f"- Primary samples: {len(ids)}",
        f"- Episode records: {len(raw)}",
        "- Four arms share the same frozen prefix, question, assistant history, role sequence, and direct-answer instruction.",
        "- F retains six prefix crops; O/R/X retain exactly two and receive original plus two retained images.",
        "- Each arm performs exactly one generation and zero tool executions.",
        "- Grounding output is recorded as a protocol violation and is never executed.",
        "- GPU workers never load GT answers, boxes, categories, coverage, or annotations.",
        "- Oracle selection uses GT coverage only during controller-side preparation and offline audit.",
        "- Baseline files are fingerprint checked and read only.", "",
        "## Problems",
        *(f"- {problem}" for problem in problems),
    ]
    if not problems:
        report.append("- None.")
    (output / "integrity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "samples": len(ids), "records": len(raw), "integrity": "PASS" if not problems else "FAIL"}, indent=2))


def validate_only() -> None:
    surgery.load_frozen_config()
    rows, bundles, episodes = surgery.build_cohort()
    ids = selected_ids("full", rows)
    if len(ids) != 88:
        print(f"WARNING: frozen primary cohort has {len(ids)} samples, not 88")
    for sample_id in ids:
        for arm in ARMS:
            indices = retained_indices(arm, sample_id, episodes[sample_id])
            expected = 6 if arm == "F_full_direct" else 2
            if len(indices) != expected or len(set(indices)) != expected:
                raise RuntimeError(f"Invalid retention plan for {sample_id}/{arm}: {indices}")
        if recursively_find_banned_keys(asdict(bundles[sample_id]), {"gt_answer", "gt_bbox", "gt_coverage"}):
            raise RuntimeError(f"GT leakage in sanitized bundle for {sample_id}")
    print(json.dumps({"primary_samples": len(ids), "validated_plans": len(ids) * len(ARMS), "created_output": False}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-shard", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--stage", choices=STAGES, default="full")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = sum((args.validate_only, args.prepare, args.run_shard, args.merge))
    if actions != 1:
        raise SystemExit("Choose exactly one of --validate-only, --prepare, --run-shard, or --merge")
    if args.validate_only:
        validate_only()
    elif args.prepare:
        prepare(args.stage)
    elif args.run_shard:
        run_shard(args.stage, args.shard_id, args.num_shards)
    else:
        merge(args.stage, args.num_shards)


if __name__ == "__main__":
    main()
