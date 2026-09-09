#!/usr/bin/env python3
"""Round-6 visual-memory surgery for the frozen VisualNeedle Mini-o3 baseline.

Preparation reads GT metadata to choose the oracle retention set, then writes a
sanitized frozen prefix and an inference plan containing only crop indices. GPU
workers never load annotations or evaluated baseline trajectories.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

import run_visualneedle_minio3_smoke20 as baseline
from run_visualneedle_minio3_smoke20_cache import CacheEnabledHFBackend


PROJECT_ROOT = Path("/home/songzhoujie/cvpr27/vstar")
BASELINE_DIR = PROJECT_ROOT / "outputs" / "visualneedle_minio3" / "full300_baseline"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "visualneedle_minio3" / "round6_memory_surgery_v1"
MODEL_PATH = Path("/mnt/data2/szj/models/Mini-o3-7B-v1-complete")
PREFIX_CROP_COUNT = 6
RETAINED_CROP_COUNT = 2
SEED = 42
DEV_SAMPLE_COUNT = 8
SMOKE_SAMPLE_COUNT = 10
DEV_RECONSTRUCTION_PASS_RATE = 0.80
NUM_SHARDS = 8
GT_HIT_THRESHOLD = 0.10
EVICTION_PLACEHOLDER = "[Visual observation evicted from working memory.]"
FORCE_ANSWER_INSTRUCTION = (
    "Do not call any more visual tools.\n"
    "Based on the visual evidence available so far, provide your best final answer now.\n"
    "Put the final answer inside <answer> and </answer>."
)
ARMS = (
    "A_full",
    "B_oracle_top2",
    "C_recent_top2",
    "D_force_answer_r6",
    "E_random_top2",
)
STAGES = ("devcheck", "smoke", "formal")
FATAL_PREFIX_ERRORS = {"oom", "runtime_error", "uncaught_episode_error"}


@dataclass(frozen=True)
class PrefixBundle:
    sample_id: str
    question: str
    original_image_path: str
    original_width: int
    original_height: int
    prefix_end_turn: int
    prefix_output_tokens: int
    turns: List[Dict[str, Any]]
    crops: List[Dict[str, Any]]
    original_image_sha256: str
    crop_image_sha256: List[str]
    raw_crop_image_sha256: List[str]
    assistant_text_sha256: str


@dataclass(frozen=True)
class InterventionPlan:
    sample_id: str
    condition: str
    retained_crop_indices: List[int]
    evicted_crop_indices: List[int]
    forced_answer: bool
    prefix_snapshot_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_config() -> Dict[str, Any]:
    config = json.loads((BASELINE_DIR / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_path": str(MODEL_PATH),
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "do_sample": False,
        "max_rounds": baseline.MAX_ROUNDS,
        "max_images_including_original": baseline.MAX_IMAGES,
        "max_new_tokens_per_round": baseline.MAX_NEW_TOKENS_PER_ROUND,
        "total_generation_budget": baseline.TOTAL_GENERATION_BUDGET,
        "max_time_per_round_seconds": baseline.ROUND_MAX_TIME_SECONDS,
        "min_pixels": baseline.MIN_PIXELS,
        "max_pixels": baseline.MAX_PIXELS,
        "use_cache": True,
        "system_prompt": baseline.SYSTEM_PROMPT,
        "observation_prompt_official": baseline.OFFICIAL_OBSERVATION_PROMPT,
        "error_prompt_official": baseline.ERROR_INFO_PROMPT,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Frozen baseline configuration mismatch: {mismatches}")
    return config


def load_baseline_episodes() -> List[Dict[str, Any]]:
    episodes = read_jsonl(BASELINE_DIR / "trajectories.jsonl")
    ids = [str(episode["sample_id"]) for episode in episodes]
    if len(episodes) != 300 or len(set(ids)) != 300:
        raise RuntimeError("Frozen baseline must contain exactly 300 unique episodes")
    return episodes


def baseline_summary_map() -> Dict[str, Dict[str, str]]:
    with (BASELINE_DIR / "episode_summary.csv").open(encoding="utf-8") as handle:
        return {str(row["sample_id"]): row for row in csv.DictReader(handle)}


def bool_field(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def sanitize_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "sample_id", "turn_index", "question", "action_type", "raw_model_output",
        "predicted_bbox_raw", "predicted_bbox_pixel", "predicted_bbox_clipped",
        "predicted_bbox_source_processed_pixel", "predicted_bbox_source_content_pixel",
        "bbox_parse_success", "bbox_valid", "bbox_was_clipped", "crop_executed",
        "crop_path", "raw_crop_path", "source", "error_type", "error_message",
        "input_tokens", "output_tokens", "max_new_tokens", "finish_reason",
        "runtime_seconds", "input_image_count", "image_grid_thw", "use_cache",
        "tokens_per_second", "original_crop_width", "original_crop_height",
        "processed_width", "processed_height", "padding_applied",
    }
    return {key: copy.deepcopy(value) for key, value in turn.items() if key in allowed}


def sanitize_crop(crop: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "sample_id", "turn_index", "crop_index", "source", "predicted_bbox_raw",
        "predicted_bbox_pixel", "predicted_bbox_clipped",
        "predicted_bbox_source_processed_pixel", "predicted_bbox_source_content_pixel",
        "bbox_was_clipped", "raw_crop_path", "crop_path", "original_crop_width",
        "original_crop_height", "processed_width", "processed_height", "padding_applied",
        "pad_left", "pad_top", "pad_right", "pad_bottom", "preprocess_scale_x",
        "preprocess_scale_y",
    }
    return {key: copy.deepcopy(value) for key, value in crop.items() if key in allowed}


def prefix_validation(episode: Dict[str, Any]) -> Tuple[List[str], Optional[PrefixBundle]]:
    reasons: List[str] = []
    if bool(episode.get("produced_final_answer")) or str(episode.get("final_answer") or "").strip():
        reasons.append("baseline_answer_present")
    crops = list(episode.get("crops") or [])
    if len(crops) < PREFIX_CROP_COUNT:
        reasons.append("fewer_than_6_successful_crops")
        return reasons, None
    prefix_crops = crops[:PREFIX_CROP_COUNT]
    prefix_end_turn = int(prefix_crops[-1]["turn_index"])
    prefix_turns = [turn for turn in episode.get("turns", []) if int(turn["turn_index"]) <= prefix_end_turn]
    if not prefix_turns or int(prefix_turns[-1]["turn_index"]) != prefix_end_turn:
        reasons.append("incomplete_prefix_turn_history")
    fatal = [turn for turn in prefix_turns if str(turn.get("error_type") or "") in FATAL_PREFIX_ERRORS]
    if fatal:
        reasons.append("fatal_error_before_prefix")
    if any("raw_model_output" not in turn or turn.get("raw_model_output") is None for turn in prefix_turns):
        reasons.append("missing_prefix_model_output")

    original_path = Path(str(episode.get("original_image_path") or ""))
    if not original_path.is_file():
        reasons.append("missing_original_image")
    missing_crop = [
        crop for crop in prefix_crops
        if not Path(str(crop.get("crop_path") or "")).is_file()
        or not Path(str(crop.get("raw_crop_path") or "")).is_file()
    ]
    if missing_crop:
        reasons.append("missing_prefix_crop_file")
    if reasons:
        return reasons, None

    clean_turns = [sanitize_turn(turn) for turn in prefix_turns]
    clean_crops = [sanitize_crop(crop) for crop in prefix_crops]
    assistant_texts = [str(turn.get("raw_model_output") or "") for turn in clean_turns]
    bundle = PrefixBundle(
        sample_id=str(episode["sample_id"]),
        question=str(episode["question"]),
        original_image_path=str(original_path),
        original_width=int(episode["original_width"]),
        original_height=int(episode["original_height"]),
        prefix_end_turn=prefix_end_turn,
        prefix_output_tokens=sum(int(turn.get("output_tokens") or 0) for turn in clean_turns),
        turns=clean_turns,
        crops=clean_crops,
        original_image_sha256=sha256_file(original_path),
        crop_image_sha256=[sha256_file(Path(crop["crop_path"])) for crop in clean_crops],
        raw_crop_image_sha256=[sha256_file(Path(crop["raw_crop_path"])) for crop in clean_crops],
        assistant_text_sha256=sha256_json(assistant_texts),
    )
    return reasons, bundle


def build_cohort() -> Tuple[List[Dict[str, Any]], Dict[str, PrefixBundle], Dict[str, Dict[str, Any]]]:
    load_frozen_config()
    summaries = baseline_summary_map()
    rows: List[Dict[str, Any]] = []
    bundles: Dict[str, PrefixBundle] = {}
    episodes = {str(ep["sample_id"]): ep for ep in load_baseline_episodes()}
    for sample_id, episode in episodes.items():
        reasons, bundle = prefix_validation(episode)
        prefix_crops = list(episode.get("crops") or [])[:PREFIX_CROP_COUNT]
        coverages = [float(crop.get("gt_coverage") or 0.0) for crop in prefix_crops]
        hit_indices = [index + 1 for index, value in enumerate(coverages) if value >= GT_HIT_THRESHOLD]
        eligible = not reasons and bundle is not None
        cohort = "primary_prefix_hit" if eligible and hit_indices else "secondary_prefix_no_hit" if eligible else "excluded"
        summary = summaries[sample_id]
        row = {
            "sample_id": sample_id,
            "baseline_status": str(episode.get("status") or ""),
            "baseline_answer_present": bool(episode.get("final_answer")),
            "baseline_correct": bool_field(summary.get("official_final_correct")),
            "num_prefix_crops": len(prefix_crops),
            "prefix_gt_hit": bool(hit_indices),
            "max_prefix_gt_coverage": max(coverages, default=0.0),
            "first_gt_hit_crop_index": hit_indices[0] if hit_indices else "",
            "eligible": eligible,
            "cohort": cohort,
            "exclusion_reason": ";".join(reasons),
        }
        rows.append(row)
        if bundle is not None and eligible:
            bundles[sample_id] = bundle
    return rows, bundles, episodes


def retained_indices(arm: str, sample_id: str, episode: Dict[str, Any]) -> List[int]:
    all_indices = list(range(1, PREFIX_CROP_COUNT + 1))
    if arm in {"A_full", "D_force_answer_r6"}:
        return all_indices
    if arm == "B_oracle_top2":
        ranked = sorted(
            all_indices,
            key=lambda index: (-float(episode["crops"][index - 1].get("gt_coverage") or 0.0), index),
        )
        return sorted(ranked[:RETAINED_CROP_COUNT])
    if arm == "C_recent_top2":
        return [PREFIX_CROP_COUNT - 1, PREFIX_CROP_COUNT]
    if arm == "E_random_top2":
        seed_bytes = hashlib.sha256(f"{SEED}:{sample_id}".encode("utf-8")).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, "big"))
        return sorted(rng.sample(all_indices, RETAINED_CROP_COUNT))
    raise ValueError(f"Unknown arm: {arm}")


def stage_directory(stage: str) -> Path:
    return OUTPUT_ROOT if stage == "formal" else OUTPUT_ROOT / stage


def stage_sample_ids(stage: str, cohort_rows: Sequence[Dict[str, Any]]) -> List[str]:
    eligible = sorted(str(row["sample_id"]) for row in cohort_rows if row["eligible"])
    primary = sorted(str(row["sample_id"]) for row in cohort_rows if row["cohort"] == "primary_prefix_hit")
    rng = random.Random(SEED)
    if stage == "devcheck":
        return sorted(rng.sample(eligible, min(DEV_SAMPLE_COUNT, len(eligible))))
    if stage == "smoke":
        return sorted(rng.sample(primary, min(SMOKE_SAMPLE_COUNT, len(primary))))
    return eligible


def prepare(stage: str) -> None:
    cohort_rows, bundles, episodes = build_cohort()
    selected = stage_sample_ids(stage, cohort_rows)
    stage_dir = stage_directory(stage)
    prefix_dir = OUTPUT_ROOT / "frozen_prefixes"
    plan_dir = stage_dir / "inference_plans"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "cohort_selection.csv", cohort_rows)
    exclusion_counts: Dict[str, int] = {}
    for row in cohort_rows:
        for reason in str(row["exclusion_reason"]).split(";"):
            if reason:
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
    baseline.atomic_write_json(OUTPUT_ROOT / "cohort_summary.json", {
        "all_baseline_samples": len(cohort_rows),
        "all_eligible_no_answer": sum(bool(row["eligible"]) for row in cohort_rows),
        "primary_prefix_hit": sum(row["cohort"] == "primary_prefix_hit" for row in cohort_rows),
        "secondary_prefix_no_hit": sum(row["cohort"] == "secondary_prefix_no_hit" for row in cohort_rows),
        "excluded": sum(row["cohort"] == "excluded" for row in cohort_rows),
        "exclusion_reason_counts_not_mutually_exclusive": exclusion_counts,
    })
    for sample_id in selected:
        payload = asdict(bundles[sample_id])
        baseline.atomic_write_json(prefix_dir / f"{sample_id}.json", payload)
        snapshot_hash = sha256_json(payload)
        for arm in ARMS:
            if stage == "devcheck" and arm != "A_full":
                continue
            retained = retained_indices(arm, sample_id, episodes[sample_id])
            plan = InterventionPlan(
                sample_id=sample_id,
                condition=arm,
                retained_crop_indices=retained,
                evicted_crop_indices=[index for index in range(1, PREFIX_CROP_COUNT + 1) if index not in retained],
                forced_answer=arm == "D_force_answer_r6",
                prefix_snapshot_sha256=snapshot_hash,
            )
            arm_plan_dir = plan_dir / arm
            arm_plan_dir.mkdir(parents=True, exist_ok=True)
            baseline.atomic_write_json(arm_plan_dir / f"{sample_id}.json", asdict(plan))
    config = {
        "experiment": "Round-6 Visual Memory Surgery Pilot",
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "baseline_directory": str(BASELINE_DIR),
        "baseline_fingerprints": {
            name: sha256_file(BASELINE_DIR / name)
            for name in ("config.json", "trajectories.jsonl", "episode_summary.csv", "official_judge_results.csv")
        },
        "model_path": str(MODEL_PATH),
        "python": "/mnt/data2/szj/envs/venv_e2m/bin/python",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "use_cache": True,
        "do_sample": False,
        "max_rounds": baseline.MAX_ROUNDS,
        "max_images_including_original": baseline.MAX_IMAGES,
        "max_new_tokens_per_round": baseline.MAX_NEW_TOKENS_PER_ROUND,
        "total_generation_budget": baseline.TOTAL_GENERATION_BUDGET,
        "max_time_per_round_seconds": baseline.ROUND_MAX_TIME_SECONDS,
        "min_pixels": baseline.MIN_PIXELS,
        "max_pixels": baseline.MAX_PIXELS,
        "prefix_crop_count": PREFIX_CROP_COUNT,
        "intervention_point": "after sixth successful crop observation and before next generation",
        "retained_crop_count_B_C_E": RETAINED_CROP_COUNT,
        "eviction_placeholder": EVICTION_PLACEHOLDER,
        "force_answer_instruction": FORCE_ANSWER_INSTRUCTION,
        "random_seed": SEED,
        "devcheck_min_match_rate": DEV_RECONSTRUCTION_PASS_RATE,
        "selected_sample_ids": selected,
        "arms": ["A_full"] if stage == "devcheck" else list(ARMS),
        "ground_truth_isolation": (
            "GPU workers load sanitized frozen_prefixes and index-only inference_plans; "
            "annotations, answers, categories, GT boxes, and coverage values are not loaded"
        ),
    }
    baseline.atomic_write_json(stage_dir / "config.json", config)
    (stage_dir / "sample_ids.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "samples": len(selected), "arms": config["arms"]}, indent=2))


def load_prefix(sample_id: str) -> PrefixBundle:
    path = OUTPUT_ROOT / "frozen_prefixes" / f"{sample_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PrefixBundle(**payload)


def load_plan(stage: str, arm: str, sample_id: str) -> InterventionPlan:
    path = stage_directory(stage) / "inference_plans" / arm / f"{sample_id}.json"
    return InterventionPlan(**json.loads(path.read_text(encoding="utf-8")))


def append_evicted_observation(
    messages: List[Dict[str, Any]], raw_output: str, action_turn: int, observation_turn: int
) -> None:
    prefix, suffix = baseline.prompt_parts(baseline.OFFICIAL_OBSERVATION_PROMPT, action_turn, observation_turn)
    messages.extend([
        {"role": "assistant", "content": raw_output},
        {"role": "user", "content": [
            {"type": "text", "text": prefix},
            {"type": "text", "text": EVICTION_PLACEHOLDER},
            {"type": "text", "text": suffix},
        ]},
    ])


def observation_from_crop(crop: Dict[str, Any], image: Image.Image) -> baseline.Observation:
    return baseline.Observation(
        source_name=f"observation_{int(crop['crop_index'])}",
        original_bbox=tuple(int(value) for value in crop["predicted_bbox_clipped"]),
        processed_image=image,
        raw_width=int(crop["original_crop_width"]),
        raw_height=int(crop["original_crop_height"]),
        scale_x=float(crop["preprocess_scale_x"]),
        scale_y=float(crop["preprocess_scale_y"]),
        pad_left=int(crop["pad_left"]),
        pad_top=int(crop["pad_top"]),
    )


def reconstruct_messages(
    prefix: PrefixBundle, plan: InterventionPlan
) -> Tuple[Image.Image, List[baseline.Observation], List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload = asdict(prefix)
    if sha256_json(payload) != plan.prefix_snapshot_sha256:
        raise RuntimeError("Prefix snapshot hash differs from the prepared inference plan")
    original_path = Path(prefix.original_image_path)
    if sha256_file(original_path) != prefix.original_image_sha256:
        raise RuntimeError(f"Original image changed: {original_path}")
    with Image.open(original_path) as handle:
        original = handle.convert("RGB")
    original_processed, original_prep = baseline.process_for_model(original)
    observations = [baseline.Observation(
        source_name="original_image",
        original_bbox=(0, 0, original.width, original.height),
        processed_image=original_processed,
        raw_width=original.width,
        raw_height=original.height,
        scale_x=float(original_prep["scale_x"]),
        scale_y=float(original_prep["scale_y"]),
        pad_left=int(original_prep["pad_left"]),
        pad_top=int(original_prep["pad_top"]),
    )]
    messages = baseline.initial_messages(original_processed, prefix.question)
    crop_by_turn = {int(crop["turn_index"]): crop for crop in prefix.crops}
    prefix_images: List[Image.Image] = []
    assistant_texts: List[str] = []
    for turn in prefix.turns:
        turn_index = int(turn["turn_index"])
        raw_output = str(turn.get("raw_model_output") or "")
        assistant_texts.append(raw_output)
        crop = crop_by_turn.get(turn_index)
        if crop is not None:
            crop_index = int(crop["crop_index"])
            crop_path = Path(crop["crop_path"])
            raw_crop_path = Path(crop["raw_crop_path"])
            if sha256_file(crop_path) != prefix.crop_image_sha256[crop_index - 1]:
                raise RuntimeError(f"Frozen crop changed: {crop_path}")
            if sha256_file(raw_crop_path) != prefix.raw_crop_image_sha256[crop_index - 1]:
                raise RuntimeError(f"Frozen raw crop changed: {raw_crop_path}")
            with Image.open(crop_path) as handle:
                crop_image = handle.convert("RGB")
            prefix_images.append(crop_image)
            observations.append(observation_from_crop(crop, crop_image))
            if crop_index in plan.retained_crop_indices:
                baseline.append_observation_message(
                    messages, raw_output, crop_image, action_turn=turn_index - 1, observation_turn=turn_index
                )
            else:
                append_evicted_observation(messages, raw_output, turn_index - 1, turn_index)
        else:
            error = str(turn.get("error_message") or "")
            baseline.append_official_error_message(messages, raw_output, error)
    if sha256_json(assistant_texts) != prefix.assistant_text_sha256:
        raise RuntimeError("Assistant prefix text changed during reconstruction")
    if plan.forced_answer:
        content = messages[-1].get("content")
        if not isinstance(content, list):
            raise RuntimeError("Force-answer intervention expected a crop observation user message")
        content.append({"type": "text", "text": "\n" + FORCE_ANSWER_INSTRUCTION})
    return original, observations, messages, prefix_images


class SurgeryBackend(CacheEnabledHFBackend):
    def generate(self, messages: Sequence[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        result = super().generate(messages, max_new_tokens)
        grids = result.get("image_grid_thw")
        merge_size = getattr(getattr(self.processor, "image_processor", None), "merge_size", None)
        per_image: Optional[List[int]] = None
        if grids is not None and merge_size:
            divisor = int(merge_size) ** 2
            per_image = [int(grid[0] * grid[1] * grid[2] // divisor) for grid in grids]
        result["visual_tokens_per_image"] = per_image
        result["visual_tokens"] = sum(per_image) if per_image is not None else None
        return result


def empty_turn(sample_id: str, question: str, turn_index: int) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "turn_index": turn_index,
        "question": question,
        "action_type": "invalid",
        "raw_model_output": "",
        "predicted_bbox_raw": None,
        "predicted_bbox_pixel": None,
        "predicted_bbox_clipped": None,
        "predicted_bbox_source_processed_pixel": None,
        "predicted_bbox_source_content_pixel": None,
        "bbox_parse_success": False,
        "bbox_valid": False,
        "bbox_was_clipped": False,
        "crop_executed": False,
        "crop_path": "",
        "raw_crop_path": "",
        "source": "",
        "error_type": "",
        "error_message": "",
        "post_intervention": True,
    }


def run_continuation(
    backend: SurgeryBackend, prefix: PrefixBundle, plan: InterventionPlan, output_dir: Path
) -> Dict[str, Any]:
    started = time.perf_counter()
    backend.torch.cuda.reset_peak_memory_stats()
    original, observations, messages, _ = reconstruct_messages(prefix, plan)
    prefix_crops = copy.deepcopy(prefix.crops)
    post_turns: List[Dict[str, Any]] = []
    post_crops: List[Dict[str, Any]] = []
    final_answer = ""
    total_output_tokens = prefix.prefix_output_tokens
    stop_reason = ""
    status = "completed"
    reached_turn_cap = False
    reached_image_cap = False
    error_type = ""
    error_message = ""
    first_generation: Optional[Dict[str, Any]] = None
    crop_directory = output_dir / "crops" / prefix.sample_id
    crop_directory.mkdir(parents=True, exist_ok=True)

    next_turn = prefix.prefix_end_turn + 1
    final_turn_limit = min(baseline.MAX_ROUNDS, next_turn) if plan.forced_answer else baseline.MAX_ROUNDS
    for round_index in range(next_turn, final_turn_limit + 1):
        remaining = baseline.TOTAL_GENERATION_BUDGET - total_output_tokens
        if remaining <= 0:
            stop_reason = "total_generation_budget"
            break
        turn_started = time.perf_counter()
        turn = empty_turn(prefix.sample_id, prefix.question, round_index)
        try:
            generation = backend.generate(messages, min(baseline.MAX_NEW_TOKENS_PER_ROUND, remaining))
            turn.update(generation)
            if first_generation is None:
                first_generation = copy.deepcopy(generation)
            total_output_tokens += int(generation["output_tokens"])
            raw_output = str(generation["raw_model_output"])
            grounding = baseline.parse_grounding(raw_output)
            answer = baseline.parse_answer(raw_output)

            if plan.forced_answer:
                if answer:
                    turn.update({"action_type": "answer", "final_answer": answer})
                    final_answer = answer
                    stop_reason = "forced_answer"
                elif grounding["present"]:
                    turn.update({
                        "action_type": "forbidden_tool_call",
                        "error_type": "tool_call_after_force_answer",
                        "error_message": "Force-answer arm prohibits post-intervention visual tool calls",
                    })
                    status = "invalid"
                    stop_reason = "tool_call_after_force_answer"
                else:
                    turn.update({
                        "action_type": "invalid",
                        "error_type": "unparsed_forced_answer",
                        "error_message": "Forced output did not contain a complete answer tag",
                    })
                    status = "invalid"
                    stop_reason = "unparsed_forced_answer"
                post_turns.append(turn)
                break

            if grounding["present"]:
                turn["action_type"] = "crop" if grounding["success"] else "invalid"
                turn["bbox_parse_success"] = bool(grounding["success"])
                if not grounding["success"]:
                    turn["error_type"] = "bbox_parse_error"
                    turn["error_message"] = grounding["error"]
                    post_turns.append(turn)
                    baseline.append_official_error_message(messages, raw_output, str(grounding["error"]))
                    continue
                turn["predicted_bbox_raw"] = grounding["bbox"]
                turn["source"] = grounding["source"]
                if len(observations) >= baseline.MAX_IMAGES or round_index == baseline.MAX_ROUNDS:
                    reached_image_cap = len(observations) >= baseline.MAX_IMAGES
                    reached_turn_cap = round_index == baseline.MAX_ROUNDS
                    turn["error_type"] = "budget_cap_before_crop"
                    turn["error_message"] = "Official policy does not execute a crop after the image/round cap"
                    post_turns.append(turn)
                    stop_reason = "image_or_turn_cap"
                    break
                try:
                    _, source_observation = baseline.resolve_source(str(grounding["source"]), observations)
                    converted = baseline.convert_bbox(grounding["bbox"], source_observation, original.size)
                    turn["predicted_bbox_source_processed_pixel"] = converted["source_processed_pixel"]
                    turn["predicted_bbox_source_content_pixel"] = converted["source_content_pixel"]
                    turn["predicted_bbox_pixel"] = converted["pixel_original"]
                    turn["predicted_bbox_clipped"] = converted["clipped_original"]
                    turn["bbox_was_clipped"] = bool(converted["was_clipped"])
                    turn["bbox_valid"] = True
                    crop_number = len(prefix_crops) + len(post_crops) + 1
                    bbox = tuple(int(value) for value in converted["clipped_original"])
                    raw_crop = original.crop(bbox).convert("RGB")
                    raw_path = crop_directory / f"turn_{round_index:02d}_raw.png"
                    processed_path = crop_directory / f"turn_{round_index:02d}_processed.png"
                    raw_crop.save(raw_path)
                    processed_crop, prep = baseline.process_for_model(raw_crop)
                    processed_crop.save(processed_path)
                    crop = {
                        "sample_id": prefix.sample_id,
                        "turn_index": round_index,
                        "crop_index": crop_number,
                        "source": grounding["source"],
                        "predicted_bbox_raw": grounding["bbox"],
                        "predicted_bbox_pixel": converted["pixel_original"],
                        "predicted_bbox_clipped": list(bbox),
                        "predicted_bbox_source_processed_pixel": converted["source_processed_pixel"],
                        "predicted_bbox_source_content_pixel": converted["source_content_pixel"],
                        "bbox_was_clipped": bool(converted["was_clipped"]),
                        "raw_crop_path": str(raw_path),
                        "crop_path": str(processed_path),
                        "original_crop_width": raw_crop.width,
                        "original_crop_height": raw_crop.height,
                        "processed_width": processed_crop.width,
                        "processed_height": processed_crop.height,
                        "padding_applied": bool(prep["padding_applied"]),
                        "pad_left": prep["pad_left"], "pad_top": prep["pad_top"],
                        "pad_right": prep["pad_right"], "pad_bottom": prep["pad_bottom"],
                        "preprocess_scale_x": prep["scale_x"], "preprocess_scale_y": prep["scale_y"],
                        "post_intervention": True,
                    }
                    post_crops.append(crop)
                    observations.append(observation_from_crop(crop, processed_crop))
                    turn.update({
                        "crop_executed": True, "crop_path": str(processed_path),
                        "raw_crop_path": str(raw_path), "original_crop_width": raw_crop.width,
                        "original_crop_height": raw_crop.height, "processed_width": processed_crop.width,
                        "processed_height": processed_crop.height, "padding_applied": bool(prep["padding_applied"]),
                    })
                    post_turns.append(turn)
                    baseline.append_observation_message(
                        messages, raw_output, processed_crop,
                        action_turn=round_index - 1, observation_turn=round_index,
                    )
                except Exception as exc:
                    turn.update({
                        "action_type": "invalid", "bbox_valid": False,
                        "error_type": "crop_execution_error", "error_message": str(exc),
                    })
                    post_turns.append(turn)
                    baseline.append_official_error_message(messages, raw_output, str(exc))
                continue
            if answer:
                turn.update({"action_type": "answer", "final_answer": answer})
                post_turns.append(turn)
                final_answer = answer
                stop_reason = "answer"
                break
            turn.update({
                "action_type": "invalid", "error_type": "unparsed_model_output",
                "error_message": "Output contains neither a complete grounding tag nor a complete answer tag",
            })
            post_turns.append(turn)
            status = "invalid"
            stop_reason = "unparsed_model_output"
            break
        except RuntimeError as exc:
            message = str(exc)
            error_type = "oom" if "out of memory" in message.lower() else "runtime_error"
            status = "oom" if error_type == "oom" else "error"
            error_message = message
            turn.update({
                "error_type": error_type, "error_message": message,
                "runtime_seconds": time.perf_counter() - turn_started,
            })
            post_turns.append(turn)
            stop_reason = error_type
            break
        except Exception as exc:
            status, error_type, error_message, stop_reason = "error", type(exc).__name__, str(exc), "error"
            turn.update({
                "error_type": error_type, "error_message": error_message,
                "traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter() - turn_started,
            })
            post_turns.append(turn)
            break
    else:
        if not final_answer and not plan.forced_answer:
            reached_turn_cap = True
            stop_reason = "turn_cap"

    all_crops = prefix_crops + post_crops
    first_visual_per_image = first_generation.get("visual_tokens_per_image") if first_generation else None
    prefix_visual_tokens = (
        sum(first_visual_per_image[1 : 1 + len(plan.retained_crop_indices)])
        if first_visual_per_image is not None else None
    )
    return {
        "sample_id": prefix.sample_id,
        "condition": plan.condition,
        "question": prefix.question,
        "original_image_path": prefix.original_image_path,
        "original_width": prefix.original_width,
        "original_height": prefix.original_height,
        "episode_finished": True,
        "status": status,
        "stop_reason": stop_reason,
        "error_type": error_type,
        "error_message": error_message,
        "prefix_crop_count": PREFIX_CROP_COUNT,
        "prefix_end_turn": prefix.prefix_end_turn,
        "prefix_snapshot_sha256": plan.prefix_snapshot_sha256,
        "original_image_sha256": prefix.original_image_sha256,
        "prefix_crop_sha256": prefix.crop_image_sha256,
        "prefix_raw_crop_sha256": prefix.raw_crop_image_sha256,
        "assistant_text_sha256": prefix.assistant_text_sha256,
        "retained_crop_indices": plan.retained_crop_indices,
        "evicted_crop_indices": plan.evicted_crop_indices,
        "retained_prefix_image_count": len(plan.retained_crop_indices),
        "evicted_prefix_image_count": len(plan.evicted_crop_indices),
        "forced_answer": plan.forced_answer,
        "post_intervention_answer_present": bool(final_answer),
        "post_intervention_final_answer": final_answer,
        "final_answer": final_answer,
        "post_intervention_status": status,
        "num_additional_crops": len(post_crops),
        "total_final_crops": len(all_crops),
        "natural_stop": bool(final_answer) and not plan.forced_answer,
        "final_round": int(post_turns[-1]["turn_index"]) if post_turns else prefix.prefix_end_turn,
        "reached_max_rounds": reached_turn_cap,
        "reached_max_images": reached_image_cap,
        "input_tokens_first_post_intervention": first_generation.get("input_tokens") if first_generation else None,
        "visual_tokens_first_post_intervention": first_generation.get("visual_tokens") if first_generation else None,
        "prefix_visual_tokens_after_intervention": prefix_visual_tokens,
        "visual_tokens_per_image_first_post_intervention": first_visual_per_image,
        "input_images_first_post_intervention": first_generation.get("input_image_count") if first_generation else None,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": backend.torch.cuda.max_memory_allocated() / (1024 * 1024),
        "oom": status == "oom",
        "invalid": status == "invalid",
        "total_output_tokens": total_output_tokens,
        "prefix_turns": prefix.turns,
        "post_intervention_turns": post_turns,
        "turns": prefix.turns + post_turns,
        "prefix_crops": prefix_crops,
        "post_intervention_crops": post_crops,
        "crops": all_crops,
    }


def complete_record(path: Path, sample_id: str, arm: str, prefix_hash: str) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = {
        "episode_finished", "sample_id", "condition", "status", "stop_reason",
        "retained_crop_indices", "evicted_crop_indices", "post_intervention_turns",
        "prefix_snapshot_sha256",
    }
    return (
        required.issubset(row)
        and row["episode_finished"] is True
        and row["status"] in {"completed", "invalid", "oom"}
        and row["sample_id"] == sample_id
        and row["condition"] == arm
        and row["prefix_snapshot_sha256"] == prefix_hash
    )


def run_shard(stage: str, arm: str, shard_id: int, num_shards: int) -> None:
    stage_dir = stage_directory(stage)
    ids = [line for line in (stage_dir / "sample_ids.txt").read_text(encoding="utf-8").splitlines() if line]
    assigned = [sample_id for index, sample_id in enumerate(ids) if index % num_shards == shard_id]
    shard_dir = stage_dir / arm / "shards" / f"shard_{shard_id}"
    raw_dir = shard_dir / "raw_episodes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    backend = SurgeryBackend(MODEL_PATH)
    log = [f"stage={stage}", f"arm={arm}", f"shard={shard_id}/{num_shards}", f"gpu={backend.device_name}"]
    for position, sample_id in enumerate(assigned, start=1):
        prefix = load_prefix(sample_id)
        plan = load_plan(stage, arm, sample_id)
        path = raw_dir / f"{sample_id}.json"
        if path.exists() and complete_record(path, sample_id, arm, plan.prefix_snapshot_sha256):
            log.append(f"resume_skip {position}/{len(assigned)} {sample_id}")
            continue
        log.append(f"start {position}/{len(assigned)} {sample_id} {time.strftime('%F %T')}")
        try:
            result = run_continuation(backend, prefix, plan, stage_dir / arm)
        except Exception as exc:
            result = {
                "sample_id": sample_id, "condition": arm, "episode_finished": True,
                "status": "error", "stop_reason": "uncaught_episode_error",
                "error_type": type(exc).__name__, "error_message": str(exc),
                "traceback": traceback.format_exc(), "prefix_snapshot_sha256": plan.prefix_snapshot_sha256,
                "retained_crop_indices": plan.retained_crop_indices,
                "evicted_crop_indices": plan.evicted_crop_indices,
                "post_intervention_turns": [], "crops": [],
            }
        baseline.atomic_write_json(path, result)
        log.append(
            f"done {sample_id} status={result['status']} answer={result.get('post_intervention_answer_present')} "
            f"additional_crops={result.get('num_additional_crops', 0)}"
        )
        (shard_dir / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    shard_episodes = []
    for sample_id in assigned:
        path = raw_dir / f"{sample_id}.json"
        if path.exists():
            shard_episodes.append(json.loads(path.read_text(encoding="utf-8")))
    baseline.write_jsonl(shard_dir / "continuation_trajectories.jsonl", shard_episodes)
    shard_fields = [
        "sample_id", "condition", "status", "stop_reason", "post_intervention_answer_present",
        "num_additional_crops", "final_round", "runtime_seconds", "peak_gpu_memory_mb",
        "oom", "invalid", "error_type", "error_message",
    ]
    write_csv(shard_dir / "episode_records.csv", [{
        field: csv_value(episode.get(field, "")) for field in shard_fields
    } for episode in shard_episodes], shard_fields)
    turn_rows = []
    for episode in shard_episodes:
        for turn in episode.get("post_intervention_turns", []):
            turn_rows.append({
                "sample_id": episode["sample_id"], "condition": arm,
                **{key: csv_value(value) for key, value in turn.items()},
            })
    turn_fields = sorted({key for row in turn_rows for key in row}) if turn_rows else ["sample_id", "condition"]
    write_csv(shard_dir / "turn_records.csv", turn_rows, turn_fields)
    write_csv(shard_dir / "oom_log.csv", [{
        "sample_id": episode["sample_id"], "condition": arm,
        "error_message": episode.get("error_message", ""),
    } for episode in shard_episodes if episode.get("status") == "oom"],
    ("sample_id", "condition", "error_message"))


def evaluate_result(result: Dict[str, Any], metadata: baseline.EvaluationMetadata) -> Dict[str, Any]:
    evaluated = baseline.evaluate_episode(result, metadata)
    summary = baseline_summary_map()[str(result["sample_id"])]
    all_crops = list(evaluated.get("crops") or [])
    prefix_crops = all_crops[:PREFIX_CROP_COUNT]
    evaluated["prefix_crops"] = prefix_crops
    evaluated["post_intervention_crops"] = all_crops[PREFIX_CROP_COUNT:]
    coverages = [float(crop.get("gt_coverage") or 0.0) for crop in prefix_crops]
    evaluated.update({
        "baseline_status": summary.get("status", ""),
        "baseline_answer_present": bool(str(summary.get("final_answer") or "").strip()),
        "baseline_correct": bool_field(summary.get("official_final_correct")),
        "prefix_gt_hit": any(value >= GT_HIT_THRESHOLD for value in coverages),
        "max_prefix_gt_coverage": max(coverages, default=0.0),
        "retained_gt_coverages": [coverages[index - 1] for index in evaluated["retained_crop_indices"]],
        "post_intervention_correct_exact": bool(evaluated.get("exact_match")),
        "post_intervention_correct_normalized": bool(evaluated.get("normalized_string_match")),
    })
    return evaluated


EPISODE_FIELDS = [
    "sample_id", "condition", "category", "baseline_answer_present", "baseline_correct", "baseline_status",
    "prefix_crop_count", "prefix_gt_hit", "max_prefix_gt_coverage", "retained_crop_indices",
    "evicted_crop_indices", "retained_gt_coverages", "retained_prefix_image_count",
    "evicted_prefix_image_count", "post_intervention_answer_present", "post_intervention_final_answer",
    "post_intervention_status", "post_intervention_correct_exact", "post_intervention_correct_normalized",
    "num_additional_crops", "total_final_crops", "natural_stop", "forced_answer", "final_round",
    "reached_max_rounds", "reached_max_images", "input_tokens_first_post_intervention",
    "visual_tokens_first_post_intervention", "visual_tokens_per_image_first_post_intervention",
    "prefix_visual_tokens_after_intervention",
    "input_images_first_post_intervention", "runtime_seconds", "peak_gpu_memory_mb", "oom", "invalid",
    "stop_reason", "error_type", "error_message", "prefix_snapshot_sha256", "original_image_sha256",
    "assistant_text_sha256",
]


def csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def merge_arm(stage: str, arm: str, num_shards: int) -> None:
    stage_dir = stage_directory(stage)
    ids = [line for line in (stage_dir / "sample_ids.txt").read_text(encoding="utf-8").splitlines() if line]
    raw_results = []
    for index, sample_id in enumerate(ids):
        path = stage_dir / arm / "shards" / f"shard_{index % num_shards}" / "raw_episodes" / f"{sample_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        plan = load_plan(stage, arm, sample_id)
        if not complete_record(path, sample_id, arm, plan.prefix_snapshot_sha256):
            raise RuntimeError(f"Incomplete/error shard record must be rerun before merge: {path}")
        raw_results.append(result)
    metadata = baseline.load_evaluation_metadata()
    evaluated = [evaluate_result(result, metadata[str(result["sample_id"])] ) for result in raw_results]
    arm_dir = stage_dir / arm
    baseline.write_jsonl(arm_dir / "continuation_trajectories.jsonl", evaluated)
    write_csv(
        arm_dir / "episode_results.csv",
        [{field: csv_value(result.get(field, "")) for field in EPISODE_FIELDS} for result in evaluated],
        EPISODE_FIELDS,
    )
    print(json.dumps({"stage": stage, "arm": arm, "episodes": len(evaluated)}, indent=2))


def action_signature(turn: Dict[str, Any]) -> Tuple[Any, ...]:
    bbox = turn.get("predicted_bbox_clipped")
    return (turn.get("action_type"), turn.get("source"), tuple(bbox) if bbox else None)


def reconstruction_devcheck(stage: str, episodes: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    baseline_by_id = {str(ep["sample_id"]): ep for ep in load_baseline_episodes()}
    rows = []
    for episode in episodes:
        frozen = baseline_by_id[str(episode["sample_id"])]
        end_turn = int(episode["prefix_end_turn"])
        baseline_suffix = [turn for turn in frozen.get("turns", []) if int(turn["turn_index"]) > end_turn]
        replay_suffix = list(episode.get("post_intervention_turns") or [])
        baseline_next = baseline_suffix[0] if baseline_suffix else {}
        replay_next = replay_suffix[0] if replay_suffix else {}
        row = {
            "sample_id": episode["sample_id"],
            "next_output_exact_match": bool(baseline_next) and baseline_next.get("raw_model_output") == replay_next.get("raw_model_output"),
            "next_action_match": bool(baseline_next) and baseline_next.get("action_type") == replay_next.get("action_type"),
            "next_bbox_match": baseline_next.get("predicted_bbox_clipped") == replay_next.get("predicted_bbox_clipped"),
            "final_answer_match": str(frozen.get("final_answer") or "") == str(episode.get("final_answer") or ""),
            "later_crop_count_match": max(0, len(frozen.get("crops", [])) - PREFIX_CROP_COUNT) == int(episode.get("num_additional_crops") or 0),
            "suffix_behavior_match": [action_signature(t) for t in baseline_suffix] == [action_signature(t) for t in replay_suffix],
        }
        rows.append(row)
    metrics = (
        "next_output_exact_match", "next_action_match", "next_bbox_match",
        "final_answer_match", "later_crop_count_match", "suffix_behavior_match",
    )
    rates = {key: sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0 for key in metrics}
    passed = bool(rows) and all(value >= DEV_RECONSTRUCTION_PASS_RATE for value in rates.values())
    baseline.atomic_write_json(OUTPUT_ROOT / "reconstruction_devcheck_summary.json", {
        "num_samples": len(rows), "minimum_required_match_rate": DEV_RECONSTRUCTION_PASS_RATE,
        "match_rates": rates, "passed": passed,
    })
    return rows, passed


def merge_stage(stage: str) -> None:
    stage_dir = stage_directory(stage)
    arms = ["A_full"] if stage == "devcheck" else list(ARMS)
    episodes: List[Dict[str, Any]] = []
    for arm in arms:
        episodes.extend(read_jsonl(stage_dir / arm / "continuation_trajectories.jsonl"))
    baseline.write_jsonl(stage_dir / "merged_trajectories.jsonl", episodes)
    write_csv(
        stage_dir / "merged_episode_results.csv",
        [{field: csv_value(ep.get(field, "")) for field in EPISODE_FIELDS} for ep in episodes],
        EPISODE_FIELDS,
    )
    details = [{
        "sample_id": ep["sample_id"], "condition": ep["condition"],
        "retained_crop_indices": json.dumps(ep["retained_crop_indices"]),
        "evicted_crop_indices": json.dumps(ep["evicted_crop_indices"]),
        "retained_gt_coverages": json.dumps(ep["retained_gt_coverages"]),
        "retained_prefix_image_count": ep["retained_prefix_image_count"],
        "evicted_prefix_image_count": ep["evicted_prefix_image_count"],
        "input_tokens_first_post_intervention": ep.get("input_tokens_first_post_intervention"),
        "visual_tokens_first_post_intervention": ep.get("visual_tokens_first_post_intervention"),
        "prefix_visual_tokens_after_intervention": ep.get("prefix_visual_tokens_after_intervention"),
        "prefix_snapshot_sha256": ep["prefix_snapshot_sha256"],
    } for ep in episodes]
    write_csv(stage_dir / "intervention_details.csv", details)

    problems: List[str] = []
    by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for episode in episodes:
        by_sample.setdefault(str(episode["sample_id"]), []).append(episode)
    for sample_id, sample_episodes in by_sample.items():
        if len(sample_episodes) != len(arms):
            problems.append(f"{sample_id}: expected {len(arms)} arms, found {len(sample_episodes)}")
            continue
        if len({ep["prefix_snapshot_sha256"] for ep in sample_episodes}) != 1:
            problems.append(f"{sample_id}: arms do not share one frozen prefix")
        if len({ep["original_image_sha256"] for ep in sample_episodes}) != 1:
            problems.append(f"{sample_id}: original image hash differs across arms")
        if len({ep["assistant_text_sha256"] for ep in sample_episodes}) != 1:
            problems.append(f"{sample_id}: assistant textual prefix differs across arms")
        for arm in ("B_oracle_top2", "C_recent_top2", "E_random_top2"):
            if arm in arms:
                ep = next(item for item in sample_episodes if item["condition"] == arm)
                if ep["retained_prefix_image_count"] != RETAINED_CROP_COUNT:
                    problems.append(f"{sample_id}/{arm}: retained image count is not 2")
    config = json.loads((stage_dir / "config.json").read_text(encoding="utf-8"))
    for name, expected_hash in config["baseline_fingerprints"].items():
        if sha256_file(BASELINE_DIR / name) != expected_hash:
            problems.append(f"Frozen baseline file changed: {name}")

    devcheck_rows: List[Dict[str, Any]] = []
    devcheck_pass = True
    if stage == "devcheck":
        devcheck_rows, devcheck_pass = reconstruction_devcheck(stage, episodes)
        write_csv(OUTPUT_ROOT / "reconstruction_devcheck.csv", devcheck_rows)
        if not devcheck_pass:
            problems.append("A_full reconstruction devcheck did not reproduce all critical suffix fields")
    report = [
        f"# Round-6 Memory Surgery {stage.title()} Integrity Report", "",
        f"- Overall: **{'PASS' if not problems else 'FAIL'}**",
        f"- Samples: {len(by_sample)}", f"- Arms: `{arms}`", f"- Episode records: {len(episodes)}",
        "- Prefixes are replayed from frozen baseline crop files; no prefix inference is run.",
        "- GPU workers read sanitized prefixes and index-only plans, with no GT answer/bbox/category/coverage fields.",
        "- B/C/E use one identical observation-eviction placeholder and retain exactly two prefix crop images.",
        "- Original image and assistant textual history remain present in every arm.",
        "- Baseline files are read-only and fingerprint-checked.", "", "## Problems",
    ]
    report.extend(f"- {problem}" for problem in problems)
    if not problems:
        report.append("- None.")
    (stage_dir / "integrity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if stage == "devcheck":
        print(json.dumps({"reconstruction_pass": devcheck_pass, "rows": devcheck_rows}, indent=2))
    if problems:
        raise RuntimeError(f"Integrity checks failed for {stage}; see {stage_dir / 'integrity_report.md'}")


def validate_only() -> None:
    rows, bundles, episodes = build_cohort()
    primary = [row for row in rows if row["cohort"] == "primary_prefix_hit"]
    secondary = [row for row in rows if row["cohort"] == "secondary_prefix_no_hit"]
    exclusions: Dict[str, int] = {}
    for row in rows:
        for reason in str(row["exclusion_reason"]).split(";"):
            if reason:
                exclusions[reason] = exclusions.get(reason, 0) + 1
    for sample_id, bundle in bundles.items():
        for arm in ARMS:
            retained = retained_indices(arm, sample_id, episodes[sample_id])
            expected = PREFIX_CROP_COUNT if arm in {"A_full", "D_force_answer_r6"} else RETAINED_CROP_COUNT
            if len(retained) != expected or len(set(retained)) != expected:
                raise RuntimeError(f"Invalid retention plan for {sample_id}/{arm}: {retained}")
    print(json.dumps({
        "baseline_episodes": len(rows), "eligible": len(bundles),
        "primary_prefix_hit": len(primary), "secondary_prefix_no_hit": len(secondary),
        "exclusions": exclusions, "validated_plans": len(bundles) * len(ARMS),
        "created_output": False,
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run-shard", action="store_true")
    modes.add_argument("--merge-arm", action="store_true")
    modes.add_argument("--merge-stage", action="store_true")
    parser.add_argument("--stage", choices=STAGES, default="smoke")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_only()
        return
    if args.run_shard or args.merge_arm:
        if not args.arm:
            raise ValueError("--arm is required for shard and arm-merge modes")
        if args.stage == "devcheck" and args.arm != "A_full":
            raise ValueError("Devcheck runs A_full only")
    if args.prepare:
        prepare(args.stage)
    elif args.run_shard:
        if not 0 <= args.shard_id < args.num_shards:
            raise ValueError("shard-id must be in [0, num-shards)")
        run_shard(args.stage, args.arm, args.shard_id, args.num_shards)
    elif args.merge_arm:
        merge_arm(args.stage, args.arm, args.num_shards)
    else:
        merge_stage(args.stage)


if __name__ == "__main__":
    main()
