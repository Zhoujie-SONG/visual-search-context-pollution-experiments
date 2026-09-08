#!/usr/bin/env python3
"""Run Mini-o3's unmodified active visual search protocol on VisualNeedle smoke20.

Inference and evaluation are deliberately separate. ``run_episode`` receives only
an original image and a question. Answers, categories, and GT boxes are loaded only
after all inference episodes in a shard have finished.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import runpy
import statistics
import time
import traceback
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/home/songzhoujie/cvpr27/vstar")
DATA_ROOT = Path("/home/songzhoujie/cvpr27/needle_data")
MODEL_PATH = Path("/mnt/data2/szj/models/Mini-o3-7B-v1-complete")
OFFICIAL_REPO = Path("/home/songzhoujie/cvpr27/Mini-o3")
OUTPUT_PARENT = PROJECT_ROOT / "outputs" / "visualneedle_minio3"
OUTPUT_DIR = OUTPUT_PARENT / "smoke20"
SAMPLE_IDS_PATH = OUTPUT_PARENT / "smoke20_sample_ids.txt"

CATEGORIES = [
    "Color Recognition",
    "Entity Recognition",
    "OCR Recognition",
    "Occluded Object Recognition",
    "Spatial Relationship",
]
SEED = 42
SAMPLES_PER_CATEGORY = 4
MAX_ROUNDS = 12
MAX_IMAGES = 12
MAX_NEW_TOKENS_PER_ROUND = 2048
TOTAL_GENERATION_BUDGET = 8192
ROUND_MAX_TIME_SECONDS = 600
MIN_PIXELS = 40_000
MAX_PIXELS = 2_000_000
MAX_PROCESSOR_ASPECT_RATIO = 199
MIN_PROCESSOR_SIDE = 28
HIGH_OVERLAP_THRESHOLD = 0.7
GT_COVERAGE_THRESHOLD = 0.10


def load_official_prompts() -> Tuple[str, str, str]:
    constants_path = OFFICIAL_REPO / "verl" / "trainer" / "constants.py"
    values = runpy.run_path(str(constants_path))
    return (
        values["TOOL_CROP_SYSTEM_PROMPT"],
        values["TOOL_CALL_CROP_MULTI_TRUN_PROMPT"],
        values["ERROR_INFO_MULTI_TURN_PROMPT"],
    )


SYSTEM_PROMPT, OFFICIAL_OBSERVATION_PROMPT, ERROR_INFO_PROMPT = load_official_prompts()
VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass(frozen=True)
class InferenceSample:
    sample_id: str
    question: str
    image_path: Path


@dataclass(frozen=True)
class EvaluationMetadata:
    sample_id: str
    category: str
    gt_answer: str
    gt_bbox: Tuple[float, float, float, float]


@dataclass
class Observation:
    source_name: str
    original_bbox: Tuple[int, int, int, int]
    processed_image: Image.Image
    raw_width: int
    raw_height: int
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_annotation_rows() -> List[Dict[str, Any]]:
    path = DATA_ROOT / "visualneedle_300en.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 300:
        raise ValueError(f"Expected 300 VisualNeedle rows, found {len(rows)}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("VisualNeedle annotation contains duplicate sample IDs")
    return rows


def load_inference_samples() -> Dict[str, InferenceSample]:
    samples: Dict[str, InferenceSample] = {}
    images_root = (DATA_ROOT / "images").resolve()
    for row in load_annotation_rows():
        image_path = (DATA_ROOT / row["image_path"]).resolve()
        if image_path.parent != images_root:
            raise ValueError(f"Refusing non-original model input path: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        sample_id = str(row["id"])
        samples[sample_id] = InferenceSample(sample_id, str(row["question"]), image_path)
    return samples


def load_evaluation_metadata() -> Dict[str, EvaluationMetadata]:
    metadata: Dict[str, EvaluationMetadata] = {}
    for row in load_annotation_rows():
        bbox = tuple(float(value) for value in row["bbox"])
        if len(bbox) != 4:
            raise ValueError(f"Invalid GT bbox for {row['id']}: {bbox}")
        metadata[str(row["id"])] = EvaluationMetadata(
            sample_id=str(row["id"]),
            category=str(row["category"]),
            gt_answer=str(row["answer"]),
            gt_bbox=bbox,
        )
    return metadata


def prepare_selection() -> List[str]:
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    rows = load_annotation_rows()
    by_id = {str(row["id"]): row for row in rows}
    if SAMPLE_IDS_PATH.exists():
        ids = [line.strip() for line in SAMPLE_IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != 20 or len(ids) != len(set(ids)):
            raise ValueError(f"Existing selection is not 20 unique IDs: {SAMPLE_IDS_PATH}")
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"Unknown IDs in frozen selection: {missing}")
    else:
        rng = random.Random(SEED)
        ids = []
        for category in CATEGORIES:
            candidates = sorted(str(row["id"]) for row in rows if row["category"] == category)
            if len(candidates) < SAMPLES_PER_CATEGORY:
                raise ValueError(f"Not enough samples in category {category}")
            ids.extend(rng.sample(candidates, SAMPLES_PER_CATEGORY))
        SAMPLE_IDS_PATH.write_text("\n".join(ids) + "\n", encoding="utf-8")
    counts = Counter(str(by_id[sample_id]["category"]) for sample_id in ids)
    if counts != Counter({category: 4 for category in CATEGORIES}):
        raise ValueError(f"Frozen selection is not balanced: {dict(counts)}")
    return ids


def prompt_parts(template: str, action_turn: int, observation_turn: int) -> Tuple[str, str]:
    rendered = template.format(action_turn=action_turn, observation_turn=observation_turn)
    if rendered.count(VISION_PLACEHOLDER) != 1:
        raise ValueError("Official observation prompt must contain exactly one vision placeholder")
    return tuple(rendered.split(VISION_PLACEHOLDER, 1))  # type: ignore[return-value]


def edge_replicate_pad(
    image: Image.Image, target_width: int, target_height: int
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    image = image.convert("RGB")
    width, height = image.size
    if target_width < width or target_height < height:
        raise ValueError("Padding target cannot shrink image")
    left = (target_width - width) // 2
    right = target_width - width - left
    top = (target_height - height) // 2
    bottom = target_height - height - top
    if not any((left, top, right, bottom)):
        return image, (0, 0, 0, 0)

    horizontal = Image.new("RGB", (target_width, height))
    horizontal.paste(image, (left, 0))
    if left:
        edge = image.crop((0, 0, 1, height)).resize((left, height), Image.Resampling.NEAREST)
        horizontal.paste(edge, (0, 0))
    if right:
        edge = image.crop((width - 1, 0, width, height)).resize((right, height), Image.Resampling.NEAREST)
        horizontal.paste(edge, (left + width, 0))

    result = Image.new("RGB", (target_width, target_height))
    result.paste(horizontal, (0, top))
    if top:
        edge = horizontal.crop((0, 0, target_width, 1)).resize((target_width, top), Image.Resampling.NEAREST)
        result.paste(edge, (0, 0))
    if bottom:
        edge = horizontal.crop((0, height - 1, target_width, height)).resize(
            (target_width, bottom), Image.Resampling.NEAREST
        )
        result.paste(edge, (0, top + height))
    return result, (left, top, right, bottom)


def process_for_model(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """Equal-scale resize plus edge padding; never stretch either axis independently."""
    raw = image.convert("RGB")
    raw_width, raw_height = raw.size
    if raw_width <= 0 or raw_height <= 0:
        raise ValueError("Empty image")
    pixels = raw_width * raw_height
    scale = 1.0
    if pixels < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / pixels)
    elif pixels > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / pixels)
    resized_width = max(1, int(round(raw_width * scale)))
    resized_height = max(1, int(round(raw_height * scale)))
    resized = raw.resize((resized_width, resized_height), Image.Resampling.LANCZOS) if scale != 1.0 else raw.copy()

    target_width = max(MIN_PROCESSOR_SIDE, resized.width)
    target_height = max(MIN_PROCESSOR_SIDE, resized.height)
    if target_width > target_height * MAX_PROCESSOR_ASPECT_RATIO:
        target_height = max(target_height, math.ceil(target_width / MAX_PROCESSOR_ASPECT_RATIO))
    if target_height > target_width * MAX_PROCESSOR_ASPECT_RATIO:
        target_width = max(target_width, math.ceil(target_height / MAX_PROCESSOR_ASPECT_RATIO))
    processed, pads = edge_replicate_pad(resized, target_width, target_height)

    # Padding can push a boundary case slightly over the pixel cap. A final equal
    # scaling keeps aspect ratio and leaves the content-to-processed map explicit.
    post_scale = 1.0
    if processed.width * processed.height > MAX_PIXELS:
        post_scale = math.sqrt(MAX_PIXELS / (processed.width * processed.height))
        new_size = (
            max(MIN_PROCESSOR_SIDE, int(math.floor(processed.width * post_scale))),
            max(MIN_PROCESSOR_SIDE, int(math.floor(processed.height * post_scale))),
        )
        processed = processed.resize(new_size, Image.Resampling.LANCZOS)

    left, top, right, bottom = pads
    metadata = {
        "original_width": raw_width,
        "original_height": raw_height,
        "resized_content_width": resized_width,
        "resized_content_height": resized_height,
        "processed_width": processed.width,
        "processed_height": processed.height,
        "scale_x": (resized_width / raw_width) * post_scale,
        "scale_y": (resized_height / raw_height) * post_scale,
        "pad_left": int(round(left * post_scale)),
        "pad_top": int(round(top * post_scale)),
        "pad_right": int(round(right * post_scale)),
        "pad_bottom": int(round(bottom * post_scale)),
        "padding_applied": any(pads),
    }
    return processed, metadata


def initial_messages(image: Image.Image, question: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]


def append_observation_message(
    messages: List[Dict[str, Any]], raw_output: str, image: Image.Image, action_turn: int, observation_turn: int
) -> None:
    prefix, suffix = prompt_parts(OFFICIAL_OBSERVATION_PROMPT, action_turn, observation_turn)
    messages.extend(
        [
            {"role": "assistant", "content": raw_output},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix},
                    {"type": "image", "image": image},
                    {"type": "text", "text": suffix},
                ],
            },
        ]
    )


def append_official_error_message(messages: List[Dict[str, Any]], raw_output: str, error: str) -> None:
    tool_output = f"ERROR occurs during grounding. Error Information: {error}.\n"
    messages.extend(
        [
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": tool_output + ERROR_INFO_PROMPT},
        ]
    )


def parse_answer(text: str) -> str:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.IGNORECASE | re.DOTALL)
    return matches[-1].strip() if matches else ""


def parse_grounding(text: str) -> Dict[str, Any]:
    matches = re.findall(r"<grounding>\s*(\{.*?\})\s*</grounding>", text or "", flags=re.DOTALL)
    if not matches:
        return {"present": False, "success": False, "error": "no_grounding_tag"}
    payload = matches[-1]
    try:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            value = ast.literal_eval(payload)
        if not isinstance(value, dict):
            raise ValueError("grounding payload is not an object")
        bbox = value.get("bbox_2d")
        source = value.get("source")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox_2d must contain four coordinates")
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        coordinates = [float(item) for item in bbox]
        if not all(math.isfinite(item) for item in coordinates):
            raise ValueError("bbox_2d contains non-finite coordinates")
        return {
            "present": True,
            "success": True,
            "payload": payload,
            "bbox": coordinates,
            "source": source,
        }
    except Exception as exc:
        return {"present": True, "success": False, "payload": payload, "error": str(exc)}


def resolve_source(source: str, observations: Sequence[Observation]) -> Tuple[int, Observation]:
    if source == "original_image":
        index = 0
    else:
        match = re.fullmatch(r"observation_([0-9]+)", source)
        if not match:
            raise ValueError(f"source '{source}' does not match original_image or observation_i")
        index = int(match.group(1))
    if index >= len(observations):
        raise ValueError(f"source '{source}' is unavailable; only {len(observations)} images exist")
    return index, observations[index]


def convert_bbox(
    normalized_bbox: Sequence[float], source_observation: Observation, original_size: Tuple[int, int]
) -> Dict[str, Any]:
    processed_width, processed_height = source_observation.processed_image.size
    source_processed = [
        normalized_bbox[0] * processed_width,
        normalized_bbox[1] * processed_height,
        normalized_bbox[2] * processed_width,
        normalized_bbox[3] * processed_height,
    ]
    if source_processed[2] <= source_processed[0] or source_processed[3] <= source_processed[1]:
        raise ValueError(f"bbox has non-positive extent before clipping: {source_processed}")

    was_clipped = any(value < 0.0 or value > 1.0 for value in normalized_bbox)
    content = [
        (source_processed[0] - source_observation.pad_left) / source_observation.scale_x,
        (source_processed[1] - source_observation.pad_top) / source_observation.scale_y,
        (source_processed[2] - source_observation.pad_left) / source_observation.scale_x,
        (source_processed[3] - source_observation.pad_top) / source_observation.scale_y,
    ]
    clipped_content = [
        max(0.0, min(float(source_observation.raw_width), content[0])),
        max(0.0, min(float(source_observation.raw_height), content[1])),
        max(0.0, min(float(source_observation.raw_width), content[2])),
        max(0.0, min(float(source_observation.raw_height), content[3])),
    ]
    if any(abs(a - b) > 1e-7 for a, b in zip(content, clipped_content)):
        was_clipped = True
    if clipped_content[2] <= clipped_content[0] or clipped_content[3] <= clipped_content[1]:
        raise ValueError("bbox lies entirely in processor padding or outside source content")

    parent = source_observation.original_bbox
    sx = (parent[2] - parent[0]) / source_observation.raw_width
    sy = (parent[3] - parent[1]) / source_observation.raw_height
    original_float = [
        parent[0] + clipped_content[0] * sx,
        parent[1] + clipped_content[1] * sy,
        parent[0] + clipped_content[2] * sx,
        parent[1] + clipped_content[3] * sy,
    ]
    original_width, original_height = original_size
    clipped_original = [
        max(0, min(original_width - 1, int(math.floor(original_float[0])))),
        max(0, min(original_height - 1, int(math.floor(original_float[1])))),
        max(1, min(original_width, int(math.ceil(original_float[2])))),
        max(1, min(original_height, int(math.ceil(original_float[3])))),
    ]
    if clipped_original[2] <= clipped_original[0] or clipped_original[3] <= clipped_original[1]:
        raise ValueError(f"bbox is invalid after clipping: {clipped_original}")
    return {
        "source_processed_pixel": source_processed,
        "source_content_pixel": content,
        "pixel_original": original_float,
        "clipped_original": clipped_original,
        "was_clipped": was_clipped,
    }


class HFBackend:
    def __init__(self, model_path: Path):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to the E2-M runner")
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device_name = torch.cuda.get_device_name(0)

    def generate(self, messages: Sequence[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images: List[Image.Image] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                images.extend(
                    item["image"]
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "image"
                )
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[1])
        image_grid_thw = inputs.get("image_grid_thw")
        image_grids = image_grid_thw.detach().cpu().tolist() if image_grid_thw is not None else None
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    max_time=ROUND_MAX_TIME_SECONDS,
                    return_dict_in_generate=True,
                )
            elapsed = time.perf_counter() - started
            sequence = output.sequences
            new_ids = sequence[:, input_tokens:]
            raw = self.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            final_token = int(sequence[0, -1].item())
            if final_token in eos_ids:
                finish_reason = "eos"
            elif int(new_ids.shape[1]) >= max_new_tokens:
                finish_reason = "length"
            else:
                finish_reason = "time_or_unknown"
            return {
                "raw_model_output": raw,
                "input_tokens": input_tokens,
                "output_tokens": int(new_ids.shape[1]),
                "finish_reason": finish_reason,
                "runtime_seconds": elapsed,
                "input_image_count": len(images),
                "image_grid_thw": image_grids,
                "max_new_tokens": max_new_tokens,
            }
        finally:
            del inputs
            self.torch.cuda.empty_cache()


def inference_audit(sample: InferenceSample, turn_index: int, input_image_count: int) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "turn_index": turn_index,
        "text_source_classes": [
            "official_system_prompt",
            "question",
            "prior_model_outputs",
            "official_tool_result_prompts",
        ],
        "initial_image_path": str(sample.image_path),
        "initial_image_under_images_directory": sample.image_path.parent == (DATA_ROOT / "images").resolve(),
        "input_image_count": input_image_count,
        "evaluation_metadata_object_available_to_run_episode": False,
        "images_bbox_used": False,
    }


def run_episode(backend: HFBackend, sample: InferenceSample, output_dir: Path) -> Dict[str, Any]:
    """Run one episode without accepting or loading evaluation metadata."""
    started = time.perf_counter()
    backend.torch.cuda.reset_peak_memory_stats()
    with Image.open(sample.image_path) as handle:
        original = handle.convert("RGB")
    original_processed, original_prep = process_for_model(original)
    observations = [
        Observation(
            source_name="original_image",
            original_bbox=(0, 0, original.width, original.height),
            processed_image=original_processed,
            raw_width=original.width,
            raw_height=original.height,
            scale_x=float(original_prep["scale_x"]),
            scale_y=float(original_prep["scale_y"]),
            pad_left=int(original_prep["pad_left"]),
            pad_top=int(original_prep["pad_top"]),
        )
    ]
    messages = initial_messages(original_processed, sample.question)
    turns: List[Dict[str, Any]] = []
    crops: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    final_answer = ""
    total_output_tokens = 0
    reached_turn_cap = False
    reached_image_cap = False
    stop_reason = ""
    status = "completed"
    error_type = ""
    error_message = ""
    preprocessing_error_count = 0
    crop_directory = output_dir / "crops" / sample.sample_id
    crop_directory.mkdir(parents=True, exist_ok=True)

    for round_index in range(1, MAX_ROUNDS + 1):
        remaining = TOTAL_GENERATION_BUDGET - total_output_tokens
        if remaining <= 0:
            stop_reason = "total_generation_budget"
            break
        turn_started = time.perf_counter()
        turn: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "turn_index": round_index,
            "question": sample.question,
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
        }
        try:
            generation = backend.generate(messages, min(MAX_NEW_TOKENS_PER_ROUND, remaining))
            turn.update(generation)
            total_output_tokens += int(generation["output_tokens"])
            raw_output = str(generation["raw_model_output"])
            audits.append(inference_audit(sample, round_index, int(generation["input_image_count"])))
            grounding = parse_grounding(raw_output)
            answer = parse_answer(raw_output)

            # The official rollout gives grounding precedence when both tags occur.
            if grounding["present"]:
                turn["action_type"] = "crop" if grounding["success"] else "invalid"
                turn["bbox_parse_success"] = bool(grounding["success"])
                if not grounding["success"]:
                    turn["error_type"] = "bbox_parse_error"
                    turn["error_message"] = grounding["error"]
                    turns.append(turn)
                    append_official_error_message(messages, raw_output, str(grounding["error"]))
                    continue

                turn["predicted_bbox_raw"] = grounding["bbox"]
                turn["source"] = grounding["source"]
                if len(observations) >= MAX_IMAGES or round_index == MAX_ROUNDS:
                    reached_image_cap = len(observations) >= MAX_IMAGES
                    reached_turn_cap = round_index == MAX_ROUNDS
                    turn["error_type"] = "budget_cap_before_crop"
                    turn["error_message"] = "Official policy does not execute a crop after the image/round cap"
                    turns.append(turn)
                    stop_reason = "image_or_turn_cap"
                    break
                try:
                    crop_phase = "bbox_validation"
                    _, source_observation = resolve_source(str(grounding["source"]), observations)
                    converted = convert_bbox(grounding["bbox"], source_observation, original.size)
                    turn["predicted_bbox_source_processed_pixel"] = converted["source_processed_pixel"]
                    turn["predicted_bbox_source_content_pixel"] = converted["source_content_pixel"]
                    turn["predicted_bbox_pixel"] = converted["pixel_original"]
                    turn["predicted_bbox_clipped"] = converted["clipped_original"]
                    turn["bbox_was_clipped"] = bool(converted["was_clipped"])
                    turn["bbox_valid"] = True

                    crop_number = len(crops) + 1
                    bbox = tuple(int(value) for value in converted["clipped_original"])
                    raw_crop = original.crop(bbox).convert("RGB")
                    raw_path = crop_directory / f"turn_{round_index:02d}_raw.png"
                    processed_path = crop_directory / f"turn_{round_index:02d}_processed.png"
                    crop_phase = "preprocessing"
                    raw_crop.save(raw_path)
                    processed_crop, prep = process_for_model(raw_crop)
                    processed_crop.save(processed_path)
                    crop = {
                        "sample_id": sample.sample_id,
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
                        "pad_left": prep["pad_left"],
                        "pad_top": prep["pad_top"],
                        "pad_right": prep["pad_right"],
                        "pad_bottom": prep["pad_bottom"],
                        "preprocess_scale_x": prep["scale_x"],
                        "preprocess_scale_y": prep["scale_y"],
                    }
                    crops.append(crop)
                    observations.append(
                        Observation(
                            source_name=f"observation_{crop_number}",
                            original_bbox=bbox,
                            processed_image=processed_crop,
                            raw_width=raw_crop.width,
                            raw_height=raw_crop.height,
                            scale_x=float(prep["scale_x"]),
                            scale_y=float(prep["scale_y"]),
                            pad_left=int(prep["pad_left"]),
                            pad_top=int(prep["pad_top"]),
                        )
                    )
                    turn.update(
                        {
                            "crop_executed": True,
                            "crop_path": str(processed_path),
                            "raw_crop_path": str(raw_path),
                            "original_crop_width": raw_crop.width,
                            "original_crop_height": raw_crop.height,
                            "processed_width": processed_crop.width,
                            "processed_height": processed_crop.height,
                            "padding_applied": bool(prep["padding_applied"]),
                        }
                    )
                    turns.append(turn)
                    append_observation_message(
                        messages,
                        raw_output,
                        processed_crop,
                        action_turn=round_index - 1,
                        observation_turn=round_index,
                    )
                except Exception as exc:
                    if crop_phase == "preprocessing":
                        preprocessing_error_count += 1
                    turn["action_type"] = "invalid"
                    turn["bbox_valid"] = False
                    turn["error_type"] = (
                        "preprocessing_error" if crop_phase == "preprocessing" else "bbox_validation_error"
                    )
                    turn["error_message"] = str(exc)
                    turns.append(turn)
                    append_official_error_message(messages, raw_output, str(exc))
                continue

            if answer:
                turn["action_type"] = "answer"
                turn["final_answer"] = answer
                turns.append(turn)
                final_answer = answer
                stop_reason = "answer"
                break

            turn["action_type"] = "invalid"
            turn["error_type"] = "unparsed_model_output"
            turn["error_message"] = "Output contains neither a complete grounding tag nor a complete answer tag"
            turns.append(turn)
            stop_reason = "unparsed_model_output"
            status = "invalid"
            break
        except RuntimeError as exc:
            message = str(exc)
            error_type = "oom" if "out of memory" in message.lower() else "runtime_error"
            status = "oom" if error_type == "oom" else "error"
            error_message = message
            turn["error_type"] = error_type
            turn["error_message"] = message
            turn["runtime_seconds"] = time.perf_counter() - turn_started
            turns.append(turn)
            stop_reason = error_type
            break
        except Exception as exc:
            error_type = type(exc).__name__
            status = "error"
            error_message = str(exc)
            turn["error_type"] = error_type
            turn["error_message"] = str(exc)
            turn["traceback"] = traceback.format_exc()
            turn["runtime_seconds"] = time.perf_counter() - turn_started
            turns.append(turn)
            stop_reason = "error"
            break
    else:
        reached_turn_cap = True
        stop_reason = "turn_cap"

    if len(turns) >= MAX_ROUNDS and not final_answer:
        reached_turn_cap = True
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "original_image_path": str(sample.image_path),
        "original_width": original.width,
        "original_height": original.height,
        "original_preprocessing": original_prep,
        "status": status,
        "stop_reason": stop_reason,
        "error_type": error_type,
        "error_message": error_message,
        "episode_finished": True,
        "num_rounds": len(turns),
        "num_crops": len(crops),
        "final_answer": final_answer,
        "produced_final_answer": bool(final_answer),
        "reached_turn_cap": reached_turn_cap,
        "reached_image_cap": reached_image_cap,
        "total_output_tokens": total_output_tokens,
        "preprocessing_error_count": preprocessing_error_count,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": backend.torch.cuda.max_memory_allocated() / (1024 * 1024),
        "turns": turns,
        "crops": crops,
        "input_audit": audits,
    }


def normalize_answer(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .,!?:;\"'`()[]{}")
    return value


def intersection_metrics(crop: Sequence[float], gt: Sequence[float]) -> Dict[str, Any]:
    ix1, iy1 = max(crop[0], gt[0]), max(crop[1], gt[1])
    ix2, iy2 = min(crop[2], gt[2]), min(crop[3], gt[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    crop_area = max(0.0, crop[2] - crop[0]) * max(0.0, crop[3] - crop[1])
    gt_area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    union = crop_area + gt_area - intersection
    gx, gy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    cx, cy = (crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2
    return {
        "intersection_area": intersection,
        "crop_area": crop_area,
        "gt_area": gt_area,
        "iou_with_gt": intersection / union if union > 0 else 0.0,
        "gt_coverage": intersection / gt_area if gt_area > 0 else 0.0,
        "contains_gt_center": crop[0] <= gx <= crop[2] and crop[1] <= gy <= crop[3],
        "crop_center_x": cx,
        "crop_center_y": cy,
        "gt_center_x": gx,
        "gt_center_y": gy,
        "center_distance": math.hypot(cx - gx, cy - gy),
    }


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    return float(intersection_metrics(first, second)["iou_with_gt"])


def evaluate_episode(raw_episode: Dict[str, Any], metadata: EvaluationMetadata) -> Dict[str, Any]:
    episode = json.loads(json.dumps(raw_episode, ensure_ascii=False))
    episode["category"] = metadata.category
    episode["gt_answer"] = metadata.gt_answer
    episode["gt_bbox"] = list(metadata.gt_bbox)
    prediction = str(episode.get("final_answer", ""))
    episode["exact_match"] = bool(prediction) and prediction.strip() == metadata.gt_answer.strip()
    episode["normalized_string_match"] = bool(prediction) and normalize_answer(prediction) == normalize_answer(metadata.gt_answer)

    prior: List[Dict[str, Any]] = []
    for crop in episode.get("crops", []):
        metrics = intersection_metrics(crop["predicted_bbox_clipped"], metadata.gt_bbox)
        crop.update(metrics)
        crop["geometric_label"] = (
            "GEOMETRICALLY_IRRELEVANT" if metrics["gt_coverage"] < GT_COVERAGE_THRESHOLD else ""
        )
        if prior:
            overlaps = [(bbox_iou(crop["predicted_bbox_clipped"], item["predicted_bbox_clipped"]), item["turn_index"]) for item in prior]
            maximum, previous_turn = max(overlaps, key=lambda item: item[0])
        else:
            maximum, previous_turn = 0.0, None
        crop["max_iou_with_previous_crop"] = maximum
        crop["most_overlapping_previous_turn"] = previous_turn
        crop["high_overlap_previous"] = maximum >= HIGH_OVERLAP_THRESHOLD
        prior.append(crop)

    crop_by_turn = {crop["turn_index"]: crop for crop in episode.get("crops", [])}
    for turn in episode.get("turns", []):
        turn["category"] = metadata.category
        if turn["turn_index"] in crop_by_turn:
            evaluated_crop = crop_by_turn[turn["turn_index"]]
            for key in (
                "intersection_area", "crop_area", "gt_area", "iou_with_gt", "gt_coverage",
                "contains_gt_center", "crop_center_x", "crop_center_y", "gt_center_x", "gt_center_y",
                "center_distance", "geometric_label", "max_iou_with_previous_crop",
                "most_overlapping_previous_turn", "high_overlap_previous",
            ):
                turn[key] = evaluated_crop[key]
    return episode


def episode_summary(episode: Dict[str, Any]) -> Dict[str, Any]:
    crops = episode.get("crops", [])
    center_hits = [crop["turn_index"] for crop in crops if crop.get("contains_gt_center")]
    coverage_hits = [crop["turn_index"] for crop in crops if crop.get("gt_coverage", 0.0) >= GT_COVERAGE_THRESHOLD]
    irrelevant = [crop for crop in crops if crop.get("geometric_label") == "GEOMETRICALLY_IRRELEVANT"]
    high_overlap = [crop for crop in crops if crop.get("high_overlap_previous")]
    parse_errors = sum(
        turn.get("error_type") in {"bbox_parse_error", "unparsed_model_output"}
        for turn in episode.get("turns", [])
    )
    return {
        "sample_id": episode["sample_id"],
        "category": episode["category"],
        "status": episode["status"],
        "num_rounds": episode["num_rounds"],
        "num_crops": len(crops),
        "produced_final_answer": episode["produced_final_answer"],
        "final_answer": episode["final_answer"],
        "gt_answer": episode["gt_answer"],
        "exact_match": episode["exact_match"],
        "normalized_string_match": episode["normalized_string_match"],
        "reached_turn_cap": episode["reached_turn_cap"],
        "reached_image_cap": episode["reached_image_cap"],
        "first_gt_center_hit_turn": min(center_hits) if center_hits else "",
        "first_gt_coverage_10_turn": min(coverage_hits) if coverage_hits else "",
        "max_gt_coverage": max((crop.get("gt_coverage", 0.0) for crop in crops), default=0.0),
        "ever_gt_center_hit": bool(center_hits),
        "num_geometrically_irrelevant": len(irrelevant),
        "irrelevant_crop_ratio": len(irrelevant) / len(crops) if crops else 0.0,
        "num_high_overlap_crops": len(high_overlap),
        "parse_error_count": parse_errors,
        "preprocessing_error_count": episode.get("preprocessing_error_count", 0),
        "oom": episode["status"] == "oom",
        "runtime_seconds": episode["runtime_seconds"],
        "stop_reason": episode["stop_reason"],
        "error_type": episode.get("error_type", ""),
        "error_message": episode.get("error_message", ""),
    }


CROP_FIELDS = [
    "sample_id", "category", "turn_index", "crop_index", "source",
    "predicted_bbox_raw", "predicted_bbox_pixel", "predicted_bbox_clipped",
    "predicted_bbox_source_processed_pixel", "predicted_bbox_source_content_pixel",
    "bbox_was_clipped", "raw_crop_path", "crop_path", "original_crop_width",
    "original_crop_height", "processed_width", "processed_height", "padding_applied",
    "pad_left", "pad_top", "pad_right", "pad_bottom", "preprocess_scale_x",
    "preprocess_scale_y", "intersection_area", "crop_area", "gt_area", "iou_with_gt",
    "gt_coverage", "contains_gt_center", "crop_center_x", "crop_center_y", "gt_center_x",
    "gt_center_y", "center_distance", "geometric_label", "max_iou_with_previous_crop",
    "most_overlapping_previous_turn", "high_overlap_previous",
]

SUMMARY_FIELDS = [
    "sample_id", "category", "status", "num_rounds", "num_crops", "produced_final_answer",
    "final_answer", "gt_answer", "exact_match", "normalized_string_match", "reached_turn_cap",
    "reached_image_cap", "first_gt_center_hit_turn", "first_gt_coverage_10_turn",
    "max_gt_coverage", "ever_gt_center_hit", "num_geometrically_irrelevant",
    "irrelevant_crop_ratio", "num_high_overlap_crops", "parse_error_count",
    "preprocessing_error_count", "oom", "runtime_seconds", "stop_reason", "error_type",
    "error_message",
]

TURN_FIELDS = [
    "sample_id", "category", "turn_index", "question", "raw_model_output", "action_type",
    "final_answer", "predicted_bbox_raw", "predicted_bbox_pixel", "predicted_bbox_clipped",
    "predicted_bbox_source_processed_pixel", "predicted_bbox_source_content_pixel",
    "bbox_parse_success", "bbox_valid", "bbox_was_clipped", "crop_executed", "crop_path",
    "raw_crop_path", "source", "input_image_count", "input_tokens", "output_tokens",
    "max_new_tokens", "finish_reason", "runtime_seconds", "image_grid_thw", "error_type",
    "error_message", "original_crop_width", "original_crop_height", "processed_width",
    "processed_height", "padding_applied", "intersection_area", "crop_area", "gt_area",
    "iou_with_gt", "gt_coverage", "contains_gt_center", "center_distance", "geometric_label",
    "max_iou_with_previous_crop", "most_overlapping_previous_turn", "high_overlap_previous",
]


def csv_ready(row: Dict[str, Any]) -> Dict[str, Any]:
    converted = {}
    for key, value in row.items():
        converted[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
    return converted


def draw_trajectory(episode: Dict[str, Any], output_path: Path) -> None:
    with Image.open(episode["original_image_path"]) as handle:
        image = handle.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, int(round(max(image.size) / 700)))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#00a6a6"]
    for index, crop in enumerate(episode.get("crops", [])):
        bbox = tuple(crop["predicted_bbox_clipped"])
        color = colors[index % len(colors)]
        draw.rectangle(bbox, outline=color, width=width)
        draw.text((bbox[0] + width, bbox[1] + width), f"T{crop['turn_index']}", fill=color, font=ImageFont.load_default())
    gt = tuple(episode["gt_bbox"])
    draw.rectangle(gt, outline="#00ff66", width=width + 1)
    draw.text((gt[0] + width, gt[1] + width), "GT (offline)", fill="#00ff66", font=ImageFont.load_default())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def aggregate(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    crops = sum(int(row["num_crops"]) for row in rows)
    irrelevant = sum(int(row["num_geometrically_irrelevant"]) for row in rows)
    crop_counts = [int(row["num_crops"]) for row in rows]
    return {
        "category": label,
        "total_episodes": len(rows),
        "completed_episodes": sum(row["status"] in {"completed", "invalid"} for row in rows),
        "final_answer_episodes": sum(bool(row["produced_final_answer"]) for row in rows),
        "no_final_answer_episodes": sum(not bool(row["produced_final_answer"]) for row in rows),
        "total_crops": crops,
        "mean_crops_per_episode": statistics.mean(crop_counts) if crop_counts else 0.0,
        "median_crops_per_episode": statistics.median(crop_counts) if crop_counts else 0.0,
        "max_crops_per_episode": max(crop_counts, default=0),
        "gt_center_hit_episodes": sum(bool(row["ever_gt_center_hit"]) for row in rows),
        "gt_center_hit_rate": sum(bool(row["ever_gt_center_hit"]) for row in rows) / len(rows) if rows else 0.0,
        "geometrically_irrelevant_crops": irrelevant,
        "geometrically_irrelevant_crop_ratio": irrelevant / crops if crops else 0.0,
        "episodes_with_irrelevant_crop": sum(int(row["num_geometrically_irrelevant"]) > 0 for row in rows),
        "high_overlap_crop_count": sum(int(row["num_high_overlap_crops"]) for row in rows),
        "parse_failures": sum(int(row["parse_error_count"]) for row in rows),
        "preprocessing_failures": sum(int(row["preprocessing_error_count"]) for row in rows),
        "oom_count": sum(bool(row["oom"]) for row in rows),
    }


def review_candidates(summaries: Sequence[Dict[str, Any]], episodes: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    by_id = {episode["sample_id"]: episode for episode in episodes}
    most_crops = sorted(summaries, key=lambda row: (-int(row["num_crops"]), row["sample_id"]))[:3]
    never_hit = sorted(
        (row for row in summaries if not row["ever_gt_center_hit"]),
        key=lambda row: (-int(row["num_crops"]), row["sample_id"]),
    )[:3]

    def after_hit(row: Dict[str, Any]) -> int:
        first = row["first_gt_center_hit_turn"]
        if first == "":
            return -1
        return sum(crop["turn_index"] > int(first) for crop in by_id[row["sample_id"]]["crops"])

    continued = sorted(
        (row for row in summaries if row["ever_gt_center_hit"]),
        key=lambda row: (-after_hit(row), row["sample_id"]),
    )[:3]

    def maximum_repeat(row: Dict[str, Any]) -> float:
        return max((float(crop["max_iou_with_previous_crop"]) for crop in by_id[row["sample_id"]]["crops"]), default=0.0)

    repeated = sorted(summaries, key=lambda row: (-maximum_repeat(row), row["sample_id"]))[:3]
    hit_no_answer = [row for row in summaries if row["ever_gt_center_hit"] and not row["produced_final_answer"]]
    early_hit_wrong = [
        row for row in summaries
        if row["ever_gt_center_hit"] and row["first_gt_center_hit_turn"] != ""
        and int(row["first_gt_center_hit_turn"]) <= 2 and not row["normalized_string_match"]
    ]
    return {
        "A_most_crops": [row["sample_id"] for row in most_crops],
        "B_never_hit_gt": [row["sample_id"] for row in never_hit],
        "C_most_crops_after_gt_hit": [row["sample_id"] for row in continued],
        "D_highest_crop_overlap": [row["sample_id"] for row in repeated],
        "E_hit_gt_but_no_final_answer": [row["sample_id"] for row in hit_no_answer],
        "F_early_hit_but_normalized_answer_wrong": [row["sample_id"] for row in early_hit_wrong],
    }


def write_config(ids: Sequence[str], command: str) -> None:
    import torch
    import transformers

    config = {
        "experiment": "VisualNeedle Mini-o3 original active visual search smoke20",
        "model_id": "Mini-o3/Mini-o3-7B-v1",
        "model_path": str(MODEL_PATH),
        "official_repo": str(OFFICIAL_REPO),
        "dataset_root": str(DATA_ROOT),
        "environment": "venv_e2m",
        "python_executable": os.environ.get("E2M_PYTHON", "/mnt/data2/szj/envs/venv_e2m/bin/python"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "do_sample": False,
        "max_rounds": MAX_ROUNDS,
        "max_images_including_original": MAX_IMAGES,
        "max_new_tokens_per_round": MAX_NEW_TOKENS_PER_ROUND,
        "total_generation_budget": TOTAL_GENERATION_BUDGET,
        "max_time_per_round_seconds": ROUND_MAX_TIME_SECONDS,
        "custom_stopping_criteria": None,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "selection_seed": SEED,
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "categories": CATEGORIES,
        "sample_ids": list(ids),
        "system_prompt": SYSTEM_PROMPT,
        "observation_prompt_official": OFFICIAL_OBSERVATION_PROMPT,
        "error_prompt_official": ERROR_INFO_PROMPT,
        "crop_policy_changes": None,
        "termination_policy_changes": None,
        "compatibility_preprocessing": "equal-scale resize plus edge-replication padding for processor limits",
        "ground_truth_isolation": "run_episode accepts InferenceSample only; EvaluationMetadata loads after shard inference",
        "command": command,
    }
    atomic_write_json(OUTPUT_DIR / "config.json", config)


def run_shard(args: argparse.Namespace) -> None:
    ids = prepare_selection()
    assigned = [sample_id for index, sample_id in enumerate(ids) if index % args.num_shards == args.shard_id]
    shard_dir = OUTPUT_DIR / "shards" / f"shard_{args.shard_id}"
    raw_dir = shard_dir / "raw_episodes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    samples = load_inference_samples()
    backend = HFBackend(MODEL_PATH)
    log_lines = [
        f"shard={args.shard_id}/{args.num_shards}",
        f"gpu={backend.device_name}",
        f"assigned={assigned}",
    ]
    for position, sample_id in enumerate(assigned, start=1):
        path = raw_dir / f"{sample_id}.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("episode_finished") is True and existing.get("sample_id") == sample_id:
                    log_lines.append(f"resume_skip {position}/{len(assigned)} {sample_id}")
                    continue
            except Exception:
                pass
        log_lines.append(f"start {position}/{len(assigned)} {sample_id} {time.strftime('%F %T')}")
        try:
            episode = run_episode(backend, samples[sample_id], OUTPUT_DIR)
        except Exception as exc:
            episode = {
                "sample_id": sample_id,
                "question": samples[sample_id].question,
                "original_image_path": str(samples[sample_id].image_path),
                "status": "error",
                "stop_reason": "uncaught_episode_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "episode_finished": True,
                "num_rounds": 0,
                "num_crops": 0,
                "final_answer": "",
                "produced_final_answer": False,
                "reached_turn_cap": False,
                "reached_image_cap": False,
                "total_output_tokens": 0,
                "preprocessing_error_count": 0,
                "runtime_seconds": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "turns": [],
                "crops": [],
                "input_audit": [],
            }
        atomic_write_json(path, episode)
        log_lines.append(
            f"done {sample_id} status={episode['status']} rounds={episode['num_rounds']} "
            f"crops={episode['num_crops']} final={episode['produced_final_answer']} runtime={episode['runtime_seconds']:.1f}s"
        )
        (shard_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    # GT metadata is intentionally loaded only after every assigned episode is complete.
    metadata = load_evaluation_metadata()
    evaluated = []
    for sample_id in assigned:
        raw = json.loads((raw_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
        evaluated.append(evaluate_episode(raw, metadata[sample_id]))
    write_jsonl(shard_dir / "trajectories.jsonl", evaluated)
    (shard_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def integrity_checks(ids: Sequence[str], episodes: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    episode_ids = [episode.get("sample_id") for episode in episodes]
    if len(episodes) != 20:
        problems.append(f"Expected 20 episode records, found {len(episodes)}")
    if len(episode_ids) != len(set(episode_ids)):
        problems.append("Duplicate episode records found")
    if set(episode_ids) != set(ids):
        problems.append(f"Episode IDs differ from frozen IDs: missing={set(ids)-set(episode_ids)}, extra={set(episode_ids)-set(ids)}")
    for episode in episodes:
        trajectory_crops = episode.get("crops", [])
        seen = set()
        for crop in trajectory_crops:
            key = (crop["sample_id"], crop["turn_index"])
            if key in seen:
                problems.append(f"Duplicate crop key {key}")
            seen.add(key)
            for path_key in ("raw_crop_path", "crop_path"):
                path = Path(crop[path_key])
                if not path.is_file():
                    problems.append(f"Missing {path_key}: {path}")
                    continue
                with Image.open(path) as handle:
                    expected = (
                        (crop["original_crop_width"], crop["original_crop_height"])
                        if path_key == "raw_crop_path" else (crop["processed_width"], crop["processed_height"])
                    )
                    if handle.size != expected:
                        problems.append(f"Dimension mismatch for {path}: {handle.size} != {expected}")
            turn = next((item for item in episode.get("turns", []) if item["turn_index"] == crop["turn_index"]), None)
            if turn is None or turn.get("predicted_bbox_clipped") != crop["predicted_bbox_clipped"]:
                problems.append(f"Trajectory/crop bbox mismatch for {key}")
        for audit in episode.get("input_audit", []):
            if not audit.get("initial_image_under_images_directory") or audit.get("images_bbox_used"):
                problems.append(f"Model-input provenance failure: {episode['sample_id']} turn {audit.get('turn_index')}")
        for turn in episode.get("turns", []):
            if "raw_model_output" in turn and turn["raw_model_output"] is None:
                problems.append(f"Missing raw output: {episode['sample_id']} turn {turn['turn_index']}")
    return not problems, problems


def merge_results(args: argparse.Namespace) -> None:
    ids = prepare_selection()
    metadata = load_evaluation_metadata()
    raw_episodes = []
    logs = []
    for index, sample_id in enumerate(ids):
        shard_id = index % args.num_shards
        shard_dir = OUTPUT_DIR / "shards" / f"shard_{shard_id}"
        path = shard_dir / "raw_episodes" / f"{sample_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard episode: {path}")
        raw_episodes.append(json.loads(path.read_text(encoding="utf-8")))
    for shard_id in range(args.num_shards):
        path = OUTPUT_DIR / "shards" / f"shard_{shard_id}" / "run_log.txt"
        if path.exists():
            logs.append(f"===== shard {shard_id} =====\n{path.read_text(encoding='utf-8')}")

    episodes = [evaluate_episode(raw, metadata[raw["sample_id"]]) for raw in raw_episodes]
    summaries = [episode_summary(episode) for episode in episodes]
    crops = [csv_ready({"category": episode["category"], **crop}) for episode in episodes for crop in episode.get("crops", [])]
    turns = [csv_ready(turn) for episode in episodes for turn in episode.get("turns", [])]
    write_jsonl(OUTPUT_DIR / "trajectories.jsonl", episodes)
    write_csv(OUTPUT_DIR / "crop_records.csv", crops, CROP_FIELDS)
    write_csv(OUTPUT_DIR / "episode_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(OUTPUT_DIR / "turn_records.csv", turns, TURN_FIELDS)
    write_jsonl(OUTPUT_DIR / "model_input_audit.jsonl", [audit for episode in episodes for audit in episode.get("input_audit", [])])
    for episode in episodes:
        draw_trajectory(episode, OUTPUT_DIR / "trajectory_vis" / f"{episode['sample_id']}.png")

    overall = aggregate(summaries, "OVERALL")
    by_category = [aggregate([row for row in summaries if row["category"] == category], category) for category in CATEGORIES]
    aggregate_fields = list(overall.keys())
    write_csv(OUTPUT_DIR / "summary_overall.csv", [overall], aggregate_fields)
    write_csv(OUTPUT_DIR / "summary_by_category.csv", by_category, aggregate_fields)
    candidates = review_candidates(summaries, episodes)
    atomic_write_json(OUTPUT_DIR / "manual_review_candidates.json", candidates)

    passed, problems = integrity_checks(ids, episodes)
    category_counts = Counter(episode["category"] for episode in episodes)
    report = [
        "# VisualNeedle Mini-o3 Smoke20 Integrity Report",
        "",
        f"- Overall: **{'PASS' if passed else 'FAIL'}**",
        f"- Frozen sample records: {len(ids)}",
        f"- Episode records: {len(episodes)}",
        f"- Unique episode IDs: {len(set(episode['sample_id'] for episode in episodes))}",
        f"- Category counts: `{dict(category_counts)}`",
        f"- Crop files checked: {len(crops)} raw + {len(crops)} processed",
        "- Inference image provenance: original files only from `images/`; generated observations only from recorded crops.",
        "- `images_bbox/`, GT bbox, category, and GT answer are not accepted by `run_episode`.",
        "- Evaluation metadata was loaded only after each shard completed all inference episodes.",
        "- No duplicate-crop guard, GT guidance, forced-answer turn, crop injection, or sample replacement is enabled.",
        "- Raw model output is retained in each turn record.",
        "",
        "## Problems",
    ]
    report.extend(f"- {problem}" for problem in problems)
    if not problems:
        report.append("- None.")
    (OUTPUT_DIR / "integrity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "run_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    write_config(ids, args.command_string)

    print(json.dumps({"integrity_pass": passed, "overall": overall, "review_candidates": candidates}, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("Integrity checks failed; inspect integrity_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run-shard", action="store_true")
    modes.add_argument("--merge", action="store_true")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--command-string", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    ids = prepare_selection()
    if args.prepare:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_config(ids, args.command_string)
        print(json.dumps({"sample_ids": ids}, ensure_ascii=False, indent=2))
    elif args.run_shard:
        run_shard(args)
    else:
        merge_results(args)


if __name__ == "__main__":
    main()
