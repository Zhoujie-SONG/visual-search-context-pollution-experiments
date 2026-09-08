import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw


DATA_ROOT = Path("/home/songzhoujie/cvpr27/needle_data")
MODEL_PATH = "/mnt/data2/szj/models/Mini-o3-7B-v1-complete"
ENV_NAME = "venv_e2m"
SELECTION_SEED = 42
SAMPLES_PER_CATEGORY = 4
MIN_PIXELS = 40_000
MAX_PIXELS = 2_000_000
MAX_GENERATION_ROUNDS = 12
LIMIT_IMAGE_NUM = 12
ROUND_MAX_NEW_TOKENS = 2048
FORCED_ANSWER_MAX_NEW_TOKENS = 512
TOTAL_OUTPUT_TOKEN_BUDGET = 8192
ROUND_MAX_TIME_SEC = 600
REPEAT_BBOX_IOU = 0.98
REPEAT_BBOX_COUNT = 3
# Qwen's vision processor requires a strict image aspect ratio below 200.
MAX_PROCESSOR_ASPECT_RATIO = 199

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based on the image provided. Output your thinking process within the <think> and </think> tags. Whenever you find anything unclear, you can zoom in a specific region in the given image to see more clearly by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, y1], \"source\": \"original_image\"}</grounding>, where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the region that you want to zoom in, respectively (suppose the width and height of the image are 1.0), and 'source' refers to the image that you zoom in and could be either 'original_image' or 'observation_i'. Once the final answer is confirmed, put it within <answer> and </answer>."""

OBSERVATION_PROMPT = """After the above Action {action_turn}, here is the zoom-in image (Observation {observation_turn}).
Continue your reasoning process inside <think> and </think>. If needed, continue to zoom in on the original image or any observation by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put the final answer inside <answer> and </answer>."""

FINAL_ANSWER_PROMPT = "Stop searching and provide your best concise final answer inside <answer> and </answer>."


@dataclass
class NeedleSample:
    sample_id: str
    category: str
    question: str
    answer: str
    image_path: Path
    bbox_xyxy: List[float]
    bbox_area_fraction: float
    area_quartile: str = ""


EPISODE_FIELDS = [
    "sample_id", "category", "area_quartile", "question", "gt_answer", "pred_answer",
    "normalized_gt_answer", "normalized_pred_answer", "strict_correct", "contains_match",
    "status", "tool_use", "turns_used", "num_crops", "target_hit", "first_target_hit_turn",
    "hit_turn_cap", "forced_answer", "stop_reason", "total_output_tokens",
    "original_image_path", "original_width", "original_height", "processed_width",
    "processed_height", "bbox_area_fraction", "generation_time_total_sec",
    "peak_gpu_memory_mb", "error_message",
]

TURN_FIELDS = [
    "sample_id", "category", "turn_id", "raw_model_output", "parsed_action_type",
    "parsed_answer", "source", "raw_model_coordinate", "bbox_on_source_image",
    "bbox_original", "bbox_valid", "bbox_clamped", "crop_path", "crop_width",
    "crop_height", "target_hit", "iou_with_gt", "normalized_center_distance",
    "input_token_count", "output_token_count", "finish_reason", "generation_time_sec",
    "error_message",
]

CROP_FIELDS = [
    "sample_id", "category", "turn_id", "source", "raw_model_coordinate",
    "bbox_on_source_image", "bbox_original", "crop_path", "crop_width", "crop_height",
    "processed_crop_width", "processed_crop_height", "target_hit", "iou_with_gt",
    "normalized_center_distance",
]

PLAIN_FIELDS = [
    "sample_id", "category", "gt_answer", "pred_answer", "normalized_gt_answer",
    "normalized_pred_answer", "strict_correct", "contains_match", "status",
    "input_token_count", "output_token_count", "finish_reason", "generation_time_sec",
    "error_message",
]


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def append_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_samples(data_root: Path) -> List[NeedleSample]:
    annotation_path = data_root / "visualneedle_300en.jsonl"
    samples = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        image_path = (data_root / row["image_path"]).resolve()
        if image_path.parent != (data_root / "images").resolve():
            raise ValueError(f"Refusing non-original model input path: {image_path}")
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        x1, y1, x2, y2 = [float(v) for v in row["bbox"]]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid bbox for {row['id']}: {row['bbox']} in {width}x{height}")
        samples.append(
            NeedleSample(
                sample_id=row["id"],
                category=row["category"],
                question=row["question"],
                answer=str(row["answer"]),
                image_path=image_path,
                bbox_xyxy=[x1, y1, x2, y2],
                bbox_area_fraction=(x2 - x1) * (y2 - y1) / (width * height),
            )
        )
    return samples


def assign_area_quartiles(
    samples: Sequence[NeedleSample], selected: Sequence[NeedleSample]
) -> None:
    ranks = {}
    category_sizes = {}
    for category in sorted({sample.category for sample in samples}):
        group = sorted(
            (sample for sample in samples if sample.category == category),
            key=lambda sample: (sample.bbox_area_fraction, sample.sample_id),
        )
        category_sizes[category] = len(group)
        ranks.update({sample.sample_id: rank for rank, sample in enumerate(group)})
    for sample in selected:
        quartile = min(3, ranks[sample.sample_id] * 4 // category_sizes[sample.category]) + 1
        sample.area_quartile = f"Q{quartile}"


def deterministic_balanced_selection(
    samples: Sequence[NeedleSample],
    samples_per_category: int,
    required_ids: Sequence[str] = (),
) -> List[NeedleSample]:
    if samples_per_category <= 0:
        raise ValueError("samples_per_category must be positive")
    by_id = {sample.sample_id: sample for sample in samples}
    missing_required = [sample_id for sample_id in required_ids if sample_id not in by_id]
    if missing_required:
        raise ValueError(f"Unknown required sample IDs: {missing_required}")
    if len(set(required_ids)) != len(required_ids):
        raise ValueError("required sample IDs contain duplicates")

    selected = [by_id[sample_id] for sample_id in required_ids]
    selected_ids = set(required_ids)
    categories = sorted({sample.category for sample in samples})
    for category in categories:
        required_count = sum(sample.category == category for sample in selected)
        needed = samples_per_category - required_count
        if needed < 0:
            raise ValueError(
                f"Required IDs already contain {required_count} {category} samples, "
                f"more than requested {samples_per_category}"
            )
        group = sorted(
            (
                sample
                for sample in samples
                if sample.category == category and sample.sample_id not in selected_ids
            ),
            key=lambda sample: (sample.bbox_area_fraction, sample.sample_id),
        )
        if needed > len(group):
            raise ValueError(f"Not enough {category} samples: need {needed}, have {len(group)}")
        for stratum in range(needed):
            lo = stratum * len(group) // needed
            hi = (stratum + 1) * len(group) // needed
            pool = group[lo:hi]
            chosen = min(
                pool,
                key=lambda sample: hashlib.sha256(
                    f"{SELECTION_SEED}|{sample.sample_id}".encode("utf-8")
                ).hexdigest(),
            )
            selected.append(chosen)
            selected_ids.add(chosen.sample_id)
    assign_area_quartiles(samples, selected)
    return selected


def selection_from_id_file(
    samples: Sequence[NeedleSample], sample_ids_file: Path
) -> List[NeedleSample]:
    sample_ids = [line.strip() for line in sample_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {sample_ids_file}")
    by_id = {sample.sample_id: sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Unknown sample IDs in {sample_ids_file}: {missing}")
    selected = [by_id[sample_id] for sample_id in sample_ids]
    assign_area_quartiles(samples, selected)
    return selected


def edge_pad(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Pad with replicated edge pixels, preserving the original content exactly."""
    width, height = image.size
    if target_width < width or target_height < height:
        raise ValueError("edge padding cannot shrink an image")

    left = (target_width - width) // 2
    right = target_width - width - left
    horizontal = Image.new("RGB", (target_width, height))
    horizontal.paste(image, (left, 0))
    if left:
        horizontal.paste(
            image.crop((0, 0, 1, height)).resize((left, height), Image.Resampling.NEAREST),
            (0, 0),
        )
    if right:
        horizontal.paste(
            image.crop((width - 1, 0, width, height)).resize(
                (right, height), Image.Resampling.NEAREST
            ),
            (left + width, 0),
        )

    top = (target_height - height) // 2
    bottom = target_height - height - top
    result = Image.new("RGB", (target_width, target_height))
    result.paste(horizontal, (0, top))
    if top:
        result.paste(
            horizontal.crop((0, 0, target_width, 1)).resize(
                (target_width, top), Image.Resampling.NEAREST
            ),
            (0, 0),
        )
    if bottom:
        result.paste(
            horizontal.crop((0, height - 1, target_width, height)).resize(
                (target_width, bottom), Image.Resampling.NEAREST
            ),
            (0, top + height),
        )
    return result


def process_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    pixels = image.width * image.height
    if pixels > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / pixels)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.NEAREST,
        )
    elif pixels < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / pixels)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.NEAREST,
        )
    if image.width < 28 or image.height < 28:
        scale = 28 / min(image.width, image.height)
        image = image.resize(
            (max(28, int(image.width * scale + 1)), max(28, int(image.height * scale + 1))),
            Image.Resampling.NEAREST,
        )
    if image.width > image.height * MAX_PROCESSOR_ASPECT_RATIO:
        image = edge_pad(image, image.width, math.ceil(image.width / MAX_PROCESSOR_ASPECT_RATIO))
    elif image.height > image.width * MAX_PROCESSOR_ASPECT_RATIO:
        image = edge_pad(image, math.ceil(image.height / MAX_PROCESSOR_ASPECT_RATIO), image.height)
    if image.width * image.height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (image.width * image.height))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.NEAREST,
        )
    return image.convert("RGB")


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower().strip()
    text = text.replace("'", "").replace("’", "")
    chars = []
    for char in text:
        category = unicodedata.category(char)
        chars.append(" " if category.startswith("P") or category.startswith("S") else char)
    text = "".join(chars)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_metrics(prediction: str, answer: str) -> Tuple[str, str, bool, bool]:
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(answer)
    exact = bool(pred_norm) and pred_norm == gt_norm
    contains = bool(pred_norm and gt_norm) and (pred_norm in gt_norm or gt_norm in pred_norm)
    return pred_norm, gt_norm, exact, contains


def parse_answer(text: str) -> str:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.IGNORECASE | re.DOTALL)
    return matches[-1].strip() if matches else ""


def parse_grounding(text: str) -> Dict[str, Any]:
    matches = re.findall(r"<grounding>\s*(\{.*?\})\s*</grounding>", text or "", flags=re.DOTALL)
    if not matches:
        return {"type": "none"}
    payload = matches[-1]
    try:
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            obj = ast.literal_eval(payload)
        bbox = obj["bbox_2d"]
        source = str(obj["source"])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox_2d must contain four values")
        bbox = [float(value) for value in bbox]
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError("bbox_2d contains a non-finite value")
        return {"type": "grounding", "bbox_2d": bbox, "source": source}
    except Exception as exc:
        return {"type": "invalid_grounding", "error": str(exc), "payload": payload}


def bbox_iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def convert_grounding(
    grounding: Dict[str, Any],
    observations: Sequence[Image.Image],
    crop_records: Sequence[Dict[str, Any]],
) -> Tuple[Image.Image, List[int], List[float], bool]:
    source = grounding["source"]
    if source == "original_image":
        observation_id = 0
    else:
        match = re.fullmatch(r"observation_([0-9]+)", source)
        if not match:
            raise ValueError(f"Invalid source: {source}")
        observation_id = int(match.group(1))
    if observation_id >= len(observations):
        raise ValueError(f"Source {source} is not available")
    source_image = observations[observation_id]
    width, height = source_image.size
    raw = grounding["bbox_2d"]
    pixel_float = [raw[0] * width, raw[1] * height, raw[2] * width, raw[3] * height]
    pixel = [int(value) for value in pixel_float]
    clamped = [
        max(0, min(width - 1, pixel[0])),
        max(0, min(height - 1, pixel[1])),
        max(1, min(width, pixel[2])),
        max(1, min(height, pixel[3])),
    ]
    if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
        raise ValueError(f"Invalid crop after clamping: {clamped}")
    if source == "original_image":
        original_bbox = [float(value) for value in clamped]
    else:
        parent = crop_records[observation_id - 1]
        px1, py1, px2, py2 = parent["bbox_original"]
        sx = (px2 - px1) / width
        sy = (py2 - py1) / height
        x1, y1, x2, y2 = clamped
        original_bbox = [px1 + x1 * sx, py1 + y1 * sy, px1 + x2 * sx, py1 + y2 * sy]
    return source_image, clamped, original_bbox, pixel != clamped


def crop_target_metrics(
    bbox: Sequence[float], gt_bbox: Sequence[float], image_size: Tuple[int, int]
) -> Tuple[bool, float, float]:
    gx = (gt_bbox[0] + gt_bbox[2]) / 2
    gy = (gt_bbox[1] + gt_bbox[3]) / 2
    hit = bbox[0] <= gx <= bbox[2] and bbox[1] <= gy <= bbox[3]
    iou = bbox_iou_xyxy(bbox, gt_bbox)
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    distance = math.hypot(cx - gx, cy - gy) / max(1.0, math.hypot(*image_size))
    return hit, iou, distance


def repeated_bbox(crops: Sequence[Dict[str, Any]]) -> bool:
    if len(crops) < REPEAT_BBOX_COUNT:
        return False
    recent = crops[-REPEAT_BBOX_COUNT:]
    anchor = recent[0]["bbox_original"]
    return all(bbox_iou_xyxy(anchor, crop["bbox_original"]) >= REPEAT_BBOX_IOU for crop in recent[1:])


class HFBackend:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(self, messages: Sequence[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                images.extend(
                    item["image"]
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "image"
                )
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        input_length = int(inputs["input_ids"].shape[1])
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        start = time.perf_counter()
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    max_time=ROUND_MAX_TIME_SEC,
                    return_dict_in_generate=True,
                )
            elapsed = time.perf_counter() - start
            sequence = output.sequences
            new_ids = sequence[:, input_length:]
            raw = self.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            last_token = int(sequence[0, -1].item())
            finish_reason = (
                "eos"
                if last_token in eos_ids
                else "length"
                if int(new_ids.shape[1]) >= max_new_tokens
                else "time_or_unknown"
            )
            return {
                "raw": raw,
                "input_token_count": input_length,
                "output_token_count": int(new_ids.shape[1]),
                "finish_reason": finish_reason,
                "generation_time_sec": elapsed,
            }
        finally:
            del inputs
            self.torch.cuda.empty_cache()


def initial_messages(image: Image.Image, question: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Question: {question}"},
            ],
        },
    ]


def turn_row(sample: NeedleSample, turn_id: int, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "turn_id": turn_id,
        "raw_model_output": result.get("raw", ""),
        "input_token_count": result.get("input_token_count", ""),
        "output_token_count": result.get("output_token_count", ""),
        "finish_reason": result.get("finish_reason", ""),
        "generation_time_sec": f"{result.get('generation_time_sec', 0):.4f}",
    }


def run_plain(backend: HFBackend, sample: NeedleSample) -> Dict[str, Any]:
    with Image.open(sample.image_path) as handle:
        processed = process_image(handle.convert("RGB"))
    messages = [
        {"role": "system", "content": "Answer the visual question concisely and put only the final answer inside <answer> and </answer>."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": processed},
                {"type": "text", "text": f"Question: {sample.question}"},
            ],
        },
    ]
    start = time.perf_counter()
    try:
        result = backend.generate(messages, FORCED_ANSWER_MAX_NEW_TOKENS)
        prediction = parse_answer(result["raw"])
        pred_norm, gt_norm, exact, contains = answer_metrics(prediction, sample.answer)
        status = "ok" if prediction else "invalid"
        error = ""
    except (RuntimeError, ValueError) as exc:
        result = {}
        prediction = ""
        pred_norm, gt_norm, exact, contains = answer_metrics("", sample.answer)
        status = "oom" if "out of memory" in str(exc).lower() else "error"
        error = str(exc)
    return {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "gt_answer": sample.answer,
        "pred_answer": prediction,
        "normalized_gt_answer": gt_norm,
        "normalized_pred_answer": pred_norm,
        "strict_correct": str(exact),
        "contains_match": str(contains),
        "status": status,
        "input_token_count": result.get("input_token_count", ""),
        "output_token_count": result.get("output_token_count", ""),
        "finish_reason": result.get("finish_reason", ""),
        "generation_time_sec": f"{time.perf_counter() - start:.4f}",
        "error_message": error,
    }


def run_episode(backend: HFBackend, sample: NeedleSample, output_dir: Path) -> Dict[str, Any]:
    start = time.perf_counter()
    backend.torch.cuda.reset_peak_memory_stats()
    with Image.open(sample.image_path) as handle:
        original = handle.convert("RGB")
    processed_original = process_image(original)
    messages = initial_messages(processed_original, sample.question)
    observations = [original]
    crops: List[Dict[str, Any]] = []
    turns: List[Dict[str, Any]] = []
    final_answer = ""
    status = "ok"
    error_message = ""
    forced_answer = False
    hit_turn_cap = False
    stop_reason = ""
    total_output_tokens = 0
    crop_dir = output_dir / "crops" / sample.sample_id
    crop_dir.mkdir(parents=True, exist_ok=True)

    for round_idx in range(MAX_GENERATION_ROUNDS):
        remaining = TOTAL_OUTPUT_TOKEN_BUDGET - total_output_tokens - FORCED_ANSWER_MAX_NEW_TOKENS
        if remaining <= 0:
            hit_turn_cap = True
            stop_reason = "total_output_token_budget"
            break
        try:
            result = backend.generate(messages, min(ROUND_MAX_NEW_TOKENS, remaining))
            total_output_tokens += int(result["output_token_count"])
            raw = result["raw"]
            answer = parse_answer(raw)
            grounding = parse_grounding(raw)
            turn = turn_row(sample, round_idx, result)
            turn.update(
                {
                    "parsed_action_type": "answer" if answer else grounding["type"],
                    "parsed_answer": answer,
                    "source": grounding.get("source", ""),
                    "raw_model_coordinate": json_dump(grounding.get("bbox_2d", "")),
                    "bbox_valid": "",
                    "bbox_clamped": "",
                    "error_message": "",
                }
            )
            if answer:
                final_answer = answer
                turns.append(turn)
                stop_reason = "answer"
                break
            if grounding["type"] == "invalid_grounding":
                turn["bbox_valid"] = "False"
                turn["error_message"] = grounding["error"]
                turns.append(turn)
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"The grounding call was invalid: {grounding['error']}. Correct it or provide a final answer."},
                    ]
                )
                continue
            if grounding["type"] != "grounding":
                turns.append(turn)
                status = "invalid"
                stop_reason = "unparsed_output"
                break
            if len(observations) >= LIMIT_IMAGE_NUM or round_idx == MAX_GENERATION_ROUNDS - 1:
                turns.append(turn)
                hit_turn_cap = True
                stop_reason = "turn_or_image_cap"
                break
            try:
                source_image, source_bbox, original_bbox, was_clamped = convert_grounding(
                    grounding, observations, crops
                )
                crop_raw = source_image.crop(tuple(source_bbox)).convert("RGB")
                processed_crop = process_image(crop_raw)
                target_hit, target_iou, center_distance = crop_target_metrics(
                    original_bbox, sample.bbox_xyxy, original.size
                )
                crop_path = crop_dir / f"turn_{round_idx + 1:02d}.png"
                crop_raw.save(crop_path)
                crop = {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "turn_id": round_idx,
                    "source": grounding["source"],
                    "raw_model_coordinate": grounding["bbox_2d"],
                    "bbox_on_source_image": source_bbox,
                    "bbox_original": original_bbox,
                    "crop_path": str(crop_path),
                    "crop_width": crop_raw.width,
                    "crop_height": crop_raw.height,
                    "processed_crop_width": processed_crop.width,
                    "processed_crop_height": processed_crop.height,
                    "target_hit": target_hit,
                    "iou_with_gt": target_iou,
                    "normalized_center_distance": center_distance,
                }
                crops.append(crop)
                observations.append(crop_raw)
                turn.update(
                    {
                        "bbox_on_source_image": json_dump(source_bbox),
                        "bbox_original": json_dump(original_bbox),
                        "bbox_valid": "True",
                        "bbox_clamped": str(was_clamped),
                        "crop_path": str(crop_path),
                        "crop_width": crop_raw.width,
                        "crop_height": crop_raw.height,
                        "target_hit": str(target_hit),
                        "iou_with_gt": f"{target_iou:.8f}",
                        "normalized_center_distance": f"{center_distance:.8f}",
                    }
                )
                turns.append(turn)
                if repeated_bbox(crops):
                    hit_turn_cap = True
                    stop_reason = "repeated_bbox_guard"
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": processed_crop},
                                {
                                    "type": "text",
                                    "text": OBSERVATION_PROMPT.format(
                                        action_turn=round_idx,
                                        observation_turn=round_idx + 1,
                                    ),
                                },
                            ],
                        },
                    ]
                )
            except Exception as exc:
                turn["bbox_valid"] = "False"
                turn["error_message"] = str(exc)
                turns.append(turn)
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"The crop failed: {exc}. Correct it or provide a final answer."},
                    ]
                )
        except (RuntimeError, ValueError) as exc:
            error_message = str(exc)
            status = "oom" if "out of memory" in error_message.lower() else "error"
            stop_reason = status
            break

    if not final_answer and status == "ok":
        forced_answer = True
        messages.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
        remaining = max(1, TOTAL_OUTPUT_TOKEN_BUDGET - total_output_tokens)
        try:
            result = backend.generate(messages, min(FORCED_ANSWER_MAX_NEW_TOKENS, remaining))
            total_output_tokens += int(result["output_token_count"])
            final_answer = parse_answer(result["raw"])
            turn = turn_row(sample, len(turns), result)
            turn.update(
                {
                    "parsed_action_type": "answer" if final_answer else "invalid",
                    "parsed_answer": final_answer,
                    "error_message": "",
                }
            )
            turns.append(turn)
            if not final_answer:
                status = "invalid"
                stop_reason = f"{stop_reason}+forced_answer_invalid"
        except (RuntimeError, ValueError) as exc:
            error_message = str(exc)
            status = "oom" if "out of memory" in error_message.lower() else "error"
            stop_reason = status

    pred_norm, gt_norm, exact, contains = answer_metrics(final_answer, sample.answer)
    first_hits = [crop["turn_id"] for crop in crops if crop["target_hit"]]
    episode = {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "area_quartile": sample.area_quartile,
        "question": sample.question,
        "gt_answer": sample.answer,
        "pred_answer": final_answer,
        "normalized_gt_answer": gt_norm,
        "normalized_pred_answer": pred_norm,
        "strict_correct": str(exact),
        "contains_match": str(contains),
        "status": status,
        "tool_use": str(bool(crops)),
        "turns_used": len(turns),
        "num_crops": len(crops),
        "target_hit": str(bool(first_hits)),
        "first_target_hit_turn": min(first_hits) if first_hits else "",
        "hit_turn_cap": str(hit_turn_cap),
        "forced_answer": str(forced_answer),
        "stop_reason": stop_reason,
        "total_output_tokens": total_output_tokens,
        "original_image_path": str(sample.image_path),
        "original_width": original.width,
        "original_height": original.height,
        "processed_width": processed_original.width,
        "processed_height": processed_original.height,
        "bbox_area_fraction": f"{sample.bbox_area_fraction:.10f}",
        "generation_time_total_sec": f"{time.perf_counter() - start:.4f}",
        "peak_gpu_memory_mb": int(backend.torch.cuda.max_memory_allocated() / 1024**2),
        "error_message": error_message,
    }
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript = "\n\n".join(
        f"TURN {turn['turn_id']} [{turn.get('parsed_action_type', '')}]\n{turn['raw_model_output']}"
        for turn in turns
    )
    (transcript_dir / f"{sample.sample_id}.txt").write_text(transcript, encoding="utf-8")
    return {"episode": episode, "turns": turns, "crops": crops, "gt_bbox_xyxy": sample.bbox_xyxy}


def load_completed(path: Path) -> Dict[str, Dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            completed[item["episode"]["sample_id"]] = item
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def append_trajectory(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare(output_dir: Path, samples: Sequence[NeedleSample], model_path: str, num_shards: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "sample_id": sample.sample_id,
            "category": sample.category,
            "area_quartile": sample.area_quartile,
            "bbox_area_fraction": sample.bbox_area_fraction,
            "question": sample.question,
            "answer": sample.answer,
            "image_path": str(sample.image_path),
            "bbox_xyxy": sample.bbox_xyxy,
        }
        for sample in samples
    ]
    settings = {
        "dataset": "VisualNeedle 300 EN",
        "data_root": str(DATA_ROOT),
        "model_path": model_path,
        "env_name": ENV_NAME,
        "selection_seed": SELECTION_SEED,
        "samples_per_category": dict(
            sorted(Counter(sample.category for sample in samples).items())
        ),
        "num_samples": len(samples),
        "num_shards": num_shards,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "do_sample": False,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_processor_aspect_ratio": MAX_PROCESSOR_ASPECT_RATIO,
        "extreme_aspect_crop_handling": "edge replication padding without stretching",
        "max_generation_rounds": MAX_GENERATION_ROUNDS,
        "limit_image_num_including_original": LIMIT_IMAGE_NUM,
        "round_max_new_tokens": ROUND_MAX_NEW_TOKENS,
        "forced_answer_max_new_tokens": FORCED_ANSWER_MAX_NEW_TOKENS,
        "total_output_token_budget": TOTAL_OUTPUT_TOKEN_BUDGET,
        "round_max_time_sec": ROUND_MAX_TIME_SEC,
        "system_prompt": SYSTEM_PROMPT,
        "observation_prompt": OBSERVATION_PROMPT,
        "model_inputs": "images/ only; images_bbox/ is never loaded",
    }
    (output_dir / "smoke_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "smoke_sample_ids.txt").write_text(
        "\n".join(sample.sample_id for sample in samples) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(samples)} samples in {output_dir}")


def run_shard(
    output_dir: Path,
    samples: Sequence[NeedleSample],
    model_path: str,
    shard_id: int,
    num_shards: int,
    only_sample_id: str,
    print_transcript: bool,
) -> None:
    shard_dir = output_dir / "shards" / f"shard{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    assigned = [sample for index, sample in enumerate(samples) if index % num_shards == shard_id]
    if only_sample_id:
        assigned = [sample for sample in samples if sample.sample_id == only_sample_id]
        if not assigned:
            raise ValueError(f"Unknown selected sample: {only_sample_id}")
    trajectory_path = shard_dir / "trajectories.jsonl"
    completed = load_completed(trajectory_path)
    plain_path = shard_dir / "plain_vqa_anchor.csv"
    plain_completed = set()
    if plain_path.exists():
        with plain_path.open("r", encoding="utf-8", newline="") as handle:
            plain_completed = {row["sample_id"] for row in csv.DictReader(handle)}
    print(f"shard={shard_id}/{num_shards} assigned={len(assigned)} completed={len(completed)}", flush=True)
    backend = HFBackend(model_path)
    for index, sample in enumerate(assigned, start=1):
        if sample.sample_id not in plain_completed:
            plain = run_plain(backend, sample)
            append_csv(plain_path, [plain], PLAIN_FIELDS)
            plain_completed.add(sample.sample_id)
            print(
                f"[{index}/{len(assigned)}] {sample.sample_id} plain status={plain['status']} correct={plain['strict_correct']}",
                flush=True,
            )
        if sample.sample_id in completed:
            print(f"[{index}/{len(assigned)}] {sample.sample_id} agent skipped=complete", flush=True)
            continue
        item = run_episode(backend, sample, output_dir)
        append_trajectory(trajectory_path, item)
        append_csv(shard_dir / "episode_summary.csv", [item["episode"]], EPISODE_FIELDS)
        append_csv(shard_dir / "turns.csv", item["turns"], TURN_FIELDS)
        append_csv(shard_dir / "crop_records.csv", item["crops"], CROP_FIELDS)
        episode = item["episode"]
        print(
            f"[{index}/{len(assigned)}] {sample.sample_id} agent status={episode['status']} "
            f"correct={episode['strict_correct']} crops={episode['num_crops']} target_hit={episode['target_hit']}",
            flush=True,
        )
        if print_transcript:
            print((output_dir / "transcripts" / f"{sample.sample_id}.txt").read_text(encoding="utf-8"), flush=True)


def summary_row(label: str, episodes: Sequence[Dict[str, Any]], plain: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    plain_by_id = {row["sample_id"]: row for row in plain}
    crop_counts = [int(row["num_crops"]) for row in episodes]
    first_hits = [int(row["first_target_hit_turn"]) for row in episodes if str(row["first_target_hit_turn"]) != ""]
    count = len(episodes)
    return {
        "category": label,
        "num_samples": count,
        "agent_strict_accuracy": sum(row["strict_correct"] == "True" for row in episodes) / count if count else "",
        "agent_contains_accuracy": sum(row["contains_match"] == "True" for row in episodes) / count if count else "",
        "plain_strict_accuracy": sum(
            plain_by_id.get(row["sample_id"], {}).get("strict_correct") == "True" for row in episodes
        ) / count if count else "",
        "plain_contains_accuracy": sum(
            plain_by_id.get(row["sample_id"], {}).get("contains_match") == "True" for row in episodes
        ) / count if count else "",
        "mean_crops": statistics.mean(crop_counts) if crop_counts else "",
        "median_crops": statistics.median(crop_counts) if crop_counts else "",
        "target_hit_rate": sum(row["target_hit"] == "True" for row in episodes) / count if count else "",
        "mean_first_target_hit_turn": statistics.mean(first_hits) if first_hits else "",
        "turn_cap_rate": sum(row["hit_turn_cap"] == "True" for row in episodes) / count if count else "",
        "forced_answer_rate": sum(row["forced_answer"] == "True" for row in episodes) / count if count else "",
        "invalid_rate": sum(row["status"] == "invalid" for row in episodes) / count if count else "",
        "oom_rate": sum(row["status"] == "oom" for row in episodes) / count if count else "",
    }


def render_overlays(output_dir: Path, trajectories: Sequence[Dict[str, Any]]) -> None:
    overlay_dir = output_dir / "trajectory_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for item in trajectories:
        episode = item["episode"]
        with Image.open(episode["original_image_path"]) as handle:
            original = handle.convert("RGB")
        canvas = process_image(original)
        sx, sy = canvas.width / original.width, canvas.height / original.height
        draw = ImageDraw.Draw(canvas)
        gx1, gy1, gx2, gy2 = item["gt_bbox_xyxy"]
        draw.rectangle((gx1 * sx, gy1 * sy, gx2 * sx, gy2 * sy), outline="cyan", width=4)
        for crop in item["crops"]:
            x1, y1, x2, y2 = crop["bbox_original"]
            color = "lime" if crop["target_hit"] else "red"
            draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=color, width=3)
            draw.text((x1 * sx + 3, y1 * sy + 3), str(crop["turn_id"] + 1), fill=color)
        canvas.save(overlay_dir / f"{episode['sample_id']}.png")


def merge_outputs(output_dir: Path, expected_samples: Sequence[NeedleSample]) -> None:
    trajectories_by_id = {}
    for path in sorted((output_dir / "shards").glob("shard*/trajectories.jsonl")):
        trajectories_by_id.update(load_completed(path))
    expected_ids = {sample.sample_id for sample in expected_samples}
    missing = sorted(expected_ids - set(trajectories_by_id))
    extra = sorted(set(trajectories_by_id) - expected_ids)
    if missing or extra:
        raise RuntimeError(f"Cannot merge: missing={missing}, extra={extra}")
    trajectories = [trajectories_by_id[sample.sample_id] for sample in expected_samples]
    episodes = [item["episode"] for item in trajectories]
    turns = [turn for item in trajectories for turn in item["turns"]]
    crops = [crop for item in trajectories for crop in item["crops"]]
    plain_by_id = {}
    for path in sorted((output_dir / "shards").glob("shard*/plain_vqa_anchor.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                plain_by_id[row["sample_id"]] = row
    missing_plain = sorted(expected_ids - set(plain_by_id))
    if missing_plain:
        raise RuntimeError(f"Cannot merge plain VQA: missing={missing_plain}")
    plain = [plain_by_id[sample.sample_id] for sample in expected_samples]
    write_csv(output_dir / "episode_summary.csv", episodes, EPISODE_FIELDS)
    write_csv(output_dir / "turns.csv", turns, TURN_FIELDS)
    write_csv(output_dir / "crop_records.csv", crops, CROP_FIELDS)
    write_csv(output_dir / "plain_vqa_anchor.csv", plain, PLAIN_FIELDS)
    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for item in trajectories:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    categories = sorted({row["category"] for row in episodes})
    summary = [summary_row("ALL", episodes, plain)]
    summary.extend(
        summary_row(category, [row for row in episodes if row["category"] == category], plain)
        for category in categories
    )
    summary_fields = list(summary[0])
    write_csv(output_dir / "category_summary.csv", summary, summary_fields)
    distribution = []
    for crop_count in sorted({int(row["num_crops"]) for row in episodes}):
        distribution.append(
            {
                "num_crops": crop_count,
                "num_episodes": sum(int(row["num_crops"]) == crop_count for row in episodes),
            }
        )
    write_csv(output_dir / "crop_count_distribution.csv", distribution, ["num_crops", "num_episodes"])
    render_overlays(output_dir, trajectories)
    overall = summary[0]
    report = [
        "# E2-M VisualNeedle Smoke Report",
        "",
        f"- Samples: {len(episodes)} ({dict(sorted(Counter(row['category'] for row in episodes).items()))})",
        f"- Agent strict accuracy: {overall['agent_strict_accuracy']:.4f}",
        f"- Agent contains-match accuracy: {overall['agent_contains_accuracy']:.4f}",
        f"- Plain VQA strict accuracy: {overall['plain_strict_accuracy']:.4f}",
        f"- Plain VQA contains-match accuracy: {overall['plain_contains_accuracy']:.4f}",
        f"- Mean crops: {overall['mean_crops']:.3f}",
        f"- Median crops: {overall['median_crops']:.3f}",
        f"- Target-hit rate: {overall['target_hit_rate']:.4f}",
        f"- Turn-cap rate: {overall['turn_cap_rate']:.4f}",
        f"- Invalid rate: {overall['invalid_rate']:.4f}",
        f"- OOM rate: {overall['oom_rate']:.4f}",
        "- Accuracy is normalized exact match. Semantic human review is still required.",
        "- A target_non_hit crop is not automatically a failed search step because intermediate clue boxes are unavailable.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"Merged outputs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-o3 VisualNeedle evaluation runner")
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--only_sample_id", default="")
    parser.add_argument("--print_transcript", action="store_true")
    parser.add_argument("--samples_per_category", type=int, default=SAMPLES_PER_CATEGORY)
    parser.add_argument("--required_sample_ids_file", type=Path)
    parser.add_argument("--sample_ids_file", type=Path)
    args = parser.parse_args()

    if args.required_sample_ids_file and args.sample_ids_file:
        raise ValueError("Use only one of --required_sample_ids_file and --sample_ids_file")
    all_samples = load_samples(args.data_root)
    if args.sample_ids_file:
        samples = selection_from_id_file(all_samples, args.sample_ids_file)
    else:
        required_ids = []
        if args.required_sample_ids_file:
            required_ids = [
                line.strip()
                for line in args.required_sample_ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        samples = deterministic_balanced_selection(
            all_samples,
            args.samples_per_category,
            required_ids,
        )
    if not samples:
        raise RuntimeError("No samples selected")
    if args.prepare_only:
        prepare(args.output_dir, samples, args.model, args.num_shards)
        return
    if args.merge_only:
        merge_outputs(args.output_dir, samples)
        return
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    run_shard(
        args.output_dir,
        samples,
        args.model,
        args.shard_id,
        args.num_shards,
        args.only_sample_id,
        args.print_transcript,
    )


if __name__ == "__main__":
    main()
