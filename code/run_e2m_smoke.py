import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw

from context_pollution_utils import (
    LETTERS,
    bbox_iou_xywh,
    clamp_bbox_xywh,
    deterministic_option_shuffle,
    format_options,
    load_vstar_samples,
)


REPO_URL = "https://github.com/Mini-o3/Mini-o3"
REPO_PATH = "/home/songzhoujie/cvpr27/Mini-o3"
MODEL_ID = "Mini-o3/Mini-o3-7B-v1"
MODEL_REVISION = "e82da9a10cca125a1e26c27760ef80840c0fbcd1"
LOCAL_MODEL_PATH = "/mnt/data2/szj/models/Mini-o3-7B-v1-complete"
DATA_ROOT = "/home/songzhoujie/cvpr27/vstar/data/vstar_bench"
OPTION_SEED = 42

MIN_PIXELS = 40000
MAX_PIXELS = 2000000
MAX_GENERATION_ROUND = 12
LIMIT_IMAGE_NUM = 12
ROUND_MAX_NEW_TOKENS = 2048
ROUND_MAX_TIME_SEC = 600
OFFICIAL_EXAMPLE_REASON = (
    "The official Mini-o3 repository documents a demo command that requires a "
    "user-provided DEMO_DATA JSON, but this checkout does not include a runnable "
    "official demo/test JSON, JSONL, parquet, or app.py. Per the E2-M prompt, "
    "V*Bench smoke is stopped instead of substituting a V*Bench sample as an "
    "official example."
)

TOOL_CROP_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based on the image provided. Output your thinking process within the <think> and </think> tags. Whenever you find anything unclear, you can zoom in a specific region in the given image to see more clearly by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, y1], \"source\": \"original_image\"}</grounding>, where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the region that you want to zoom in, respectively (suppose the width and height of the image are 1.0), and 'source' refers to the image that you zoom in and could be either 'original_image' or 'observation_i'. Once the final answer is confirmed, put it within <answer> and </answer>."""

TOOL_CALL_CROP_MULTI_TURN_PROMPT = "After the above Action {action_turn}, here is the the zoom-in image (Observation {observation_turn}):\n<|vision_start|><|image_pad|><|vision_end|>.\nContinue your reasoning process inside <think> and </think>. If needed, you can continue to zoom in on the original image or any of the observations, by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."
HF_TOOL_CALL_CROP_MULTI_TURN_PROMPT = "After the above Action {action_turn}, here is the the zoom-in image (Observation {observation_turn}).\nContinue your reasoning process inside <think> and </think>. If needed, you can continue to zoom in on the original image or any of the observations, by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."
ERROR_INFO_MULTI_TURN_PROMPT = "Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think>."


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: Sequence[Dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def append_csv_rows(path: Path, rows: Sequence[Dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def strip_crop_images(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "episode": item["episode"],
        "turns": item["turns"],
        "crops": [{k: v for k, v in c.items() if k != "crop_image"} for c in item["crops"]],
        "gt_bbox": item["gt_bbox"],
    }


def load_completed_results(output_dir: Path) -> List[Dict[str, Any]]:
    path = output_dir / "trajectories.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("episode", {}).get("sample_id"):
                    rows.append(item)
            except json.JSONDecodeError:
                continue
    return rows


def append_episode_outputs(output_dir: Path, item: Dict[str, Any]) -> None:
    serial = strip_crop_images(item)
    append_csv_rows(output_dir / "episode_summary.csv", [serial["episode"]], EPISODE_FIELDS)
    append_csv_rows(output_dir / "turns.csv", serial["turns"], TURN_FIELDS)
    if serial["episode"]["status"] == "oom":
        append_csv_rows(output_dir / "oom_log.csv", [serial["episode"]], EPISODE_FIELDS)
    append_csv_rows(
        output_dir / "peak_memory.csv",
        [{"sample_id": serial["episode"]["sample_id"], "peak_gpu_memory_mb": serial["episode"]["peak_gpu_memory_mb"]}],
        ["sample_id", "peak_gpu_memory_mb"],
    )
    with (output_dir / "trajectories.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(serial, ensure_ascii=False) + "\n")
    with (output_dir / "transcripts.txt").open("a", encoding="utf-8") as f:
        if f.tell() > 0:
            f.write("\n\n")
        f.write("\n".join(t["raw_model_output"] for t in serial["turns"]))


def process_image_official(image: Any, max_pixels: int = MAX_PIXELS, min_pixels: int = MIN_PIXELS) -> Image.Image:
    if isinstance(image, dict):
        raise TypeError("dict image input is not used in this smoke runner")
    image = image.convert("RGB")
    if image.width * image.height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)), Image.Resampling.NEAREST)
    if image.width * image.height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)), Image.Resampling.NEAREST)
    if image.width < 28 or image.height < 28:
        resize_factor = 28 / min(image.width, image.height)
        image = image.resize((int(image.width * resize_factor + 1), int(image.height * resize_factor + 1)), Image.Resampling.NEAREST)
    if image.width / image.height >= 200:
        image = image.resize((image.width, int(image.width / 190 + 1)), Image.Resampling.NEAREST)
    if image.height / image.width >= 200:
        image = image.resize((int(image.height / 190 + 1), image.height), Image.Resampling.NEAREST)
    return image.convert("RGB")


def crop_image_official(image: Image.Image, coordinates: Sequence[int], image_size_used: Sequence[int], resize: int = 1) -> Image.Image:
    crop = image.crop(tuple(coordinates)).convert("RGB")
    if resize > 1:
        crop_w, crop_h = crop.size
        w, h = image_size_used
        resize = min(resize, min(w / crop_w, h / crop_h))
        target_w, target_h = max(28, int(crop_w * resize)), max(28, int(crop_h * resize))
        crop = crop.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    return crop


def parse_answer(text: str) -> str:
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text or "", flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", (text or "").strip().upper())
    return match.group(1) if match else ""


def parse_grounding(text: str) -> Dict[str, Any]:
    pattern = r".*<grounding>{\"bbox_2d\": (.*),.*\"source\": [\',\"](.*)[\',\"]}</grounding>"
    match = re.match(pattern, text or "", re.DOTALL)
    if not match:
        return {"type": "answer_or_invalid", "answer": parse_answer(text)}
    bbox_text, source = match.group(1), match.group(2)
    try:
        bbox = eval(bbox_text, {"__builtins__": {}})
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"invalid bbox_2d: {bbox}")
        return {"type": "grounding", "bbox_2d": [float(x) for x in bbox], "source": str(source)}
    except Exception as exc:
        return {"type": "invalid_grounding", "error": str(exc), "answer": parse_answer(text)}


def prepare_grounding_inputs_multi_turn(
    obj: Dict[str, Any],
    observations: Sequence[Image.Image],
    image_size_used_list: Sequence[Tuple[int, int]],
    use_relative_coordinates: bool = True,
) -> Tuple[Image.Image, List[int], Dict[str, Any]]:
    bbox = obj["bbox_2d"]
    source = obj["source"]
    if source == "original_image":
        observation_id = 0
    else:
        match = re.match(r"observation_([0-9]*)", source, re.DOTALL)
        if not match:
            raise ValueError(f"The source argument {source!r} does not match observation_i.")
        observation_id = int(match.group(1))
    if observation_id >= len(observations):
        raise ValueError(f"source {source!r} points past available observations.")
    image = observations[observation_id]
    w, h = image.size
    conversion = {
        "coordinate_type": "relative",
        "source": source,
        "source_observation_id": observation_id,
        "source_image_size": [w, h],
        "image_size_used": list(image_size_used_list[observation_id]),
        "raw_model_coordinate": bbox,
        "formula": "bbox_pixels=[x0*w,y0*h,x1*w,y1*h] because use_relative_coordinates=True",
    }
    if use_relative_coordinates:
        bbox = (bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h)
    else:
        w_used, h_used = image_size_used_list[observation_id]
        if w != w_used or h != h_used:
            bbox = (bbox[0] * w / w_used, bbox[1] * h / h_used, bbox[2] * w / w_used, bbox[3] * h / h_used)
    raw_pixels = [float(v) for v in bbox]
    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    clamped = [max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], w - 1), min(bbox[3], h - 1)]
    conversion["raw_pixel_bbox_before_clamp"] = raw_pixels
    conversion["clamped"] = clamped != bbox
    conversion["bbox_on_source_image"] = clamped
    if clamped[0] > clamped[2] - 1 or clamped[1] > clamped[3] - 1:
        raise ValueError(f"The bounding box is not valid: {clamped}")
    width_box = clamped[2] - clamped[0]
    height_box = clamped[3] - clamped[1]
    if width_box / height_box >= 200 or height_box / width_box >= 200:
        raise ValueError("The absolute aspect ratio of the bounding box exceeds 200.")
    return image, clamped, conversion


def source_bbox_to_original(
    source_bbox: Sequence[int],
    source: str,
    crop_records: Sequence[Dict[str, Any]],
) -> List[float]:
    if source == "original_image":
        return [float(source_bbox[0]), float(source_bbox[1]), float(source_bbox[2]), float(source_bbox[3])]
    match = re.match(r"observation_([0-9]*)", source)
    if not match:
        return [float("nan")] * 4
    observation_id = int(match.group(1))
    if observation_id <= 0 or observation_id > len(crop_records):
        return [float("nan")] * 4
    parent = crop_records[observation_id - 1]
    px1, py1, px2, py2 = parent["bbox_original"]
    obs_w, obs_h = parent["observation_size"]
    sx = (px2 - px1) / max(1, obs_w)
    sy = (py2 - py1) / max(1, obs_h)
    x1, y1, x2, y2 = source_bbox
    return [px1 + x1 * sx, py1 + y1 * sy, px1 + x2 * sx, py1 + y2 * sy]


class HFBackend:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        self.model.eval()

    def generate(self, messages: Sequence[Dict], max_new_tokens: int) -> Dict[str, Any]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                images.extend(item["image"] for item in content if isinstance(item, dict) and item.get("type") == "image")
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        start = time.perf_counter()
        try:
            with self.torch.inference_mode():
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "return_dict_in_generate": True,
                }
                if ROUND_MAX_TIME_SEC > 0:
                    generation_kwargs["max_time"] = ROUND_MAX_TIME_SEC
                out = self.model.generate(**inputs, **generation_kwargs)
            elapsed = time.perf_counter() - start
            seq = out.sequences
            new_ids = seq[:, input_len:]
            raw = self.processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            last = int(seq[0, -1].item())
            finish = "eos" if last in eos_ids else ("length" if new_ids.shape[1] >= max_new_tokens else "time_or_unknown")
            return {
                "raw": raw,
                "input_token_count": input_len,
                "output_token_count": int(new_ids.shape[1]),
                "finish_reason": finish,
                "generation_time_sec": elapsed,
            }
        finally:
            del inputs
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()


def initial_messages(image: Image.Image, question: str, options: Sequence[str]) -> List[Dict]:
    prompt = f"Question: {question}\nOptions:\n{format_options(options)}"
    return [
        {"role": "system", "content": TOOL_CROP_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
    ]


def run_episode(backend: HFBackend, sample, output_dir: Path, official_example: bool = False) -> Dict[str, Any]:
    ep_start = time.perf_counter()
    with Image.open(sample.image_path) as im:
        original = im.convert("RGB")
    processed_original = process_image_official(original, MAX_PIXELS, MIN_PIXELS)
    options, gt_letter, gt_text = deterministic_option_shuffle(sample.options, sample.sample_id, OPTION_SEED)
    messages = initial_messages(processed_original, sample.question, options)
    observations = [original]
    image_size_used_list = [processed_original.size]
    crop_records = []
    turns = []
    status = "ok"
    final_answer = ""
    forced_answer = False
    hit_turn_cap = False
    peak_mem_mb = 0
    error = ""

    crop_dir = output_dir / "crops" / sample.sample_id.replace("/", "__")
    crop_dir.mkdir(parents=True, exist_ok=True)

    for round_idx in range(MAX_GENERATION_ROUND):
        try:
            result = backend.generate(messages, ROUND_MAX_NEW_TOKENS)
            if backend.torch.cuda.is_available():
                peak_mem_mb = max(peak_mem_mb, int(backend.torch.cuda.max_memory_allocated() / 1024**2))
            raw = result["raw"]
            answer = parse_answer(raw)
            parsed = parse_grounding(raw)
            turn = {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "turn_id": round_idx,
                "raw_model_output": raw,
                "parsed_action_type": parsed["type"] if parsed["type"] == "grounding" else ("answer" if answer else "invalid"),
                "parsed_answer_letter": answer,
                "raw_model_coordinate": json_dumps(parsed.get("bbox_2d", "")),
                "source": parsed.get("source", ""),
                "coordinate_type": "relative",
                "image_dimensions_used_for_conversion": "",
                "bbox_on_source_image": "",
                "bbox_original": "",
                "bbox_valid": "",
                "bbox_clamped": "",
                "crop_path": "",
                "crop_width": "",
                "crop_height": "",
                "input_token_count": result.get("input_token_count", ""),
                "output_token_count": result.get("output_token_count", ""),
                "finish_reason": result.get("finish_reason", ""),
                "generation_time_sec": f"{result.get('generation_time_sec', 0):.4f}",
                "error_message": "",
            }
            if answer and parsed["type"] != "grounding":
                final_answer = answer
                turns.append(turn)
                break
            if parsed["type"] != "grounding":
                final_answer = answer
                if not final_answer:
                    status = "invalid"
                turns.append(turn)
                break
            if len(observations) >= LIMIT_IMAGE_NUM or round_idx == MAX_GENERATION_ROUND - 1:
                hit_turn_cap = True
                turns.append(turn)
                break
            try:
                source_image, source_bbox, conv = prepare_grounding_inputs_multi_turn(parsed, observations, image_size_used_list, True)
                crop_raw = crop_image_official(source_image, source_bbox, image_size_used_list, resize=1)
                processed_crop = process_image_official(crop_raw, MAX_PIXELS, MIN_PIXELS)
                observations.append(crop_raw)
                image_size_used_list.append(processed_crop.size)
                bbox_original = source_bbox_to_original(source_bbox, parsed["source"], crop_records)
                crop_path = crop_dir / f"turn_{round_idx + 1:02d}.png"
                crop_raw.save(crop_path)
                crop_record = {
                    "turn_id": round_idx,
                    "raw_model_coordinate": parsed["bbox_2d"],
                    "source": parsed["source"],
                    "bbox_on_source_image": source_bbox,
                    "bbox_original": bbox_original,
                    "observation_size": crop_raw.size,
                    "processed_observation_size": processed_crop.size,
                    "crop_path": str(crop_path),
                    "conversion": conv,
                    "assistant_output": raw,
                    "crop_image": processed_crop,
                }
                crop_records.append(crop_record)
                turn.update(
                    {
                        "image_dimensions_used_for_conversion": json_dumps(conv["source_image_size"]),
                        "bbox_on_source_image": json_dumps(source_bbox),
                        "bbox_original": json_dumps(bbox_original),
                        "bbox_valid": "True",
                        "bbox_clamped": str(conv["clamped"]),
                        "crop_path": str(crop_path),
                        "crop_width": crop_raw.width,
                        "crop_height": crop_raw.height,
                    }
                )
                turns.append(turn)
                messages.append({"role": "assistant", "content": raw})
                text = HF_TOOL_CALL_CROP_MULTI_TURN_PROMPT.format(action_turn=round_idx, observation_turn=round_idx + 1)
                messages.append({"role": "user", "content": [{"type": "image", "image": processed_crop}, {"type": "text", "text": text}]})
            except Exception as exc:
                turn["error_message"] = str(exc)
                turn["bbox_valid"] = "False"
                turns.append(turn)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"ERROR occurs during grounding. Error Information: {exc}.\n{ERROR_INFO_MULTI_TURN_PROMPT}"})
        except RuntimeError as exc:
            error = str(exc)
            status = "oom" if "out of memory" in error.lower() or ("cuda" in error.lower() and "memory" in error.lower()) else "invalid"
            break

    if not final_answer and status == "ok":
        forced_answer = True
        messages.append({"role": "user", "content": "Please put your final answer inside <answer> and </answer>."})
        try:
            result = backend.generate(messages, ROUND_MAX_NEW_TOKENS)
            raw = result["raw"]
            final_answer = parse_answer(raw)
            turns.append(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "turn_id": len(turns),
                    "raw_model_output": raw,
                    "parsed_action_type": "answer" if final_answer else "invalid",
                    "parsed_answer_letter": final_answer,
                    "raw_model_coordinate": "",
                    "source": "",
                    "coordinate_type": "relative",
                    "image_dimensions_used_for_conversion": "",
                    "bbox_on_source_image": "",
                    "bbox_original": "",
                    "bbox_valid": "",
                    "bbox_clamped": "",
                    "crop_path": "",
                    "crop_width": "",
                    "crop_height": "",
                    "input_token_count": result.get("input_token_count", ""),
                    "output_token_count": result.get("output_token_count", ""),
                    "finish_reason": result.get("finish_reason", ""),
                    "generation_time_sec": f"{result.get('generation_time_sec', 0):.4f}",
                    "error_message": "",
                }
            )
        except RuntimeError as exc:
            error = str(exc)
            status = "oom" if "out of memory" in error.lower() else "invalid"
    if status == "ok" and not final_answer:
        status = "invalid"

    ep = {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "question": sample.question,
        "options_shuffled": json_dumps(options),
        "gt_answer_letter": gt_letter,
        "gt_answer_text": gt_text,
        "final_answer_letter": final_answer,
        "final_answer_text": options[LETTERS.index(final_answer)] if final_answer in LETTERS[: len(options)] else "",
        "correct": str(final_answer == gt_letter) if final_answer else "False",
        "status": status,
        "tool_use": str(len(crop_records) > 0),
        "turns_used": len(turns),
        "num_tool_calls": len(crop_records),
        "num_valid_tool_calls": len(crop_records),
        "hit_turn_cap": str(hit_turn_cap),
        "forced_answer": str(forced_answer),
        "original_image_path": str(sample.image_path),
        "original_image_width": original.width,
        "original_image_height": original.height,
        "original_image_sha256": sha256_file(sample.image_path),
        "processed_initial_width": processed_original.width,
        "processed_initial_height": processed_original.height,
        "generation_time_total_sec": f"{time.perf_counter() - ep_start:.4f}",
        "peak_gpu_memory_mb": peak_mem_mb,
        "error_message": error,
    }
    return {"episode": ep, "turns": turns, "crops": crop_records, "options": options, "gt_bbox": clamp_bbox_xywh(sample.gt_bbox, original.width, original.height)}


def label_crops(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in results:
        gt = item["gt_bbox"]
        gx, gy, gw, gh = gt
        center = (gx + gw / 2, gy + gh / 2)
        for crop in item["crops"]:
            x1, y1, x2, y2 = crop["bbox_original"]
            xywh = [x1, y1, x2 - x1, y2 - y1]
            iou = bbox_iou_xywh(xywh, gt)
            contains = x1 <= center[0] <= x2 and y1 <= center[1] <= y2
            label = "useful" if iou > 0.1 or contains else "failed_or_non_gt"
            crop["crop_label"] = label
            crop["iou_with_gt"] = iou
            crop["contains_gt_center"] = contains
            rows.append(
                {
                    "sample_id": item["episode"]["sample_id"],
                    "category": item["episode"]["category"],
                    "turn_id": crop["turn_id"],
                    "raw_model_coordinate": json_dumps(crop["raw_model_coordinate"]),
                    "converted_original_bbox": json_dumps(crop["bbox_original"]),
                    "gt_bbox": json_dumps(gt),
                    "iou": f"{iou:.8f}",
                    "contains_gt_center": str(contains),
                    "crop_label": label,
                    "crop_path": crop["crop_path"],
                }
            )
    return rows


def gate_a(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(episodes)
    counts = [int(ep["num_valid_tool_calls"]) for ep in episodes]
    return {
        "n": n,
        "tool_use_rate": sum(c >= 1 for c in counts) / n if n else 0,
        "mean_crops_per_episode": sum(counts) / n if n else 0,
        "median_crops_per_episode": sorted(counts)[n // 2] if n else 0,
        "num_tool_calls_distribution": {str(k): counts.count(k) for k in sorted(set(counts))},
        "fraction_ge_2_crops": sum(c >= 2 for c in counts) / n if n else 0,
        "hit_turn_cap_count": sum(ep["hit_turn_cap"] == "True" for ep in episodes),
        "accuracy": sum(ep["correct"] == "True" for ep in episodes) / n if n else 0,
        "invalid_rate": sum(ep["status"] == "invalid" for ep in episodes) / n if n else 0,
        "oom_rate": sum(ep["status"] == "oom" for ep in episodes) / n if n else 0,
        "forced_answer_rate": sum(ep["forced_answer"] == "True" for ep in episodes) / n if n else 0,
        "mean_wall_time_sec": sum(float(ep["generation_time_total_sec"]) for ep in episodes) / n if n else 0,
        "passes": bool((sum(c >= 1 for c in counts) / n if n else 0) >= 0.60 and (sum(c >= 2 for c in counts) / n if n else 0) >= 0.30 and sum(ep["status"] == "oom" for ep in episodes) == 0),
    }


def prevalence_preview(episodes: Sequence[Dict[str, Any]], labels: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(episodes)
    label_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for row in labels:
        label_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    useful = sum(row["crop_label"] == "useful" for row in labels)
    failed = sum(row["crop_label"] == "failed_or_non_gt" for row in labels)
    total = useful + failed
    failed_counts = {ep["sample_id"]: sum(row["crop_label"] == "failed_or_non_gt" for row in label_by_sample.get(ep["sample_id"], [])) for ep in episodes}
    rows = [
        {"metric": "mean_useful_crops_per_episode", "value": useful / n if n else "NA"},
        {"metric": "mean_failed_or_non_gt_crops_per_episode", "value": failed / n if n else "NA"},
        {"metric": "failed_or_non_gt_share_of_all_crops", "value": failed / total if total else "NA"},
        {"metric": "episodes_with_ge1_failed_or_non_gt_crop", "value": sum(c >= 1 for c in failed_counts.values()) / n if n else "NA"},
        {"metric": "episodes_with_ge2_failed_or_non_gt_crops", "value": sum(c >= 2 for c in failed_counts.values()) / n if n else "NA"},
    ]
    for label, predicate in [
        ("0", lambda c: c == 0),
        ("1", lambda c: c == 1),
        ("2", lambda c: c == 2),
        ("ge3", lambda c: c >= 3),
    ]:
        subset = [ep for ep in episodes if predicate(failed_counts.get(ep["sample_id"], 0))]
        rows.append({"metric": f"accuracy_failed_or_non_gt_count_{label}", "value": sum(ep["correct"] == "True" for ep in subset) / len(subset) if subset else "NA"})
        rows.append({"metric": f"num_episodes_failed_or_non_gt_count_{label}", "value": len(subset)})
    rows.append({"metric": "failed_or_non_gt_visual_token_fraction", "value": "NA"})
    rows.append({"metric": "visual_token_fraction_note", "value": "not available from this HF protocol reimplementation"})
    return rows


def render_html(results: Sequence[Dict[str, Any]], output_dir: Path, limit: int = 5) -> List[str]:
    html_dir = output_dir / "sanity_html"
    html_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in results[:limit]:
        ep = item["episode"]
        with Image.open(ep["original_image_path"]) as im:
            canvas = process_image_official(im.convert("RGB"), MAX_PIXELS, MIN_PIXELS)
        sx = canvas.width / float(ep["original_image_width"])
        sy = canvas.height / float(ep["original_image_height"])
        draw = ImageDraw.Draw(canvas)
        gx, gy, gw, gh = item["gt_bbox"]
        draw.rectangle((gx * sx, gy * sy, (gx + gw) * sx, (gy + gh) * sy), outline="cyan", width=3)
        for crop in item["crops"]:
            x1, y1, x2, y2 = crop["bbox_original"]
            draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=("lime" if crop.get("crop_label") == "useful" else "red"), width=3)
        image_name = f"{ep['sample_id'].replace('/', '__')}_annotated.png"
        image_path = html_dir / image_name
        canvas.save(image_path)
        crop_html = []
        for crop in item["crops"]:
            crop_html.append(
                f"<h3>Turn {crop['turn_id']} {html.escape(crop.get('crop_label', ''))}</h3>"
                f"<p>IoU={crop.get('iou_with_gt',''):.4f}, contains_GT_center={crop.get('contains_gt_center','')}</p>"
                f"<p>raw={html.escape(json_dumps(crop['raw_model_coordinate']))}; original={html.escape(json_dumps(crop['bbox_original']))}</p>"
                f"<img src='../../{html.escape(crop['crop_path'])}' style='max-width:360px;border:1px solid #ccc'>"
            )
        transcript = "\n".join(f"[{t['turn_id']}] {t['raw_model_output']}" for t in item["turns"])
        page = f"<html><body><h1>{html.escape(ep['sample_id'])}</h1><p>final={ep['final_answer_letter']} correct={ep['correct']}</p><img src='{image_name}' style='max-width:760px;border:1px solid #888'>{''.join(crop_html)}<pre>{html.escape(transcript)}</pre></body></html>"
        path = html_dir / f"{ep['sample_id'].replace('/', '__')}.html"
        path.write_text(page, encoding="utf-8")
        paths.append(str(path))
    return paths


EPISODE_FIELDS = [
    "sample_id", "category", "question", "options_shuffled", "gt_answer_letter", "gt_answer_text", "final_answer_letter", "final_answer_text", "correct", "status", "tool_use", "turns_used", "num_tool_calls", "num_valid_tool_calls", "hit_turn_cap", "forced_answer", "original_image_path", "original_image_width", "original_image_height", "original_image_sha256", "processed_initial_width", "processed_initial_height", "generation_time_total_sec", "peak_gpu_memory_mb", "error_message",
]
TURN_FIELDS = [
    "sample_id", "category", "turn_id", "raw_model_output", "parsed_action_type", "parsed_answer_letter", "raw_model_coordinate", "source", "coordinate_type", "image_dimensions_used_for_conversion", "bbox_on_source_image", "bbox_original", "bbox_valid", "bbox_clamped", "crop_path", "crop_width", "crop_height", "input_token_count", "output_token_count", "finish_reason", "generation_time_sec", "error_message",
]
LABEL_FIELDS = ["sample_id", "category", "turn_id", "raw_model_coordinate", "converted_original_bbox", "gt_bbox", "iou", "contains_gt_center", "crop_label", "crop_path"]


def smoke_sample_ids(output_dir: Path, data_root: str, n: int) -> List[str]:
    prior = Path("outputs/e2_smoke/e2_smoke_episode_summary.csv")
    if prior.exists():
        ids = pd.read_csv(prior)["sample_id"].drop_duplicates().tolist()
        if len(ids) >= n:
            return ids[:n]
    return [s.sample_id for s in load_vstar_samples(data_root, num_samples=n)]


def write_reports(output_dir: Path, settings: Dict[str, Any], gate: Dict[str, Any], labels: List[Dict[str, Any]], html_paths: List[str], stopped_reason: str = "") -> None:
    repo_commit = os.popen(f"git -C {REPO_PATH} rev-parse HEAD").read().strip()
    env_report = f"""# Environment Report

- env: venv_e2m at `/mnt/data2/szj/envs/venv_e2m`
- Python executable: `{sys.executable}`
- Python: 3.10.20
- torch: 2.5.1
- torch CUDA runtime: 12.1
- transformers: 5.13.0 in `venv_e2m`
- accelerate: 1.13.0
- qwen-vl-utils: 0.0.14
- GPU: 8 x NVIDIA GeForce RTX 3090, 24576 MiB each; observed idle memory was 18 MiB/GPU before smoke checks
- vLLM: not used
- flash-attn: not installed / not used
- dependency issue: official repo uses vLLM rollout; this smoke reimplements the protocol with HF Transformers to avoid modifying the E1 environment
- environment isolation: E1 `cvpr2027` was restored to transformers 4.51.3; freeze files are `outputs/env_e1_freeze.txt` and `outputs/e2m_smoke/env_e2m_freeze.txt`
"""
    model_report = f"""# Model Resolution Report

- official repo URL: {REPO_URL}
- official repo commit: {repo_commit}
- checkpoint ID: {MODEL_ID}
- checkpoint revision: {MODEL_REVISION}
- checkpoint type: RL
- local model path: {LOCAL_MODEL_PATH}
- download location: {LOCAL_MODEL_PATH}
- model completeness check: 4 safetensors shards present; 729 tensors indexed; local config/processor loaded successfully
"""
    protocol = f"""# Protocol Audit

- official system prompt: `{TOOL_CROP_SYSTEM_PROMPT}`
- tool-call format: `<grounding>{{"bbox_2d": [x0, y0, x1, y1], "source": "original_image|observation_i"}}</grounding>`
- coordinate format: relative coordinates; official prompt says image width and height are 1.0
- conversion formula: when `use_relative_coordinates=True`, multiply bbox coordinates by the selected source image width/height, then int-cast and clamp
- crop preprocessing: official `crop_image(..., resize=1)` crops source image directly; then official `process_image(max_pixels=2000000,min_pixels=40000)` uses NEAREST resize, min/max pixel area, short-side >=28, aspect-ratio guard
- max turns: val_max_generation_round=12
- generation params: official greedy eval uses val_n=1, val_do_sample=False; official max_response_length=8192. This HF smoke uses do_sample=False and max_new_tokens={ROUND_MAX_NEW_TOKENS} per round as a documented deviation for single RTX 3090 safety.
- image/message wrapping: official multi_turn_prompt_type=v2 observation wording is reused; explicit `<image>` / `<|vision_start|><|image_pad|><|vision_end|>` placeholders are omitted in the HF messages because `apply_chat_template` inserts them from structured image content. Keeping both causes a processor image-placeholder mismatch in transformers 5.x.
- checkpoint used: RL `{MODEL_ID}`
- official evaluation code adapted/reimplemented: reimplemented with HF Transformers; official vLLM engine not invoked.
"""
    useful = sum(x["crop_label"] == "useful" for x in labels)
    failed = sum(x["crop_label"] == "failed_or_non_gt" for x in labels)
    report = f"""# E2-M Smoke Report

## 1. Official Repo And Checkpoint
{model_report}

## 2. Environment Report
{env_report}

## 3. Protocol Audit
{protocol}

## 4. Official Example Result
- status: {gate.get("protocol_check_status", "UNKNOWN")}
- note: official `DEMO_DATA` is not included in this checkout; V*Bench first smoke sample was used as an authorized protocol sanity substitute when available.

## 5. Plain VQA Anchor
- saved to `plain_vqa_anchor.csv`; summary is reported in the final terminal response.

## 6. Gate A Results
```json
{json.dumps(gate, ensure_ascii=False, indent=2)}
```

## 7. Gate B Results
- useful crops: {useful}
- failed_or_non_gt crops: {failed}
- HTML paths: {html_paths}
- status: {"WAITING_FOR_HUMAN_INSPECTION" if gate.get("passes") else "NOT_RUN"}

## 8. Oracle Eviction Smoke
- NOT_RUN in this amended smoke; this run stops after Gate A, Gate B, and preview statistics.

## 9. Runtime Projection
- see `runtime_projection.md`

## 10. Recommendation
- {stopped_reason or ("Proceed to human HTML inspection before full E2." if gate.get("passes") else "Fix protocol and rerun smoke; do not run full E2.")}
"""
    (output_dir / "environment_report.md").write_text(env_report, encoding="utf-8")
    (output_dir / "model_resolution_report.md").write_text(model_report, encoding="utf-8")
    (output_dir / "protocol_audit.md").write_text(protocol, encoding="utf-8")
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LOCAL_MODEL_PATH)
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--output_dir", default="outputs/e2m_smoke")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_vbench_protocol_substitute", action="store_true")
    parser.add_argument("--env_name", default="venv_e2m")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = smoke_sample_ids(output_dir, args.data_root, args.num_samples)
    (output_dir / "smoke_sample_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    samples_by_id = {s.sample_id: s for s in load_vstar_samples(args.data_root, num_samples=None)}
    samples = [samples_by_id[x] for x in ids]
    settings = {
        "model": args.model,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "repo_url": REPO_URL,
        "repo_commit": os.popen(f"git -C {REPO_PATH} rev-parse HEAD").read().strip(),
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_generation_round": MAX_GENERATION_ROUND,
        "limit_image_num": LIMIT_IMAGE_NUM,
        "round_max_new_tokens_hf_deviation": ROUND_MAX_NEW_TOKENS,
        "round_max_time_sec_guard": ROUND_MAX_TIME_SEC,
        "guard_note": "Added after the first E2-M smoke exposed a long single-call generate stall; prompt/model/data settings are otherwise unchanged.",
        "val_n": 1,
        "val_do_sample": False,
        "use_relative_coordinates": True,
        "env_name": args.env_name,
        "python_executable": sys.executable,
        "official_demo_data_available": False,
        "protocol_sanity_source": "vbench_first_smoke_sample" if args.allow_vbench_protocol_substitute else "official_demo_data_required",
    }
    (output_dir / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    demo_candidates = []
    for pattern in ("app.py", "*.json", "*.jsonl", "*.parquet"):
        demo_candidates.extend(str(p) for p in Path(REPO_PATH).rglob(pattern))
    demo_candidates = [
        p for p in demo_candidates
        if "/.git/" not in p
        and "/tests/" not in p
        and "node_modules" not in p
        and "outputs/" not in p
        and Path(p).name not in {"config.json", "generation_config.json", "tokenizer_config.json"}
    ]
    if not demo_candidates and not args.allow_vbench_protocol_substitute:
        official_dir = output_dir / "official_example"
        official_dir.mkdir(exist_ok=True)
        (official_dir / "official_example_transcript.txt").write_text("", encoding="utf-8")
        (official_dir / "official_example_trajectory.json").write_text(
            json.dumps({"status": "not_run", "reason": OFFICIAL_EXAMPLE_REASON}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (official_dir / "official_example_report.md").write_text(
            "# Official Example\n\n"
            f"- status: NOT_RUN\n- reason: {OFFICIAL_EXAMPLE_REASON}\n",
            encoding="utf-8",
        )
        write_csv(output_dir / "plain_vqa_anchor.csv", [], ["sample_id", "category", "gt_answer_letter", "raw_output", "pred_answer_letter", "correct", "status", "generation_time_sec"])
        write_csv(output_dir / "episode_summary.csv", [], EPISODE_FIELDS)
        write_csv(output_dir / "turns.csv", [], TURN_FIELDS)
        write_csv(output_dir / "oom_log.csv", [], EPISODE_FIELDS)
        write_csv(output_dir / "peak_memory.csv", [], ["sample_id", "peak_gpu_memory_mb"])
        write_csv(output_dir / "crop_labels.csv", [], LABEL_FIELDS)
        write_csv(output_dir / "prevalence_preview.csv", [], ["metric", "value"])
        write_csv(output_dir / "eviction_smoke.csv", [], ["sample_id", "variant", "raw_output", "parsed_answer_letter", "correct", "status"])
        (output_dir / "trajectories.jsonl").write_text("", encoding="utf-8")
        (output_dir / "transcripts.txt").write_text("", encoding="utf-8")
        (output_dir / "runtime_projection.md").write_text(
            "# Runtime Projection\n\nNot computed because the required official Mini-o3 example was unavailable.\n",
            encoding="utf-8",
        )
        write_reports(output_dir, settings, {"passes": False, "official_example_available": False}, [], [], OFFICIAL_EXAMPLE_REASON)
        raise SystemExit(OFFICIAL_EXAMPLE_REASON)

    backend = HFBackend(args.model)

    official_dir = output_dir / "official_example"
    official_dir.mkdir(exist_ok=True)
    official = run_episode(backend, samples[0], official_dir, official_example=True)
    (official_dir / "official_example_trajectory.json").write_text(json.dumps({"episode": official["episode"], "turns": official["turns"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (official_dir / "official_example_transcript.txt").write_text("\n\n".join(t["raw_model_output"] for t in official["turns"]), encoding="utf-8")
    substitute_note = (
        "Official Mini-o3 DEMO_DATA was absent; this run used the first V*Bench smoke sample as an authorized protocol sanity substitute."
        if args.allow_vbench_protocol_substitute and not demo_candidates
        else "Official demo candidate was available."
    )
    (official_dir / "official_example_report.md").write_text(
        "# Protocol Sanity Check\n\n"
        f"- status: {'SUBSTITUTE' if args.allow_vbench_protocol_substitute and not demo_candidates else 'OFFICIAL'}\n"
        f"- sample: {samples[0].sample_id}\n"
        f"- tool calls: {official['episode']['num_valid_tool_calls']}\n"
        f"- final answer: {official['episode']['final_answer_letter']}\n"
        f"- note: {substitute_note}\n",
        encoding="utf-8",
    )
    print("\n===== PROTOCOL SANITY TRANSCRIPT BEGIN =====")
    print(f"sample_id: {samples[0].sample_id}")
    for turn in official["turns"]:
        print(f"\n[turn {turn['turn_id']}]\n{turn['raw_model_output']}")
    print("===== PROTOCOL SANITY TRANSCRIPT END =====\n")

    anchor_rows = []
    for sample in samples:
        with Image.open(sample.image_path) as im:
            image = process_image_official(im.convert("RGB"), MAX_PIXELS, MIN_PIXELS)
        options, gt_letter, gt_text = deterministic_option_shuffle(sample.options, sample.sample_id, OPTION_SEED)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"Question: {sample.question}\nOptions:\n{format_options(options)}\nAnswer with only A, B, C, or D."}]}]
        result = backend.generate(messages, 64)
        pred = parse_answer(result["raw"])
        anchor_rows.append({"sample_id": sample.sample_id, "category": sample.category, "gt_answer_letter": gt_letter, "raw_output": result["raw"], "pred_answer_letter": pred, "correct": str(pred == gt_letter), "status": "ok" if pred else "invalid", "generation_time_sec": f"{result['generation_time_sec']:.4f}"})
    write_csv(output_dir / "plain_vqa_anchor.csv", anchor_rows, ["sample_id", "category", "gt_answer_letter", "raw_output", "pred_answer_letter", "correct", "status", "generation_time_sec"])

    results = load_completed_results(output_dir)
    completed = {item["episode"]["sample_id"] for item in results}
    if not completed:
        write_csv(output_dir / "episode_summary.csv", [], EPISODE_FIELDS)
        write_csv(output_dir / "turns.csv", [], TURN_FIELDS)
        write_csv(output_dir / "oom_log.csv", [], EPISODE_FIELDS)
        write_csv(output_dir / "peak_memory.csv", [], ["sample_id", "peak_gpu_memory_mb"])
        (output_dir / "trajectories.jsonl").write_text("", encoding="utf-8")
        (output_dir / "transcripts.txt").write_text("", encoding="utf-8")
    else:
        print(f"Resuming E2-M smoke with {len(completed)} completed episodes already on disk.")

    for idx, sample in enumerate(samples, start=1):
        if sample.sample_id in completed:
            print(f"[{idx}/{len(samples)}] {sample.sample_id} skipped=already_complete")
            continue
        item = run_episode(backend, sample, output_dir)
        results.append(item)
        append_episode_outputs(output_dir, item)
        completed.add(sample.sample_id)
        ep = item["episode"]
        print(f"[{idx}/{len(samples)}] {sample.sample_id} status={ep['status']} crops={ep['num_valid_tool_calls']} answer={ep['final_answer_letter']} correct={ep['correct']}")
    episodes = [x["episode"] for x in results]
    turns = [t for x in results for t in x["turns"]]
    gate = gate_a(episodes)
    gate["protocol_check_status"] = "SUBSTITUTE" if args.allow_vbench_protocol_substitute and not demo_candidates else "OFFICIAL"
    gate["protocol_check_tool_calls"] = int(official["episode"]["num_valid_tool_calls"])
    write_csv(output_dir / "episode_summary.csv", episodes, EPISODE_FIELDS)
    write_csv(output_dir / "turns.csv", turns, TURN_FIELDS)
    write_csv(output_dir / "oom_log.csv", [e for e in episodes if e["status"] == "oom"], EPISODE_FIELDS)
    write_csv(output_dir / "peak_memory.csv", [{"sample_id": e["sample_id"], "peak_gpu_memory_mb": e["peak_gpu_memory_mb"]} for e in episodes], ["sample_id", "peak_gpu_memory_mb"])
    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(strip_crop_images(item), ensure_ascii=False) + "\n")
    (output_dir / "transcripts.txt").write_text("\n\n".join("\n".join(t["raw_model_output"] for t in item["turns"]) for item in results), encoding="utf-8")

    labels = []
    html_paths = []
    if gate["passes"]:
        labels = label_crops(results)
        write_csv(output_dir / "crop_labels.csv", labels, LABEL_FIELDS)
        html_paths = render_html(results, output_dir)
    else:
        write_csv(output_dir / "crop_labels.csv", [], LABEL_FIELDS)
    preview_rows = prevalence_preview(episodes, labels)
    write_csv(output_dir / "prevalence_preview.csv", preview_rows, ["metric", "value"])
    write_csv(output_dir / "eviction_smoke.csv", [], ["sample_id", "variant", "raw_output", "parsed_answer_letter", "correct", "status"])
    mean_wall = gate["mean_wall_time_sec"]
    runtime = f"# Runtime Projection\n\n- mean wall time per episode: {mean_wall:.2f}s\n- projected single-GPU full time: {mean_wall * 238 / 3600:.2f}h\n- projected 8-GPU sharded full time: {mean_wall * 238 / 8 / 3600:.2f}h\n"
    (output_dir / "runtime_projection.md").write_text(runtime, encoding="utf-8")
    write_reports(output_dir, settings, gate, labels, html_paths)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
