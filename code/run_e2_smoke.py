import argparse
import copy
import csv
import hashlib
import html
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from context_pollution_utils import (
    LETTERS,
    bbox_iou_xywh,
    clamp_bbox_xywh,
    deterministic_option_shuffle,
    format_options,
    json_dumps,
    load_vstar_samples,
)
from run_context_pollution_smoke_v21 import Backend, resize_crop_long_side, sha256_file


MODEL_PATH = "/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct"
DATA_ROOT = "/home/songzhoujie/cvpr27/vstar/data/vstar_bench"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
DISPLAY_LONG_SIDE = 1024
CROP_LONG_SIDE = 448
INTERMEDIATE_MAX_NEW_TOKENS = 192
FINAL_MAX_NEW_TOKENS = 64
MAX_CROP_TURNS = 8
MAX_INVALID_RETRIES = 2
OPTION_SEED = 42


SYSTEM_PROMPT_TEMPLATE = """You are a visual assistant answering multiple-choice questions about an image.

The image you see is downscaled, so small objects and text may be unclear.

You can request a zoomed-in crop of any region using exactly this format:

<tool_call>{{"name":"crop","arguments":{{"bbox":[x1,y1,x2,y2]}}}}</tool_call>

The provided image has size {width} x {height} pixels.
All crop coordinates must be in this {width} x {height} coordinate system.

Examine regions relevant to the question before answering.
When you are confident, answer using exactly this format:

<answer>X</answer>

where X is one of A, B, C, or D.

Use at most 8 crops.

Do not output anything outside the tool_call or answer tags."""

FEW_SHOT_EXEMPLAR = """

Example:
User shows a downscaled image and asks which number is printed on a small badge.
Assistant:
<tool_call>{"name":"crop","arguments":{"bbox":[120,80,260,210]}}</tool_call>
User:
Tool result:
<zoomed crop image>
Assistant:
<answer>C</answer>
"""

INVALID_ACTION_PROMPT = "Invalid action. Either request a crop with the exact tool_call format or answer with <answer>X</answer>."
FORCED_ANSWER_PROMPT = "You must answer now. Respond with <answer>X</answer> only."


EPISODE_FIELDS = [
    "sample_id",
    "category",
    "question",
    "options_shuffled",
    "gt_answer_letter",
    "gt_answer_text",
    "final_answer_letter",
    "final_answer_text",
    "correct",
    "status",
    "tool_use",
    "turns_used",
    "num_tool_calls",
    "num_valid_tool_calls",
    "num_invalid_actions",
    "retry_count",
    "forced_answer",
    "original_image_path",
    "display_image_path",
    "display_image_width",
    "display_image_height",
    "display_image_sha256",
    "original_image_width",
    "original_image_height",
    "original_image_sha256",
    "scale_x",
    "scale_y",
    "total_input_tokens_at_final_answer",
    "finish_reason_final",
    "output_token_count_final",
    "generation_time_total_sec",
]

TURN_FIELDS = [
    "sample_id",
    "category",
    "turn_id",
    "raw_model_output",
    "parsed_action_type",
    "parsed_answer_letter",
    "bbox_display",
    "bbox_original",
    "bbox_valid",
    "crop_path",
    "crop_width",
    "crop_height",
    "input_token_count",
    "output_token_count",
    "finish_reason",
    "generation_time_sec",
    "error_message",
]

LABEL_FIELDS = [
    "sample_id",
    "category",
    "turn_id",
    "crop_label",
    "iou_with_gt",
    "contains_gt_center",
    "gt_bbox",
    "bbox_original",
    "crop_path",
]

EVICTION_FIELDS = [
    "sample_id",
    "category",
    "variant",
    "live_answer_letter",
    "live_correct",
    "raw_output",
    "parsed_answer_letter",
    "correct",
    "status",
    "input_token_count",
    "output_token_count",
    "finish_reason",
    "generation_time_sec",
]


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256_image_pixels(image: Image.Image) -> str:
    image = image.convert("RGB")
    return hashlib.sha256(image.tobytes()).hexdigest()


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    image = image.convert("RGB")
    current = max(image.size)
    if current <= long_side:
        return image.copy()
    scale = long_side / current
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.BICUBIC)


def first_smoke_sample_ids(output_dir: Path, data_root: str, num_samples: int) -> List[str]:
    smoke = output_dir.parent / "results_context_pollution_smoke_v21.csv"
    if smoke.exists():
        ids: List[str] = []
        with smoke.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sid = row.get("sample_id")
                if sid and sid not in ids:
                    ids.append(sid)
                if len(ids) >= num_samples:
                    return ids
    return [sample.sample_id for sample in load_vstar_samples(data_root, num_samples=num_samples)]


def parse_action(raw: str) -> Dict[str, object]:
    text = raw or ""
    answer_match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, flags=re.IGNORECASE)
    tool_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    candidates = []
    if answer_match:
        candidates.append((answer_match.start(), "answer", answer_match))
    if tool_match:
        candidates.append((tool_match.start(), "tool_call", tool_match))
    if not candidates:
        return {"type": "invalid"}
    _, kind, match = sorted(candidates, key=lambda item: item[0])[0]
    if kind == "answer":
        return {"type": "answer", "answer_letter": match.group(1).upper()}
    try:
        payload = json.loads(match.group(1))
        bbox = payload.get("arguments", {}).get("bbox")
        if payload.get("name") != "crop" or not isinstance(bbox, list) or len(bbox) != 4:
            return {"type": "invalid"}
        bbox = [float(x) for x in bbox]
        return {"type": "tool_call", "bbox_display": bbox}
    except Exception:
        return {"type": "invalid"}


def answer_text(letter: str, options: Sequence[str]) -> str:
    if letter in LETTERS[: len(options)]:
        return options[LETTERS.index(letter)]
    return ""


def display_to_original_bbox(
    bbox_display: Sequence[float],
    scale_x: float,
    scale_y: float,
    original_width: int,
    original_height: int,
    min_side: float = 8.0,
) -> Tuple[List[float], bool]:
    x1, y1, x2, y2 = [float(v) for v in bbox_display]
    if not all(math.isfinite(v) for v in [x1, y1, x2, y2]):
        return [0.0, 0.0, 1.0, 1.0], False
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    ox1, oy1, ox2, oy2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
    cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    w, h = max(min_side, ox2 - ox1), max(min_side, oy2 - oy1)
    ox1, ox2 = cx - w / 2, cx + w / 2
    oy1, oy2 = cy - h / 2, cy + h / 2
    ox1 = max(0.0, min(float(original_width - 1), ox1))
    oy1 = max(0.0, min(float(original_height - 1), oy1))
    ox2 = max(ox1 + 1.0, min(float(original_width), ox2))
    oy2 = max(oy1 + 1.0, min(float(original_height), oy2))
    return [ox1, oy1, ox2, oy2], True


def crop_original_xyxy(image: Image.Image, bbox_original: Sequence[float]) -> Image.Image:
    x1, y1, x2, y2 = bbox_original
    crop = image.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))).convert("RGB")
    return resize_crop_long_side(crop, CROP_LONG_SIDE)


def make_system_prompt(width: int, height: int, few_shot: bool) -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE.format(width=width, height=height)
    if few_shot:
        prompt += FEW_SHOT_EXEMPLAR
    return prompt


def make_initial_user(display_image: Image.Image, question: str, options: Sequence[str]) -> Dict:
    text = f"Question: {question}\nOptions:\n{format_options(options)}"
    return {"role": "user", "content": [{"type": "image", "image": display_image}, {"type": "text", "text": text}]}


def generate(backend: Backend, messages: Sequence[Dict], max_new_tokens: int) -> Dict[str, object]:
    result = backend.generate(messages, extract_images(messages), max_new_tokens=max_new_tokens)
    return result


def extract_images(messages: Sequence[Dict]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    images.append(item["image"])
    return images


def serializable_messages(messages: Sequence[Dict]) -> List[Dict]:
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            serial_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    serial_content.append({"type": "image", "image": "<PIL.Image>"})
                else:
                    serial_content.append(item)
            out.append({"role": message.get("role"), "content": serial_content})
        else:
            out.append(dict(message))
    return out


def run_episode(
    backend: Backend,
    sample,
    output_dir: Path,
    few_shot: bool,
    run_tag: str,
) -> Dict:
    episode_start = time.perf_counter()
    display_dir = output_dir / "display_images"
    crop_dir = output_dir / "crops" / sample.sample_id.replace("/", "__")
    display_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(sample.image_path) as original_in:
        original = original_in.convert("RGB")
    display = resize_long_side(original, DISPLAY_LONG_SIDE)
    display_path = display_dir / f"{sample.sample_id.replace('/', '__')}.png"
    display.save(display_path)
    original_w, original_h = original.size
    display_w, display_h = display.size
    scale_x = original_w / display_w
    scale_y = original_h / display_h
    original_sha = sha256_file(sample.image_path)
    display_sha = sha256_image_pixels(display)

    options, gt_letter, gt_text = deterministic_option_shuffle(sample.options, sample.sample_id, OPTION_SEED)
    messages: List[Dict] = [
        {"role": "system", "content": make_system_prompt(display_w, display_h, few_shot)},
        make_initial_user(display, sample.question, options),
    ]

    turns = []
    crops = []
    invalid_actions = 0
    retries = 0
    forced_answer = False
    status = "ok"
    final_answer = ""
    result_final: Dict[str, object] = {}
    messages_before_final: List[Dict] = []
    raw_final = ""
    error_final = ""
    crop_turns = 0

    for turn_id in range(1, MAX_CROP_TURNS + MAX_INVALID_RETRIES + 2):
        max_tokens = FINAL_MAX_NEW_TOKENS if forced_answer else INTERMEDIATE_MAX_NEW_TOKENS
        try:
            messages_before_this = copy.deepcopy(messages)
            result = generate(backend, messages, max_tokens)
            raw = str(result.get("raw", ""))
            action = parse_action(raw)
            parsed_type = str(action["type"])
            turn_row = {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "turn_id": turn_id,
                "raw_model_output": raw,
                "parsed_action_type": parsed_type,
                "parsed_answer_letter": action.get("answer_letter", ""),
                "bbox_display": json_dumps(action.get("bbox_display", "")) if action.get("bbox_display") else "",
                "bbox_original": "",
                "bbox_valid": "",
                "crop_path": "",
                "crop_width": "",
                "crop_height": "",
                "input_token_count": result.get("input_token_count", ""),
                "output_token_count": result.get("output_token_count", ""),
                "finish_reason": result.get("finish_reason", ""),
                "generation_time_sec": result.get("generation_time_sec", ""),
                "error_message": "",
            }
            if parsed_type == "answer":
                final_answer = str(action["answer_letter"])
                messages_before_final = messages_before_this
                result_final = result
                raw_final = raw
                turns.append(turn_row)
                break
            if parsed_type == "tool_call" and crop_turns < MAX_CROP_TURNS:
                bbox_display = action["bbox_display"]
                bbox_original, bbox_valid = display_to_original_bbox(bbox_display, scale_x, scale_y, original_w, original_h)
                crop = crop_original_xyxy(original, bbox_original)
                crop_turns += 1
                crop_path = crop_dir / f"turn_{crop_turns:02d}.png"
                crop.save(crop_path)
                turn_row.update(
                    {
                        "bbox_original": json_dumps(bbox_original),
                        "bbox_valid": str(bbox_valid),
                        "crop_path": str(crop_path),
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                    }
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": [{"type": "text", "text": "Tool result:"}, {"type": "image", "image": crop}]})
                crops.append(
                    {
                        "turn_id": turn_id,
                        "assistant_output": raw,
                        "bbox_display": bbox_display,
                        "bbox_original": bbox_original,
                        "bbox_valid": bbox_valid,
                        "crop_path": str(crop_path),
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                        "crop_image": crop,
                    }
                )
                turns.append(turn_row)
                continue
            invalid_actions += 1
            retries += 1
            turns.append(turn_row)
            if retries <= MAX_INVALID_RETRIES:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": INVALID_ACTION_PROMPT})
                continue
            forced_answer = True
            messages.append({"role": "user", "content": FORCED_ANSWER_PROMPT})
        except RuntimeError as exc:
            error_final = str(exc)
            if "out of memory" in error_final.lower() or ("cuda" in error_final.lower() and "memory" in error_final.lower()):
                status = "oom"
            else:
                status = "invalid"
            if backend.torch.cuda.is_available():
                backend.torch.cuda.empty_cache()
            break
        if crop_turns >= MAX_CROP_TURNS and not forced_answer:
            forced_answer = True
            messages.append({"role": "user", "content": FORCED_ANSWER_PROMPT})

    if not final_answer and status == "ok":
        try:
            forced_answer = True
            messages.append({"role": "user", "content": FORCED_ANSWER_PROMPT})
            messages_before_final = copy.deepcopy(messages)
            result_final = generate(backend, messages, FINAL_MAX_NEW_TOKENS)
            raw_final = str(result_final.get("raw", ""))
            action = parse_action(raw_final)
            final_answer = str(action.get("answer_letter", "")) if action.get("type") == "answer" else ""
            turns.append(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "turn_id": len(turns) + 1,
                    "raw_model_output": raw_final,
                    "parsed_action_type": action.get("type", "invalid"),
                    "parsed_answer_letter": final_answer,
                    "bbox_display": "",
                    "bbox_original": "",
                    "bbox_valid": "",
                    "crop_path": "",
                    "crop_width": "",
                    "crop_height": "",
                    "input_token_count": result_final.get("input_token_count", ""),
                    "output_token_count": result_final.get("output_token_count", ""),
                    "finish_reason": result_final.get("finish_reason", ""),
                    "generation_time_sec": result_final.get("generation_time_sec", ""),
                    "error_message": "",
                }
            )
        except RuntimeError as exc:
            error_final = str(exc)
            status = "oom" if "out of memory" in error_final.lower() else "invalid"

    if status == "ok" and not final_answer:
        status = "invalid"

    correct = str(final_answer == gt_letter) if final_answer else "False"
    episode = {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "question": sample.question,
        "options_shuffled": json_dumps(options),
        "gt_answer_letter": gt_letter,
        "gt_answer_text": gt_text,
        "final_answer_letter": final_answer,
        "final_answer_text": answer_text(final_answer, options),
        "correct": correct,
        "status": status,
        "tool_use": str(len(crops) > 0),
        "turns_used": len(turns),
        "num_tool_calls": len(crops),
        "num_valid_tool_calls": len([c for c in crops if c["bbox_valid"]]),
        "num_invalid_actions": invalid_actions,
        "retry_count": retries,
        "forced_answer": str(forced_answer),
        "original_image_path": str(sample.image_path),
        "display_image_path": str(display_path),
        "display_image_width": display_w,
        "display_image_height": display_h,
        "display_image_sha256": display_sha,
        "original_image_width": original_w,
        "original_image_height": original_h,
        "original_image_sha256": original_sha,
        "scale_x": f"{scale_x:.8f}",
        "scale_y": f"{scale_y:.8f}",
        "total_input_tokens_at_final_answer": result_final.get("input_token_count", ""),
        "finish_reason_final": result_final.get("finish_reason", ""),
        "output_token_count_final": result_final.get("output_token_count", ""),
        "generation_time_total_sec": f"{time.perf_counter() - episode_start:.4f}",
    }
    trajectory = {
        "run_tag": run_tag,
        "few_shot": few_shot,
        "episode": episode,
        "turns": turns,
        "crops": [{k: v for k, v in crop.items() if k != "crop_image"} for crop in crops],
        "messages_before_final_serialized": serializable_messages(messages_before_final),
        "raw_final_output": raw_final,
        "error_final": error_final,
    }
    return {
        "episode": episode,
        "turns": turns,
        "crops": crops,
        "trajectory": trajectory,
        "messages_before_final": messages_before_final,
        "options": options,
        "gt_bbox": clamp_bbox_xywh(sample.gt_bbox, original_w, original_h),
        "display_image": display,
    }


def gate_a(episodes: Sequence[Dict]) -> Dict[str, object]:
    n = len(episodes)
    tool_counts = [int(ep["num_valid_tool_calls"]) for ep in episodes]
    ok = [ep for ep in episodes if ep["status"] == "ok"]
    metrics = {
        "n": n,
        "tool_use_rate": sum(c >= 1 for c in tool_counts) / n if n else 0,
        "mean_crops_per_episode": sum(tool_counts) / n if n else 0,
        "median_crops_per_episode": sorted(tool_counts)[n // 2] if n else 0,
        "num_tool_calls_distribution": {str(k): tool_counts.count(k) for k in sorted(set(tool_counts))},
        "fraction_ge_2_crops": sum(c >= 2 for c in tool_counts) / n if n else 0,
        "accuracy": sum(ep["correct"] == "True" for ep in episodes) / n if n else 0,
        "invalid_rate": sum(ep["status"] == "invalid" for ep in episodes) / n if n else 0,
        "oom_rate": sum(ep["status"] == "oom" for ep in episodes) / n if n else 0,
        "forced_answer_rate": sum(ep["forced_answer"] == "True" for ep in episodes) / n if n else 0,
        "ok_episodes": len(ok),
    }
    metrics["passes"] = bool(metrics["tool_use_rate"] >= 0.60 and metrics["fraction_ge_2_crops"] >= 0.30 and metrics["oom_rate"] == 0)
    return metrics


def label_crops(results: Sequence[Dict]) -> List[Dict]:
    rows = []
    for item in results:
        sample_id = item["episode"]["sample_id"]
        category = item["episode"]["category"]
        gt_bbox = item["gt_bbox"]
        gx, gy, gw, gh = gt_bbox
        center = (gx + gw / 2, gy + gh / 2)
        for crop in item["crops"]:
            x1, y1, x2, y2 = crop["bbox_original"]
            crop_xywh = [x1, y1, x2 - x1, y2 - y1]
            iou = bbox_iou_xywh(crop_xywh, gt_bbox)
            contains = x1 <= center[0] <= x2 and y1 <= center[1] <= y2
            label = "useful" if iou > 0.1 or contains else "failed_or_non_gt"
            crop["crop_label"] = label
            crop["iou_with_gt"] = iou
            crop["contains_gt_center"] = contains
            rows.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "turn_id": crop["turn_id"],
                    "crop_label": label,
                    "iou_with_gt": f"{iou:.8f}",
                    "contains_gt_center": str(contains),
                    "gt_bbox": json_dumps(gt_bbox),
                    "bbox_original": json_dumps(crop["bbox_original"]),
                    "crop_path": crop["crop_path"],
                }
            )
    return rows


def render_html(results: Sequence[Dict], output_dir: Path, limit: int = 5) -> List[str]:
    html_dir = output_dir / "sanity_html"
    html_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in results[:limit]:
        ep = item["episode"]
        display = item["display_image"].copy()
        draw = ImageDraw.Draw(display)
        sx = float(ep["display_image_width"]) / float(ep["original_image_width"])
        sy = float(ep["display_image_height"]) / float(ep["original_image_height"])
        for crop in item["crops"]:
            x1, y1, x2, y2 = crop["bbox_original"]
            color = "lime" if crop.get("crop_label") == "useful" else "red"
            draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=color, width=3)
        annotated_path = html_dir / f"{ep['sample_id'].replace('/', '__')}_display_annotated.png"
        display.save(annotated_path)
        crop_blocks = []
        for crop in item["crops"]:
            crop_blocks.append(
                f"<div><h3>Turn {crop['turn_id']} - {html.escape(crop.get('crop_label', ''))}</h3>"
                f"<p>IoU={crop.get('iou_with_gt', ''):.4f} contains_GT_center={crop.get('contains_gt_center', '')}</p>"
                f"<p>bbox_original={html.escape(json_dumps(crop['bbox_original']))}</p>"
                f"<img src='../../{html.escape(crop['crop_path'])}' style='max-width:360px;border:1px solid #ccc'></div>"
            )
        body = f"""
        <html><body>
        <h1>{html.escape(ep['sample_id'])}</h1>
        <p>Category: {html.escape(ep['category'])}</p>
        <p>Question: {html.escape(ep['question'])}</p>
        <p>Final answer: {html.escape(ep['final_answer_letter'])}, correct={html.escape(ep['correct'])}</p>
        <img src='{html.escape(annotated_path.name)}' style='max-width:720px;border:1px solid #888'>
        {''.join(crop_blocks)}
        </body></html>
        """
        path = html_dir / f"{ep['sample_id'].replace('/', '__')}.html"
        path.write_text(body, encoding="utf-8")
        paths.append(str(path))
    return paths


def build_replay_messages(item: Dict, variant: str) -> List[Dict]:
    ep = item["episode"]
    options = item["options"]
    display = item["display_image"]
    messages = [
        {"role": "system", "content": make_system_prompt(int(ep["display_image_width"]), int(ep["display_image_height"]), False)},
        make_initial_user(display, ep["question"], options),
    ]
    for crop in item["crops"]:
        label = crop.get("crop_label")
        if variant == "evict-silent" and label == "failed_or_non_gt":
            continue
        messages.append({"role": "assistant", "content": crop["assistant_output"]})
        if variant == "evict-marker" and label == "failed_or_non_gt":
            x1, y1, x2, y2 = [int(round(v)) for v in crop["bbox_original"]]
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool result: [searched region ({x1},{y1},{x2},{y2}) in the original image: nothing relevant found]",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Tool result:"}, {"type": "image", "image": crop["crop_image"]}],
                }
            )
    return messages


def run_eviction(backend: Backend, results: Sequence[Dict]) -> List[Dict]:
    rows = []
    subset = [item for item in results if any(crop.get("crop_label") == "failed_or_non_gt" for crop in item["crops"])]
    for item in subset:
        ep = item["episode"]
        for variant in ["original-rebuild", "evict-silent", "evict-marker"]:
            messages = item["messages_before_final"] if variant == "original-rebuild" else build_replay_messages(item, variant)
            status = "ok"
            raw = ""
            parsed = ""
            result = {}
            try:
                result = generate(backend, messages, FINAL_MAX_NEW_TOKENS)
                raw = str(result.get("raw", ""))
                action = parse_action(raw)
                parsed = str(action.get("answer_letter", "")) if action.get("type") == "answer" else ""
                if not parsed:
                    status = "invalid"
            except RuntimeError as exc:
                status = "oom" if "out of memory" in str(exc).lower() else "invalid"
            rows.append(
                {
                    "sample_id": ep["sample_id"],
                    "category": ep["category"],
                    "variant": variant,
                    "live_answer_letter": ep["final_answer_letter"],
                    "live_correct": ep["correct"],
                    "raw_output": raw,
                    "parsed_answer_letter": parsed,
                    "correct": str(parsed == ep["gt_answer_letter"]) if parsed else "False",
                    "status": status,
                    "input_token_count": result.get("input_token_count", ""),
                    "output_token_count": result.get("output_token_count", ""),
                    "finish_reason": result.get("finish_reason", ""),
                    "generation_time_sec": result.get("generation_time_sec", ""),
                }
            )
    return rows


def write_outputs(output_dir: Path, results: Sequence[Dict], crop_labels: Sequence[Dict], eviction_rows: Sequence[Dict], settings: Dict, gate: Dict, html_paths: Sequence[str], report: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [item["episode"] for item in results]
    turns = [turn for item in results for turn in item["turns"]]
    oom = [ep for ep in episodes if ep["status"] == "oom"]
    write_csv(output_dir / "e2_smoke_episode_summary.csv", episodes, EPISODE_FIELDS)
    write_csv(output_dir / "e2_smoke_turns.csv", turns, TURN_FIELDS)
    write_csv(output_dir / "e2_smoke_oom_log.csv", oom, EPISODE_FIELDS)
    write_csv(output_dir / "e2_smoke_crop_labels.csv", crop_labels, LABEL_FIELDS)
    write_csv(output_dir / "e2_smoke_eviction.csv", eviction_rows, EVICTION_FIELDS)
    (output_dir / "e2_smoke_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "e2_smoke_report.md").write_text(report, encoding="utf-8")
    with (output_dir / "e2_smoke_trajectories.jsonl").open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item["trajectory"], ensure_ascii=False) + "\n")
    transcript_blocks = []
    for item in results:
        ep = item["episode"]
        lines = [f"=== {ep['sample_id']} ===", f"final={ep['final_answer_letter']} gt={ep['gt_answer_letter']} correct={ep['correct']}"]
        for turn in item["turns"]:
            lines.append(f"[turn {turn['turn_id']} {turn['parsed_action_type']}] {turn['raw_model_output']}")
        transcript_blocks.append("\n".join(lines))
    (output_dir / "e2_smoke_transcripts.txt").write_text("\n\n".join(transcript_blocks), encoding="utf-8")


def make_report(attempt: str, gate: Dict, results: Sequence[Dict], crop_labels: Sequence[Dict], html_paths: Sequence[str], eviction_rows: Sequence[Dict]) -> str:
    episodes = [item["episode"] for item in results]
    useful = sum(row["crop_label"] == "useful" for row in crop_labels)
    failed = sum(row["crop_label"] == "failed_or_non_gt" for row in crop_labels)
    failed_episode_ids = {row["sample_id"] for row in crop_labels if row["crop_label"] == "failed_or_non_gt"}
    lines = [
        "# E2 Smoke Report",
        "",
        f"1. Few-shot mode: {attempt}",
        "",
        "2. Gate A tool-use numbers",
        json.dumps(gate, ensure_ascii=False, indent=2),
        "",
        "3. Accuracy and invalid/OOM audit",
        f"- Accuracy: {gate['accuracy']:.4f}",
        f"- Invalid rate: {gate['invalid_rate']:.4f}",
        f"- OOM rate: {gate['oom_rate']:.4f}",
        f"- Forced-answer rate: {gate['forced_answer_rate']:.4f}",
        "",
        "4. Crop-label prevalence on smoke",
        "- This oracle label is based on annotated GT target overlap and may undercount contextual evidence, especially for relation/context questions.",
        f"- useful crops: {useful}",
        f"- failed_or_non_gt crops: {failed}",
        f"- episodes with >=1 failed_or_non_gt crop: {len(failed_episode_ids)} / {len(episodes)}",
        "",
        "5. Gate B HTML paths and label sanity notes",
    ]
    if html_paths:
        lines.extend(f"- {path}" for path in html_paths)
    else:
        lines.append("- Gate B not run because Gate A failed.")
    lines.extend(["", "6. Oracle eviction smoke results"])
    if eviction_rows:
        import pandas as pd

        df = pd.DataFrame(eviction_rows)
        for variant, group in df.groupby("variant"):
            acc = group["correct"].eq("True").mean()
            lines.append(f"- {variant}: accuracy={acc:.4f}, n={len(group)}")
        live = df[df["variant"].eq("original-rebuild")]
        if not live.empty:
            mismatch = (live["parsed_answer_letter"] != live["live_answer_letter"]).mean()
            lines.append(f"- original-rebuild vs live answer mismatch rate: {mismatch:.4f}")
    else:
        lines.append("- Not run because Gate A failed or no failed_or_non_gt crops were found.")
    lines.extend(["", "7. Replay sanity: original-rebuild vs live-run mismatch rate"])
    if eviction_rows:
        import pandas as pd

        df = pd.DataFrame(eviction_rows)
        live = df[df["variant"].eq("original-rebuild")]
        mismatch = (live["parsed_answer_letter"] != live["live_answer_letter"]).mean() if not live.empty else 0
        lines.append(f"- mismatch_rate={mismatch:.4f}")
    else:
        lines.append("- NA")
    lines.extend(["", "8. Recommendation"])
    if gate["passes"]:
        lines.append("- Gate A passed. Gate B HTML was rendered; stop for human inspection before full E2.")
    else:
        lines.append("- Gate A failed. Stop and consider adjusted prompt or DeepEyes checkpoint; do not run full E2.")
    return "\n".join(lines)


def run_pass(backend: Backend, samples: Sequence, output_dir: Path, few_shot: bool, run_tag: str) -> Tuple[List[Dict], Dict]:
    results = []
    for idx, sample in enumerate(samples, start=1):
        item = run_episode(backend, sample, output_dir, few_shot=few_shot, run_tag=run_tag)
        results.append(item)
        ep = item["episode"]
        print(f"[{run_tag}] [{idx}/{len(samples)}] {sample.sample_id} status={ep['status']} crops={ep['num_valid_tool_calls']} answer={ep['final_answer_letter']} correct={ep['correct']}")
        if backend.torch.cuda.is_available():
            backend.torch.cuda.empty_cache()
    return results, gate_a([item["episode"] for item in results])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2 smoke with real multi-turn crop trajectories.")
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output_dir", default="outputs/e2_smoke")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = first_smoke_sample_ids(output_dir, args.data_root, args.num_samples)
    all_samples = {sample.sample_id: sample for sample in load_vstar_samples(args.data_root, num_samples=None)}
    samples = [all_samples[sid] for sid in sample_ids if sid in all_samples]
    if len(samples) != args.num_samples:
        raise RuntimeError(f"Expected {args.num_samples} samples, found {len(samples)}.")

    settings = {
        "model": args.model,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "batch_size": 1,
        "do_sample": False,
        "processor_min_pixels": MIN_PIXELS,
        "processor_max_pixels": MAX_PIXELS,
        "display_long_side": DISPLAY_LONG_SIDE,
        "crop_long_side": CROP_LONG_SIDE,
        "intermediate_max_new_tokens": INTERMEDIATE_MAX_NEW_TOKENS,
        "final_max_new_tokens": FINAL_MAX_NEW_TOKENS,
        "max_crop_turns": MAX_CROP_TURNS,
        "max_invalid_retries": MAX_INVALID_RETRIES,
        "sample_ids": sample_ids,
    }
    (output_dir / "e2_smoke_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("E2 smoke settings:")
    print(json.dumps(settings, ensure_ascii=False, indent=2))

    backend = Backend(args.model, MIN_PIXELS, MAX_PIXELS, "sdpa")
    results, gate = run_pass(backend, samples, output_dir, few_shot=False, run_tag="no_fewshot")
    attempt = "without few-shot"
    if not gate["passes"]:
        print("Gate A failed without few-shot; rerunning once with one generic few-shot exemplar.")
        shutil.rmtree(output_dir / "crops", ignore_errors=True)
        results, gate = run_pass(backend, samples, output_dir, few_shot=True, run_tag="fewshot")
        attempt = "with one generic few-shot fallback"

    crop_labels: List[Dict] = []
    html_paths: List[str] = []
    eviction_rows: List[Dict] = []
    if gate["passes"]:
        crop_labels = label_crops(results)
        html_paths = render_html(results, output_dir, limit=5)
        eviction_rows = run_eviction(backend, results)
    report = make_report(attempt, gate, results, crop_labels, html_paths, eviction_rows)
    write_outputs(output_dir, results, crop_labels, eviction_rows, settings | {"few_shot_attempt": attempt, "gate_a": gate}, gate, html_paths, report)
    print(report)


if __name__ == "__main__":
    main()
