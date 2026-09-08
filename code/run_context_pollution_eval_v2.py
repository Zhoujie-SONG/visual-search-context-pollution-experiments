import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from context_pollution_utils import (
    LETTERS,
    bbox_xywh_to_xyxy,
    clamp_bbox_xywh,
    deterministic_option_shuffle,
    format_options,
    json_dumps,
    load_vstar_samples,
    parse_answer_letter,
    sanitize_id,
    stable_seed,
)
from generate_crops import sample_irrelevant_bbox


SYSTEM_PROMPT = (
    "You are a visual assistant that can request zoomed-in crops of the image using\n"
    '<tool_call>{"name":"crop","arguments":{"bbox":[x1,y1,x2,y2]}}</tool_call>.'
)
ASSISTANT_TOOL_TEMPLATE = (
    'I will examine this region more closely.\n'
    '<tool_call>{"name":"crop","arguments":{"bbox":[%d,%d,%d,%d]}}</tool_call>'
)
FINAL_USER_PROMPT = (
    "Based on the original image and all examined regions, answer the question.\n"
    "Respond with ONLY the option letter."
)
CONDITION_TO_K = {
    "original_only": -1,
    "gt_crop_only": 0,
    "gt_plus_1_irrelevant": 1,
    "gt_plus_4_irrelevant": 4,
    "gt_plus_8_irrelevant": 8,
}


def condition_from_k(k: int) -> str:
    if k < 0:
        return "original_only"
    if k == 0:
        return "gt_crop_only"
    return f"gt_plus_{k}_irrelevant"


def crop_box_tuple(bbox: Sequence[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xywh_to_xyxy(bbox)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def xywh_to_xyxy_int(bbox: Sequence[float]) -> List[int]:
    return list(crop_box_tuple(bbox))


def zoom_crop_from_original(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    crop = image.crop(crop_box_tuple(bbox)).convert("RGB")
    long_side = max(crop.size)
    if long_side < 448:
        scale = 448 / max(1, long_side)
        new_size = (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale))))
        crop = crop.resize(new_size, Image.Resampling.BICUBIC)
    return crop


def hash_tensor(tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def robust_parse_answer(raw: str, options: Sequence[str]) -> Tuple[str, str]:
    letter = parse_answer_letter(raw, LETTERS[: len(options)])
    if letter:
        return letter, options[LETTERS.index(letter)]
    raw_lower = str(raw).strip().lower()
    for idx, option in enumerate(options):
        option_norm = str(option).strip().lower()
        if option_norm and option_norm in raw_lower:
            return LETTERS[idx], option
    for idx, option in enumerate(options):
        # Fallback for concise options embedded in full-sentence options.
        words = [w for w in str(option).lower().replace(".", "").replace('"', "").split() if len(w) > 2]
        tail = " ".join(words[-3:])
        if tail and tail in raw_lower:
            return LETTERS[idx], option
    return "", ""


def build_messages(
    original: Image.Image,
    crops: Sequence[Dict],
    question: str,
    options: Sequence[str],
) -> Tuple[List[Dict], List[Image.Image], str]:
    first_text = f"Question: {question}\nOptions:\n{format_options(options)}"
    messages: List[Dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": original}, {"type": "text", "text": first_text}]},
    ]
    images = [original]
    transcript = [
        f"[system] {SYSTEM_PROMPT}",
        "[user] <full original image>",
        first_text,
    ]
    for item in crops:
        bbox = item["bbox_xyxy"]
        assistant_text = ASSISTANT_TOOL_TEMPLATE % tuple(bbox)
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "Tool result:"}, {"type": "image", "image": item["image"]}],
            }
        )
        images.append(item["image"])
        transcript.append(f"[assistant] {assistant_text}")
        transcript.append(f"[user] Tool result:\n<{item['kind']} crop image; bbox={bbox}>")
    messages.append({"role": "user", "content": FINAL_USER_PROMPT})
    transcript.append(f"[user] {FINAL_USER_PROMPT}")
    return messages, images, "\n".join(transcript)


class QwenV2Backend:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "auto"})
        else:
            kwargs.update({"torch_dtype": torch.float32})
        self.model = ModelClass.from_pretrained(model_path, **kwargs)
        self.model.eval()

    def image_tensor_hash(self, image: Image.Image) -> Tuple[str, int]:
        values = self.processor.image_processor(images=[image], return_tensors="pt")
        pixel_values = values["pixel_values"]
        visual_tokens = ""
        if "image_grid_thw" in values:
            grid = values["image_grid_thw"][0].tolist()
            visual_tokens = int(grid[0] * grid[1] * grid[2])
        return hash_tensor(pixel_values), visual_tokens

    def generate(self, messages: Sequence[Dict], images: Sequence[Image.Image], max_new_tokens: int) -> Dict[str, object]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=list(images), padding=True, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        visual_tokens = ""
        if "image_grid_thw" in inputs:
            visual_tokens = int(sum(int(g[0] * g[1] * g[2]) for g in inputs["image_grid_thw"]))
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        elapsed = time.perf_counter() - start
        seq = generated.sequences
        new_ids = seq[:, input_len:]
        raw = self.processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        last = int(seq[0, -1].item())
        finish_reason = "eos" if last in eos_ids else ("length" if new_ids.shape[1] >= max_new_tokens else "unknown")
        return {
            "raw_output": raw,
            "input_token_count": input_len,
            "estimated_visual_token_count": visual_tokens,
            "output_token_count": int(new_ids.shape[1]),
            "finish_reason": finish_reason,
            "inference_time_sec": f"{elapsed:.4f}",
        }


def make_condition_crops(
    original: Image.Image,
    gt_bbox: Sequence[float],
    irrelevant_bboxes: Sequence[Sequence[float]],
    sample_id: str,
    condition: str,
    seed: int,
) -> Tuple[List[Dict], str]:
    k = CONDITION_TO_K[condition]
    if k < 0:
        return [], ""
    items = [{"kind": "gt", "bbox_xywh": gt_bbox}] + [
        {"kind": "irrelevant", "bbox_xywh": bbox} for bbox in irrelevant_bboxes[:k]
    ]
    rng = random.Random(stable_seed(seed, sample_id, condition, "v2_gt_position"))
    rng.shuffle(items)
    crops = []
    gt_position = ""
    for idx, item in enumerate(items, start=1):
        if item["kind"] == "gt":
            gt_position = str(idx)
        crop = zoom_crop_from_original(original, item["bbox_xywh"])
        crops.append(
            {
                "kind": item["kind"],
                "bbox_xywh": item["bbox_xywh"],
                "bbox_xyxy": xywh_to_xyxy_int(item["bbox_xywh"]),
                "image": crop,
                "width": crop.width,
                "height": crop.height,
            }
        )
    return crops, gt_position


def sample_irrelevant_bboxes(
    original: Image.Image,
    gt_bbox: Sequence[float],
    sample_id: str,
    seed: int,
    max_k: int,
) -> List[List[float]]:
    rng = random.Random(stable_seed(seed, sample_id, "v2_irrelevant_crops"))
    return [sample_irrelevant_bbox(rng, gt_bbox, original.width, original.height) for _ in range(max_k)]


def write_header(path: Path, fieldnames: Sequence[str], overwrite: bool, resume: bool) -> set:
    if resume and path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            return {
                (r["seed"], r["sample_id"], r["condition"])
                for r in csv.DictReader(f)
                if r.get("seed") and r.get("sample_id") and r.get("condition")
            }
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    return set()


def append_row(path: Path, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1-v2 separate-image context pollution evaluation.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_csv", default="outputs/results_context_pollution_full_v2.csv")
    parser.add_argument("--transcript_examples", default="")
    parser.add_argument("--num_samples", type=int, default=238)
    parser.add_argument("--conditions", nargs="+", default=["original_only", "gt_crop_only", "gt_plus_1_irrelevant", "gt_plus_4_irrelevant", "gt_plus_8_irrelevant"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--option_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    samples = load_vstar_samples(args.data_root, num_samples=args.num_samples)
    max_k = max(CONDITION_TO_K[c] for c in args.conditions)
    backend = QwenV2Backend(args.model)

    fieldnames = [
        "seed",
        "sample_id",
        "category",
        "condition",
        "k_irrelevant",
        "image_path",
        "question",
        "options_shuffled",
        "gt_answer_letter",
        "gt_answer_text",
        "gt_bbox_xywh",
        "gt_position",
        "num_tool_turns",
        "num_images",
        "crop_bboxes_xyxy",
        "crop_kinds_in_order",
        "crop_sizes_in_order",
        "full_image_hash",
        "gt_crop_hash",
        "full_image_visual_tokens",
        "gt_crop_visual_tokens",
        "input_token_count",
        "estimated_visual_token_count",
        "max_new_tokens",
        "output_token_count",
        "finish_reason",
        "model_raw_output",
        "pred_answer_letter",
        "pred_answer_text",
        "valid",
        "correct",
        "inference_time_sec",
        "error",
    ]
    output_csv = Path(args.output_csv)
    completed = write_header(output_csv, fieldnames, args.overwrite, args.resume)
    transcript_lines: List[str] = []
    hash_registry: Dict[str, Tuple[str, str]] = {}

    for seed in args.seeds:
        for sample_index, sample in enumerate(samples, start=1):
            with Image.open(sample.image_path) as original_in:
                original = original_in.convert("RGB")
                gt_bbox = clamp_bbox_xywh(sample.gt_bbox, original.width, original.height)
                gt_crop = zoom_crop_from_original(original, gt_bbox)
                full_hash, full_visual_tokens = backend.image_tensor_hash(original)
                gt_hash, gt_visual_tokens = backend.image_tensor_hash(gt_crop)
                prior = hash_registry.setdefault(sample.sample_id, (full_hash, gt_hash))
                if prior != (full_hash, gt_hash):
                    raise RuntimeError(f"Processed full/GT hash changed for sample {sample.sample_id}")
                irrelevant_bboxes = sample_irrelevant_bboxes(original, gt_bbox, sample.sample_id, seed, max(0, max_k))

                options, gt_letter, gt_text = deterministic_option_shuffle(sample.options, sample.sample_id, args.option_seed)
                for condition in args.conditions:
                    key = (str(seed), sample.sample_id, condition)
                    if key in completed:
                        continue
                    error = ""
                    raw = ""
                    pred_letter = ""
                    pred_text = ""
                    valid = "False"
                    correct = "False"
                    try:
                        crops, gt_position = make_condition_crops(
                            original, gt_bbox, irrelevant_bboxes, sample.sample_id, condition, seed
                        )
                        messages, images, transcript = build_messages(original, crops, sample.question, options)
                        result = backend.generate(messages, images, args.max_new_tokens)
                        raw = str(result["raw_output"])
                        pred_letter, pred_text = robust_parse_answer(raw, options)
                        valid = str(bool(pred_letter))
                        correct = str(pred_letter == gt_letter) if pred_letter else "False"
                        if len(transcript_lines) < 5:
                            transcript_lines.append(
                                f"=== seed={seed} sample={sample.sample_id} condition={condition} ===\n"
                                f"{transcript}\n[assistant raw] {raw}\n[pred] {pred_letter} [gt] {gt_letter}\n"
                            )
                    except Exception as exc:
                        result = {}
                        crops = []
                        gt_position = ""
                        error = str(exc)
                    row = {
                        "seed": seed,
                        "sample_id": sample.sample_id,
                        "category": sample.category,
                        "condition": condition,
                        "k_irrelevant": CONDITION_TO_K[condition],
                        "image_path": str(sample.image_path),
                        "question": sample.question,
                        "options_shuffled": json_dumps(options),
                        "gt_answer_letter": gt_letter,
                        "gt_answer_text": gt_text,
                        "gt_bbox_xywh": json_dumps(gt_bbox),
                        "gt_position": gt_position,
                        "num_tool_turns": len(crops),
                        "num_images": 1 + len(crops),
                        "crop_bboxes_xyxy": json_dumps([c["bbox_xyxy"] for c in crops]),
                        "crop_kinds_in_order": json_dumps([c["kind"] for c in crops]),
                        "crop_sizes_in_order": json_dumps([[c["width"], c["height"]] for c in crops]),
                        "full_image_hash": full_hash,
                        "gt_crop_hash": gt_hash,
                        "full_image_visual_tokens": full_visual_tokens,
                        "gt_crop_visual_tokens": gt_visual_tokens,
                        "input_token_count": result.get("input_token_count", ""),
                        "estimated_visual_token_count": result.get("estimated_visual_token_count", ""),
                        "max_new_tokens": args.max_new_tokens,
                        "output_token_count": result.get("output_token_count", ""),
                        "finish_reason": result.get("finish_reason", ""),
                        "model_raw_output": raw,
                        "pred_answer_letter": pred_letter,
                        "pred_answer_text": pred_text,
                        "valid": valid,
                        "correct": correct,
                        "inference_time_sec": result.get("inference_time_sec", ""),
                        "error": error,
                    }
                    append_row(output_csv, fieldnames, row)
            print(f"[seed {seed}] [{sample_index}/{len(samples)}] {sample.sample_id}")

    if args.transcript_examples:
        Path(args.transcript_examples).write_text("\n\n".join(transcript_lines), encoding="utf-8")
    print(f"wrote results: {output_csv}")


if __name__ == "__main__":
    main()
