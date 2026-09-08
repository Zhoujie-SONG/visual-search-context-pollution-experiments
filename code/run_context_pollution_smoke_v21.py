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
    deterministic_option_shuffle,
    format_options,
    json_dumps,
    load_vstar_samples,
    parse_answer_letter,
    read_crop_manifest,
    stable_seed,
)


SYSTEM_PROMPT = "You are a visual assistant that can examine zoomed-in crops of an image."
ASSISTANT_TOOL_TEMPLATE = (
    'I will examine this region more closely.\n'
    '<tool_call>{"name":"crop","arguments":{"bbox":[%d,%d,%d,%d]}}</tool_call>'
)
FINAL_USER_PROMPT = (
    "Based on the original image and all examined regions, answer the question.\n"
    "Output only a single option letter: A, B, C, or D. Do not explain."
)
CONDITIONS = ["original_only", "gt_crop_only", "gt_plus_8_irrelevant"]
CONDITION_TO_K = {"original_only": -1, "gt_crop_only": 0, "gt_plus_8_irrelevant": 8}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_image_pixels(image: Image.Image) -> str:
    image = image.convert("RGB")
    return hashlib.sha256(image.tobytes()).hexdigest()


def hash_tensor(tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def bbox_xywh_to_xyxy_int(bbox: Sequence[float]) -> List[int]:
    x, y, w, h = [float(v) for v in bbox[:4]]
    return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]


def resize_crop_long_side(image: Image.Image, long_side: int) -> Image.Image:
    image = image.convert("RGB")
    current = max(image.size)
    if current != long_side:
        scale = long_side / max(1, current)
        new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
        image = image.resize(new_size, Image.Resampling.BICUBIC)
    return pad_image_min_edge(image, min_size=28)


def pad_image_min_edge(image: Image.Image, min_size: int = 28) -> Image.Image:
    image = image.convert("RGB")
    target_w = max(image.width, min_size)
    target_h = max(image.height, min_size)
    if image.width >= min_size and image.height >= min_size:
        return image

    left = (target_w - image.width) // 2
    right = target_w - image.width - left
    top = (target_h - image.height) // 2
    bottom = target_h - image.height - top

    horizontal = Image.new("RGB", (target_w, image.height))
    horizontal.paste(image, (left, 0))
    if left:
        horizontal.paste(image.crop((0, 0, 1, image.height)).resize((left, image.height)), (0, 0))
    if right:
        horizontal.paste(
            image.crop((image.width - 1, 0, image.width, image.height)).resize((right, image.height)),
            (left + image.width, 0),
        )

    padded = Image.new("RGB", (target_w, target_h))
    padded.paste(horizontal, (0, top))
    if top:
        padded.paste(horizontal.crop((0, 0, target_w, 1)).resize((target_w, top)), (0, 0))
    if bottom:
        padded.paste(
            horizontal.crop((0, image.height - 1, target_w, image.height)).resize((target_w, bottom)),
            (0, top + image.height),
        )
    return padded


def robust_parse(raw: str, options: Sequence[str]) -> Tuple[str, str]:
    letter = parse_answer_letter(raw, LETTERS[: len(options)])
    if letter:
        return letter, options[LETTERS.index(letter)]
    raw_lower = str(raw).lower()
    for i, opt in enumerate(options):
        if str(opt).lower() in raw_lower:
            return LETTERS[i], opt
    return "", ""


def build_messages(
    original: Image.Image,
    crop_items: Sequence[Dict],
    question: str,
    options: Sequence[str],
) -> Tuple[List[Dict], List[Image.Image], str]:
    first_text = f"Question: {question}\nOptions:\n{format_options(options)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": original}, {"type": "text", "text": first_text}]},
    ]
    images = [original]
    transcript = [f"[system] {SYSTEM_PROMPT}", f"[user] <original image>\n{first_text}"]
    for item in crop_items:
        assistant = ASSISTANT_TOOL_TEMPLATE % tuple(item["bbox_xyxy"])
        messages.append({"role": "assistant", "content": assistant})
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": "Tool result:"}, {"type": "image", "image": item["image"]}]}
        )
        images.append(item["image"])
        transcript.append(f"[assistant] {assistant}")
        transcript.append(f"[user] Tool result:\n<image path={item['path']} bbox={item['bbox_xyxy']}>")
    messages.append({"role": "user", "content": FINAL_USER_PROMPT})
    transcript.append(f"[user] {FINAL_USER_PROMPT}")
    return messages, images, "\n".join(transcript)


class Backend:
    def __init__(self, model_path: str, min_pixels: int, max_pixels: int, attn: str):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass

        self.torch = torch
        self.dtype = "bfloat16" if torch.cuda.is_available() else "float32"
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        kwargs = {"trust_remote_code": True, "attn_implementation": attn}
        if torch.cuda.is_available():
            kwargs.update({"torch_dtype": torch.bfloat16, "device_map": {"": 0}})
        else:
            kwargs.update({"torch_dtype": torch.float32})
        try:
            self.model = ModelClass.from_pretrained(model_path, **kwargs)
            self.attn_implementation = attn
        except Exception:
            if attn != "sdpa":
                kwargs["attn_implementation"] = "sdpa"
                self.model = ModelClass.from_pretrained(model_path, **kwargs)
                self.attn_implementation = "sdpa"
            else:
                raise
        self.model.eval()

    def processed_hash(self, image: Image.Image) -> Tuple[str, int]:
        values = self.processor.image_processor(images=[image], return_tensors="pt")
        visual_tokens = ""
        if "image_grid_thw" in values:
            g = values["image_grid_thw"][0].tolist()
            visual_tokens = int(g[0] * g[1] * g[2])
        return hash_tensor(values["pixel_values"]), visual_tokens

    def generate(self, messages: Sequence[Dict], images: Sequence[Image.Image], max_new_tokens: int) -> Dict[str, object]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=list(images), padding=True, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        visual_tokens = ""
        if "image_grid_thw" in inputs:
            visual_tokens = int(sum(int(g[0] * g[1] * g[2]) for g in inputs["image_grid_thw"]))
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        start = time.perf_counter()
        try:
            with self.torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
            elapsed = time.perf_counter() - start
            seq = out.sequences
            new_ids = seq[:, input_len:]
            raw = self.processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            last = int(seq[0, -1].item())
            finish = "eos" if last in eos_ids else ("length" if new_ids.shape[1] >= max_new_tokens else "unknown")
            return {
                "raw": raw,
                "input_token_count": input_len,
                "estimated_visual_token_count": visual_tokens,
                "output_token_count": int(new_ids.shape[1]),
                "finish_reason": finish,
                "generation_time_sec": f"{elapsed:.4f}",
            }
        finally:
            del inputs
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1-v2.1 smoke with processor limits and strict OOM logging.")
    parser.add_argument("--data_root", default="/home/songzhoujie/cvpr27/vstar/data/vstar_bench")
    parser.add_argument("--model", default="/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--crop_dir", default="outputs/crops")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--crop_long_side", type=int, default=448)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_paths = {
        "results": Path("outputs/results_context_pollution_smoke_v21.csv"),
        "summary": Path("outputs/summary_context_pollution_smoke_v21.csv"),
        "transcripts": Path("outputs/transcripts_smoke_v21.txt"),
        "format": Path("outputs/format_audit_smoke_v21.csv"),
        "gt": Path("outputs/gt_crop_consistency_smoke_v21.csv"),
        "oom": Path("outputs/oom_log_smoke_v21.csv"),
        "settings": Path("outputs/settings_smoke_v21.json"),
    }
    if not args.overwrite:
        existing = [str(p) for p in output_paths.values() if p.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite without --overwrite:\n" + "\n".join(existing))

    backend = Backend(args.model, args.min_pixels, args.max_pixels, args.attn_implementation)
    samples = load_vstar_samples(args.data_root, num_samples=args.num_samples)
    manifest = read_crop_manifest(Path(args.crop_dir) / "manifest_context_pollution.jsonl")

    settings = {
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "crop_long_side": args.crop_long_side,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": backend.attn_implementation,
        "dtype": backend.dtype,
        "device": "cuda:0" if backend.torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "conditions": CONDITIONS,
    }
    output_paths["settings"].write_text(json.dumps(settings, indent=2), encoding="utf-8")

    result_rows: List[Dict] = []
    format_rows: List[Dict] = []
    gt_rows: List[Dict] = []
    oom_rows: List[Dict] = []
    transcript_blocks: List[str] = []
    gt_consistency: Dict[str, Tuple[str, int, int, str]] = {}
    full_hashes: Dict[str, str] = {}

    for sample_index, sample in enumerate(samples, start=1):
        if sample.sample_id not in manifest:
            raise RuntimeError(f"Missing crop manifest for {sample.sample_id}")
        info = manifest[sample.sample_id]
        gt_path = Path(info["gt_crop_path"])
        if not gt_path.exists():
            raise RuntimeError(f"Missing GT crop file: {gt_path}")
        gt_hash_file = sha256_file(gt_path)
        with Image.open(gt_path) as gt_in:
            gt_img_base = resize_crop_long_side(gt_in.convert("RGB"), args.crop_long_side)
            gt_width, gt_height = gt_img_base.size
            gt_pixel_hash = sha256_image_pixels(gt_img_base)
        gt_key = (str(gt_path), gt_width, gt_height, gt_pixel_hash)
        prior_gt = gt_consistency.setdefault(sample.sample_id, gt_key)
        if prior_gt != gt_key:
            raise RuntimeError(f"GT crop consistency failed for {sample.sample_id}")

        with Image.open(sample.image_path) as orig_in:
            original = orig_in.convert("RGB")
            original_file_hash = sha256_file(sample.image_path)
            full_processed_hash, _ = backend.processed_hash(original)
            prior_full = full_hashes.setdefault(sample.sample_id, full_processed_hash)
            if prior_full != full_processed_hash:
                raise RuntimeError(f"Full image processed hash changed for {sample.sample_id}")

            options, gt_letter, gt_text = deterministic_option_shuffle(sample.options, sample.sample_id, 42)
            for condition in CONDITIONS:
                k = CONDITION_TO_K[condition]
                crop_entries = []
                if k >= 0:
                    all_items = [{"kind": "gt", "path": str(gt_path), "bbox": info["gt_bbox"]}]
                    all_items.extend(
                        {
                            "kind": "irrelevant",
                            "path": item["path"],
                            "bbox": item["bbox"],
                        }
                        for item in info["irrelevant_crops"][:k]
                    )
                    rng = random.Random(stable_seed(args.seed, sample.sample_id, condition, "v21_gt_position"))
                    rng.shuffle(all_items)
                    for item in all_items:
                        path = Path(item["path"])
                        with Image.open(path) as crop_in:
                            crop_img = resize_crop_long_side(crop_in.convert("RGB"), args.crop_long_side)
                        crop_entries.append(
                            {
                                "kind": item["kind"],
                                "path": str(path),
                                "bbox_xyxy": bbox_xywh_to_xyxy_int(item["bbox"]),
                                "image": crop_img,
                            }
                        )
                gt_position = next((str(i + 1) for i, c in enumerate(crop_entries) if c["kind"] == "gt"), "")
                messages, images, transcript = build_messages(original, crop_entries, sample.question, options)
                all_paths = [str(sample.image_path)] + [c["path"] for c in crop_entries]
                gt_crop_path_for_row = str(gt_path) if k >= 0 else ""
                status = "ok"
                error = ""
                result = {}
                raw = ""
                pred = ""
                pred_text = ""
                valid = "False"
                correct = "False"
                try:
                    result = backend.generate(messages, images, args.max_new_tokens)
                    raw = result["raw"]
                    pred, pred_text = robust_parse(raw, options)
                    if pred:
                        valid = "True"
                        correct = str(pred == gt_letter)
                    else:
                        status = "invalid"
                except RuntimeError as exc:
                    error = str(exc)
                    if "out of memory" in error.lower() or "cuda" in error.lower() and "memory" in error.lower():
                        status = "oom"
                        if backend.torch.cuda.is_available():
                            backend.torch.cuda.empty_cache()
                    else:
                        status = "invalid"
                row = {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "condition": condition,
                    "k_irrelevant": k,
                    "status": status,
                    "original_image_path": str(sample.image_path),
                    "gt_crop_path": gt_crop_path_for_row,
                    "irrelevant_crop_paths": json_dumps([c["path"] for c in crop_entries if c["kind"] == "irrelevant"]),
                    "all_image_paths_used": json_dumps(all_paths),
                    "gt_crop_position": gt_position,
                    "gt_crop_hash": gt_pixel_hash if k >= 0 else "",
                    "gt_crop_file_hash": gt_hash_file if k >= 0 else "",
                    "gt_crop_width": gt_width if k >= 0 else "",
                    "gt_crop_height": gt_height if k >= 0 else "",
                    "original_image_hash": original_file_hash,
                    "full_image_processed_hash": full_processed_hash,
                    "prompt_text": sample.question + "\n" + format_options(options),
                    "gt_answer_letter": gt_letter,
                    "gt_answer_text": gt_text,
                    "model_raw_output": raw,
                    "pred_answer_letter": pred,
                    "pred_answer_text": pred_text,
                    "correct": correct,
                    "valid_output": valid,
                    "input_token_count": result.get("input_token_count", ""),
                    "estimated_visual_token_count": result.get("estimated_visual_token_count", ""),
                    "output_token_count": result.get("output_token_count", ""),
                    "finish_reason": result.get("finish_reason", ""),
                    "generation_time_sec": result.get("generation_time_sec", ""),
                    "processor_min_pixels": args.min_pixels,
                    "processor_max_pixels": args.max_pixels,
                    "crop_long_side": args.crop_long_side,
                    "attn_implementation": backend.attn_implementation,
                    "error_message": error,
                }
                result_rows.append(row)
                if status == "oom":
                    oom_rows.append(row)
                format_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "condition": condition,
                        "uses_contact_sheet": any("contact_sheet" in p for p in all_paths),
                        "num_messages": len(messages),
                        "num_images": len(images),
                        "expected_images": 1 + max(0, k + 1),
                        "separate_image_messages_pass": len(images) == 1 + max(0, k + 1),
                        "has_system_prompt": messages[0]["role"] == "system",
                        "has_final_instruction": messages[-1]["role"] == "user" and "Output only" in str(messages[-1]["content"]),
                    }
                )
                gt_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "condition": condition,
                        "gt_crop_path": gt_crop_path_for_row,
                        "gt_crop_width": gt_width if k >= 0 else "",
                        "gt_crop_height": gt_height if k >= 0 else "",
                        "gt_crop_hash": gt_pixel_hash if k >= 0 else "",
                        "full_image_processed_hash": full_processed_hash,
                        "gt_consistency_pass": True,
                        "full_image_consistency_pass": True,
                    }
                )
                if len(transcript_blocks) < 5 and status != "oom":
                    transcript_blocks.append(
                        f"=== {sample.sample_id} {condition} ===\n"
                        f"image_paths={json_dumps(all_paths)}\n{transcript}\n[assistant] {raw}\n[pred={pred} gt={gt_letter} status={status}]\n"
                    )
                del images, messages, crop_entries
                if backend.torch.cuda.is_available():
                    backend.torch.cuda.empty_cache()
        print(f"[{sample_index}/{len(samples)}] {sample.sample_id}")
        if oom_rows:
            print("OOM detected; stopping smoke early.")
            break

    result_fields = [
        "sample_id",
        "category",
        "condition",
        "k_irrelevant",
        "status",
        "original_image_path",
        "gt_crop_path",
        "irrelevant_crop_paths",
        "all_image_paths_used",
        "gt_crop_position",
        "gt_crop_hash",
        "gt_crop_file_hash",
        "gt_crop_width",
        "gt_crop_height",
        "original_image_hash",
        "full_image_processed_hash",
        "prompt_text",
        "gt_answer_letter",
        "gt_answer_text",
        "model_raw_output",
        "pred_answer_letter",
        "pred_answer_text",
        "correct",
        "valid_output",
        "input_token_count",
        "estimated_visual_token_count",
        "output_token_count",
        "finish_reason",
        "generation_time_sec",
        "processor_min_pixels",
        "processor_max_pixels",
        "crop_long_side",
        "attn_implementation",
        "error_message",
    ]
    write_csv(output_paths["results"], result_rows, result_fields)
    write_csv(output_paths["format"], format_rows, list(format_rows[0].keys()) if format_rows else ["sample_id"])
    write_csv(output_paths["gt"], gt_rows, list(gt_rows[0].keys()) if gt_rows else ["sample_id"])
    write_csv(output_paths["oom"], oom_rows, result_fields)
    output_paths["transcripts"].write_text("\n\n".join(transcript_blocks), encoding="utf-8")

    summary_rows = []
    for condition in CONDITIONS:
        rows = [r for r in result_rows if r["condition"] == condition]
        ok_or_invalid = [r for r in rows if r["status"] != "oom"]
        valid = [r for r in rows if r["valid_output"] == "True"]
        correct = [r for r in rows if r["correct"] == "True"]
        oom = [r for r in rows if r["status"] == "oom"]
        summary_rows.append(
            {
                "condition": condition,
                "n": len(rows),
                "oom_count": len(oom),
                "oom_rate": len(oom) / len(rows) if rows else "",
                "invalid_count": sum(r["status"] == "invalid" for r in rows),
                "invalid_rate_excluding_oom": sum(r["status"] == "invalid" for r in rows) / len(ok_or_invalid)
                if ok_or_invalid
                else "",
                "accuracy_invalid_as_wrong_excluding_oom": len(correct) / len(ok_or_invalid) if ok_or_invalid else "",
                "valid_only_accuracy": len(correct) / len(valid) if valid else "",
                "mean_input_token_count": sum(float(r["input_token_count"]) for r in rows if r["input_token_count"]) / max(1, sum(bool(r["input_token_count"]) for r in rows)),
            }
        )
    write_csv(output_paths["summary"], summary_rows, list(summary_rows[0].keys()))
    print("wrote E1-v2.1 smoke outputs")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
