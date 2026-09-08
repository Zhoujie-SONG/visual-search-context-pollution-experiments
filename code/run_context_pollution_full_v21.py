import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from context_pollution_utils import (
    bbox_iou_xywh,
    clamp_bbox_xywh,
    deterministic_option_shuffle,
    format_options,
    json_dumps,
    load_vstar_samples,
    read_crop_manifest,
    sanitize_id,
    stable_seed,
)
from generate_crops import sample_irrelevant_bbox, save_crop
from run_context_pollution_smoke_v21 import (
    ASSISTANT_TOOL_TEMPLATE,
    FINAL_USER_PROMPT,
    SYSTEM_PROMPT,
    Backend,
    bbox_xywh_to_xyxy_int,
    build_messages,
    resize_crop_long_side,
    robust_parse,
    sha256_file,
    sha256_image_pixels,
)


CONDITIONS = [
    "original_only",
    "gt_crop_only",
    "gt_plus_1_irrelevant",
    "gt_plus_4_irrelevant",
    "gt_plus_8_irrelevant",
]
CONDITION_TO_K = {
    "original_only": -1,
    "gt_crop_only": 0,
    "gt_plus_1_irrelevant": 1,
    "gt_plus_4_irrelevant": 4,
    "gt_plus_8_irrelevant": 8,
}
SEED42_CROP_SEED = 42
OPTION_SEED = 42


FIELDNAMES = [
    "seed",
    "shard_id",
    "num_shards",
    "sample_index",
    "sample_id",
    "category",
    "condition",
    "k_irrelevant",
    "status",
    "original_image_path",
    "gt_crop_path",
    "irrelevant_crop_paths",
    "all_image_paths_used",
    "image_kinds_in_order",
    "image_sizes_in_order",
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
    "per_image_visual_token_counts",
    "distractor_visual_token_fraction",
    "output_token_count",
    "finish_reason",
    "generation_time_sec",
    "max_new_tokens",
    "do_sample",
    "processor_min_pixels",
    "processor_max_pixels",
    "crop_long_side",
    "attn_implementation",
    "dtype",
    "error_message",
]


def load_smoke_settings(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        settings = json.load(f)
    print("Frozen smoke settings:")
    print(json.dumps(settings, ensure_ascii=False, indent=2))
    return settings


def is_complete_row(row: Dict[str, str]) -> bool:
    if row.get("status") not in {"ok", "invalid", "oom"}:
        return False
    required = [
        "seed",
        "sample_id",
        "condition",
        "status",
        "gt_answer_letter",
        "correct",
        "valid_output",
        "processor_min_pixels",
        "processor_max_pixels",
        "crop_long_side",
        "max_new_tokens",
        "do_sample",
    ]
    return all(key in row and str(row.get(key, "")) != "" for key in required)


def prepare_output(path: Path, overwrite: bool, resume: bool) -> set:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and resume:
        complete_rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if is_complete_row(row):
                    complete_rows.append(row)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in complete_rows:
                writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
        return {(str(row["seed"]), row["sample_id"], row["condition"]) for row in complete_rows}
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    return set()


def append_row(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def seed_irrelevant_items(
    seed: int,
    sample,
    original: Image.Image,
    gt_bbox: Sequence[float],
    base_info: Dict,
    output_root: Path,
    max_k: int,
) -> List[Dict]:
    if seed == SEED42_CROP_SEED:
        return [
            {
                "path": item["path"],
                "bbox": item["bbox"],
                "iou_with_gt": item.get("iou_with_gt", ""),
            }
            for item in base_info["irrelevant_crops"][:max_k]
        ]

    sample_dir = output_root / f"seed{seed}" / sanitize_id(sample.sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(stable_seed(seed, sample.sample_id, "irrelevant_crops"))
    items = []
    for crop_index in range(max_k):
        bbox = sample_irrelevant_bbox(rng, gt_bbox, original.width, original.height)
        crop_path = sample_dir / f"irrelevant_{crop_index:02d}.png"
        if not crop_path.exists():
            save_crop(original, bbox, crop_path, overwrite=True)
        items.append(
            {
                "path": str(crop_path),
                "bbox": bbox,
                "iou_with_gt": bbox_iou_xywh(bbox, gt_bbox),
            }
        )
    return items


def condition_crop_items(
    seed: int,
    sample_id: str,
    condition: str,
    gt_path: Path,
    gt_bbox: Sequence[float],
    irrelevant_items: Sequence[Dict],
    crop_long_side: int,
) -> Tuple[List[Dict], str]:
    k = CONDITION_TO_K[condition]
    if k < 0:
        return [], ""

    items = [{"kind": "gt", "path": str(gt_path), "bbox": gt_bbox}]
    items.extend(
        {
            "kind": "irrelevant",
            "path": item["path"],
            "bbox": item["bbox"],
        }
        for item in irrelevant_items[:k]
    )
    rng = random.Random(stable_seed(seed, sample_id, condition, "v21_gt_position"))
    rng.shuffle(items)

    crops = []
    gt_position = ""
    for index, item in enumerate(items, start=1):
        if item["kind"] == "gt":
            gt_position = str(index)
        path = Path(item["path"])
        with Image.open(path) as crop_in:
            crop_img = resize_crop_long_side(crop_in.convert("RGB"), crop_long_side)
        crops.append(
            {
                "kind": item["kind"],
                "path": str(path),
                "bbox_xyxy": bbox_xywh_to_xyxy_int(item["bbox"]),
                "image": crop_img,
            }
        )
    return crops, gt_position


def generate_with_diagnostics(
    backend: Backend,
    messages: Sequence[Dict],
    images: Sequence[Image.Image],
    image_kinds: Sequence[str],
    max_new_tokens: int,
) -> Dict[str, object]:
    text = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = backend.processor(text=[text], images=list(images), padding=True, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[1])

    per_image_tokens = ""
    total_visual_tokens = ""
    distractor_fraction = ""
    if "image_grid_thw" in inputs:
        grids = inputs["image_grid_thw"].tolist()
        if len(grids) == len(images):
            counts = [int(g[0] * g[1] * g[2]) for g in grids]
            total = sum(counts)
            distractor = sum(count for count, kind in zip(counts, image_kinds) if kind == "irrelevant")
            per_image_tokens = json_dumps(counts)
            total_visual_tokens = total
            distractor_fraction = f"{(distractor / total):.8f}" if total else ""
        else:
            total_visual_tokens = int(sum(int(g[0] * g[1] * g[2]) for g in grids))

    device = next(backend.model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    start = time.perf_counter()
    try:
        with backend.torch.inference_mode():
            out = backend.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        elapsed = time.perf_counter() - start
        seq = out.sequences
        new_ids = seq[:, input_len:]
        raw = backend.processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        eos = backend.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        last = int(seq[0, -1].item())
        finish = "eos" if last in eos_ids else ("length" if new_ids.shape[1] >= max_new_tokens else "unknown")
        return {
            "raw": raw,
            "input_token_count": input_len,
            "estimated_visual_token_count": total_visual_tokens,
            "per_image_visual_token_counts": per_image_tokens,
            "distractor_visual_token_fraction": distractor_fraction,
            "output_token_count": int(new_ids.shape[1]),
            "finish_reason": finish,
            "generation_time_sec": f"{elapsed:.4f}",
        }
    finally:
        del inputs
        if backend.torch.cuda.is_available():
            backend.torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one E1-v2.1 full seed/shard with shard-safe CSV output.")
    parser.add_argument("--data_root", default="/home/songzhoujie/cvpr27/vstar/data/vstar_bench")
    parser.add_argument("--model", default="/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--crop_dir", default="outputs/crops")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=238)
    parser.add_argument("--settings_json", default="outputs/settings_smoke_v21.json")
    parser.add_argument("--seed_crop_root", default="outputs/crops_v21")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")

    settings = load_smoke_settings(Path(args.settings_json))
    min_pixels = int(settings["min_pixels"])
    max_pixels = int(settings["max_pixels"])
    crop_long_side = int(settings["crop_long_side"])
    max_new_tokens = int(settings["max_new_tokens"])
    attn = str(settings["attn_implementation"])

    output_csv = Path(args.output_csv) if args.output_csv else Path(f"outputs/v21_full_seed{args.seed}_shard{args.shard_id}.csv")
    completed = prepare_output(output_csv, args.overwrite, args.resume)

    print("Full-run fixed settings:")
    print(
        json.dumps(
            {
                "model": args.model,
                "dtype": settings.get("dtype"),
                "attn_implementation": attn,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "crop_long_side": crop_long_side,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "system_prompt": SYSTEM_PROMPT,
                "assistant_tool_template": ASSISTANT_TOOL_TEMPLATE,
                "final_user_prompt": FINAL_USER_PROMPT,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    backend = Backend(args.model, min_pixels, max_pixels, attn)
    samples = load_vstar_samples(args.data_root, num_samples=args.num_samples)
    shard_samples = [(idx, sample) for idx, sample in enumerate(samples) if idx % args.num_shards == args.shard_id]
    base_manifest = read_crop_manifest(Path(args.crop_dir) / "manifest_context_pollution.jsonl")
    max_k = max(CONDITION_TO_K.values())
    seed_crop_root = Path(args.seed_crop_root)

    full_hashes: Dict[str, str] = {}
    gt_keys: Dict[str, Tuple[str, int, int, str]] = {}

    for local_index, (sample_index, sample) in enumerate(shard_samples, start=1):
        if sample.sample_id not in base_manifest:
            raise RuntimeError(f"Missing base crop manifest row for {sample.sample_id}")
        base_info = base_manifest[sample.sample_id]
        gt_path = Path(base_info["gt_crop_path"])
        if not gt_path.exists():
            raise RuntimeError(f"Missing fixed GT crop: {gt_path}")

        with Image.open(gt_path) as gt_in:
            gt_img = resize_crop_long_side(gt_in.convert("RGB"), crop_long_side)
            gt_width, gt_height = gt_img.size
            gt_pixel_hash = sha256_image_pixels(gt_img)
        gt_key = (str(gt_path), gt_width, gt_height, gt_pixel_hash)
        if gt_keys.setdefault(sample.sample_id, gt_key) != gt_key:
            raise RuntimeError(f"GT crop consistency failed for {sample.sample_id}")

        with Image.open(sample.image_path) as orig_in:
            original = orig_in.convert("RGB")
            original_file_hash = sha256_file(sample.image_path)
            full_processed_hash, _ = backend.processed_hash(original)
            if full_hashes.setdefault(sample.sample_id, full_processed_hash) != full_processed_hash:
                raise RuntimeError(f"Full image processed hash changed for {sample.sample_id}")
            gt_bbox = clamp_bbox_xywh(base_info["gt_bbox"], original.width, original.height)
            irrelevant_items = seed_irrelevant_items(
                args.seed,
                sample,
                original,
                gt_bbox,
                base_info,
                seed_crop_root,
                max_k,
            )
            options, gt_letter, gt_text = deterministic_option_shuffle(sample.options, sample.sample_id, OPTION_SEED)

            for condition in CONDITIONS:
                key = (str(args.seed), sample.sample_id, condition)
                if key in completed:
                    continue

                k = CONDITION_TO_K[condition]
                crop_entries: List[Dict] = []
                gt_position = ""
                raw = ""
                pred = ""
                pred_text = ""
                valid = "False"
                correct = "False"
                status = "ok"
                error = ""
                result: Dict[str, object] = {}
                all_paths = [str(sample.image_path)]
                image_kinds = ["original"]
                image_sizes = [[original.width, original.height]]

                try:
                    crop_entries, gt_position = condition_crop_items(
                        args.seed,
                        sample.sample_id,
                        condition,
                        gt_path,
                        gt_bbox,
                        irrelevant_items,
                        crop_long_side,
                    )
                    messages, images, _ = build_messages(original, crop_entries, sample.question, options)
                    all_paths.extend(c["path"] for c in crop_entries)
                    image_kinds.extend(c["kind"] for c in crop_entries)
                    image_sizes.extend([[c["image"].width, c["image"].height] for c in crop_entries])
                    result = generate_with_diagnostics(backend, messages, images, image_kinds, max_new_tokens)
                    raw = str(result["raw"])
                    pred, pred_text = robust_parse(raw, options)
                    if pred:
                        valid = "True"
                        correct = str(pred == gt_letter)
                    else:
                        status = "invalid"
                except RuntimeError as exc:
                    error = str(exc)
                    if "out of memory" in error.lower() or ("cuda" in error.lower() and "memory" in error.lower()):
                        status = "oom"
                    else:
                        status = "invalid"
                    if backend.torch.cuda.is_available():
                        backend.torch.cuda.empty_cache()
                except Exception as exc:
                    status = "invalid"
                    error = str(exc)
                    if backend.torch.cuda.is_available():
                        backend.torch.cuda.empty_cache()

                row = {
                    "seed": args.seed,
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    "sample_index": sample_index,
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "condition": condition,
                    "k_irrelevant": k,
                    "status": status,
                    "original_image_path": str(sample.image_path),
                    "gt_crop_path": str(gt_path) if k >= 0 else "",
                    "irrelevant_crop_paths": json_dumps([c["path"] for c in crop_entries if c["kind"] == "irrelevant"]),
                    "all_image_paths_used": json_dumps(all_paths),
                    "image_kinds_in_order": json_dumps(image_kinds),
                    "image_sizes_in_order": json_dumps(image_sizes),
                    "gt_crop_position": gt_position,
                    "gt_crop_hash": gt_pixel_hash if k >= 0 else "",
                    "gt_crop_file_hash": sha256_file(gt_path) if k >= 0 else "",
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
                    "per_image_visual_token_counts": result.get("per_image_visual_token_counts", ""),
                    "distractor_visual_token_fraction": result.get("distractor_visual_token_fraction", ""),
                    "output_token_count": result.get("output_token_count", ""),
                    "finish_reason": result.get("finish_reason", ""),
                    "generation_time_sec": result.get("generation_time_sec", ""),
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "processor_min_pixels": min_pixels,
                    "processor_max_pixels": max_pixels,
                    "crop_long_side": crop_long_side,
                    "attn_implementation": backend.attn_implementation,
                    "dtype": backend.dtype,
                    "error_message": error,
                }
                append_row(output_csv, row)
                completed.add(key)
                del crop_entries
                if backend.torch.cuda.is_available():
                    backend.torch.cuda.empty_cache()

        print(
            f"[seed {args.seed} shard {args.shard_id}/{args.num_shards}] "
            f"[{local_index}/{len(shard_samples)}] {sample.sample_id}"
        )

    print(f"wrote shard CSV: {output_csv}")


if __name__ == "__main__":
    main()
