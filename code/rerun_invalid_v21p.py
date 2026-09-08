import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

from context_pollution_utils import clamp_bbox_xywh, deterministic_option_shuffle, format_options, json_dumps, load_vstar_samples, read_crop_manifest
from run_context_pollution_full_v21 import (
    CONDITIONS,
    CONDITION_TO_K,
    FIELDNAMES,
    OPTION_SEED,
    condition_crop_items,
    generate_with_diagnostics,
    seed_irrelevant_items,
)
from run_context_pollution_smoke_v21 import Backend, build_messages, robust_parse, sha256_file, sha256_image_pixels, resize_crop_long_side


def load_smoke_settings(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        settings = json.load(f)
    print("Frozen smoke settings:")
    print(json.dumps(settings, ensure_ascii=False, indent=2))
    return settings


def rerun_row(
    source_row: pd.Series,
    sample,
    base_info: Dict,
    backend: Backend,
    args,
    settings: Dict,
) -> Dict[str, object]:
    seed = int(source_row["seed"])
    condition = str(source_row["condition"])
    k = CONDITION_TO_K[condition]
    min_pixels = int(settings["min_pixels"])
    max_pixels = int(settings["max_pixels"])
    crop_long_side = int(settings["crop_long_side"])
    max_new_tokens = int(settings["max_new_tokens"])

    gt_path = Path(base_info["gt_crop_path"])
    with Image.open(gt_path) as gt_in:
        gt_img = resize_crop_long_side(gt_in.convert("RGB"), crop_long_side)
        gt_width, gt_height = gt_img.size
        gt_pixel_hash = sha256_image_pixels(gt_img)

    with Image.open(sample.image_path) as orig_in:
        original = orig_in.convert("RGB")
        original_file_hash = sha256_file(sample.image_path)
        full_processed_hash, _ = backend.processed_hash(original)
        gt_bbox = clamp_bbox_xywh(base_info["gt_bbox"], original.width, original.height)
        irrelevant_items = seed_irrelevant_items(
            seed,
            sample,
            original,
            gt_bbox,
            base_info,
            Path(args.seed_crop_root),
            max(CONDITION_TO_K.values()),
        )
        options, gt_letter, gt_text = deterministic_option_shuffle(sample.options, sample.sample_id, OPTION_SEED)

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
                seed,
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

    return {
        "seed": seed,
        "shard_id": source_row.get("shard_id", ""),
        "num_shards": source_row.get("num_shards", ""),
        "sample_index": source_row.get("sample_index", ""),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch E1-v2.1 by rerunning only invalid rows with min-28 edge padding.")
    parser.add_argument("--input_csv", default="outputs/results_context_pollution_full_v21_all_seeds.csv")
    parser.add_argument("--output_csv", default="outputs/results_context_pollution_full_v21_all_seeds.csv")
    parser.add_argument("--rerun_csv", default="outputs/rerun_invalid_rows_v21p.csv")
    parser.add_argument("--backup_csv", default="outputs/results_context_pollution_full_v21_all_seeds_before_v21p.csv")
    parser.add_argument("--data_root", default="/home/songzhoujie/cvpr27/vstar/data/vstar_bench")
    parser.add_argument("--model", default="/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--crop_dir", default="outputs/crops")
    parser.add_argument("--seed_crop_root", default="outputs/crops_v21")
    parser.add_argument("--settings_json", default="outputs/settings_smoke_v21.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    rerun_csv = Path(args.rerun_csv)
    backup_csv = Path(args.backup_csv)
    if output_csv.exists() and output_csv != input_csv and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {output_csv}")
    if rerun_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {rerun_csv}")

    settings = load_smoke_settings(Path(args.settings_json))
    backend = Backend(args.model, int(settings["min_pixels"]), int(settings["max_pixels"]), str(settings["attn_implementation"]))
    samples = {sample.sample_id: sample for sample in load_vstar_samples(args.data_root, num_samples=None)}
    manifest = read_crop_manifest(Path(args.crop_dir) / "manifest_context_pollution.jsonl")

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    invalid = df[df["status"].eq("invalid")].copy()
    print(f"invalid rows to rerun: {len(invalid)}")
    replacement_rows = []
    for index, row in invalid.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in samples:
            raise RuntimeError(f"Unknown sample_id: {sample_id}")
        if sample_id not in manifest:
            raise RuntimeError(f"Missing crop manifest row: {sample_id}")
        replacement = rerun_row(row, samples[sample_id], manifest[sample_id], backend, args, settings)
        replacement_rows.append(replacement)
        print(
            f"[{len(replacement_rows)}/{len(invalid)}] seed={replacement['seed']} "
            f"{sample_id} {replacement['condition']} -> {replacement['status']} {replacement['pred_answer_letter']}"
        )

    rep = pd.DataFrame(replacement_rows)
    rerun_csv.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(rerun_csv, index=False)

    patched = df.copy()
    for row in replacement_rows:
        mask = (
            patched["seed"].astype(str).eq(str(row["seed"]))
            & patched["sample_id"].eq(str(row["sample_id"]))
            & patched["condition"].eq(str(row["condition"]))
        )
        if int(mask.sum()) != 1:
            raise RuntimeError(f"Expected exactly one row to replace for {row['seed']} {row['sample_id']} {row['condition']}")
        for field in FIELDNAMES:
            patched.loc[mask, field] = row.get(field, "")

    duplicated = patched.duplicated(["seed", "sample_id", "condition"], keep=False)
    if duplicated.any():
        raise RuntimeError("Duplicate seed/sample/condition rows remain after patch.")
    if len(patched) != 3570:
        raise RuntimeError(f"Expected 3570 rows after patch, found {len(patched)}")

    if input_csv.resolve() == output_csv.resolve() and not backup_csv.exists():
        shutil.copy2(input_csv, backup_csv)
        print(f"wrote backup: {backup_csv}")

    patched = patched[FIELDNAMES]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    patched.to_csv(output_csv, index=False)
    print(f"wrote patched CSV: {output_csv}")
    print(f"rows: {len(patched)}")
    print(f"status counts: {patched['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
