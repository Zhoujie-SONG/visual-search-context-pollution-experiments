import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw

from context_pollution_utils import (
    LETTERS,
    assert_can_write,
    deterministic_option_shuffle,
    json_dumps,
    load_vstar_samples,
    parse_answer_letter,
    read_crop_manifest,
    sanitize_id,
    stable_seed,
    strict_prompt,
)


CONDITION_ORIGINAL = "original_only"
CONDITION_GT = "gt_crop_only"


def resolve_model_path(model: str) -> str:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path)
    local_qwen = Path("/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct")
    if model == "Qwen/Qwen2.5-VL-7B-Instruct" and local_qwen.exists():
        return str(local_qwen)
    return model


class QwenVLBackend:
    def __init__(self, model: str):
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForImageTextToText as ModelClass
        except Exception as exc:
            raise RuntimeError(
                "Qwen2.5-VL inference dependencies are missing. Install torch, transformers, "
                "accelerate, and qwen-vl-utils in the active environment."
            ) from exc

        self.torch = torch
        self.model_name_or_path = resolve_model_path(model)
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model = ModelClass.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = ModelClass.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        self.model.eval()

    def generate(self, image_paths: Sequence[str], prompt: str, max_new_tokens: int = 8) -> str:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated = generated[:, input_len:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def fit_image(image: Image.Image, max_size: tuple) -> Image.Image:
    copied = image.copy()
    copied.thumbnail(max_size, Image.Resampling.LANCZOS)
    return copied


def labeled_panel(image: Image.Image, label: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(40, 40, 40), width=2)
    draw.rectangle((0, 0, width, 30), fill=(245, 245, 245))
    draw.text((8, 8), label, fill=(0, 0, 0))
    fitted = fit_image(image, (width - 16, height - 42))
    panel.paste(fitted, ((width - fitted.width) // 2, 36 + (height - 42 - fitted.height) // 2))
    return panel


def create_contact_sheet(
    original_path: str,
    evidence_paths: Sequence[str],
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(original_path) as original_in:
        original = original_in.convert("RGB")
    evidence_images = []
    for path in evidence_paths:
        with Image.open(path) as image_in:
            evidence_images.append(image_in.convert("RGB"))

    if not evidence_images:
        sheet = labeled_panel(original, "Original image", 1100, 760)
        sheet.save(output_path)
        return

    original_panel = labeled_panel(original, "Original image", 820, 760)
    tile_w, tile_h = 260, 220
    cols = 2 if len(evidence_images) <= 4 else 3
    rows = (len(evidence_images) + cols - 1) // cols
    evidence_area = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for index, image in enumerate(evidence_images):
        panel = labeled_panel(image, f"Evidence-{index + 1}", tile_w, tile_h)
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        evidence_area.paste(panel, (x, y))

    sheet_h = max(original_panel.height, evidence_area.height)
    sheet = Image.new("RGB", (original_panel.width + evidence_area.width + 20, sheet_h), "white")
    sheet.paste(original_panel, (0, 0))
    sheet.paste(evidence_area, (original_panel.width + 20, 0))
    sheet.save(output_path)


def evidence_order(sample_id: str, condition: str, gt_path: str, irrelevant_paths: Sequence[str], seed: int):
    items = [{"kind": "gt", "path": gt_path}] + [
        {"kind": "irrelevant", "path": path} for path in irrelevant_paths
    ]
    rng = random.Random(stable_seed(seed, sample_id, condition, "evidence_order"))
    rng.shuffle(items)
    gt_position = next((index + 1 for index, item in enumerate(items) if item["kind"] == "gt"), "")
    return items, gt_position


def write_header_if_needed(path: Path, fieldnames: Sequence[str], overwrite: bool) -> None:
    assert_can_write(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def load_completed_rows(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            (row.get("sample_id", ""), row.get("condition", ""))
            for row in csv.DictReader(f)
            if row.get("sample_id") and row.get("condition")
        }


def append_row(path: Path, fieldnames: Sequence[str], row: Dict) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context pollution evaluation with Qwen2.5-VL.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--crop_dir", default="outputs/crops")
    parser.add_argument("--output_csv", default="outputs/results_context_pollution.csv")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--k_values", nargs="+", type=int, default=[0, 1, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_mode", choices=["contact_sheet", "multi_image"], default="contact_sheet")
    parser.add_argument("--contact_sheet_dir", default="outputs/contact_sheets")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true", help="Create rows/contact sheets without loading the model.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append missing sample-condition rows to an existing CSV.")
    args = parser.parse_args()

    samples = load_vstar_samples(args.data_root, num_samples=args.num_samples)
    manifest = read_crop_manifest(Path(args.crop_dir) / "manifest_context_pollution.jsonl")
    output_csv = Path(args.output_csv)
    fieldnames = [
        "sample_id",
        "condition",
        "k_irrelevant",
        "image_path",
        "contact_sheet_path",
        "gt_bbox",
        "gt_crop_position",
        "question",
        "options_shuffled",
        "gt_answer_letter",
        "gt_answer_text",
        "model_raw_output",
        "pred_answer_letter",
        "pred_answer_text",
        "correct",
        "num_evidence_crops",
        "seed",
        "inference_time_sec",
        "error",
    ]
    completed = set()
    if args.resume and output_csv.exists():
        completed = load_completed_rows(output_csv)
        print(f"[resume] found {len(completed)} completed sample-condition rows in {output_csv}")
    else:
        write_header_if_needed(output_csv, fieldnames, args.overwrite)

    backend = None
    backend_error = ""
    if args.dry_run:
        backend_error = "dry_run_no_model_inference"
    else:
        try:
            backend = QwenVLBackend(args.model)
        except Exception as exc:
            backend_error = str(exc)
            print(f"[WARN] Qwen backend unavailable; rows will be marked failed: {backend_error}")
    contact_dir = Path(args.contact_sheet_dir)

    for sample_index, sample in enumerate(samples):
        if sample.sample_id not in manifest:
            print(f"[WARN] missing crop manifest for {sample.sample_id}; skipping")
            continue
        crop_info = manifest[sample.sample_id]
        shuffled_options, gt_letter, gt_text = deterministic_option_shuffle(
            sample.options, sample.sample_id, args.seed
        )
        prompt = strict_prompt(sample.question, shuffled_options)
        conditions = [(CONDITION_ORIGINAL, 0)]
        for k in args.k_values:
            if k == 0:
                conditions.append((CONDITION_GT, 0))
            else:
                conditions.append((f"gt_plus_{k}_irrelevant", k))

        for condition, k in conditions:
            if (sample.sample_id, condition) in completed:
                continue

            raw_output = ""
            pred_letter = ""
            pred_text = ""
            correct = ""
            error = ""
            elapsed = ""
            gt_position = ""
            evidence_paths: List[str] = []

            try:
                if condition != CONDITION_ORIGINAL:
                    available = crop_info["irrelevant_crops"]
                    if len(available) < k:
                        raise RuntimeError(f"Only {len(available)} irrelevant crops available, need {k}.")
                    ordered, gt_position = evidence_order(
                        sample.sample_id,
                        condition,
                        crop_info["gt_crop_path"],
                        [item["path"] for item in available[:k]],
                        args.seed,
                    )
                    evidence_paths = [item["path"] for item in ordered]

                sheet_path = contact_dir / sanitize_id(sample.sample_id) / f"{condition}.jpg"
                create_contact_sheet(str(sample.image_path), evidence_paths, sheet_path, args.overwrite)
                image_inputs = (
                    [str(sheet_path)]
                    if args.input_mode == "contact_sheet"
                    else [str(sample.image_path)] + evidence_paths
                )

                start = time.perf_counter()
                if backend is None:
                    error = backend_error or "model_backend_unavailable"
                    raw_output = ""
                else:
                    raw_output = backend.generate(image_inputs, prompt, max_new_tokens=args.max_new_tokens)
                elapsed = f"{time.perf_counter() - start:.4f}"
                pred_letter = parse_answer_letter(raw_output, LETTERS[: len(shuffled_options)])
                if pred_letter:
                    pred_text = shuffled_options[LETTERS.index(pred_letter)]
                    correct = str(pred_letter == gt_letter)
                else:
                    correct = "False"
            except Exception as exc:
                error = str(exc)
                correct = "False"
                sheet_path = contact_dir / sanitize_id(sample.sample_id) / f"{condition}.jpg"
                print(f"[WARN] {sample.sample_id} {condition}: {exc}")

            row = {
                "sample_id": sample.sample_id,
                "condition": condition,
                "k_irrelevant": k,
                "image_path": str(sample.image_path),
                "contact_sheet_path": str(sheet_path),
                "gt_bbox": json_dumps(crop_info.get("gt_bbox", sample.gt_bbox)),
                "gt_crop_position": gt_position,
                "question": sample.question,
                "options_shuffled": json_dumps(shuffled_options),
                "gt_answer_letter": gt_letter,
                "gt_answer_text": gt_text,
                "model_raw_output": raw_output,
                "pred_answer_letter": pred_letter,
                "pred_answer_text": pred_text,
                "correct": correct,
                "num_evidence_crops": len(evidence_paths),
                "seed": args.seed,
                "inference_time_sec": elapsed,
                "error": error,
            }
            append_row(output_csv, fieldnames, row)
        print(f"[{sample_index + 1}/{len(samples)}] evaluated {sample.sample_id}")

    print(f"wrote results: {output_csv}")


if __name__ == "__main__":
    main()
