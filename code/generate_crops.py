import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence

from PIL import Image

from context_pollution_utils import (
    assert_can_write,
    bbox_iou_xywh,
    clamp_bbox_xywh,
    json_dumps,
    load_vstar_samples,
    read_crop_manifest,
    sanitize_id,
    stable_seed,
)


def crop_box_tuple(bbox: Sequence[float]) -> tuple:
    x, y, w, h = bbox
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


def sample_irrelevant_bbox(
    rng: random.Random,
    gt_bbox: Sequence[float],
    image_width: int,
    image_height: int,
    max_attempts: int = 3000,
) -> List[float]:
    gt_area = max(1.0, float(gt_bbox[2]) * float(gt_bbox[3]))
    gt_aspect = max(0.05, float(gt_bbox[2]) / max(1.0, float(gt_bbox[3])))

    for _ in range(max_attempts):
        area = gt_area * rng.uniform(0.5, 2.0)
        aspect = gt_aspect * rng.uniform(0.6, 1.67)
        width = max(8.0, min(float(image_width), (area * aspect) ** 0.5))
        height = max(8.0, min(float(image_height), area / width))
        if width >= image_width:
            x = 0.0
        else:
            x = rng.uniform(0.0, image_width - width)
        if height >= image_height:
            y = 0.0
        else:
            y = rng.uniform(0.0, image_height - height)
        candidate = clamp_bbox_xywh([x, y, width, height], image_width, image_height)
        candidate_area = candidate[2] * candidate[3]
        if not (0.5 * gt_area <= candidate_area <= 2.0 * gt_area):
            continue
        if bbox_iou_xywh(candidate, gt_bbox) < 0.1:
            return candidate

    raise RuntimeError("Could not sample an irrelevant crop with IoU < 0.1 and matched area.")


def save_crop(image: Image.Image, bbox: Sequence[float], path: Path, overwrite: bool) -> None:
    assert_can_write(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box_tuple(bbox)).save(path)


def row_complete(row: dict, max_k: int) -> bool:
    if not row:
        return False
    gt_path = Path(row.get("gt_crop_path", ""))
    if not gt_path.exists():
        return False
    irrelevant = row.get("irrelevant_crops") or []
    if len(irrelevant) < max_k:
        return False
    return all(Path(item.get("path", "")).exists() for item in irrelevant[:max_k])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GT and irrelevant crops for context pollution.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="outputs/crops")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--k_values", nargs="+", type=int, default=[0, 1, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse complete existing crop rows and generate only missing rows.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest_context_pollution.jsonl"
    if not args.resume:
        assert_can_write(manifest_path, args.overwrite)

    samples = load_vstar_samples(args.data_root, num_samples=args.num_samples)
    if not samples:
        raise SystemExit("No valid samples found.")

    max_k = max(args.k_values) if args.k_values else 0
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = read_crop_manifest(manifest_path) if args.resume and manifest_path.exists() else {}

    rows = []
    failures = []
    for index, sample in enumerate(samples):
        existing_row = existing.get(sample.sample_id)
        if args.resume and row_complete(existing_row, max_k):
            rows.append(existing_row)
            print(f"[{index + 1}/{len(samples)}] reused crops for {sample.sample_id}")
            continue

        sample_dir = output_dir / sanitize_id(sample.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(sample.image_path) as image_in:
                image = image_in.convert("RGB")
                width, height = image.size
                gt_bbox = clamp_bbox_xywh(sample.gt_bbox, width, height)
                gt_crop_path = sample_dir / "gt.png"
                save_crop(image, gt_bbox, gt_crop_path, args.overwrite or args.resume)

                rng = random.Random(stable_seed(args.seed, sample.sample_id, "irrelevant_crops"))
                irrelevant = []
                for crop_index in range(max_k):
                    crop_bbox = sample_irrelevant_bbox(rng, gt_bbox, width, height)
                    crop_path = sample_dir / f"irrelevant_{crop_index:02d}.png"
                    save_crop(image, crop_bbox, crop_path, args.overwrite or args.resume)
                    irrelevant.append(
                        {
                            "path": str(crop_path),
                            "bbox": crop_bbox,
                            "iou_with_gt": bbox_iou_xywh(crop_bbox, gt_bbox),
                        }
                    )

            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "image_path": str(sample.image_path),
                    "annotation_path": str(sample.annotation_path),
                    "category": sample.category,
                    "question": sample.question,
                    "options": sample.options,
                    "gt_answer_text": sample.options[0],
                    "gt_bbox": gt_bbox,
                    "gt_crop_path": str(gt_crop_path),
                    "irrelevant_crops": irrelevant,
                    "seed": args.seed,
                }
            )
            print(f"[{index + 1}/{len(samples)}] generated crops for {sample.sample_id}")
        except Exception as exc:
            failures.append({"sample_id": sample.sample_id, "error": str(exc)})
            print(f"[WARN] failed {sample.sample_id}: {exc}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")

    print(f"wrote manifest: {manifest_path}")
    print(f"processed: {len(rows)}")
    print(f"failed: {len(failures)}")
    if failures:
        failure_path = output_dir / "crop_generation_failures.json"
        assert_can_write(failure_path, args.overwrite)
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"failure details: {failure_path}")


if __name__ == "__main__":
    main()
