import argparse
from collections import Counter
from pathlib import Path

from context_pollution_utils import load_vstar_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local V*Bench structure and detected fields.")
    parser.add_argument("--data_root", required=True, help="Path to the local V*Bench root.")
    parser.add_argument("--num_preview", type=int, default=3)
    args = parser.parse_args()

    samples = load_vstar_samples(args.data_root)
    if not samples:
        raise SystemExit(f"No V*Bench samples detected under {args.data_root}")

    counts = Counter(sample.category for sample in samples)
    root = Path(args.data_root).expanduser().resolve()

    print("Detected V*Bench layout")
    print(f"data_root: {root}")
    print(f"num_samples: {len(samples)}")
    print(f"categories: {dict(sorted(counts.items()))}")
    print()
    print("Detected fields")
    print("image_path: paired image file with the same stem as each annotation JSON")
    print("annotation/question JSON: <category>/<sample_id>.json")
    print("question field: question")
    print("options field: options")
    print("correct answer: options[0] before deterministic shuffling")
    print("target bbox field: bbox, interpreted as [x, y, width, height]")
    print("target object field: target_object")
    print()
    print("Preview")
    for sample in samples[: args.num_preview]:
        print(f"- sample_id: {sample.sample_id}")
        print(f"  image_path: {sample.image_path.relative_to(root)}")
        print(f"  annotation_path: {sample.annotation_path.relative_to(root)}")
        print(f"  question: {sample.question}")
        print(f"  options: {sample.options}")
        print(f"  gt_bbox: {sample.gt_bbox}")
        print(f"  target_objects: {sample.target_objects}")


if __name__ == "__main__":
    main()
