import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class VStarSample:
    sample_id: str
    category: str
    image_path: Path
    annotation_path: Path
    question: str
    options: List[str]
    target_objects: List[str]
    bboxes: List[List[float]]
    gt_bbox: List[float]


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sanitize_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", sample_id)


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image_for_annotation(annotation_path: Path) -> Optional[Path]:
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = annotation_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    lower_stem = annotation_path.stem.lower()
    for candidate in annotation_path.parent.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            if candidate.stem.lower() == lower_stem:
                return candidate
    return None


def clamp_bbox_xywh(bbox: Sequence[float], width: int, height: int) -> List[float]:
    x, y, w, h = [float(v) for v in bbox[:4]]
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + max(1.0, w)))
    y2 = max(0.0, min(float(height), y + max(1.0, h)))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return [x1, y1, x2 - x1, y2 - y1]


def union_bbox_xywh(bboxes: Sequence[Sequence[float]]) -> List[float]:
    if not bboxes:
        raise ValueError("No bbox values were found.")
    x1 = min(float(b[0]) for b in bboxes)
    y1 = min(float(b[1]) for b in bboxes)
    x2 = max(float(b[0]) + float(b[2]) for b in bboxes)
    y2 = max(float(b[1]) + float(b[3]) for b in bboxes)
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def bbox_xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox[:4]]
    return x, y, x + w, y + h


def bbox_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def load_vstar_samples(data_root: str, num_samples: Optional[int] = None) -> List[VStarSample]:
    root = Path(data_root).expanduser().resolve()
    if (root / "data" / "vstar_bench").exists():
        root = (root / "data" / "vstar_bench").resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    samples: List[VStarSample] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for annotation_path in sorted(category_dir.glob("*.json")):
            image_path = find_image_for_annotation(annotation_path)
            if image_path is None:
                continue
            annotation = load_json(annotation_path)
            question = annotation.get("question")
            options = annotation.get("options")
            bboxes = annotation.get("bbox")
            if not isinstance(question, str) or not isinstance(options, list) or not isinstance(bboxes, list):
                continue
            normalized_bboxes: List[List[float]] = []
            for bbox in bboxes:
                if isinstance(bbox, list) and len(bbox) >= 4:
                    normalized_bboxes.append([float(v) for v in bbox[:4]])
            if not normalized_bboxes or len(options) < 2:
                continue
            target_objects = annotation.get("target_object") or []
            if not isinstance(target_objects, list):
                target_objects = [str(target_objects)]
            sample_id = f"{category_dir.name}/{annotation_path.stem}"
            samples.append(
                VStarSample(
                    sample_id=sample_id,
                    category=category_dir.name,
                    image_path=image_path,
                    annotation_path=annotation_path,
                    question=question,
                    options=[str(option) for option in options],
                    target_objects=[str(obj) for obj in target_objects],
                    bboxes=normalized_bboxes,
                    gt_bbox=union_bbox_xywh(normalized_bboxes),
                )
            )
            if num_samples is not None and len(samples) >= num_samples:
                return samples
    return samples


def deterministic_option_shuffle(
    options: Sequence[str], sample_id: str, seed: int
) -> Tuple[List[str], str, str]:
    import random

    indexed = list(enumerate(str(option) for option in options))
    rng = random.Random(stable_seed(seed, sample_id, "option_shuffle"))
    rng.shuffle(indexed)
    shuffled = [option for _, option in indexed]
    gt_position = next(i for i, (original_idx, _) in enumerate(indexed) if original_idx == 0)
    return shuffled, LETTERS[gt_position], shuffled[gt_position]


def format_options(options: Sequence[str]) -> str:
    return "\n".join(f"({LETTERS[i]}) {option}" for i, option in enumerate(options))


def strict_prompt(question: str, options: Sequence[str]) -> str:
    return (
        "You are given the original image and several evidence crops. "
        "Some evidence crops may be irrelevant. Answer the question by selecting the correct option. "
        "Output only the option letter.\n\n"
        f"Question: {question}\n"
        f"Options:\n{format_options(options)}"
    )


def parse_answer_letter(text: str, valid_letters: Iterable[str]) -> str:
    if not text:
        return ""
    valid = "".join(valid_letters)
    pattern = rf"(?<![A-Za-z])([{re.escape(valid)}])(?![A-Za-z])"
    match = re.search(pattern, text.strip().upper())
    if match:
        return match.group(1)
    stripped = text.strip().upper()
    return stripped[0] if stripped[:1] in valid else ""


def assert_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output without --overwrite: {path}")


def read_crop_manifest(path: Path) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row
    return result


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
