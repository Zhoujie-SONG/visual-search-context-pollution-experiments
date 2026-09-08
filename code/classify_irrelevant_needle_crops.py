#!/usr/bin/env python3
"""Classify geometrically irrelevant VisualNeedle crops with a local VLM judge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


SYSTEM_PROMPT = """你是视觉搜索轨迹的分析员。一个视觉搜索 agent 在回答关于图像的问题时,会多轮裁剪图像局部作为观察。现在给你一次裁剪,它与最终目标区域几何上基本不重叠。你的任务是判断这次裁剪属于哪一类,并只输出 JSON。

四个类别的定义:

- MISLEADING(误导性):crop 中包含与目标同类别的其他实例,或包含部分满足问题约束的内容(例如问题找\"穿白短袖的男人\",crop 里有穿白衣的女人、或不穿白衣的男人)。判断标准:一个只看这张 crop 的人,是否可能把其中内容误认为目标、或据此写下错误的属性/答案。
- EXPLORATION(良性探索):合理的搜索步骤。crop 对应问题中提到的参照物、目标可能出现的候选区域、或系统性扫图的一步;内容与目标不相似,不会被误认,其价值是\"排除该区域\"或\"提供定位线索\"。
- REDUNDANT(冗余):与之前某次裁剪覆盖高度重叠的区域,或几乎不含可辨识物体的无信息区域(纯背景、天空、模糊纹理)。
- NEAR_MISS(近失):crop 落在目标紧邻位置或裁到了目标的一小部分,但截掉了回答问题所需的关键属性(问耳机颜色但耳机被裁出画面)。

判定优先级:先判 REDUNDANT(看历史裁剪列表),再判 NEAR_MISS(看与目标框的位置关系),再区分 MISLEADING 与 EXPLORATION(看内容是否可能被误认为目标)。只能选一类。

只输出一个 JSON 对象,不要输出任何其他文字。"""

OUTPUT_SCHEMA = """{
  \"type\": \"MISLEADING | EXPLORATION | REDUNDANT | NEAR_MISS\",
  \"contains_same_category_instance\": true/false,
  \"matched_constraints\": [\"crop 内容满足了问题中的哪些约束,如 '白色衣物'、'男性',没有则空数组\"],
  \"could_yield_wrong_answer\": true/false,
  \"reason\": \"一句话,不超过 40 字\"
}"""

VALID_TYPES = {"MISLEADING", "EXPLORATION", "REDUNDANT", "NEAR_MISS"}
TARGET_DESCRIPTIONS = {
    "TPROMPTa6afff2f9bd6": "图中的吉他",
    "TPROMPT98301a2c2575": "图中的打火机",
    "TPROMPT8f6f95bc1628": "双手拿水杯的人偶的手",
    "TPROMPT31a0621e1bbc": "蜡笔小新玩具左侧的盒子",
    "TPROMPT3e12bdceaf5f": "白色鬃毛马背上承载的物体",
    "TPROMPT937d576541c5": "最接近心形的盒子右上方物体",
    "TPROMPTe31648324dbb": "章鱼哥头上的物体",
    "TPROMPT98d34d3f2f7e": "图中的心形物体",
    "TPROMPT6064c9a5c9cc": "文字 bo 下方的数字",
    "TPROMPT2d3a530d79de": "咖啡右侧的文字",
    "TPROMPT776fb7077320": "CALL 后面的文字内容",
    "TPROMPT00e974d39841": "玻璃栏杆上的文字",
    "TPROMPTf33ce6bcff4e": "红裙女人与右侧大树之间被遮挡的物体",
    "TPROMPTd8cea30a3f40": "右侧建筑左边被植物遮挡的白色物体",
    "TPROMPTcb887313e74d": "最高植物左后方的浅绿色物体",
    "TPROMPT1c4e65857519": "窗外高楼旁边的塔式起重机",
    "TPROMPT42a1f591e921": "深色长裙女人携带的包",
    "TPROMPTf580f92c43df": "穿绿衣者头部右上方的文字",
    "TPROMPT727f619343da": "水杯后面的物品",
    "TPROMPT6a40e4600d95": "黄瓜左侧第二种零食",
}
RESULT_FIELDS = [
    "sample_id",
    "category",
    "question",
    "gt_answer",
    "target_description",
    "turn_idx",
    "total_turns",
    "prev_bboxes_list",
    "crop_bbox",
    "gt_bbox",
    "target_coverage",
    "crop_gt_iou",
    "max_iou_with_previous",
    "crop_path",
    "overlay_path",
    "type",
    "contains_same_category_instance",
    "matched_constraints",
    "could_yield_wrong_answer",
    "reason",
    "parse_status",
    "attempts",
    "input_token_count",
    "output_token_count",
    "raw_output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        default="/mnt/data2/szj/experiments/e2m_needle_smoke/trajectories.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data2/szj/experiments/e2m_needle_smoke/irrelevant_crop_classification",
    )
    parser.add_argument(
        "--model",
        default="/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.10)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bbox_intersection(a: Sequence[float], b: Sequence[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = bbox_intersection(a, b)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def target_coverage(crop: Sequence[float], target: Sequence[float]) -> float:
    area = bbox_area(target)
    return bbox_intersection(crop, target) / area if area > 0 else 0.0


def resize_long_side(image: Image.Image, long_side: int = 1024) -> Image.Image:
    width, height = image.size
    scale = min(1.0, long_side / max(width, height))
    if scale == 1.0:
        return image.copy()
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)


def edge_pad(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Replicate edge pixels so tiny/extreme crops satisfy Qwen's image constraints."""
    image = image.convert("RGB")
    width, height = image.size
    left = (target_width - width) // 2
    right = target_width - width - left
    horizontal = Image.new("RGB", (target_width, height))
    horizontal.paste(image, (left, 0))
    if left:
        horizontal.paste(
            image.crop((0, 0, 1, height)).resize((left, height), Image.Resampling.NEAREST),
            (0, 0),
        )
    if right:
        horizontal.paste(
            image.crop((width - 1, 0, width, height)).resize(
                (right, height), Image.Resampling.NEAREST
            ),
            (left + width, 0),
        )
    top = (target_height - height) // 2
    bottom = target_height - height - top
    result = Image.new("RGB", (target_width, target_height))
    result.paste(horizontal, (0, top))
    if top:
        result.paste(
            horizontal.crop((0, 0, target_width, 1)).resize(
                (target_width, top), Image.Resampling.NEAREST
            ),
            (0, 0),
        )
    if bottom:
        result.paste(
            horizontal.crop((0, height - 1, target_width, height)).resize(
                (target_width, bottom), Image.Resampling.NEAREST
            ),
            (0, top + height),
        )
    return result


def processor_safe_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    target_width = max(width, 28, math.ceil(height / 199))
    target_height = max(height, 28, math.ceil(width / 199))
    if (target_width, target_height) == (width, height):
        return image.convert("RGB")
    return edge_pad(image, target_width, target_height)


def draw_overlay(
    image_path: Path,
    gt_bbox: Sequence[float],
    crop_bbox: Sequence[float],
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    width, height = original.size
    overlay = resize_long_side(original)
    sx, sy = overlay.width / width, overlay.height / height
    line_width = max(4, round(max(overlay.size) / 180))
    draw = ImageDraw.Draw(overlay)

    def scaled(bbox: Sequence[float]) -> Tuple[int, int, int, int]:
        return (
            round(bbox[0] * sx),
            round(bbox[1] * sy),
            round(bbox[2] * sx),
            round(bbox[3] * sy),
        )

    draw.rectangle(scaled(crop_bbox), outline=(255, 40, 40), width=line_width)
    # Draw GT last so a tiny target remains visible when boxes touch.
    draw.rectangle(scaled(gt_bbox), outline=(20, 230, 70), width=line_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path, quality=95)


def make_manifest(args: argparse.Namespace) -> List[Dict[str, Any]]:
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "irrelevant_crop_manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        return read_jsonl(manifest_path)

    records: List[Dict[str, Any]] = []
    for trajectory in read_jsonl(Path(args.trajectories)):
        episode = trajectory["episode"]
        gt_bbox = [float(value) for value in trajectory["gt_bbox_xyxy"]]
        crops = trajectory.get("crops", [])
        previous: List[List[float]] = []
        for crop in crops:
            crop_bbox = [float(value) for value in crop["bbox_original"]]
            coverage = target_coverage(crop_bbox, gt_bbox)
            if coverage < args.coverage_threshold:
                turn_idx = int(crop["turn_id"]) + 1
                overlay_path = output_dir / "overlays" / episode["sample_id"] / f"turn_{turn_idx:02d}.jpg"
                draw_overlay(
                    Path(episode["original_image_path"]),
                    gt_bbox,
                    crop_bbox,
                    overlay_path,
                )
                records.append(
                    {
                        "sample_id": episode["sample_id"],
                        "category": episode["category"],
                        "question": episode["question"],
                        "gt_answer": episode["gt_answer"],
                        "target_description": TARGET_DESCRIPTIONS.get(
                            episode["sample_id"], episode["category"]
                        ),
                        "turn_idx": turn_idx,
                        "total_turns": len(crops),
                        "prev_bboxes_list": [list(bbox) for bbox in previous],
                        "crop_bbox": crop_bbox,
                        "gt_bbox": gt_bbox,
                        "target_coverage": coverage,
                        "crop_gt_iou": bbox_iou(crop_bbox, gt_bbox),
                        "max_iou_with_previous": max(
                            (bbox_iou(crop_bbox, bbox) for bbox in previous), default=0.0
                        ),
                        "crop_path": crop["crop_path"],
                        "overlay_path": str(overlay_path),
                    }
                )
            previous.append(crop_bbox)

    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def user_prompt(record: Dict[str, Any]) -> str:
    return f"""问题:{record['question']}
正确答案:{record['gt_answer']}
目标描述:{record['target_description']}
当前是第 {record['turn_idx']} 次裁剪(共 {record['total_turns']} 次)
之前各次裁剪的框(原图坐标):{json.dumps(record['prev_bboxes_list'], ensure_ascii=False)}
本次裁剪的框:{json.dumps(record['crop_bbox'], ensure_ascii=False)}

[图1] 原图(低分辨率)。绿框 = 最终目标 GT;红框 = 本次裁剪位置。
[图2] 本次裁剪的高清内容。

输出格式:
{OUTPUT_SCHEMA}"""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE))
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_result(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not value or value.get("type") not in VALID_TYPES:
        return None
    if not isinstance(value.get("contains_same_category_instance"), bool):
        return None
    if not isinstance(value.get("matched_constraints"), list):
        return None
    if not all(isinstance(item, str) for item in value["matched_constraints"]):
        return None
    if not isinstance(value.get("could_yield_wrong_answer"), bool):
        return None
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        return None
    return {
        "type": value["type"],
        "contains_same_category_instance": value["contains_same_category_instance"],
        "matched_constraints": value["matched_constraints"],
        "could_yield_wrong_answer": value["could_yield_wrong_answer"],
        "reason": value["reason"].strip(),
    }


class Judge:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            min_pixels=28 * 28 * 16,
            max_pixels=1280 * 28 * 28,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(
        self,
        record: Dict[str, Any],
        max_new_tokens: int,
        correction_text: str = "",
    ) -> Dict[str, Any]:
        with Image.open(record["overlay_path"]) as image:
            overlay = processor_safe_image(image)
        with Image.open(record["crop_path"]) as image:
            crop = processor_safe_image(image)
        prompt = user_prompt(record)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.split("[图1]")[0] + "[图1] 原图(低分辨率)。绿框 = 最终目标 GT;红框 = 本次裁剪位置。"},
                    {"type": "image", "image": overlay},
                    {"type": "text", "text": "[图2] 本次裁剪的高清内容。"},
                    {"type": "image", "image": crop},
                    {
                        "type": "text",
                        "text": "输出格式:\n" + OUTPUT_SCHEMA + correction_text,
                    },
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[overlay, crop], padding=True, return_tensors="pt")
        input_length = int(inputs["input_ids"].shape[1])
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        new_ids = output.sequences[:, input_length:]
        raw = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return {
            "raw_output": raw,
            "input_token_count": input_length,
            "output_token_count": int(new_ids.shape[1]),
        }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_shard(args: argparse.Namespace, manifest: List[Dict[str, Any]]) -> None:
    if args.shard_id is None or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    output_path = Path(args.output_dir) / "shards" / f"shard_{args.shard_id}.jsonl"
    existing = {}
    if output_path.exists() and not args.overwrite:
        for row in read_jsonl(output_path):
            existing[(row["sample_id"], row["turn_idx"])] = row
    shard_records = [record for index, record in enumerate(manifest) if index % args.num_shards == args.shard_id]
    if len(existing) == len(shard_records):
        print(f"shard {args.shard_id}: already complete ({len(existing)})", flush=True)
        return

    judge = Judge(args.model)
    results = list(existing.values())
    for record in shard_records:
        key = (record["sample_id"], record["turn_idx"])
        if key in existing:
            continue
        attempts = []
        generated = judge.generate(record, args.max_new_tokens)
        attempts.append(generated)
        parsed = validate_result(extract_json_object(generated["raw_output"]))
        correction_text = ""
        if parsed is None:
            correction_text = (
                "\n\n上一次输出格式不合规。严格只输出一个满足上述字段和类型要求的 JSON 对象，"
                "不要输出思考过程或 Markdown。"
            )
        elif parsed["type"] == "REDUNDANT" and record["max_iou_with_previous"] < 0.98:
            correction_text = (
                "\n\n重要复核：本次框与历史框的最大 IoU 仅为 "
                f"{record['max_iou_with_previous']:.3f}，不属于历史区域高度重复。"
                "请重新仔细查看图2：只要其中有清晰可辨识的物体，就不能以‘无信息区域’判为 REDUNDANT；"
                "此时应按定义在 NEAR_MISS、MISLEADING、EXPLORATION 中选择。"
                "只有图2确实几乎是纯背景、天空或模糊纹理时，才保留 REDUNDANT。"
                "严格只输出修正后的 JSON。"
            )
        elif parsed["type"] != "REDUNDANT" and record["max_iou_with_previous"] >= 0.98:
            correction_text = (
                "\n\n重要复核：本次框与历史框的最大 IoU 不低于 0.98。"
                "按题目给定优先级，高度重复必须优先判为 REDUNDANT。严格只输出修正后的 JSON。"
            )
        if correction_text:
            generated = judge.generate(record, args.max_new_tokens, correction_text=correction_text)
            attempts.append(generated)
            parsed = validate_result(extract_json_object(generated["raw_output"]))
        result = dict(record)
        if parsed is None:
            result.update(
                {
                    "type": "",
                    "contains_same_category_instance": None,
                    "matched_constraints": [],
                    "could_yield_wrong_answer": None,
                    "reason": "",
                    "parse_status": "invalid",
                }
            )
        else:
            result.update(parsed)
            result["parse_status"] = "ok"
        result.update(
            {
                "attempts": len(attempts),
                "input_token_count": attempts[-1]["input_token_count"],
                "output_token_count": attempts[-1]["output_token_count"],
                "raw_output": attempts[-1]["raw_output"],
                "raw_attempts": [attempt["raw_output"] for attempt in attempts],
            }
        )
        results.append(result)
        write_jsonl(output_path, sorted(results, key=lambda row: (row["sample_id"], row["turn_idx"])))
        print(
            f"shard {args.shard_id}: {len(results)}/{len(shard_records)} "
            f"{record['sample_id']} turn {record['turn_idx']} -> {result['type'] or 'PARSE_INVALID'}",
            flush=True,
        )


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def merge_results(args: argparse.Namespace, manifest: List[Dict[str, Any]]) -> None:
    output_dir = Path(args.output_dir)
    results = []
    for shard_id in range(args.num_shards):
        shard_path = output_dir / "shards" / f"shard_{shard_id}.jsonl"
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        results.extend(read_jsonl(shard_path))
    results.sort(key=lambda row: (row["sample_id"], row["turn_idx"]))
    keys = {(row["sample_id"], row["turn_idx"]) for row in results}
    expected = {(row["sample_id"], row["turn_idx"]) for row in manifest}
    if len(results) != len(keys) or keys != expected:
        raise RuntimeError(
            f"merge integrity failed: rows={len(results)}, unique={len(keys)}, expected={len(expected)}"
        )
    write_jsonl(output_dir / "irrelevant_crop_classifications.jsonl", results)
    with (output_dir / "irrelevant_crop_classifications.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({field: csv_value(row.get(field, "")) for field in RESULT_FIELDS})

    counts = Counter(row["type"] or "PARSE_INVALID" for row in results)
    by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        by_category[row["category"]][row["type"] or "PARSE_INVALID"] += 1
    summary_rows = []
    for category in ["ALL", *sorted(by_category)]:
        category_counts = counts if category == "ALL" else by_category[category]
        total = sum(category_counts.values())
        for label in [*sorted(VALID_TYPES), "PARSE_INVALID"]:
            count = category_counts[label]
            summary_rows.append(
                {
                    "category": category,
                    "type": label,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                    "total": total,
                }
            )
    with (output_dir / "classification_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "type", "count", "fraction", "total"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"rows": len(results), "counts": counts}, ensure_ascii=False, default=dict))


def main() -> None:
    args = parse_args()
    if args.prepare_only and args.merge_only:
        raise ValueError("--prepare-only and --merge-only are mutually exclusive")
    manifest = make_manifest(args)
    print(f"manifest: {len(manifest)} irrelevant crops", flush=True)
    if args.prepare_only:
        return
    if args.merge_only:
        merge_results(args, manifest)
        return
    run_shard(args, manifest)


if __name__ == "__main__":
    main()
