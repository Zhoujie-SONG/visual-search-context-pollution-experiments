#!/usr/bin/env python3
"""Semantically rescore VisualNeedle agent and plain-VQA answers with a local judge."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SYSTEM_PROMPT = """你是严格但允许自然语言改写的 VQA 答案评审员。请结合问题，判断候选输出是否表达了与参考答案相同的最终答案。

规则：
1. 忽略大小写、冠词、标点、单复数和不改变答案含义的完整句式。
2. 同义词、简称、上位/下位表达只有在问题语境中明确等价时才算正确。
3. 数字、文字识别、颜色、方位和实体身份中的实质差异必须判错。
4. 候选若只有未完成推理、列出多个可能、拒答或没有明确最终结论，判为 NO_ANSWER。
5. 即使缺少 <answer> 标签，只要候选明确给出唯一最终答案，也可以判为 CORRECT 或 INCORRECT。
6. 不要因为候选偶然提到参考答案就判对；它必须是候选最终采纳的答案。

只输出一个 JSON 对象，不要输出 Markdown 或额外文字。"""

OUTPUT_SCHEMA = """{
  "verdict": "CORRECT | INCORRECT | NO_ANSWER",
  "extracted_answer": "候选明确表达的最终答案，没有则为空字符串",
  "reason": "一句简短理由"
}"""

FIELDS = [
    "evaluation_source", "sample_id", "category", "question", "gt_answer",
    "candidate_output", "original_status", "original_strict_correct",
    "original_contains_match", "verdict", "semantic_correct", "extracted_answer",
    "reason", "parse_status", "attempts", "input_token_count", "output_token_count",
    "raw_judge_output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_items(input_dir: Path) -> List[Dict[str, Any]]:
    episodes = read_csv(input_dir / "episode_summary.csv")
    turns = read_csv(input_dir / "turns.csv")
    final_raw: Dict[str, str] = {}
    final_turn: Dict[str, int] = {}
    for turn in turns:
        sample_id = turn["sample_id"]
        turn_id = int(turn["turn_id"])
        if turn_id >= final_turn.get(sample_id, -1):
            final_turn[sample_id] = turn_id
            final_raw[sample_id] = turn.get("raw_model_output", "").strip()

    items = []
    for row in episodes:
        candidate = row.get("pred_answer", "").strip() or final_raw.get(row["sample_id"], "")
        items.append(
            {
                "evaluation_source": "agent",
                "sample_id": row["sample_id"],
                "category": row["category"],
                "question": row["question"],
                "gt_answer": row["gt_answer"],
                "candidate_output": candidate,
                "original_status": row.get("status", ""),
                "original_strict_correct": row.get("strict_correct", ""),
                "original_contains_match": row.get("contains_match", ""),
            }
        )

    for row in read_csv(input_dir / "plain_vqa_anchor.csv"):
        items.append(
            {
                "evaluation_source": "plain_vqa",
                "sample_id": row["sample_id"],
                "category": row["category"],
                "question": next(
                    episode["question"] for episode in episodes if episode["sample_id"] == row["sample_id"]
                ),
                "gt_answer": row["gt_answer"],
                "candidate_output": row.get("pred_answer", "").strip(),
                "original_status": row.get("status", ""),
                "original_strict_correct": row.get("strict_correct", ""),
                "original_contains_match": row.get("contains_match", ""),
            }
        )
    return items


def prompt(item: Dict[str, Any], correction: str = "") -> str:
    candidate = item["candidate_output"] or "[EMPTY]"
    return f"""问题：{item['question']}
参考答案：{item['gt_answer']}
候选输出：{candidate}

请按语义等价规则评审。输出格式：
{OUTPUT_SCHEMA}{correction}"""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S))
    for start in (index for index, char in enumerate(text) if char == "{"):
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:index + 1])
                    break
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def validate(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not value or value.get("verdict") not in {"CORRECT", "INCORRECT", "NO_ANSWER"}:
        return None
    if not isinstance(value.get("extracted_answer"), str) or not isinstance(value.get("reason"), str):
        return None
    return {
        "verdict": value["verdict"],
        "extracted_answer": value["extracted_answer"].strip(),
        "reason": value["reason"].strip(),
    }


class Judge:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
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

    def generate(self, text: str, max_new_tokens: int) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[rendered], padding=True, return_tensors="pt")
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
            "raw": raw,
            "input_token_count": input_length,
            "output_token_count": int(new_ids.shape[1]),
        }


def run_shard(args: argparse.Namespace, items: Sequence[Dict[str, Any]]) -> None:
    if args.shard_id is None or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    output_path = args.output_dir / "shards" / f"shard_{args.shard_id}.jsonl"
    existing = {} if args.overwrite else {
        (row["evaluation_source"], row["sample_id"]): row for row in read_jsonl(output_path)
    }
    assigned = [item for index, item in enumerate(items) if index % args.num_shards == args.shard_id]
    if len(existing) == len(assigned):
        print(f"shard {args.shard_id}: already complete ({len(existing)})", flush=True)
        return
    judge = Judge(args.model)
    results = list(existing.values())
    for item in assigned:
        key = (item["evaluation_source"], item["sample_id"])
        if key in existing:
            continue
        attempts = []
        generated = judge.generate(prompt(item), args.max_new_tokens)
        attempts.append(generated)
        parsed = validate(extract_json(generated["raw"]))
        if parsed is None:
            generated = judge.generate(
                prompt(item, "\n上次格式不合规。只输出字段完整、可解析的 JSON 对象。"),
                args.max_new_tokens,
            )
            attempts.append(generated)
            parsed = validate(extract_json(generated["raw"]))
        result = dict(item)
        if parsed is None:
            result.update({"verdict": "", "semantic_correct": False, "extracted_answer": "", "reason": "", "parse_status": "invalid"})
        else:
            result.update(parsed)
            result["semantic_correct"] = parsed["verdict"] == "CORRECT"
            result["parse_status"] = "ok"
        result.update(
            {
                "attempts": len(attempts),
                "input_token_count": attempts[-1]["input_token_count"],
                "output_token_count": attempts[-1]["output_token_count"],
                "raw_judge_output": attempts[-1]["raw"],
            }
        )
        results.append(result)
        results.sort(key=lambda row: (row["evaluation_source"], row["sample_id"]))
        write_jsonl(output_path, results)
        print(
            f"shard {args.shard_id}: {len(results)}/{len(assigned)} "
            f"{item['evaluation_source']} {item['sample_id']} -> {result['verdict'] or 'PARSE_INVALID'}",
            flush=True,
        )


def fraction(rows: Sequence[Dict[str, Any]], field: str, value: Any = True) -> float:
    return sum(row.get(field) == value for row in rows) / len(rows) if rows else 0.0


def merge(args: argparse.Namespace, items: Sequence[Dict[str, Any]]) -> None:
    results = []
    for shard_id in range(args.num_shards):
        results.extend(read_jsonl(args.output_dir / "shards" / f"shard_{shard_id}.jsonl"))
    expected = {(row["evaluation_source"], row["sample_id"]) for row in items}
    actual = {(row["evaluation_source"], row["sample_id"]) for row in results}
    if len(results) != len(actual) or actual != expected:
        raise RuntimeError(f"merge integrity failed: rows={len(results)} unique={len(actual)} expected={len(expected)}")
    results.sort(key=lambda row: (row["evaluation_source"], row["sample_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "semantic_rescore.jsonl", results)
    with (args.output_dir / "semantic_rescore.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in results)

    summary = []
    for source in ["agent", "plain_vqa"]:
        source_rows = [row for row in results if row["evaluation_source"] == source]
        for category in ["ALL", *sorted({row["category"] for row in source_rows})]:
            group = source_rows if category == "ALL" else [row for row in source_rows if row["category"] == category]
            counts = Counter(row["verdict"] or "PARSE_INVALID" for row in group)
            summary.append(
                {
                    "evaluation_source": source,
                    "category": category,
                    "num_samples": len(group),
                    "semantic_correct": counts["CORRECT"],
                    "semantic_accuracy": counts["CORRECT"] / len(group) if group else 0.0,
                    "incorrect": counts["INCORRECT"],
                    "no_answer": counts["NO_ANSWER"],
                    "judge_parse_invalid": counts["PARSE_INVALID"],
                    "original_strict_accuracy": fraction(group, "original_strict_correct", "True"),
                    "original_contains_accuracy": fraction(group, "original_contains_match", "True"),
                }
            )
    summary_fields = list(summary[0])
    with (args.output_dir / "semantic_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    overall = [row for row in summary if row["category"] == "ALL"]
    report = [
        "# VisualNeedle 100-sample Semantic Rescore",
        "",
        "Semantic correctness ignores formatting and accepts meaning-preserving paraphrases.",
        "A missing tag is accepted only when the output still states one unambiguous final answer.",
        "",
    ]
    for row in overall:
        report.extend(
            [
                f"## {row['evaluation_source']}",
                "",
                f"- Semantic accuracy: {row['semantic_accuracy']:.4f} ({row['semantic_correct']}/{row['num_samples']})",
                f"- Incorrect: {row['incorrect']}",
                f"- No answer: {row['no_answer']}",
                f"- Judge parse invalid: {row['judge_parse_invalid']}",
                f"- Original strict accuracy: {row['original_strict_accuracy']:.4f}",
                f"- Original contains accuracy: {row['original_contains_accuracy']:.4f}",
                "",
            ]
        )
    (args.output_dir / "semantic_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    items = load_items(args.input_dir)
    if args.merge_only:
        merge(args, items)
    else:
        run_shard(args, items)


if __name__ == "__main__":
    main()
