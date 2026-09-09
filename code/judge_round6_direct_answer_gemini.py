#!/usr/bin/env python3
"""Resume-safe Gemini judging for lenient-valid direct answers."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import judge_visualneedle_full300_gemini as official_judge
import score_visualneedle_full300 as official_score
from run_round6_direct_answer import ARMS, read_jsonl, stage_dir


def needs_judge(row: Dict[str, Any]) -> bool:
    answer = str(row.get("parsed_answer") or "").strip()
    return bool(row.get("lenient_valid_answer")) and bool(answer) and not official_score.answers_match(
        answer, str(row["gt_answer"]), question=str(row["question"])
    )


def judge_one(row: Dict[str, Any], model: str, prompt: str, key: str, raw_root: Path) -> Dict[str, Any]:
    payload = {
        "sample_id": row["sample_id"],
        "question": row["question"],
        "gt_answer": row["gt_answer"],
        "final_answer": row["parsed_answer"],
        "original_image_path": row["original_image_path"],
        "turns": [{"raw_model_output": row["raw_model_output"]}],
    }
    record = official_judge.judge_one(payload, model, prompt, key, raw_root / str(row["condition"]))
    return {**record, "condition": row["condition"]}


def load_cached(raw_root: Path, model: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    cached: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for arm in ARMS:
        for path in (raw_root / arm).glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if row.get("model") == model:
                row["condition"] = arm
                cached[(str(row["sample_id"]), arm)] = row
    return cached


def write_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "sample_id", "condition", "model", "judge_result", "judge_parse_success",
        "raw_output", "finish_reason", "runtime_seconds", "prompt_tokens",
        "candidate_tokens", "thought_tokens", "total_tokens", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["sample_id"], item["condition"])):
            attempt = (row.get("attempts") or [{}])[-1]
            usage = attempt.get("usage_metadata") or {}
            writer.writerow({
                "sample_id": row["sample_id"], "condition": row["condition"], "model": row["model"],
                "judge_result": row.get("judge_result"), "judge_parse_success": row.get("judge_parse_success"),
                "raw_output": attempt.get("raw_output", ""), "finish_reason": attempt.get("finish_reason", ""),
                "runtime_seconds": attempt.get("runtime_seconds", ""), "prompt_tokens": usage.get("promptTokenCount", ""),
                "candidate_tokens": usage.get("candidatesTokenCount", ""), "thought_tokens": usage.get("thoughtsTokenCount", ""),
                "total_tokens": usage.get("totalTokenCount", ""), "error": attempt.get("error_message", ""),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sanity", "full"), default="full")
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("VISUALNEEDLE_MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or VISUALNEEDLE_MODEL_API_KEY is required")
    output = stage_dir(args.stage)
    needed = [row for row in read_jsonl(output / "merged_results.jsonl") if needs_judge(row)]
    needed_keys = {(str(row["sample_id"]), str(row["condition"])) for row in needed}
    raw_root = output / "official_judge_raw"
    cached = load_cached(raw_root, args.model)
    pending = [row for row in needed if (str(row["sample_id"]), str(row["condition"])) not in cached or not cached[(str(row["sample_id"]), str(row["condition"]))].get("judge_parse_success")]
    system_prompt = official_judge.load_official_system_prompt()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, row, args.model, system_prompt, api_key, raw_root): (row["sample_id"], row["condition"]) for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            cached[(str(record["sample_id"]), str(record["condition"]))] = record
            print(f"[{index}/{len(pending)}] {record['sample_id']} {record['condition']} result={record.get('judge_result')}", flush=True)
    relevant = [cached[key] for key in sorted(cached) if key in needed_keys]
    write_results(output / "official_judge_results.csv", relevant)
    print(json.dumps({
        "required": len(needed), "records": len(relevant),
        "yes": sum(row.get("judge_result") is True for row in relevant),
        "no": sum(row.get("judge_result") is False for row in relevant),
        "pending": sum(row.get("judge_result") is None for row in relevant),
    }, indent=2))


if __name__ == "__main__":
    main()
