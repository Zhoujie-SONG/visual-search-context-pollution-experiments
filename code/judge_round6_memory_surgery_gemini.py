#!/usr/bin/env python3
"""Resume-safe official image-aware judging for memory-surgery outputs."""

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
from run_round6_memory_surgery import stage_directory


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def needs_judge(episode: Dict[str, Any]) -> bool:
    prediction = str(episode.get("final_answer") or "").strip()
    return bool(prediction) and not official_score.answers_match(
        prediction, str(episode["gt_answer"]), question=str(episode["question"])
    )


def judge_episode(
    episode: Dict[str, Any], model: str, system_prompt: str, api_key: str, root: Path
) -> Dict[str, Any]:
    arm = str(episode["condition"])
    record = official_judge.judge_one(episode, model, system_prompt, api_key, root / arm)
    return {**record, "condition": arm}


def write_results(path: Path, records: List[Dict[str, Any]]) -> None:
    fields = [
        "sample_id", "condition", "model", "judge_result", "judge_parse_success",
        "raw_output", "finish_reason", "runtime_seconds", "prompt_tokens",
        "candidate_tokens", "thought_tokens", "total_tokens", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda row: (row["sample_id"], row["condition"])):
            attempt = (record.get("attempts") or [{}])[-1]
            usage = attempt.get("usage_metadata") or {}
            writer.writerow({
                "sample_id": record["sample_id"], "condition": record["condition"],
                "model": record["model"], "judge_result": record.get("judge_result"),
                "judge_parse_success": record.get("judge_parse_success"),
                "raw_output": attempt.get("raw_output", ""),
                "finish_reason": attempt.get("finish_reason", ""),
                "runtime_seconds": attempt.get("runtime_seconds", ""),
                "prompt_tokens": usage.get("promptTokenCount", ""),
                "candidate_tokens": usage.get("candidatesTokenCount", ""),
                "thought_tokens": usage.get("thoughtsTokenCount", ""),
                "total_tokens": usage.get("totalTokenCount", ""),
                "error": attempt.get("error_message", ""),
            })


def load_cached(raw_root: Path, model: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    records = {}
    if not raw_root.exists():
        return records
    for arm_dir in raw_root.iterdir():
        if not arm_dir.is_dir():
            continue
        for path in arm_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.get("model") == model:
                record["condition"] = arm_dir.name
                records[(str(record["sample_id"]), arm_dir.name)] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("VISUALNEEDLE_MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or VISUALNEEDLE_MODEL_API_KEY is required")
    output_dir = stage_directory(args.stage)
    episodes = [ep for ep in read_jsonl(output_dir / "merged_trajectories.jsonl") if needs_judge(ep)]
    raw_root = output_dir / "official_judge_raw"
    cached = load_cached(raw_root, args.model)
    pending = [ep for ep in episodes if (str(ep["sample_id"]), str(ep["condition"])) not in cached or not cached[(str(ep["sample_id"]), str(ep["condition"]))].get("judge_parse_success")]
    if args.limit is not None:
        pending = pending[: args.limit]
    system_prompt = official_judge.load_official_system_prompt()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(judge_episode, episode, args.model, system_prompt, api_key, raw_root):
            (episode["sample_id"], episode["condition"])
            for episode in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            cached[(str(record["sample_id"]), str(record["condition"]))] = record
            print(
                f"[{index}/{len(pending)}] {record['sample_id']} {record['condition']} "
                f"result={record.get('judge_result')}", flush=True,
            )
    relevant = [cached[key] for key in sorted(cached) if key in {(str(ep["sample_id"]), str(ep["condition"])) for ep in episodes}]
    write_results(output_dir / "official_judge_results.csv", relevant)
    print(json.dumps({
        "required": len(episodes), "records": len(relevant),
        "yes": sum(record.get("judge_result") is True for record in relevant),
        "no": sum(record.get("judge_result") is False for record in relevant),
        "unparsed": sum(record.get("judge_result") is None for record in relevant),
    }, indent=2))


if __name__ == "__main__":
    main()

