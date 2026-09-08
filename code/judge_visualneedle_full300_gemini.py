#!/usr/bin/env python3
"""Apply the official VisualNeedle VLM-judge protocol with Gemini Flash."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional


OFFICIAL_AGENT = Path("/home/songzhoujie/cvpr27/VisualNeedle-official/visualneedle_eval/visualneedle_agent.py")
DEFAULT_MODEL = "gemini-3.8-flash"


def load_official_system_prompt() -> str:
    tree = ast.parse(OFFICIAL_AGENT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "VLM_JUDGE_SYSTEM_PROMPT" for target in targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    return value
    raise RuntimeError("VLM_JUDGE_SYSTEM_PROMPT not found in official repository")


def parse_yes_no(output: str) -> Optional[bool]:
    if not output:
        return None
    tokens = output.strip().split()
    if tokens:
        token = tokens[0].strip().rstrip(".)，,。:：").upper()
        if token == "YES":
            return True
        if token == "NO":
            return False
    upper = output.upper()
    if "YES" in upper and "NO" not in upper:
        return True
    if "NO" in upper and "YES" not in upper:
        return False
    return None


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def request_judge(
    model: str, system_prompt: str, image_path: Path, prompt: str, api_key: str
) -> Dict[str, Any]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": mime, "data": image_data}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {
            "temperature": 1.0,
            "thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": True},
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read())
    elapsed = time.perf_counter() - started
    candidate = (result.get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts", [])
    visible_text = "".join(str(part.get("text", "")) for part in parts if not part.get("thought")).strip()
    thought_text = "".join(str(part.get("text", "")) for part in parts if part.get("thought")).strip()
    return {
        "raw_output": visible_text,
        "thought_output": thought_text,
        "finish_reason": candidate.get("finishReason", ""),
        "usage_metadata": result.get("usageMetadata", {}),
        "runtime_seconds": elapsed,
    }


def judge_one(episode: Dict[str, Any], model: str, system_prompt: str, api_key: str, raw_dir: Path) -> Dict[str, Any]:
    sample_id = str(episode["sample_id"])
    output_path = raw_dir / f"{sample_id}.json"
    if output_path.exists():
        try:
            prior = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                prior.get("finished") is True
                and prior.get("model") == model
                and prior.get("judge_parse_success") is True
            ):
                return prior
        except Exception:
            pass
    model_response = str((episode.get("turns") or [{}])[-1].get("raw_model_output", ""))
    base_prompt = (
        f"**Question asked about the image:**\n{episode['question']}\n\n"
        f"**Ground-truth label answer:**\n{episode['gt_answer']}\n\n"
        f"**Model's parsed answer:**\n{episode.get('final_answer') or '<NONE>'}\n\n"
        f"**Model's full response:**\n{model_response.strip()[:8000]}\n\n"
        "Based on the image, question, and the answers above, is the model's answer correct?\n"
        "Output EXACTLY one word: YES or NO."
    )
    attempts = []
    parsed: Optional[bool] = None
    for retry in range(2):
        prompt = base_prompt if retry == 0 else base_prompt + "\n\nCRITICAL: Output ONLY YES or NO, nothing else."
        result: Dict[str, Any] = {}
        for transport_retry in range(6):
            try:
                result = request_judge(model, system_prompt, Path(episode["original_image_path"]), prompt, api_key)
                result["parsed_result"] = parse_yes_no(result["raw_output"])
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                result = {
                    "error_type": type(exc).__name__,
                    "error_message": f"HTTP {exc.code}",
                    "transport_retry": transport_retry,
                    "parsed_result": None,
                }
                if not retryable or transport_retry == 5:
                    break
                time.sleep(min(60, 5 * (2 ** transport_retry)))
            except Exception as exc:
                result = {"error_type": type(exc).__name__, "error_message": str(exc)[:1000], "parsed_result": None}
                if transport_retry == 5:
                    break
                time.sleep(min(60, 5 * (2 ** transport_retry)))
        attempts.append(result)
        parsed = result["parsed_result"]
        if parsed is not None:
            break
        time.sleep(1)
    record = {
        "sample_id": sample_id,
        "model": model,
        "judge_result": parsed,
        "judge_parse_success": parsed is not None,
        "attempts": attempts,
        "finished": True,
    }
    atomic_write_json(output_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("VISUALNEEDLE_MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or VISUALNEEDLE_MODEL_API_KEY is required")
    output_dir = args.output_dir.resolve()
    summaries = {row["sample_id"]: row for row in csv.DictReader((output_dir / "episode_summary.csv").open())}
    episodes = [json.loads(line) for line in (output_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    pending = [ep for ep in episodes if summaries[ep["sample_id"]].get("official_judge_pending") == "True"]
    if args.limit is not None:
        pending = pending[: args.limit]
    system_prompt = load_official_system_prompt()
    raw_dir = output_dir / "official_judge_raw"
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, ep, args.model, system_prompt, api_key, raw_dir): ep["sample_id"] for ep in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"[{index}/{len(pending)}] {record['sample_id']} result={record['judge_result']}", flush=True)
    all_records = []
    for path in sorted(raw_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("model") == args.model:
            all_records.append(record)
    fields = ["sample_id", "model", "judge_result", "judge_parse_success", "raw_output", "finish_reason", "runtime_seconds", "prompt_tokens", "candidate_tokens", "thought_tokens", "total_tokens", "error"]
    with (output_dir / "official_judge_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in all_records:
            attempt = record.get("attempts", [{}])[-1]
            usage = attempt.get("usage_metadata", {})
            writer.writerow({
                "sample_id": record["sample_id"], "model": record["model"],
                "judge_result": record.get("judge_result"), "judge_parse_success": record.get("judge_parse_success"),
                "raw_output": attempt.get("raw_output", ""), "finish_reason": attempt.get("finish_reason", ""),
                "runtime_seconds": attempt.get("runtime_seconds", ""), "prompt_tokens": usage.get("promptTokenCount", ""),
                "candidate_tokens": usage.get("candidatesTokenCount", ""), "thought_tokens": usage.get("thoughtsTokenCount", ""),
                "total_tokens": usage.get("totalTokenCount", ""),
                "error": attempt.get("error_message", ""),
            })
    print(json.dumps({"records": len(all_records), "yes": sum(r.get("judge_result") is True for r in all_records), "no": sum(r.get("judge_result") is False for r in all_records), "unparsed": sum(r.get("judge_result") is None for r in all_records)}, indent=2))


if __name__ == "__main__":
    main()
