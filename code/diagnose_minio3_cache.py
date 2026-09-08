#!/usr/bin/env python3
"""First-round Mini-o3 KV-cache performance A/B without touching baseline outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image

from run_visualneedle_minio3_smoke20 import (
    DATA_ROOT,
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_PATH,
    SYSTEM_PROMPT,
    initial_messages,
    process_for_model,
)


SAMPLE_ID = "TPROMPT65ab072c3b73"
OUTPUT_DIR = Path("/home/songzhoujie/cvpr27/vstar/outputs/visualneedle_minio3/cache_diagnosis")
MAX_NEW_TOKENS = 128
MAX_TIME_SECONDS = 600
SEED = 42


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_inference_fields() -> Dict[str, str]:
    annotation = DATA_ROOT / "visualneedle_300en.jsonl"
    for line in annotation.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if str(row["id"]) == SAMPLE_ID:
            image_path = (DATA_ROOT / row["image_path"]).resolve()
            if image_path.parent != (DATA_ROOT / "images").resolve():
                raise ValueError(f"Refusing non-original image path: {image_path}")
            return {
                "sample_id": SAMPLE_ID,
                "question": str(row["question"]),
                "image_path": str(image_path),
            }
    raise KeyError(SAMPLE_ID)


def cache_description(cache: Any) -> Dict[str, Any]:
    if cache is None:
        return {
            "cache_returned": False,
            "cache_type": None,
            "cache_num_layers": None,
            "cache_sequence_length": None,
            "first_key_shape": None,
        }
    description: Dict[str, Any] = {
        "cache_returned": True,
        "cache_type": f"{type(cache).__module__}.{type(cache).__name__}",
        "cache_num_layers": None,
        "cache_sequence_length": None,
        "first_key_shape": None,
    }
    try:
        description["cache_num_layers"] = len(cache)
    except Exception:
        pass
    for method_name in ("get_seq_length", "get_usable_length"):
        method = getattr(cache, method_name, None)
        if callable(method):
            try:
                description["cache_sequence_length"] = int(method())
                break
            except Exception:
                pass
    try:
        first = cache[0]
        key = first[0] if isinstance(first, (tuple, list)) else getattr(first, "key", None)
        if key is not None and hasattr(key, "shape"):
            description["first_key_shape"] = list(key.shape)
    except Exception:
        pass
    return description


class Diagnosis:
    def __init__(self) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.transformers = transformers
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run this diagnostic in the host GPU environment")
        self.processor = AutoProcessor.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def environment(self) -> Dict[str, Any]:
        config = self.model.config
        generation_config = self.model.generation_config
        dtype = str(next(self.model.parameters()).dtype)
        return {
            "sample_id": SAMPLE_ID,
            "model_path": str(MODEL_PATH),
            "model_config_use_cache": getattr(config, "use_cache", None),
            "model_text_config_use_cache": getattr(getattr(config, "text_config", None), "use_cache", None),
            "generation_config_use_cache": getattr(generation_config, "use_cache", None),
            "model_config_attn_implementation": getattr(config, "_attn_implementation", None),
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers.__version__,
            "cuda_device_name": self.torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "model_dtype": dtype,
            "processor_min_pixels": MIN_PIXELS,
            "processor_max_pixels": MAX_PIXELS,
            "system_prompt": SYSTEM_PROMPT,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_time_seconds": MAX_TIME_SECONDS,
        }

    def prepare(self, image: Image.Image, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        processed, preprocess_metadata = process_for_model(image)
        messages = initial_messages(processed, question)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[processed], padding=True, return_tensors="pt")
        processor_time = time.perf_counter() - started

        self.torch.cuda.synchronize()
        h2d_started = time.perf_counter()
        device_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        self.torch.cuda.synchronize()
        h2d_time = time.perf_counter() - h2d_started
        return {
            "inputs": device_inputs,
            "processor_time_seconds": processor_time,
            "h2d_time_seconds": h2d_time,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "preprocess_metadata": preprocess_metadata,
            "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None,
        }

    def one_run(self, label: str, use_cache: bool, image: Image.Image, question: str, measured: bool) -> Dict[str, Any]:
        prepared = self.prepare(image, question)
        inputs = prepared.pop("inputs")
        input_tokens = int(prepared["input_tokens"])
        self.torch.manual_seed(SEED)
        self.torch.cuda.manual_seed_all(SEED)
        self.torch.cuda.reset_peak_memory_stats()
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=use_cache,
                max_time=MAX_TIME_SECONDS,
                return_dict_in_generate=True,
            )
        self.torch.cuda.synchronize()
        generate_time = time.perf_counter() - started

        sequence = output.sequences
        generated_ids = sequence[:, input_tokens:]
        output_tokens = int(generated_ids.shape[1])
        raw_output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        final_token = int(sequence[0, -1].item())
        if final_token in eos_ids:
            finish_reason = "eos"
        elif output_tokens >= MAX_NEW_TOKENS:
            finish_reason = "length"
        else:
            finish_reason = "time_or_unknown"
        result = {
            "label": label,
            "measured": measured,
            "use_cache": use_cache,
            **prepared,
            "generate_time_seconds": generate_time,
            "output_tokens": output_tokens,
            "tokens_per_second": output_tokens / generate_time if generate_time else None,
            "peak_gpu_memory_mb": self.torch.cuda.max_memory_allocated() / (1024 * 1024),
            "finish_reason": finish_reason,
            "raw_output": raw_output,
            "generated_token_ids": generated_ids[0].detach().cpu().tolist(),
            **cache_description(getattr(output, "past_key_values", None)),
        }
        del output, sequence, generated_ids, inputs
        return result


def write_report(environment: Dict[str, Any], measured: Sequence[Dict[str, Any]]) -> None:
    by_cache = {bool(row["use_cache"]): row for row in measured}
    without = by_cache[False]
    with_cache = by_cache[True]
    speedup = without["generate_time_seconds"] / with_cache["generate_time_seconds"]
    outputs_identical = without["generated_token_ids"] == with_cache["generated_token_ids"]
    no_cache_tps = float(without["tokens_per_second"])
    cache_tps = float(with_cache["tokens_per_second"])
    explains = speedup >= 1.5 and no_cache_tps <= 1.0
    recommendation = (
        "下一步在独立 smoke 副本上验证显式 `use_cache=True`，确认轨迹逐 token 一致后再考虑正式重跑。"
        if explains
        else "KV cache 单变量不足以解释异常；下一步按计划检查 SDPA、GPU utilization/power/clock、kernel 与视觉编码耗时。"
    )
    lines = [
        "# Mini-o3 KV Cache Diagnosis",
        "",
        "## 配置",
        "",
        f"- checkpoint `model.config.use_cache`: `{environment['model_config_use_cache']}`",
        f"- checkpoint `model.config.text_config.use_cache`: `{environment['model_text_config_use_cache']}`",
        f"- checkpoint `generation_config.use_cache`: `{environment['generation_config_use_cache']}`",
        f"- attention implementation: `{environment['model_config_attn_implementation']}`",
        f"- torch / transformers: `{environment['torch_version']}` / `{environment['transformers_version']}`",
        f"- GPU: `{environment['cuda_device_name']}` (`CUDA_VISIBLE_DEVICES={environment['cuda_visible_devices']}`)",
        f"- model dtype: `{environment['model_dtype']}`",
        "- A/B: same first-round image, question, official system prompt, processor, BF16, SDPA, greedy decoding, `max_new_tokens=128`; only `use_cache` differs.",
        "",
        "## 结果",
        "",
        "| setting | generate time (s) | output tokens | tok/s | peak GPU MB | finish | cache returned |",
        "|---|---:|---:|---:|---:|---|---|",
        f"| use_cache=False | {without['generate_time_seconds']:.3f} | {without['output_tokens']} | {no_cache_tps:.3f} | {without['peak_gpu_memory_mb']:.1f} | {without['finish_reason']} | {without['cache_returned']} |",
        f"| use_cache=True | {with_cache['generate_time_seconds']:.3f} | {with_cache['output_tokens']} | {cache_tps:.3f} | {with_cache['peak_gpu_memory_mb']:.1f} | {with_cache['finish_reason']} | {with_cache['cache_returned']} |",
        "",
        f"- speedup: **{speedup:.2f}x**",
        f"- generated token IDs / outputs identical: **{outputs_identical}**",
        f"- `use_cache=True` returned cache type: `{with_cache['cache_type']}`",
        f"- returned cache sequence length: `{with_cache['cache_sequence_length']}`",
        f"- 是否足以解释 0.4-0.9 token/s: **{'是' if explains else '否'}**",
        "",
        "## 结论与下一步",
        "",
        recommendation,
        "",
        "本诊断未修改正式 runner、prompt、crop/termination policy、图像预处理、样本选择或 smoke20 输出。",
    ]
    (OUTPUT_DIR / "diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = load_inference_fields()
    with Image.open(sample["image_path"]) as handle:
        image = handle.convert("RGB")
    diagnosis = Diagnosis()
    environment = diagnosis.environment()
    environment["question"] = sample["question"]
    environment["image_path"] = sample["image_path"]
    write_json(OUTPUT_DIR / "environment_config.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)

    all_runs: List[Dict[str, Any]] = []
    for label, use_cache in (("A", False), ("B", True)):
        print(f"[{label}] warmup use_cache={use_cache}", flush=True)
        warmup = diagnosis.one_run(label, use_cache, image, sample["question"], measured=False)
        all_runs.append(warmup)
        write_json(OUTPUT_DIR / "runs_partial.json", all_runs)
        print(f"[{label}] measured use_cache={use_cache}", flush=True)
        measured = diagnosis.one_run(label, use_cache, image, sample["question"], measured=True)
        all_runs.append(measured)
        write_json(OUTPUT_DIR / "runs_partial.json", all_runs)
        print(
            f"[{label}] {measured['output_tokens']} tokens in {measured['generate_time_seconds']:.3f}s "
            f"({measured['tokens_per_second']:.3f} tok/s), cache={measured['cache_type']}",
            flush=True,
        )

    write_json(OUTPUT_DIR / "cache_ab_results.json", all_runs)
    measured_rows = [row for row in all_runs if row["measured"]]
    csv_fields = [
        "label", "measured", "use_cache", "processor_time_seconds", "h2d_time_seconds",
        "generate_time_seconds", "input_tokens", "output_tokens", "tokens_per_second",
        "peak_gpu_memory_mb", "finish_reason", "cache_returned", "cache_type",
        "cache_num_layers", "cache_sequence_length", "first_key_shape", "raw_output",
    ]
    with (OUTPUT_DIR / "cache_ab_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measured_rows)
    write_report(environment, measured_rows)
    print((OUTPUT_DIR / "diagnosis_report.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
