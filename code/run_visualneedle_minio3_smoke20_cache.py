#!/usr/bin/env python3
"""Cache-enabled wrapper around the frozen VisualNeedle Mini-o3 smoke20 runner.

The imported runner owns every protocol and evaluation decision. This module only
redirects outputs and enables the model's standard within-call KV cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image

import run_visualneedle_minio3_smoke20 as baseline


CACHE_OUTPUT_DIR = baseline.OUTPUT_PARENT / "smoke20_cache"
ORIGINAL_WRITE_CONFIG = baseline.write_config


class CacheEnabledHFBackend(baseline.HFBackend):
    def __init__(self, model_path: Path):
        super().__init__(model_path)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        self.model.config.text_config.use_cache = True
        self.model.generation_config.use_cache = True

    def generate(self, messages: Sequence[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images: List[Image.Image] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                images.extend(
                    item["image"]
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "image"
                )
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[1])
        image_grid_thw = inputs.get("image_grid_thw")
        image_grids = image_grid_thw.detach().cpu().tolist() if image_grid_thw is not None else None
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        started = baseline.time.perf_counter()
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    max_time=baseline.ROUND_MAX_TIME_SECONDS,
                    return_dict_in_generate=True,
                )
            elapsed = baseline.time.perf_counter() - started
            sequence = output.sequences
            new_ids = sequence[:, input_tokens:]
            raw = self.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            final_token = int(sequence[0, -1].item())
            if final_token in eos_ids:
                finish_reason = "eos"
            elif int(new_ids.shape[1]) >= max_new_tokens:
                finish_reason = "length"
            else:
                finish_reason = "time_or_unknown"
            output_tokens = int(new_ids.shape[1])
            return {
                "raw_model_output": raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "runtime_seconds": elapsed,
                "tokens_per_second": output_tokens / elapsed if elapsed else None,
                "input_image_count": len(images),
                "image_grid_thw": image_grids,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
            }
        finally:
            del inputs
            self.torch.cuda.empty_cache()


def write_cache_config(ids: Sequence[str], command: str) -> None:
    ORIGINAL_WRITE_CONFIG(ids, command)
    path = CACHE_OUTPUT_DIR / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(
        {
            "experiment": "VisualNeedle Mini-o3 original active visual search smoke20 with KV cache",
            "use_cache": True,
            "model_config_text_config_use_cache": True,
            "model_generation_config_use_cache": True,
            "model_generate_explicit_use_cache": True,
            "cross_round_kv_reuse": False,
            "only_difference_from_smoke20": "standard within-generate KV cache enabled",
            "baseline_output_directory": str(baseline.OUTPUT_PARENT / "smoke20"),
            "output_directory": str(CACHE_OUTPUT_DIR),
        }
    )
    baseline.atomic_write_json(path, config)


def main() -> None:
    baseline.OUTPUT_DIR = CACHE_OUTPUT_DIR
    baseline.HFBackend = CacheEnabledHFBackend
    baseline.write_config = write_cache_config
    for field in ("use_cache", "tokens_per_second"):
        if field not in baseline.TURN_FIELDS:
            baseline.TURN_FIELDS.append(field)
    baseline.main()


if __name__ == "__main__":
    main()
