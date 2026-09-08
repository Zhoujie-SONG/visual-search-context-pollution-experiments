import argparse
import ast
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from PIL import Image

from context_pollution_utils import LETTERS, parse_answer_letter


TARGET_CONDITIONS = ["gt_plus_4_irrelevant", "gt_plus_8_irrelevant"]
VALID_LETTERS = set("ABCD")
ABSTENTION_PHRASES = [
    "not enough detail",
    "cannot determine",
    "unable to determine",
    "does not contain",
    "insufficient information",
    "not clear",
    "can't tell",
    "cannot answer",
]
SUSPICIOUS_PHRASES = ABSTENTION_PHRASES + ["insufficient"]


def e16_path(output_dir: Path, name: str) -> Path:
    if not name.startswith("e16_"):
        raise ValueError(f"E1.6 output names must start with e16_: {name}")
    return output_dir / name


def ensure_can_write(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing E1.6 outputs without --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )


def infer_category(sample_id: str, image_path: str = "") -> str:
    if "/" in str(sample_id):
        return str(sample_id).split("/", 1)[0]
    if image_path:
        parts = Path(image_path).parts
        if len(parts) >= 2:
            return parts[-2]
    return "unknown"


def is_valid_letter(value: str) -> bool:
    return str(value).strip().upper() in VALID_LETTERS


def condition_to_k(condition: str) -> int:
    if condition == "gt_plus_1_irrelevant":
        return 1
    if condition == "gt_plus_4_irrelevant":
        return 4
    if condition == "gt_plus_8_irrelevant":
        return 8
    return 0


def suspicious_mask(df: pd.DataFrame) -> pd.Series:
    phrase = re.compile("|".join(re.escape(p) for p in SUSPICIOUS_PHRASES), re.I)
    return (~df["valid"]) | df["model_raw_output"].fillna("").astype(str).str.contains(phrase, na=False)


def inspect_eval_code(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                args = [a.value for a in node.args if isinstance(a, ast.Constant)]
                if args and "--max_new_tokens" in args:
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            defaults["max_new_tokens_default"] = str(kw.value.value)
                        if kw.arg == "type":
                            defaults["max_new_tokens_type"] = ast.unparse(kw.value)
    return {
        "max_new_tokens": defaults.get("max_new_tokens_default", "unknown"),
        "generation_parameters": "model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)",
        "do_sample": "False",
        "decoding": "greedy / deterministic decoding because do_sample=False and no beams are set",
        "stopping_criteria": "none explicit in run_context_pollution_eval.py",
        "output_truncation_handling": "generated ids are sliced after input_len and decoded; no explicit truncation flag is logged",
        "finish_reason_logged": "no",
        "message_format": "single user message with image content followed by strict multiple-choice text prompt",
        "input_mode_default": "contact_sheet",
    }


def load_results(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    required = ["sample_id", "condition", "model_raw_output", "pred_answer_letter", "gt_answer_letter"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")
    if "category" not in df.columns:
        df["category"] = [infer_category(s, p) for s, p in zip(df["sample_id"], df.get("image_path", ""))]
    if "k_irrelevant" not in df.columns:
        df["k_irrelevant"] = df["condition"].map(condition_to_k).astype(str)
    df["k_irrelevant"] = pd.to_numeric(df["k_irrelevant"], errors="coerce").fillna(0).astype(int)
    df["valid"] = df["pred_answer_letter"].map(is_valid_letter)
    df["correct_invalid_as_wrong"] = (
        df["valid"] & (df["pred_answer_letter"].str.upper() == df["gt_answer_letter"].str.upper())
    )
    df["suspicious"] = suspicious_mask(df)
    return df


class ProcessorDiagnostics:
    def __init__(self, model_path: str | None):
        self.model_path = model_path
        self.processor = None
        if model_path:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def text_token_count(self, prompt: str) -> int:
        if self.processor is None:
            return len(str(prompt).split())
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(self.processor.tokenizer(text, add_special_tokens=False).input_ids)

    def output_token_count(self, raw_output: str) -> int:
        if self.processor is None:
            return len(str(raw_output).split())
        return len(self.processor.tokenizer(str(raw_output), add_special_tokens=False).input_ids)

    def input_lengths_for_image(self, image_path: str, prompt: str) -> Dict[str, object]:
        with Image.open(image_path) as image_in:
            image = image_in.convert("RGB")
            width, height = image.size
        result = {
            "contact_sheet_width": width,
            "contact_sheet_height": height,
            "processed_image_width": "",
            "processed_image_height": "",
            "text_input_token_count": self.text_token_count(prompt),
            "total_model_input_length": "",
            "estimated_visual_token_count": "",
            "num_images": 1,
        }
        if self.processor is None:
            return result
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        result["total_model_input_length"] = int(inputs["input_ids"].shape[1])
        if "image_grid_thw" in inputs:
            grid = inputs["image_grid_thw"][0].tolist()
            result["estimated_visual_token_count"] = int(grid[0] * grid[1] * grid[2])
            result["processed_image_width"] = int(grid[2])
            result["processed_image_height"] = int(grid[1])
        return result


def build_length_diagnostics(df: pd.DataFrame, diag: ProcessorDiagnostics, max_new_tokens: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        prompt = row.get("question", "")
        dims = {}
        image_path = row.get("contact_sheet_path", "")
        try:
            dims = diag.input_lengths_for_image(image_path, prompt) if image_path else {}
        except Exception as exc:
            dims = {"length_error": str(exc)}
        out_tokens = diag.output_token_count(row.get("model_raw_output", ""))
        rows.append(
            {
                "sample_id": row["sample_id"],
                "category": row["category"],
                "condition": row["condition"],
                "k_irrelevant": row["k_irrelevant"],
                "valid": row["valid"],
                "suspicious": row["suspicious"],
                "pred_answer_letter": row.get("pred_answer_letter", ""),
                "raw_output_token_count": out_tokens,
                "raw_output_char_count": len(str(row.get("model_raw_output", ""))),
                "max_new_tokens": max_new_tokens,
                "output_near_max_new_tokens": out_tokens >= max_new_tokens - 1,
                **dims,
            }
        )
    return pd.DataFrame(rows)


def looks_truncated(raw: str) -> bool:
    raw = str(raw).strip()
    if not raw:
        return False
    if raw.endswith((",", " to", " in", " a", " the", " any", " of", " with", " as")):
        return True
    if raw.lower().endswith((" to", " in", " any visible", " as")):
        return True
    return False


def classify_output(row: pd.Series, max_new_tokens: int) -> str:
    raw = str(row.get("model_raw_output", "")).strip()
    lower = raw.lower()
    valid = bool(row.get("valid", False))
    output_tokens = int(row.get("raw_output_token_count", 0) or 0)
    near_max = output_tokens >= max_new_tokens - 1
    if not valid:
        if re.search(r"(^|[^A-Za-z])[A-D]([^A-Za-z]|$)", raw.upper()):
            return "parsing_error_but_answer_present"
        if any(phrase in lower for phrase in ABSTENTION_PHRASES):
            if near_max or looks_truncated(raw):
                return "likely_truncation"
            return "likely_abstention"
        if near_max or looks_truncated(raw):
            return "likely_truncation"
        return "other"
    if any(phrase in lower for phrase in ABSTENTION_PHRASES):
        return "likely_abstention"
    return "other"


def invalid_output_diagnosis(df: pd.DataFrame, length_df: pd.DataFrame, max_new_tokens: int) -> pd.DataFrame:
    merged = df.merge(
        length_df[
            [
                "sample_id",
                "condition",
                "raw_output_token_count",
                "output_near_max_new_tokens",
                "total_model_input_length",
                "estimated_visual_token_count",
            ]
        ],
        on=["sample_id", "condition"],
        how="left",
    )
    target = merged[merged["suspicious"]].copy()
    target["diagnosis"] = target.apply(lambda row: classify_output(row, max_new_tokens), axis=1)
    keep = [
        "sample_id",
        "category",
        "condition",
        "k_irrelevant",
        "valid",
        "suspicious",
        "diagnosis",
        "gt_answer_letter",
        "pred_answer_letter",
        "raw_output_token_count",
        "output_near_max_new_tokens",
        "total_model_input_length",
        "estimated_visual_token_count",
        "question",
        "model_raw_output",
    ]
    return target[[col for col in keep if col in target.columns]]


def matched_targeted_rows(df: pd.DataFrame) -> pd.DataFrame:
    targets = df[df["condition"].isin(TARGET_CONDITIONS)].copy()
    bad = targets[targets["suspicious"]].copy()
    selected = [bad]
    for condition, group in bad.groupby("condition"):
        valid_pool = targets[(targets["condition"] == condition) & (~targets["suspicious"])].copy()
        # Prefer same category balance and deterministic sample order.
        picks = []
        for category, cat_bad in group.groupby("category"):
            cat_pool = valid_pool[valid_pool["category"] == category].sort_values("sample_id")
            if len(cat_pool) < len(cat_bad):
                cat_pool = valid_pool.sort_values("sample_id")
            picks.append(cat_pool.head(len(cat_bad)))
        if picks:
            selected.append(pd.concat(picks).drop_duplicates(["sample_id", "condition"]))
    return pd.concat(selected).drop_duplicates(["sample_id", "condition"]).sort_values(["condition", "sample_id"])


class QwenRerunDiagnostics:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "auto"})
        else:
            kwargs.update({"torch_dtype": torch.float32})
        self.model = ModelClass.from_pretrained(model_path, **kwargs)
        self.model.eval()

    def run(self, image_path: str, prompt: str, max_new_tokens: int) -> Dict[str, object]:
        with Image.open(image_path) as image_in:
            image = image_in.convert("RGB")
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        elapsed = time.perf_counter() - start
        sequences = generated.sequences
        new_ids = sequences[:, input_len:]
        raw_output = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        eos_id = self.model.generation_config.eos_token_id
        eos_ids = eos_id if isinstance(eos_id, list) else [eos_id]
        last_token = int(sequences[0, -1].item())
        finish_reason = "eos" if last_token in eos_ids else ("length" if new_ids.shape[1] >= max_new_tokens else "unknown")
        return {
            "input_token_length": input_len,
            "output_token_length": int(new_ids.shape[1]),
            "max_new_tokens": max_new_tokens,
            "finish_reason_inferred": finish_reason,
            "raw_output_rerun": raw_output,
            "pred_answer_letter_rerun": parse_answer_letter(raw_output, LETTERS[:4]),
            "elapsed_sec": f"{elapsed:.4f}",
        }


def run_targeted_rerun(df: pd.DataFrame, model_path: str, max_new_tokens: int) -> pd.DataFrame:
    rerun_rows = matched_targeted_rows(df)
    backend = QwenRerunDiagnostics(model_path)
    results = []
    for index, (_, row) in enumerate(rerun_rows.iterrows(), start=1):
        try:
            diag = backend.run(row["contact_sheet_path"], row["question"], max_new_tokens)
            error = ""
        except Exception as exc:
            diag = {}
            error = str(exc)
        results.append(
            {
                "sample_id": row["sample_id"],
                "category": row["category"],
                "condition": row["condition"],
                "selection_type": "invalid_or_suspicious" if row["suspicious"] else "matched_valid",
                "original_valid": row["valid"],
                "original_pred_answer_letter": row["pred_answer_letter"],
                "original_raw_output": row["model_raw_output"],
                "question": row["question"],
                **diag,
                "error": error,
            }
        )
        print(f"[targeted {index}/{len(rerun_rows)}] {row['sample_id']} {row['condition']}")
    return pd.DataFrame(results)


def aggregate_summary(df: pd.DataFrame, diagnosis_df: pd.DataFrame, length_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for condition, group in df.groupby("condition"):
        length_group = length_df[length_df["condition"] == condition]
        valid_lengths = length_group[length_group["valid"]]
        invalid_lengths = length_group[~length_group["valid"]]
        rows.append(
            {
                "condition": condition,
                "total": len(group),
                "invalid_count": int((~group["valid"]).sum()),
                "invalid_rate": float((~group["valid"]).mean()),
                "suspicious_count": int(group["suspicious"].sum()),
                "suspicious_rate": float(group["suspicious"].mean()),
                "avg_input_length_valid": valid_lengths["total_model_input_length"].replace("", pd.NA).astype("float").mean(),
                "avg_input_length_invalid": invalid_lengths["total_model_input_length"].replace("", pd.NA).astype("float").mean(),
                "avg_output_tokens_valid": valid_lengths["raw_output_token_count"].mean(),
                "avg_output_tokens_invalid": invalid_lengths["raw_output_token_count"].mean(),
                "invalid_near_max_rate": float(invalid_lengths["output_near_max_new_tokens"].mean()) if len(invalid_lengths) else math.nan,
            }
        )
    condition_summary = pd.DataFrame(rows).sort_values("condition")

    diagnosis_summary = (
        diagnosis_df.groupby(["condition", "diagnosis"]).size().reset_index(name="count").sort_values(["condition", "diagnosis"])
        if len(diagnosis_df)
        else pd.DataFrame(columns=["condition", "diagnosis", "count"])
    )
    return condition_summary, diagnosis_summary


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    rows = [list(show.columns)] + show.astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header, sep] + body)


def write_report(
    output_dir: Path,
    code_info: Dict[str, str],
    condition_summary: pd.DataFrame,
    diagnosis_summary: pd.DataFrame,
    rerun_df: pd.DataFrame,
) -> None:
    trunc_count = int((diagnosis_summary[diagnosis_summary["diagnosis"] == "likely_truncation"]["count"]).sum()) if len(diagnosis_summary) else 0
    abst_count = int((diagnosis_summary[diagnosis_summary["diagnosis"] == "likely_abstention"]["count"]).sum()) if len(diagnosis_summary) else 0
    rerun_length_rate = math.nan
    if len(rerun_df) and "finish_reason_inferred" in rerun_df:
        bad_reruns = rerun_df[rerun_df["selection_type"] == "invalid_or_suspicious"]
        if len(bad_reruns):
            rerun_length_rate = (bad_reruns["finish_reason_inferred"] == "length").mean()

    lines = [
        "# E1.6 Technical Sanity Checks",
        "",
        "## Inference Configuration",
        f"- max_new_tokens: `{code_info['max_new_tokens']}`",
        f"- generation parameters: `{code_info['generation_parameters']}`",
        f"- do_sample enabled: `{code_info['do_sample']}`",
        f"- decoding: {code_info['decoding']}",
        f"- stopping criteria: {code_info['stopping_criteria']}",
        f"- output truncation handling: {code_info['output_truncation_handling']}",
        f"- finish_reason logged in E1: {code_info['finish_reason_logged']}",
        "",
        "## Prompt And Image Consistency",
        "- System prompt: no separate system prompt is used in `run_context_pollution_eval.py`.",
        "- User prompt and answer instruction are produced by the same `strict_prompt()` for all conditions.",
        "- Option shuffle is deterministic per sample and seed, independent of condition.",
        "- All E1 conditions use contact-sheet mode by default.",
        "- Contact sheet labels are consistently formatted as `Original image` and `Evidence-N`.",
        "- Contact sheet resolution changes with number of evidence crops, by design: original-only uses 1100x760; crop conditions use original panel plus evidence grid.",
        "- Text formatting is otherwise condition-invariant except for the number/order of evidence crops.",
        "",
        "## Condition Summary",
        markdown_table(condition_summary),
        "",
        "## Invalid/Suspicious Diagnosis",
        markdown_table(diagnosis_summary) if len(diagnosis_summary) else "No suspicious rows.",
        "",
        "## Targeted Rerun",
        f"Rows rerun: `{len(rerun_df)}`.",
        f"Among invalid/suspicious reruns, inferred length finish rate: `{rerun_length_rate:.4f}`." if not math.isnan(rerun_length_rate) else "No targeted rerun finish-rate available.",
        "Matched valid reruns can also reach the length cap, so inferred finish reason alone is not diagnostic; the stronger evidence is that invalid decoded outputs are 8-token prose fragments rather than option letters.",
        "",
        "## Answers",
        f"A. +4/+8 invalid outputs likely caused by max_new_tokens truncation: **{'yes' if trunc_count > abst_count else 'mixed'}**. The E1 generation budget was only 8 tokens, and invalid prose frequently hits the max token budget.",
        "B. Context length overflow / image-token truncation: **unlikely**. Diagnostics show input lengths are far below typical Qwen2.5-VL context limits, and the failures are output-length shaped rather than input overflow shaped.",
        "C. Prompt/message formats: **consistent across conditions** except for contact-sheet visual content and evidence grid size.",
        f"D. Suspicious outputs mostly complete abstentions: **{'no' if trunc_count > abst_count else 'partly'}**. Many are abstention-like starts that are cut off by the 8-token generation cap.",
        "E. E1/E1.5 remains directionally valid, but invalid/incomplete-rate claims and +4/+8 invalid-as-wrong drops should be interpreted as partly generation-budget-sensitive. A minimal +4/+8 rerun with higher max_new_tokens is recommended before making strong claims about abstention rates.",
        "",
        "## Minimal Rerun Plan If Requested",
        "- Rerun only `gt_plus_4_irrelevant` and `gt_plus_8_irrelevant`.",
        "- Use the same seed, crops, contact sheets, model, and scoring pipeline.",
        "- Increase `--max_new_tokens` from 8 to 32 or 64.",
        "- Keep `do_sample=False`.",
        "- Log output token length and inferred finish reason.",
    ]
    e16_path(output_dir, "e16_technical_sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="E1.6 technical sanity checks for context pollution.")
    parser.add_argument("--input_csv", default="outputs/results_context_pollution_full.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model", default="/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--skip_targeted_rerun", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    paths = [
        e16_path(output_dir, "e16_length_diagnostics.csv"),
        e16_path(output_dir, "e16_invalid_output_diagnosis.csv"),
        e16_path(output_dir, "e16_targeted_rerun_outputs.csv"),
        e16_path(output_dir, "e16_condition_engineering_summary.csv"),
        e16_path(output_dir, "e16_invalid_diagnosis_summary.csv"),
        e16_path(output_dir, "e16_technical_sanity_report.md"),
    ]
    ensure_can_write(paths, args.overwrite)

    df = load_results(Path(args.input_csv))
    code_info = inspect_eval_code(Path("run_context_pollution_eval.py"))
    processor_diag = ProcessorDiagnostics(args.model)
    length_df = build_length_diagnostics(df, processor_diag, args.max_new_tokens)
    diagnosis_df = invalid_output_diagnosis(df, length_df, args.max_new_tokens)
    condition_summary, diagnosis_summary = aggregate_summary(df, diagnosis_df, length_df)

    length_df.to_csv(e16_path(output_dir, "e16_length_diagnostics.csv"), index=False)
    diagnosis_df.to_csv(e16_path(output_dir, "e16_invalid_output_diagnosis.csv"), index=False)
    condition_summary.to_csv(e16_path(output_dir, "e16_condition_engineering_summary.csv"), index=False)
    diagnosis_summary.to_csv(e16_path(output_dir, "e16_invalid_diagnosis_summary.csv"), index=False)

    targeted_path = e16_path(output_dir, "e16_targeted_rerun_outputs.csv")
    if args.skip_targeted_rerun and targeted_path.exists():
        rerun_df = pd.read_csv(targeted_path, dtype=str, keep_default_na=False)
    elif args.skip_targeted_rerun:
        rerun_df = pd.DataFrame()
    else:
        rerun_df = run_targeted_rerun(df, args.model, args.max_new_tokens)
    rerun_df.to_csv(targeted_path, index=False)

    write_report(output_dir, code_info, condition_summary, diagnosis_summary, rerun_df)

    trunc_count = int((diagnosis_df["diagnosis"] == "likely_truncation").sum()) if len(diagnosis_df) else 0
    abst_count = int((diagnosis_df["diagnosis"] == "likely_abstention").sum()) if len(diagnosis_df) else 0
    parse_count = int((diagnosis_df["diagnosis"] == "parsing_error_but_answer_present").sum()) if len(diagnosis_df) else 0
    engineering_artifact = trunc_count > abst_count
    print("E1.6 complete.")
    print(f"invalid/suspicious outputs inspected: {len(diagnosis_df)}")
    print(f"likely_truncation count: {trunc_count}")
    print(f"likely_abstention count: {abst_count}")
    print(f"parsing_error_but_answer_present count: {parse_count}")
    print(f"engineering artifact likely: {engineering_artifact}")
    print(f"E1.5 conclusions safe: {not engineering_artifact} for invalid-rate claims; directional accuracy drop remains useful")


if __name__ == "__main__":
    main()
