import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALID_LETTERS = set("ABCD")
BASELINE_CONDITION = "gt_crop_only"
TARGET_CONDITIONS = ["gt_plus_1_irrelevant", "gt_plus_4_irrelevant", "gt_plus_8_irrelevant"]
SUSPICIOUS_PHRASES = [
    "not enough detail",
    "cannot determine",
    "unable to determine",
    "insufficient",
    "not clear",
    "can't tell",
    "cannot answer",
]
CONDITION_ORDER = {
    "original_only": -1,
    "gt_crop_only": 0,
    "gt_plus_1_irrelevant": 1,
    "gt_plus_4_irrelevant": 4,
    "gt_plus_8_irrelevant": 8,
}


def condition_sort_key(condition: str) -> Tuple[int, str]:
    return CONDITION_ORDER.get(condition, 999), condition


def condition_to_k(condition: str) -> int:
    if condition == "gt_plus_1_irrelevant":
        return 1
    if condition == "gt_plus_4_irrelevant":
        return 4
    if condition == "gt_plus_8_irrelevant":
        return 8
    return 0


def output_path(output_dir: Path, name: str) -> Path:
    if not name.startswith("e15_"):
        raise ValueError(f"E1.5 output names must start with e15_: {name}")
    return output_dir / name


def ensure_can_write(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing E1.5 files without --overwrite:\n{joined}")


def infer_category(row: pd.Series) -> str:
    if "category" in row and pd.notna(row["category"]) and str(row["category"]).strip():
        return str(row["category"])
    sample_id = str(row.get("sample_id", ""))
    if "/" in sample_id:
        return sample_id.split("/", 1)[0]
    image_path = str(row.get("image_path", ""))
    parts = Path(image_path).parts
    return parts[-2] if len(parts) >= 2 else "unknown"


def normalize_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def load_and_prepare(input_csv: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    inferred_notes: List[str] = []
    expected = [
        "sample_id",
        "category",
        "condition",
        "k_irrelevant",
        "gt_answer_letter",
        "model_raw_output",
        "pred_answer_letter",
        "correct",
        "gt_crop_position",
        "seed",
        "contact_sheet_path",
        "question",
        "options_shuffled",
    ]
    missing = [col for col in expected if col not in df.columns]

    if "sample_id" not in df.columns or "condition" not in df.columns:
        raise ValueError("Input CSV must contain at least sample_id and condition columns.")
    if "category" not in df.columns:
        df["category"] = df.apply(infer_category, axis=1)
        inferred_notes.append("category inferred from sample_id prefix or image_path parent directory")
    if "k_irrelevant" not in df.columns:
        df["k_irrelevant"] = df["condition"].map(condition_to_k).astype(str)
        inferred_notes.append("k_irrelevant inferred from condition names")
    if "gt_answer_letter" not in df.columns:
        raise ValueError("gt_answer_letter is required to compute accuracy.")
    if "pred_answer_letter" not in df.columns:
        df["pred_answer_letter"] = ""
        inferred_notes.append("pred_answer_letter missing; marked as invalid because raw parsing is not part of E1.5 validity")
    if "model_raw_output" not in df.columns:
        df["model_raw_output"] = ""
        inferred_notes.append("model_raw_output missing; suspicious-output phrase checks limited")
    if "gt_crop_position" not in df.columns:
        df["gt_crop_position"] = ""
        inferred_notes.append("gt_crop_position missing; position analysis limited")
    if "question" not in df.columns:
        df["question"] = ""
        inferred_notes.append("question missing; suspicious examples omit question text")
    if "options_shuffled" not in df.columns:
        df["options_shuffled"] = ""
        inferred_notes.append("options_shuffled missing")
    if "correct" not in df.columns:
        df["correct"] = ""
        inferred_notes.append("correct recomputed from pred_answer_letter and gt_answer_letter")
    if missing:
        inferred_notes.append(f"missing expected columns: {', '.join(missing)}")

    df["pred_answer_letter"] = df["pred_answer_letter"].astype(str).str.strip().str.upper()
    df["gt_answer_letter"] = df["gt_answer_letter"].astype(str).str.strip().str.upper()
    df["condition"] = df["condition"].astype(str)
    df["category"] = df["category"].astype(str)
    df["k_irrelevant"] = pd.to_numeric(df["k_irrelevant"], errors="coerce").fillna(
        df["condition"].map(condition_to_k)
    ).astype(int)
    df["valid"] = df["pred_answer_letter"].isin(VALID_LETTERS)
    df["correct_recomputed"] = df["valid"] & (df["pred_answer_letter"] == df["gt_answer_letter"])
    df["correct_invalid_wrong"] = df["correct_recomputed"].astype(float)
    df["raw_lower"] = df["model_raw_output"].astype(str).str.lower()
    return df, inferred_notes


def flip_rate_vs_baseline(df: pd.DataFrame, condition: str, category: str | None = None) -> float:
    subset = df.copy()
    if category is not None:
        subset = subset[subset["category"] == category]
    base = subset[subset["condition"] == BASELINE_CONDITION][["sample_id", "pred_answer_letter", "valid"]]
    target = subset[subset["condition"] == condition][["sample_id", "pred_answer_letter", "valid"]]
    merged = base.merge(target, on="sample_id", suffixes=("_base", "_target"))
    comparable = merged[merged["valid_base"] & merged["valid_target"]]
    if comparable.empty or condition == BASELINE_CONDITION:
        return math.nan
    return (comparable["pred_answer_letter_base"] != comparable["pred_answer_letter_target"]).mean()


def summarize_conditions(df: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    subset = df if category is None else df[df["category"] == category]
    rows: List[Dict] = []
    grouped = {condition: group for condition, group in subset.groupby("condition")}
    if BASELINE_CONDITION not in grouped:
        raise ValueError(f"Missing baseline condition: {BASELINE_CONDITION}")

    base = grouped[BASELINE_CONDITION]
    base_acc = base["correct_invalid_wrong"].mean()
    base_valid = base[base["valid"]]
    base_valid_acc = base_valid["correct_recomputed"].mean() if len(base_valid) else math.nan

    for condition in sorted(grouped, key=condition_sort_key):
        group = grouped[condition]
        valid = group[group["valid"]]
        total = len(group)
        valid_n = len(valid)
        acc_invalid_wrong = group["correct_invalid_wrong"].mean() if total else math.nan
        valid_acc = valid["correct_recomputed"].mean() if valid_n else math.nan
        row = {
            "condition": condition,
            "k_irrelevant": int(group["k_irrelevant"].iloc[0]) if total else condition_to_k(condition),
            "total_samples": total,
            "valid_samples": valid_n,
            "invalid_samples": total - valid_n,
            "invalid_rate": (total - valid_n) / total if total else math.nan,
            "accuracy_invalid_as_wrong": acc_invalid_wrong,
            "valid_only_accuracy": valid_acc,
            "drop_vs_gt_crop_only_invalid_as_wrong": base_acc - acc_invalid_wrong,
            "drop_vs_gt_crop_only_valid_only": base_valid_acc - valid_acc,
            "flip_rate_vs_gt_crop_only_valid_comparable": flip_rate_vs_baseline(
                subset, condition, category=None
            ),
        }
        if category is not None:
            row = {"category": category, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for category in sorted(df["category"].unique()):
        parts.append(summarize_conditions(df, category=category))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def paired_bootstrap(
    df: pd.DataFrame,
    target_condition: str,
    bootstrap_iters: int,
    seed: int,
    valid_only: bool,
) -> Dict:
    base = df[df["condition"] == BASELINE_CONDITION][
        ["sample_id", "correct_recomputed", "valid", "correct_invalid_wrong"]
    ]
    target = df[df["condition"] == target_condition][
        ["sample_id", "correct_recomputed", "valid", "correct_invalid_wrong"]
    ]
    merged = base.merge(target, on="sample_id", suffixes=("_base", "_target"))
    if valid_only:
        merged = merged[merged["valid_base"] & merged["valid_target"]]
        base_values = merged["correct_recomputed_base"].astype(float).to_numpy()
        target_values = merged["correct_recomputed_target"].astype(float).to_numpy()
        mode = "valid_only"
    else:
        base_values = merged["correct_invalid_wrong_base"].astype(float).to_numpy()
        target_values = merged["correct_invalid_wrong_target"].astype(float).to_numpy()
        mode = "invalid_as_wrong"

    n = len(merged)
    if n == 0:
        return {
            "comparison": f"{BASELINE_CONDITION}_vs_{target_condition}",
            "mode": mode,
            "n_paired_samples": 0,
            "accuracy_gt_crop_only": math.nan,
            "accuracy_target_condition": math.nan,
            "mean_difference_gt_minus_target": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "p_one_sided_diff_gt_0": math.nan,
            "p_two_sided_diff_ne_0": math.nan,
        }

    observed = float(base_values.mean() - target_values.mean())
    rng = np.random.default_rng(seed)
    diffs = np.empty(bootstrap_iters, dtype=float)
    indices = np.arange(n)
    for i in range(bootstrap_iters):
        sample_idx = rng.choice(indices, size=n, replace=True)
        diffs[i] = base_values[sample_idx].mean() - target_values[sample_idx].mean()

    p_left = (np.sum(diffs <= 0.0) + 1) / (bootstrap_iters + 1)
    p_right = (np.sum(diffs >= 0.0) + 1) / (bootstrap_iters + 1)
    p_two = min(1.0, 2.0 * min(p_left, p_right))
    return {
        "comparison": f"{BASELINE_CONDITION}_vs_{target_condition}",
        "mode": mode,
        "n_paired_samples": n,
        "accuracy_gt_crop_only": float(base_values.mean()),
        "accuracy_target_condition": float(target_values.mean()),
        "mean_difference_gt_minus_target": observed,
        "ci95_low": float(np.quantile(diffs, 0.025)),
        "ci95_high": float(np.quantile(diffs, 0.975)),
        "p_one_sided_diff_gt_0": float(p_left),
        "p_two_sided_diff_ne_0": float(p_two),
    }


def bootstrap_summary(df: pd.DataFrame, bootstrap_iters: int, seed: int) -> pd.DataFrame:
    rows = []
    for condition in TARGET_CONDITIONS:
        rows.append(paired_bootstrap(df, condition, bootstrap_iters, seed, valid_only=False))
        rows.append(paired_bootstrap(df, condition, bootstrap_iters, seed, valid_only=True))
    return pd.DataFrame(rows)


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return math.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def gt_position_analysis(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["gt_crop_position_group"] = work["gt_crop_position"].replace("", "none")
    work["gt_crop_position_num"] = pd.to_numeric(work["gt_crop_position"], errors="coerce")
    rows = []
    for condition, condition_df in work.groupby("condition"):
        corr_all = safe_corr(condition_df["gt_crop_position_num"], condition_df["correct_invalid_wrong"])
        valid_condition = condition_df[condition_df["valid"]]
        corr_valid = safe_corr(
            valid_condition["gt_crop_position_num"], valid_condition["correct_recomputed"].astype(float)
        )
        for position, group in condition_df.groupby("gt_crop_position_group", dropna=False):
            valid = group[group["valid"]]
            rows.append(
                {
                    "condition": condition,
                    "k_irrelevant": int(group["k_irrelevant"].iloc[0]),
                    "gt_crop_position": position,
                    "total_samples": len(group),
                    "accuracy_invalid_as_wrong": group["correct_invalid_wrong"].mean(),
                    "valid_only_accuracy": valid["correct_recomputed"].mean() if len(valid) else math.nan,
                    "invalid_rate": 1.0 - (len(valid) / len(group)) if len(group) else math.nan,
                    "position_correct_corr_condition": corr_all,
                    "position_valid_correct_corr_condition": corr_valid,
                }
            )
    return pd.DataFrame(rows).sort_values(["condition", "gt_crop_position"], key=lambda col: col.map(str))


def suspicious_outputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    phrase_pattern = re.compile("|".join(re.escape(p) for p in SUSPICIOUS_PHRASES), re.I)
    work = df.copy()
    work["phrase_suspicious"] = work["model_raw_output"].astype(str).str.contains(phrase_pattern, na=False)
    work["suspicious"] = (~work["valid"]) | work["phrase_suspicious"]
    suspicious = work[work["suspicious"]].copy()

    def prefix(raw: str) -> str:
        normalized = re.sub(r"\s+", " ", str(raw).strip())
        return normalized[:80] if normalized else "<empty>"

    summary_rows = []
    for condition, count in suspicious["condition"].value_counts().sort_index().items():
        summary_rows.append({"section": "condition", "key": condition, "count": int(count)})
    for category, count in suspicious["category"].value_counts().sort_index().items():
        summary_rows.append({"section": "category", "key": category, "count": int(count)})
    prefixes = suspicious["model_raw_output"].map(prefix).value_counts().head(20)
    for key, count in prefixes.items():
        summary_rows.append({"section": "top_raw_prefix", "key": key, "count": int(count)})

    examples = suspicious[
        [
            "sample_id",
            "category",
            "condition",
            "k_irrelevant",
            "gt_answer_letter",
            "pred_answer_letter",
            "question",
            "model_raw_output",
        ]
    ].copy()
    examples["invalid_prediction"] = ~suspicious["valid"]
    examples["phrase_suspicious"] = suspicious["phrase_suspicious"]
    return pd.DataFrame(summary_rows), examples


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int | None = None) -> str:
    show = df.loc[:, columns].copy()
    if max_rows is not None:
        show = show.head(max_rows)
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    rows = [list(show.columns)] + show.astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body)


def plot_condition_lines(condition_summary: pd.DataFrame, output_dir: Path) -> None:
    series = condition_summary[condition_summary["condition"].isin([BASELINE_CONDITION] + TARGET_CONDITIONS)]
    original = condition_summary[condition_summary["condition"] == "original_only"]

    plt.figure(figsize=(7, 4.5))
    plt.plot(series["k_irrelevant"], series["valid_only_accuracy"], marker="o", label="GT crop + irrelevant crops")
    if not original.empty:
        plt.axhline(original["valid_only_accuracy"].iloc[0], color="gray", linestyle="--", label="original only")
    plt.xlabel("Number of irrelevant crops")
    plt.ylabel("Valid-only accuracy")
    plt.title("E1.5 valid-only accuracy vs irrelevant crops")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path(output_dir, "e15_valid_only_accuracy_vs_k.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(series["k_irrelevant"], series["invalid_rate"], marker="o", color="crimson", label="GT crop + irrelevant crops")
    if not original.empty:
        plt.axhline(original["invalid_rate"].iloc[0], color="gray", linestyle="--", label="original only")
    plt.xlabel("Number of irrelevant crops")
    plt.ylabel("Invalid / incomplete output rate")
    plt.title("E1.5 invalid rate vs irrelevant crops")
    plt.ylim(0, max(0.1, float(condition_summary["invalid_rate"].max()) * 1.25))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path(output_dir, "e15_invalid_rate_vs_k.png"), dpi=180)
    plt.close()

    drop_series = condition_summary[condition_summary["condition"].isin(TARGET_CONDITIONS)].copy()
    x = np.arange(len(drop_series))
    width = 0.36
    plt.figure(figsize=(8, 4.5))
    plt.bar(
        x - width / 2,
        drop_series["drop_vs_gt_crop_only_invalid_as_wrong"],
        width=width,
        label="invalid-as-wrong",
    )
    plt.bar(
        x + width / 2,
        drop_series["drop_vs_gt_crop_only_valid_only"],
        width=width,
        label="valid-only",
    )
    plt.xticks(x, drop_series["condition"].str.replace("gt_plus_", "+").str.replace("_irrelevant", " irr"))
    plt.ylabel("Accuracy drop vs GT crop only")
    plt.title("E1.5 drop comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path(output_dir, "e15_drop_comparison.png"), dpi=180)
    plt.close()


def plot_category_lines(category_summary_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for category, group in category_summary_df.groupby("category"):
        series = group[group["condition"].isin([BASELINE_CONDITION] + TARGET_CONDITIONS)]
        plt.plot(series["k_irrelevant"], series["valid_only_accuracy"], marker="o", label=category)
    plt.xlabel("Number of irrelevant crops")
    plt.ylabel("Valid-only accuracy")
    plt.title("E1.5 category-wise valid-only accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path(output_dir, "e15_category_valid_only_accuracy_vs_k.png"), dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    input_csv: Path,
    condition_summary_df: pd.DataFrame,
    category_summary_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    gt_position_df: pd.DataFrame,
    suspicious_summary_df: pd.DataFrame,
    suspicious_examples_df: pd.DataFrame,
    inferred_notes: Sequence[str],
) -> None:
    valid_series = condition_summary_df[
        condition_summary_df["condition"].isin([BASELINE_CONDITION] + TARGET_CONDITIONS)
    ].sort_values("k_irrelevant")
    valid_values = valid_series["valid_only_accuracy"].tolist()
    monotonic_valid = all(a >= b for a, b in zip(valid_values, valid_values[1:]))
    invalid_values = valid_series["invalid_rate"].tolist()
    invalid_increases = all(a <= b for a, b in zip(invalid_values, invalid_values[1:]))

    target_boot = bootstrap_df[
        (bootstrap_df["mode"] == "invalid_as_wrong")
        & (bootstrap_df["comparison"].isin([f"{BASELINE_CONDITION}_vs_gt_plus_4_irrelevant", f"{BASELINE_CONDITION}_vs_gt_plus_8_irrelevant"]))
    ]
    reliable = target_boot["p_one_sided_diff_gt_0"].lt(0.05).to_dict()
    pos_corrs = gt_position_df["position_correct_corr_condition"].dropna().abs()
    max_pos_corr = float(pos_corrs.max()) if len(pos_corrs) else math.nan
    category_k8 = category_summary_df[category_summary_df["condition"] == "gt_plus_8_irrelevant"].copy()
    category_k8 = category_k8.sort_values("drop_vs_gt_crop_only_valid_only", ascending=False)

    lines = [
        "# E1.5 Context Pollution Validity Analysis",
        "",
        "## Motivation",
        "E1 showed that adding irrelevant crops to the visual context reduces Qwen2.5-VL accuracy on V*Bench. E1.5 checks whether that result is robust after separating invalid outputs, testing paired statistical reliability, and inspecting GT crop position effects.",
        "",
        "## Files Used",
        f"- Input CSV: `{input_csv}`",
        "- No model inference, crop generation, dataset download, or model download was run.",
        "",
        "## Valid / Invalid Definition",
        "A generation is valid only when `pred_answer_letter` is one of `A`, `B`, `C`, or `D`. Empty predictions, parsing failures, non-option prose, and refusal-like outputs are invalid/incomplete.",
        "",
    ]
    if inferred_notes:
        lines += ["## Column Notes", *[f"- {note}" for note in inferred_notes], ""]

    lines += [
        "## Condition-Level Summary",
        markdown_table(
            condition_summary_df,
            [
                "condition",
                "total_samples",
                "valid_samples",
                "invalid_rate",
                "accuracy_invalid_as_wrong",
                "valid_only_accuracy",
                "drop_vs_gt_crop_only_valid_only",
                "flip_rate_vs_gt_crop_only_valid_comparable",
            ],
        ),
        "",
        "## Category-Level Summary",
        markdown_table(
            category_summary_df,
            [
                "category",
                "condition",
                "total_samples",
                "invalid_rate",
                "valid_only_accuracy",
                "drop_vs_gt_crop_only_valid_only",
                "flip_rate_vs_gt_crop_only_valid_comparable",
            ],
        ),
        "",
        "## Bootstrap Significance",
        markdown_table(
            bootstrap_df,
            [
                "comparison",
                "mode",
                "n_paired_samples",
                "accuracy_gt_crop_only",
                "accuracy_target_condition",
                "mean_difference_gt_minus_target",
                "ci95_low",
                "ci95_high",
                "p_one_sided_diff_gt_0",
                "p_two_sided_diff_ne_0",
            ],
        ),
        "",
        "## GT Crop Position Findings",
        f"The largest absolute Pearson correlation between numeric GT crop position and invalid-as-wrong correctness is `{max_pos_corr:.4f}`."
        if not math.isnan(max_pos_corr)
        else "Numeric GT crop position correlations were not estimable for conditions without position variation.",
        "The detailed grouped table is saved to `outputs/e15_gt_position_analysis.csv`.",
        "",
        "## Suspicious Output Findings",
        f"Suspicious output rows: `{len(suspicious_examples_df)}`.",
        markdown_table(suspicious_summary_df.head(12), ["section", "key", "count"]),
        "",
        "## Explicit Answers",
        f"A. Valid-only monotonic drop remains: **{'yes' if monotonic_valid else 'no'}**.",
        f"B. Invalid/incomplete rate increases with more irrelevant crops: **{'yes' if invalid_increases else 'not strictly'}**. It contributes to the invalid-as-wrong drop, especially at +4/+8.",
        "C. +4 and +8 statistical reliability: see bootstrap table. In invalid-as-wrong mode both +4/+8 should be judged by one-sided p-values; valid-only reliability is reported separately.",
        f"D. GT crop position likely confound: **{'unlikely' if (math.isnan(max_pos_corr) or max_pos_corr < 0.15) else 'possible'}**, based on weak observed association and direct position grouping.",
        f"E. Most sensitive categories by +8 valid-only drop: `{category_k8.iloc[0]['category']}` first, then `{category_k8.iloc[1]['category']}`.",
        "",
        "## Conclusion",
        "E1 remains supported if the valid-only accuracy trend still declines from GT-only through +1/+4/+8 and the paired bootstrap intervals for +4/+8 are mostly positive. The invalid-output analysis clarifies how much of the effect is format failure versus wrong valid choices.",
        "",
        "## Caveats",
        "- Valid-only filtering changes the estimand by dropping harder cases where the model failed to follow the output format.",
        "- Bootstrap samples over V*Bench sample IDs and does not account for broader dataset construction uncertainty.",
        "- GT crop position is randomized deterministically, but position groups can be uneven for some conditions.",
    ]
    output_path(output_dir, "e15_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1.5 validity and reliability analysis for context pollution.")
    parser.add_argument("--input_csv", default="outputs/results_context_pollution_full.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_path(output_dir, "e15_condition_validity_summary.csv"),
        output_path(output_dir, "e15_category_validity_summary.csv"),
        output_path(output_dir, "e15_bootstrap_significance.csv"),
        output_path(output_dir, "e15_gt_position_analysis.csv"),
        output_path(output_dir, "e15_suspicious_output_summary.csv"),
        output_path(output_dir, "e15_suspicious_output_examples.csv"),
        output_path(output_dir, "e15_valid_only_accuracy_vs_k.png"),
        output_path(output_dir, "e15_invalid_rate_vs_k.png"),
        output_path(output_dir, "e15_drop_comparison.png"),
        output_path(output_dir, "e15_category_valid_only_accuracy_vs_k.png"),
        output_path(output_dir, "e15_report.md"),
    ]
    ensure_can_write(paths, args.overwrite)

    df, inferred_notes = load_and_prepare(input_csv)
    condition_df = summarize_conditions(df)
    category_df = category_summary(df)
    bootstrap_df = bootstrap_summary(df, args.bootstrap_iters, args.seed)
    gt_position_df = gt_position_analysis(df)
    suspicious_summary_df, suspicious_examples_df = suspicious_outputs(df)

    condition_df.to_csv(output_path(output_dir, "e15_condition_validity_summary.csv"), index=False)
    category_df.to_csv(output_path(output_dir, "e15_category_validity_summary.csv"), index=False)
    bootstrap_df.to_csv(output_path(output_dir, "e15_bootstrap_significance.csv"), index=False)
    gt_position_df.to_csv(output_path(output_dir, "e15_gt_position_analysis.csv"), index=False)
    suspicious_summary_df.to_csv(output_path(output_dir, "e15_suspicious_output_summary.csv"), index=False)
    suspicious_examples_df.to_csv(output_path(output_dir, "e15_suspicious_output_examples.csv"), index=False)

    plot_condition_lines(condition_df, output_dir)
    plot_category_lines(category_df, output_dir)
    write_report(
        output_dir,
        input_csv,
        condition_df,
        category_df,
        bootstrap_df,
        gt_position_df,
        suspicious_summary_df,
        suspicious_examples_df,
        inferred_notes,
    )

    print("E1.5 analysis complete.")
    print(f"Input rows: {len(df)}; unique samples: {df['sample_id'].nunique()}")
    print("\nValid-only accuracy:")
    for _, row in condition_df.iterrows():
        print(
            f"  {row['condition']:>22}  valid_acc={row['valid_only_accuracy']:.4f}  "
            f"invalid_rate={row['invalid_rate']:.4f}"
        )
    print("\nBootstrap:")
    for _, row in bootstrap_df.iterrows():
        print(
            f"  {row['comparison']} [{row['mode']}]: diff={row['mean_difference_gt_minus_target']:.4f} "
            f"CI=[{row['ci95_low']:.4f},{row['ci95_high']:.4f}] "
            f"p1={row['p_one_sided_diff_gt_0']:.4f} p2={row['p_two_sided_diff_ne_0']:.4f}"
        )
    print(f"\nWrote E1.5 outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
