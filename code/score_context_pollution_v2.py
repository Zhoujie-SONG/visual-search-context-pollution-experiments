import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = {
    "original_only": -1,
    "gt_crop_only": 0,
    "gt_plus_1_irrelevant": 1,
    "gt_plus_4_irrelevant": 4,
    "gt_plus_8_irrelevant": 8,
}
TARGETS = ["gt_plus_1_irrelevant", "gt_plus_4_irrelevant", "gt_plus_8_irrelevant"]
BASELINE = "gt_crop_only"


def can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    df["k_irrelevant"] = pd.to_numeric(df["k_irrelevant"], errors="coerce").astype(int)
    df["valid_bool"] = df["valid"].astype(str).str.lower().eq("true") & df["pred_answer_letter"].isin(list("ABCD"))
    df["correct_bool"] = df["valid_bool"] & df["pred_answer_letter"].eq(df["gt_answer_letter"])
    df["input_token_count"] = pd.to_numeric(df["input_token_count"], errors="coerce")
    df["output_token_count"] = pd.to_numeric(df["output_token_count"], errors="coerce")
    return df


def per_seed_condition(df: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    sub = df.copy()
    if category is not None:
        sub = sub[sub["category"] == category]
    rows = []
    for (seed, condition), group in sub.groupby(["seed", "condition"]):
        valid = group[group["valid_bool"]]
        rows.append(
            {
                "seed": seed,
                "category": category or "ALL",
                "condition": condition,
                "k_irrelevant": int(group["k_irrelevant"].iloc[0]),
                "num_samples": len(group),
                "valid_samples": len(valid),
                "invalid_rate": 1.0 - len(valid) / len(group) if len(group) else math.nan,
                "accuracy_invalid_as_wrong": group["correct_bool"].mean() if len(group) else math.nan,
                "valid_only_accuracy": valid["correct_bool"].mean() if len(valid) else math.nan,
                "mean_input_tokens": group["input_token_count"].mean(),
                "mean_output_tokens": group["output_token_count"].mean(),
                "finish_length_rate": group["finish_reason"].eq("length").mean(),
                "finish_eos_rate": group["finish_reason"].eq("eos").mean(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_seed_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (category, condition), group in per_seed.groupby(["category", "condition"]):
        row = {
            "category": category,
            "condition": condition,
            "k_irrelevant": int(group["k_irrelevant"].iloc[0]),
            "num_samples_per_seed": int(group["num_samples"].iloc[0]),
            "num_seeds": group["seed"].nunique(),
        }
        for col in [
            "accuracy_invalid_as_wrong",
            "valid_only_accuracy",
            "invalid_rate",
            "mean_input_tokens",
            "mean_output_tokens",
            "finish_length_rate",
            "finish_eos_rate",
        ]:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["category", "k_irrelevant", "condition"])


def bootstrap(df: pd.DataFrame, mode: str, iters: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    unit = df.copy()
    unit["unit_id"] = unit["seed"].astype(str) + "::" + unit["sample_id"]
    for target in TARGETS:
        base = unit[unit["condition"] == BASELINE][["unit_id", "correct_bool", "valid_bool"]]
        targ = unit[unit["condition"] == target][["unit_id", "correct_bool", "valid_bool"]]
        merged = base.merge(targ, on="unit_id", suffixes=("_base", "_target"))
        if mode == "valid_only":
            merged = merged[merged["valid_bool_base"] & merged["valid_bool_target"]]
        b = merged["correct_bool_base"].astype(float).to_numpy()
        t = merged["correct_bool_target"].astype(float).to_numpy()
        n = len(merged)
        diffs = np.empty(iters)
        idx = np.arange(n)
        for i in range(iters):
            s = rng.choice(idx, size=n, replace=True)
            diffs[i] = b[s].mean() - t[s].mean()
        obs = b.mean() - t.mean()
        p_left = (np.sum(diffs <= 0) + 1) / (iters + 1)
        rows.append(
            {
                "comparison": f"{BASELINE}_vs_{target}",
                "mode": mode,
                "n_paired_seed_samples": n,
                "accuracy_gt_crop_only": b.mean(),
                "accuracy_target_condition": t.mean(),
                "mean_difference_gt_minus_target": obs,
                "ci95_low": np.quantile(diffs, 0.025),
                "ci95_high": np.quantile(diffs, 0.975),
                "p_one_sided_diff_gt_0": p_left,
                "p_two_sided_diff_ne_0": min(1.0, 2 * min(p_left, (np.sum(diffs >= 0) + 1) / (iters + 1))),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy(summary: pd.DataFrame, output_png: Path, metric_prefix: str, title: str) -> None:
    all_rows = summary[summary["category"].eq("ALL") & summary["condition"].ne("original_only")].sort_values("k_irrelevant")
    x = all_rows["k_irrelevant"].to_numpy()
    y = all_rows[f"{metric_prefix}_mean"].to_numpy()
    err = all_rows[f"{metric_prefix}_std"].fillna(0).to_numpy()
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(x, y, yerr=err, marker="o", capsize=4, label=metric_prefix)
    orig = summary[summary["category"].eq("ALL") & summary["condition"].eq("original_only")]
    if not orig.empty:
        plt.axhline(orig[f"{metric_prefix}_mean"].iloc[0], color="gray", linestyle="--", label="original_only")
    plt.ylim(0, 1.05)
    plt.xlabel("Number of irrelevant crops")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def write_report(summary: pd.DataFrame, by_category: pd.DataFrame, bootstrap_df: pd.DataFrame, old_summary: Path, output: Path) -> None:
    overall = summary[summary["category"].eq("ALL")]
    old_text = "旧 contact-sheet 结果未找到。"
    if old_summary.exists():
        old = pd.read_csv(old_summary)
        old_gt = float(old[old["condition"].eq("gt_crop_only")]["accuracy"].iloc[0])
        old_k8 = float(old[old["condition"].eq("gt_plus_8_irrelevant")]["accuracy"].iloc[0])
        new_gt = float(overall[overall["condition"].eq("gt_crop_only")]["accuracy_invalid_as_wrong_mean"].iloc[0])
        new_k8 = float(overall[overall["condition"].eq("gt_plus_8_irrelevant")]["accuracy_invalid_as_wrong_mean"].iloc[0])
        old_text = (
            f"旧 contact-sheet 的 GT-only 到 +8 drop 为 {old_gt - old_k8:.4f}；"
            f"v2 separate-image 的 invalid-as-wrong drop 为 {new_gt - new_k8:.4f}。"
        )
    lines = [
        "# E1-v2 Corrected Context-Pollution Report",
        "",
        "## Overall",
        overall.to_csv(index=False),
        "## By Category",
        by_category.to_csv(index=False),
        "## Bootstrap",
        bootstrap_df.to_csv(index=False),
        "## Contact-Sheet Comparison",
        old_text,
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score E1-v2 context pollution results.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--by_category_csv", required=True)
    parser.add_argument("--bootstrap_csv", required=True)
    parser.add_argument("--plot_invalid_as_wrong_png", required=True)
    parser.add_argument("--plot_valid_only_png", required=True)
    parser.add_argument("--report_md", required=True)
    parser.add_argument("--old_summary_csv", default="outputs/summary_context_pollution_full.csv")
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = [
        Path(args.summary_csv),
        Path(args.by_category_csv),
        Path(args.bootstrap_csv),
        Path(args.plot_invalid_as_wrong_png),
        Path(args.plot_valid_only_png),
        Path(args.report_md),
    ]
    for path in paths:
        can_write(path, args.overwrite)
    df = load_results(Path(args.input_csv))
    parts = [per_seed_condition(df)]
    for category in sorted(df["category"].unique()):
        parts.append(per_seed_condition(df, category=category))
    per_seed = pd.concat(parts, ignore_index=True)
    summary = aggregate_seed_stats(per_seed)
    summary.to_csv(args.summary_csv, index=False)
    summary[summary["category"].ne("ALL")].to_csv(args.by_category_csv, index=False)
    boot = pd.concat(
        [bootstrap(df, "invalid_as_wrong", args.bootstrap_iters, args.seed), bootstrap(df, "valid_only", args.bootstrap_iters, args.seed)],
        ignore_index=True,
    )
    boot.to_csv(args.bootstrap_csv, index=False)
    plot_accuracy(summary, Path(args.plot_invalid_as_wrong_png), "accuracy_invalid_as_wrong", "E1-v2 invalid-as-wrong accuracy")
    plot_accuracy(summary, Path(args.plot_valid_only_png), "valid_only_accuracy", "E1-v2 valid-only accuracy")
    write_report(summary, summary[summary["category"].ne("ALL")], boot, Path(args.old_summary_csv), Path(args.report_md))

    print("E1-v2 scoring complete.")
    print(summary[summary["category"].eq("ALL")][["condition", "accuracy_invalid_as_wrong_mean", "accuracy_invalid_as_wrong_std", "valid_only_accuracy_mean", "valid_only_accuracy_std", "invalid_rate_mean", "mean_input_tokens_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
