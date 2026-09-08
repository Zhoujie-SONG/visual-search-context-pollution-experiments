import argparse
import math
from pathlib import Path

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


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    df["k_irrelevant"] = pd.to_numeric(df["k_irrelevant"], errors="coerce").astype(int)
    df["is_oom"] = df["status"].eq("oom")
    df["is_invalid"] = df["status"].eq("invalid")
    df["is_preprocess_error"] = df["is_invalid"] & df["model_raw_output"].eq("") & df["error_message"].ne("")
    df["is_model_invalid"] = df["is_invalid"] & ~df["is_preprocess_error"]
    df["valid_bool"] = bool_series(df["valid_output"]) & df["pred_answer_letter"].isin(list("ABCD")) & ~df["is_oom"]
    df["correct_bool"] = bool_series(df["correct"]) & ~df["is_oom"]
    df["correct_primary"] = df["correct_bool"].astype(float)
    df.loc[df["is_oom"], "correct_primary"] = np.nan
    df["correct_oom_as_wrong"] = df["correct_bool"].astype(float)
    for col in [
        "input_token_count",
        "estimated_visual_token_count",
        "output_token_count",
        "distractor_visual_token_fraction",
        "generation_time_sec",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def per_seed_condition(df: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    sub = df.copy()
    if category is not None:
        sub = sub[sub["category"] == category]
    rows = []
    for (seed, condition), group in sub.groupby(["seed", "condition"], sort=False):
        non_oom = group[~group["is_oom"]]
        generated = non_oom[~non_oom["is_preprocess_error"]]
        valid = non_oom[non_oom["valid_bool"]]
        rows.append(
            {
                "seed": seed,
                "category": category or "ALL",
                "condition": condition,
                "k_irrelevant": int(group["k_irrelevant"].iloc[0]),
                "num_rows": len(group),
                "num_non_oom": len(non_oom),
                "oom_count": int(group["is_oom"].sum()),
                "oom_rate": float(group["is_oom"].mean()) if len(group) else math.nan,
                "invalid_count_excluding_oom": int(non_oom["is_invalid"].sum()),
                "invalid_rate_excluding_oom": float(non_oom["is_invalid"].mean()) if len(non_oom) else math.nan,
                "preprocess_error_count_excluding_oom": int(non_oom["is_preprocess_error"].sum()),
                "preprocess_error_rate_excluding_oom": float(non_oom["is_preprocess_error"].mean()) if len(non_oom) else math.nan,
                "model_invalid_count_excluding_oom_preprocess": int(generated["is_model_invalid"].sum()),
                "model_invalid_rate_excluding_oom_preprocess": float(generated["is_model_invalid"].mean()) if len(generated) else math.nan,
                "accuracy_primary_excluding_oom": float(non_oom["correct_bool"].mean()) if len(non_oom) else math.nan,
                "accuracy_generation_only_excluding_oom_preprocess": float(generated["correct_bool"].mean()) if len(generated) else math.nan,
                "accuracy_oom_as_wrong": float(group["correct_oom_as_wrong"].mean()) if len(group) else math.nan,
                "valid_only_accuracy_excluding_oom": float(valid["correct_bool"].mean()) if len(valid) else math.nan,
                "mean_input_tokens": group["input_token_count"].mean(),
                "mean_visual_tokens": group["estimated_visual_token_count"].mean(),
                "mean_distractor_token_fraction": group["distractor_visual_token_fraction"].mean(),
                "mean_output_tokens": group["output_token_count"].mean(),
                "finish_length_rate": group["finish_reason"].eq("length").mean(),
                "finish_eos_rate": group["finish_reason"].eq("eos").mean(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_seed_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "oom_rate",
        "invalid_rate_excluding_oom",
        "preprocess_error_rate_excluding_oom",
        "model_invalid_rate_excluding_oom_preprocess",
        "accuracy_primary_excluding_oom",
        "accuracy_generation_only_excluding_oom_preprocess",
        "accuracy_oom_as_wrong",
        "valid_only_accuracy_excluding_oom",
        "mean_input_tokens",
        "mean_visual_tokens",
        "mean_distractor_token_fraction",
        "mean_output_tokens",
        "finish_length_rate",
        "finish_eos_rate",
    ]
    for (category, condition), group in per_seed.groupby(["category", "condition"], sort=False):
        row = {
            "category": category,
            "condition": condition,
            "k_irrelevant": int(group["k_irrelevant"].iloc[0]),
            "num_rows_per_seed_mean": group["num_rows"].mean(),
            "num_non_oom_per_seed_mean": group["num_non_oom"].mean(),
            "num_seeds": group["seed"].nunique(),
        }
        for col in metrics:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_order"] = out["condition"].map(ORDER)
    return out.sort_values(["category", "_order"]).drop(columns=["_order"])


def add_drops(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["drop_vs_gt_crop_only_primary"] = np.nan
    out["drop_vs_gt_crop_only_oom_as_wrong"] = np.nan
    for category, group in out.groupby("category"):
        base = group[group["condition"].eq(BASELINE)]
        if base.empty:
            continue
        base_primary = float(base["accuracy_primary_excluding_oom_mean"].iloc[0])
        base_oom = float(base["accuracy_oom_as_wrong_mean"].iloc[0])
        idx = out["category"].eq(category)
        out.loc[idx, "drop_vs_gt_crop_only_primary"] = base_primary - out.loc[idx, "accuracy_primary_excluding_oom_mean"]
        out.loc[idx, "drop_vs_gt_crop_only_oom_as_wrong"] = base_oom - out.loc[idx, "accuracy_oom_as_wrong_mean"]
    return out


def sample_mean_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = (
        df.groupby(["sample_id", "condition"], as_index=False)[metric]
        .mean()
        .pivot(index="sample_id", columns="condition", values=metric)
    )
    return table


def bootstrap(df: pd.DataFrame, metric: str, mode: str, iters: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    table = sample_mean_table(df, metric)
    rows = []
    for target in TARGETS:
        pair = table[[BASELINE, target]].dropna()
        base = pair[BASELINE].to_numpy(dtype=float)
        targ = pair[target].to_numpy(dtype=float)
        n = len(pair)
        if n == 0:
            continue
        idx = np.arange(n)
        diffs = np.empty(iters)
        for i in range(iters):
            chosen = rng.choice(idx, size=n, replace=True)
            diffs[i] = base[chosen].mean() - targ[chosen].mean()
        obs = base.mean() - targ.mean()
        p_left = (np.sum(diffs <= 0) + 1) / (iters + 1)
        p_right = (np.sum(diffs >= 0) + 1) / (iters + 1)
        rows.append(
            {
                "comparison": f"{BASELINE}_vs_{target}",
                "mode": mode,
                "unit": "sample_mean_over_seeds",
                "n_samples": n,
                "accuracy_gt_crop_only": base.mean(),
                "accuracy_target_condition": targ.mean(),
                "mean_difference_gt_minus_target": obs,
                "ci95_low": np.quantile(diffs, 0.025),
                "ci95_high": np.quantile(diffs, 0.975),
                "p_one_sided_diff_gt_0": p_left,
                "p_two_sided_diff_ne_0": min(1.0, 2 * min(p_left, p_right)),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy(
    summary: pd.DataFrame,
    output_png: Path,
    metric: str,
    title: str,
    y_min: float = 0.0,
) -> None:
    overall = summary[summary["category"].eq("ALL")]
    series = overall[overall["condition"].ne("original_only")].sort_values("k_irrelevant")
    x = series["k_irrelevant"].to_numpy()
    y = series[f"{metric}_mean"].to_numpy()
    err = series[f"{metric}_std"].fillna(0).to_numpy()
    # A high-contrast, colour-blind-friendly palette that remains clear on slides.
    series_color = "#0072B2"
    baseline_color = "#D55E00"
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")
    ax.errorbar(
        x,
        y,
        yerr=err,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.2,
        color=series_color,
        linewidth=2.8,
        elinewidth=1.8,
        capsize=5,
        capthick=1.8,
        label="GT crop + irrelevant crops",
        zorder=3,
    )
    for x_value, y_value, error_value in zip(x, y, err):
        label_y = min(y_value + error_value, 0.985)
        ax.annotate(
            f"{y_value:.3f}",
            xy=(x_value, label_y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#005A8D",
            fontsize=10.5,
            fontweight="semibold",
            zorder=4,
        )
    original = overall[overall["condition"].eq("original_only")]
    if not original.empty:
        ax.axhline(
            original[f"{metric}_mean"].iloc[0],
            linestyle=(0, (5, 3)),
            color=baseline_color,
            linewidth=2.2,
            label="Original image only",
            zorder=2,
        )
    ax.set_ylim(y_min, 1.0)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(y_min, 1.01, 0.1))
    ax.set_xlabel("Number of irrelevant crops", fontsize=12, labelpad=9)
    ax.set_ylabel("Accuracy", fontsize=12, labelpad=9)
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#64748B")
    ax.tick_params(labelsize=11, colors="#334155")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout(pad=1.1)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(summary: pd.DataFrame, by_category: pd.DataFrame, boot: pd.DataFrame, output: Path, visual_tokens_available: bool) -> None:
    overall = summary[summary["category"].eq("ALL")].copy()
    gt = overall[overall["condition"].eq(BASELINE)]["accuracy_primary_excluding_oom_mean"].iloc[0]
    k8 = overall[overall["condition"].eq("gt_plus_8_irrelevant")]["accuracy_primary_excluding_oom_mean"].iloc[0]
    oom_any = overall["oom_rate_mean"].max() > 0
    visual_note = (
        "Per-image visual token counts were available from `image_grid_thw`."
        if visual_tokens_available
        else "Per-image visual token counts were not available; visual-token fields are NA."
    )
    preprocess_rate = overall["preprocess_error_rate_excluding_oom_mean"].max()
    lines = [
        "# E1-v2.1 Full Context-Pollution Report",
        "",
        "## Overall Seed-Level Summary",
        overall.to_csv(index=False),
        "## Category Summary",
        by_category.to_csv(index=False),
        "## Bootstrap",
        boot.to_csv(index=False),
        "## Notes",
        f"- Primary accuracy excludes OOM rows. OOM-as-wrong sensitivity is reported separately.",
        f"- Any OOM observed: {bool(oom_any)}.",
        f"- Max preprocessing-error rate by condition: {preprocess_rate:.4f}. These rows remain in the raw CSV as `status=invalid`.",
        f"- GT-only minus +8 primary drop: {gt - k8:.4f}.",
        f"- {visual_note}",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score E1-v2.1 full results with OOM-aware metrics.")
    parser.add_argument("--input_csv", default="outputs/results_context_pollution_full_v21_all_seeds.csv")
    parser.add_argument("--summary_csv", default="outputs/summary_context_pollution_full_v21.csv")
    parser.add_argument("--by_category_csv", default="outputs/summary_context_pollution_full_v21_by_category.csv")
    parser.add_argument("--bootstrap_csv", default="outputs/bootstrap_context_pollution_full_v21.csv")
    parser.add_argument("--plot_primary_png", default="outputs/accuracy_primary_excluding_oom_full_v21.png")
    parser.add_argument("--plot_oom_as_wrong_png", default="outputs/accuracy_oom_as_wrong_full_v21.png")
    parser.add_argument("--report_md", default="outputs/report_context_pollution_full_v21.md")
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_paths = [
        Path(args.summary_csv),
        Path(args.by_category_csv),
        Path(args.bootstrap_csv),
        Path(args.plot_primary_png),
        Path(args.plot_oom_as_wrong_png),
        Path(args.report_md),
    ]
    for path in output_paths:
        can_write(path, args.overwrite)

    df = load_results(Path(args.input_csv))
    per_seed_parts = [per_seed_condition(df)]
    for category in sorted(df["category"].unique()):
        per_seed_parts.append(per_seed_condition(df, category=category))
    per_seed = pd.concat(per_seed_parts, ignore_index=True)
    summary = add_drops(aggregate_seed_stats(per_seed))
    by_category = summary[summary["category"].ne("ALL")].copy()

    boot = pd.concat(
        [
            bootstrap(df, "correct_primary", "primary_excluding_oom", args.bootstrap_iters, args.seed),
            bootstrap(df, "correct_oom_as_wrong", "oom_as_wrong_sensitivity", args.bootstrap_iters, args.seed),
        ],
        ignore_index=True,
    )
    visual_tokens_available = df["per_image_visual_token_counts"].astype(str).str.len().gt(0).any()

    summary.to_csv(args.summary_csv, index=False)
    by_category.to_csv(args.by_category_csv, index=False)
    boot.to_csv(args.bootstrap_csv, index=False)
    plot_accuracy(summary, Path(args.plot_primary_png), "accuracy_primary_excluding_oom", "E1-v2.1 primary accuracy excluding OOM")
    plot_accuracy(
        summary,
        Path(args.plot_oom_as_wrong_png),
        "accuracy_oom_as_wrong",
        "E1-v2.1 OOM-as-wrong sensitivity",
        y_min=0.5,
    )
    write_report(summary, by_category, boot, Path(args.report_md), visual_tokens_available)

    overall = summary[summary["category"].eq("ALL")]
    cols = [
        "condition",
        "accuracy_primary_excluding_oom_mean",
        "accuracy_primary_excluding_oom_std",
        "accuracy_generation_only_excluding_oom_preprocess_mean",
        "accuracy_oom_as_wrong_mean",
        "oom_rate_mean",
        "invalid_rate_excluding_oom_mean",
        "preprocess_error_rate_excluding_oom_mean",
        "model_invalid_rate_excluding_oom_preprocess_mean",
        "mean_input_tokens_mean",
        "mean_distractor_token_fraction_mean",
    ]
    print("E1-v2.1 scoring complete.")
    print(overall[cols].to_string(index=False))


if __name__ == "__main__":
    main()
