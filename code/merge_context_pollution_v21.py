import argparse
from pathlib import Path

import pandas as pd


CONDITION_ORDER = {
    "original_only": -1,
    "gt_crop_only": 0,
    "gt_plus_1_irrelevant": 1,
    "gt_plus_4_irrelevant": 4,
    "gt_plus_8_irrelevant": 8,
}


def write_checked(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype(int)
    out["sample_index"] = pd.to_numeric(out["sample_index"], errors="coerce")
    out["_condition_order"] = out["condition"].map(CONDITION_ORDER)
    out = out.sort_values(["seed", "sample_index", "_condition_order", "sample_id"]).drop(columns=["_condition_order"])
    return out


def merge_seed(seed: int, num_shards: int, output_dir: Path, overwrite: bool) -> Path:
    parts = []
    missing = []
    for shard_id in range(num_shards):
        path = output_dir / f"v21_full_seed{seed}_shard{shard_id}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        parts.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    if missing:
        raise FileNotFoundError("Missing shard CSVs:\n" + "\n".join(missing))
    df = sort_results(pd.concat(parts, ignore_index=True))
    duplicated = df.duplicated(["seed", "sample_id", "condition"], keep=False)
    if duplicated.any():
        dup = df.loc[duplicated, ["seed", "sample_id", "condition", "shard_id"]]
        raise RuntimeError("Duplicate seed/sample/condition rows:\n" + dup.to_string(index=False))
    out = output_dir / f"results_context_pollution_full_v21_seed{seed}.csv"
    write_checked(df, out, overwrite)
    print(f"seed {seed}: wrote {len(df)} rows -> {out}")
    return out


def merge_all(seeds: list[int], output_dir: Path, overwrite: bool) -> Path:
    parts = []
    missing = []
    for seed in seeds:
        path = output_dir / f"results_context_pollution_full_v21_seed{seed}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        parts.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    if missing:
        raise FileNotFoundError("Missing seed CSVs:\n" + "\n".join(missing))
    df = sort_results(pd.concat(parts, ignore_index=True))
    duplicated = df.duplicated(["seed", "sample_id", "condition"], keep=False)
    if duplicated.any():
        dup = df.loc[duplicated, ["seed", "sample_id", "condition", "shard_id"]]
        raise RuntimeError("Duplicate seed/sample/condition rows:\n" + dup.to_string(index=False))
    out = output_dir / "results_context_pollution_full_v21_all_seeds.csv"
    write_checked(df, out, overwrite)
    print(f"all seeds: wrote {len(df)} rows -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge E1-v2.1 shard CSVs safely.")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--seed", type=int, default=None, help="Merge one seed only.")
    parser.add_argument("--all", action="store_true", help="Merge all seed-level CSVs.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.seed is not None:
        merge_seed(args.seed, args.num_shards, output_dir, args.overwrite)
    if args.all:
        merge_all(args.seeds, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
