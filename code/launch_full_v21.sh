#!/usr/bin/env bash
set -euo pipefail

cd /home/songzhoujie/cvpr27/vstar

export TMPDIR=/mnt/data2/szj/tmp
export HF_HOME=/mnt/data2/szj/hf_cache
export TRANSFORMERS_CACHE=/mnt/data2/szj/hf_cache

PYTHON=/home/songzhoujie/miniconda3/envs/cvpr2027/bin/python
MODEL=/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct
DATA_ROOT=/home/songzhoujie/cvpr27/vstar/data/vstar_bench
NUM_SHARDS=8
NUM_SAMPLES=238

mkdir -p outputs

for seed in 42 43 44; do
  echo "=== launching seed ${seed} with ${NUM_SHARDS} shards ==="
  pids=()
  for shard in 0 1 2 3 4 5 6 7; do
    log="outputs/log_v21_full_seed${seed}_shard${shard}.txt"
    CUDA_VISIBLE_DEVICES=${shard} "${PYTHON}" run_context_pollution_full_v21.py \
      --data_root "${DATA_ROOT}" \
      --model "${MODEL}" \
      --crop_dir outputs/crops \
      --seed "${seed}" \
      --shard_id "${shard}" \
      --num_shards "${NUM_SHARDS}" \
      --num_samples "${NUM_SAMPLES}" \
      --resume \
      > "${log}" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one shard failed for seed ${seed}. See outputs/log_v21_full_seed${seed}_shard*.txt" >&2
    exit 1
  fi

  "${PYTHON}" merge_context_pollution_v21.py \
    --seed "${seed}" \
    --num_shards "${NUM_SHARDS}" \
    --overwrite
done

"${PYTHON}" merge_context_pollution_v21.py \
  --all \
  --seeds 42 43 44 \
  --num_shards "${NUM_SHARDS}" \
  --overwrite

"${PYTHON}" score_context_pollution_v21.py \
  --input_csv outputs/results_context_pollution_full_v21_all_seeds.csv \
  --summary_csv outputs/summary_context_pollution_full_v21.csv \
  --by_category_csv outputs/summary_context_pollution_full_v21_by_category.csv \
  --bootstrap_csv outputs/bootstrap_context_pollution_full_v21.csv \
  --plot_primary_png outputs/accuracy_primary_excluding_oom_full_v21.png \
  --plot_oom_as_wrong_png outputs/accuracy_oom_as_wrong_full_v21.png \
  --report_md outputs/report_context_pollution_full_v21.md \
  --overwrite
