#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR=/mnt/data2/szj/experiments/e2m_needle_smoke
CLASS_DIR="$EXPERIMENT_DIR/irrelevant_crop_classification_100"
SEMANTIC_DIR="$EXPERIMENT_DIR/semantic_rescore_100"
PYTHON=/home/songzhoujie/miniconda3/envs/cvpr2027/bin/python
MODEL=/home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct
ROOT=/home/songzhoujie/cvpr27/vstar
LOG_DIR="$EXPERIMENT_DIR/reanalysis_logs"

mkdir -p "$LOG_DIR" "$CLASS_DIR/shards" "$SEMANTIC_DIR/shards"
printf '%s\n' "$(date '+%F %T') reanalysis started" >"$EXPERIMENT_DIR/REANALYSIS_100_RUNNING"

run_stage() {
  local stage=$1
  shift
  local pids=()
  for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$shard" "$@" --num-shards 8 --shard-id "$shard" \
      >"$LOG_DIR/${stage}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    printf '%s\n' "$(date '+%F %T') ${stage} failed" >"$EXPERIMENT_DIR/REANALYSIS_100_FAILED"
    return 1
  fi
}

run_stage crop_classification "$PYTHON" "$ROOT/classify_irrelevant_needle_crops.py" \
  --trajectories "$EXPERIMENT_DIR/trajectories.jsonl" \
  --output-dir "$CLASS_DIR" \
  --model "$MODEL" \
  --coverage-threshold 0.10 \
  --max-new-tokens 256

"$PYTHON" "$ROOT/classify_irrelevant_needle_crops.py" \
  --trajectories "$EXPERIMENT_DIR/trajectories.jsonl" \
  --output-dir "$CLASS_DIR" \
  --model "$MODEL" \
  --coverage-threshold 0.10 \
  --num-shards 8 \
  --merge-only \
  >"$LOG_DIR/crop_classification_merge.log" 2>&1

run_stage semantic_rescore "$PYTHON" "$ROOT/semantic_rescore_needle.py" \
  --input-dir "$EXPERIMENT_DIR" \
  --output-dir "$SEMANTIC_DIR" \
  --model "$MODEL" \
  --max-new-tokens 128

"$PYTHON" "$ROOT/semantic_rescore_needle.py" \
  --input-dir "$EXPERIMENT_DIR" \
  --output-dir "$SEMANTIC_DIR" \
  --model "$MODEL" \
  --num-shards 8 \
  --merge-only \
  >"$LOG_DIR/semantic_rescore_merge.log" 2>&1

"$PYTHON" "$ROOT/summarize_needle_100_checks.py" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --classification-dir "$CLASS_DIR" \
  --semantic-dir "$SEMANTIC_DIR" \
  --coverage-threshold 0.10 \
  >"$LOG_DIR/reanalysis_summary.log" 2>&1

printf '%s\n' "$(date '+%F %T') reanalysis complete" >"$EXPERIMENT_DIR/REANALYSIS_100_COMPLETE"
printf '%s\n' "$(date '+%F %T') complete" >>"$EXPERIMENT_DIR/REANALYSIS_100_RUNNING"
