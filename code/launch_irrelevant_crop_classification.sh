#!/usr/bin/env bash
set -euo pipefail

PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER=/home/songzhoujie/cvpr27/vstar/classify_irrelevant_needle_crops.py
OUTPUT_DIR=/mnt/data2/szj/experiments/e2m_needle_smoke/irrelevant_crop_classification
GPUS=(4 5 6 7)

mkdir -p "$OUTPUT_DIR/logs"
pids=()
for shard_id in 0 1 2 3; do
  gpu=${GPUS[$shard_id]}
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --num-shards 4 \
    --shard-id "$shard_id" \
    --overwrite \
    >"$OUTPUT_DIR/logs/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
  echo "launched shard=$shard_id gpu=$gpu pid=$!"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
