#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=/mnt/data2/szj/experiments/e2m_needle_smoke
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER=/home/songzhoujie/cvpr27/vstar/run_e2m_needle_smoke.py
MODEL=/mnt/data2/szj/models/Mini-o3-7B-v1-complete
SAMPLE_IDS="$OUTPUT_DIR/smoke_sample_ids.txt"
LOG="$OUTPUT_DIR/logs/shard1_100_recovery.log"

mkdir -p "$OUTPUT_DIR/logs"
printf '%s\n' "$(date '+%F %T') recovery started: shard1 missing episodes only" >"$OUTPUT_DIR/RUN100_RECOVERY_RUNNING"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --num_shards 8 \
  --shard_id 1 \
  --sample_ids_file "$SAMPLE_IDS" \
  >"$LOG" 2>&1

"$PYTHON" "$RUNNER" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --num_shards 8 \
  --sample_ids_file "$SAMPLE_IDS" \
  --merge_only \
  >"$OUTPUT_DIR/logs/merge_100_recovery.log" 2>&1

rm -f "$OUTPUT_DIR/RUN100_FAILED" "$OUTPUT_DIR/RUN100_RECOVERY_RUNNING"
printf '%s\n' "$(date '+%F %T') 100-sample run recovered and merged" >"$OUTPUT_DIR/E2M_NEEDLE_100_COMPLETE"
