#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/songzhoujie/cvpr27/vstar
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER="$PROJECT/run_round6_memory_surgery.py"
OUTPUT="$PROJECT/outputs/visualneedle_minio3/round6_memory_surgery_v1/devcheck"
NUM_SHARDS=8

cd "$PROJECT"
"$PYTHON" "$RUNNER" --prepare --stage devcheck

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  mkdir -p "$OUTPUT/A_full/shards/shard_$shard"
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" "$RUNNER" \
    --run-shard --stage devcheck --arm A_full --num-shards "$NUM_SHARDS" --shard-id "$shard" \
    >"$OUTPUT/A_full/shards/shard_$shard/process_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" "$RUNNER" --merge-arm --stage devcheck --arm A_full --num-shards "$NUM_SHARDS"
"$PYTHON" "$RUNNER" --merge-stage --stage devcheck
"$PYTHON" "$PROJECT/score_round6_memory_surgery.py" --stage devcheck
"$PROJECT/publish_reports_to_github.sh"

