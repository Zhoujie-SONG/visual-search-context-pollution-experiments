#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/songzhoujie/cvpr27/vstar
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER="$PROJECT/run_visualneedle_minio3_smoke20.py"
OUTPUT="$PROJECT/outputs/visualneedle_minio3/smoke20"
NUM_SHARDS=8
COMMAND="$0"

mkdir -p "$OUTPUT/shards"
cd "$PROJECT" || exit 1
E2M_PYTHON="$PYTHON" "$PYTHON" "$RUNNER" --prepare --num-shards "$NUM_SHARDS" --command-string "$COMMAND" || exit 1

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_dir="$OUTPUT/shards/shard_$shard"
  mkdir -p "$shard_dir"
  CUDA_VISIBLE_DEVICES="$shard" E2M_PYTHON="$PYTHON" "$PYTHON" "$RUNNER" \
    --run-shard --num-shards "$NUM_SHARDS" --shard-id "$shard" \
    >"$shard_dir/process_stdout.log" 2>&1 &
  pids+=("$!")
done

failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "shard $index failed; see $OUTPUT/shards/shard_$index/process_stdout.log" >&2
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  exit 1
fi

E2M_PYTHON="$PYTHON" "$PYTHON" "$RUNNER" --merge --num-shards "$NUM_SHARDS" --command-string "$COMMAND"
