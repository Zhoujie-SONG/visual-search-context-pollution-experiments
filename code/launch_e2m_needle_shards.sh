#!/usr/bin/env bash
set -u

OUTPUT_DIR="${1:-/mnt/data2/szj/experiments/e2m_needle_smoke}"
shift || true
if [ "$#" -eq 0 ]; then
  set -- 0 1 2 3 4 5 6 7
fi

PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER=/home/songzhoujie/cvpr27/vstar/run_e2m_needle_smoke.py
MODEL=/mnt/data2/szj/models/Mini-o3-7B-v1-complete
mkdir -p "$OUTPUT_DIR/logs"

pids=()
shards=()
for shard in "$@"; do
  log="$OUTPUT_DIR/logs/shard${shard}.log"
  echo "starting shard $shard on GPU $shard; log=$log"
  env CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" "$RUNNER" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --num_shards 8 \
    --shard_id "$shard" \
    >"$log" 2>&1 &
  pids+=("$!")
  shards+=("$shard")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "shard ${shards[$index]} complete"
  else
    echo "shard ${shards[$index]} failed; inspect $OUTPUT_DIR/logs/shard${shards[$index]}.log" >&2
    status=1
  fi
done
exit "$status"
