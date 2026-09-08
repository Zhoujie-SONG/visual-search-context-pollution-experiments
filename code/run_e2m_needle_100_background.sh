#!/usr/bin/env bash
set -u

OUTPUT_DIR=/mnt/data2/szj/experiments/e2m_needle_smoke
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER=/home/songzhoujie/cvpr27/vstar/run_e2m_needle_smoke.py
MODEL=/mnt/data2/szj/models/Mini-o3-7B-v1-complete
SAMPLE_IDS="$OUTPUT_DIR/smoke_sample_ids.txt"
MIN_FREE_MIB=19000
MAX_ATTEMPTS=3

mkdir -p "$OUTPUT_DIR/logs"
exec 9>"$OUTPUT_DIR/run100.lock"
if ! flock -n 9; then
  echo "Another 100-sample controller already holds $OUTPUT_DIR/run100.lock" >&2
  exit 4
fi
rm -f "$OUTPUT_DIR/RUN100_FAILED"
printf '%s\n' "$(date '+%F %T') controller started" >"$OUTPUT_DIR/RUN100_RUNNING"

wait_for_gpu() {
  local gpu=$1
  local free_mib
  while true; do
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      return 0
    fi
    echo "$(date '+%F %T') gpu=$gpu waiting free_mib=${free_mib:-unknown} threshold=$MIN_FREE_MIB"
    sleep 60
  done
}

run_shard() {
  local shard=$1
  local gpu=$2
  local log="$OUTPUT_DIR/logs/shard${shard}_100.log"
  local attempt
  : >"$log"
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    wait_for_gpu "$gpu" >>"$log" 2>&1
    echo "$(date '+%F %T') shard=$shard gpu=$gpu attempt=$attempt starting" >>"$log"
    if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
      --output_dir "$OUTPUT_DIR" \
      --model "$MODEL" \
      --num_shards 8 \
      --shard_id "$shard" \
      --sample_ids_file "$SAMPLE_IDS" \
      >>"$log" 2>&1; then
      echo "$(date '+%F %T') shard=$shard complete" >>"$log"
      return 0
    fi
    echo "$(date '+%F %T') shard=$shard attempt=$attempt failed" >>"$log"
    sleep 120
  done
  echo "$(date '+%F %T') shard=$shard exhausted retries" >>"$log"
  return 1
}

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  run_shard "$shard" "$shard" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status != 0 )); then
  printf '%s\n' "$(date '+%F %T') one or more shards failed" >"$OUTPUT_DIR/RUN100_FAILED"
  exit 2
fi

if ! "$PYTHON" "$RUNNER" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --num_shards 8 \
  --sample_ids_file "$SAMPLE_IDS" \
  --merge_only \
  >"$OUTPUT_DIR/logs/merge_100.log" 2>&1; then
  printf '%s\n' "$(date '+%F %T') merge failed" >"$OUTPUT_DIR/RUN100_FAILED"
  exit 3
fi

printf '%s\n' "$(date '+%F %T') 100-sample run and merge complete" >"$OUTPUT_DIR/E2M_NEEDLE_100_COMPLETE"
printf '%s\n' "$(date '+%F %T') complete" >>"$OUTPUT_DIR/RUN100_RUNNING"
