#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/songzhoujie/cvpr27/vstar
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER="$PROJECT/run_round6_memory_surgery.py"
ROOT="$PROJECT/outputs/visualneedle_minio3/round6_memory_surgery_v1"
OUTPUT="$ROOT"
NUM_SHARDS=8
ARMS=(A_full B_oracle_top2 C_recent_top2 D_force_answer_r6 E_random_top2)

if ! grep -q 'Overall: \*\*PASS\*\*' "$ROOT/smoke/integrity_report.md" 2>/dev/null; then
  echo "Smoke integrity has not passed; refusing to start formal inference." >&2
  exit 1
fi

cd "$PROJECT"
"$PYTHON" "$RUNNER" --prepare --stage formal

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  (
    for arm in "${ARMS[@]}"; do
      mkdir -p "$OUTPUT/$arm/shards/shard_$shard"
      CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" "$RUNNER" \
        --run-shard --stage formal --arm "$arm" --num-shards "$NUM_SHARDS" --shard-id "$shard" \
        >"$OUTPUT/$arm/shards/shard_$shard/process_stdout.log" 2>&1
    done
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

for arm in "${ARMS[@]}"; do
  "$PYTHON" "$RUNNER" --merge-arm --stage formal --arm "$arm" --num-shards "$NUM_SHARDS"
done
"$PYTHON" "$RUNNER" --merge-stage --stage formal
"$PYTHON" "$PROJECT/score_round6_memory_surgery.py" --stage formal
"$PROJECT/publish_reports_to_github.sh"

echo "Formal inference is complete. No downstream experiment has been started."

