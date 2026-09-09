#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/songzhoujie/cvpr27/vstar
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER="$PROJECT/run_round6_direct_answer.py"
OUTPUT="$PROJECT/outputs/visualneedle_minio3/round6_direct_answer_v1/full"
NUM_SHARDS=8

if ! grep -q 'Overall: \*\*PASS\*\*' "$PROJECT/outputs/visualneedle_minio3/round6_direct_answer_v1/sanity/integrity_report.md" 2>/dev/null; then
  echo "Direct-answer sanity has not passed; refusing to start full cohort." >&2
  exit 1
fi

cd "$PROJECT"
"$PYTHON" "$RUNNER" --prepare --stage full
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" "$RUNNER" \
    --run-shard --stage full --num-shards "$NUM_SHARDS" --shard-id "$shard" \
    >"$OUTPUT/shard_${shard}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
"$PYTHON" "$RUNNER" --merge --stage full --num-shards "$NUM_SHARDS"
"$PYTHON" "$PROJECT/score_round6_direct_answer.py" --stage full
"$PROJECT/publish_reports_to_github.sh"
