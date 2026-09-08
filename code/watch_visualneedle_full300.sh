#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/songzhoujie/cvpr27/vstar
OUTPUT="$PROJECT/outputs/visualneedle_minio3/full300_baseline"
WATCH_PID="${1:?launcher pid required}"

while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 30
done

if [[ -s "$OUTPUT/full300_baseline_report.md" && -s "$OUTPUT/trajectories.jsonl" ]]; then
  exit 0
fi

cd "$PROJECT" || exit 1
exec ./launch_visualneedle_minio3_full300.sh >>"$OUTPUT/detached_launcher.log" 2>&1
