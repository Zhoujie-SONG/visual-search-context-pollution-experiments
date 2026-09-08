#!/usr/bin/env bash
set -u

OUTPUT_DIR="${1:-/mnt/data2/szj/experiments/e2m_needle_smoke}"
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER=/home/songzhoujie/cvpr27/vstar/run_e2m_needle_smoke.py
MODEL=/mnt/data2/szj/models/Mini-o3-7B-v1-complete

count_unique() {
  "$PYTHON" -c 'import glob,json,sys
ids=set()
for path in glob.glob(sys.argv[1] + "/shards/shard*/trajectories.jsonl"):
    for line in open(path, encoding="utf-8"):
        try:
            ids.add(json.loads(line)["episode"]["sample_id"])
        except Exception:
            pass
print(len(ids))' "$OUTPUT_DIR"
}

while true; do
  count="$(count_unique)"
  echo "$(date '+%F %T') completed_unique=$count/20"
  if [ "$count" -ge 20 ]; then
    break
  fi
  if ! pgrep -f 'run_e2m_needle_smoke.py.*--shard_id' >/dev/null; then
    echo "No shard workers remain, but only $count/20 samples completed." >&2
    touch "$OUTPUT_DIR/FINALIZER_FAILED"
    exit 2
  fi
  sleep 60
done

"$PYTHON" "$RUNNER" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --num_shards 8 \
  --merge_only
status=$?
if [ "$status" -eq 0 ]; then
  touch "$OUTPUT_DIR/SMOKE_COMPLETE"
  echo "$(date '+%F %T') merge complete"
else
  touch "$OUTPUT_DIR/FINALIZER_FAILED"
  echo "$(date '+%F %T') merge failed with status $status" >&2
fi
exit "$status"
