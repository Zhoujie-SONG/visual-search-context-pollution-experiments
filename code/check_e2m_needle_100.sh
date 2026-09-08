#!/usr/bin/env bash
set -u

OUTPUT_DIR=/mnt/data2/szj/experiments/e2m_needle_smoke
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python

"$PYTHON" -c 'import glob,json,sys
from collections import Counter
out=sys.argv[1]
ids=set()
status=Counter()
for path in glob.glob(out + "/shards/shard*/trajectories.jsonl"):
    for line in open(path, encoding="utf-8"):
        try:
            episode=json.loads(line)["episode"]
            ids.add(episode["sample_id"])
            status[episode.get("status", "unknown")]+=1
        except Exception:
            pass
print(f"completed_unique={len(ids)}/100 status={dict(status)}")' "$OUTPUT_DIR"

for shard in 0 1 2 3 4 5 6 7; do
  count=$("$PYTHON" -c 'import json,sys
seen=set()
try:
    lines=open(sys.argv[1], encoding="utf-8")
except FileNotFoundError:
    lines=[]
for line in lines:
    try: seen.add(json.loads(line)["episode"]["sample_id"])
    except Exception: pass
print(len(seen))' "$OUTPUT_DIR/shards/shard${shard}/trajectories.jsonl")
  echo "shard${shard} completed=$count"
  tail -2 "$OUTPUT_DIR/logs/shard${shard}_100.log" 2>/dev/null || true
done

if [[ -f "$OUTPUT_DIR/E2M_NEEDLE_100_COMPLETE" ]]; then
  cat "$OUTPUT_DIR/E2M_NEEDLE_100_COMPLETE"
elif [[ -f "$OUTPUT_DIR/RUN100_FAILED" ]]; then
  cat "$OUTPUT_DIR/RUN100_FAILED"
else
  echo "run_status=running_or_waiting"
fi
