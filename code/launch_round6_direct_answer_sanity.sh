#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/songzhoujie/cvpr27/vstar
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
RUNNER="$PROJECT/run_round6_direct_answer.py"

cd "$PROJECT"
"$PYTHON" "$RUNNER" --prepare --stage sanity
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" --run-shard --stage sanity --num-shards 1 --shard-id 0
"$PYTHON" "$RUNNER" --merge --stage sanity --num-shards 1
"$PYTHON" "$PROJECT/score_round6_direct_answer.py" --stage sanity
