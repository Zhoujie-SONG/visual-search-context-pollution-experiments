#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/songzhoujie/cvpr27/vstar
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
STAGE="${1:?Usage: $0 smoke|formal}"
ENV_FILE=/home/songzhoujie/.config/visualneedle/gemini.env

if [[ "$STAGE" != "smoke" && "$STAGE" != "formal" ]]; then
  echo "Stage must be smoke or formal." >&2
  exit 2
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Gemini environment file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

cd "$PROJECT"
"$PYTHON" "$PROJECT/judge_round6_memory_surgery_gemini.py" --stage "$STAGE" --model gemini-3.8-flash
"$PYTHON" "$PROJECT/score_round6_memory_surgery.py" --stage "$STAGE"
"$PROJECT/publish_reports_to_github.sh"

