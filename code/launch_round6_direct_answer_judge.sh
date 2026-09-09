#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/songzhoujie/cvpr27/vstar
PYTHON=/mnt/data2/szj/envs/venv_e2m/bin/python
ENV_FILE=/home/songzhoujie/.config/visualneedle/gemini.env

if ! grep -q 'Overall: \*\*PASS\*\*' "$PROJECT/outputs/visualneedle_minio3/round6_direct_answer_v1/full/integrity_report.md" 2>/dev/null; then
  echo "Direct-answer full integrity has not passed; refusing to run judge." >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Gemini environment file not found: $ENV_FILE" >&2
  exit 1
fi
set -a
source "$ENV_FILE"
set +a

cd "$PROJECT"
"$PYTHON" "$PROJECT/judge_round6_direct_answer_gemini.py" --stage full --model gemini-3.8-flash
"$PYTHON" "$PROJECT/score_round6_direct_answer.py" --stage full
"$PROJECT/publish_reports_to_github.sh"
