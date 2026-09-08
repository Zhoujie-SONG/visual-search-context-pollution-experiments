#!/usr/bin/env bash
set -euo pipefail

PUBLISHER="/home/songzhoujie/cvpr27/visual-search-context-pollution-experiments/scripts/sync_and_publish.py"

if [[ ! -f "$PUBLISHER" ]]; then
  echo "Publisher not found: $PUBLISHER" >&2
  exit 1
fi

python "$PUBLISHER" --push

