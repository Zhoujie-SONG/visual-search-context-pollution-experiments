#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <experiment command> [args...]" >&2
  exit 2
fi

"$@"
/home/songzhoujie/cvpr27/vstar/publish_reports_to_github.sh

