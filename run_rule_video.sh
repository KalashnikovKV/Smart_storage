#!/bin/bash
# Rule-based pipeline — webcam or video file.
# Usage: ./run_rule_video.sh
#        ./run_rule_video.sh training/test_video/Video_10.MOV
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ $# -ge 1 ]]; then
  exec ./run.sh --mode video --source "$1"
fi
exec ./run.sh --mode video
