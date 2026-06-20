#!/bin/bash
# Rule-based pipeline — all images in a folder (recursive).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
FOLDER="${1:-training/test_images/mouse}"
exec ./run.sh --mode batch --source "$FOLDER"
