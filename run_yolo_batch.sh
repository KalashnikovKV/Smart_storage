#!/bin/bash
# YOLO-Seg pipeline — batch folder (trained best.pt).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
FOLDER="${1:-training/test_images/mouse}"
MODEL="${SMART_STORAGE_MODEL:-models/smart_storage_final_best.pt}"
DEVICE="${SMART_STORAGE_DEVICE:-cuda}"
exec ./run.sh --mode batch --source "$FOLDER" --model "$MODEL" --device "$DEVICE"
