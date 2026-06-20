#!/bin/bash
# YOLO-Seg pipeline — single image (trained best.pt) with OpenCV dashboard.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PHOTO="${1:-training/test_images/Charger_Adapter/Image_43.jpg}"
MODEL="${SMART_STORAGE_MODEL:-models/smart_storage_final_best.pt}"
DEVICE="${SMART_STORAGE_DEVICE:-cuda}"
exec ./run.sh --mode image --source "$PHOTO" --model "$MODEL" --device "$DEVICE"
