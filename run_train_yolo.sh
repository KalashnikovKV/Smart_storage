#!/bin/bash
# Train YOLO-Seg on dataset/ (after Roboflow import).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
EPOCHS="${SMART_STORAGE_EPOCHS:-50}"
DEVICE="${SMART_STORAGE_DEVICE:-cuda}"
exec uv run python scripts/train_yolo.py \
  --data dataset/dataset.yaml \
  --model models/yolo11n-seg.pt \
  --epochs "$EPOCHS" \
  --device "$DEVICE"
