#!/bin/bash
# YOLO-Seg pipeline — webcam or video file (trained best.pt).
#
# Multi-object scenes (e.g. Video_10): all classes are detected by default.
# Webcam — focus on headphones only:
#   SMART_STORAGE_YOLO_PREFER=headphones ./run_yolo_video.sh
#
# Usage: ./run_yolo_video.sh
#        ./run_yolo_video.sh training/test_video/Video_10.MOV
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODEL="${SMART_STORAGE_MODEL:-models/smart_storage_final_best.pt}"
DEVICE="${SMART_STORAGE_DEVICE:-cuda}"
YOLO_PREFER="${SMART_STORAGE_YOLO_PREFER:-}"

ARGS=(--mode video --model "$MODEL" --device "$DEVICE")
if [[ -n "$YOLO_PREFER" ]]; then
  ARGS+=(--yolo-prefer "$YOLO_PREFER" --yolo-prefer-strict)
fi
if [[ $# -ge 1 ]]; then
  exec ./run.sh "${ARGS[@]}" --source "$1"
fi
exec ./run.sh "${ARGS[@]}"
