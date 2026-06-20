#!/bin/bash
# Rule-based pipeline — single image with OpenCV dashboard.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PHOTO="${1:-training/test_images/Charger_Adapter/Image_43.jpg}"
exec ./run.sh --mode image --source "$PHOTO"
