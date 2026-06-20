#!/bin/bash
# Check dataset/ layout and YOLO label files (no training).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec uv run python scripts/train_yolo.py --validate-only --data dataset/dataset.yaml
