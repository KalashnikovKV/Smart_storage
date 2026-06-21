#!/bin/bash
# Import Roboflow YOLOv8 Segmentation export into dataset/.
# Usage: ./run_import_roboflow.sh path/to/roboflow-export
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
EXPORT="${1:?Usage: ./run_import_roboflow.sh path/to/roboflow-export}"
exec uv run python scripts/build_dataset.py --import-from "$EXPORT" --clean
