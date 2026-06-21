#!/usr/bin/env python3
"""CLI entry point for YOLO-Seg dataset preparation (see training/build_dataset.py)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.build_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
