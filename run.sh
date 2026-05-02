#!/bin/bash
# Smart Storage runner — always works from any directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHONPATH=. uv run python -m src.main "$@"
