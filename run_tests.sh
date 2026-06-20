#!/bin/bash
# Run unit tests (core pipeline, no optional web stack).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec uv run pytest tests/ -q
