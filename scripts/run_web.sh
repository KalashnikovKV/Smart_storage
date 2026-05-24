#!/bin/bash
# Smart Storage ML Studio — FastAPI backend (+ optional frontend build)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8765}"
HOST="${HOST:-0.0.0.0}"

if [ -d "$ROOT/frontend/dist" ]; then
  echo "Serving built frontend from frontend/dist"
else
  echo "No frontend/dist — API only. Run: cd frontend && npm run dev"
fi

echo ""
echo "  Open in browser:  http://localhost:${PORT}"
echo ""

PYTHONPATH=. uv run python -m src.web --host "$HOST" --port "$PORT"
