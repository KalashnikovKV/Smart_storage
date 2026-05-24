"""CLI entry: uv run python -m src.web [--port 8765]"""

from __future__ import annotations

import argparse

import uvicorn

from src.web.config import WebSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Storage ML Studio web server")
    parser.add_argument("--host", default=WebSettings.default().host)
    parser.add_argument("--port", type=int, default=WebSettings.default().port)
    args = parser.parse_args()

    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
