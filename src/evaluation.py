"""Command-line evaluation entry point for labeled Smart Storage CSV results.

Usage:
    uv run python -m src.evaluation

    uv run python -m src.evaluation --csv output/labels.csv

    uv run python -m src.evaluation --csv output/results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.app.data_exporter import EvaluationTracker


DEFAULT_EVALUATION_CSV = "output/labels.csv"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for evaluation mode."""
    parser = argparse.ArgumentParser(
        description="Evaluate Smart Storage predictions using labeled CSV rows."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_EVALUATION_CSV,
        help=(
            "Path to CSV with category and ground_truth columns. "
            f"Default: {DEFAULT_EVALUATION_CSV}"
        ),
    )

    return parser


def main() -> None:
    """Load labeled CSV rows and print evaluation report."""
    parser = build_parser()
    args = parser.parse_args()

    csv_path = Path(args.csv)

    tracker = EvaluationTracker()

    try:
        tracker.load_labeled(str(csv_path))
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        print(
            "\nCreate labeled data first, for example:\n"
            "  uv run python -m src.main --mode label --source test_images/Image.jpeg\n"
            "\nOr pass an existing CSV explicitly:\n"
            "  uv run python -m src.evaluation --csv output/results.csv",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    tracker.print_report()


if __name__ == "__main__":
    main()