"""Tests for results CSV loading in web layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.app.data_exporter import DataExporter
from src.web.services.label_queue import load_labeled_names
from src.web.services.results_csv import read_results_csv


def test_read_empty_header_only_csv(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(",".join(DataExporter.COLUMNS) + "\n", encoding="utf-8")

    df = read_results_csv(csv_path)
    assert list(df.columns) == DataExporter.COLUMNS
    assert len(df) == 0


def test_load_labeled_names_from_exporter_csv(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    row = {column: "" for column in DataExporter.COLUMNS}
    row.update(
        {
            "timestamp": "2026-01-01T00:00:00",
            "image_name": "Image_1.jpeg",
            "object_id": 1,
            "category": "Mouse",
            "ground_truth": "Mouse",
        }
    )
    pd.DataFrame([row], columns=DataExporter.COLUMNS).to_csv(csv_path, index=False)

    assert load_labeled_names(csv_path) == {"Image_1.jpeg"}
