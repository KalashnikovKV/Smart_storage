"""Load output/results.csv for the web layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.app.data_exporter import DataExporter


def read_results_csv(csv_path: Path) -> pd.DataFrame:
    """Load results CSV written by DataExporter."""
    if not csv_path.is_file():
        return pd.DataFrame(columns=DataExporter.COLUMNS)

    df = pd.read_csv(csv_path, keep_default_na=False)
    for column in DataExporter.COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[DataExporter.COLUMNS]
