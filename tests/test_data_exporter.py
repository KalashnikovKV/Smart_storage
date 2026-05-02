"""Tests for DataExporter."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_exporter import DataExporter
from src.models import ColorResult, Decision, DetectionResult


@pytest.fixture
def tmp_csv(tmp_path):
    return str(tmp_path / "test_results.csv")


@pytest.fixture
def sample_decision():
    return Decision(
        category="Зарядка iPhone",
        confidence=0.87,
        color="white",
        size="small",
        method_used="combined",
        is_unknown=False,
        timestamp="2026-04-24T12:00:00",
    )


@pytest.fixture
def sample_detection():
    return DetectionResult(
        bbox=(100, 100, 40, 30),
        color_hsv=ColorResult("white", 0.91, "hsv"),
        color_kmeans=ColorResult("white", 0.85, "kmeans"),
        primary_color="white",
        size_category="small",
        area_pixels=15234,
        aspect_ratio=1.3,
    )


def test_export_creates_csv(tmp_csv, sample_decision, sample_detection):
    exporter = DataExporter(output_path=tmp_csv)
    exporter.export(sample_decision, sample_detection)
    assert os.path.exists(tmp_csv)
    df = pd.read_csv(tmp_csv)
    assert len(df) == 1
    assert df.iloc[0]["category"] == "Зарядка iPhone"
    assert df.iloc[0]["color"] == "white"


def test_export_appends_rows(tmp_csv, sample_decision, sample_detection):
    exporter = DataExporter(output_path=tmp_csv)
    exporter.export(sample_decision, sample_detection)
    exporter.export(sample_decision, sample_detection)
    df = pd.read_csv(tmp_csv)
    assert len(df) == 2
