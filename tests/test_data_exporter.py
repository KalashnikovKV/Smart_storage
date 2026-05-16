"""Tests for DataExporter."""

import os

import numpy as np
import pandas as pd
import pytest

from src.app.data_exporter import DataExporter
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
        bbox=(10, 10, 40, 30),
        color_hsv=ColorResult("white", 0.91, "hsv"),
        color_kmeans=ColorResult("white", 0.85, "kmeans"),
        primary_color="white",
        size_category="small",
        area_pixels=15234,
        aspect_ratio=1.3,
    )


@pytest.fixture
def sample_image():
    """480x640 BGR image: white rectangle on black background."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[10:40, 10:50] = (255, 255, 255)
    return img


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


def test_export_creates_images_dir_when_enabled(tmp_csv, sample_decision, sample_detection):
    from src.config import AppConfig
    cfg = AppConfig(csv_output_path=tmp_csv, save_roi_images=True)
    exporter = DataExporter(output_path=tmp_csv, config=cfg)
    assert exporter.images_dir.exists()
    assert exporter.images_dir.is_dir()


def test_export_does_not_create_images_dir_when_disabled(tmp_path):
    from src.config import AppConfig
    csv_path = str(tmp_path / "results.csv")
    cfg = AppConfig(csv_output_path=csv_path, save_roi_images=False)
    exporter = DataExporter(output_path=csv_path, config=cfg)
    assert not exporter.images_dir.exists()


def test_export_saves_roi_image(tmp_csv, sample_decision, sample_detection, sample_image):
    exporter = DataExporter(output_path=tmp_csv)
    roi = sample_image[10:40, 10:50]
    exporter.export(sample_decision, sample_detection, roi=roi)

    saved_files = list(exporter.images_dir.glob("*.jpg"))
    assert len(saved_files) == 1


def test_export_roi_name_written_to_csv(tmp_csv, sample_decision, sample_detection, sample_image):
    exporter = DataExporter(output_path=tmp_csv)
    roi = sample_image[10:40, 10:50]
    exporter.export(sample_decision, sample_detection, roi=roi)

    df = pd.read_csv(tmp_csv)
    assert "roi_image" in df.columns
    assert df.iloc[0]["roi_image"].endswith(".jpg")


def test_export_without_roi_leaves_roi_image_empty(tmp_csv, sample_decision, sample_detection):
    exporter = DataExporter(output_path=tmp_csv)
    exporter.export(sample_decision, sample_detection)

    df = pd.read_csv(tmp_csv)
    assert "roi_image" in df.columns
    assert df.iloc[0]["roi_image"] != df.iloc[0]["roi_image"] or df.iloc[0]["roi_image"] == ""


def test_export_many_saves_roi_per_detection(tmp_csv, sample_decision, sample_detection, sample_image):
    exporter = DataExporter(output_path=tmp_csv)
    exporter.export_many(
        [sample_decision],
        [sample_detection],
        image_name="test.jpg",
        original_image=sample_image,
    )

    df = pd.read_csv(tmp_csv)
    assert len(df) == 1
    assert "roi_image" in df.columns
    assert df.iloc[0]["roi_image"].endswith(".jpg")

    saved_files = list(exporter.images_dir.glob("*.jpg"))
    assert len(saved_files) == 1


def test_export_many_without_original_image_leaves_roi_empty(
    tmp_csv, sample_decision, sample_detection
):
    exporter = DataExporter(output_path=tmp_csv)
    exporter.export_many([sample_decision], [sample_detection], image_name="test.jpg")

    df = pd.read_csv(tmp_csv)
    assert "roi_image" in df.columns
    roi_val = df.iloc[0]["roi_image"]
    assert roi_val != roi_val or roi_val == ""


def test_export_writes_ground_truth_column(tmp_csv, sample_decision, sample_detection):
    exporter = DataExporter(output_path=tmp_csv)
    exporter.export(
        sample_decision,
        sample_detection,
        ground_truth="Flash Drive",
    )
    df = pd.read_csv(tmp_csv)
    assert "ground_truth" in df.columns
    assert df.iloc[0]["ground_truth"] == "Flash Drive"
    assert pd.isna(df.iloc[0]["ground_truth"]) is False


def test_export_many_writes_ground_truth(tmp_csv, sample_decision, sample_detection, sample_image):
    exporter = DataExporter(output_path=tmp_csv)
    exporter.export_many(
        [sample_decision],
        [sample_detection],
        image_name="test.jpg",
        original_image=sample_image,
        ground_truth="Mouse",
    )
    df = pd.read_csv(tmp_csv)
    assert df.iloc[0]["ground_truth"] == "Mouse"


def test_export_does_not_save_roi_when_disabled(tmp_path, sample_decision, sample_detection, sample_image):
    from src.config import AppConfig
    csv_path = str(tmp_path / "results.csv")
    cfg = AppConfig(csv_output_path=csv_path, save_roi_images=False)
    exporter = DataExporter(output_path=csv_path, config=cfg)

    roi = sample_image[10:40, 10:50]
    exporter.export(sample_decision, sample_detection, roi=roi)

    assert not exporter.images_dir.exists()
    df = pd.read_csv(csv_path)
    roi_val = df.iloc[0]["roi_image"]
    assert roi_val != roi_val or roi_val == ""
