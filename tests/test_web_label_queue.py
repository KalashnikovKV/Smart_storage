"""Tests for web label queue service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.app.data_exporter import DataExporter
from src.web.services.label_queue import (
    build_queue,
    list_image_files,
    load_labeled_names,
    queue_stats,
)


def _write_label_row(csv_path: Path, *, image_name: str, ground_truth: str) -> None:
    row = {column: "" for column in DataExporter.COLUMNS}
    row.update(
        {
            "timestamp": "2026-01-01T00:00:00",
            "image_name": image_name,
            "object_id": 1,
            "category": ground_truth,
            "ground_truth": ground_truth,
            "confidence": 0.9,
        }
    )
    pd.DataFrame([row], columns=DataExporter.COLUMNS).to_csv(csv_path, index=False)


def test_list_image_files(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("skip")
    (tmp_path / "c.png").write_bytes(b"x")

    names = list_image_files(tmp_path)
    assert names == ["a.jpg", "c.png"]


def test_build_unlabeled_queue(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "one.jpg").write_bytes(b"x")
    (images_dir / "two.jpg").write_bytes(b"x")

    csv_path = tmp_path / "results.csv"
    _write_label_row(csv_path, image_name="one.jpg", ground_truth="Mouse")

    queue = build_queue(images_dir=images_dir, csv_path=csv_path, label_filter="unlabeled")
    assert queue == ["two.jpg"]

    labeled = load_labeled_names(csv_path)
    assert labeled == {"one.jpg"}


def test_queue_stats(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (images_dir / name).write_bytes(b"x")

    csv_path = tmp_path / "results.csv"
    _write_label_row(csv_path, image_name="a.jpg", ground_truth="Mouse")

    total, labeled, remaining = queue_stats(
        images_dir=images_dir,
        csv_path=csv_path,
        label_filter="unlabeled",
    )
    assert total == 3
    assert labeled == 1
    assert remaining == 2
