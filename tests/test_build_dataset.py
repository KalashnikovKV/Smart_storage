"""Tests for YOLO-Seg dataset builder (training/build_dataset.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from training.build_dataset import (
    CATEGORY_TO_SLUG,
    YOLO_CLASS_NAMES,
    build_dataset_yaml,
    category_to_slug,
    deduplicate_rows,
    mask_to_polygon,
    polygon_to_yolo_line,
)
from training.train_yolo import validate_dataset


@pytest.fixture
def white_mask():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:80, 40:160] = 255
    return mask


def test_category_to_slug_maps_known_classes():
    assert category_to_slug("Mouse") == "mouse"
    assert category_to_slug("Charger Adapter") == "charger"
    assert category_to_slug("USB-C Cable") == "cable"
    assert category_to_slug("iPhone Charger") == "charger"
    assert category_to_slug("Unknown Object") is None


def test_normalise_slug_maps_legacy_aliases():
    from training.build_dataset import normalise_slug

    assert normalise_slug("charger_adapter") == "charger"
    assert normalise_slug("usb_cable") == "cable"


def test_mask_to_polygon_returns_normalized_points(white_mask):
    polygon = mask_to_polygon(white_mask, image_width=200, image_height=100)
    assert polygon is not None
    assert len(polygon) >= 3
    for x, y in polygon:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_polygon_to_yolo_line_format():
    line = polygon_to_yolo_line(0, [(0.1, 0.2), (0.5, 0.2), (0.5, 0.8)])
    parts = line.split()
    assert parts[0] == "0"
    assert len(parts) == 1 + 2 * 3


def test_deduplicate_prefers_ground_truth_row():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T10:00:00",
                "image_name": "a.jpg",
                "object_id": 1,
                "category": "Mouse",
                "ground_truth": "",
            },
            {
                "timestamp": "2026-01-01T11:00:00",
                "image_name": "a.jpg",
                "object_id": 1,
                "category": "Keyboard",
                "ground_truth": "Mouse",
            },
        ]
    )
    result = deduplicate_rows(df)
    assert len(result) == 1
    assert result.iloc[0]["ground_truth"] == "Mouse"


def test_build_dataset_yaml_names(tmp_path):
    output_dir = tmp_path / "dataset"
    yaml_path = build_dataset_yaml(output_dir, YOLO_CLASS_NAMES)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"][0] == "mouse"
    assert data["names"][2] == "charger"
    assert data["names"][3] == "cable"
    assert len(data["names"]) == 7


def test_end_to_end_dataset_build(tmp_path):
    """Build a tiny dataset from synthetic CSV, image, and cleaned mask."""
    from training.build_dataset import main as build_main

    images_dir = tmp_path / "frames"
    images_dir.mkdir()
    stages_dir = tmp_path / "stages" / "cleaned_mask"
    stages_dir.mkdir(parents=True)

    image_path = images_dir / "item.jpg"
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[30:90, 40:120] = (255, 255, 255)
    cv2.imwrite(str(image_path), image)

    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[30:90, 40:120] = 255
    cv2.imwrite(str(stages_dir / "item_cleaned_mask.jpg"), mask)

    csv_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-16T12:00:00",
                "image_name": "item.jpg",
                "object_id": 1,
                "category": "Mouse",
                "ground_truth": "Mouse",
                "confidence": 0.9,
            }
        ]
    ).to_csv(csv_path, index=False)

    dataset_dir = tmp_path / "dataset"
    exit_code = build_main(
        [
            "--csv",
            str(csv_path),
            "--images-dir",
            str(images_dir),
            "--stages-dir",
            str(tmp_path / "stages"),
            "--output",
            str(dataset_dir),
        ]
    )
    assert exit_code == 0

    yaml_path = dataset_dir / "dataset.yaml"
    assert yaml_path.is_file()

    train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    assert train_images
    assert train_labels

    label_line = train_labels[0].read_text(encoding="utf-8").strip()
    parts = label_line.split()
    assert parts[0] == "0"
    assert len(parts) >= 7

    image_count, label_count = validate_dataset(yaml_path)
    assert image_count >= 1
    assert label_count >= 1


def test_all_category_slugs_are_listed():
    for slug in set(CATEGORY_TO_SLUG.values()):
        assert slug in YOLO_CLASS_NAMES


def test_load_csv_rows_handles_mixed_legacy_and_extended(tmp_path):
    from training.build_dataset import load_csv_rows

    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "timestamp,category,color,size_category,confidence,method,area_pixels,aspect_ratio,color_hsv,color_kmeans,confidence_hsv,confidence_kmeans\n"
        "2026-04-24T13:54:03,Mouse,black,large,0.364,combined,716791,1.15,blue,black,0.353,0.405\n"
        "2026-05-02T20:41:29,Image_1.jpeg,1,USB-C Cable,black,large,0.738,combined,894462,3.47,rectangular,0.538,0.989,0.939,black,black,0.971,0.788\n",
        encoding="utf-8",
    )
    df = load_csv_rows(csv_path)
    assert len(df) == 2
    assert df.iloc[0]["category"] == "Mouse"
    assert df.iloc[0]["image_name"] == ""
    assert df.iloc[1]["image_name"] == "Image_1.jpeg"
    assert df.iloc[1]["category"] == "USB-C Cable"


def test_import_yolo_export(tmp_path):
    from training.build_dataset import import_yolo_export

    export_root = tmp_path / "roboflow"
    train_images = export_root / "train" / "images"
    train_labels = export_root / "train" / "labels"
    val_images = export_root / "valid" / "images"
    val_labels = export_root / "valid" / "labels"
    for directory in (train_images, train_labels, val_images, val_labels):
        directory.mkdir(parents=True)

    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:60, 20:60] = 255
    cv2.imwrite(str(train_images / "a.jpg"), image)
    cv2.imwrite(str(val_images / "b.jpg"), image)

    (train_labels / "a.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    (val_labels / "b.txt").write_text(
        "2 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    (export_root / "data.yaml").write_text(
        "names:\n  0: mouse\n  2: charger\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "dataset"
    train_counts, val_counts = import_yolo_export(export_root, output_dir)
    assert train_counts["mouse"] == 1
    assert val_counts["charger"] == 1
    assert (output_dir / "images" / "train" / "a.jpg").is_file()
    assert (output_dir / "labels" / "val" / "b.txt").read_text(encoding="utf-8").startswith("2 ")
