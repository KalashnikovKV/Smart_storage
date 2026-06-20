"""Tests for Segmenter protocol and YOLOSegmenter (mocked — no weights)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.yolo_segmenter import (
    YOLOSegmenter,
    _parse_results,
    combine_instance_masks,
    consolidate_best_per_class,
    deduplicate_overlapping_masks,
    detection_area_ratio,
    filter_preferred_classes,
    mask_iou,
    reclassify_by_geometry,
    trim_low_confidence_detections,
)
from src.models import YOLODetection
from src.pipeline.protocols import Segmenter


def _make_yolo_result(
    conf: float = 0.88,
    cls_id: int = 0,
    class_names: dict | None = None,
) -> SimpleNamespace:
    names = class_names or {0: "mouse"}

    box = SimpleNamespace()
    box.cls = np.array([cls_id])
    box.conf = np.array([conf])
    box.xyxy = np.array([[10.0, 20.0, 110.0, 120.0]])

    mask_tensor = MagicMock()
    mask_tensor.cpu.return_value.numpy.return_value = np.ones((64, 64), dtype=np.float32)

    masks = SimpleNamespace(data=[mask_tensor])

    return SimpleNamespace(boxes=[box], masks=masks, names=names)


def test_yolo_segmenter_satisfies_segmenter_protocol():
    segmenter = YOLOSegmenter.__new__(YOLOSegmenter)
    assert isinstance(segmenter, Segmenter)


def test_parse_results_returns_yolo_detection_dataclasses():
    result = _make_yolo_result()
    detections = _parse_results([result], (480, 640))

    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, YOLODetection)
    assert det.class_name == "mouse"
    assert det.confidence == pytest.approx(0.88)
    assert det.bbox == (10, 20, 100, 100)
    assert det.mask.shape == (480, 640)
    assert set(np.unique(det.mask)).issubset({0, 255})


@patch("ultralytics.YOLO")
def test_segment_delegates_to_model(mock_yolo_class):
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model
    mock_model.predict.return_value = [_make_yolo_result()]

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    segmenter = YOLOSegmenter("fake.pt", device="cpu")
    detections = segmenter.segment(image)

    mock_model.predict.assert_called_once()
    assert len(detections) == 1
    assert detections[0].class_name == "mouse"


def test_mask_iou_for_identical_masks():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    assert mask_iou(mask, mask) == pytest.approx(1.0)


def test_deduplicate_overlapping_masks_keeps_highest_confidence():
    base = np.zeros((120, 120), dtype=np.uint8)
    base[20:100, 20:100] = 255

    overlap = base.copy()
    overlap[30:90, 30:90] = 255

    separate = np.zeros((120, 120), dtype=np.uint8)
    separate[5:15, 5:15] = 255

    detections = [
        YOLODetection("charger", (20, 20, 80, 80), base, 0.997),
        YOLODetection("charger", (30, 30, 60, 60), overlap, 0.607),
        YOLODetection("mouse", (5, 5, 10, 10), separate, 0.900),
    ]

    filtered = deduplicate_overlapping_masks(detections, min_mask_iou=0.80)

    assert len(filtered) == 2
    assert filtered[0].confidence == pytest.approx(0.997)
    assert filtered[0].class_name == "charger"
    assert filtered[1].class_name == "mouse"


def test_combine_instance_masks_unions_all_objects():
    mask_a = np.zeros((80, 80), dtype=np.uint8)
    mask_a[10:30, 10:30] = 255

    mask_b = np.zeros((80, 80), dtype=np.uint8)
    mask_b[50:70, 50:70] = 255

    detections = [
        YOLODetection("mouse", (10, 10, 20, 20), mask_a, 0.9),
        YOLODetection("mouse", (50, 50, 20, 20), mask_b, 0.8),
    ]

    combined = combine_instance_masks(detections, (80, 80))

    assert combined[15, 15] == 255
    assert combined[60, 60] == 255
    assert combined[40, 40] == 0


def test_filter_preferred_classes_keeps_headphones_over_charger():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:50, 10:50] = 255

    detections = [
        YOLODetection("charger", (10, 10, 40, 40), mask, 0.95),
        YOLODetection("headphones", (12, 12, 36, 36), mask, 0.12),
    ]

    filtered = filter_preferred_classes(
        detections,
        ["headphones"],
        min_conf=0.05,
        strict=False,
    )

    assert len(filtered) == 1
    assert filtered[0].class_name == "headphones"


def test_filter_preferred_classes_strict_returns_empty_when_missing():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    detections = [YOLODetection("charger", (10, 10, 40, 40), mask, 0.95)]

    filtered = filter_preferred_classes(
        detections,
        ["headphones"],
        strict=True,
    )

    assert filtered == []


def test_trim_low_confidence_detections_keeps_top_three_objects():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    detections = [
        YOLODetection("charger", (10, 10, 40, 40), mask, 0.99),
        YOLODetection("headphones", (12, 12, 36, 36), mask, 0.63),
        YOLODetection("mouse", (20, 20, 20, 20), mask, 0.41),
        YOLODetection("headphones", (15, 15, 30, 30), mask, 0.12),
    ]

    trimmed = trim_low_confidence_detections(
        detections,
        max_objects=8,
        min_confidence=0.15,
    )

    assert len(trimmed) == 3
    assert [det.class_name for det in trimmed] == ["charger", "headphones", "mouse"]


def test_reclassify_by_geometry_maps_large_flash_drive_to_mouse():
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[400:700, 400:700] = 255
    detections = [
        YOLODetection("flash_drive", (400, 400, 300, 300), mask, 0.50),
    ]

    corrected = reclassify_by_geometry(detections, (1000, 1000))

    assert corrected[0].class_name == "mouse"
    assert detection_area_ratio(corrected[0], (1000, 1000)) >= 0.032


def test_trim_low_confidence_detections_drops_tiny_secondary_objects():
    full_mask = np.zeros((1000, 1000), dtype=np.uint8)
    full_mask[100:500, 100:500] = 255
    tiny_mask = np.zeros((1000, 1000), dtype=np.uint8)
    tiny_mask[990:998, 990:998] = 255
    detections = [
        YOLODetection("charger", (100, 100, 400, 400), full_mask, 0.99),
        YOLODetection("headphones", (100, 100, 400, 400), full_mask, 0.70),
        YOLODetection("charger", (990, 990, 8, 8), tiny_mask, 0.24),
    ]

    trimmed = trim_low_confidence_detections(
        detections,
        min_confidence=0.15,
        min_area_ratio=0.008,
        image_shape=(1000, 1000),
    )

    assert len(trimmed) == 2


def test_consolidate_best_per_class_keeps_one_detection_per_label():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    detections = [
        YOLODetection("headphones", (10, 10, 40, 40), mask, 0.50),
        YOLODetection("headphones", (12, 12, 36, 36), mask, 0.30),
        YOLODetection("cable", (10, 10, 40, 40), mask, 0.09),
        YOLODetection("charger", (10, 10, 40, 40), mask, 0.99),
    ]

    consolidated = consolidate_best_per_class(detections)

    assert len(consolidated) == 3
    assert [det.class_name for det in consolidated] == ["charger", "headphones", "cable"]


def test_trim_low_confidence_allows_low_conf_cable():
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[100:500, 100:500] = 255
    detections = [
        YOLODetection("charger", (100, 100, 400, 400), mask, 0.99),
        YOLODetection("headphones", (100, 100, 400, 400), mask, 0.40),
        YOLODetection("cable", (100, 100, 400, 400), mask, 0.09),
    ]

    trimmed = trim_low_confidence_detections(
        consolidate_best_per_class(detections),
        min_confidence=0.15,
        min_area_ratio=0.008,
        image_shape=(1000, 1000),
    )

    assert len(trimmed) == 3
    assert trimmed[-1].class_name == "cable"
