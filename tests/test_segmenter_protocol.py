"""Tests for Segmenter protocol and YOLOSegmenter (mocked — no weights)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import YOLODetection
from src.segment.protocol import Segmenter
from src.segment.yolo import YOLOSegmenter, _parse_results


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
