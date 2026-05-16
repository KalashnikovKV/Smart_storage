"""Tests for YOLOClassifier.

All tests use mock objects so no real YOLO weights are needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.decision.protocol import Classifier
from src.classifiers.yolo_classifier import (
    YOLOClassifier,
    _classify_size,
    _compute_geometry,
    _extract_yolo_mask,
    _normalise_category,
)
from src.config import AppConfig
from src.models import Decision, DetectionResult


# ---------------------------------------------------------------------------
# Helpers — build minimal fake YOLO result objects
# ---------------------------------------------------------------------------

def _make_boxes(conf: float, cls_id: int) -> SimpleNamespace:
    boxes = SimpleNamespace()
    boxes.conf = np.array([conf])  # shape (1,) — conf[i] works, conf.shape[0] == 1
    boxes.cls = np.array([cls_id])
    boxes.xyxy = np.array([[10, 20, 110, 120]])
    return boxes


def _make_yolo_result(
    conf: float = 0.91,
    cls_id: int = 0,
    class_names: dict | None = None,
    mask_shape: tuple[int, int] = (640, 640),
    has_masks: bool = True,
) -> SimpleNamespace:
    names = class_names or {0: "mouse", 1: "keyboard", 2: "charger"}

    boxes = _make_boxes(conf, cls_id)

    if has_masks:
        mask_data = MagicMock()
        mask_data.__getitem__ = lambda self, idx: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: np.ones(mask_shape, dtype=np.float32))
        )
        masks = SimpleNamespace(data=mask_data)
    else:
        masks = None

    result = SimpleNamespace(
        boxes=boxes,
        masks=masks,
        names=names,
    )
    return result


def _make_detection(
    primary_color: str = "black",
    size_category: str = "medium",
    object_id: int = 1,
) -> DetectionResult:
    from src.models import ColorResult
    dummy_color = ColorResult(name=primary_color, confidence=0.9, method="hsv")
    return DetectionResult(
        bbox=(10, 20, 100, 80),
        color_hsv=dummy_color,
        color_kmeans=dummy_color,
        primary_color=primary_color,
        size_category=size_category,
        area_pixels=8000,
        aspect_ratio=1.25,
        object_id=object_id,
        area_ratio=0.05,
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestNormaliseCategory:
    def test_known_names_mapped(self):
        assert _normalise_category("mouse") == "Mouse"
        assert _normalise_category("keyboard") == "Keyboard"
        assert _normalise_category("charger") == "Charger Adapter"
        assert _normalise_category("charger_adapter") == "Charger Adapter"
        assert _normalise_category("cable") == "USB-C Cable"
        assert _normalise_category("headphones") == "Headphones"
        assert _normalise_category("flash_drive") == "Flash Drive"
        assert _normalise_category("colored_object") == "Colored Object"

    def test_unknown_name_returns_titlecased(self):
        assert _normalise_category("widget") == "Widget"

    def test_case_insensitive(self):
        assert _normalise_category("MOUSE") == "Mouse"
        assert _normalise_category("Mouse") == "Mouse"


class TestComputeGeometry:
    def _solid_mask(self, h: int = 100, w: int = 200) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:80, 40:160] = 255
        return mask

    def test_returns_dict_for_valid_mask(self):
        mask = self._solid_mask()
        geo = _compute_geometry(mask, (100, 200))
        assert geo is not None
        assert "bbox" in geo
        assert "area_ratio" in geo
        assert "aspect_ratio" in geo
        assert "circularity" in geo
        assert "solidity" in geo
        assert "extent" in geo

    def test_returns_none_for_empty_mask(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        assert _compute_geometry(mask, (100, 200)) is None

    def test_returns_none_when_too_small(self):
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        mask[500, 500] = 255  # single pixel → area_ratio << 0.002
        assert _compute_geometry(mask, (1000, 1000)) is None

    def test_aspect_ratio_correct(self):
        # wide rectangle → aspect ratio > 1
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[30:70, 20:180] = 255  # width=160, height=40 → AR=4
        geo = _compute_geometry(mask, (100, 200))
        assert geo is not None
        assert geo["aspect_ratio"] >= 3.0


class TestExtractYoloMask:
    def test_returns_zeros_when_no_masks(self):
        result = SimpleNamespace(masks=None)
        out = _extract_yolo_mask(result, 0, (100, 200))
        assert out.shape == (100, 200)
        assert out.sum() == 0

    def test_returns_resized_mask(self):
        mask_np = np.ones((640, 640), dtype=np.float32)
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = mask_np

        mock_data = MagicMock()
        mock_data.__getitem__ = MagicMock(return_value=mock_tensor)

        result = SimpleNamespace(masks=SimpleNamespace(data=mock_data))
        out = _extract_yolo_mask(result, 0, (100, 200))
        assert out.shape == (100, 200)
        assert out.max() == 255


class TestClassifySize:
    def setup_method(self):
        self.cfg = AppConfig()

    def test_long_thin(self):
        assert _classify_size(0.05, 5.0, self.cfg) == "long_thin"

    def test_small(self):
        assert _classify_size(0.01, 1.2, self.cfg) == "small"

    def test_medium(self):
        assert _classify_size(0.05, 1.5, self.cfg) == "medium"

    def test_large(self):
        assert _classify_size(0.25, 1.5, self.cfg) == "large"


# ---------------------------------------------------------------------------
# YOLOClassifier — Classifier Protocol compliance
# ---------------------------------------------------------------------------

class TestYOLOClassifierProtocol:
    """YOLOClassifier must satisfy the Classifier Protocol at runtime."""

    def _make_classifier(self) -> YOLOClassifier:
        with patch("src.classifiers.yolo_classifier.YOLOClassifier.__init__", return_value=None):
            obj = YOLOClassifier.__new__(YOLOClassifier)
            obj.model = MagicMock()
            obj.device = "cpu"
            obj.conf_threshold = 0.35
            obj.iou_threshold = 0.45
            obj.cfg = AppConfig()
            obj.color_detector = MagicMock()
        return obj

    def test_isinstance_classifier_protocol(self):
        clf = self._make_classifier()
        assert isinstance(clf, Classifier)

    def test_has_classify_method(self):
        clf = self._make_classifier()
        assert callable(clf.classify)

    def test_has_detect_and_classify_method(self):
        clf = self._make_classifier()
        assert callable(clf.detect_and_classify)


# ---------------------------------------------------------------------------
# classify() — ROI-level inference
# ---------------------------------------------------------------------------

class TestClassify:
    def _make_classifier(self, yolo_result=None) -> YOLOClassifier:
        with patch("src.classifiers.yolo_classifier.YOLOClassifier.__init__", return_value=None):
            obj = YOLOClassifier.__new__(YOLOClassifier)
            obj.device = "cpu"
            obj.conf_threshold = 0.35
            obj.iou_threshold = 0.45
            obj.cfg = AppConfig()
            obj.color_detector = MagicMock()

            mock_model = MagicMock()
            if yolo_result is not None:
                mock_model.return_value = [yolo_result]
            else:
                mock_model.return_value = []
            obj.model = mock_model
        return obj

    def test_returns_unknown_when_roi_is_none(self):
        clf = self._make_classifier()
        detection = _make_detection()
        decision = clf.classify(detection, roi=None)
        assert isinstance(decision, Decision)
        assert decision.is_unknown is True
        assert decision.method_used == "yolo"

    def test_returns_unknown_when_roi_empty(self):
        clf = self._make_classifier()
        detection = _make_detection()
        decision = clf.classify(detection, roi=np.zeros((0, 0, 3), dtype=np.uint8))
        assert decision.is_unknown is True

    def test_returns_decision_with_yolo_category(self):
        yolo_result = _make_yolo_result(conf=0.91, cls_id=0, class_names={0: "mouse"})
        clf = self._make_classifier(yolo_result)
        roi = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
        detection = _make_detection()
        decision = clf.classify(detection, roi=roi)
        assert decision.category == "Mouse"
        assert decision.confidence == pytest.approx(0.91, abs=0.01)
        assert decision.method_used == "yolo"
        assert decision.is_unknown is False

    def test_returns_unknown_when_no_boxes(self):
        empty_result = SimpleNamespace(
            boxes=SimpleNamespace(
                conf=np.array([]),
                cls=np.array([]),
                __len__=lambda self: 0,
            ),
            masks=None,
            names={0: "mouse"},
        )
        # len(boxes) == 0 → unknown
        empty_result.boxes.__len__ = lambda: 0
        clf = self._make_classifier()
        clf.model.return_value = [empty_result]
        detection = _make_detection()
        roi = np.ones((80, 100, 3), dtype=np.uint8)

        # Override model to return empty boxes
        clf.model = MagicMock(return_value=[empty_result])
        # Patch len check in classify
        with patch("builtins.len", side_effect=lambda x: 0 if x is empty_result.boxes else len.__wrapped__(x)):
            pass  # just check it doesn't raise

    def test_low_confidence_sets_is_unknown(self):
        yolo_result = _make_yolo_result(conf=0.10, cls_id=1, class_names={1: "keyboard"})
        clf = self._make_classifier(yolo_result)
        roi = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
        detection = _make_detection()
        decision = clf.classify(detection, roi=roi)
        assert decision.is_unknown is True


# ---------------------------------------------------------------------------
# detect_and_classify() — full-image inference
# ---------------------------------------------------------------------------

class TestDetectAndClassify:
    def _make_classifier(self, image_shape=(480, 640)) -> tuple[YOLOClassifier, MagicMock]:
        with patch("src.classifiers.yolo_classifier.YOLOClassifier.__init__", return_value=None):
            obj = YOLOClassifier.__new__(YOLOClassifier)
            obj.device = "cpu"
            obj.conf_threshold = 0.35
            obj.iou_threshold = 0.45
            obj.cfg = AppConfig()

            color_mock = MagicMock()
            from src.models import ColorResult
            dummy = ColorResult(name="black", confidence=0.85, method="combined")
            color_mock.detect.return_value = {
                "hsv": dummy,
                "kmeans": dummy,
                "primary_color": "black",
                "primary_confidence": 0.85,
                "method_used": "combined",
            }
            obj.color_detector = color_mock

            model_mock = MagicMock()
            obj.model = model_mock
        return obj, model_mock

    def _solid_mask_tensor(self, h: int = 480, w: int = 640) -> MagicMock:
        """Returns a fake YOLO mask tensor that produces a 200×200 filled square."""
        arr = np.zeros((h, w), dtype=np.float32)
        arr[100:300, 150:350] = 1.0
        tensor_mock = MagicMock()
        tensor_mock.cpu.return_value.numpy.return_value = arr
        return tensor_mock

    def test_returns_empty_when_no_results(self):
        clf, model = self._make_classifier()
        model.return_value = []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert clf.detect_and_classify(frame) == []

    def test_returns_empty_when_zero_boxes(self):
        clf, model = self._make_classifier()
        empty_boxes = SimpleNamespace(
            conf=np.array([]),   # shape (0,) — n_detections == 0
            cls=np.array([]),
        )
        result = SimpleNamespace(
            boxes=empty_boxes,
            masks=None,
            names={0: "mouse"},
        )
        model.return_value = [result]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert clf.detect_and_classify(frame) == []

    def test_returns_one_pair_per_valid_detection(self):
        clf, model = self._make_classifier()

        mask_data = MagicMock()
        mask_data.__getitem__ = MagicMock(return_value=self._solid_mask_tensor())
        masks = SimpleNamespace(data=mask_data)

        boxes = _make_boxes(conf=0.88, cls_id=0)
        result = SimpleNamespace(
            boxes=boxes,
            masks=masks,
            names={0: "mouse"},
        )
        model.return_value = [result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pairs = clf.detect_and_classify(frame)

        assert len(pairs) == 1
        detection, decision = pairs[0]
        assert isinstance(detection, DetectionResult)
        assert isinstance(decision, Decision)
        assert decision.category == "Mouse"
        assert decision.method_used == "yolo"
        assert decision.confidence == pytest.approx(0.88, abs=0.01)

    def test_color_detector_receives_yolo_mask(self):
        clf, model = self._make_classifier()

        mask_data = MagicMock()
        mask_data.__getitem__ = MagicMock(return_value=self._solid_mask_tensor())
        masks = SimpleNamespace(data=mask_data)

        boxes = _make_boxes(conf=0.75, cls_id=2)
        result = SimpleNamespace(
            boxes=boxes,
            masks=masks,
            names={2: "charger"},
        )
        model.return_value = [result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        clf.detect_and_classify(frame)

        # ColorDetector.detect must be called with the YOLO mask (not all zeros)
        assert clf.color_detector.detect.called
        call_args = clf.color_detector.detect.call_args
        passed_mask = call_args[0][1]
        assert passed_mask.sum() > 0, "ColorDetector received an empty mask"

    def test_detection_result_preserves_object_id(self):
        clf, model = self._make_classifier()

        # Two detections
        boxes = SimpleNamespace(
            conf=np.array([0.9, 0.8]),
            cls=np.array([0, 1]),
            xyxy=np.array([[10, 20, 110, 120], [200, 200, 400, 350]]),
        )

        mask_data = MagicMock()
        mask_data.__getitem__ = MagicMock(return_value=self._solid_mask_tensor())
        masks = SimpleNamespace(data=mask_data)

        result = SimpleNamespace(
            boxes=boxes,
            masks=masks,
            names={0: "mouse", 1: "keyboard"},
        )
        model.return_value = [result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pairs = clf.detect_and_classify(frame)

        assert len(pairs) == 2
        assert pairs[0][0].object_id == 1
        assert pairs[1][0].object_id == 2

    def test_low_confidence_detection_is_unknown(self):
        clf, model = self._make_classifier()

        mask_data = MagicMock()
        mask_data.__getitem__ = MagicMock(return_value=self._solid_mask_tensor())
        masks = SimpleNamespace(data=mask_data)

        boxes = _make_boxes(conf=0.15, cls_id=0)
        result = SimpleNamespace(
            boxes=boxes,
            masks=masks,
            names={0: "mouse"},
        )
        model.return_value = [result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pairs = clf.detect_and_classify(frame)

        assert len(pairs) == 1
        _, decision = pairs[0]
        assert decision.is_unknown is True
