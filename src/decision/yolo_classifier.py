"""YOLO-Seg classifier — wraps ultralytics YOLO behind the Classifier Protocol.

Flow per frame:
    image
    → YOLO-Seg inference  (class + bbox + mask + confidence)
    → ColorDetector.detect(image, yolo_mask)   (HSV + K-means inside mask)
    → (DetectionResult, Decision)

The Classifier Protocol interface (classify) is also supported for ROI-level
inference when the caller already holds a cropped image of one object.
"""

from __future__ import annotations

import logging
from datetime import datetime

import cv2
import numpy as np

from src.config import AppConfig
from src.decision.protocol import Classifier  # noqa: F401 — satisfies Protocol
from src.detect.color import ColorDetector
from src.models import Decision, DetectionResult

LOGGER = logging.getLogger(__name__)

# Maps YOLO training class names → canonical app category strings.
# Keys must match names used in dataset.yaml exactly (lowercase).
_CLASS_NAME_MAP: dict[str, str] = {
    "mouse": "Mouse",
    "keyboard": "Keyboard",
    "charger": "Charger Adapter",
    "charger_adapter": "Charger Adapter",
    "cable": "USB-C Cable",
    "usb_cable": "USB-C Cable",
    "usb-c_cable": "USB-C Cable",
    "headphones": "Headphones",
    "flash_drive": "Flash Drive",
    "flashdrive": "Flash Drive",
    "colored_object": "Colored Object",
}

_LOW_CONF = 0.25


def _normalise_category(raw: str) -> str:
    """Map a raw YOLO class name to the canonical category string."""
    return _CLASS_NAME_MAP.get(raw.lower().replace(" ", "_"), raw.title())


def _extract_yolo_mask(
    result: object,
    idx: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Return a binary uint8 mask (255/0) resized to image_shape (h, w)."""
    h, w = image_shape
    if result.masks is None:  # type: ignore[union-attr]
        return np.zeros((h, w), dtype=np.uint8)

    mask_tensor = result.masks.data[idx]  # type: ignore[index]
    mask_np = mask_tensor.cpu().numpy().astype(np.float32)
    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 0.5).astype(np.uint8) * 255


def _compute_geometry(
    mask: np.ndarray,
    image_shape: tuple[int, int],
) -> dict | None:
    """Compute geometric features from a binary mask.

    Returns None if the mask is empty or the object is implausibly large/small.
    """
    h, w = image_shape
    frame_area = h * w

    area_pixels = int(np.sum(mask > 0))
    if area_pixels <= 0:
        return None

    area_ratio = area_pixels / frame_area
    if area_ratio < 0.002 or area_ratio > 0.70:
        return None

    points = cv2.findNonZero(mask)
    if points is None:
        return None

    x, y, bw, bh = cv2.boundingRect(points)
    aspect_ratio = float(max(bw, bh)) / max(min(bw, bh), 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_points = np.vstack(contours)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    circularity = (
        float((4 * np.pi * area_pixels) / (perimeter * perimeter))
        if perimeter > 0
        else 0.0
    )

    hull = cv2.convexHull(all_points)
    hull_area = float(cv2.contourArea(hull))
    solidity = area_pixels / hull_area if hull_area > 0 else 0.0

    bbox_area = bw * bh
    extent = area_pixels / bbox_area if bbox_area > 0 else 0.0

    display_contour = hull

    return {
        "bbox": (x, y, bw, bh),
        "area_pixels": area_pixels,
        "area_ratio": area_ratio,
        "aspect_ratio": round(aspect_ratio, 2),
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
        "extent": round(extent, 3),
        "bbox_width_ratio": round(bw / w, 4),
        "bbox_height_ratio": round(bh / h, 4),
        "contour": display_contour,
    }


def _classify_size(area_ratio: float, aspect_ratio: float, cfg: AppConfig) -> str:
    """Map area_ratio + aspect_ratio to a size category string."""
    if aspect_ratio > cfg.long_thin_aspect:
        return "long_thin"
    if area_ratio < cfg.small_max_ratio:
        return "small"
    if area_ratio < cfg.medium_max_ratio:
        return "medium"
    return "large"


class YOLOClassifier:
    """YOLO-Seg based classifier that satisfies the Classifier Protocol.

    Usage — full-image inference (recommended):
        classifier = YOLOClassifier("models/yolo_seg_best.pt")
        pairs = classifier.detect_and_classify(frame)
        for detection, decision in pairs:
            ...

    Usage — ROI-level inference (Classifier Protocol):
        decision = classifier.classify(detection, roi=roi_image)
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        config: AppConfig | None = None,
    ) -> None:
        from ultralytics import YOLO  # lazy import — ultralytics is optional dep

        self.model = YOLO(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.cfg = config or AppConfig()
        self.color_detector = ColorDetector(self.cfg)

    def classify(
        self,
        detection: DetectionResult,
        roi: np.ndarray | None = None,
    ) -> Decision:
        """Classify a single object.

        If *roi* is provided the model runs inference on it to determine the
        category.  Otherwise the detection's existing color/size fields are
        used and category falls back to "Unknown Object".
        """
        if roi is None or roi.size == 0:
            LOGGER.warning("YOLOClassifier.classify called without roi — returning Unknown")
            return Decision(
                category="Unknown Object",
                confidence=0.0,
                color=detection.primary_color,
                size=detection.size_category,
                method_used="yolo",
                is_unknown=True,
                object_id=detection.object_id,
            )

        results = self.model(
            roi,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        boxes = results[0].boxes if results else None
        if boxes is None or boxes.conf.shape[0] == 0:
            return Decision(
                category="Unknown Object",
                confidence=0.0,
                color=detection.primary_color,
                size=detection.size_category,
                method_used="yolo",
                is_unknown=True,
                object_id=detection.object_id,
            )

        best_idx = int(boxes.conf.argmax())
        conf = float(boxes.conf[best_idx])
        cls_id = int(boxes.cls[best_idx])
        category = _normalise_category(results[0].names[cls_id])

        return Decision(
            category=category,
            confidence=round(conf, 3),
            color=detection.primary_color,
            size=detection.size_category,
            method_used="yolo",
            is_unknown=conf < _LOW_CONF,
            closest_match="" if conf >= _LOW_CONF else category,
            object_id=detection.object_id,
        )

    def detect_and_classify(
        self,
        image: np.ndarray,
    ) -> list[tuple[DetectionResult, Decision]]:
        """Run YOLO-Seg on *image* and return one (DetectionResult, Decision) per object.

        ColorDetector runs inside each YOLO mask for accurate color estimation
        that is independent of background.
        """
        results = self.model(
            image,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        n_detections = result.boxes.conf.shape[0]

        if n_detections == 0:
            return []

        image_shape = image.shape[:2]
        output: list[tuple[DetectionResult, Decision]] = []

        for i in range(n_detections):
            conf = float(result.boxes.conf[i])
            cls_id = int(result.boxes.cls[i])
            category = _normalise_category(result.names[cls_id])

            object_mask = _extract_yolo_mask(result, i, image_shape)

            geo = _compute_geometry(object_mask, image_shape)
            if geo is None:
                LOGGER.debug("Skipping detection %d — mask geometry invalid", i)
                continue

            color_result = self.color_detector.detect(image, object_mask)

            size_category = _classify_size(
                geo["area_ratio"],
                geo["aspect_ratio"],
                self.cfg,
            )

            detection = DetectionResult(
                bbox=geo["bbox"],
                color_hsv=color_result["hsv"],
                color_kmeans=color_result["kmeans"],
                primary_color=color_result["primary_color"],
                size_category=size_category,
                area_pixels=geo["area_pixels"],
                aspect_ratio=geo["aspect_ratio"],
                circularity=geo["circularity"],
                solidity=geo["solidity"],
                extent=geo["extent"],
                shape_category="",
                contour=geo["contour"],
                object_id=i + 1,
                area_ratio=geo["area_ratio"],
                bbox_width_ratio=geo["bbox_width_ratio"],
                bbox_height_ratio=geo["bbox_height_ratio"],
                visual_size_label=size_category,
                size_confidence=round(conf, 3),
            )

            decision = Decision(
                category=category,
                confidence=round(conf, 3),
                color=color_result["primary_color"],
                size=size_category,
                method_used="yolo",
                is_unknown=conf < _LOW_CONF,
                closest_match="" if conf >= _LOW_CONF else category,
                object_id=i + 1,
                timestamp=datetime.now().isoformat(),
            )

            output.append((detection, decision))

        return output
