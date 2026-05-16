"""Backward-compatible re-export."""

from src.decision.yolo_classifier import (
    YOLOClassifier,
    _classify_size,
    _compute_geometry,
    _extract_yolo_mask,
    _normalise_category,
)

__all__ = [
    "YOLOClassifier",
    "_classify_size",
    "_compute_geometry",
    "_extract_yolo_mask",
    "_normalise_category",
]
