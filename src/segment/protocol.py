"""Segmenter protocol — abstraction over instance segmentation."""

from typing import Protocol, runtime_checkable

import numpy as np

from src.models import YOLODetection


@runtime_checkable
class Segmenter(Protocol):
    """Structural protocol for any instance segmenter (e.g. YOLO-Seg)."""

    def segment(self, image: np.ndarray) -> list[YOLODetection]: ...
