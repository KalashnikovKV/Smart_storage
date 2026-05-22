"""Pipeline protocols — Segmenter and Classifier abstractions."""

from typing import Protocol, runtime_checkable

import numpy as np

from src.models import Decision, DetectionResult, YOLODetection


@runtime_checkable
class Segmenter(Protocol):
    """Structural protocol for any instance segmenter (e.g. YOLO-Seg)."""

    def segment(self, image: np.ndarray) -> list[YOLODetection]: ...


@runtime_checkable
class Classifier(Protocol):
    """Structural protocol for any IT-peripheral classifier."""

    def classify(
        self,
        detection: DetectionResult,
        roi: np.ndarray | None = None,
    ) -> Decision: ...
