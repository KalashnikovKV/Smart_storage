"""YOLO-Seg segmenter — implements the Segmenter protocol.

Returns per-object class name, bbox, mask and confidence. Color and
classification in the full app are handled by Pipeline + ColorDetector.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.models import YOLODetection
from src.pipeline.protocols import Segmenter  # noqa: F401 — satisfies Protocol

LOGGER = logging.getLogger(__name__)


def _parse_results(results: object, image_shape: tuple[int, int]) -> list[YOLODetection]:
    """Convert ultralytics Results into YOLODetection instances."""
    image_height, image_width = image_shape
    detections: list[YOLODetection] = []

    for result in results:  # type: ignore[union-attr]
        if result.boxes is None or result.masks is None:
            LOGGER.debug("No boxes or masks in YOLO result — skipping.")
            continue

        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            mask_data = result.masks.data[index].cpu().numpy()
            mask = cv2.resize(
                mask_data,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask > 0.5).astype(np.uint8) * 255

            detections.append(
                YOLODetection(
                    class_name=result.names[class_id],
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    mask=mask,
                    confidence=round(confidence, 3),
                )
            )

    return sorted(detections, key=lambda d: d.confidence, reverse=True)


class YOLOSegmenter:
    """YOLO-Seg wrapper that satisfies the Segmenter protocol."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO  # lazy import — optional dependency
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Cannot load ultralytics/torch. Install YOLO deps with "
                "'uv sync --group yolo' (CPU torch). For GPU, see "
                "aidlc-docs/construction/build-and-test/build-instructions.md."
            ) from exc

        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def segment(self, image: np.ndarray) -> list[YOLODetection]:
        """Run YOLO-Seg and return detections sorted by confidence (desc)."""
        results = self.model.predict(
            image,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return _parse_results(results, image.shape[:2])
