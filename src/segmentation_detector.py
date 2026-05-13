"""YOLO-Seg instance segmentation detector.

Runs a trained YOLO-Seg model on an image and returns raw per-object results:
    class_id, category, confidence, bbox (x, y, w, h), binary mask.

Color and size analysis are NOT done here — that responsibility belongs to
Pipeline, which calls ColorDetector with the returned mask.

Usage:
    detector = SegmentationDetector("models/yolo_segmentation_best.pt")
    detections = detector.detect(image)
    for d in detections:
        color = color_detector.detect(image, d["mask"])
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class SegmentationDetector:
    """Wraps a YOLO-Seg model and returns per-object class + bbox + mask."""

    def __init__(
        self,
        model_path: str = "models/yolo_segmentation_best.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # lazy import — optional dependency

        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, image: np.ndarray) -> list[dict]:
        """Run inference and return a list of detections sorted by confidence.

        Each detection dict contains:
            class_id   (int)
            category   (str)   — class name from dataset.yaml
            confidence (float) — 0.0–1.0
            bbox       (tuple) — (x, y, w, h) in pixels
            mask       (np.ndarray uint8) — binary mask, same size as image,
                                            255 = object pixel, 0 = background
        """
        results = self.model.predict(
            image,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []
        image_height, image_width = image.shape[:2]

        for result in results:
            if result.boxes is None or result.masks is None:
                LOGGER.debug("No boxes or masks in YOLO result — skipping.")
                continue

            for index, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Resize YOLO mask to original image resolution
                mask_data = result.masks.data[index].cpu().numpy()
                mask = cv2.resize(
                    mask_data,
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask = (mask > 0.5).astype(np.uint8) * 255

                detections.append(
                    {
                        "class_id": class_id,
                        "category": result.names[class_id],
                        "confidence": round(confidence, 3),
                        "bbox": (
                            int(x1),
                            int(y1),
                            int(x2 - x1),
                            int(y2 - y1),
                        ),
                        "mask": mask,
                    }
                )

        return sorted(detections, key=lambda d: d["confidence"], reverse=True)
