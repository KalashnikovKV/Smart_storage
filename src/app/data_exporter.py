"""CSV export module for classification results."""

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import AppConfig
from src.models import Decision, DetectionResult


class DataExporter:
    """Exports classification results to CSV."""

    COLUMNS = [
        "timestamp",
        "image_name",
        "object_id",
        "category",
        "color",
        "size_category",
        "confidence",
        "method",
        "area_pixels",
        "aspect_ratio",
        "shape_category",
        "circularity",
        "solidity",
        "extent",
        "color_hsv",
        "color_kmeans",
        "confidence_hsv",
        "confidence_kmeans",
        "roi_image",
        "ground_truth",
    ]

    def __init__(
        self,
        output_path: str | None = None,
        config: AppConfig | None = None,
    ) -> None:
        cfg = config or AppConfig()
        self.output_path = output_path or cfg.csv_output_path
        self.save_roi_images = cfg.save_roi_images
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.images_dir = Path(self.output_path).parent / "images"
        if self.save_roi_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        decision: Decision,
        detection: DetectionResult,
        image_name: str = "",
        roi: np.ndarray | None = None,
        ground_truth: str = "",
    ) -> None:
        """Write a single classification result to CSV and save ROI if enabled."""
        roi_image_name = self._save_roi(roi) if (roi is not None and self.save_roi_images) else ""
        row = self._build_row(
            decision,
            detection,
            image_name,
            roi_image_name,
            ground_truth=ground_truth,
        )
        self._append_rows([row])

    def export_many(
        self,
        decisions: list[Decision],
        detections: list[DetectionResult],
        image_name: str = "",
        original_image: np.ndarray | None = None,
        ground_truth: str = "",
    ) -> None:
        """Write multiple classification results to CSV, saving one ROI per detection."""
        detection_by_id = {
            detection.object_id: detection
            for detection in detections
        }

        rows = []

        for decision in decisions:
            detection = detection_by_id.get(decision.object_id)

            if detection is not None:
                roi_image_name = ""
                if original_image is not None and self.save_roi_images:
                    roi = self._extract_roi(original_image, detection.bbox)
                    roi_image_name = self._save_roi(roi)
                rows.append(
                    self._build_row(
                        decision,
                        detection,
                        image_name,
                        roi_image_name,
                        ground_truth=ground_truth,
                    )
                )

        if rows:
            self._append_rows(rows)

    def _extract_roi(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop ROI from image using bbox, clamped to image bounds."""
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        return image[y1:y2, x1:x2]

    def _save_roi(self, roi: np.ndarray) -> str:
        """Save ROI image to images_dir and return the filename. Empty string on failure."""
        if roi is None or roi.size == 0:
            return ""
        img_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(self.images_dir / img_name), roi)
        return img_name

    def _build_row(
        self,
        decision: Decision,
        detection: DetectionResult,
        image_name: str,
        roi_image_name: str = "",
        ground_truth: str = "",
    ) -> dict:
        """Build one CSV row."""
        return {
            "timestamp": decision.timestamp,
            "image_name": image_name,
            "object_id": decision.object_id,
            "category": decision.category,
            "color": decision.color,
            "size_category": decision.size,
            "confidence": round(decision.confidence, 3),
            "method": decision.method_used,
            "area_pixels": detection.area_pixels,
            "aspect_ratio": detection.aspect_ratio,
            "shape_category": detection.shape_category,
            "circularity": detection.circularity,
            "solidity": detection.solidity,
            "extent": detection.extent,
            "color_hsv": detection.color_hsv.name,
            "color_kmeans": detection.color_kmeans.name,
            "confidence_hsv": round(detection.color_hsv.confidence, 3),
            "confidence_kmeans": round(detection.color_kmeans.confidence, 3),
            "roi_image": roi_image_name,
            "ground_truth": ground_truth,
        }

    def _append_rows(self, rows: list[dict]) -> None:
        """Append rows to CSV."""
        df = pd.DataFrame(rows, columns=self.COLUMNS)

        write_header = not os.path.exists(self.output_path)
        df.to_csv(
            self.output_path,
            mode="a",
            header=write_header,
            index=False,
        )