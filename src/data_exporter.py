"""CSV export module for classification results."""

import os
from pathlib import Path

import pandas as pd

from src import config
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
    ]

    def __init__(self, output_path: str | None = None) -> None:
        self.output_path = output_path or config.CSV_OUTPUT_PATH
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        decision: Decision,
        detection: DetectionResult,
        image_name: str = "",
    ) -> None:
        """Write a single classification result to CSV."""
        row = self._build_row(decision, detection, image_name)
        self._append_rows([row])

    def export_many(
        self,
        decisions: list[Decision],
        detections: list[DetectionResult],
        image_name: str = "",
    ) -> None:
        """Write multiple classification results to CSV."""
        detection_by_id = {
            detection.object_id: detection
            for detection in detections
        }

        rows = []

        for decision in decisions:
            detection = detection_by_id.get(decision.object_id)

            if detection is not None:
                rows.append(self._build_row(decision, detection, image_name))

        if rows:
            self._append_rows(rows)

    def _build_row(
        self,
        decision: Decision,
        detection: DetectionResult,
        image_name: str,
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