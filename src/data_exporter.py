"""CSV export module for classification results."""

import os
from pathlib import Path

import pandas as pd

from src import config
from src.models import Decision, DetectionResult


class DataExporter:
    """Exports classification results to CSV."""

    COLUMNS = [
        "timestamp", "category", "color", "size_category",
        "confidence", "method", "area_pixels", "aspect_ratio",
        "color_hsv", "color_kmeans", "confidence_hsv", "confidence_kmeans",
    ]

    def __init__(self, output_path: str | None = None) -> None:
        self.output_path = output_path or config.CSV_OUTPUT_PATH
        # Ensure output directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def export(self, decision: Decision, detection: DetectionResult) -> None:
        """Write a single classification result to CSV.

        Args:
            decision: Final classification decision.
            detection: Detection result with color details.
        """
        row = {
            "timestamp": decision.timestamp,
            "category": decision.category,
            "color": decision.color,
            "size_category": decision.size,
            "confidence": round(decision.confidence, 3),
            "method": decision.method_used,
            "area_pixels": detection.area_pixels,
            "aspect_ratio": detection.aspect_ratio,
            "color_hsv": detection.color_hsv.name,
            "color_kmeans": detection.color_kmeans.name,
            "confidence_hsv": round(detection.color_hsv.confidence, 3),
            "confidence_kmeans": round(detection.color_kmeans.confidence, 3),
        }
        df = pd.DataFrame([row], columns=self.COLUMNS)

        write_header = not os.path.exists(self.output_path)
        df.to_csv(self.output_path, mode="a", header=write_header, index=False)
