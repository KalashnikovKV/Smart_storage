"""Data models for the Smart Storage pipeline."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class ColorResult:
    """Result of color detection from a single method."""

    name: str
    confidence: float
    method: str
    rgb: tuple[int, int, int] = (0, 0, 0)


@dataclass
class DetectionResult:
    """Result of object detection including color, size, and shape analysis."""

    bbox: tuple[int, int, int, int]
    color_hsv: ColorResult
    color_kmeans: ColorResult
    primary_color: str
    size_category: str
    area_pixels: int
    aspect_ratio: float

    circularity: float = 0.0
    solidity: float = 0.0
    extent: float = 0.0
    shape_category: str = ""
    contour: np.ndarray | None = None

    object_id: int = 1
    edge_density: float = 0.0
    area_ratio: float = 0.0
    bbox_width_ratio: float = 0.0
    bbox_height_ratio: float = 0.0
    visual_size_label: str = ""
    size_confidence: float = 0.0


@dataclass
class Decision:
    """Final classification decision."""

    category: str
    confidence: float
    color: str
    size: str
    method_used: str
    is_unknown: bool = False
    closest_match: str = ""
    object_id: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PipelineResult:
    """Complete result of the CV pipeline with all intermediate outputs."""

    original: np.ndarray
    enhanced: np.ndarray
    mask: np.ndarray
    cleaned_mask: np.ndarray
    detection: DetectionResult
    decision: Decision
    processing_time_ms: float = 0.0
    detections: list[DetectionResult] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)