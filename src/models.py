"""Data models for the Smart Storage pipeline."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class ColorResult:
    """Result of color detection from a single method."""

    name: str  # "white", "black", "gray", "silver", etc.
    confidence: float  # 0.0 - 1.0
    method: str  # "hsv" or "kmeans"
    rgb: tuple[int, int, int] = (0, 0, 0)  # Dominant color RGB values


@dataclass
class DetectionResult:
    """Result of object detection including color, size, and shape analysis."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    color_hsv: ColorResult
    color_kmeans: ColorResult
    primary_color: str  # Final determined color name
    size_category: str  # "small", "medium", "large", "long_thin"
    area_pixels: int
    aspect_ratio: float
    # Shape features
    circularity: float = 0.0  # 1.0 = perfect circle, 0.0 = very irregular
    solidity: float = 0.0  # area / convex_hull_area — how "solid" the shape is
    extent: float = 0.0  # area / bounding_rect_area — how much of bbox is filled
    shape_category: str = ""  # "oval", "rectangular", "irregular"
    contour: np.ndarray | None = None


@dataclass
class Decision:
    """Final classification decision."""

    category: str  # "Зарядка iPhone", "Кабель питания", etc.
    confidence: float  # 0.0 - 1.0
    color: str
    size: str
    method_used: str  # "hsv", "kmeans", or "combined"
    is_unknown: bool = False
    closest_match: str = ""
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
