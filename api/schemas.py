"""Pydantic response schemas for the Smart Storage API.

These mirror the internal dataclasses from src/models.py but are JSON-serialisable
and safe to expose over HTTP (no numpy arrays, no contours).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColorResultOut(BaseModel):
    """Color detection result from a single method."""

    name: str = Field(description="Color name: white, black, gray, silver, red, blue, unknown")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    method: str = Field(description="Detection method: hsv or kmeans")
    rgb: tuple[int, int, int] = Field(default=(0, 0, 0), description="Dominant color as (R, G, B)")


class DetectionOut(BaseModel):
    """Object detection result: bounding box, color and size properties."""

    bbox: tuple[int, int, int, int] = Field(description="Bounding box as (x, y, w, h)")
    primary_color: str = Field(description="Final determined color name")
    size_category: str = Field(description="small | medium | large | long_thin")
    area_pixels: int = Field(description="Object area in pixels")
    aspect_ratio: float = Field(description="max(w,h) / min(w,h)")
    circularity: float = Field(default=0.0, description="Shape circularity 0..1")
    solidity: float = Field(default=0.0, description="area / convex_hull_area")
    extent: float = Field(default=0.0, description="area / bounding_rect_area")
    shape_category: str = Field(default="", description="oval | rectangular | irregular")
    color_hsv: ColorResultOut
    color_kmeans: ColorResultOut


class DecisionOut(BaseModel):
    """Final classification decision."""

    category: str = Field(description="Category name in Russian, e.g. Зарядка iPhone")
    confidence: float = Field(ge=0.0, le=1.0)
    color: str
    size: str
    method_used: str = Field(description="hsv | kmeans | combined")
    is_unknown: bool = Field(default=False)
    closest_match: str = Field(default="")
    timestamp: str = Field(description="ISO 8601 timestamp")


class ArtifactsOut(BaseModel):
    """Base64-encoded pipeline stage images (JPEG).

    Clients can decode these with ``atob()`` / ``Buffer.from(b64, 'base64')``
    and display them without a second round-trip.
    All fields are optional — pass ``include_artifacts=false`` to skip them.
    """

    original_b64: str | None = None
    enhanced_b64: str | None = None
    mask_b64: str | None = None
    cleaned_mask_b64: str | None = None
    detection_b64: str | None = None


class AnalyzeResponse(BaseModel):
    """Full response for POST /api/analyze.

    When ``is_detected`` is False, ``decision`` and ``detection`` are None.
    This happens for valid images where no foreground object was found
    (e.g. empty frame, full-frame background, very small/large object).
    """

    is_detected: bool = Field(description="False if no object was found in the image")
    decision: DecisionOut | None = None
    detection: DetectionOut | None = None
    artifacts: ArtifactsOut
    processing_time_ms: float = Field(description="Total pipeline time in milliseconds")


class StreamFrame(BaseModel):
    """Single frame payload sent over the WebSocket /api/stream.

    Artifacts are omitted to keep latency low.
    """

    decision: DecisionOut
    detection: DetectionOut
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
