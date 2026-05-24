"""Pipeline execution and preview rendering for labeling."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.models import PipelineResult
from src.pipeline import Pipeline
from src.web.schemas.label import LabelItem


def read_image(image_path: Path) -> np.ndarray | None:
    """Load a BGR image from disk."""
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        return None
    return image


def run_pipeline(pipeline: Pipeline, image_path: Path) -> PipelineResult | None:
    """Run CV pipeline on a single image file."""
    image = read_image(image_path)
    if image is None:
        return None
    return pipeline.run(image)


def render_preview(
    image: np.ndarray,
    result: PipelineResult | None,
    *,
    overlay: bool = True,
) -> np.ndarray:
    """Draw bbox and mask overlay for the labeling UI."""
    preview = image.copy()

    if result is None or not overlay:
        return preview

    if result.cleaned_mask is not None and result.cleaned_mask.size > 0:
        mask = result.cleaned_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        colored = np.zeros_like(preview)
        colored[:, :, 1] = mask  # green channel
        preview = cv2.addWeighted(preview, 0.65, colored, 0.35, 0)

    if result.detection is not None:
        x, y, w, h = result.detection.bbox
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = result.decision.category if result.decision else ""
        if label:
            cv2.putText(
                preview,
                label,
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return preview


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode BGR image as JPEG bytes."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode preview image")
    return buffer.tobytes()


def build_label_item(
    *,
    image_name: str,
    result: PipelineResult | None,
) -> LabelItem:
    """Build API response for one labeling candidate."""
    has_detection = result is not None and result.decision is not None

    if not has_detection:
        return LabelItem(
            image_name=image_name,
            image_url=f"/api/label/images/{image_name}",
            preview_url=f"/api/label/images/{image_name}/preview",
            has_detection=False,
        )

    decision = result.decision
    return LabelItem(
        image_name=image_name,
        image_url=f"/api/label/images/{image_name}",
        preview_url=f"/api/label/images/{image_name}/preview",
        predicted_category=decision.category,
        confidence=float(decision.confidence),
        color=decision.color,
        size=decision.size,
        method_used=decision.method_used,
        processing_time_ms=float(result.processing_time_ms),
        has_detection=True,
    )
