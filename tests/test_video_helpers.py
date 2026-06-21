"""Tests for video-mode geometry helpers."""

from __future__ import annotations

import numpy as np

from src.modes.video import _map_result_to_frame, _resize_for_pipeline, _scale_detection
from src.models import ColorResult, Decision, DetectionResult, PipelineResult


def _detection() -> DetectionResult:
    color = ColorResult(name="black", confidence=0.8, method="hsv")
    return DetectionResult(
        bbox=(10, 20, 30, 40),
        color_hsv=color,
        color_kmeans=color,
        primary_color="black",
        size_category="medium",
        area_pixels=1200,
        aspect_ratio=1.3,
        object_id=1,
        area_ratio=0.05,
        bbox_width_ratio=0.10,
        bbox_height_ratio=0.10,
    )


def _result(small_frame: np.ndarray) -> PipelineResult:
    mask = np.zeros(small_frame.shape[:2], dtype=np.uint8)
    decision = Decision(
        category="Headphones",
        confidence=0.8,
        color="black",
        size="medium",
        method_used="combined",
        object_id=1,
    )
    detection = _detection()
    return PipelineResult(
        original=small_frame,
        enhanced=small_frame,
        mask=mask,
        cleaned_mask=mask,
        detection=detection,
        decision=decision,
        detections=[detection],
        decisions=[decision],
        processing_time_ms=42.0,
    )


def test_resize_for_pipeline_downscales_wide_frames():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    resized, scale = _resize_for_pipeline(frame, max_width=640)

    assert resized.shape[1] == 640
    assert scale == 0.5


def test_resize_for_pipeline_skips_downscale_when_disabled():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    resized, scale = _resize_for_pipeline(frame, max_width=0)

    assert resized is frame
    assert scale == 1.0


def test_scale_detection_expands_bbox_to_full_resolution():
    detection = _detection()
    scaled = _scale_detection(detection, inv_scale=2.0)

    assert scaled.bbox == (20, 40, 60, 80)


def test_map_result_to_frame_restores_original_size():
    display = np.zeros((480, 640, 3), dtype=np.uint8)
    small = np.zeros((240, 320, 3), dtype=np.uint8)
    mapped = _map_result_to_frame(_result(small), display, scale=0.5)

    assert mapped.original.shape == display.shape
    assert mapped.mask.shape == display.shape[:2]
    assert mapped.detections[0].bbox[0] == 20
