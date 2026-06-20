"""Tests for dashboard decision rendering."""

from __future__ import annotations

import numpy as np

from src.app.visualizer import Visualizer
from src.models import ColorResult, Decision, DetectionResult, PipelineResult


def _detection(object_id: int, color: str = "white") -> DetectionResult:
    color_result = ColorResult(name=color, confidence=0.8, method="hsv")
    return DetectionResult(
        bbox=(10, 10, 40, 40),
        color_hsv=color_result,
        color_kmeans=color_result,
        primary_color=color,
        size_category="medium",
        area_pixels=1600,
        aspect_ratio=1.0,
        shape_category="oval",
        object_id=object_id,
        area_ratio=0.05,
        bbox_width_ratio=0.10,
        bbox_height_ratio=0.10,
    )


def _decision(object_id: int, category: str) -> Decision:
    return Decision(
        category=category,
        confidence=0.9,
        color="white",
        size="medium",
        method_used="combined",
        object_id=object_id,
    )


def _pipeline_result(
    detections: list[DetectionResult],
    decisions: list[Decision],
) -> PipelineResult:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    return PipelineResult(
        original=image,
        enhanced=image,
        mask=mask,
        cleaned_mask=mask,
        detection=detections[0],
        decision=decisions[0],
        detections=detections,
        decisions=decisions,
        processing_time_ms=12.0,
    )


def test_paired_objects_aligns_decisions_with_detections():
    result = _pipeline_result(
        detections=[_detection(1), _detection(2, "black")],
        decisions=[_decision(1, "Mouse"), _decision(2, "Flash Drive")],
    )

    pairs = Visualizer()._paired_objects(result)

    assert len(pairs) == 2
    assert pairs[0][0].category == "Mouse"
    assert pairs[0][1].object_id == 1
    assert pairs[1][0].category == "Flash Drive"
    assert pairs[1][1].primary_color == "black"


def test_create_video_triple_view_has_three_columns():
    visualizer = Visualizer()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    masks = np.full((240, 320, 3), 80, dtype=np.uint8)
    contours = np.full((240, 320, 3), 40, dtype=np.uint8)

    combined = visualizer.create_video_triple_view(
        frame,
        masks,
        contours,
        status_line="LIVE",
    )

    assert combined.shape[0] == 240 + 48
    assert combined.shape[1] == 320 * 3


def test_build_contour_overlay_panel_draws_contours():
    result = _pipeline_result(
        detections=[_detection(1)],
        decisions=[_decision(1, "Headphones")],
    )
    panel = Visualizer().build_contour_overlay_panel(result.original, result)

    assert panel.shape == result.original.shape
    assert np.any(panel != result.original)


def test_create_dashboard_title_for_multiple_objects():
    result = _pipeline_result(
        detections=[_detection(1), _detection(2)],
        decisions=[_decision(1, "Mouse"), _decision(2, "Flash Drive")],
    )

    dashboard = Visualizer().create_dashboard(result)

    assert dashboard is not None
    assert dashboard.shape[0] > 0
    assert Visualizer()._paired_objects(result)[1][0].object_id == 2
