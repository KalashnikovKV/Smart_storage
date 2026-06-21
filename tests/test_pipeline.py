"""Tests for Pipeline."""

import cv2
import numpy as np
import pytest

from src.pipeline import Pipeline


@pytest.fixture
def pipeline():
    return Pipeline()


@pytest.fixture
def white_object_on_black():
    """White rectangle on black background — simulates a charger."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Small white object (charger-sized)
    image[200:240, 280:320] = [255, 255, 255]
    return image


@pytest.fixture
def black_object_on_white():
    """Black rectangle on white background — simulates a mouse."""
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    # Medium black object (mouse-sized)
    image[180:260, 250:350] = [10, 10, 10]
    return image


def test_enhance_returns_same_shape(pipeline, white_object_on_black):
    result = pipeline.enhance(white_object_on_black)
    assert result.shape == white_object_on_black.shape
    assert result.dtype == np.uint8


def test_segment_returns_binary_mask(pipeline, white_object_on_black):
    enhanced = pipeline.enhance(white_object_on_black)
    mask = pipeline.segment(enhanced)
    assert mask.shape == (480, 640)
    assert set(np.unique(mask)).issubset({0, 255})


def test_clean_returns_binary_mask(pipeline, white_object_on_black):
    enhanced = pipeline.enhance(white_object_on_black)
    mask = pipeline.segment(enhanced)
    cleaned = pipeline.clean(mask)
    assert cleaned.shape == mask.shape
    assert set(np.unique(cleaned)).issubset({0, 255})


def test_full_pipeline_white_object(pipeline, white_object_on_black):
    result = pipeline.run(white_object_on_black)
    # May or may not detect depending on adaptive threshold behavior
    # with synthetic images — this tests the pipeline doesn't crash
    if result is not None:
        assert result.decision is not None
        assert result.processing_time_ms >= 0


def test_full_pipeline_black_object(pipeline, black_object_on_white):
    result = pipeline.run(black_object_on_white)
    if result is not None:
        assert result.decision is not None
        assert result.enhanced.shape == black_object_on_white.shape


def test_pipeline_empty_image(pipeline):
    """Pipeline should handle an image with no object gracefully."""
    image = np.full((480, 640, 3), 128, dtype=np.uint8)  # Uniform gray
    result = pipeline.run(image)
    # Uniform image may or may not produce a detection — just ensure no crash
    assert result is None or result.decision is not None


def test_enhance_preserves_content(pipeline):
    """Enhancement should not drastically change image dimensions."""
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    enhanced = pipeline.enhance(image)
    assert enhanced.shape == image.shape


def test_find_object_contours_detects_two_objects(pipeline):
    """Two separate blobs in one mask should yield two ranked contours."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (180, 160), 255, -1)
    cv2.rectangle(mask, (400, 300), (520, 380), 255, -1)

    contours = pipeline.mask_ops.find_object_contours(mask, max_objects=8)

    assert len(contours) == 2


def test_detect_all_returns_multiple_objects(pipeline):
    """Multi-contour path should produce one DetectionResult per object."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (180, 160), 255, -1)
    cv2.rectangle(mask, (400, 300), (520, 380), 255, -1)

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[mask > 0] = (200, 200, 200)

    detections = pipeline.detect_all(image, image, mask)

    assert len(detections) == 2
    assert [detection.object_id for detection in detections] == [1, 2]


def test_run_multi_object_decisions(pipeline):
    """Full pipeline should classify each detected object separately."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(image, (420, 280), (560, 400), (255, 255, 255), -1)

    result = pipeline.run(image)

    assert result is not None
    assert len(result.detections) == 2
    assert len(result.decisions) == 2
    assert result.decisions[0].object_id == 1
    assert result.decisions[1].object_id == 2


def _four_white_objects_image() -> np.ndarray:
    """Four separated white blocks on black — synthetic multi-object scene."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = (
        (60, 60, 160, 150),
        (470, 60, 570, 150),
        (60, 320, 160, 410),
        (470, 320, 570, 410),
    )
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return image


def test_find_object_contours_detects_four_objects(pipeline):
    """Up to four separate blobs should all be returned."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    for x1, y1, x2, y2 in (
        (60, 60, 160, 150),
        (470, 60, 570, 150),
        (60, 320, 160, 410),
        (470, 320, 570, 410),
    ):
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    contours = pipeline.mask_ops.find_object_contours(mask, max_objects=8)

    assert len(contours) == 4


def test_run_four_object_pipeline(pipeline):
    """Full pipeline should detect and classify four separated objects."""
    result = pipeline.run(_four_white_objects_image())

    assert result is not None
    assert len(result.detections) == 4
    assert len(result.decisions) == 4
    assert all(detection.contour is not None for detection in result.detections)


def test_build_colored_object_masks(pipeline):
    """Each object mask should get a distinct color in the combined view."""
    import cv2
    import numpy as np
    from src.app.visualizer import Visualizer

    mask_a = np.zeros((120, 160), dtype=np.uint8)
    mask_b = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(mask_a, (20, 20), (70, 80), 255, -1)
    cv2.rectangle(mask_b, (90, 40), (140, 100), 255, -1)

    visualizer = Visualizer()
    view = visualizer.build_colored_object_masks([mask_a, mask_b])

    assert view.shape == (120, 160, 3)
    assert np.any(view[30, 45] > 0)
    assert np.any(view[60, 115] > 0)
    assert not np.array_equal(view[30, 45], view[60, 115])


def test_remove_shadow_from_mask(pipeline):
    """Dark shadow attached to a bright object should be trimmed from the mask."""
    image = np.full((200, 200, 3), 120, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (100, 100), (240, 240, 240), -1)
    cv2.rectangle(image, (100, 70), (150, 110), (55, 55, 55), -1)

    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (40, 40), (150, 110), 255, -1)

    cleaned = pipeline.mask_ops.remove_shadow_from_mask(image, mask)
    cleaned_area = int(np.sum(cleaned > 0))
    object_only_area = int(np.sum(mask[40:101, 40:101] > 0))

    assert cleaned_area < int(np.sum(mask > 0))
    assert cleaned_area >= int(object_only_area * 0.75)
    assert int(np.sum(cleaned[70:111, 100:151] > 0)) < int(np.sum(mask[70:111, 100:151] > 0) * 0.35)


def test_yolo_path_dashboard_uses_yolo_masks_not_threshold():
    """YOLO mode should expose YOLO instance masks on dashboard panels 3-4."""
    from src.models import YOLODetection

    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    yolo_mask = np.zeros((480, 640), dtype=np.uint8)
    yolo_mask[180:260, 250:350] = 255

    class FakeSegmenter:
        def segment(self, _image: np.ndarray) -> list[YOLODetection]:
            return [
                YOLODetection(
                    class_name="mouse",
                    bbox=(250, 180, 100, 80),
                    mask=yolo_mask,
                    confidence=0.95,
                ),
            ]

    pipeline = Pipeline(segmenter=FakeSegmenter())
    result = pipeline.run(image)

    assert result is not None
    assert np.array_equal(result.mask, yolo_mask)
    assert np.array_equal(result.cleaned_mask, yolo_mask)
    assert len(result.detections) == 1
    assert result.decision.method_used == "yolo"
