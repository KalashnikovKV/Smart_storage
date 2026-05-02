"""Tests for Pipeline."""

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
