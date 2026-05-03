"""Tests for ColorDetector."""

import numpy as np
import pytest

from src.color_detector import ColorDetector


@pytest.fixture
def detector():
    return ColorDetector()


@pytest.fixture
def white_image_and_mask():
    """Create a white object on black background."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = [255, 255, 255]  # White square (BGR)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    return image, mask


@pytest.fixture
def black_image_and_mask():
    """Create a black object on white background."""
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[30:70, 30:70] = [10, 10, 10]  # Near-black square
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    return image, mask


def test_detect_hsv_white(detector, white_image_and_mask):
    image, mask = white_image_and_mask
    result = detector.detect_hsv(image, mask)
    assert result.name == "white"
    assert result.confidence > 0.5
    assert result.method == "hsv"


def test_detect_hsv_black(detector, black_image_and_mask):
    image, mask = black_image_and_mask
    result = detector.detect_hsv(image, mask)
    assert result.name == "black"
    assert result.confidence > 0.5
    assert result.method == "hsv"


def test_detect_kmeans_white(detector, white_image_and_mask):
    image, mask = white_image_and_mask
    result = detector.detect_kmeans(image, mask)
    assert result.name == "white"
    assert result.confidence > 0.5
    assert result.method == "kmeans"


def test_detect_kmeans_black(detector, black_image_and_mask):
    image, mask = black_image_and_mask
    result = detector.detect_kmeans(image, mask)
    assert result.name == "black"
    assert result.confidence > 0.5
    assert result.method == "kmeans"


def test_detect_combined_agreement(detector, white_image_and_mask):
    image, mask = white_image_and_mask
    result = detector.detect(image, mask)
    assert result["primary_color"] == "white"
    assert result["method_used"] == "combined"
    assert result["primary_confidence"] > 0.5


def test_detect_empty_mask(detector):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    result_hsv = detector.detect_hsv(image, mask)
    assert result_hsv.name == "unknown"
    assert result_hsv.confidence == 0.0

    result_kmeans = detector.detect_kmeans(image, mask)
    assert result_kmeans.name == "unknown"
    assert result_kmeans.confidence == 0.0
