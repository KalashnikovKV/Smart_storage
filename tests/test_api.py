"""API integration tests using FastAPI TestClient.

These tests run without a real webcam and without any files on disk — all
images are generated synthetically via NumPy and encoded in-memory.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(image: np.ndarray) -> bytes:
    """Encode a BGR numpy array to JPEG bytes (in-memory)."""
    _, buf = cv2.imencode(".jpg", image)
    return buf.tobytes()


def _white_object_image(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    """White rectangle on dark background with realistic noise.

    Noise ensures the adaptive threshold and Otsu segmentation can find
    meaningful gradients — pure flat images produce degenerate masks.
    """
    rng = np.random.default_rng(seed)
    # Dark noisy background
    bg_noise = rng.integers(0, 30, (h, w, 3), dtype=np.uint8)
    img = bg_noise.copy()
    # Bright object (slightly noisy to avoid CLAHE flattening)
    obj_noise = rng.integers(220, 256, (180, 280, 3), dtype=np.uint8)
    img[150:330, 180:460] = obj_noise
    return img


def _black_object_image(h: int = 480, w: int = 640, seed: int = 99) -> np.ndarray:
    """Dark rectangle on light background with realistic noise."""
    rng = np.random.default_rng(seed)
    # Light noisy background
    bg_noise = rng.integers(180, 220, (h, w, 3), dtype=np.uint8)
    img = bg_noise.copy()
    # Dark object
    obj_noise = rng.integers(0, 25, (180, 280, 3), dtype=np.uint8)
    img[150:330, 180:460] = obj_noise
    return img


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------

def test_analyze_white_object_returns_200():
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert response.status_code == 200


def test_analyze_response_schema_complete():
    """Top-level schema keys must always be present regardless of detection outcome."""
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    data = response.json()
    assert "is_detected" in data
    assert "artifacts" in data
    assert "processing_time_ms" in data


def test_analyze_decision_fields_present():
    """When object is detected, decision contains all required fields."""
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    data = response.json()
    if not data["is_detected"]:
        pytest.skip("Pipeline did not detect an object in the synthetic image")
    decision = data["decision"]
    for field in ("category", "confidence", "color", "size", "method_used", "is_unknown", "timestamp"):
        assert field in decision, f"Missing field: {field}"


def test_analyze_detection_fields_present():
    """When object is detected, detection contains all required fields."""
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    data = response.json()
    if not data["is_detected"]:
        pytest.skip("Pipeline did not detect an object in the synthetic image")
    detection = data["detection"]
    for field in ("bbox", "primary_color", "size_category", "area_pixels", "aspect_ratio"):
        assert field in detection, f"Missing field: {field}"
    assert len(detection["bbox"]) == 4
    assert "color_hsv" in detection
    assert "color_kmeans" in detection


def test_analyze_confidence_in_range():
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    data = response.json()
    if not data["is_detected"]:
        pytest.skip("Pipeline did not detect an object in the synthetic image")
    conf = data["decision"]["confidence"]
    assert 0.0 <= conf <= 1.0


def test_analyze_artifacts_included_by_default():
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    data = response.json()
    if not data["is_detected"]:
        pytest.skip("Pipeline did not detect an object — artifacts only populated on detection")
    artifacts = data["artifacts"]
    assert any(v is not None for v in artifacts.values())


def test_analyze_artifacts_excluded_when_disabled():
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze?include_artifacts=false",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert response.status_code == 200
    artifacts = response.json()["artifacts"]
    assert all(v is None for v in artifacts.values())


def test_analyze_black_object_returns_200():
    jpeg = _make_jpeg_bytes(_black_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert response.status_code == 200


def test_analyze_no_detection_returns_is_detected_false():
    """An empty (uniform) image with no object should return is_detected=False, not an error."""
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    jpeg = _make_jpeg_bytes(blank)
    response = client.post(
        "/api/analyze",
        files={"file": ("blank.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_detected" in data
    # Blank image — pipeline should return None → is_detected=False
    assert data["is_detected"] is False
    assert data["decision"] is None
    assert data["detection"] is None


def test_analyze_unsupported_content_type_returns_415():
    response = client.post(
        "/api/analyze",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
    )
    assert response.status_code == 415


def test_analyze_empty_file_returns_400():
    response = client.post(
        "/api/analyze",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code == 400


def test_analyze_corrupted_image_returns_422():
    response = client.post(
        "/api/analyze",
        files={"file": ("bad.jpg", io.BytesIO(b"not-an-image"), "image/jpeg")},
    )
    assert response.status_code == 422


def test_analyze_processing_time_is_positive():
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert response.json()["processing_time_ms"] > 0


def test_analyze_is_detected_flag_present():
    """is_detected must always be in the response."""
    jpeg = _make_jpeg_bytes(_white_object_image())
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert "is_detected" in response.json()
