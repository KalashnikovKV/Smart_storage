"""Tests for FastAPI ML Studio endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.app import create_app
from src.web.config import WebSettings


@pytest.fixture
def web_client(tmp_path: Path):
    images_dir = tmp_path / "test_images"
    images_dir.mkdir()
    image_path = images_dir / "sample.jpg"
    # Minimal valid JPEG header bytes — cv2 may still fail; health should work.
    image_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"
    )

    csv_path = tmp_path / "results.csv"
    csv_path.write_text("image_name,ground_truth,category,confidence\n", encoding="utf-8")

    settings = WebSettings(
        project_root=tmp_path,
        images_dir=images_dir,
        csv_path=csv_path,
        dataset_dir=tmp_path / "dataset",
        dataset_yaml=tmp_path / "dataset" / "dataset.yaml",
        stages_dir=tmp_path / "stages",
        frontend_dist=tmp_path / "frontend" / "dist",
    )

    app = create_app()
    app.state.settings = settings
    app.state.preview_cache = {}
    from src.modes.common import build_pipeline
    from src.web.services.job_runner import JobStore

    app.state.pipeline = build_pipeline(None, device="cpu")
    app.state.job_store = JobStore()

    return TestClient(app)


def test_health(web_client: TestClient):
    response = web_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_label_classes(web_client: TestClient):
    response = web_client.get("/api/label/classes")
    assert response.status_code == 200
    classes = response.json()
    assert len(classes) == 8
    assert classes[0]["label"] == "Mouse"


def test_dashboard_stats(web_client: TestClient):
    response = web_client.get("/api/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 1
    assert data["labeled_images"] == 0
