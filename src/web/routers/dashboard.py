"""Dashboard API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.web.services.dashboard import build_dashboard_stats

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
def get_stats(request: Request):
    settings = request.app.state.settings
    return build_dashboard_stats(
        images_dir=settings.images_dir,
        csv_path=settings.csv_path,
        dataset_yaml=settings.dataset_yaml,
    )
