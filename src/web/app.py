"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.modes.common import build_pipeline
from src.web.config import WebSettings
from src.web.routers import dashboard, dataset, label, train
from src.web.services.job_runner import JobStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = WebSettings.default()
    app.state.settings = settings
    app.state.pipeline = build_pipeline(model_path=None, device="cpu")
    app.state.preview_cache = {}
    app.state.job_store = JobStore()
    yield


def create_app() -> FastAPI:
    """Create and configure the ML Studio FastAPI app."""
    app = FastAPI(
        title="Smart Storage ML Studio",
        description="Web UI for dataset labeling and YOLO training",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(label.router)
    app.include_router(dashboard.router)
    app.include_router(dataset.router)
    app.include_router(train.router)

    @app.get("/api/health")
    def health():
        settings: WebSettings = app.state.settings
        return {
            "status": "ok",
            "images_dir": str(settings.images_dir),
            "csv_path": str(settings.csv_path),
        }

    dist = WebSettings.default().frontend_dist
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")

    return app


app = create_app()
