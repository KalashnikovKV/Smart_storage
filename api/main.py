"""Smart Storage — FastAPI application entry point.

Start the server:
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs (Swagger UI):
    http://localhost:8000/docs

Endpoints:
    GET  /api/health          — Health check
    POST /api/analyze         — Analyse a single uploaded image
    WS   /api/stream          — Stream results from server-side webcam
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analyze import router as analyze_router
from api.routes.stream import router as stream_router
from api.schemas import HealthResponse

app = FastAPI(
    title="Smart Storage API",
    description=(
        "Computer Vision API for automatic IT peripheral recognition.\n\n"
        "Send an image → get back the classification (category, confidence, color, size) "
        "plus optional base64-encoded pipeline stage images.\n\n"
        "Or connect via WebSocket to receive a live stream of results from the server camera."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins by default so any React Native / web client works.
# Tighten this list in production by setting ALLOWED_ORIGINS env variable.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(analyze_router, prefix="/api", tags=["Analysis"])
app.include_router(stream_router, prefix="/api", tags=["Stream"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Return server status. Use this to verify the API is reachable before sending images."""
    return HealthResponse(status="ok", version="0.1.0")
