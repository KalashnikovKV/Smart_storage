"""POST /api/analyze — accept an image file, run the CV pipeline, return JSON."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, UploadFile

from api.schemas import AnalyzeResponse, ArtifactsOut, DetectionOut, DecisionOut, ColorResultOut
from src.pipeline import Pipeline
from src.app.visualizer import Visualizer

router = APIRouter()

# Single shared pipeline instance (stateless — safe for concurrent requests)
_pipeline = Pipeline()
_visualizer = Visualizer()


def _encode_image(image: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def _encode_mask(mask: np.ndarray) -> str:
    """Encode a binary mask to a base64 JPEG string (converted to BGR first)."""
    bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return _encode_image(bgr)


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyse a single image")
async def analyze_image(
    file: UploadFile,
    include_artifacts: bool = Query(
        default=True,
        description="Include base64-encoded pipeline stage images in the response",
    ),
) -> AnalyzeResponse:
    """Accept a JPEG/PNG image, run the full CV pipeline and return the classification result.

    **Request**: multipart/form-data with a single ``file`` field.

    **Response**: JSON with decision, detection details and optional base64 artifacts.

    React Native example::

        const form = new FormData();
        form.append('file', { uri: photoUri, type: 'image/jpeg', name: 'photo.jpg' });
        const res = await fetch('http://<host>:8000/api/analyze', { method: 'POST', body: form });
        const data = await res.json();
        console.log(data.decision.category);
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file received")

    # Decode uploaded bytes → BGR numpy array
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Cannot decode image — invalid or corrupted file")

    import time as _time
    t0 = _time.perf_counter()

    # Run the full CV pipeline
    result = _pipeline.run(image)

    elapsed_ms = (_time.perf_counter() - t0) * 1000

    # Valid image but no detectable object — return 200 with is_detected=False
    if result is None:
        return AnalyzeResponse(
            is_detected=False,
            artifacts=ArtifactsOut(),
            processing_time_ms=round(elapsed_ms, 2),
        )

    det = result.detection
    dec = result.decision

    detection_out = DetectionOut(
        bbox=det.bbox,
        primary_color=det.primary_color,
        size_category=det.size_category,
        area_pixels=det.area_pixels,
        aspect_ratio=round(det.aspect_ratio, 3),
        circularity=round(det.circularity, 3),
        solidity=round(det.solidity, 3),
        extent=round(det.extent, 3),
        shape_category=det.shape_category,
        color_hsv=ColorResultOut(
            name=det.color_hsv.name,
            confidence=round(det.color_hsv.confidence, 3),
            method=det.color_hsv.method,
            rgb=det.color_hsv.rgb,
        ),
        color_kmeans=ColorResultOut(
            name=det.color_kmeans.name,
            confidence=round(det.color_kmeans.confidence, 3),
            method=det.color_kmeans.method,
            rgb=det.color_kmeans.rgb,
        ),
    )

    decision_out = DecisionOut(
        category=dec.category,
        confidence=round(dec.confidence, 3),
        color=dec.color,
        size=dec.size,
        method_used=dec.method_used,
        is_unknown=dec.is_unknown,
        closest_match=dec.closest_match,
        timestamp=dec.timestamp,
    )

    # Build artifacts — draw detection overlay on a copy for the detection panel
    if include_artifacts:
        detection_img = result.enhanced.copy()
        _visualizer.draw_detection(detection_img, result.detection)
        artifacts = ArtifactsOut(
            original_b64=_encode_image(result.original),
            enhanced_b64=_encode_image(result.enhanced),
            mask_b64=_encode_mask(result.mask),
            cleaned_mask_b64=_encode_mask(result.cleaned_mask),
            detection_b64=_encode_image(detection_img),
        )
    else:
        artifacts = ArtifactsOut()

    return AnalyzeResponse(
        is_detected=True,
        decision=decision_out,
        detection=detection_out,
        artifacts=artifacts,
        processing_time_ms=round(result.processing_time_ms, 2),
    )
