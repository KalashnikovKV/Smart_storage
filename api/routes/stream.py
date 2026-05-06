"""WebSocket /api/stream — stream CV pipeline results from a server-side camera.

Connection lifecycle:
  1. Client connects via WebSocket.
  2. Server opens the webcam (camera_id query param, default 0).
  3. Server processes frames in a loop and sends JSON (StreamFrame) to the client.
  4. Client can send a text message "stop" to close the stream gracefully.
  5. On disconnect the webcam is released automatically.

Each message is a JSON-serialised ``StreamFrame`` with decision + detection fields.
Artifacts (base64 images) are intentionally omitted to keep latency low.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.schemas import DecisionOut, DetectionOut, ColorResultOut, StreamFrame
from src.pipeline import Pipeline
from src.video_processor import VideoProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

_pipeline = Pipeline()

# Interval between frames sent to client (seconds). 100 ms ≈ 10 FPS ceiling.
_FRAME_INTERVAL = 0.10


def _build_stream_frame(result) -> dict:
    """Convert a PipelineResult to a StreamFrame dict ready for JSON serialisation."""
    det = result.detection
    dec = result.decision

    frame = StreamFrame(
        decision=DecisionOut(
            category=dec.category,
            confidence=round(dec.confidence, 3),
            color=dec.color,
            size=dec.size,
            method_used=dec.method_used,
            is_unknown=dec.is_unknown,
            closest_match=dec.closest_match,
            timestamp=dec.timestamp,
        ),
        detection=DetectionOut(
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
        ),
        processing_time_ms=round(result.processing_time_ms, 2),
    )
    return frame.model_dump()


@router.websocket("/stream")
async def stream_camera(
    websocket: WebSocket,
    camera_id: int = 0,
) -> None:
    """Stream real-time CV pipeline results from the server webcam.

    Query params:
      - ``camera_id`` (int, default 0): OpenCV VideoCapture device index.

    Messages sent to client: JSON-serialised ``StreamFrame`` objects.
    Messages accepted from client: ``"stop"`` to gracefully close the stream.

    JavaScript / React Native example::

        const ws = new WebSocket('ws://<host>:8000/api/stream?camera_id=0');
        ws.onmessage = (event) => {
            const frame = JSON.parse(event.data);
            if (frame.error) { console.error(frame.error); return; }
            console.log(frame.decision.category, frame.decision.confidence);
        };
        // To stop: ws.send('stop');
    """
    await websocket.accept()
    logger.info("WebSocket client connected (camera_id=%d)", camera_id)

    video_processor = VideoProcessor(camera_id=camera_id)
    started = video_processor.start()

    if not started:
        await websocket.send_text(json.dumps({"error": f"Cannot open camera {camera_id}"}))
        await websocket.close()
        return

    try:
        while True:
            # Check for incoming client messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if msg.strip().lower() == "stop":
                    logger.info("Client requested stream stop")
                    break
            except asyncio.TimeoutError:
                pass  # No message — continue streaming

            frame = video_processor.get_frame()
            if frame is None:
                await websocket.send_text(json.dumps({"error": "Camera read failed"}))
                break

            result = _pipeline.run(frame)
            if result is None:
                # No object detected — send a lightweight status message
                await websocket.send_text(json.dumps({"status": "no_object"}))
            else:
                await websocket.send_text(json.dumps(_build_stream_frame(result)))

            # Yield control so FastAPI can handle other requests
            await asyncio.sleep(_FRAME_INTERVAL)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        video_processor.stop()
        logger.info("Camera released (camera_id=%d)", camera_id)
