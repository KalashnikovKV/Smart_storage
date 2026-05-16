"""Video capture module for webcam and video file processing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig

SUPPORTED_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}


def is_video_file(path: str | Path) -> bool:
    """Return True when *path* has a supported video file extension."""
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


class VideoProcessor:
    """Captures frames from a webcam or a video file."""

    def __init__(self, source: str | int | None = None, camera_id: int | None = None) -> None:
        """Create a capture helper.

        Args:
            source: Webcam index (int), video file path (str), or None for default webcam.
            camera_id: Used when *source* is None (defaults to AppConfig.camera_id).
        """
        self._source = source
        self.camera_id = camera_id if camera_id is not None else AppConfig().camera_id
        self._cap: cv2.VideoCapture | None = None
        self.source_path: Path | None = (
            Path(source) if isinstance(source, str) else None
        )

    def start(self) -> bool:
        """Open the video capture device or file.

        Returns:
            True if opened successfully, False otherwise.
        """
        if isinstance(self._source, str):
            self._cap = cv2.VideoCapture(self._source)
        elif isinstance(self._source, int):
            self._cap = cv2.VideoCapture(self._source)
        else:
            self._cap = cv2.VideoCapture(self.camera_id)

        if not self._cap.isOpened():
            self._cap = None
            return False
        return True

    def stop(self) -> None:
        """Release the video capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> np.ndarray | None:
        """Capture a single frame from the camera.

        Returns:
            BGR image or None if capture failed.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def is_running(self) -> bool:
        """Check if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()
