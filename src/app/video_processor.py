"""Video capture module for webcam and video file processing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig

SUPPORTED_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}
DEFAULT_PLAYBACK_FPS = 30.0


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
        self._paused = False
        self._frozen_frame: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None
        self._fps = DEFAULT_PLAYBACK_FPS
        self._loop_file = True

    @property
    def is_file_source(self) -> bool:
        """True when frames are read from a video file (not a live camera)."""
        return self.source_path is not None

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def playback_fps(self) -> float:
        return self._fps

    def start(self) -> bool:
        """Open the video capture device or file."""
        if isinstance(self._source, str):
            self._cap = cv2.VideoCapture(self._source)
        elif isinstance(self._source, int):
            self._cap = cv2.VideoCapture(self._source)
        else:
            self._cap = cv2.VideoCapture(self.camera_id)

        if self._cap is None or not self._cap.isOpened():
            self._cap = None
            return False

        if self.is_file_source:
            reported = float(self._cap.get(cv2.CAP_PROP_FPS))
            if 1.0 <= reported <= 120.0:
                self._fps = reported

        return True

    def stop(self) -> None:
        """Release the video capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._paused = False
        self._frozen_frame = None
        self._last_frame = None

    def pause(self, frame: np.ndarray | None = None) -> None:
        """Stop advancing playback; optionally freeze on a specific frame."""
        self._paused = True
        if frame is not None:
            self._frozen_frame = frame.copy()

    def resume(self) -> None:
        """Resume live capture or file playback."""
        self._paused = False
        self._frozen_frame = None

    def toggle_pause(self, frame: np.ndarray | None = None) -> bool:
        """Toggle pause state. Returns True when paused after the toggle."""
        if self._paused:
            self.resume()
            return False
        self.pause(frame if frame is not None else self._last_frame)
        return True

    def wait_delay_ms(self) -> int:
        """Suggested cv2.waitKey delay to approximate real-time file playback."""
        if self.is_file_source and not self._paused:
            return max(1, int(1000 / self._fps))
        return 1

    def get_frame(self) -> np.ndarray | None:
        """Read the next frame, or return the frozen frame while paused."""
        if self._paused:
            if self._frozen_frame is not None:
                return self._frozen_frame.copy()
            return self._last_frame.copy() if self._last_frame is not None else None

        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._last_frame = frame
            return frame

        if self.is_file_source and self._loop_file:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._last_frame = frame
                return frame

        return self._last_frame.copy() if self._last_frame is not None else None

    def is_running(self) -> bool:
        """Check if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()
