"""Video capture module for webcam processing."""

import cv2
import numpy as np

from src import config


class VideoProcessor:
    """Captures frames from a webcam."""

    def __init__(self, camera_id: int | None = None) -> None:
        self.camera_id = camera_id if camera_id is not None else config.CAMERA_ID
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> bool:
        """Open the video capture device.

        Returns:
            True if camera opened successfully, False otherwise.
        """
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
