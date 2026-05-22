"""Tests for VideoProcessor file-source helpers and playback controls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.app.video_processor import DEFAULT_PLAYBACK_FPS, SUPPORTED_VIDEO_EXTENSIONS, VideoProcessor, is_video_file


def test_is_video_file_recognises_mov_and_mp4():
    assert is_video_file("training/test_video/Video_1.MOV")
    assert is_video_file(Path("clip.mp4"))


def test_is_video_file_rejects_images():
    assert not is_video_file("training/test_images/Image_1.jpeg")


def test_supported_video_extensions_include_common_formats():
    assert ".mov" in SUPPORTED_VIDEO_EXTENSIONS
    assert ".mp4" in SUPPORTED_VIDEO_EXTENSIONS


def test_pause_returns_frozen_frame_without_reading():
    video = VideoProcessor(source="training/test_video/Video_1.MOV")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    video._cap = MagicMock()
    video.pause(frame)

    result = video.get_frame()

    assert result is not None
    assert np.array_equal(result, frame)
    video._cap.read.assert_not_called()


def test_wait_delay_ms_uses_file_fps():
    video = VideoProcessor(source="clip.mp4")
    video._fps = 25.0
    assert video.wait_delay_ms() == 40


def test_wait_delay_ms_is_one_for_webcam():
    video = VideoProcessor()
    assert video.wait_delay_ms() == 1


def test_start_reads_fps_from_capture():
    video = VideoProcessor(source="clip.mp4")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 24.0

    with patch("src.app.video_processor.cv2.VideoCapture", return_value=mock_cap):
        assert video.start() is True

    assert video.playback_fps == 24.0


def test_start_falls_back_to_default_fps_when_reported_fps_invalid():
    video = VideoProcessor(source="clip.mp4")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 0.0

    with patch("src.app.video_processor.cv2.VideoCapture", return_value=mock_cap):
        assert video.start() is True

    assert video.playback_fps == DEFAULT_PLAYBACK_FPS
