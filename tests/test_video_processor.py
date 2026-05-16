"""Tests for VideoProcessor file-source helpers."""

from pathlib import Path

from src.app.video_processor import SUPPORTED_VIDEO_EXTENSIONS, is_video_file


def test_is_video_file_recognises_mov_and_mp4():
    assert is_video_file("training/test_video/Video_1.MOV")
    assert is_video_file(Path("clip.mp4"))


def test_is_video_file_rejects_images():
    assert not is_video_file("training/test_images/Image_1.jpeg")


def test_supported_video_extensions_include_common_formats():
    assert ".mov" in SUPPORTED_VIDEO_EXTENSIONS
    assert ".mp4" in SUPPORTED_VIDEO_EXTENSIONS
