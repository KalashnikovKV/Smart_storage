"""CLI processing modes."""

from src.modes.batch import run_batch_mode
from src.modes.image import run_image_mode
from src.modes.video import run_video_mode

__all__ = [
    "run_batch_mode",
    "run_image_mode",
    "run_video_mode",
]
