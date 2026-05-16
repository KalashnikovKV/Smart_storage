"""Detection stage: color + geometry from masks."""

from src.detect.color import ColorDetector
from src.detect.from_mask import ObjectMaskDetector

__all__ = ["ColorDetector", "ObjectMaskDetector"]
