"""Segmentation stage (threshold + optional YOLO protocol implementations)."""

from src.segment.protocol import Segmenter
from src.segment.threshold import ThresholdSegmenter
from src.segment.yolo import YOLOSegmenter, _parse_results

__all__ = [
    "Segmenter",
    "ThresholdSegmenter",
    "YOLOSegmenter",
    "_parse_results",
]
