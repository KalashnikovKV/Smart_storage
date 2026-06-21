"""YOLO-Seg integration (optional ML backend)."""

from src.ml.yolo_category import yolo_category
from src.ml.yolo_segmenter import YOLOSegmenter, _parse_results

__all__ = [
    "YOLOSegmenter",
    "_parse_results",
    "yolo_category",
]
