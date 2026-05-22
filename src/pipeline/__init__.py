"""CV pipeline stages: enhance → segment → clean → detect → decide."""

from src.pipeline.protocols import Classifier, Segmenter
from src.pipeline.runner import Pipeline

__all__ = ["Classifier", "Pipeline", "Segmenter"]
