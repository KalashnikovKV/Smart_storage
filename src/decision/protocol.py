"""Classifier protocol — abstraction over the decision/classify step.

Allows swapping the rule-based classifier for an ML model without
touching Pipeline.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from src.models import Decision, DetectionResult


@runtime_checkable
class Classifier(Protocol):
    """Structural protocol for any IT-peripheral classifier."""

    def classify(
        self,
        detection: DetectionResult,
        roi: np.ndarray | None = None,
    ) -> Decision: ...
