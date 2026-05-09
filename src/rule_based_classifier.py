"""Rule-based classifier — wraps Pipeline.decide() behind the Classifier protocol."""

import numpy as np

from src.classifier_protocol import Classifier  # noqa: F401 — re-export hint
from src.models import Decision, DetectionResult


class RuleBasedClassifier:
    """Classifier that delegates to Pipeline.decide() for rule-based decisions.

    Satisfies the Classifier protocol, making it swappable for an ML model.
    """

    def __init__(self) -> None:
        # Lazy import avoids a circular dependency with Pipeline → config chain
        from src.pipeline import Pipeline  # noqa: PLC0415

        self._pipeline = Pipeline()

    def classify(
        self,
        detection: DetectionResult,
        roi: np.ndarray | None = None,  # noqa: ARG002 — reserved for ML models
    ) -> Decision:
        """Return a Decision using the rule-based heuristics from Pipeline."""
        return self._pipeline.decide(detection)
