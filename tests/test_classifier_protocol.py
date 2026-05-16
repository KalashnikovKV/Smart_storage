"""Tests verifying that RuleBasedClassifier satisfies the Classifier protocol."""

import numpy as np
import pytest

from src.decision.protocol import Classifier
from src.decision.rule_based import RuleBasedClassifier
from src.models import ColorResult, Decision, DetectionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rule_based():
    return RuleBasedClassifier()


@pytest.fixture
def sample_detection() -> DetectionResult:
    """Minimal DetectionResult representing a small white object (charger-like)."""
    color = ColorResult(name="white", confidence=0.85, method="hsv", rgb=(240, 240, 240))
    return DetectionResult(
        bbox=(280, 200, 40, 40),
        color_hsv=color,
        color_kmeans=ColorResult(name="white", confidence=0.80, method="kmeans", rgb=(235, 235, 235)),
        primary_color="white",
        size_category="small",
        area_pixels=1600,
        aspect_ratio=1.0,
        area_ratio=0.005,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_rule_based_classifier_is_instance_of_protocol(rule_based):
    """RuleBasedClassifier must be a structural subtype of Classifier."""
    assert isinstance(rule_based, Classifier)


def test_classifier_protocol_has_classify_method():
    """Classifier protocol requires a 'classify' method — verified structurally."""
    assert hasattr(Classifier, "classify")


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------

def test_classify_returns_decision(rule_based, sample_detection):
    result = rule_based.classify(sample_detection)
    assert isinstance(result, Decision)


def test_classify_with_roi_kwarg(rule_based, sample_detection):
    """roi parameter must be accepted without error (reserved for ML models)."""
    roi = np.zeros((40, 40, 3), dtype=np.uint8)
    result = rule_based.classify(sample_detection, roi=roi)
    assert isinstance(result, Decision)


def test_classify_with_roi_none(rule_based, sample_detection):
    result = rule_based.classify(sample_detection, roi=None)
    assert isinstance(result, Decision)


def test_classify_confidence_in_range(rule_based, sample_detection):
    result = rule_based.classify(sample_detection)
    assert 0.0 <= result.confidence <= 1.0


def test_classify_returns_nonempty_category(rule_based, sample_detection):
    result = rule_based.classify(sample_detection)
    assert isinstance(result.category, str)
    assert len(result.category) > 0


# ---------------------------------------------------------------------------
# Protocol is exportable from src package
# ---------------------------------------------------------------------------

def test_classifier_exported_from_src():
    """Classifier must be importable from the src top-level package."""
    from src import Classifier as C  # noqa: PLC0415
    assert C is Classifier
