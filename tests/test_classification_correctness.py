"""Classification correctness tests for synthetic object detections."""

import pytest

from src.models import ColorResult, DetectionResult
from src.pipeline import Pipeline


def make_color_result(name: str, confidence: float, method: str) -> ColorResult:
    """Create a synthetic color detection result."""
    return ColorResult(
        name=name,
        confidence=confidence,
        method=method,
        rgb=(0, 0, 0),
    )


def make_detection(
    *,
    primary_color: str,
    size_category: str,
    shape_category: str,
    area_ratio: float,
    bbox_width_ratio: float,
    bbox_height_ratio: float,
    aspect_ratio: float,
    circularity: float,
    solidity: float,
    extent: float,
    edge_density: float,
) -> DetectionResult:
    """Create a synthetic detection result for classification tests."""
    return DetectionResult(
        bbox=(
            10,
            10,
            int(bbox_width_ratio * 1000),
            int(bbox_height_ratio * 1000),
        ),
        color_hsv=make_color_result(primary_color, 0.90, "hsv"),
        color_kmeans=make_color_result(primary_color, 0.85, "kmeans"),
        primary_color=primary_color,
        size_category=size_category,
        area_pixels=int(area_ratio * 1_000_000),
        aspect_ratio=aspect_ratio,
        circularity=circularity,
        solidity=solidity,
        extent=extent,
        shape_category=shape_category,
        contour=None,
        object_id=1,
        edge_density=edge_density,
        area_ratio=area_ratio,
        bbox_width_ratio=bbox_width_ratio,
        bbox_height_ratio=bbox_height_ratio,
        visual_size_label=size_category,
        size_confidence=0.90,
    )


def test_keyboard_classification() -> None:
    """A large dark rectangular object should be classified as Keyboard."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="black",
        size_category="large",
        shape_category="rectangular",
        area_ratio=0.24,
        bbox_width_ratio=0.36,
        bbox_height_ratio=0.71,
        aspect_ratio=1.97,
        circularity=0.30,
        solidity=0.90,
        extent=0.80,
        edge_density=0.04,
    )

    decision = pipeline.decide(detection)

    assert decision.category == "Keyboard"
    assert decision.color == "black"
    assert decision.size == "large"
    assert not decision.is_unknown


def test_mouse_classification() -> None:
    """A dark compact oval object should be classified as Mouse."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="black",
        size_category="medium",
        shape_category="oval",
        area_ratio=0.057,
        bbox_width_ratio=0.33,
        bbox_height_ratio=0.26,
        aspect_ratio=1.27,
        circularity=0.65,
        solidity=0.82,
        extent=0.62,
        edge_density=0.03,
    )

    decision = pipeline.decide(detection)

    assert decision.category == "Mouse"
    assert decision.color == "black"
    assert decision.size == "medium"
    assert not decision.is_unknown


def test_large_close_mouse_can_still_be_mouse() -> None:
    """A close mouse can visually appear large and should still be Mouse."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="black",
        size_category="large",
        shape_category="oval",
        area_ratio=0.19,
        bbox_width_ratio=0.66,
        bbox_height_ratio=0.43,
        aspect_ratio=1.53,
        circularity=0.62,
        solidity=0.84,
        extent=0.67,
        edge_density=0.03,
    )

    decision = pipeline.decide(detection)

    assert decision.category == "Mouse"
    assert decision.color == "black"
    assert decision.size == "large"
    assert not decision.is_unknown


def test_charger_adapter_classification() -> None:
    """A compact light rectangular object should be Charger Adapter."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="silver",
        size_category="medium",
        shape_category="rectangular",
        area_ratio=0.045,
        bbox_width_ratio=0.29,
        bbox_height_ratio=0.24,
        aspect_ratio=1.21,
        circularity=0.42,
        solidity=0.80,
        extent=0.65,
        edge_density=0.04,
    )

    decision = pipeline.decide(detection)

    assert decision.category == "Charger Adapter"
    assert decision.color == "silver"
    assert decision.size == "medium"
    assert not decision.is_unknown


def test_charger_adapter_is_not_classified_as_mouse() -> None:
    """A light charger-like object must not be classified as Mouse."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="white",
        size_category="medium",
        shape_category="rectangular",
        area_ratio=0.045,
        bbox_width_ratio=0.29,
        bbox_height_ratio=0.24,
        aspect_ratio=1.21,
        circularity=0.42,
        solidity=0.80,
        extent=0.65,
        edge_density=0.04,
    )

    decision = pipeline.decide(detection)

    assert decision.category != "Mouse"
    assert decision.category == "Charger Adapter"


def test_headphones_classification() -> None:
    """A large dark ring-like object should be classified as Headphones."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="black",
        size_category="large",
        shape_category="ring_like",
        area_ratio=0.207,
        bbox_width_ratio=0.88,
        bbox_height_ratio=0.52,
        aspect_ratio=1.69,
        circularity=0.25,
        solidity=0.70,
        extent=0.45,
        edge_density=0.10,
    )

    decision = pipeline.decide(detection)

    assert decision.category == "Headphones"
    assert decision.color == "black"
    assert decision.size == "large"
    assert not decision.is_unknown


def test_wired_earphones_classification() -> None:
    """A medium silver ring-like object should be classified as Headphones."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="silver",
        size_category="medium",
        shape_category="ring_like",
        area_ratio=0.122,
        bbox_width_ratio=0.62,
        bbox_height_ratio=0.40,
        aspect_ratio=1.55,
        circularity=0.22,
        solidity=0.68,
        extent=0.49,
        edge_density=0.09,
    )

    decision = pipeline.decide(detection)

    assert decision.category == "Headphones"
    assert decision.color == "silver"
    assert decision.size == "medium"
    assert not decision.is_unknown


def test_keyboard_is_not_classified_as_usb_cable() -> None:
    """A solid large rectangular object must not become USB-C Cable."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="black",
        size_category="large",
        shape_category="rectangular",
        area_ratio=0.24,
        bbox_width_ratio=0.36,
        bbox_height_ratio=0.71,
        aspect_ratio=1.97,
        circularity=0.30,
        solidity=0.90,
        extent=0.80,
        edge_density=0.04,
    )

    decision = pipeline.decide(detection)

    assert decision.category != "USB-C Cable"
    assert decision.category == "Keyboard"


def test_usb_cable_classification_for_long_thin_object() -> None:
    """A long thin neutral object should be classified as USB-C Cable."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color="white",
        size_category="long_thin",
        shape_category="irregular",
        area_ratio=0.035,
        bbox_width_ratio=0.70,
        bbox_height_ratio=0.12,
        aspect_ratio=5.83,
        circularity=0.10,
        solidity=0.50,
        extent=0.35,
        edge_density=0.08,
    )

    decision = pipeline.decide(detection)

    assert decision.category == "USB-C Cable"
    assert decision.color == "white"
    assert decision.size == "long_thin"
    assert not decision.is_unknown


@pytest.mark.parametrize(
    ("primary_color", "expected_category"),
    [
        ("red", "Colored Object"),
        ("green", "Colored Object"),
        ("blue", "Colored Object"),
    ],
)
def test_colored_objects_fallback(
    primary_color: str,
    expected_category: str,
) -> None:
    """Chromatic unsupported objects should become Colored Object."""
    pipeline = Pipeline()

    detection = make_detection(
        primary_color=primary_color,
        size_category="medium",
        shape_category="irregular",
        area_ratio=0.08,
        bbox_width_ratio=0.35,
        bbox_height_ratio=0.30,
        aspect_ratio=1.17,
        circularity=0.20,
        solidity=0.55,
        extent=0.40,
        edge_density=0.05,
    )

    decision = pipeline.decide(detection)

    assert decision.category == expected_category
    assert decision.color == primary_color