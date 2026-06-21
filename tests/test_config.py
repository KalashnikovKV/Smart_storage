"""Tests for AppConfig dataclass serialisation."""

import json

from src.config import AppConfig, ClassificationRule


def test_appconfig_defaults():
    cfg = AppConfig()
    assert cfg.clahe_clip_limit == 2.0
    assert cfg.clahe_tile_size == (8, 8)
    assert cfg.blur_kernel == (5, 5)
    assert cfg.camera_id == 0
    assert cfg.csv_output_path == "output/results.csv"


def test_appconfig_hsv_ranges_present():
    cfg = AppConfig()
    assert "white" in cfg.hsv_ranges
    assert "black" in cfg.hsv_ranges
    assert isinstance(cfg.hsv_ranges["white"]["h"], tuple)


def test_appconfig_rules_are_dataclass_instances():
    cfg = AppConfig()
    assert len(cfg.rules) == 6
    assert all(isinstance(r, ClassificationRule) for r in cfg.rules)
    assert cfg.rules[0].rule_id == "R01"
    assert all(hasattr(r, "colors") for r in cfg.rules)
    assert all(hasattr(r, "shape_hints") for r in cfg.rules)


def test_to_json_creates_file(tmp_path):
    cfg = AppConfig()
    path = str(tmp_path / "config.json")
    cfg.to_json(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["clahe_clip_limit"] == 2.0
    assert data["clahe_tile_size"] == [8, 8]
    assert "white" in data["hsv_ranges"]
    assert len(data["rules"]) == 6


def test_from_json_roundtrip(tmp_path):
    original = AppConfig(clahe_clip_limit=3.5, camera_id=2)
    path = str(tmp_path / "config.json")
    original.to_json(path)

    restored = AppConfig.from_json(path)

    assert restored.clahe_clip_limit == 3.5
    assert restored.camera_id == 2
    assert restored.clahe_tile_size == (8, 8)
    assert isinstance(restored.clahe_tile_size, tuple)
    assert isinstance(restored.hsv_ranges["white"]["h"], tuple)
    assert all(isinstance(r, ClassificationRule) for r in restored.rules)


def test_from_json_preserves_hsv_tuples(tmp_path):
    cfg = AppConfig()
    path = str(tmp_path / "cfg.json")
    cfg.to_json(path)
    loaded = AppConfig.from_json(path)

    for color, ranges in loaded.hsv_ranges.items():
        for key, value in ranges.items():
            assert isinstance(value, tuple), (
                f"hsv_ranges[{color!r}][{key!r}] should be tuple, got {type(value)}"
            )


def test_pipeline_uses_appconfig():
    """Pipeline should accept and store an AppConfig instance."""
    from src.pipeline import Pipeline

    cfg = AppConfig(low_confidence=0.10)
    pipeline = Pipeline(config=cfg)
    assert pipeline.config.low_confidence == 0.10


def test_color_detector_uses_appconfig():
    """ColorDetector should use the passed AppConfig's hsv_ranges."""
    from src.pipeline.detect_color import ColorDetector

    cfg = AppConfig()
    detector = ColorDetector(cfg)
    assert detector.hsv_ranges is cfg.hsv_ranges
    assert detector.kmeans_clusters == cfg.kmeans_clusters
