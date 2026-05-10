"""Configuration for Smart Storage — Object Sorting System."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field


@dataclass
class ClassificationRule:
    """Single classification rule mapping color + size to a category."""

    rule_id: str
    color: str
    size: str
    category_ru: str
    category_en: str
    base_confidence: float


def _default_hsv_ranges() -> dict[str, dict[str, tuple[int, int]]]:
    return {
        "white":    {"h": (0, 180),   "s": (0, 45),   "v": (185, 255)},
        "black":    {"h": (0, 180),   "s": (0, 120),  "v": (0, 100)},
        "gray":     {"h": (0, 180),   "s": (0, 65),   "v": (85, 210)},
        "silver":   {"h": (0, 180),   "s": (0, 55),   "v": (140, 230)},
        "red_low":  {"h": (0, 10),    "s": (60, 255), "v": (50, 255)},
        "red_high": {"h": (170, 180), "s": (60, 255), "v": (50, 255)},
        "orange":   {"h": (11, 24),   "s": (60, 255), "v": (60, 255)},
        "yellow":   {"h": (25, 34),   "s": (60, 255), "v": (70, 255)},
        "green":    {"h": (35, 85),   "s": (45, 255), "v": (45, 255)},
        "blue":     {"h": (90, 135),  "s": (45, 255), "v": (45, 255)},
        "purple":   {"h": (136, 160), "s": (45, 255), "v": (45, 255)},
        "brown":    {"h": (10, 25),   "s": (50, 180), "v": (40, 170)},
    }


def _default_rules() -> list[ClassificationRule]:
    return [
        ClassificationRule("R01", "white",  "small",     "Зарядка iPhone",      "iPhone Charger",    0.85),
        ClassificationRule("R02", "black",  "small",     "Зарядка Android",     "Android Charger",   0.80),
        ClassificationRule("R03", "black",  "long_thin", "Кабель питания",      "Power Cable",       0.85),
        ClassificationRule("R04", "white",  "long_thin", "Кабель USB-C",        "USB-C Cable",       0.80),
        ClassificationRule("R05", "black",  "medium",    "Мышь",                "Mouse",             0.85),
        ClassificationRule("R06", "white",  "medium",    "Мышь (белая)",        "Mouse (White)",     0.80),
        ClassificationRule("R07", "black",  "large",     "Клавиатура",          "Keyboard",          0.90),
        ClassificationRule("R08", "white",  "large",     "Клавиатура (белая)",  "Keyboard (White)",  0.85),
        ClassificationRule("R09", "gray",   "small",     "Флешка",              "Flash Drive",       0.80),
        ClassificationRule("R10", "silver", "small",     "Флешка",              "Flash Drive",       0.80),
    ]


_TUPLE_FIELDS = frozenset({
    "clahe_tile_size",
    "blur_kernel",
    "pre_blur_kernel",
    "close_kernel_size",
    "open_kernel_size",
})


@dataclass
class AppConfig:
    """Single source of truth for all pipeline parameters."""

    # Enhancement
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = (8, 8)
    blur_kernel: tuple = (5, 5)
    gamma_value: float = 1.15

    # Segmentation
    pre_blur_kernel: tuple = (9, 9)
    adaptive_block_size: int = 31
    adaptive_c: int = 5

    # Mask cleaning
    close_kernel_size: tuple = (11, 11)
    close_iterations: int = 2
    open_kernel_size: tuple = (5, 5)
    open_iterations: int = 1

    # Object filtering
    min_object_ratio: float = 0.002
    max_object_ratio: float = 0.70
    max_objects: int = 8

    # Size thresholds
    small_max_ratio: float = 0.025
    medium_max_ratio: float = 0.18
    long_thin_aspect: float = 4.0

    # K-means
    kmeans_clusters: int = 3
    kmeans_max_iter: int = 80
    kmeans_n_init: int = 5
    kmeans_max_pixels: int = 5000

    # Confidence thresholds
    high_confidence: float = 0.60
    low_confidence: float = 0.25

    # Camera
    camera_id: int = 0

    # Export paths
    csv_output_path: str = "output/results.csv"
    output_dir: str = "output"
    stages_dir: str = "output/stages"

    # HSV color ranges
    hsv_ranges: dict = field(default_factory=_default_hsv_ranges)

    # Classification rules
    rules: list = field(default_factory=_default_rules)

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_json(cls, path: str) -> "AppConfig":
        """Load configuration from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for field_name in _TUPLE_FIELDS:
            if field_name in data and isinstance(data[field_name], list):
                data[field_name] = tuple(data[field_name])

        if "hsv_ranges" in data:
            data["hsv_ranges"] = {
                color: {
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in ranges.items()
                }
                for color, ranges in data["hsv_ranges"].items()
            }

        if "rules" in data:
            data["rules"] = [
                ClassificationRule(**r) if isinstance(r, dict) else r
                for r in data["rules"]
            ]

        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_serializable(), f, indent=2, ensure_ascii=False)

    def _to_serializable(self) -> dict:
        """Return a JSON-serialisable representation of this config."""
        result: dict = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                result[f.name] = list(value)
            elif f.name == "rules":
                result[f.name] = [dataclasses.asdict(r) for r in value]
            elif f.name == "hsv_ranges":
                result[f.name] = {
                    color: {
                        k: list(v) if isinstance(v, tuple) else v
                        for k, v in ranges.items()
                    }
                    for color, ranges in value.items()
                }
            else:
                result[f.name] = value
        return result
