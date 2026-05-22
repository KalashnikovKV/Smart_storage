"""Configuration for Smart Storage — Object Sorting System."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field


@dataclass
class ClassificationRule:
    """Single classification rule mapping color + size + shape to a category."""

    rule_id: str
    category_en: str
    colors: list  # accepted primary_color values
    sizes: list   # accepted size_category values
    shape_hints: list  # accepted shape_category values
    base_confidence: float
    category_ru: str = ""
    # Optional hard-limit guards evaluated in Pipeline._score_rule
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 999.0
    min_solidity: float = 0.0
    max_solidity: float = 1.0
    min_extent: float = 0.0
    max_edge_density: float = 1.0
    min_area_ratio: float = 0.0
    max_area_ratio: float = 999.0
    # Circularity guard applied only when shape_category == "rectangular"
    min_circularity_if_rectangular: float = 0.0


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
        # R01: Keyboard — large dark elongated rectangular object
        ClassificationRule(
            rule_id="R01",
            category_en="Keyboard",
            category_ru="Клавиатура",
            colors=["black", "gray", "blue", "silver"],
            sizes=["large"],
            shape_hints=["rectangular"],
            base_confidence=0.90,
            min_aspect_ratio=1.70,
            min_area_ratio=0.12,
            min_extent=0.45,
            min_solidity=0.60,
        ),
        # R02: Mouse — compact dark rounded object
        ClassificationRule(
            rule_id="R02",
            category_en="Mouse",
            category_ru="Мышь",
            colors=["black", "gray", "blue"],
            sizes=["medium", "large"],
            shape_hints=["oval", "block", "rectangular"],
            base_confidence=0.88,
            max_aspect_ratio=2.35,
            min_solidity=0.58,
            min_extent=0.28,
            min_area_ratio=0.02,
            max_area_ratio=0.32,
        ),
        # R03: Charger Adapter — compact light solid block/rect/oval
        # Rectangular shape requires circularity >= 0.25 to separate from cable hollow
        ClassificationRule(
            rule_id="R03",
            category_en="Charger Adapter",
            category_ru="Зарядка",
            colors=["white", "silver", "gray"],
            sizes=["small", "medium"],
            shape_hints=["block", "oval", "rectangular"],
            base_confidence=0.84,
            max_aspect_ratio=2.20,
            min_solidity=0.40,
            min_extent=0.25,
            max_area_ratio=0.18,
            max_edge_density=0.18,
            min_circularity_if_rectangular=0.25,
        ),
        # R04: Flash Drive — small compact object in any supported color
        ClassificationRule(
            rule_id="R04",
            category_en="Flash Drive",
            category_ru="Флешка",
            colors=["gray", "silver", "black", "green", "blue", "red"],
            sizes=["small"],
            shape_hints=["block", "rectangular", "oval", "irregular"],
            base_confidence=0.76,
            max_aspect_ratio=3.5,
            min_extent=0.30,
        ),
        # R05: USB-C Cable — elongated or loop-shaped neutral object
        # Solidity < 0.62 separates cable ring_like from headphones ring_like
        ClassificationRule(
            rule_id="R05",
            category_en="USB-C Cable",
            category_ru="Кабель USB-C",
            colors=["white", "silver", "gray", "black"],
            sizes=["long_thin", "small", "medium"],
            shape_hints=["irregular", "ring_like", "rectangular", "block"],
            base_confidence=0.76,
            max_solidity=0.619,
        ),
        # R06: Headphones — medium/large ring-like object with sufficient solidity
        # Solidity >= 0.62 is the primary separator from cables
        ClassificationRule(
            rule_id="R06",
            category_en="Headphones",
            category_ru="Наушники",
            colors=["black", "white", "silver", "gray"],
            sizes=["medium", "large"],
            shape_hints=["ring_like", "irregular"],
            base_confidence=0.78,
            min_solidity=0.62,
        ),
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
    min_component_area_ratio: float = 0.0015

    # Color-distance segmentation (LAB space)
    color_distance_sigma: float = 2.8
    color_distance_min_delta: float = 14.0
    color_distance_soft_sigma: float = 1.4

    # In-contour refinement (multi-color objects on dark backgrounds)
    min_saturation: int = 35
    brightness_delta: int = 28
    min_detection_extent: float = 0.20

    # Shadow suppression inside object masks
    shadow_bg_distance_ratio: float = 0.90
    shadow_l_delta: float = 12.0
    shadow_max_saturation: float = 48.0
    shadow_bg_l_margin: float = 10.0
    shadow_min_keep_ratio: float = 0.28
    shadow_min_core_distance: float = 10.0
    shadow_core_dilate_iterations: int = 2

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

    # YOLO inference
    device: str = "cpu"              # "cpu", "cuda", "cuda:0", "mps" (Apple Silicon)
    yolo_weights: str = "models/yolo_seg_best.pt"
    yolo_conf_threshold: float = 0.5
    yolo_iou_threshold: float = 0.45

    # Export paths
    csv_output_path: str = "output/results.csv"
    output_dir: str = "output"
    stages_dir: str = "output/stages"
    save_roi_images: bool = True

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
