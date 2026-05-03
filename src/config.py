"""Configuration for Smart Storage — Object Sorting System."""

# --- Enhancement Parameters ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
BLUR_KERNEL = (5, 5)
GAMMA_VALUE = 1.15

# --- Segmentation Parameters ---
PRE_BLUR_KERNEL = (9, 9)
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 5

# --- Mask Cleaning Parameters ---
CLOSE_KERNEL_SIZE = (11, 11)
CLOSE_ITERATIONS = 2
OPEN_KERNEL_SIZE = (5, 5)
OPEN_ITERATIONS = 1

# --- Object Filtering ---
MIN_OBJECT_RATIO = 0.002
MAX_OBJECT_RATIO = 0.70
MAX_OBJECTS = 8

# --- Size Thresholds ---
SMALL_MAX_RATIO = 0.025
MEDIUM_MAX_RATIO = 0.18
LONG_THIN_ASPECT = 4.0

# --- K-Means Parameters ---
KMEANS_CLUSTERS = 3
KMEANS_MAX_ITER = 80
KMEANS_N_INIT = 5
KMEANS_MAX_PIXELS = 5000

# --- Confidence Thresholds ---
HIGH_CONFIDENCE = 0.60
LOW_CONFIDENCE = 0.25

# --- Camera ---
CAMERA_ID = 0

# --- Export ---
CSV_OUTPUT_PATH = "output/results.csv"

# --- Output folders ---
OUTPUT_DIR = "output"
STAGES_DIR = "output/stages"

# --- HSV Color Ranges ---
# Format: {color_name: {"h": (min, max), "s": (min, max), "v": (min, max)}}
# Special handling: white, black, gray checked by S/V first
HSV_RANGES = {
    "white": {"h": (0, 180), "s": (0, 45), "v": (185, 255)},
    "black": {"h": (0, 180), "s": (0, 120), "v": (0, 100)},
    "gray": {"h": (0, 180), "s": (0, 65), "v": (85, 210)},
    "silver": {"h": (0, 180), "s": (0, 55), "v": (140, 230)},

    "red_low": {"h": (0, 10), "s": (60, 255), "v": (50, 255)},
    "red_high": {"h": (170, 180), "s": (60, 255), "v": (50, 255)},
    "orange": {"h": (11, 24), "s": (60, 255), "v": (60, 255)},
    "yellow": {"h": (25, 34), "s": (60, 255), "v": (70, 255)},
    "green": {"h": (35, 85), "s": (45, 255), "v": (45, 255)},
    "blue": {"h": (90, 135), "s": (45, 255), "v": (45, 255)},
    "purple": {"h": (136, 160), "s": (45, 255), "v": (45, 255)},
    "brown": {"h": (10, 25), "s": (50, 180), "v": (40, 170)},
}

# --- Classification Rules ---
# These rules document the expected sorting categories.
# Format: list of dicts {color, size, category_ru, category_en, base_confidence}
CLASSIFICATION_RULES = [
    {
        "rule_id": "R01",
        "colors": ["black", "gray", "blue"],
        "sizes": ["medium", "large"],
        "shape_hints": ["oval", "block", "rectangular"],
        "category": "Mouse",
        "base_confidence": 0.86,
    },
    {
        "rule_id": "R02",
        "colors": ["black", "gray", "blue", "white", "silver"],
        "sizes": ["medium", "large", "long_thin"],
        "shape_hints": ["rectangular"],
        "category": "Keyboard",
        "base_confidence": 0.88,
    },
    {
        "rule_id": "R03",
        "colors": ["white", "silver", "gray"],
        "sizes": ["small", "medium"],
        "shape_hints": ["rectangular", "block", "oval"],
        "category": "Charger Adapter",
        "base_confidence": 0.84,
    },
    {
        "rule_id": "R04",
        "colors": ["black", "white", "silver", "gray"],
        "sizes": ["medium", "large"],
        "shape_hints": ["irregular", "ring_like", "block"],
        "category": "Headphones",
        "base_confidence": 0.78,
    },
    {
        "rule_id": "R05",
        "colors": ["white", "silver", "gray", "black"],
        "sizes": ["long_thin", "medium"],
        "shape_hints": ["ring_like", "irregular"],
        "category": "USB-C Cable",
        "base_confidence": 0.76,
    },
    {
        "rule_id": "R06",
        "colors": ["gray", "silver", "black", "green", "blue", "red"],
        "sizes": ["small"],
        "shape_hints": ["rectangular", "block"],
        "category": "Flash Drive",
        "base_confidence": 0.76,
    },
    {
        "rule_id": "R07",
        "colors": ["red", "green", "blue", "yellow", "orange", "purple", "brown"],
        "sizes": ["small", "medium", "large"],
        "shape_hints": ["oval", "rectangular", "block", "irregular", "ring_like"],
        "category": "Colored Object",
        "base_confidence": 0.65,
    },
]

# --- Size Match Factors for Confidence ---
SIZE_ADJACENCY = {
    ("small", "medium"): 0.75,
    ("medium", "small"): 0.75,
    ("medium", "large"): 0.75,
    ("large", "medium"): 0.75,
    ("small", "long_thin"): 0.35,
    ("long_thin", "small"): 0.35,
    ("medium", "long_thin"): 0.45,
    ("long_thin", "medium"): 0.45,
    ("large", "long_thin"): 0.35,
    ("long_thin", "large"): 0.35,
    ("small", "large"): 0.25,
    ("large", "small"): 0.25,
}