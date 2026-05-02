"""Configuration and classification rules for Smart Storage."""

# --- Enhancement Parameters ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
BLUR_KERNEL = (5, 5)

# --- Segmentation Parameters ---
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2
PRE_BLUR_KERNEL = (11, 11)

# --- Mask Cleaning Parameters ---
CLOSE_KERNEL_SIZE = (7, 7)
CLOSE_ITERATIONS = 2
OPEN_KERNEL_SIZE = (5, 5)
OPEN_ITERATIONS = 2

# --- Size Thresholds ---
SMALL_MAX_RATIO = 0.04
MEDIUM_MAX_RATIO = 0.35
LONG_THIN_ASPECT = 4.0
MIN_OBJECT_RATIO = 0.005
MAX_OBJECT_RATIO = 0.60

# --- K-Means Parameters ---
KMEANS_CLUSTERS = 3
KMEANS_MAX_ITER = 100
KMEANS_N_INIT = 10

# --- Confidence Thresholds ---
HIGH_CONFIDENCE = 0.60
LOW_CONFIDENCE = 0.25

# --- Camera ---
CAMERA_ID = 0

# --- Export ---
CSV_OUTPUT_PATH = "output/results.csv"

# --- HSV Color Ranges ---
# Format: {color_name: {"h": (min, max), "s": (min, max), "v": (min, max)}}
# Special handling: white, black, gray checked by S/V first
HSV_RANGES = {
    "white": {"h": (0, 180), "s": (0, 50), "v": (200, 255)},
    "black": {"h": (0, 180), "s": (0, 255), "v": (0, 50)},
    "gray": {"h": (0, 180), "s": (0, 50), "v": (50, 200)},
    "silver": {"h": (0, 180), "s": (0, 50), "v": (150, 200)},
    "red_low": {"h": (0, 10), "s": (50, 255), "v": (50, 255)},
    "red_high": {"h": (170, 180), "s": (50, 255), "v": (50, 255)},
    "blue": {"h": (100, 130), "s": (50, 255), "v": (50, 255)},
    "green": {"h": (35, 85), "s": (50, 255), "v": (50, 255)},
}

# --- Classification Rules ---
# Format: list of dicts {color, size, category_ru, category_en, base_confidence}
CLASSIFICATION_RULES = [
    {"rule_id": "R01", "color": "white", "size": "small",
     "category_ru": "iPhone Charger", "category_en": "iPhone Charger",
     "base_confidence": 0.85},
    {"rule_id": "R02", "color": "black", "size": "small",
     "category_ru": "Android Charger", "category_en": "Android Charger",
     "base_confidence": 0.80},
    {"rule_id": "R03", "color": "black", "size": "long_thin",
     "category_ru": "Power Cable", "category_en": "Power Cable",
     "base_confidence": 0.85},
    {"rule_id": "R04", "color": "white", "size": "long_thin",
     "category_ru": "USB-C Cable", "category_en": "USB-C Cable",
     "base_confidence": 0.80},
    {"rule_id": "R05", "color": "black", "size": "medium",
     "category_ru": "Mouse", "category_en": "Mouse",
     "base_confidence": 0.85},
    {"rule_id": "R06", "color": "white", "size": "medium",
     "category_ru": "Mouse (White)", "category_en": "Mouse (White)",
     "base_confidence": 0.80},
    {"rule_id": "R07", "color": "black", "size": "large",
     "category_ru": "Keyboard", "category_en": "Keyboard",
     "base_confidence": 0.90},
    {"rule_id": "R08", "color": "white", "size": "large",
     "category_ru": "Keyboard (White)", "category_en": "Keyboard (White)",
     "base_confidence": 0.85},
    {"rule_id": "R09", "color": "gray", "size": "small",
     "category_ru": "Flash Drive", "category_en": "Flash Drive",
     "base_confidence": 0.80},
    {"rule_id": "R10", "color": "silver", "size": "small",
     "category_ru": "Flash Drive", "category_en": "Flash Drive",
     "base_confidence": 0.80},
    # Extra rules for dark-blue objects (common for Logitech mice)
    {"rule_id": "R11", "color": "blue", "size": "medium",
     "category_ru": "Mouse", "category_en": "Mouse",
     "base_confidence": 0.80},
    {"rule_id": "R12", "color": "blue", "size": "large",
     "category_ru": "Mouse", "category_en": "Mouse",
     "base_confidence": 0.75},
    # Mouse can appear large when close to camera
    {"rule_id": "R13", "color": "black", "size": "large",
     "category_ru": "Mouse", "category_en": "Mouse",
     "base_confidence": 0.75},
    # Keyboard rules — gray/dark keyboards (very common)
    {"rule_id": "R14", "color": "gray", "size": "medium",
     "category_ru": "Keyboard", "category_en": "Keyboard",
     "base_confidence": 0.85},
    {"rule_id": "R15", "color": "gray", "size": "large",
     "category_ru": "Keyboard", "category_en": "Keyboard",
     "base_confidence": 0.90},
    {"rule_id": "R16", "color": "blue", "size": "medium",
     "category_ru": "Keyboard", "category_en": "Keyboard",
     "base_confidence": 0.70},
]

# --- Size Match Factors for Confidence ---
SIZE_ADJACENCY = {
    ("small", "medium"): 0.7,
    ("medium", "small"): 0.7,
    ("medium", "large"): 0.7,
    ("large", "medium"): 0.7,
    ("small", "long_thin"): 0.3,
    ("long_thin", "small"): 0.3,
    ("medium", "long_thin"): 0.3,
    ("long_thin", "medium"): 0.3,
    ("large", "long_thin"): 0.3,
    ("long_thin", "large"): 0.3,
    ("small", "large"): 0.3,
    ("large", "small"): 0.3,
}
