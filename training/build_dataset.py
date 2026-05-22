#!/usr/bin/env python3
"""Build a YOLO-Seg dataset from pipeline CSV exports and cleaned masks.

Reads ``output/results.csv``, resolves source frames and ``cleaned_mask`` artifacts,
converts mask contours to normalized polygon labels, splits train/val, and writes
``dataset/dataset.yaml``.

Also supports importing a Roboflow / CVAT YOLOv8 Segmentation export via
``--import-from``.

Usage:
    uv run python training/build_dataset.py
    uv run python training/build_dataset.py --csv output/results.csv --images-dir training/test_images
    uv run python training/build_dataset.py --regenerate-masks
    uv run python training/build_dataset.py --init
    uv run python training/build_dataset.py --import-from path/to/roboflow-export --clean
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# Project root on sys.path for ``from src...`` when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.app.data_exporter import DataExporter
from src.config import AppConfig

# YOLO training class slugs — must match dataset.yaml and yolo_classifier._CLASS_NAME_MAP keys.
YOLO_CLASS_NAMES: list[str] = [
    "mouse",
    "keyboard",
    "charger",
    "cable",
    "headphones",
    "flash_drive",
    "colored_object",
]

# Minimum images per class for the first training run (see project ML roadmap).
MIN_CLASS_COUNTS: dict[str, int] = {
    "mouse": 30,
    "keyboard": 20,
    "charger": 40,
    "cable": 40,
    "headphones": 30,
    "flash_drive": 30,
    "colored_object": 20,
}

# Legacy slug aliases from earlier dataset iterations / Roboflow exports.
SLUG_ALIASES: dict[str, str] = {
    "charger_adapter": "charger",
    "usb_cable": "cable",
    "usb-c_cable": "cable",
}

CATEGORY_TO_SLUG: dict[str, str] = {
    "mouse": "mouse",
    "keyboard": "keyboard",
    "charger adapter": "charger",
    "charger_adapter": "charger",
    "iphone charger": "charger",
    "android charger": "charger",
    "charger": "charger",
    "usb-c cable": "cable",
    "usb_cable": "cable",
    "power cable": "cable",
    "cable": "cable",
    "headphones": "headphones",
    "flash drive": "flash_drive",
    "flash_drive": "flash_drive",
    "flashdrive": "flash_drive",
    "colored object": "colored_object",
    "colored_object": "colored_object",
    "coloredobject": "colored_object",
}

DEFAULT_IMAGE_SEARCH_DIRS = (
    Path("training/test_images"),
    Path("output/images"),
)

SKIP_CATEGORY_PATTERNS = (
    "unknown",
    "неизвест",
    "объект не найден",
    "too small",
    "too large",
    "remote",
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SampleRecord:
    """One training sample derived from a CSV row."""

    image_name: str
    class_slug: str
    source_image: Path
    cleaned_mask: Path | None
    row_index: int


def _normalise_category_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalise_slug(slug: str) -> str | None:
    """Map a raw slug or alias to a canonical YOLO class slug."""
    key = _normalise_category_key(slug).replace("-", "_")
    if key in SLUG_ALIASES:
        key = SLUG_ALIASES[key]
    if key in YOLO_CLASS_NAMES:
        return key
    return None


def category_to_slug(category: str) -> str | None:
    """Map CSV category / ground_truth to a YOLO class slug, or None to skip."""
    if not category or not str(category).strip():
        return None

    key = _normalise_category_key(category)
    if any(pattern in key for pattern in SKIP_CATEGORY_PATTERNS):
        return None

    if key in CATEGORY_TO_SLUG:
        return CATEGORY_TO_SLUG[key]

    slug = key.replace(" ", "_").replace("-", "_")
    return normalise_slug(slug)


def resolve_label(row: pd.Series) -> str:
    """Pick ground_truth when set, otherwise predicted category."""
    gt = str(row.get("ground_truth", "")).strip()
    if gt:
        return gt
    return str(row.get("category", "")).strip()


LEGACY_CSV_COLUMNS = [
    "timestamp",
    "category",
    "color",
    "size_category",
    "confidence",
    "method",
    "area_pixels",
    "aspect_ratio",
    "color_hsv",
    "color_kmeans",
    "confidence_hsv",
    "confidence_kmeans",
]


def _looks_like_extended_csv_row(fields: list[str]) -> bool:
    """Heuristic: extended rows have image_name in column 2 and object_id in column 3."""
    if len(fields) < 4:
        return False
    image_token = fields[1].strip().lower()
    object_token = fields[2].strip()
    has_image_ext = any(image_token.endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)
    has_object_id = object_token.isdigit()
    return has_image_ext and has_object_id


def _parse_csv_row(fields: list[str], columns: list[str]) -> dict[str, str]:
    """Map one CSV line to the canonical DataExporter column layout."""
    if len(fields) <= len(LEGACY_CSV_COLUMNS) and not _looks_like_extended_csv_row(fields):
        padded = fields + [""] * max(0, len(LEGACY_CSV_COLUMNS) - len(fields))
        legacy = dict(zip(LEGACY_CSV_COLUMNS, padded[: len(LEGACY_CSV_COLUMNS)], strict=False))
        row = {column: "" for column in columns}
        row.update(legacy)
        return row

    padded = fields + [""] * max(0, len(columns) - len(fields))
    return dict(zip(columns, padded[: len(columns)], strict=False))


def load_csv_rows(csv_path: Path) -> pd.DataFrame:
    """Load results CSV, tolerating legacy rows with fewer columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    columns = DataExporter.COLUMNS
    rows: list[dict[str, str]] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for line_no, fields in enumerate(reader, start=1):
            if not fields or all(not field.strip() for field in fields):
                continue

            if line_no == 1 and fields[0].strip().lower() == "timestamp":
                continue

            rows.append(_parse_csv_row(fields, columns))

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    df = pd.DataFrame(rows, columns=columns)
    if "image_name" not in df.columns:
        df["image_name"] = ""

    return df


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest row per (image_name, object_id), preferring labeled ground_truth."""
    if df.empty:
        return df

    sort_cols = ["timestamp"] if "timestamp" in df.columns else []
    ordered = df.sort_values(sort_cols) if sort_cols else df.copy()

    def _row_priority(row: pd.Series) -> tuple[int, int]:
        has_gt = int(bool(str(row.get("ground_truth", "")).strip()))
        has_image = int(bool(str(row.get("image_name", "")).strip()))
        return (has_gt, has_image)

    key_cols = ["image_name"]
    if "object_id" in ordered.columns:
        key_cols.append("object_id")

    best_rows: list[pd.Series] = []
    for _, group in ordered.groupby(key_cols, dropna=False):
        ranked = sorted(group.iterrows(), key=lambda item: _row_priority(item[1]), reverse=True)
        best_rows.append(ranked[0][1])

    return pd.DataFrame(best_rows).reset_index(drop=True)


def find_source_image(image_name: str, search_dirs: list[Path]) -> Path | None:
    """Locate a source frame by file name across search directories."""
    if not image_name or image_name == "webcam_frame":
        return None

    name = Path(image_name).name
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        direct = directory / name
        if direct.is_file():
            return direct
        for path in directory.rglob(name):
            if path.is_file():
                return path
    return None


def find_cleaned_mask(image_name: str, stages_dir: Path) -> Path | None:
    """Return cleaned_mask artifact path for an image name stem."""
    if not image_name:
        return None

    stem = Path(image_name).stem
    mask_path = stages_dir / "cleaned_mask" / f"{stem}_cleaned_mask.jpg"
    if mask_path.is_file():
        return mask_path

    # Fallback: any cleaned mask whose name starts with the stem.
    mask_dir = stages_dir / "cleaned_mask"
    if not mask_dir.is_dir():
        return None

    candidates = sorted(mask_dir.glob(f"{stem}*_cleaned_mask.jpg"))
    return candidates[-1] if candidates else None


def mask_to_polygon(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    epsilon_ratio: float = 0.002,
    min_points: int = 3,
    max_points: int = 128,
) -> list[tuple[float, float]] | None:
    """Extract a simplified contour polygon from a binary mask."""
    if mask is None or mask.size == 0:
        return None

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()

    if gray.shape[:2] != (image_height, image_width):
        gray = cv2.resize(gray, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(epsilon_ratio * perimeter, 1.0)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < min_points:
        return None

    points = approx.reshape(-1, 2).astype(np.float64)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]

    normalized: list[tuple[float, float]] = []
    for x, y in points:
        nx = min(max(float(x) / image_width, 0.0), 1.0)
        ny = min(max(float(y) / image_height, 0.0), 1.0)
        normalized.append((nx, ny))

    return normalized


def polygon_to_yolo_line(class_id: int, polygon: list[tuple[float, float]]) -> str:
    """Format one YOLO-Seg label line."""
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
    return f"{class_id} {coords}"


def bbox_to_polygon(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    """Fallback rectangle polygon from (x, y, w, h) when mask is unavailable."""
    x, y, w, h = bbox
    x2 = min(x + w, image_width)
    y2 = min(y + h, image_height)
    x1 = max(x, 0)
    y1 = max(y, 0)
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [
        (min(max(px / image_width, 0.0), 1.0), min(max(py / image_height, 0.0), 1.0))
        for px, py in corners
    ]


def generate_mask_from_pipeline(image: np.ndarray) -> np.ndarray | None:
    """Run the rule-based pipeline to obtain a cleaned mask for one frame."""
    from src.pipeline import Pipeline

    pipeline = Pipeline(AppConfig())
    result = pipeline.run(image)
    if result is None:
        return None
    return result.cleaned_mask


def build_dataset_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """Write dataset.yaml for Ultralytics training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "dataset.yaml"
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return yaml_path


def init_dataset_layout(output_dir: Path) -> Path:
    """Create empty YOLO-Seg directory tree and dataset.yaml."""
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    return build_dataset_yaml(output_dir, YOLO_CLASS_NAMES)


def _load_export_names(export_root: Path) -> dict[int, str]:
    """Read class names from a Roboflow/CVAT data.yaml export."""
    for candidate in (export_root / "data.yaml", export_root / "dataset.yaml"):
        if not candidate.is_file():
            continue
        with candidate.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        names = cfg.get("names", {})
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
    return {}


def _remap_label_line(line: str, export_names: dict[int, str]) -> str | None:
    """Convert one YOLO-Seg label line to canonical class IDs."""
    parts = line.strip().split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        return None

    class_id = int(parts[0])
    raw_name = export_names.get(class_id, str(class_id))
    slug = category_to_slug(raw_name) or normalise_slug(raw_name)
    if slug is None:
        return None

    canonical_id = YOLO_CLASS_NAMES.index(slug)
    return f"{canonical_id} {' '.join(parts[1:])}"


def _discover_export_splits(export_root: Path) -> list[tuple[str, Path, Path]]:
    """Return (split_name, images_dir, labels_dir) tuples for a YOLO export."""
    split_aliases = {
        "train": "train",
        "valid": "val",
        "validation": "val",
        "val": "val",
    }
    found: list[tuple[str, Path, Path]] = []

    for images_dir in sorted(export_root.rglob("images")):
        if not images_dir.is_dir():
            continue
        split_key = images_dir.parent.name.lower()
        if split_key not in split_aliases and images_dir.name.lower() not in split_aliases:
            continue

        split_name = split_aliases.get(images_dir.name.lower(), split_aliases.get(split_key, "train"))
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            labels_dir = export_root / "labels" / images_dir.name
        if labels_dir.is_dir():
            found.append((split_name, images_dir, labels_dir))

    if found:
        return found

    for split_name in ("train", "val"):
        images_dir = export_root / "images" / split_name
        labels_dir = export_root / "labels" / split_name
        if images_dir.is_dir() and labels_dir.is_dir():
            found.append((split_name, images_dir, labels_dir))

    return found


def import_yolo_export(export_root: Path, output_dir: Path) -> tuple[Counter[str], Counter[str]]:
    """Import Roboflow/CVAT YOLOv8 Segmentation export into project layout."""
    export_root = export_root.resolve()
    if not export_root.is_dir():
        raise FileNotFoundError(f"Export directory not found: {export_root}")

    splits = _discover_export_splits(export_root)
    if not splits:
        raise ValueError(
            f"No YOLO split folders found under {export_root}. "
            "Expected train/images + train/labels or images/train + labels/train.",
        )

    export_names = _load_export_names(export_root)
    train_counts: Counter[str] = Counter()
    val_counts: Counter[str] = Counter()

    for split_name, images_dir, labels_dir in splits:
        out_images = output_dir / "images" / split_name
        out_labels = output_dir / "labels" / split_name
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        counter = train_counts if split_name == "train" else val_counts

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue

            remapped_lines: list[str] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                remapped = _remap_label_line(line, export_names)
                if remapped:
                    remapped_lines.append(remapped)

            if not remapped_lines:
                continue

            shutil.copy2(image_path, out_images / image_path.name)
            (out_labels / f"{image_path.stem}.txt").write_text(
                "\n".join(remapped_lines) + "\n",
                encoding="utf-8",
            )

            class_id = int(remapped_lines[0].split()[0])
            counter[YOLO_CLASS_NAMES[class_id]] += 1

    return train_counts, val_counts


def print_minimum_gaps(train_counts: Counter[str], val_counts: Counter[str]) -> None:
    """Report progress toward per-class minimum image counts."""
    combined = Counter(train_counts)
    combined.update(val_counts)
    total_required = sum(MIN_CLASS_COUNTS.values())

    print(f"\nMinimum dataset targets ({total_required} images total):")
    for slug in YOLO_CLASS_NAMES:
        have = combined.get(slug, 0)
        need = MIN_CLASS_COUNTS[slug]
        status = "OK" if have >= need else f"need {need - have} more"
        print(f"  {slug}: {have}/{need} — {status}")

    total_have = sum(combined.values())
    if total_have < total_required:
        print(f"\nOverall: {total_have}/{total_required} — add more labeled images before training.")
    else:
        print(f"\nOverall: {total_have}/{total_required} — minimum reached.")


def stratified_split(
    samples: list[SampleRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split samples into train/val, stratifying by class when possible."""
    if not samples:
        return [], []

    rng = random.Random(seed)
    by_class: dict[str, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        by_class[sample.class_slug].append(sample)

    train: list[SampleRecord] = []
    val: list[SampleRecord] = []

    for class_samples in by_class.values():
        items = class_samples.copy()
        rng.shuffle(items)
        if len(items) == 1:
            train.append(items[0])
            continue

        val_count = max(1, int(round(len(items) * val_ratio)))
        val_count = min(val_count, len(items) - 1)
        val.extend(items[:val_count])
        train.extend(items[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def collect_samples(
    df: pd.DataFrame,
    *,
    search_dirs: list[Path],
    stages_dir: Path,
    min_confidence: float,
    regenerate_masks: bool,
) -> tuple[list[SampleRecord], list[str]]:
    """Build sample list from CSV; return (samples, skip reasons)."""
    samples: list[SampleRecord] = []
    skipped: list[str] = []

    for index, row in df.iterrows():
        label = resolve_label(row)
        slug = category_to_slug(label)
        if slug is None:
            skipped.append(f"row {index}: unmapped category '{label}'")
            continue

        if min_confidence > 0 and "confidence" in row.index:
            try:
                confidence = float(row["confidence"])
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < min_confidence:
                skipped.append(f"row {index}: low confidence {confidence:.3f}")
                continue

        image_name = str(row.get("image_name", "")).strip()
        source_image = find_source_image(image_name, search_dirs)
        if source_image is None:
            skipped.append(f"row {index}: source image not found for '{image_name}'")
            continue

        cleaned_mask = find_cleaned_mask(image_name, stages_dir)
        if cleaned_mask is None and regenerate_masks:
            image = cv2.imread(str(source_image))
            if image is not None:
                mask = generate_mask_from_pipeline(image)
                if mask is not None:
                    mask_dir = stages_dir / "cleaned_mask"
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    stem = Path(image_name).stem
                    cleaned_mask = mask_dir / f"{stem}_cleaned_mask.jpg"
                    cv2.imwrite(str(cleaned_mask), mask)

        samples.append(
            SampleRecord(
                image_name=image_name,
                class_slug=slug,
                source_image=source_image,
                cleaned_mask=cleaned_mask,
                row_index=int(index),
            )
        )

    return samples, skipped


def write_split(
    split_samples: list[SampleRecord],
    split_name: str,
    output_dir: Path,
    class_to_id: dict[str, int],
    *,
    epsilon_ratio: float,
    use_bbox_fallback: bool,
) -> tuple[int, Counter[str]]:
    """Copy images and write YOLO-Seg label files for one split."""
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    written = 0

    for sample in split_samples:
        image = cv2.imread(str(sample.source_image))
        if image is None:
            continue

        height, width = image.shape[:2]
        polygon: list[tuple[float, float]] | None = None

        if sample.cleaned_mask and sample.cleaned_mask.is_file():
            mask = cv2.imread(str(sample.cleaned_mask), cv2.IMREAD_GRAYSCALE)
            polygon = mask_to_polygon(
                mask,
                width,
                height,
                epsilon_ratio=epsilon_ratio,
            )

        if polygon is None and use_bbox_fallback:
            from src.pipeline import Pipeline

            pipeline = Pipeline(AppConfig())
            result = pipeline.run(image)
            if result is not None and result.detection is not None:
                polygon = bbox_to_polygon(result.detection.bbox, width, height)

        if polygon is None:
            continue

        stem = Path(sample.image_name).stem
        out_stem = f"{stem}_{sample.row_index}"
        out_image = images_dir / f"{out_stem}.jpg"
        out_label = labels_dir / f"{out_stem}.txt"

        shutil.copy2(sample.source_image, out_image)

        class_id = class_to_id[sample.class_slug]
        line = polygon_to_yolo_line(class_id, polygon)
        out_label.write_text(line + "\n", encoding="utf-8")

        class_counts[sample.class_slug] += 1
        written += 1

    return written, class_counts


def print_statistics(
    train_counts: Counter[str],
    val_counts: Counter[str],
    skipped: list[str],
    yaml_path: Path,
) -> None:
    """Print dataset build summary."""
    print("Smart Storage — YOLO-Seg Dataset Builder")
    print("=" * 50)
    print(f"dataset.yaml: {yaml_path}")

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"\nSamples written: train={total_train}, val={total_val}, total={total_train + total_val}")

    print("\nClass distribution (train):")
    for slug in YOLO_CLASS_NAMES:
        count = train_counts.get(slug, 0)
        if count:
            print(f"  {slug}: {count}")

    print("\nClass distribution (val):")
    for slug in YOLO_CLASS_NAMES:
        count = val_counts.get(slug, 0)
        if count:
            print(f"  {slug}: {count}")

    if skipped:
        print(f"\nSkipped rows: {len(skipped)}")
        for reason in skipped[:10]:
            print(f"  - {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    print_minimum_gaps(train_counts, val_counts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO-Seg dataset from pipeline CSV and cleaned masks.",
    )
    parser.add_argument("--csv", default="output/results.csv", help="Path to results CSV.")
    parser.add_argument(
        "--images-dir",
        action="append",
        default=[],
        help="Directory to search for source frames (repeatable). Default: training/test_images.",
    )
    parser.add_argument("--stages-dir", default="output/stages", help="Pipeline stages directory.")
    parser.add_argument("--output", default="dataset", help="Output dataset root directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument(
        "--require-ground-truth",
        action="store_true",
        help="Only include rows with non-empty ground_truth.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Skip rows below this confidence (0 = no filter).",
    )
    parser.add_argument(
        "--regenerate-masks",
        action="store_true",
        help="Run pipeline to create cleaned_mask when artifact is missing.",
    )
    parser.add_argument(
        "--bbox-fallback",
        action="store_true",
        help="Use detection bbox rectangle if mask contour extraction fails.",
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.002,
        help="approxPolyDP epsilon as fraction of contour perimeter.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output directory before building.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create empty dataset/ layout and dataset.yaml (no CSV processing).",
    )
    parser.add_argument(
        "--import-from",
        metavar="DIR",
        help="Import Roboflow/CVAT YOLOv8 Segmentation export from DIR.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output)

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    if args.init:
        yaml_path = init_dataset_layout(output_dir)
        print("Smart Storage — YOLO-Seg Dataset Builder")
        print("=" * 50)
        print(f"Initialized empty dataset at: {output_dir.resolve()}")
        print(f"dataset.yaml: {yaml_path}")
        print("\nNext steps:")
        print("  1. Label images in Roboflow/CVAT (YOLOv8 Segmentation export)")
        print("  2. uv run python scripts/build_dataset.py --import-from <export-dir> --clean")
        print("  3. uv run python scripts/train_yolo.py --validate-only --data dataset/dataset.yaml")
        print_minimum_gaps(Counter(), Counter())
        return 0

    if args.import_from:
        export_root = Path(args.import_from)
        try:
            train_counts, val_counts = import_yolo_export(export_root, output_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return 1

        if sum(train_counts.values()) + sum(val_counts.values()) == 0:
            print("No labeled images imported. Check export format and class names.")
            return 1

        yaml_path = build_dataset_yaml(output_dir, YOLO_CLASS_NAMES)
        print_statistics(train_counts, val_counts, [], yaml_path)
        return 0

    csv_path = Path(args.csv)
    stages_dir = Path(args.stages_dir)

    search_dirs = [Path(p) for p in args.images_dir] if args.images_dir else list(DEFAULT_IMAGE_SEARCH_DIRS)

    df = load_csv_rows(csv_path)
    if args.require_ground_truth:
        df = df[df["ground_truth"].astype(str).str.strip() != ""].copy()

    df = deduplicate_rows(df)
    samples, skipped = collect_samples(
        df,
        search_dirs=search_dirs,
        stages_dir=stages_dir,
        min_confidence=args.min_confidence,
        regenerate_masks=args.regenerate_masks,
    )

    if not samples:
        print("No valid samples found. Check CSV labels, image paths, and stages masks.")
        for reason in skipped[:15]:
            print(f"  - {reason}")
        return 1

    train_samples, val_samples = stratified_split(samples, args.val_ratio, args.seed)
    class_to_id = {name: index for index, name in enumerate(YOLO_CLASS_NAMES)}

    train_written, train_counts = write_split(
        train_samples,
        "train",
        output_dir,
        class_to_id,
        epsilon_ratio=args.epsilon_ratio,
        use_bbox_fallback=args.bbox_fallback,
    )
    val_written, val_counts = write_split(
        val_samples,
        "val",
        output_dir,
        class_to_id,
        epsilon_ratio=args.epsilon_ratio,
        use_bbox_fallback=args.bbox_fallback,
    )

    if train_written == 0 and val_written == 0:
        print("No label files written — cleaned_mask artifacts may be missing.")
        print("Re-run batch mode with stage saving, or pass --regenerate-masks.")
        return 1

    yaml_path = build_dataset_yaml(output_dir, YOLO_CLASS_NAMES)
    print_statistics(train_counts, val_counts, skipped, yaml_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
