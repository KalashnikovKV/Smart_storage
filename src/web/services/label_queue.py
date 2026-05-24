"""Image queue for web labeling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


from src.modes.common import SUPPORTED_IMAGE_EXTENSIONS
from src.web.services.results_csv import read_results_csv

LabelFilter = str  # "unlabeled" | "low_confidence" | "all"


def list_image_files(images_dir: Path) -> list[str]:
    """Return sorted image filenames from the training folder."""
    if not images_dir.is_dir():
        return []

    names = [
        path.name
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return names


def load_labeled_names(csv_path: Path) -> set[str]:
    """Image names that already have a non-empty ground_truth in CSV."""
    df = read_results_csv(csv_path)
    if df.empty or "ground_truth" not in df.columns:
        return set()

    labeled: set[str] = set()
    for _, row in df.iterrows():
        ground_truth = str(row.get("ground_truth", "")).strip()
        image_name = str(row.get("image_name", "")).strip()
        if ground_truth and image_name:
            labeled.add(image_name)
    return labeled


def load_latest_predictions(csv_path: Path) -> dict[str, dict[str, object]]:
    """Latest CSV row per image_name (for low-confidence filter)."""
    df = read_results_csv(csv_path)
    if df.empty or "image_name" not in df.columns:
        return {}

    latest: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        image_name = str(row.get("image_name", "")).strip()
        if not image_name:
            continue
        latest[image_name] = row.to_dict()
    return latest


def build_queue(
    *,
    images_dir: Path,
    csv_path: Path,
    label_filter: LabelFilter = "unlabeled",
    low_confidence_threshold: float = 0.7,
) -> list[str]:
    """Build ordered list of image names for the labeling queue."""
    all_images = list_image_files(images_dir)
    if label_filter == "all":
        return all_images

    labeled = load_labeled_names(csv_path)
    predictions = load_latest_predictions(csv_path)

    if label_filter == "unlabeled":
        return [name for name in all_images if name not in labeled]

    if label_filter == "low_confidence":
        queue: list[str] = []
        for name in all_images:
            if name in labeled:
                continue
            row = predictions.get(name)
            if row is None:
                queue.append(name)
                continue
            try:
                confidence = float(row.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < low_confidence_threshold:
                queue.append(name)
        return queue

    return all_images


def queue_stats(
    *,
    images_dir: Path,
    csv_path: Path,
    label_filter: LabelFilter,
    low_confidence_threshold: float = 0.7,
) -> tuple[int, int, int]:
    """Return (total, labeled, remaining) for the current filter context."""
    all_images = list_image_files(images_dir)
    labeled = load_labeled_names(csv_path)
    queue = build_queue(
        images_dir=images_dir,
        csv_path=csv_path,
        label_filter=label_filter,
        low_confidence_threshold=low_confidence_threshold,
    )
    return len(all_images), len(labeled), len(queue)


def count_labels_by_slug(csv_path: Path) -> Counter[str]:
    """Count ground_truth labels mapped to YOLO slugs."""
    from training.build_dataset import MIN_CLASS_COUNTS, YOLO_CLASS_NAMES, category_to_slug

    counts: Counter[str] = Counter()
    df = read_results_csv(csv_path)
    if df.empty or "ground_truth" not in df.columns:
        return counts

    for _, row in df.iterrows():
        ground_truth = str(row.get("ground_truth", "")).strip()
        if not ground_truth:
            continue
        slug = category_to_slug(ground_truth)
        if slug and slug in YOLO_CLASS_NAMES:
            counts[slug] += 1

    # Ensure all keys exist for dashboard
    for slug in MIN_CLASS_COUNTS:
        counts.setdefault(slug, 0)
    return counts
