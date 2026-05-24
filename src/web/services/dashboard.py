"""Dashboard statistics aggregation."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.web.schemas.dashboard import ClassProgress, DashboardStats
from src.web.services.label_queue import (
    count_labels_by_slug,
    list_image_files,
    load_labeled_names,
)
from src.web.services.results_csv import read_results_csv
from training.build_dataset import MIN_CLASS_COUNTS


def _count_dataset_split(dataset_yaml: Path, split: str) -> int:
    if not dataset_yaml.is_file():
        return 0

    with dataset_yaml.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    root = dataset_yaml.parent
    rel = cfg.get(split, f"images/{split}")
    images_dir = root / rel
    if not images_dir.is_dir():
        return 0

    return sum(1 for path in images_dir.iterdir() if path.is_file())


def build_dashboard_stats(
    *,
    images_dir: Path,
    csv_path: Path,
    dataset_yaml: Path,
) -> DashboardStats:
    """Aggregate stats for the ML Studio overview page."""
    all_images = list_image_files(images_dir)
    labeled = load_labeled_names(csv_path)
    slug_counts = count_labels_by_slug(csv_path)

    csv_rows = len(read_results_csv(csv_path))

    class_progress = [
        ClassProgress(
            slug=slug,
            have=slug_counts.get(slug, 0),
            need=need,
            ok=slug_counts.get(slug, 0) >= need,
        )
        for slug, need in MIN_CLASS_COUNTS.items()
    ]

    dataset_train = _count_dataset_split(dataset_yaml, "train")
    dataset_val = _count_dataset_split(dataset_yaml, "val")

    checklist = {
        "has_images": len(all_images) > 0,
        "has_labels": len(labeled) > 0,
        "has_dataset": dataset_train > 0,
        "min_targets_met": all(item.ok for item in class_progress),
    }

    return DashboardStats(
        total_images=len(all_images),
        labeled_images=len(labeled),
        csv_rows=csv_rows,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        class_progress=class_progress,
        checklist=checklist,
    )
