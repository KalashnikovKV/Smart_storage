#!/usr/bin/env python3
"""Train or validate a YOLO-Seg model on a dataset built by build_dataset.py.

Usage:
    uv run python training/train_yolo.py --validate-only --data dataset/dataset.yaml
    uv sync --group yolo
    uv run python training/train_yolo.py --data dataset/dataset.yaml --model yolo11n-seg.pt --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_yaml(data_yaml: Path) -> dict:
    with data_yaml.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_dataset_root(data_yaml: Path, cfg: dict) -> Path:
    root = cfg.get("path", "")
    if not root:
        return data_yaml.parent
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = (data_yaml.parent / root_path).resolve()
    return root_path


def validate_dataset(data_yaml: Path) -> tuple[int, int]:
    """Verify dataset layout and YOLO-Seg label format. Returns (image_count, label_count)."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    cfg = _load_yaml(data_yaml)
    root = _resolve_dataset_root(data_yaml, cfg)

    train_rel = cfg.get("train", "images/train")
    val_rel = cfg.get("val", "images/val")
    names = cfg.get("names", {})

    if not names:
        raise ValueError("dataset.yaml must define 'names' with at least one class.")

    image_count = 0
    label_count = 0
    train_count = 0
    errors: list[str] = []

    for split_rel, required in ((train_rel, True), (val_rel, False)):
        images_dir = root / split_rel
        labels_dir = root / "labels" / Path(split_rel).name

        if not images_dir.is_dir():
            if required:
                errors.append(f"missing images directory: {images_dir}")
            continue

        split_images = [
            path
            for path in sorted(images_dir.iterdir())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        if required and not split_images:
            errors.append(f"no images in required split: {images_dir}")

        for image_path in split_images:
            image_count += 1
            if required:
                train_count += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"missing label for {image_path.name}")
                continue

            label_count += 1
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 7 or len(parts) % 2 == 0:
                    errors.append(
                        f"{label_path.name}:{line_no} invalid YOLO-Seg line "
                        f"(need class_id + even number of coords, got {len(parts)} tokens)",
                    )
                    continue

                class_id = int(parts[0])
                if class_id < 0 or class_id >= len(names):
                    errors.append(
                        f"{label_path.name}:{line_no} class_id {class_id} out of range "
                        f"(names has {len(names)} classes)",
                    )

                coords = list(map(float, parts[1:]))
                if any(not (0.0 <= value <= 1.0) for value in coords):
                    errors.append(
                        f"{label_path.name}:{line_no} coordinates must be normalized to [0, 1]",
                    )

    if train_count == 0:
        errors.append("no images found in train split")

    if errors:
        detail = "\n".join(f"  - {item}" for item in errors[:20])
        raise ValueError(f"Dataset validation failed:\n{detail}")

    return image_count, label_count


def train(
    data_yaml: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
) -> Path:
    """Run Ultralytics YOLO-Seg training."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run: uv sync --group yolo",
        ) from exc

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        task="segment",
    )
    return Path(results.save_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or validate YOLO-Seg on a local dataset.")
    parser.add_argument("--data", default="dataset/dataset.yaml", help="Path to dataset.yaml.")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Base YOLO-Seg weights.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, cuda:0, mps.")
    parser.add_argument("--project", default="runs/segment", help="Ultralytics project directory.")
    parser.add_argument("--name", default="smart_storage", help="Run name inside project.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate dataset.yaml and label files (no training).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_yaml = Path(args.data)

    try:
        image_count, label_count = validate_dataset(data_yaml)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    cfg = _load_yaml(data_yaml)
    names = cfg.get("names", {})
    print("Smart Storage — YOLO-Seg Trainer")
    print("=" * 50)
    print(f"dataset: {data_yaml.resolve()}")
    print(f"classes ({len(names)}): {', '.join(str(names[k]) for k in sorted(names, key=int))}")
    print(f"images: {image_count}, labels: {label_count}")

    if args.validate_only:
        print("\nDataset validation passed.")
        return 0

    try:
        save_dir = train(
            data_yaml=data_yaml,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
    except ImportError as exc:
        print(exc)
        return 1

    print(f"\nTraining complete. Artifacts: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
