"""Shared helpers for CLI processing modes."""

from __future__ import annotations

import logging
from pathlib import Path

from src.app.data_exporter import DataExporter
from src.config import AppConfig
from src.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CSV_PATH = "output/results.csv"


def collect_image_paths(root: Path) -> list[Path]:
    """Return supported image files under *root* (file or directory, recursive)."""
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else []

    if not root.is_dir():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def image_key(image_path: Path, images_root: Path) -> str:
    """Relative posix path used in CSV exports."""
    return image_path.resolve().relative_to(images_root.resolve()).as_posix()


def stage_output_stem(image_key: str) -> str:
    """Filesystem-safe stem for output/stages/* artifacts."""
    parts = Path(image_key.replace("\\", "/")).parts
    if len(parts) == 1:
        return Path(parts[0]).stem
    *folders, file_name = parts
    return "__".join([*folders, Path(file_name).stem])


def resolve_image_path(images_dir: Path, image_key: str) -> Path:
    """Resolve a user-facing image key under *images_dir* (blocks path traversal)."""
    normalized = Path(image_key.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or image_key.startswith("."):
        raise ValueError(f"Invalid image path: {image_key}")

    resolved = (images_dir / normalized).resolve()
    root = images_dir.resolve()
    if not str(resolved).startswith(str(root)):
        raise ValueError(f"Invalid image path: {image_key}")
    if not resolved.is_file():
        raise FileNotFoundError(image_key)
    return resolved


def configure_logging(debug: bool) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def resolve_device(requested: str) -> str:
    """Return the effective device string, falling back to cpu if CUDA unavailable."""
    if requested in {"cpu", "mps"}:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return requested
        LOGGER.warning(
            "CUDA requested but torch.cuda.is_available() returned False. "
            "Falling back to cpu.",
        )
        print("Warning: CUDA not available — falling back to cpu.")
    except (ImportError, OSError) as exc:
        LOGGER.warning(
            "torch unavailable (%s); cannot verify CUDA. Falling back to cpu.",
            exc,
        )
        print(
            "Warning: PyTorch failed to load — falling back to cpu. "
            "If you need YOLO, run: uv sync --group yolo  "
            "(installs CPU-only torch; see build-instructions.md for CUDA)."
        )
    return "cpu"


def build_pipeline(
    model_path: str | None,
    device: str = "cpu",
    max_objects: int | None = None,
    *,
    fast: bool = False,
    yolo_conf: float | None = None,
    yolo_imgsz: int | None = None,
    yolo_prefer: list[str] | None = None,
    yolo_prefer_strict: bool = False,
) -> Pipeline:
    """Build Pipeline, optionally with a YOLO-Seg segmenter."""
    config = AppConfig()
    if max_objects is not None:
        config.max_objects = max_objects

    if fast:
        config.kmeans_max_iter = 25
        config.kmeans_n_init = 2
        config.kmeans_max_pixels = 1500
        config.pre_blur_kernel = (7, 7)
        config.adaptive_block_size = 21

    if model_path:
        from src.ml.yolo_segmenter import YOLOSegmenter

        effective_device = resolve_device(device)
        segmenter = YOLOSegmenter(
            model_path,
            device=effective_device,
            conf=yolo_conf if yolo_conf is not None else config.yolo_conf_threshold,
            imgsz=yolo_imgsz if yolo_imgsz is not None else 640,
            max_objects=config.max_objects,
            preferred_classes=yolo_prefer,
            strict_preferred=yolo_prefer_strict,
        )
        LOGGER.info("YOLO-Seg model loaded: %s (device=%s)", model_path, effective_device)
        prefer_msg = (
            f" | prefer: {','.join(yolo_prefer)}"
            + (" (strict)" if yolo_prefer_strict else "")
            if yolo_prefer
            else ""
        )
        print(
            f"YOLO-Seg model: {model_path} | device: {effective_device} | "
            f"conf: {segmenter.conf} | imgsz: {segmenter.imgsz}{prefer_msg}",
        )
        return Pipeline(config=config, segmenter=segmenter)
    return Pipeline(config=config)


def resolve_csv_path(output_path: str | None) -> str:
    """Default CSV for all modes (predictions and ground_truth)."""
    return output_path or DEFAULT_CSV_PATH


def export_result(
    exporter: DataExporter,
    result,
    image_name: str,
    ground_truth: str = "",
) -> None:
    """Export pipeline result to CSV with ROI images."""
    LOGGER.debug(
        "Exporting result for image_name=%s ground_truth=%s",
        image_name,
        ground_truth or "(empty)",
    )

    if hasattr(exporter, "export_many"):
        exporter.export_many(
            result.decisions,
            result.detections,
            image_name=image_name,
            original_image=result.original,
            ground_truth=ground_truth,
        )
        return

    roi = None
    if exporter.save_roi_images and result.original is not None:
        roi = exporter._extract_roi(result.original, result.detection.bbox)

    exporter.export(
        result.decision,
        result.detection,
        image_name=image_name,
        roi=roi,
        ground_truth=ground_truth,
    )


def save_pipeline_outputs(visualizer, result, base_name: str) -> None:
    """Save pipeline stage images if the visualizer supports this method."""
    LOGGER.debug("Saving pipeline outputs for base_name=%s", base_name)

    if hasattr(visualizer, "save_pipeline_outputs"):
        visualizer.save_pipeline_outputs(result, base_name=base_name)
        print("Pipeline stage images saved to output/stages/")
    else:
        LOGGER.warning("Visualizer has no save_pipeline_outputs() method.")
        print("Stage image saving skipped: Visualizer has no save_pipeline_outputs().")


def print_result(result) -> None:
    """Print pipeline result to console."""
    print(f"Objects detected: {len(result.decisions)}")
    print(f"Processing time: {result.processing_time_ms:.0f}ms")

    for decision in result.decisions:
        print("-" * 30)
        print(f"Object #{decision.object_id}")
        print(f"Category: {decision.category}")
        print(f"Confidence: {decision.confidence:.0%}")
        print(f"Color: {decision.color}")
        print(f"Size: {decision.size}")
        print(f"Method: {decision.method_used}")

        if decision.is_unknown:
            print(f"Closest match: {decision.closest_match}")
