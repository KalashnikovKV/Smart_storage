"""Batch folder processing mode."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

from src.app import DataExporter, Visualizer
from src.modes.common import (
    SUPPORTED_IMAGE_EXTENSIONS,
    build_pipeline,
    export_result,
    save_pipeline_outputs,
)

LOGGER = logging.getLogger(__name__)


def run_batch_mode(
    source: str,
    output_path: str | None = None,
    save_outputs: bool = True,
    model_path: str | None = None,
    device: str = "cpu",
) -> None:
    """Run the pipeline on all supported images in a folder."""
    source_dir = Path(source)

    LOGGER.debug(
        "Starting batch mode. source=%s output_path=%s save_outputs=%s device=%s",
        source_dir,
        output_path,
        save_outputs,
        device,
    )

    if not source_dir.exists() or not source_dir.is_dir():
        LOGGER.error("Invalid batch source directory: %s", source)
        print(f"Error: '{source}' is not a valid directory.")
        sys.exit(1)

    image_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    if not image_paths:
        LOGGER.error("No supported images found in directory: %s", source)
        print(f"No supported images found in '{source}'.")
        sys.exit(1)

    LOGGER.debug("Batch images found: %s", [path.name for path in image_paths])

    pipeline = build_pipeline(model_path, device=device)
    visualizer = Visualizer()
    exporter = DataExporter(output_path) if output_path else DataExporter()

    print(f"Smart Storage — Batch Mode: {source_dir}")
    print(f"Images found: {len(image_paths)}")
    print("-" * 50)

    processed = 0
    failed = 0

    for image_path in image_paths:
        LOGGER.debug("Processing image: %s", image_path.name)

        image = cv2.imread(str(image_path))

        if image is None:
            LOGGER.warning("Cannot read image: %s", image_path.name)
            print(f"[FAILED] {image_path.name}: cannot read image.")
            failed += 1
            continue

        result = pipeline.run(image)

        if result is None:
            LOGGER.debug("No object detected in image: %s", image_path.name)
            print(f"[NO OBJECT] {image_path.name}")
            failed += 1
            continue

        export_result(
            exporter=exporter,
            result=result,
            image_name=image_path.name,
        )

        if save_outputs:
            save_pipeline_outputs(
                visualizer=visualizer,
                result=result,
                base_name=image_path.stem,
            )

        categories = ", ".join(
            f"#{decision.object_id} {decision.category} "
            f"({decision.color}, {decision.size})"
            for decision in result.decisions
        )

        LOGGER.debug(
            "Image processed successfully: %s categories=%s",
            image_path.name,
            categories,
        )

        print(f"[OK] {image_path.name}: {categories}")
        processed += 1

    print("-" * 50)
    print(f"Processed: {processed}")
    print(f"Failed / no object: {failed}")
    print(f"CSV saved to: {exporter.output_path}")

    LOGGER.debug(
        "Batch mode finished. processed=%s failed=%s csv=%s",
        processed,
        failed,
        exporter.output_path,
    )

    if save_outputs:
        print("Pipeline stage images saved to output/stages/")
