"""Batch folder processing mode."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

from src.app import DataExporter, Visualizer
from src.modes.common import (
    build_pipeline,
    collect_image_paths,
    export_result,
    image_key,
    save_pipeline_outputs,
    stage_output_stem,
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

    image_paths = collect_image_paths(source_dir)

    if not image_paths:
        LOGGER.error("No supported images found in directory: %s", source)
        print(f"No supported images found in '{source}' (searched subfolders recursively).")
        sys.exit(1)

    LOGGER.debug("Batch images found: %s", [image_key(p, source_dir) for p in image_paths])

    pipeline = build_pipeline(model_path, device=device)
    visualizer = Visualizer()
    exporter = DataExporter(output_path) if output_path else DataExporter()

    print(f"Smart Storage — Batch Mode: {source_dir}")
    print(f"Images found: {len(image_paths)}")
    print("-" * 50)

    processed = 0
    failed = 0

    for image_path in image_paths:
        key = image_key(image_path, source_dir)
        LOGGER.debug("Processing image: %s", key)

        image = cv2.imread(str(image_path))

        if image is None:
            LOGGER.warning("Cannot read image: %s", key)
            print(f"[FAILED] {key}: cannot read image.")
            failed += 1
            continue

        result = pipeline.run(image)

        if result is None:
            LOGGER.debug("No object detected in image: %s", key)
            print(f"[NO OBJECT] {key}")
            failed += 1
            continue

        export_result(
            exporter=exporter,
            result=result,
            image_name=key,
        )

        if save_outputs:
            save_pipeline_outputs(
                visualizer=visualizer,
                result=result,
                base_name=stage_output_stem(key),
            )

        categories = ", ".join(
            f"#{decision.object_id} {decision.category} "
            f"({decision.color}, {decision.size})"
            for decision in result.decisions
        )

        LOGGER.debug(
            "Image processed successfully: %s categories=%s",
            key,
            categories,
        )

        print(f"[OK] {key}: {categories}")
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
