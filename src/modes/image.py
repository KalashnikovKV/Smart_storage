"""Single-image processing mode."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from src.app import DataExporter, Visualizer, WindowClosed
from src.modes.common import (
    build_pipeline,
    export_result,
    print_result,
    save_pipeline_outputs,
)

LOGGER = logging.getLogger(__name__)
CONTOUR_DEBUG_DIR = Path("output/stages/contours")


def _save_contour_debug(
    pipeline,
    image: np.ndarray,
    base_name: str,
) -> Path:
    """Run enhance/segment/clean and save numbered contour overlay."""
    CONTOUR_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    enhanced = pipeline.enhance(image)
    mask = pipeline.segment(enhanced)
    cleaned = pipeline.clean(mask)

    overlay = pipeline.debug_contours_overlay(image, cleaned)
    output_path = CONTOUR_DEBUG_DIR / f"{base_name}_contours.jpg"
    cv2.imwrite(str(output_path), overlay)

    contours = pipeline.mask_ops.find_object_contours(
        np.where(cleaned > 0, 255, 0).astype(np.uint8),
        max_objects=pipeline.config.max_objects,
    )
    print(f"Debug contours: {len(contours)} object(s) — saved to {output_path}")
    return output_path


def run_image_mode(
    source: str,
    output_path: str | None = None,
    save_outputs: bool = True,
    show_window: bool = True,
    model_path: str | None = None,
    device: str = "cpu",
    debug_contours: bool = False,
    max_objects: int | None = None,
) -> None:
    """Run the pipeline on a single image file."""
    source_path = Path(source)

    LOGGER.debug(
        "Starting image mode. source=%s output_path=%s save_outputs=%s show_window=%s device=%s",
        source_path,
        output_path,
        save_outputs,
        show_window,
        device,
    )

    pipeline = build_pipeline(model_path, device=device, max_objects=max_objects)
    visualizer = Visualizer()
    exporter = DataExporter(output_path) if output_path else DataExporter()

    image = cv2.imread(str(source_path))

    if image is None:
        LOGGER.error("Could not read image: %s", source)
        print(f"Error: Could not read image '{source}'.")
        sys.exit(1)

    LOGGER.debug("Image loaded successfully. shape=%s", image.shape)

    print(f"Smart Storage — Image Mode: {source_path.name}")
    print("-" * 50)

    if debug_contours and model_path is None:
        _save_contour_debug(pipeline, image, source_path.stem)

    result = pipeline.run(image)

    if result is None:
        LOGGER.debug("No object detected in image: %s", source_path.name)
        print("No object detected in the image.")

        if show_window:
            no_det_window = "Smart Storage — No Detection"
            cv2.namedWindow(no_det_window, cv2.WINDOW_NORMAL)
            cv2.imshow(no_det_window, image)
            print("\nPress any key in the OpenCV window to close (or click X)...")
            try:
                Visualizer().wait_until_dismissed(no_det_window, image)
            except WindowClosed:
                pass
            cv2.destroyAllWindows()

        return

    LOGGER.debug(
        "Image processed. objects=%s category=%s confidence=%.3f",
        len(result.decisions),
        result.decision.category,
        result.decision.confidence,
    )

    print_result(result)

    if save_outputs:
        save_pipeline_outputs(
            visualizer=visualizer,
            result=result,
            base_name=source_path.stem,
        )

    export_result(
        exporter=exporter,
        result=result,
        image_name=source_path.name,
    )
    LOGGER.debug("Image result exported to CSV: %s", exporter.output_path)
    print(f"\nResult saved to CSV: {exporter.output_path}")

    if show_window:
        visualizer.show_pipeline(result)
        print("\nPress any key in the OpenCV window to close (or click X)...")
        cv2.waitKey(1)
        try:
            visualizer.wait_until_key_or_close(result)
        except WindowClosed:
            pass
        cv2.destroyAllWindows()
