"""Smart Storage — Object Sorting System.

Main application entry point with video, image and batch processing modes.

Usage:
    uv run python -m src.main --mode video

    uv run python -m src.main --mode video --debug

    uv run python -m src.main --mode image --source "test_images/Image.jpeg"

    uv run python -m src.main --mode image --source "test_images/Image.jpeg" --no-window

    uv run python -m src.main --mode image --source "test_images/Image.jpeg" --no-display

    uv run python -m src.main --mode image --source "test_images/Image.jpeg" --debug

    uv run python -m src.main --mode batch --source "test_images"

    uv run python -m src.main --mode batch --source "test_images" --no-display

    uv run python -m src.main --mode batch --source "test_images" --output "output/results.csv"

Controls in video mode:
    q - quit
    s - save current result to CSV
    c - capture and freeze current frame
    p - pause/resume video
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from src.data_exporter import DataExporter
from src.pipeline import Pipeline
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LOGGER = logging.getLogger(__name__)


def configure_logging(debug: bool) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def run_video_mode(output_path: str | None = None) -> None:
    """Run the pipeline on live webcam video."""
    LOGGER.debug("Starting video mode. output_path=%s", output_path)

    pipeline = Pipeline()
    visualizer = Visualizer()
    video = VideoProcessor()
    exporter = DataExporter(output_path) if output_path else DataExporter()

    if not video.start():
        LOGGER.error("Could not open webcam.")
        print("Error: Could not open webcam. Check camera connection.")
        print("Tip: Try --mode image --source <path> for file processing.")
        sys.exit(1)

    print("Smart Storage — Video Mode")
    print("Controls: q=quit, s=save to CSV, c=capture, p=pause")
    print("-" * 50)

    paused = False
    last_result = None

    try:
        while True:
            if not paused:
                frame = video.get_frame()

                if frame is None:
                    LOGGER.warning("Failed to capture frame.")
                    print("Warning: Failed to capture frame.")
                    continue

                LOGGER.debug("Captured video frame with shape=%s", frame.shape)

                result = pipeline.run(frame)

                if result is not None:
                    last_result = result
                    LOGGER.debug(
                        "Video frame processed. Category=%s confidence=%.3f",
                        result.decision.category,
                        result.decision.confidence,
                    )
                    visualizer.show_pipeline(result)
                else:
                    LOGGER.debug("No object detected in current video frame.")
                    display = frame.copy()
                    cv2.putText(
                        display,
                        "No object detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Smart Storage — Pipeline Dashboard", display)  # noqa: E501

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                LOGGER.debug("Quit key pressed.")
                break

            if key == ord("s") and last_result is not None:
                export_result(
                    exporter=exporter,
                    result=last_result,
                    image_name="webcam_frame",
                )

                LOGGER.debug("Saved current video result to CSV.")

                print(
                    f"Saved: {last_result.decision.category} "
                    f"({last_result.decision.confidence:.0%})"
                )

            if key == ord("c"):
                paused = True
                LOGGER.debug("Video frame captured/frozen.")
                print("Frame captured. Press 'p' to resume.")

            if key == ord("p"):
                paused = not paused
                state = "Paused" if paused else "Resumed"
                LOGGER.debug("Video state changed: %s", state)
                print(f"Video {state}")

    except KeyboardInterrupt:
        LOGGER.debug("Video mode interrupted by user.")

    finally:
        video.stop()
        cv2.destroyAllWindows()
        LOGGER.debug("Video mode stopped.")
        print("Smart Storage stopped.")


def run_image_mode(
    source: str,
    output_path: str | None = None,
    save_outputs: bool = True,
    show_window: bool = True,
) -> None:
    """Run the pipeline on a single image file."""
    source_path = Path(source)

    LOGGER.debug(
        "Starting image mode. source=%s output_path=%s save_outputs=%s show_window=%s",
        source_path,
        output_path,
        save_outputs,
        show_window,
    )

    pipeline = Pipeline()
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

    result = pipeline.run(image)

    if result is None:
        LOGGER.debug("No object detected in image: %s", source_path.name)
        print("No object detected in the image.")

        if show_window:
            cv2.namedWindow("Smart Storage — No Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Smart Storage — No Detection", image)
            print("\nPress any key in the OpenCV window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return

    LOGGER.debug(
        "Image processed. objects=%s category=%s confidence=%.3f",
        len(result.decisions),
        result.decision.category,
        result.decision.confidence,
    )

    print_result(result)

    export_result(
        exporter=exporter,
        result=result,
        image_name=source_path.name,
    )
    LOGGER.debug("Image result exported to CSV: %s", exporter.output_path)
    print(f"\nResult saved to CSV: {exporter.output_path}")

    if save_outputs:
        save_pipeline_outputs(
            visualizer=visualizer,
            result=result,
            base_name=source_path.stem,
        )

    if show_window:
        visualizer.show_pipeline(result)
        print("\nPress any key in the OpenCV window to close...")
        cv2.waitKey(1)   # force initial render
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_batch_mode(
    source: str,
    output_path: str | None = None,
    save_outputs: bool = True,
) -> None:
    """Run the pipeline on all supported images in a folder."""
    source_dir = Path(source)

    LOGGER.debug(
        "Starting batch mode. source=%s output_path=%s save_outputs=%s",
        source_dir,
        output_path,
        save_outputs,
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

    pipeline = Pipeline()
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


def export_result(
    exporter: DataExporter,
    result,
    image_name: str,
) -> None:
    """Export pipeline result to CSV.

    Supports both exporter versions:
    - export(decision, detection)
    - export_many(decisions, detections, image_name)
    """
    LOGGER.debug("Exporting result for image_name=%s", image_name)

    if hasattr(exporter, "export_many"):
        exporter.export_many(
            result.decisions,
            result.detections,
            image_name=image_name,
        )
        return

    exporter.export(result.decision, result.detection)


def save_pipeline_outputs(
    visualizer: Visualizer,
    result,
    base_name: str,
) -> None:
    """Save pipeline stage images if the visualizer supports this method."""
    LOGGER.debug("Saving pipeline outputs for base_name=%s", base_name)

    if hasattr(visualizer, "save_pipeline_outputs"):
        visualizer.save_pipeline_outputs(
            result,
            base_name=base_name,
        )
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


def main() -> None:
    """Parse CLI arguments and run the selected mode."""
    parser = argparse.ArgumentParser(
        description="Smart Storage — Object Sorting System"
    )

    parser.add_argument(
        "--mode",
        choices=["video", "image", "batch"],
        default="video",
        help="Processing mode: video, image or batch.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image path for image mode or folder path for batch mode.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path. Default: output/results.csv.",
    )

    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Do not save pipeline stage output images.",
    )

    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open OpenCV window in image mode.",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open OpenCV window in image mode.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    configure_logging(args.debug)
    LOGGER.debug("Parsed arguments: %s", args)

    if args.mode in {"image", "batch"} and args.source is None:
        parser.error("--source is required for image and batch modes")

    should_show_window = not (args.no_window or args.no_display)

    if args.mode == "video":
        run_video_mode(args.output)

    elif args.mode == "image":
        run_image_mode(
            source=args.source,
            output_path=args.output,
            save_outputs=not args.no_save_images,
            show_window=should_show_window,
        )

    elif args.mode == "batch":
        run_batch_mode(
            source=args.source,
            output_path=args.output,
            save_outputs=not args.no_save_images,
        )


if __name__ == "__main__":
    main()