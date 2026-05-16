"""Smart Storage — Object Sorting System.

Main application entry point with video, image and batch processing modes.

Usage:
    uv run python -m src.main --mode video

    uv run python -m src.main --mode video --debug

    uv run python -m src.main --mode image --source "training/test_images/Image.jpeg"

    uv run python -m src.main --mode image --source "training/test_images/Image.jpeg" --no-window

    uv run python -m src.main --mode image --source "training/test_images/Image.jpeg" --no-display

    uv run python -m src.main --mode image --source "training/test_images/Image.jpeg" --debug

    uv run python -m src.main --mode batch --source "training/test_images"

    uv run python -m src.main --mode batch --source "training/test_images" --no-display

    uv run python -m src.main --mode label

    uv run python -m src.main --mode label --source "training/test_images/Image.jpeg"

    uv run python -m src.main --mode label --source "training/test_video/Video_1.MOV"

    uv run python -m src.main --mode label --source "training/test_images/Image.jpeg" --no-display

    uv run python -m src.main --mode batch --source "training/test_images" --output "output/results.csv"

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
import numpy as np

from src.app import (
    DataExporter,
    VideoProcessor,
    Visualizer,
    WindowClosed,
    prompt_ground_truth,
)
from src.app.video_processor import SUPPORTED_VIDEO_EXTENSIONS, is_video_file
from src.pipeline import Pipeline


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CSV_PATH = "output/results.csv"

LOGGER = logging.getLogger(__name__)


def configure_logging(debug: bool) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _resolve_device(requested: str) -> str:
    """Return the effective device string, falling back to cpu if CUDA unavailable."""
    if requested in {"cpu", "mps"}:
        return requested
    # cuda / cuda:N — verify availability
    try:
        import torch
        if torch.cuda.is_available():
            return requested
        LOGGER.warning(
            "CUDA requested but torch.cuda.is_available() returned False. "
            "Falling back to cpu."
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


def _build_pipeline(model_path: str | None, device: str = "cpu") -> Pipeline:
    """Build Pipeline, optionally with a YOLO-Seg segmenter."""
    if model_path:
        from src.segment.yolo import YOLOSegmenter
        effective_device = _resolve_device(device)
        segmenter = YOLOSegmenter(model_path, device=effective_device)
        LOGGER.info("YOLO-Seg model loaded: %s (device=%s)", model_path, effective_device)
        print(f"YOLO-Seg model: {model_path} | device: {effective_device}")
        return Pipeline(segmenter=segmenter)
    return Pipeline()


def run_video_mode(
    output_path: str | None = None,
    model_path: str | None = None,
    device: str = "cpu",
) -> None:
    """Run the pipeline on live webcam video."""
    LOGGER.debug("Starting video mode. output_path=%s device=%s", output_path, device)

    pipeline = _build_pipeline(model_path, device=device)
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
    window_name = visualizer.WINDOW_NAME

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
                    if not visualizer._window_created:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        visualizer._window_created = True
                    cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            # Exit if user presses q or closes the window with the X button
            if key == ord("q"):
                LOGGER.debug("Quit key pressed.")
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                LOGGER.debug("Window closed by user.")
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
    model_path: str | None = None,
    device: str = "cpu",
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

    pipeline = _build_pipeline(model_path, device=device)
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
        cv2.waitKey(1)  # force initial render
        try:
            visualizer.wait_until_key_or_close(result)
        except WindowClosed:
            pass
        cv2.destroyAllWindows()


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

    pipeline = _build_pipeline(model_path, device=device)
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


def _resolve_csv_path(output_path: str | None) -> str:
    """Default CSV for all modes (predictions and ground_truth)."""
    return output_path or DEFAULT_CSV_PATH


def _label_and_export(
    exporter: DataExporter,
    result,
    image_name: str,
    show_window: bool = False,
    visualizer: Visualizer | None = None,
) -> str:
    """Prompt for ground truth, export labeled sample, return chosen class."""
    pump = None
    if show_window and visualizer is not None:
        pump = lambda: visualizer.pump_events(result)

    try:
        ground_truth = prompt_ground_truth(
            result.decision.category,
            result.decision.confidence,
            pump=pump,
        )
    except WindowClosed:
        if visualizer is not None:
            visualizer.close_window()
        raise

    export_result(
        exporter=exporter,
        result=result,
        image_name=image_name,
        ground_truth=ground_truth,
    )
    return ground_truth


def run_label_image_mode(
    source: str,
    output_path: str | None = None,
    show_window: bool = True,
    model_path: str | None = None,
    device: str = "cpu",
) -> None:
    """Label a single image: pipeline prediction + interactive ground-truth prompt."""
    source_path = Path(source)
    csv_path = _resolve_csv_path(output_path)

    LOGGER.debug(
        "Starting label image mode. source=%s csv=%s show_window=%s",
        source_path,
        csv_path,
        show_window,
    )

    pipeline = _build_pipeline(model_path, device=device)
    visualizer = Visualizer()
    exporter = DataExporter(csv_path)

    image = cv2.imread(str(source_path))
    if image is None:
        LOGGER.error("Could not read image: %s", source)
        print(f"Error: Could not read image '{source}'.")
        sys.exit(1)

    print(f"Smart Storage — Label Mode (image): {source_path.name}")
    print(f"CSV: {csv_path}")
    print("-" * 50)

    result = pipeline.run(image)
    if result is None:
        print("No object detected — nothing to label.")
        if show_window:
            no_det_window = "Smart Storage — No Detection"
            cv2.namedWindow(no_det_window, cv2.WINDOW_NORMAL)
            cv2.imshow(no_det_window, image)
            print("Press any key in the OpenCV window to close (or click X)...")
            try:
                Visualizer().wait_until_dismissed(no_det_window, image)
            except WindowClosed:
                pass
            cv2.destroyAllWindows()
        sys.exit(1)

    print_result(result)

    if show_window:
        visualizer.show_pipeline(result)
        cv2.waitKey(1)
        print("Review the prediction in the OpenCV window, then answer in the terminal.")

    try:
        ground_truth = _label_and_export(
            exporter,
            result,
            source_path.name,
            show_window=show_window,
            visualizer=visualizer,
        )
    except WindowClosed:
        cv2.destroyAllWindows()
        print("Окно закрыто — выход без сохранения.")
        return

    print(f"Saved ground_truth={ground_truth!r} → {csv_path}")
    if exporter.save_roi_images:
        print(f"ROI image → {exporter.images_dir}/")

    if show_window and visualizer.is_window_open():
        print("Press any key in the OpenCV window to close (or click X)...")
        try:
            visualizer.wait_until_key_or_close(result)
        except WindowClosed:
            pass
    cv2.destroyAllWindows()


def _capture_and_label_video_sample(
    pipeline: Pipeline,
    video: VideoProcessor,
    exporter: DataExporter,
    image_name: str,
    csv_path: str,
    show_window: bool = False,
    visualizer: Visualizer | None = None,
    frame: np.ndarray | None = None,
) -> bool:
    """Capture one video frame, label it, and export. Returns True if a sample was saved."""
    if frame is None:
        frame = video.get_frame()
    if frame is None:
        print("Warning: Failed to capture frame.")
        return False

    result = pipeline.run(frame)
    if result is None:
        print("No object detected — adjust the scene and try again.")
        return False

    print_result(result)

    if show_window and visualizer is not None:
        visualizer.show_pipeline(result)
        cv2.waitKey(1)

    try:
        ground_truth = _label_and_export(
            exporter,
            result,
            image_name,
            show_window=show_window,
            visualizer=visualizer,
        )
    except WindowClosed:
        if visualizer is not None:
            visualizer.close_window()
        raise

    print(f"Saved ground_truth={ground_truth!r} → {csv_path}")
    if exporter.save_roi_images:
        print(f"ROI image → {exporter.images_dir}/")
    return True


def _run_label_video_capture_loop(
    video: VideoProcessor,
    *,
    sample_prefix: str,
    output_path: str | None,
    show_window: bool,
    model_path: str | None,
    device: str,
) -> None:
    """Shared interactive loop for webcam and video-file labeling."""
    csv_path = _resolve_csv_path(output_path)
    pipeline = _build_pipeline(model_path, device=device)
    visualizer = Visualizer()
    exporter = DataExporter(csv_path)
    window_name = visualizer.WINDOW_NAME
    sample_index = 0
    last_frame: np.ndarray | None = None

    try:
        if not show_window:
            while True:
                line = input("> ").strip().lower()
                if line in {"q", "quit"}:
                    break
                if line not in {"c", "capture"}:
                    continue
                if _capture_and_label_video_sample(
                    pipeline,
                    video,
                    exporter,
                    f"{sample_prefix}_{sample_index:04d}",
                    csv_path,
                    show_window=False,
                    visualizer=None,
                ):
                    sample_index += 1
        else:
            while True:
                frame = video.get_frame()
                if frame is not None:
                    last_frame = frame
                    result = pipeline.run(frame)
                    if result is not None:
                        visualizer.show_pipeline(result)
                    else:
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
                        if not visualizer._window_created:
                            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                            visualizer._window_created = True
                        cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if visualizer._window_created and not visualizer.is_window_open():
                    break

                if key == ord("c"):
                    try:
                        if _capture_and_label_video_sample(
                            pipeline,
                            video,
                            exporter,
                            f"{sample_prefix}_{sample_index:04d}",
                            csv_path,
                            show_window=True,
                            visualizer=visualizer,
                            frame=last_frame,
                        ):
                            sample_index += 1
                            print("Ready for next capture (press 'c').")
                    except WindowClosed:
                        break

    except WindowClosed:
        pass
    except KeyboardInterrupt:
        LOGGER.debug("Label video capture loop interrupted.")


def run_label_video_mode(
    output_path: str | None = None,
    show_window: bool = True,
    model_path: str | None = None,
    device: str = "cpu",
) -> None:
    """Label from webcam: capture with 'c', confirm or correct class, save ROI + CSV."""
    csv_path = _resolve_csv_path(output_path)
    video = VideoProcessor()

    if not video.start():
        LOGGER.error("Could not open webcam for label mode.")
        print("Error: Could not open webcam. Check camera connection.")
        print("Tip: Use --mode label --source <image or video path> for file-based labeling.")
        sys.exit(1)

    print("Smart Storage — Label Mode (webcam)")
    print(f"CSV: {csv_path}")
    if show_window:
        print("Controls: c=capture & label, q=quit")
    else:
        print("Headless: type c + Enter to capture & label, q + Enter to quit")
    print("-" * 50)

    try:
        _run_label_video_capture_loop(
            video,
            sample_prefix="webcam",
            output_path=output_path,
            show_window=show_window,
            model_path=model_path,
            device=device,
        )
    finally:
        video.stop()
        if show_window:
            cv2.destroyAllWindows()
        print("Label mode stopped.")


def run_label_video_file_mode(
    source: str,
    output_path: str | None = None,
    show_window: bool = True,
    model_path: str | None = None,
    device: str = "cpu",
) -> None:
    """Label from a video file: preview playback, c=capture frame and label."""
    source_path = Path(source)
    csv_path = _resolve_csv_path(output_path)

    if not source_path.is_file():
        LOGGER.error("Video file not found: %s", source)
        print(f"Error: Video file not found: '{source}'.")
        sys.exit(1)

    if source_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        print(f"Error: Unsupported video format '{source_path.suffix}'.")
        print(f"Supported: {supported}")
        sys.exit(1)

    video = VideoProcessor(source=str(source_path))
    if not video.start():
        LOGGER.error("Could not open video file: %s", source)
        print(f"Error: Could not open video '{source}'.")
        sys.exit(1)

    print(f"Smart Storage — Label Mode (video): {source_path.name}")
    print(f"CSV: {csv_path}")
    if show_window:
        print("Controls: c=capture & label current frame, q=quit")
    else:
        print("Headless: type c + Enter to capture & label, q + Enter to quit")
    print("-" * 50)

    try:
        _run_label_video_capture_loop(
            video,
            sample_prefix=source_path.stem,
            output_path=output_path,
            show_window=show_window,
            model_path=model_path,
            device=device,
        )
    finally:
        video.stop()
        if show_window:
            cv2.destroyAllWindows()
        print("Label mode stopped.")


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
        choices=["video", "image", "batch", "label"],
        default="video",
        help="Processing mode: video, image, batch, or label (dataset collection).",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Path for --mode: image/label (file), batch (folder), "
            "label (image or video: .mov, .mp4, …)."
        ),
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
        "--model",
        type=str,
        default=None,
        help=(
            "Path to YOLO-Seg weights (.pt). "
            "When provided, YOLO-Seg is used for detection and segmentation "
            "instead of the built-in threshold pipeline. "
            "Example: --model models/yolo_segmentation_best.pt"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Inference device: 'cpu', 'cuda', 'cuda:0', 'mps'. "
            "When omitted, auto-detects CUDA and falls back to cpu if unavailable. "
            "Only used when --model is provided."
        ),
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

    # Resolve device: explicit arg > auto-detect CUDA > cpu
    if args.device is not None:
        effective_device = args.device
    elif args.model is None:
        effective_device = "cpu"
    else:
        try:
            import torch
            effective_device = "cuda" if torch.cuda.is_available() else "cpu"
        except (ImportError, OSError):
            effective_device = "cpu"

    LOGGER.debug("Effective inference device: %s", effective_device)

    if args.mode == "video":
        run_video_mode(args.output, model_path=args.model, device=effective_device)

    elif args.mode == "image":
        run_image_mode(
            source=args.source,
            output_path=args.output,
            save_outputs=not args.no_save_images,
            show_window=should_show_window,
            model_path=args.model,
            device=effective_device,
        )

    elif args.mode == "batch":
        run_batch_mode(
            source=args.source,
            output_path=args.output,
            save_outputs=not args.no_save_images,
            model_path=args.model,
            device=effective_device,
        )

    elif args.mode == "label":
        if args.source is None:
            run_label_video_mode(
                output_path=args.output,
                show_window=should_show_window,
                model_path=args.model,
                device=effective_device,
            )
        elif is_video_file(args.source):
            run_label_video_file_mode(
                source=args.source,
                output_path=args.output,
                show_window=should_show_window,
                model_path=args.model,
                device=effective_device,
            )
        else:
            run_label_image_mode(
                source=args.source,
                output_path=args.output,
                show_window=should_show_window,
                model_path=args.model,
                device=effective_device,
            )


if __name__ == "__main__":
    main()