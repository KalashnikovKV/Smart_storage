"""Interactive ground-truth labeling mode."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from src.app import DataExporter, VideoProcessor, Visualizer, WindowClosed, prompt_ground_truth
from src.app.video_processor import SUPPORTED_VIDEO_EXTENSIONS, is_video_file
from src.modes.common import (
    build_pipeline,
    export_result,
    print_result,
    resolve_csv_path,
)

LOGGER = logging.getLogger(__name__)

_PREVIEW_HINT = "c=capture & label   space=pause   q=quit"
_PREVIEW_MAX_WIDTH = 1280


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


def _resize_for_preview(frame: np.ndarray, max_width: int = _PREVIEW_MAX_WIDTH) -> np.ndarray:
    """Downscale wide frames for responsive OpenCV preview (capture stays full-res)."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def _show_video_preview(
    visualizer: Visualizer,
    frame: np.ndarray,
    *,
    paused: bool,
    is_file: bool,
) -> None:
    """Fast raw-frame preview — no CV pipeline (pipeline runs only on capture)."""
    display = _resize_for_preview(frame)
    status = "PAUSED" if paused else ("FILE" if is_file else "LIVE")
    cv2.putText(
        display,
        f"{status}  |  {_PREVIEW_HINT}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    window_name = visualizer.WINDOW_NAME
    if not visualizer._window_created:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        visualizer._window_created = True
    cv2.imshow(window_name, display)


def run_label_image_mode(
    source: str,
    output_path: str | None = None,
    show_window: bool = True,
    model_path: str | None = None,
    device: str = "cpu",
) -> None:
    """Label a single image: pipeline prediction + interactive ground-truth prompt."""
    source_path = Path(source)
    csv_path = resolve_csv_path(output_path)

    LOGGER.debug(
        "Starting label image mode. source=%s csv=%s show_window=%s",
        source_path,
        csv_path,
        show_window,
    )

    pipeline = build_pipeline(model_path, device=device)
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
    pipeline,
    video: VideoProcessor,
    exporter: DataExporter,
    image_name: str,
    csv_path: str,
    show_window: bool = False,
    visualizer: Visualizer | None = None,
    frame: np.ndarray | None = None,
) -> bool:
    """Run pipeline on one frame, label it, and export."""
    if frame is None:
        frame = video.get_frame()
    if frame is None:
        print("Warning: Failed to capture frame.")
        return False

    capture = frame.copy()
    video.pause(capture)

    print("Running pipeline on captured frame...")
    result = pipeline.run(capture)
    if result is None:
        print("No object detected — adjust the scene and try again.")
        video.resume()
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
    finally:
        video.resume()

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
    csv_path = resolve_csv_path(output_path)
    pipeline = build_pipeline(model_path, device=device)
    visualizer = Visualizer()
    exporter = DataExporter(csv_path)
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
            return

        print("Preview: raw video only (pipeline runs when you press c).")
        if video.is_file_source:
            print(f"Playback ~{video.playback_fps:.0f} FPS — space to pause/resume.")

        while True:
            frame = video.get_frame()
            if frame is not None:
                last_frame = frame
                _show_video_preview(
                    visualizer,
                    frame,
                    paused=video.is_paused,
                    is_file=video.is_file_source,
                )

            key = cv2.waitKey(video.wait_delay_ms()) & 0xFF

            if key == ord("q"):
                break

            if visualizer._window_created and not visualizer.is_window_open():
                break

            if key == ord(" "):
                paused = video.toggle_pause(last_frame)
                print("Paused." if paused else "Resumed.")

            if key == ord("c") and last_frame is not None:
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
    csv_path = resolve_csv_path(output_path)
    video = VideoProcessor()

    if not video.start():
        LOGGER.error("Could not open webcam for label mode.")
        print("Error: Could not open webcam. Check camera connection.")
        print("Tip: Use --mode label --source <image or video path> for file-based labeling.")
        sys.exit(1)

    print("Smart Storage — Label Mode (webcam)")
    print(f"CSV: {csv_path}")
    if show_window:
        print(f"Controls: {_PREVIEW_HINT}")
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
    csv_path = resolve_csv_path(output_path)

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
        print(f"Controls: {_PREVIEW_HINT}")
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
