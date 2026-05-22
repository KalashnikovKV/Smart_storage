"""Live webcam and video-file processing mode."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

from src.app import DataExporter, VideoProcessor, Visualizer
from src.modes.common import build_pipeline, export_result, print_result

LOGGER = logging.getLogger(__name__)

VIDEO_WINDOW_NAME = "Smart Storage — Video + Masks"
_FILE_HINT = "space=pause   a=analyze (mask panel)   s=save CSV   q=quit"
_WEBCAM_HINT = "q=quit   s=save to CSV   c=capture   p=pause"


def _ensure_window(visualizer: Visualizer, window_name: str) -> None:
    if not visualizer._window_created:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        visualizer._window_created = True


def _show_side_by_side(
    visualizer: Visualizer,
    frame,
    mask_panel,
    window_name: str,
    *,
    status_line: str,
    right_title: str = "Object Masks",
) -> None:
    display = visualizer.create_video_side_by_side(
        frame,
        mask_panel,
        left_title="Video",
        right_title=right_title,
        status_line=status_line,
    )
    _ensure_window(visualizer, window_name)
    cv2.imshow(window_name, display)


def _build_mask_panel(visualizer: Visualizer, pipeline, result):
    object_masks = pipeline.get_object_masks(result.cleaned_mask, result.original)
    return visualizer.build_colored_object_masks(
        object_masks,
        detections=result.detections,
        decisions=result.decisions,
    )


def run_video_mode(
    output_path: str | None = None,
    model_path: str | None = None,
    device: str = "cpu",
    source: str | None = None,
    max_objects: int | None = None,
) -> None:
    """Run the pipeline on a webcam or a video file."""
    LOGGER.debug(
        "Starting video mode. source=%s output_path=%s device=%s",
        source,
        output_path,
        device,
    )

    pipeline = build_pipeline(model_path, device=device, max_objects=max_objects)
    visualizer = Visualizer()
    video = VideoProcessor(source=source)
    exporter = DataExporter(output_path) if output_path else DataExporter()

    if not video.start():
        if source:
            LOGGER.error("Could not open video source: %s", source)
            print(f"Error: Could not open video '{source}'.")
        else:
            LOGGER.error("Could not open webcam.")
            print("Error: Could not open webcam. Check camera connection.")
            print("Tip: Use --mode video --source training/test_video/Video_6.MOV")
        sys.exit(1)

    is_file = video.is_file_source
    source_name = Path(source).name if source else "webcam"
    window_name = VIDEO_WINDOW_NAME

    print("Smart Storage — Video Mode")
    if is_file:
        print(f"Source: {source_name}")
        print("Layout: video (left) + object masks (right)")
        print(f"Controls: {_FILE_HINT}")
        print("Note: analysis takes ~5–15 s per frame — press 'a' on a paused frame.")
    else:
        print(f"Controls: {_WEBCAM_HINT}")
    print("-" * 50)

    paused = False
    last_result = None
    current_frame = None
    current_mask_panel = None

    try:
        while True:
            if not paused:
                frame = video.get_frame()

                if frame is None:
                    LOGGER.warning("Failed to capture frame.")
                    if is_file:
                        print("End of video (will loop).")
                    else:
                        print("Warning: Failed to capture frame.")
                    continue

                current_frame = frame

                if is_file:
                    height, width = frame.shape[:2]
                    placeholder = visualizer.create_mask_placeholder(height, width)
                    status = "PLAYING" if not paused else "PAUSED"
                    _show_side_by_side(
                        visualizer,
                        frame,
                        current_mask_panel if current_mask_panel is not None else placeholder,
                        window_name,
                        status_line=f"{status}  |  {_FILE_HINT}",
                        right_title=(
                            "Object Masks (analyzed)"
                            if current_mask_panel is not None
                            else "Object Masks"
                        ),
                    )
                else:
                    LOGGER.debug("Captured video frame with shape=%s", frame.shape)
                    result = pipeline.run(frame)

                    if result is not None:
                        last_result = result
                        mask_panel = _build_mask_panel(visualizer, pipeline, result)
                        _show_side_by_side(
                            visualizer,
                            frame,
                            mask_panel,
                            window_name,
                            status_line=f"LIVE  |  objects: {len(result.detections)}",
                            right_title="Object Masks",
                        )
                    else:
                        height, width = frame.shape[:2]
                        _show_side_by_side(
                            visualizer,
                            frame,
                            visualizer.create_mask_placeholder(height, width),
                            window_name,
                            status_line="LIVE  |  no objects detected",
                        )

            key = cv2.waitKey(video.wait_delay_ms() if is_file else 1) & 0xFF

            if key == ord("q"):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord(" ") and is_file:
                paused = video.toggle_pause(current_frame)
                state = "Paused" if paused else "Resumed"
                print(f"Video {state}")

                if current_frame is not None:
                    height, width = current_frame.shape[:2]
                    mask_panel = (
                        current_mask_panel
                        if current_mask_panel is not None
                        else visualizer.create_mask_placeholder(height, width)
                    )
                    _show_side_by_side(
                        visualizer,
                        current_frame,
                        mask_panel,
                        window_name,
                        status_line=f"{'PAUSED' if paused else 'PLAYING'}  |  {_FILE_HINT}",
                        right_title=(
                            "Object Masks (analyzed)"
                            if current_mask_panel is not None
                            else "Object Masks"
                        ),
                    )

            if key == ord("a") and is_file and current_frame is not None:
                if not paused:
                    video.pause(current_frame)
                    paused = True

                print("Analyzing frame…")
                result = pipeline.run(current_frame)

                if result is None:
                    print("No objects detected on this frame.")
                    height, width = current_frame.shape[:2]
                    current_mask_panel = visualizer.create_mask_placeholder(height, width)
                else:
                    last_result = result
                    print_result(result)
                    current_mask_panel = _build_mask_panel(visualizer, pipeline, result)

                _show_side_by_side(
                    visualizer,
                    current_frame,
                    current_mask_panel,
                    window_name,
                    status_line=(
                        f"PAUSED  |  objects: {len(last_result.detections)}  |  {_FILE_HINT}"
                        if last_result is not None
                        else f"PAUSED  |  no objects  |  {_FILE_HINT}"
                    ),
                    right_title=(
                        "Object Masks (analyzed)"
                        if last_result is not None
                        else "Object Masks"
                    ),
                )

            if key == ord("s") and last_result is not None:
                image_name = f"{Path(source_name).stem}_frame" if is_file else "webcam_frame"
                export_result(
                    exporter=exporter,
                    result=last_result,
                    image_name=image_name,
                )
                print(f"Saved {len(last_result.decisions)} object(s) to CSV.")

            if not is_file:
                if key == ord("c"):
                    paused = True
                    print("Frame captured. Press 'p' to resume.")

                if key == ord("p"):
                    paused = not paused
                    print(f"Video {'Paused' if paused else 'Resumed'}")

    except KeyboardInterrupt:
        LOGGER.debug("Video mode interrupted by user.")

    finally:
        video.stop()
        cv2.destroyAllWindows()
        print("Smart Storage stopped.")
