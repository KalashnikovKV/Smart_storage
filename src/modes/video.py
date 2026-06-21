"""Live webcam and video-file processing mode."""

from __future__ import annotations

import dataclasses
import logging
import sys
import threading
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np

from src.app import DataExporter, VideoProcessor, Visualizer
from src.modes.common import build_pipeline, export_result, print_result
from src.models import DetectionResult, PipelineResult

LOGGER = logging.getLogger(__name__)

VIDEO_WINDOW_NAME = "Smart Storage — Video + Masks + Contours"
VIDEO_RULE_MAX_WIDTH = 640
VIDEO_YOLO_MAX_WIDTH = 0  # 0 = full camera resolution for YOLO
VIDEO_PROCESS_INTERVAL = 2
VIDEO_YOLO_PROCESS_INTERVAL = 6
VIDEO_YOLO_CONF = 0.05
VIDEO_YOLO_IMGSZ = 1280
_FILE_HINT = "space=pause+analyze   a=re-analyze   s=save CSV   q=quit"
_WEBCAM_HINT = "q=quit   s=save to CSV   c=capture   p=pause"


def _ensure_window(
    visualizer: Visualizer,
    window_name: str,
    content: np.ndarray | None = None,
) -> None:
    if not visualizer._window_created:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if content is not None:
            h, w = content.shape[:2]
            win_w, win_h = visualizer._fit_window_size(w, h)
            cv2.resizeWindow(window_name, win_w, win_h)
        visualizer._window_created = True


def _resize_for_pipeline(
    frame: np.ndarray,
    max_width: int,
) -> tuple[np.ndarray, float]:
    """Downscale wide frames for faster rule-based processing."""
    if max_width <= 0:
        return frame, 1.0

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0

    scale = max_width / width
    new_width = max_width
    new_height = max(1, int(height * scale))
    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _scale_detection(detection: DetectionResult, inv_scale: float) -> DetectionResult:
    """Map detection geometry from a downscaled frame back to full resolution."""
    x, y, width, height = detection.bbox
    contour = detection.contour
    if contour is not None and len(contour) > 0:
        contour = (contour.astype(np.float32) * inv_scale).astype(np.int32)

    return dataclasses.replace(
        detection,
        bbox=(
            int(x * inv_scale),
            int(y * inv_scale),
            max(1, int(width * inv_scale)),
            max(1, int(height * inv_scale)),
        ),
        contour=contour,
        area_pixels=int(detection.area_pixels * inv_scale * inv_scale),
    )


def _map_result_to_frame(
    result: PipelineResult,
    display_frame: np.ndarray,
    scale: float,
) -> PipelineResult:
    """Reproject pipeline outputs onto the original display frame."""
    if abs(scale - 1.0) < 1e-6:
        return dataclasses.replace(result, original=display_frame)

    inv_scale = 1.0 / scale
    target_height, target_width = display_frame.shape[:2]

    def upscale_binary(mask: np.ndarray) -> np.ndarray:
        return cv2.resize(
            mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    scaled_detections = [
        _scale_detection(detection, inv_scale)
        for detection in result.detections
    ]

    return dataclasses.replace(
        result,
        original=display_frame,
        enhanced=cv2.resize(
            result.enhanced,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        ),
        mask=upscale_binary(result.mask),
        cleaned_mask=upscale_binary(result.cleaned_mask),
        detections=scaled_detections,
        detection=scaled_detections[0],
    )


def _detection_masks_from_result(
    pipeline,
    result: PipelineResult,
) -> list[np.ndarray]:
    """Build one binary mask per detection, aligned with *result.detections*."""
    height, width = result.original.shape[:2]

    if pipeline.segmenter is not None:
        masks: list[np.ndarray] = []
        for detection in result.detections:
            mask = np.zeros((height, width), dtype=np.uint8)
            if detection.contour is not None and len(detection.contour) > 0:
                cv2.drawContours(mask, [detection.contour], -1, 255, cv2.FILLED)
            else:
                x, y, box_w, box_h = detection.bbox
                mask[y : y + box_h, x : x + box_w] = 255
            masks.append(mask)
        return masks

    return pipeline.get_object_masks(result.cleaned_mask, result.original)


def _build_panels(
    visualizer: Visualizer,
    pipeline,
    result: PipelineResult,
) -> tuple[np.ndarray, np.ndarray]:
    object_masks = _detection_masks_from_result(pipeline, result)
    mask_panel = visualizer.build_colored_object_masks(
        object_masks,
        detections=result.detections,
        decisions=result.decisions,
    )
    contour_panel = visualizer.build_contour_overlay_panel(result.original, result)
    return mask_panel, contour_panel


def _analyze_frame(
    pipeline,
    visualizer: Visualizer,
    frame: np.ndarray,
    *,
    max_width: int = VIDEO_RULE_MAX_WIDTH,
) -> tuple[PipelineResult | None, np.ndarray, np.ndarray]:
    """Run the pipeline on *frame* and build mask + contour panels."""
    small_frame, scale = _resize_for_pipeline(frame, max_width)
    result = pipeline.run(small_frame)

    if result is None:
        height, width = frame.shape[:2]
        return (
            None,
            visualizer.create_mask_placeholder(height, width),
            visualizer.create_contour_placeholder(height, width),
        )

    display_result = _map_result_to_frame(result, frame, scale)
    mask_panel, contour_panel = _build_panels(visualizer, pipeline, display_result)
    return display_result, mask_panel, contour_panel


def _show_triple_view(
    visualizer: Visualizer,
    frame: np.ndarray,
    mask_panel: np.ndarray,
    contour_panel: np.ndarray,
    window_name: str,
    *,
    status_line: str,
    mask_title: str = "Object Masks",
    contour_title: str = "Contours",
) -> None:
    display = visualizer.create_video_triple_view(
        frame,
        mask_panel,
        contour_panel,
        left_title="Video",
        center_title=mask_title,
        right_title=contour_title,
        status_line=status_line,
    )
    _ensure_window(visualizer, window_name, display)
    cv2.imshow(window_name, display)


class _BackgroundAnalyzer:
    """Process webcam frames on a worker thread so the preview stays smooth."""

    def __init__(
        self,
        pipeline,
        visualizer: Visualizer,
        *,
        max_width: int = VIDEO_RULE_MAX_WIDTH,
        process_interval: int = VIDEO_PROCESS_INTERVAL,
    ) -> None:
        self._pipeline = pipeline
        self._visualizer = visualizer
        self._max_width = max_width
        self._process_interval = process_interval
        self._frame_counter = 0
        self._queue: Queue[np.ndarray] = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._busy = False
        self._last_result: PipelineResult | None = None
        self._mask_panel: np.ndarray | None = None
        self._contour_panel: np.ndarray | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def submit(self, frame: np.ndarray) -> None:
        self._frame_counter += 1
        if self._frame_counter % self._process_interval != 0:
            return

        frame_copy = frame.copy()
        try:
            self._queue.put_nowait(frame_copy)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(frame_copy)
            except Full:
                pass

    def snapshot(
        self,
        frame: np.ndarray,
    ) -> tuple[PipelineResult | None, np.ndarray, np.ndarray, bool]:
        with self._lock:
            result = self._last_result
            mask_panel = self._mask_panel
            contour_panel = self._contour_panel
            busy = self._busy

        height, width = frame.shape[:2]
        if mask_panel is None:
            mask_panel = self._visualizer.create_mask_placeholder(height, width)
        if contour_panel is None:
            contour_panel = self._visualizer.create_contour_placeholder(height, width)

        return result, mask_panel, contour_panel, busy

    def _worker(self) -> None:
        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except Empty:
                continue

            with self._lock:
                self._busy = True

            try:
                display_result, mask_panel, contour_panel = _analyze_frame(
                    self._pipeline,
                    self._visualizer,
                    frame,
                    max_width=self._max_width,
                )
            except Exception:
                LOGGER.exception("Background frame analysis failed.")
                continue
            finally:
                with self._lock:
                    self._busy = False

            with self._lock:
                if display_result is not None:
                    self._last_result = display_result
                    self._mask_panel = mask_panel
                    self._contour_panel = contour_panel


def run_video_mode(
    output_path: str | None = None,
    model_path: str | None = None,
    device: str = "cpu",
    source: str | None = None,
    max_objects: int | None = None,
    yolo_prefer: list[str] | None = None,
    yolo_prefer_strict: bool = False,
) -> None:
    """Run the pipeline on a webcam or a video file."""
    LOGGER.debug(
        "Starting video mode. source=%s output_path=%s device=%s",
        source,
        output_path,
        device,
    )

    is_yolo = model_path is not None
    pipeline = build_pipeline(
        model_path,
        device=device,
        max_objects=max_objects,
        fast=not is_yolo,
        yolo_conf=VIDEO_YOLO_CONF if is_yolo else None,
        yolo_imgsz=VIDEO_YOLO_IMGSZ if is_yolo else None,
        yolo_prefer=yolo_prefer if is_yolo else None,
        yolo_prefer_strict=yolo_prefer_strict if is_yolo else False,
    )
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
    video_max_width = VIDEO_YOLO_MAX_WIDTH if is_yolo else VIDEO_RULE_MAX_WIDTH
    video_interval = (
        VIDEO_YOLO_PROCESS_INTERVAL if is_yolo else VIDEO_PROCESS_INTERVAL
    )
    analyzer = (
        None
        if is_file
        else _BackgroundAnalyzer(
            pipeline,
            visualizer,
            max_width=video_max_width,
            process_interval=video_interval,
        )
    )

    print("Smart Storage — Video Mode")
    if is_file:
        print(f"Source: {source_name}")
        print("Layout: video | object masks | contours on camera")
        print(f"Controls: {_FILE_HINT}")
        print("Tip: pause with space — analysis starts automatically.")
    else:
        print("Layout: video | object masks | contours on camera")
        print(f"Controls: {_WEBCAM_HINT}")
        if is_yolo:
            prefer_note = (
                f", prefer={','.join(yolo_prefer)}"
                if yolo_prefer
                else ""
            )
            print(
                f"YOLO live: full resolution, conf={VIDEO_YOLO_CONF}, "
                f"every {video_interval} frames, imgsz={VIDEO_YOLO_IMGSZ}{prefer_note}"
            )
        else:
            print(
                f"Rule live: every {video_interval} frames, "
                f"max width {video_max_width}px"
            )
    print("-" * 50)

    paused = False
    last_result = None
    current_frame = None
    current_mask_panel = None
    current_contour_panel = None
    analyzing = False

    if analyzer is not None:
        analyzer.start()

    def placeholders(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        return (
            visualizer.create_mask_placeholder(height, width),
            visualizer.create_contour_placeholder(height, width),
        )

    def apply_analysis(
        frame: np.ndarray,
        *,
        status_prefix: str,
    ) -> None:
        nonlocal last_result, current_mask_panel, current_contour_panel, analyzing

        analyzing = True
        display_result, mask_panel, contour_panel = _analyze_frame(
            pipeline,
            visualizer,
            frame,
            max_width=video_max_width,
        )
        analyzing = False

        if display_result is None:
            print("No objects detected on this frame.")
            current_mask_panel, current_contour_panel = placeholders(frame)
            status = f"{status_prefix}  |  no objects  |  {_FILE_HINT}"
            mask_title = "Object Masks"
            contour_title = "Contours"
        else:
            last_result = display_result
            current_mask_panel = mask_panel
            current_contour_panel = contour_panel
            print_result(display_result)
            status = (
                f"{status_prefix}  |  objects: {len(display_result.detections)}  "
                f"|  {display_result.processing_time_ms:.0f} ms  |  {_FILE_HINT}"
            )
            mask_title = "Object Masks (analyzed)"
            contour_title = "Contours (analyzed)"

        _show_triple_view(
            visualizer,
            frame,
            current_mask_panel,
            current_contour_panel,
            window_name,
            status_line=status,
            mask_title=mask_title,
            contour_title=contour_title,
        )

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
                    if current_mask_panel is None or current_contour_panel is None:
                        current_mask_panel, current_contour_panel = placeholders(frame)

                    status = "PLAYING" if not paused else "PAUSED"
                    if analyzing:
                        status = f"ANALYZING…  |  {_FILE_HINT}"

                    _show_triple_view(
                        visualizer,
                        frame,
                        current_mask_panel,
                        current_contour_panel,
                        window_name,
                        status_line=f"{status}  |  {_FILE_HINT}",
                        mask_title=(
                            "Object Masks (analyzed)"
                            if last_result is not None
                            else "Object Masks"
                        ),
                        contour_title=(
                            "Contours (analyzed)"
                            if last_result is not None
                            else "Contours"
                        ),
                    )
                else:
                    assert analyzer is not None
                    analyzer.submit(frame)
                    result, mask_panel, contour_panel, busy = analyzer.snapshot(frame)

                    if result is not None:
                        last_result = result

                    object_count = len(result.detections) if result is not None else 0
                    if result is not None:
                        timing = f"{result.processing_time_ms:.0f} ms"
                        status = f"LIVE  |  objects: {object_count}  |  last: {timing}"
                    elif busy:
                        status = "LIVE  |  analyzing…"
                    else:
                        status = "LIVE  |  waiting for first detection…"

                    has_panels = result is not None
                    _show_triple_view(
                        visualizer,
                        frame,
                        mask_panel,
                        contour_panel,
                        window_name,
                        status_line=status,
                        mask_title=(
                            "Object Masks (analyzed)" if has_panels else "Object Masks"
                        ),
                        contour_title=(
                            "Contours (analyzed)" if has_panels else "Contours"
                        ),
                    )

            key = cv2.waitKey(video.wait_delay_ms() if is_file else 1) & 0xFF

            if key == ord("q"):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord(" ") and is_file and current_frame is not None:
                was_paused = paused
                paused = video.toggle_pause(current_frame)
                state = "Paused" if paused else "Resumed"
                print(f"Video {state}")

                if paused and not was_paused:
                    print("Analyzing paused frame…")
                    apply_analysis(current_frame, status_prefix="PAUSED")
                elif current_frame is not None:
                    if current_mask_panel is None or current_contour_panel is None:
                        current_mask_panel, current_contour_panel = placeholders(
                            current_frame,
                        )
                    _show_triple_view(
                        visualizer,
                        current_frame,
                        current_mask_panel,
                        current_contour_panel,
                        window_name,
                        status_line=f"{'PAUSED' if paused else 'PLAYING'}  |  {_FILE_HINT}",
                        mask_title=(
                            "Object Masks (analyzed)"
                            if last_result is not None
                            else "Object Masks"
                        ),
                        contour_title=(
                            "Contours (analyzed)"
                            if last_result is not None
                            else "Contours"
                        ),
                    )

            if key == ord("a") and is_file and current_frame is not None:
                if not paused:
                    video.pause(current_frame)
                    paused = True

                print("Re-analyzing frame…")
                apply_analysis(current_frame, status_prefix="PAUSED")

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
        if analyzer is not None:
            analyzer.stop()
        video.stop()
        cv2.destroyAllWindows()
        print("Smart Storage stopped.")
