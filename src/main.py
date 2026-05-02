"""Smart Storage — IT Peripheral Recognition System.

Main application entry point with video and image processing modes.

Usage:
    # Video mode (webcam):
    uv run python -m src.main --mode video

    # Image mode (single file):
    uv run python -m src.main --mode image --source path/to/image.jpg

    # Image mode with CSV export:
    uv run python -m src.main --mode image --source photo.jpg --output results.csv

Controls (video mode):
    q - Quit
    s - Save current result to CSV
    c - Capture and freeze current frame
    p - Pause/resume video
"""

import argparse
import sys

import cv2

from src.data_exporter import DataExporter
from src.pipeline import Pipeline
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer


def run_video_mode(output_path: str | None = None) -> None:
    """Run the pipeline on live webcam video.

    Args:
        output_path: Optional CSV output path.
    """
    pipeline = Pipeline()
    visualizer = Visualizer()
    video = VideoProcessor()
    exporter = DataExporter(output_path) if output_path else DataExporter()

    if not video.start():
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
                    print("Warning: Failed to capture frame.")
                    continue

                result = pipeline.run(frame)
                if result is not None:
                    last_result = result
                    visualizer.show_pipeline(result)
                else:
                    # Show original frame with "No object" message
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
                    cv2.imshow("Smart Storage — Pipeline Dashboard", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s") and last_result is not None:
                exporter.export(last_result.decision, last_result.detection)
                print(
                    f"Saved: {last_result.decision.category} "
                    f"({last_result.decision.confidence:.0%})"
                )
            elif key == ord("c"):
                paused = True
                print("Frame captured. Press 'p' to resume.")
            elif key == ord("p"):
                paused = not paused
                state = "Paused" if paused else "Resumed"
                print(f"Video {state}")
    except KeyboardInterrupt:
        pass
    finally:
        video.stop()
        cv2.destroyAllWindows()
        print("Smart Storage stopped.")


def run_image_mode(source: str, output_path: str | None = None) -> None:
    """Run the pipeline on a single image file.

    Args:
        source: Path to the input image.
        output_path: Optional CSV output path.
    """
    pipeline = Pipeline()
    visualizer = Visualizer()
    exporter = DataExporter(output_path) if output_path else DataExporter()

    image = cv2.imread(source)
    if image is None:
        print(f"Error: Could not read image '{source}'.")
        sys.exit(1)

    print(f"Smart Storage — Image Mode: {source}")
    print("-" * 50)

    result = pipeline.run(image)
    if result is None:
        print("No object detected in the image.")
        cv2.namedWindow("Smart Storage - No Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Smart Storage - No Detection", image)
        cv2.waitKey(1)
        print("\nPress any key in the OpenCV window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Display results
    d = result.decision
    print(f"Category: {d.category}")
    print(f"Confidence: {d.confidence:.0%}")
    print(f"Color: {d.color}")
    print(f"Size: {d.size}")
    print(f"Method: {d.method_used}")
    print(f"Processing time: {result.processing_time_ms:.0f}ms")

    if d.is_unknown:
        print(f"Closest match: {d.closest_match}")

    # Export to CSV
    exporter.export(d, result.detection)
    print(f"\nResult saved to CSV: {exporter.output_path}")

    # Show dashboard
    visualizer.show_pipeline(result)
    print("\nPress any key in the OpenCV window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    """Parse arguments and run the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Smart Storage — IT Peripheral Recognition System"
    )
    parser.add_argument(
        "--mode",
        choices=["video", "image"],
        default="video",
        help="Processing mode: 'video' for webcam, 'image' for file (default: video)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to input image (required for image mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to CSV output file (default: output/results.csv)",
    )

    args = parser.parse_args()

    if args.mode == "image" and args.source is None:
        parser.error("--source is required for image mode")

    if args.mode == "video":
        run_video_mode(args.output)
    else:
        run_image_mode(args.source, args.output)


if __name__ == "__main__":
    main()
