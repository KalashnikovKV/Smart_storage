"""Smart Storage — Object Sorting System CLI entry point."""

from __future__ import annotations

import argparse
import logging

from src.app.video_processor import is_video_file
from src.modes import (
    run_batch_mode,
    run_image_mode,
    run_label_image_mode,
    run_label_video_file_mode,
    run_label_video_mode,
    run_video_mode,
)
from src.modes.common import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and run the selected mode."""
    parser = argparse.ArgumentParser(
        description="Smart Storage — Object Sorting System",
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
            "video (webcam if omitted, or .mov/.mp4 file), "
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
            "instead of the built-in threshold pipeline."
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

    parser.add_argument(
        "--debug-contours",
        action="store_true",
        help=(
            "Save a debug overlay with numbered contour outlines "
            "(rule-based path only; output/stages/contours/*_contours.jpg)."
        ),
    )

    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        metavar="N",
        help="Maximum objects to detect per frame (default: 8).",
    )

    args = parser.parse_args()

    configure_logging(args.debug)
    LOGGER.debug("Parsed arguments: %s", args)

    if args.mode in {"image", "batch"} and args.source is None:
        parser.error("--source is required for image and batch modes")

    should_show_window = not (args.no_window or args.no_display)

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
        run_video_mode(
            output_path=args.output,
            model_path=args.model,
            device=effective_device,
            source=args.source,
            max_objects=args.max_objects,
        )

    elif args.mode == "image":
        run_image_mode(
            source=args.source,
            output_path=args.output,
            save_outputs=not args.no_save_images,
            show_window=should_show_window,
            model_path=args.model,
            device=effective_device,
            debug_contours=args.debug_contours,
            max_objects=args.max_objects,
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
