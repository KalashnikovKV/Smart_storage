"""YOLO-Seg segmenter — implements the Segmenter protocol.

Returns per-object class name, bbox, mask and confidence. Color and
classification in the full app are handled by Pipeline + ColorDetector.
"""

from __future__ import annotations

import dataclasses
import logging

import cv2
import numpy as np

from src.models import YOLODetection
from src.pipeline.protocols import Segmenter  # noqa: F401 — satisfies Protocol

LOGGER = logging.getLogger(__name__)

# Trained flash drives are compact; larger blobs are often misclassified mice.
_FLASH_DRIVE_MAX_AREA_RATIO = 0.032
_MOUSE_LIKE_MIN_AREA_RATIO = 0.032

# Some classes are harder in video; allow a lower keep threshold after best-per-class merge.
_DEFAULT_MIN_KEEP_CONFIDENCE = 0.15
_CLASS_MIN_KEEP_CONFIDENCE: dict[str, float] = {
    "cable": 0.06,
}


def _parse_results(results: object, image_shape: tuple[int, int]) -> list[YOLODetection]:
    """Convert ultralytics Results into YOLODetection instances."""
    image_height, image_width = image_shape
    detections: list[YOLODetection] = []

    for result in results:  # type: ignore[union-attr]
        if result.boxes is None or result.masks is None:
            LOGGER.debug("No boxes or masks in YOLO result — skipping.")
            continue

        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            mask_data = result.masks.data[index].cpu().numpy()
            mask = cv2.resize(
                mask_data,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask > 0.5).astype(np.uint8) * 255

            detections.append(
                YOLODetection(
                    class_name=result.names[class_id],
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    mask=mask,
                    confidence=round(confidence, 3),
                )
            )

    return sorted(detections, key=lambda d: d.confidence, reverse=True)


def detection_area_ratio(detection: YOLODetection, image_shape: tuple[int, int]) -> float:
    """Foreground area of *detection* as a fraction of the full frame."""
    frame_area = image_shape[0] * image_shape[1]
    if frame_area <= 0:
        return 0.0
    return float(np.count_nonzero(detection.mask > 0)) / frame_area


def reclassify_by_geometry(
    detections: list[YOLODetection],
    image_shape: tuple[int, int],
) -> list[YOLODetection]:
    """Correct common class swaps using mask size heuristics."""
    corrected: list[YOLODetection] = []

    for detection in detections:
        class_name = detection.class_name.lower().replace(" ", "_")
        area_ratio = detection_area_ratio(detection, image_shape)

        if class_name == "flash_drive" and area_ratio >= _MOUSE_LIKE_MIN_AREA_RATIO:
            corrected.append(
                dataclasses.replace(detection, class_name="mouse"),
            )
            LOGGER.debug(
                "YOLO reclassify flash_drive -> mouse (area=%.3f)",
                area_ratio,
            )
            continue

        corrected.append(detection)

    return corrected


def resolve_overlapping_class_conflicts(
    detections: list[YOLODetection],
    image_shape: tuple[int, int],
    *,
    min_mask_iou: float = 0.45,
) -> list[YOLODetection]:
    """When two classes overlap the same region, pick the better semantic fit."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[YOLODetection] = []

    for candidate in ordered:
        replaced = False

        for index, existing in enumerate(kept):
            if mask_iou(candidate.mask, existing.mask) < min_mask_iou:
                continue

            winner = _pick_conflicting_detection(existing, candidate, image_shape)
            kept[index] = winner
            replaced = True
            break

        if not replaced:
            kept.append(candidate)

    return sorted(kept, key=lambda d: d.confidence, reverse=True)


def _pick_conflicting_detection(
    left: YOLODetection,
    right: YOLODetection,
    image_shape: tuple[int, int],
) -> YOLODetection:
    """Resolve two different classes that cover largely the same pixels."""
    left_name = left.class_name.lower().replace(" ", "_")
    right_name = right.class_name.lower().replace(" ", "_")

    best = left if left.confidence >= right.confidence else right

    if {left_name, right_name} == {"mouse", "flash_drive"}:
        if detection_area_ratio(best, image_shape) >= _MOUSE_LIKE_MIN_AREA_RATIO:
            return dataclasses.replace(best, class_name="mouse")
        return best

    return best


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union for two binary masks."""
    foreground_a = mask_a > 0
    foreground_b = mask_b > 0
    intersection = int(np.logical_and(foreground_a, foreground_b).sum())

    if intersection == 0:
        return 0.0

    union = int(np.logical_or(foreground_a, foreground_b).sum())
    return intersection / union if union > 0 else 0.0


def deduplicate_overlapping_masks(
    detections: list[YOLODetection],
    min_mask_iou: float = 0.80,
) -> list[YOLODetection]:
    """Drop lower-confidence detections when instance masks overlap heavily."""
    if len(detections) <= 1:
        return detections

    kept: list[YOLODetection] = []

    for candidate in detections:
        if any(
            mask_iou(candidate.mask, existing.mask) >= min_mask_iou
            for existing in kept
        ):
            continue
        kept.append(candidate)

    if len(kept) < len(detections):
        LOGGER.debug(
            "YOLO mask NMS: %d -> %d detections (mask IoU >= %.2f)",
            len(detections),
            len(kept),
            min_mask_iou,
        )

    return kept


def consolidate_best_per_class(
    detections: list[YOLODetection],
) -> list[YOLODetection]:
    """Keep the highest-confidence detection for each YOLO class."""
    best_by_class: dict[str, YOLODetection] = {}

    for detection in detections:
        class_key = detection.class_name.lower().replace(" ", "_")
        current = best_by_class.get(class_key)
        if current is None or detection.confidence > current.confidence:
            best_by_class[class_key] = detection

    consolidated = sorted(
        best_by_class.values(),
        key=lambda detection: detection.confidence,
        reverse=True,
    )

    if len(consolidated) < len(detections):
        LOGGER.debug(
            "YOLO best-per-class: %d -> %d",
            len(detections),
            len(consolidated),
        )

    return consolidated


def trim_low_confidence_detections(
    detections: list[YOLODetection],
    *,
    max_objects: int = 8,
    min_confidence: float = _DEFAULT_MIN_KEEP_CONFIDENCE,
    min_area_ratio: float = 0.008,
    image_shape: tuple[int, int] | None = None,
    class_min_confidence: dict[str, float] | None = None,
) -> list[YOLODetection]:
    """Keep the top detection always; filter weak or tiny additional objects."""
    if not detections:
        return detections

    class_thresholds = class_min_confidence or _CLASS_MIN_KEEP_CONFIDENCE
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = [ordered[0]]

    for candidate in ordered[1:]:
        if len(kept) >= max_objects:
            break

        class_key = candidate.class_name.lower().replace(" ", "_")
        threshold = class_thresholds.get(class_key, min_confidence)
        if candidate.confidence < threshold:
            continue
        if (
            image_shape is not None
            and detection_area_ratio(candidate, image_shape) < min_area_ratio
        ):
            continue
        kept.append(candidate)

    if len(kept) < len(detections):
        LOGGER.debug(
            "YOLO confidence trim: %d -> %d (default_min_conf=%.2f, min_area=%.3f)",
            len(detections),
            len(kept),
            min_confidence,
            min_area_ratio,
        )

    return kept


def filter_preferred_classes(
    detections: list[YOLODetection],
    preferred_classes: list[str] | None,
    *,
    min_conf: float = 0.05,
    strict: bool = False,
) -> list[YOLODetection]:
    """Keep only preferred classes when they are present in *detections*.

    When *strict* is True and no preferred class is found, return an empty list
    instead of falling back to unrelated high-confidence classes (e.g. cable vs
    headphones).
    """
    if not preferred_classes or not detections:
        return detections

    preferred = {name.lower().replace(" ", "_") for name in preferred_classes}
    kept = [
        detection
        for detection in detections
        if detection.class_name.lower().replace(" ", "_") in preferred
        and detection.confidence >= min_conf
    ]

    if kept:
        filtered = deduplicate_overlapping_masks(kept)
        LOGGER.debug(
            "YOLO class filter %s -> %d detection(s)",
            preferred_classes,
            len(filtered),
        )
        return filtered

    if strict:
        LOGGER.debug(
            "YOLO class filter %s: no matches (strict — returning empty)",
            preferred_classes,
        )
        return []

    return detections


def combine_instance_masks(
    detections: list[YOLODetection],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Merge per-object YOLO masks into one binary mask for visualization."""
    height, width = image_shape
    combined = np.zeros((height, width), dtype=np.uint8)

    for detection in detections:
        combined = cv2.bitwise_or(combined, detection.mask)

    return combined


class YOLOSegmenter:
    """YOLO-Seg wrapper that satisfies the Segmenter protocol."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        conf: float = 0.25,
        imgsz: int = 640,
        mask_nms_iou: float = 0.80,
        max_det: int = 30,
        max_objects: int = 8,
        min_keep_confidence: float = 0.15,
        preferred_classes: list[str] | None = None,
        strict_preferred: bool = False,
        preferred_min_conf: float = 0.05,
    ) -> None:
        try:
            from ultralytics import YOLO  # lazy import — optional dependency
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Cannot load ultralytics/torch. Install YOLO deps with "
                "'uv sync --group yolo' (CPU torch). For GPU, see "
                "aidlc-docs/construction/build-and-test/build-instructions.md."
            ) from exc

        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.mask_nms_iou = mask_nms_iou
        self.max_det = max_det
        self.max_objects = max_objects
        self.min_keep_confidence = min_keep_confidence
        self.preferred_classes = preferred_classes
        self.strict_preferred = strict_preferred
        self.preferred_min_conf = preferred_min_conf

    def segment(self, image: np.ndarray) -> list[YOLODetection]:
        """Run YOLO-Seg and return detections sorted by confidence (desc)."""
        results = self.model.predict(
            image,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        detections = _parse_results(results, image.shape[:2])
        image_shape = image.shape[:2]
        detections = deduplicate_overlapping_masks(
            detections,
            min_mask_iou=self.mask_nms_iou,
        )
        detections = reclassify_by_geometry(detections, image_shape)
        detections = resolve_overlapping_class_conflicts(
            detections,
            image_shape,
            min_mask_iou=0.45,
        )
        detections = deduplicate_overlapping_masks(
            detections,
            min_mask_iou=self.mask_nms_iou,
        )
        detections = consolidate_best_per_class(detections)
        detections = trim_low_confidence_detections(
            detections,
            max_objects=self.max_objects,
            min_confidence=self.min_keep_confidence,
            min_area_ratio=0.008,
            image_shape=image_shape,
        )
        return filter_preferred_classes(
            detections,
            self.preferred_classes,
            min_conf=self.preferred_min_conf,
            strict=self.strict_preferred,
        )
