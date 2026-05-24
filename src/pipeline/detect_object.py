"""Build DetectionResult from a binary object mask."""

import cv2
import numpy as np

from src.config import AppConfig
from src.pipeline.detect_color import ColorDetector
from src.pipeline.mask_ops import MaskOps
from src.models import DetectionResult


class ObjectMaskDetector:
    """Extract bbox, shape, and color metrics from a foreground mask."""

    def __init__(
        self,
        config: AppConfig,
        mask_ops: MaskOps,
        color_detector: ColorDetector,
    ) -> None:
        self.config = config
        self.mask_ops = mask_ops
        self.color_detector = color_detector

    def _prepare_binary_mask(self, mask: np.ndarray) -> np.ndarray:
        """Normalize mask to uint8 binary and remove border-connected artifacts."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        return self.mask_ops.remove_border_connected_components(binary)

    def build_object_mask_for_contour(
        self,
        binary: np.ndarray,
        contour: np.ndarray,
        color_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a single-object mask clipped to the cleaned foreground."""
        return self.mask_ops.extract_object_mask_from_contour(
            binary,
            contour,
            color_image=color_image,
        )

    def build_all_object_masks(
        self,
        mask: np.ndarray,
        color_image: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Build one binary mask per detected foreground object."""
        binary = self._prepare_binary_mask(mask)
        contours = self.mask_ops.find_object_contours(
            binary,
            max_objects=self.config.max_objects,
        )

        object_masks: list[np.ndarray] = []
        for contour in contours:
            object_masks.append(
                self.build_object_mask_for_contour(binary, contour, color_image),
            )

        return object_masks

    def build_final_object_mask(self, mask: np.ndarray) -> np.ndarray | None:
        """Build a mask for the single best-scoring object (legacy helper)."""
        object_masks = self.build_all_object_masks(mask)
        return object_masks[0] if object_masks else None

    def detect_all_from_mask(
        self,
        color_image: np.ndarray,
        processing_image: np.ndarray,
        mask: np.ndarray,
    ) -> list[DetectionResult]:
        """Detect and classify every foreground object in *mask*."""
        detections: list[DetectionResult] = []

        for object_id, object_mask in enumerate(
            self.build_all_object_masks(mask, color_image),
            start=1,
        ):
            detection = self.build_detection_from_mask(
                color_image=color_image,
                processing_image=processing_image,
                object_mask=object_mask,
                object_id=object_id,
            )
            if detection is not None:
                detections.append(detection)

        return detections

    def build_detection_from_mask(
        self,
        color_image: np.ndarray,
        processing_image: np.ndarray,
        object_mask: np.ndarray,
        object_id: int,
    ) -> DetectionResult | None:
        object_pixels = object_mask > 0
        area_pixels = int(np.sum(object_pixels))

        if area_pixels <= 0:
            return None

        image_height, image_width = processing_image.shape[:2]
        frame_area = image_height * image_width
        area_ratio = area_pixels / frame_area

        if area_ratio < self.config.min_object_ratio or area_ratio > self.config.max_object_ratio:
            return None

        points = cv2.findNonZero(object_mask)

        if points is None:
            return None

        x, y, w, h = cv2.boundingRect(points)
        bbox_width_ratio = w / image_width
        bbox_height_ratio = h / image_height

        if self.mask_ops.looks_like_background_region(
            x=x, y=y, w=w, h=h,
            area_ratio=area_ratio,
            image_width=image_width,
            image_height=image_height,
        ):
            return None

        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        main_contour = max(contours, key=cv2.contourArea)
        display_contour = main_contour
        aspect_ratio = max(w, h) / max(min(w, h), 1)

        perimeter = cv2.arcLength(main_contour, True)
        circularity = (
            (4 * np.pi * area_pixels) / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )

        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_pixels / hull_area if hull_area > 0 else 0.0

        bbox_area = w * h
        extent = area_pixels / bbox_area if bbox_area > 0 else 0.0
        edge_density = self._calculate_edge_density(processing_image, object_mask)

        shape_category = self._classify_shape(
            circularity=circularity,
            solidity=solidity,
            extent=extent,
            aspect_ratio=aspect_ratio,
            edge_density=edge_density,
        )

        if extent < self.config.min_detection_extent:
            if shape_category not in {"ring_like", "irregular"}:
                return None

        size_category, size_confidence = self._classify_visual_size(
            area_ratio=area_ratio,
            bbox_width_ratio=bbox_width_ratio,
            bbox_height_ratio=bbox_height_ratio,
            aspect_ratio=aspect_ratio,
            shape_category=shape_category,
        )

        color_result = self.color_detector.detect(color_image, object_mask)

        return DetectionResult(
            bbox=(x, y, w, h),
            color_hsv=color_result["hsv"],
            color_kmeans=color_result["kmeans"],
            primary_color=color_result["primary_color"],
            size_category=size_category,
            area_pixels=area_pixels,
            aspect_ratio=round(aspect_ratio, 2),
            circularity=round(circularity, 3),
            solidity=round(solidity, 3),
            extent=round(extent, 3),
            shape_category=shape_category,
            contour=display_contour,
            object_id=object_id,
            edge_density=round(edge_density, 3),
            area_ratio=round(area_ratio, 4),
            bbox_width_ratio=round(bbox_width_ratio, 4),
            bbox_height_ratio=round(bbox_height_ratio, 4),
            visual_size_label=size_category,
            size_confidence=round(size_confidence, 3),
        )

    def _calculate_edge_density(
        self, image: np.ndarray, mask: np.ndarray,
    ) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 130)

        object_pixels = mask > 0
        object_area = int(np.sum(object_pixels))

        if object_area == 0:
            return 0.0

        edge_pixels = int(np.sum((edges > 0) & object_pixels))
        return edge_pixels / object_area

    def _classify_shape(
        self,
        circularity: float,
        solidity: float,
        extent: float,
        aspect_ratio: float,
        edge_density: float,
    ) -> str:
        is_loop_like = (
            edge_density > 0.055
            and extent < 0.72
            and solidity < 0.90
            and circularity > 0.08
        )

        if is_loop_like:
            return "ring_like"

        is_oval_like = (
            circularity > 0.42
            and solidity > 0.58
            and aspect_ratio < 2.55
            and extent > 0.30
        )

        if is_oval_like:
            return "oval"

        is_rectangular_like = extent > 0.60 and solidity > 0.65

        if is_rectangular_like:
            return "rectangular"

        is_block_like = (
            aspect_ratio < 2.2
            and extent > 0.30
            and solidity > 0.45
        )

        if is_block_like:
            return "block"

        return "irregular"

    def _classify_visual_size(
        self,
        area_ratio: float,
        bbox_width_ratio: float,
        bbox_height_ratio: float,
        aspect_ratio: float,
        shape_category: str,
    ) -> tuple[str, float]:
        max_bbox_ratio = max(bbox_width_ratio, bbox_height_ratio)

        if aspect_ratio >= self.config.long_thin_aspect:
            return ("long_thin", 0.85)

        if shape_category in {"ring_like", "irregular"}:
            if area_ratio < self.config.small_max_ratio and max_bbox_ratio < 0.38:
                return ("small", 0.70)
            if area_ratio < self.config.medium_max_ratio and max_bbox_ratio < 0.70:
                return ("medium", 0.75)
            return ("large", 0.70)

        if area_ratio < self.config.small_max_ratio and max_bbox_ratio < 0.32:
            return ("small", 0.85)
        if area_ratio < self.config.medium_max_ratio and max_bbox_ratio < 0.62:
            return ("medium", 0.85)

        return ("large", 0.85)
