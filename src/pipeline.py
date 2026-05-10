"""CV Pipeline for Smart Storage — Object Sorting System.

Pipeline:
image -> enhance -> segment -> clean -> detect -> decide

--- Classification order in decide() ---

1. Keyboard        — large dark elongated rectangular
2. Mouse           — compact dark rounded
3. Charger Adapter — compact light solid block/rect, circ >= 0.25
4. Flash Drive     — small compact any color
5. USB-C Cable     — loop/elongated/hollow-coil, neutral color
6. Headphones      — large/medium ring_like/irregular, solid >= 0.62
7. Colored Object  — chromatic fallback
8. Unknown Object

--- Calibration data (from real --debug measurements) ---

Object                  edge   solid  circ   shape       area
White charger (Image_13) 0.011  0.534  0.132  block       0.071
Silver charger (Image_6) 0.018  0.851  0.386  rectangular 0.044
Cable hollow (Image_5)   0.017  0.819  0.181  rectangular 0.045
Big headphones (Image_2) 0.076  0.640  0.161  ring_like   0.207
Small headphones (Image_4) 0.060 0.729 0.196  ring_like   0.122

Key separator — cable hollow vs charger:
  Cable hollow:     circ=0.181  (low — elongated blob from coil inner circle)
  Silver charger:   circ=0.386  (higher — compact rectangular block)
  White charger:    circ=0.132  (also low — BUT shape=block, not rectangular)

  The edge < 0.025 guard WRONGLY rejects both chargers (edge=0.011 and 0.018).
  REMOVED. New approach:

  For shape=rectangular AND edge < 0.025:
    → cable hollow if circ < 0.25  (cable: circ=0.181 → cable)
    → charger      if circ >= 0.25 (silver charger: circ=0.386 → charger)

  For shape=block:
    → always allow charger (white charger has shape=block, never a cable)
    → cable hollow never produces shape=block (it's a nearly circular blob
      which classifies as rectangular or oval, not block)
"""

import logging
import time

import cv2
import numpy as np

from src.classifier_protocol import Classifier
from src.color_detector import ColorDetector
from src.config import AppConfig
from src.models import Decision, DetectionResult, PipelineResult

LOGGER = logging.getLogger(__name__)


class Pipeline:
    """Main CV pipeline: enhance -> segment -> clean -> detect -> decide."""

    def __init__(
        self,
        config: AppConfig | None = None,
        classifier: Classifier | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.color_detector = ColorDetector(self.config)
        self.classifier = classifier
        self._current_frame: np.ndarray | None = None  # set during run()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance image using CLAHE, gamma correction and light blur."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_size,
        )
        l_enhanced = clahe.apply(l_channel)

        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced = self._apply_gamma(enhanced, self.config.gamma_value)
        enhanced = cv2.GaussianBlur(enhanced, self.config.blur_kernel, 0)

        return enhanced

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment the main object from background."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config.pre_blur_kernel, 0)

        _, otsu_inv = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        _, otsu_norm = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        adaptive_inv = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.config.adaptive_block_size, self.config.adaptive_c,
        )
        adaptive_norm = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, self.config.adaptive_block_size, self.config.adaptive_c,
        )

        color_distance_mask = self._create_color_distance_mask(image)
        edge_mask = self._create_edge_object_mask(gray)
        local_contrast_mask = self._create_local_contrast_mask(gray)

        candidates = [
            otsu_inv,
            otsu_norm,
            adaptive_inv,
            adaptive_norm,
            color_distance_mask,
            edge_mask,
            local_contrast_mask,
            cv2.bitwise_or(color_distance_mask, edge_mask),
            cv2.bitwise_or(local_contrast_mask, edge_mask),
            cv2.bitwise_or(otsu_inv, edge_mask),
            cv2.bitwise_or(adaptive_inv, edge_mask),
        ]

        best_object_mask = None
        best_score = -1.0

        for candidate in candidates:
            prepared = self._prepare_candidate_mask(candidate)
            object_mask, score = self._select_best_object_mask(prepared)

            if object_mask is not None and score > best_score:
                best_object_mask = object_mask
                best_score = score

        if best_object_mask is None:
            return otsu_inv

        return best_object_mask

    def clean(self, mask: np.ndarray) -> np.ndarray:
        """Clean selected object mask."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        binary = self._remove_border_connected_components(binary)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.config.close_kernel_size,
        )
        closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, close_kernel,
            iterations=self.config.close_iterations,
        )

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.config.open_kernel_size,
        )
        opened = cv2.morphologyEx(
            closed, cv2.MORPH_OPEN, open_kernel,
            iterations=self.config.open_iterations,
        )

        main_contour = self._find_best_contour(opened)

        if main_contour is None:
            main_contour = self._find_best_contour(binary)

        if main_contour is None:
            return opened

        main_features = self._estimate_basic_features(main_contour)
        main_mask = np.zeros_like(binary)
        cv2.drawContours(main_mask, [main_contour], -1, 255, cv2.FILLED)

        edge_density = self._calculate_edge_density_from_mask(binary, main_mask)

        is_thin_or_fragmented_object = (
            edge_density >= 0.08
            or main_features["solidity"] < 0.78
            or main_features["extent"] < 0.55
        )

        if is_thin_or_fragmented_object:
            clean_mask = self._preserve_nearby_object_parts(
                source_mask=binary,
                main_contour=main_contour,
            )
        else:
            clean_mask = main_mask

            should_fill_holes = (
                main_features["extent"] > 0.35
                and main_features["solidity"] > 0.50
                and edge_density < 0.16
            )

            if should_fill_holes:
                clean_mask = self._fill_holes(clean_mask)

        clean_mask = self._remove_border_noise(clean_mask)

        if int(np.sum(clean_mask > 0)) == 0:
            return main_mask

        return clean_mask

    def detect(self, image: np.ndarray, mask: np.ndarray) -> DetectionResult | None:
        """Detect one object. Kept for compatibility with earlier code."""
        detections = self.detect_all(
            color_image=image,
            processing_image=image,
            mask=mask,
        )
        return detections[0] if detections else None

    def detect_all(
        self,
        color_image: np.ndarray,
        processing_image: np.ndarray,
        mask: np.ndarray,
    ) -> list[DetectionResult]:
        """Detect the main foreground object."""
        object_mask = self._build_final_object_mask(mask)

        if object_mask is None:
            return []

        detection = self._build_detection_from_mask(
            color_image=color_image,
            processing_image=processing_image,
            object_mask=object_mask,
            object_id=1,
        )

        return [detection] if detection is not None else []

    def _extract_roi(self, detection: DetectionResult) -> np.ndarray | None:
        """Crop the bounding-box region from the current frame for ML classifiers."""
        if self._current_frame is None:
            return None
        x, y, w, h = detection.bbox
        roi = self._current_frame[y : y + h, x : x + w]
        return roi if roi.size > 0 else None

    def decide(self, detection: DetectionResult) -> Decision:
        """Delegate to injected classifier or fall back to rule-based logic."""
        if self.classifier is not None:
            roi = self._extract_roi(detection)
            return self.classifier.classify(detection, roi)
        return self._rule_based_decide(detection)

    def _rule_based_decide(self, detection: DetectionResult) -> Decision:
        """Produce final automatic decision using hard-coded rules."""
        color = detection.primary_color
        size = detection.size_category

        LOGGER.debug(
            "DETECT: color=%s size=%s shape=%s area=%.3f circ=%.3f solid=%.3f "
            "extent=%.3f edge=%.3f bbox_w=%.3f bbox_h=%.3f aspect=%.2f",
            color, size, detection.shape_category,
            detection.area_ratio, detection.circularity, detection.solidity,
            detection.extent, detection.edge_density,
            detection.bbox_width_ratio, detection.bbox_height_ratio,
            detection.aspect_ratio,
        )

        color_confidence = max(
            detection.color_hsv.confidence,
            detection.color_kmeans.confidence,
        )

        category = "Unknown Object"
        confidence = 0.0
        is_unknown = False
        closest_match = ""

        chromatic_colors = {
            "red", "green", "blue", "yellow", "orange", "purple", "brown",
        }

        is_keyboard    = self._is_keyboard(detection, color)
        is_mouse       = self._is_mouse(detection, color)
        is_charger     = self._is_charger_adapter(detection, color)
        is_flash_drive = self._is_flash_drive(detection, color)
        is_usb_cable   = self._is_usb_cable(detection, color)
        is_headphones  = self._is_headphones(detection, color)

        if is_keyboard:
            category = "Keyboard"
            confidence = 0.90 * color_confidence

        elif is_mouse:
            category = "Mouse"
            confidence = 0.88 * color_confidence

        elif is_charger:
            category = "Charger Adapter"
            confidence = 0.84 * color_confidence

        elif is_flash_drive:
            category = "Flash Drive"
            confidence = 0.76 * color_confidence

        elif is_usb_cable:
            category = "USB-C Cable"
            confidence = 0.76 * color_confidence

        elif is_headphones:
            category = "Headphones"
            confidence = 0.78 * color_confidence

        elif color in chromatic_colors:
            category = "Colored Object"
            confidence = 0.65 * color_confidence

        else:
            category = "Unknown Object"
            confidence = max(0.30 * color_confidence, 0.15)
            is_unknown = True
            closest_match = self._guess_closest_category(detection)

        if confidence < self.config.low_confidence:
            is_unknown = True
            closest_match = closest_match or category
            category = "Unknown Object"

        return Decision(
            category=category,
            confidence=round(float(confidence), 3),
            color=color,
            size=size,
            method_used="combined",
            is_unknown=is_unknown,
            closest_match=closest_match,
            object_id=getattr(detection, "object_id", 1),
        )

    def decide_all(self, detections: list[DetectionResult]) -> list[Decision]:
        """Make decisions for all detections."""
        return [self.decide(detection) for detection in detections]

    def run(self, image: np.ndarray) -> PipelineResult | None:
        """Execute the full pipeline."""
        start = time.time()

        enhanced = self.enhance(image)
        mask = self.segment(enhanced)
        cleaned = self.clean(mask)

        detections = self.detect_all(
            color_image=image,
            processing_image=enhanced,
            mask=cleaned,
        )

        if not detections:
            detections = self.detect_all(
                color_image=image,
                processing_image=enhanced,
                mask=mask,
            )

        if not detections:
            return None

        self._current_frame = image
        decisions = self.decide_all(detections)
        self._current_frame = None
        elapsed_ms = (time.time() - start) * 1000

        return PipelineResult(
            original=image,
            enhanced=enhanced,
            mask=mask,
            cleaned_mask=cleaned,
            detection=detections[0],
            decision=decisions[0],
            detections=detections,
            decisions=decisions,
            processing_time_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # Classification helpers — thresholds calibrated from real debug data
    # ------------------------------------------------------------------

    def _is_keyboard(self, detection: DetectionResult, color: str) -> bool:
        """Check whether object looks like a keyboard."""
        is_keyboard_color = color in {"black", "gray", "blue", "silver"}

        large_rectangular_object = (
            detection.shape_category == "rectangular"
            and detection.area_ratio >= 0.12
            and max(detection.bbox_width_ratio, detection.bbox_height_ratio) >= 0.55
            and min(detection.bbox_width_ratio, detection.bbox_height_ratio) >= 0.22
        )

        elongated_object = detection.aspect_ratio >= 1.70

        solid_mask = (
            detection.extent > 0.45
            and detection.solidity > 0.60
        )

        not_mouse_size = detection.area_ratio > 0.12

        return (
            is_keyboard_color
            and large_rectangular_object
            and elongated_object
            and solid_mask
            and not_mouse_size
        )

    def _is_mouse(self, detection: DetectionResult, color: str) -> bool:
        """Check whether object looks like a mouse."""
        is_mouse_color = color in {"black", "gray", "blue"}

        compact_bbox = (
            detection.bbox_width_ratio < 0.78
            and detection.bbox_height_ratio < 0.82
        )

        not_keyboard_like = detection.aspect_ratio < 2.35

        rounded_or_compact_shape = detection.shape_category in {
            "oval", "block", "rectangular",
        }

        solid_object = (
            detection.solidity > 0.58
            and detection.extent > 0.28
        )

        plausible_area = 0.02 <= detection.area_ratio <= 0.32

        return (
            is_mouse_color
            and compact_bbox
            and not_keyboard_like
            and rounded_or_compact_shape
            and solid_object
            and plausible_area
        )

    def _is_charger_adapter(self, detection: DetectionResult, color: str) -> bool:
        """Check whether object looks like a compact charger adapter.

        Calibrated from real data:

        White charger (Image_13):  edge=0.011, solid=0.534, circ=0.132, shape=block
        Silver charger (Image_6):  edge=0.018, solid=0.851, circ=0.386, shape=rectangular
        Cable hollow (Image_5):    edge=0.017, solid=0.819, circ=0.181, shape=rectangular

        Previous approach (edge < 0.025 guard) incorrectly rejected both chargers
        because their edge values are similar to the cable hollow.

        New approach — use circularity to separate cable hollow from charger:
        - Cable hollow inner circle: circ=0.181 (elongated blob, not circular)
        - Silver charger:            circ=0.386 (compact block, more circular)
        - Threshold: circ >= 0.25 → charger (rectangular shape only)

        For shape=block: always allow charger. A cable hollow never produces
        shape=block — it is a near-circular or rectangular blob.

        For shape=oval: allow charger (some chargers photograph as oval blobs).

        White charger (Image_13) has shape=block AND solid=0.534 (below the old
        solid > 0.50 threshold with some margin). The solid_shape check now
        accepts solid > 0.40 to include this case.
        """
        is_light = color in {"white", "silver", "gray"}

        compact_block = (
            detection.aspect_ratio < 2.20
            and detection.bbox_width_ratio < 0.60
            and detection.bbox_height_ratio < 0.70
            and detection.area_ratio < 0.18
        )

        # shape=block → always a candidate for charger (never cable hollow)
        # shape=rectangular → only if circularity >= 0.25 (cable hollow has circ=0.181)
        # shape=oval → allow (some charger photographs look oval)
        shape_ok = (
            detection.shape_category == "block"
            or detection.shape_category == "oval"
            or (
                detection.shape_category == "rectangular"
                and detection.circularity >= 0.25
            )
        )

        # Relaxed solid threshold to include white charger (solid=0.534)
        solid_shape = (
            detection.extent > 0.25
            and detection.solidity > 0.40
        )

        not_loop = detection.shape_category != "ring_like"

        not_too_fragmented = detection.edge_density < 0.18

        return (
            is_light
            and compact_block
            and shape_ok
            and solid_shape
            and not_loop
            and not_too_fragmented
        )

    def _is_flash_drive(self, detection: DetectionResult, color: str) -> bool:
        """Check whether object looks like a small flash drive."""
        return (
            color in {"gray", "silver", "black", "green", "blue", "red"}
            and detection.size_category == "small"
            and detection.aspect_ratio < 3.5
            and detection.extent > 0.30
        )

    def _is_usb_cable(self, detection: DetectionResult, color: str) -> bool:
        """Check whether object looks like a USB-C cable.

        Calibrated from real data:

        Cable hollow (Image_5): edge=0.017, solid=0.819, circ=0.181, shape=rectangular
          → Caught by: hollow_rectangular_coil (shape=rectangular AND circ < 0.25)

        Cable ring_like (other images): solid < 0.62
          → Caught by: ring_like_cable

        NOT cable — chargers pass through _is_charger_adapter() first in decide()
        so they never reach this check.

        NOT cable — headphones have solid >= 0.62 (both 0.640 and 0.729).
        """
        is_cable_color = color in {"white", "silver", "gray", "black"}

        if not is_cable_color:
            return False

        # Solid large dark keyboard-like rectangle → not a cable
        solid_large_rectangle = (
            detection.shape_category == "rectangular"
            and detection.area_ratio >= 0.12
            and detection.extent > 0.45
            and detection.solidity > 0.60
            and detection.edge_density < 0.05
        )

        if solid_large_rectangle:
            return False

        # Signal 1: straight elongated cable
        elongated_shape = (
            detection.size_category == "long_thin"
            or detection.aspect_ratio >= 2.4
        )

        # Signal 2: hollow rectangular coil mask
        # Calibrated: cable hollow circ=0.181 (< 0.25)
        # Silver charger circ=0.386 (>= 0.25) → NOT caught here → goes to charger
        hollow_rectangular_coil = (
            detection.shape_category in {"rectangular", "block"}
            and detection.circularity < 0.25
            and detection.area_ratio < 0.10
            and detection.edge_density < 0.025
        )

        # Signal 3: ring_like shape with LOW solidity
        # Calibrated: headphones solid=0.640 and 0.729 (both >= 0.62)
        # Cable ring_like masks are more fragmented → solid < 0.62
        ring_like_cable = (
            detection.shape_category in {"ring_like", "irregular"}
            and detection.solidity < 0.62
        )

        # Signal 4: fragmented strand structure
        fragmented_strand = (
            detection.edge_density >= 0.09
            and detection.solidity < 0.55
        )

        return (
            elongated_shape
            or hollow_rectangular_coil
            or ring_like_cable
            or fragmented_strand
        )

    def _is_headphones(self, detection: DetectionResult, color: str) -> bool:
        """Check whether object looks like headphones.

        Calibrated from real data:
        - Big headphones:   solid=0.640, edge=0.076, area=0.207, ring_like
        - Small headphones: solid=0.729, edge=0.060, area=0.122, ring_like

        Both have solid >= 0.62. Primary separator from cables.
        """
        headphones_colors = {"black", "white", "silver", "gray"}

        if color not in headphones_colors:
            return False

        complex_shape = (
            detection.shape_category in {"irregular", "ring_like", "block"}
            or detection.edge_density >= 0.060
            or detection.solidity < 0.82
        )

        not_keyboard_like = not (
            color in {"black", "gray", "blue"}
            and detection.aspect_ratio >= 2.35
            and detection.extent > 0.42
            and detection.solidity > 0.65
        )

        not_charger_like = not self._is_charger_adapter(detection, color)

        # Primary separator from cables: solid >= 0.62
        solid_enough_for_headphones = detection.solidity >= 0.62

        return (
            detection.size_category in {"medium", "large"}
            and complex_shape
            and not_keyboard_like
            and not_charger_like
            and solid_enough_for_headphones
        )

    def _guess_closest_category(self, detection: DetectionResult) -> str:
        """Return fallback category hint."""
        if detection.shape_category == "oval":
            return "Mouse-like object"

        if detection.shape_category == "rectangular":
            return "Rectangular object"

        if detection.shape_category in {"ring_like", "irregular"}:
            return "Cable-like or headphones-like object"

        return "Unknown visual object"

    # ------------------------------------------------------------------
    # Internal pipeline helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        if gamma <= 0:
            return image
        inverse_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inverse_gamma * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, table)

    def _build_final_object_mask(self, mask: np.ndarray) -> np.ndarray | None:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        binary = self._remove_border_connected_components(binary)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        main_contour = self._find_best_contour(binary)

        if main_contour is None:
            return None

        result = self._preserve_nearby_object_parts(
            source_mask=binary,
            main_contour=main_contour,
        )

        if int(np.sum(result > 0)) == 0:
            result = np.zeros_like(binary)
            cv2.drawContours(result, [main_contour], -1, 255, cv2.FILLED)

        return result

    def _build_detection_from_mask(
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

        if self._looks_like_background_region(
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

        contour_points = np.vstack(contours)
        display_contour = cv2.convexHull(contour_points)
        aspect_ratio = max(w, h) / max(min(w, h), 1)

        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        circularity = (
            (4 * np.pi * area_pixels) / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )

        hull = cv2.convexHull(contour_points)
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

    def _prepare_candidate_mask(self, mask: np.ndarray) -> np.ndarray:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        binary = self._remove_border_connected_components(binary)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
        return binary

    def _select_best_object_mask(
        self, mask: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None, -1.0

        best_contour = None
        best_score = -1.0

        for contour in contours:
            score = self._score_contour(contour, mask.shape[:2])
            if score > best_score:
                best_contour = contour
                best_score = score

        if best_contour is None:
            return None, -1.0

        object_mask = np.zeros_like(mask)
        cv2.drawContours(object_mask, [best_contour], -1, 255, cv2.FILLED)

        features = self._estimate_basic_features(best_contour)
        edge_density = self._calculate_edge_density_from_mask(mask, object_mask)

        is_fragmented_or_cable_like = (
            edge_density >= 0.055
            or features["solidity"] < 0.85
            or features["extent"] < 0.60
        )

        if is_fragmented_or_cable_like:
            preserved_mask = self._preserve_nearby_object_parts(
                source_mask=mask,
                main_contour=best_contour,
            )
            if int(np.sum(preserved_mask > 0)) > int(np.sum(object_mask > 0)):
                object_mask = preserved_mask

        return object_mask, best_score

    def _find_best_contour(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        scored_contours = []
        for contour in contours:
            score = self._score_contour(contour, mask.shape[:2])
            if score > -0.5:
                scored_contours.append((score, contour))

        if scored_contours:
            scored_contours.sort(key=lambda item: item[0], reverse=True)
            return scored_contours[0][1]

        return self._find_largest_reasonable_contour(mask)

    def _find_largest_reasonable_contour(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        height, width = mask.shape[:2]
        frame_area = height * width
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            area_ratio = area / frame_area
            if self.config.min_object_ratio <= area_ratio <= self.config.max_object_ratio:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                if aspect_ratio <= 25.0:
                    return contour

        return None

    def _score_contour(
        self, contour: np.ndarray, image_shape: tuple[int, int],
    ) -> float:
        height, width = image_shape
        frame_area = height * width
        area = cv2.contourArea(contour)

        if area <= 0:
            return -1.0

        area_ratio = area / frame_area
        if area_ratio < self.config.min_object_ratio or area_ratio > self.config.max_object_ratio:
            return -1.0

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h

        if bbox_area <= 0:
            return -1.0

        aspect_ratio = max(w, h) / max(min(w, h), 1)
        extent = area / bbox_area
        bbox_width_ratio = w / width
        bbox_height_ratio = h / height

        if self._is_obvious_artifact(
            x=x, y=y, w=w, h=h,
            area_ratio=area_ratio,
            aspect_ratio=aspect_ratio,
            extent=extent,
            image_width=width,
            image_height=height,
        ):
            return -1.0

        object_center_x = x + w / 2
        object_center_y = y + h / 2
        image_center_x = width / 2
        image_center_y = height / 2

        distance = np.sqrt(
            (object_center_x - image_center_x) ** 2
            + (object_center_y - image_center_y) ** 2
        )

        max_distance = np.sqrt(image_center_x ** 2 + image_center_y ** 2)
        centrality = 1.0 - min(distance / max_distance, 1.0)

        border_penalty = self._calculate_border_penalty(
            x=x, y=y, w=w, h=h, image_width=width, image_height=height,
        )

        huge_bbox_penalty = 0.0
        if bbox_width_ratio > 0.88 and bbox_height_ratio > 0.55:
            huge_bbox_penalty -= 1.5
        if bbox_width_ratio > 0.96 or bbox_height_ratio > 0.96:
            huge_bbox_penalty -= 1.2

        shadow_like_penalty = 0.0
        if area_ratio > 0.35 and extent > 0.80:
            shadow_like_penalty -= 1.0

        if aspect_ratio > 25.0:
            return -1.0

        extent_bonus = 0.25 if 0.04 <= extent <= 0.98 else -0.20
        aspect_bonus = 0.15 if aspect_ratio <= 18.0 else -0.30

        return (
            area_ratio * 3.0
            + centrality * 1.4
            + extent_bonus
            + aspect_bonus
            - border_penalty
            + huge_bbox_penalty
            + shadow_like_penalty
        )

    def _is_obvious_artifact(
        self,
        x: int, y: int, w: int, h: int,
        area_ratio: float, aspect_ratio: float, extent: float,
        image_width: int, image_height: int,
    ) -> bool:
        relative_width = w / image_width
        relative_height = h / image_height

        if area_ratio < self.config.min_object_ratio:
            return True
        if area_ratio > self.config.max_object_ratio:
            return True
        if aspect_ratio > 25.0 and extent < 0.35:
            return True
        if relative_width > 0.94 and relative_height < 0.08:
            return True
        if relative_height > 0.94 and relative_width < 0.08:
            return True
        if relative_width > 0.78 and y < image_height * 0.08:
            return True
        if relative_width > 0.82 and y > image_height * 0.86:
            return True
        if relative_width > 0.92 and relative_height > 0.70:
            return True
        if x <= 2 and y <= 2 and relative_width > 0.60 and relative_height > 0.45:
            return True

        return False

    def _looks_like_background_region(
        self,
        x: int, y: int, w: int, h: int,
        area_ratio: float, image_width: int, image_height: int,
    ) -> bool:
        bbox_width_ratio = w / image_width
        bbox_height_ratio = h / image_height

        touches_left = x <= 2
        touches_right = x + w >= image_width - 2
        touches_top = y <= 2
        touches_bottom = y + h >= image_height - 2

        touched_borders = sum(
            [touches_left, touches_right, touches_top, touches_bottom]
        )

        if touched_borders >= 2 and area_ratio > 0.25:
            return True
        if bbox_width_ratio > 0.95 and bbox_height_ratio > 0.60:
            return True
        if bbox_width_ratio > 0.90 and area_ratio > 0.35:
            return True
        if bbox_height_ratio > 0.90 and area_ratio > 0.35:
            return True

        return False

    def _calculate_border_penalty(
        self,
        x: int, y: int, w: int, h: int,
        image_width: int, image_height: int,
    ) -> float:
        penalty = 0.0
        if x <= 2:
            penalty += 0.35
        if y <= 2:
            penalty += 0.35
        if x + w >= image_width - 2:
            penalty += 0.35
        if y + h >= image_height - 2:
            penalty += 0.35
        return penalty

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape[:2]
        flood_filled = mask.copy()
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
        flood_filled_inv = cv2.bitwise_not(flood_filled)
        return cv2.bitwise_or(mask, flood_filled_inv)

    def _remove_border_connected_components(self, mask: np.ndarray) -> np.ndarray:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        height, width = binary.shape[:2]
        frame_area = height * width

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )

        cleaned = np.zeros_like(binary)

        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            area_ratio = area / frame_area
            bbox_width_ratio = w / width
            bbox_height_ratio = h / height

            touches_left = x <= 1
            touches_right = x + w >= width - 1
            touches_top = y <= 1
            touches_bottom = y + h >= height - 1

            touched_borders = sum(
                [touches_left, touches_right, touches_top, touches_bottom]
            )

            is_border_artifact = (
                touched_borders >= 2 and area_ratio > 0.015
            ) or (
                touched_borders >= 1
                and area_ratio > 0.20
                and (bbox_width_ratio > 0.70 or bbox_height_ratio > 0.70)
            )

            if not is_border_artifact:
                cleaned[labels == label] = 255

        return cleaned

    def _remove_border_noise(self, mask: np.ndarray) -> np.ndarray:
        cleaned = self._remove_border_connected_components(mask)
        if int(np.sum(cleaned > 0)) == 0:
            return mask
        return cleaned

    def _preserve_nearby_object_parts(
        self,
        source_mask: np.ndarray,
        main_contour: np.ndarray,
    ) -> np.ndarray:
        height, width = source_mask.shape[:2]
        frame_area = height * width

        x, y, w, h = cv2.boundingRect(main_contour)

        pad_x = max(20, int(w * 0.55))
        pad_y = max(20, int(h * 0.55))

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        expanded_roi = np.zeros_like(source_mask)
        expanded_roi[y1:y2, x1:x2] = 255

        candidate = cv2.bitwise_and(source_mask, expanded_roi)
        candidate = self._remove_border_connected_components(candidate)

        connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        connected = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, connect_kernel, iterations=2,
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            connected, connectivity=8,
        )

        result = np.zeros_like(source_mask)

        main_center_x = x + w / 2
        main_center_y = y + h / 2
        max_allowed_distance = max(w, h) * 1.45

        for label in range(1, num_labels):
            component_x = int(stats[label, cv2.CC_STAT_LEFT])
            component_y = int(stats[label, cv2.CC_STAT_TOP])
            component_w = int(stats[label, cv2.CC_STAT_WIDTH])
            component_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            component_area = int(stats[label, cv2.CC_STAT_AREA])

            if component_area <= 0:
                continue

            component_area_ratio = component_area / frame_area

            if component_area_ratio > 0.35:
                continue

            component_center_x, component_center_y = centroids[label]

            distance_to_main = np.sqrt(
                (component_center_x - main_center_x) ** 2
                + (component_center_y - main_center_y) ** 2
            )

            close_to_main = distance_to_main <= max_allowed_distance
            not_tiny_noise = component_area >= max(16, int(frame_area * 0.00005))
            not_large_background = not self._looks_like_background_region(
                x=component_x, y=component_y,
                w=component_w, h=component_h,
                area_ratio=component_area_ratio,
                image_width=width, image_height=height,
            )

            if close_to_main and not_tiny_noise and not_large_background:
                result[labels == label] = 255

        if int(np.sum(result > 0)) == 0:
            cv2.drawContours(result, [main_contour], -1, 255, cv2.FILLED)

        final_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = cv2.morphologyEx(
            result, cv2.MORPH_CLOSE, final_kernel, iterations=1,
        )

        return result

    def _create_edge_object_mask(self, gray: np.ndarray) -> np.ndarray:
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 120)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        dilated = cv2.dilate(edges, dilate_kernel, iterations=1)
        closed = cv2.morphologyEx(
            dilated, cv2.MORPH_CLOSE, close_kernel, iterations=2,
        )
        closed = self._remove_border_connected_components(closed)

        return closed

    def _create_local_contrast_mask(self, gray: np.ndarray) -> np.ndarray:
        gray_float = gray.astype(np.float32)
        blur_large = cv2.GaussianBlur(gray_float, (31, 31), 0)
        difference = cv2.absdiff(gray_float, blur_large)

        normalized = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        _, mask = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = self._remove_border_connected_components(mask)

        return mask

    def _create_color_distance_mask(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        height, width = image.shape[:2]
        border_size = max(8, int(min(height, width) * 0.04))

        border_pixels = np.concatenate(
            [
                lab[:border_size, :, :].reshape(-1, 3),
                lab[-border_size:, :, :].reshape(-1, 3),
                lab[:, :border_size, :].reshape(-1, 3),
                lab[:, -border_size:, :].reshape(-1, 3),
            ],
            axis=0,
        )

        background_color = np.median(border_pixels, axis=0)
        distance = np.linalg.norm(lab - background_color, axis=2)

        normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        _, mask = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = self._remove_border_connected_components(mask)

        return mask

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

    def _calculate_edge_density_from_mask(
        self, source_mask: np.ndarray, object_mask: np.ndarray,
    ) -> float:
        object_pixels = object_mask > 0
        object_area = int(np.sum(object_pixels))

        if object_area == 0:
            return 0.0

        source_pixels = source_mask > 0
        active_pixels = int(np.sum(source_pixels & object_pixels))
        return active_pixels / object_area

    def _estimate_basic_features(self, contour: np.ndarray) -> dict[str, float]:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        return {"extent": float(extent), "solidity": float(solidity)}

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