"""CV Pipeline for IT peripheral recognition."""

import time

import cv2
import numpy as np

from src import config
from src.color_detector import ColorDetector
from src.models import Decision, DetectionResult, PipelineResult


class Pipeline:
    """Main CV pipeline: enhance -> segment -> clean -> detect -> decide."""

    def __init__(self) -> None:
        self.color_detector = ColorDetector()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance image using CLAHE and Gaussian blur.

        Args:
            image: BGR input image.

        Returns:
            Enhanced BGR image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_SIZE,
        )
        l_enhanced = clahe.apply(l_channel)

        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced = cv2.GaussianBlur(enhanced, config.BLUR_KERNEL, 0)
        return enhanced

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment object from background using multi-strategy thresholding.

        Tries Otsu (best for solid-color backgrounds), then adaptive threshold
        as fallback. Picks the strategy that produces the largest clean region.

        Args:
            image: Enhanced BGR image.

        Returns:
            Binary mask (object=255, background=0).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, config.PRE_BLUR_KERNEL, 0)

        # Strategy 1: Otsu — works well for solid-color backgrounds
        _, otsu_inv = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        _, otsu_norm = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Strategy 2: Adaptive threshold (fallback for uneven lighting)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            config.ADAPTIVE_BLOCK_SIZE,
            config.ADAPTIVE_C,
        )

        # Pick the mask whose largest contour covers the most area
        # (after a quick clean pass to remove noise)
        # Prefer masks where object is 2%–60% of frame (not the background)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        candidates = [otsu_inv, otsu_norm, adaptive]
        best_mask = otsu_inv
        best_area = 0

        for candidate in candidates:
            cleaned = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # Sort by area descending
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                largest_area = cv2.contourArea(sorted_contours[0])
                frame_area = image.shape[0] * image.shape[1]
                ratio = largest_area / frame_area

                # Skip if largest contour is the whole frame (background captured)
                if ratio > 0.80:
                    # Try second largest if exists
                    if len(sorted_contours) > 1:
                        second_area = cv2.contourArea(sorted_contours[1])
                        ratio = second_area / frame_area
                        if 0.01 < ratio < 0.70 and second_area > best_area:
                            best_area = second_area
                            best_mask = candidate
                    continue

                if 0.01 < ratio < 0.70 and largest_area > best_area:
                    best_area = largest_area
                    best_mask = candidate

        return best_mask

    def clean(self, mask: np.ndarray) -> np.ndarray:
        """Clean segmentation mask using morphological operations.

        Args:
            mask: Binary segmentation mask.

        Returns:
            Cleaned binary mask with only the largest contour.
        """
        # Close: fill holes inside object (larger kernel for real photos)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)

        # Open: remove small noise outside object
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=2)

        # Keep only the largest contour
        contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            # Fallback: try on original mask without open
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        if not contours:
            return opened

        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(opened)
        cv2.drawContours(clean_mask, [largest], -1, 255, cv2.FILLED)
        return clean_mask

    def detect(
        self, image: np.ndarray, mask: np.ndarray
    ) -> DetectionResult | None:
        """Detect object properties: bounding box, color, size, and shape.

        Args:
            image: Enhanced BGR image.
            mask: Cleaned binary mask.

        Returns:
            DetectionResult or None if no object found.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        frame_area = image.shape[0] * image.shape[1]
        area_ratio = area / frame_area

        if area_ratio < config.MIN_OBJECT_RATIO or area_ratio > config.MAX_OBJECT_RATIO:
            # If too large, the mask likely captured the background — try inverting
            if area_ratio > config.MAX_OBJECT_RATIO:
                inverted = cv2.bitwise_not(mask)
                contours2, _ = cv2.findContours(
                    inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours2:
                    largest2 = max(contours2, key=cv2.contourArea)
                    area2 = cv2.contourArea(largest2)
                    ratio2 = area2 / frame_area
                    if config.MIN_OBJECT_RATIO < ratio2 < config.MAX_OBJECT_RATIO:
                        largest = largest2
                        area = area2
                        area_ratio = ratio2
                    else:
                        return None
                else:
                    return None
            else:
                return None

        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        perimeter = cv2.arcLength(largest, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0

        # Debug output
        print(f"[detect] area_ratio={area_ratio:.3f} aspect={aspect_ratio:.2f} "
              f"circ={circularity:.3f} solid={solidity:.3f} extent={extent:.3f}")

        # Shape classification
        if circularity > 0.65 and solidity > 0.85:
            shape_category = "oval"  # Mouse, charger
        elif extent > 0.75 and solidity > 0.85:
            shape_category = "rectangular"  # Keyboard, phone
        else:
            shape_category = "irregular"  # Cables, headphones

        # Size categorization
        if aspect_ratio > config.LONG_THIN_ASPECT:
            size_category = "long_thin"
        elif area_ratio < config.SMALL_MAX_RATIO:
            size_category = "small"
        elif area_ratio < config.MEDIUM_MAX_RATIO:
            size_category = "medium"
        else:
            size_category = "large"

        # Color detection
        color_result = self.color_detector.detect(image, mask)

        return DetectionResult(
            bbox=(x, y, w, h),
            color_hsv=color_result["hsv"],
            color_kmeans=color_result["kmeans"],
            primary_color=color_result["primary_color"],
            size_category=size_category,
            area_pixels=int(area),
            aspect_ratio=round(aspect_ratio, 2),
            circularity=round(circularity, 3),
            solidity=round(solidity, 3),
            extent=round(extent, 3),
            shape_category=shape_category,
            contour=largest,
        )

    def decide(self, detection: DetectionResult) -> Decision:
        """Classify object based on color, size, and shape features.

        Shape is the primary differentiator between object types:
        - oval + small/medium → Mouse
        - rectangular + medium/large → Keyboard
        - irregular + any → Headphones / Cable
        - any + small → Charger / Flash Drive (by color)

        Args:
            detection: DetectionResult from detect stage.

        Returns:
            Decision with category and confidence.
        """
        color = detection.primary_color
        size = detection.size_category
        shape = detection.shape_category
        color_conf = max(
            detection.color_hsv.confidence,
            detection.color_kmeans.confidence,
        )

        DARK = {"black", "blue", "gray", "silver"}

        # Shape-based classification with color refinement
        category = "Unknown Object"
        confidence = 0.0
        is_unknown = False

        # Special case: coiled cable — circular shape with low circularity
        # (ring shape: circ 0.25-0.55, not a solid oval)
        # Also catches cables with low solidity
        is_coiled = (0.20 < detection.circularity < 0.60 and detection.extent < 0.70)

        # Special case: charger/adapter with pins — very low circularity due to pins
        # aspect ~1.0 (square), circ < 0.20, solidity > 0.72 (solid compact block)
        # Headphones also have low circularity but LOW solidity (gaps/cups) — excluded here
        is_charger_block = (
            detection.circularity < 0.20
            and detection.aspect_ratio < 1.8
            and size in ("small", "medium")
            and detection.solidity > 0.72
        )

        if is_charger_block:
            if color in {"white", "gray", "silver"}:
                category = "iPhone Charger"
                confidence = 0.80 * color_conf
            else:
                category = "Android Charger"
                confidence = 0.75 * color_conf
        elif is_coiled or (detection.solidity < 0.65 and size in ("small", "long_thin")):
            # Cables: coiled ring shape OR long thin with low solidity
            # Medium/large irregular objects (headphones) are excluded here
            if color in {"white", "silver", "gray"}:
                category = "USB-C Cable"
                confidence = 0.78 * color_conf
            else:
                category = "Power Cable"
                confidence = 0.75 * color_conf

        elif size == "long_thin":
            # Cables
            if color in DARK:
                category = "Power Cable"
                confidence = 0.80 * color_conf
            else:
                category = "USB-C Cable"
                confidence = 0.75 * color_conf

        elif size == "small":
            # Small objects: phone chargers, flash drives
            if shape == "rectangular":
                # Small rectangular = phone charger block
                if color == "white":
                    category = "iPhone Charger"
                    confidence = 0.85 * color_conf
                else:
                    category = "Android Charger"
                    confidence = 0.80 * color_conf
            elif color == "white":
                category = "iPhone Charger"
                confidence = 0.80 * color_conf
            elif color in {"gray", "silver"}:
                category = "Flash Drive"
                confidence = 0.80 * color_conf
            elif color in DARK:
                category = "Android Charger"
                confidence = 0.80 * color_conf
            else:
                category = "Android Charger"
                confidence = 0.60 * color_conf

        elif shape == "rectangular":
            # Rectangular objects
            if size == "large":
                # Large rectangular = keyboard
                if color == "white":
                    category = "Keyboard (White)"
                    confidence = 0.85 * color_conf
                else:
                    category = "Keyboard"
                    confidence = 0.90 * color_conf
            elif size == "medium":
                # Medium rectangular: could be keyboard or laptop charger
                # Laptop chargers are more square (aspect_ratio < 2.0)
                if detection.aspect_ratio < 2.0:
                    # Compact rectangular block = laptop charger
                    if color == "white":
                        category = "MacBook Charger"
                        confidence = 0.80 * color_conf
                    else:
                        category = "Laptop Charger"
                        confidence = 0.75 * color_conf
                else:
                    # Wide rectangular = keyboard
                    if color == "white":
                        category = "Keyboard (White)"
                        confidence = 0.80 * color_conf
                    else:
                        category = "Keyboard"
                        confidence = 0.85 * color_conf

        elif shape == "oval":
            # Oval objects: mouse
            category = "Mouse"
            if color in DARK:
                confidence = 0.85 * color_conf
            else:
                category = "Mouse (White)"
                confidence = 0.80 * color_conf

        elif shape == "irregular":
            # Irregular: headphones, adapters
            if size in ("medium", "large"):
                category = "Headphones"
                confidence = 0.70 * color_conf
            else:
                category = "Adapter"
                confidence = 0.60 * color_conf

        else:
            # Fallback: use size + color
            if size in ("medium", "large") and color in DARK:
                # Could be mouse or keyboard — use extent to decide
                if detection.extent > 0.70:
                    category = "Keyboard"
                    confidence = 0.70 * color_conf
                else:
                    category = "Mouse"
                    confidence = 0.65 * color_conf
            else:
                is_unknown = True
                confidence = 0.20 * color_conf

        if confidence < config.LOW_CONFIDENCE:
            is_unknown = True
            closest = category
            category = "Unknown Object"
        else:
            closest = ""

        return Decision(
            category=category,
            confidence=round(confidence, 3),
            color=color,
            size=size,
            method_used="combined",
            is_unknown=is_unknown,
            closest_match=closest,
        )

    def run(self, image: np.ndarray) -> PipelineResult | None:
        """Execute the full pipeline on an image.

        Args:
            image: BGR input image.

        Returns:
            PipelineResult with all intermediate outputs, or None if no object.
        """
        start = time.time()

        enhanced = self.enhance(image)
        mask = self.segment(enhanced)
        cleaned = self.clean(mask)
        detection = self.detect(enhanced, cleaned)

        if detection is None:
            return None

        decision = self.decide(detection)
        elapsed_ms = (time.time() - start) * 1000

        return PipelineResult(
            original=image,
            enhanced=enhanced,
            mask=mask,
            cleaned_mask=cleaned,
            detection=detection,
            decision=decision,
            processing_time_ms=round(elapsed_ms, 1),
        )
