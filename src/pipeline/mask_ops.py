"""Shared binary-mask utilities used by segment, clean, and detect stages."""

import cv2
import numpy as np

from src.config import AppConfig


class MaskOps:
    """Contour scoring, border filtering, and mask geometry helpers."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape[:2]
        flood_filled = mask.copy()
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
        flood_filled_inv = cv2.bitwise_not(flood_filled)
        return cv2.bitwise_or(mask, flood_filled_inv)

    def remove_border_connected_components(self, mask: np.ndarray) -> np.ndarray:
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

    def remove_border_noise(self, mask: np.ndarray) -> np.ndarray:
        cleaned = self.remove_border_connected_components(mask)
        if int(np.sum(cleaned > 0)) == 0:
            return mask
        return cleaned

    def remove_small_components(
        self,
        mask: np.ndarray,
        min_area_ratio: float | None = None,
    ) -> np.ndarray:
        """Drop connected components smaller than *min_area_ratio* of the frame."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        height, width = binary.shape[:2]
        frame_area = height * width
        min_ratio = min_area_ratio or self.config.min_component_area_ratio
        min_area = max(24, int(frame_area * min_ratio))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )

        cleaned = np.zeros_like(binary)
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                cleaned[labels == label] = 255

        return cleaned

    def keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )

        if num_labels <= 1:
            return binary

        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        result = np.zeros_like(binary)
        result[labels == largest_label] = 255
        return result

    def background_lab_stats(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return LAB image, background color, mean/std of border color distance."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        height, width = lab.shape[:2]
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
        border_distance = np.linalg.norm(border_pixels - background_color, axis=1)

        return (
            lab,
            background_color,
            float(np.mean(border_distance)),
            float(np.std(border_distance)),
        )

    def _is_overfilled_mask(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        max_extent: float = 0.72,
    ) -> bool:
        """Return True when a mask looks like a solid bbox fill rather than an object."""
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if box_w <= 0 or box_h <= 0:
            return True

        bbox_area = box_w * box_h
        bbox_region = mask[y : y + box_h, x : x + box_w]
        bbox_area_pixels = int(np.sum(bbox_region > 0))
        extent = bbox_area_pixels / bbox_area

        if extent <= max_extent:
            return False

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return True

        main = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main)
        hull = cv2.convexHull(main)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        return extent > max_extent and solidity > 0.88

    def segment_inside_contour(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        binary: np.ndarray,
    ) -> np.ndarray:
        """Refine a mask inside the contour silhouette without expanding to bbox."""
        height, width = image.shape[:2]
        x, y, box_w, box_h = cv2.boundingRect(contour)
        pad = max(6, int(0.04 * max(box_w, box_h)))

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + box_w + pad)
        y2 = min(height, y + box_h + pad)

        roi = image[y1:y2, x1:x2]
        roi_binary = binary[y1:y2, x1:x2]
        roi_height, roi_width = roi.shape[:2]

        contour_region = np.zeros((roi_height, roi_width), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x1
        shifted[:, :, 1] -= y1
        cv2.drawContours(contour_region, [shifted], -1, 255, cv2.FILLED)

        base = cv2.bitwise_and(contour_region, roi_binary)

        lab, background_color, bg_mean, bg_std = self.background_lab_stats(image)
        roi_lab = lab[y1:y2, x1:x2]
        distance = np.linalg.norm(roi_lab - background_color, axis=2)

        soft_threshold = bg_mean + max(
            8.0,
            self.config.color_distance_soft_sigma * bg_std,
        )
        color_fg = np.where(distance >= soft_threshold, 255, 0).astype(np.uint8)
        color_fg = cv2.bitwise_and(color_fg, contour_region)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat_fg = np.where(
            hsv[:, :, 1] >= self.config.min_saturation,
            255,
            0,
        ).astype(np.uint8)
        sat_fg = cv2.bitwise_and(sat_fg, contour_region)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        border_gray = np.concatenate(
            [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]],
        )
        bg_gray = float(np.median(border_gray))
        bright_fg = np.where(
            gray >= bg_gray + self.config.brightness_delta,
            255,
            0,
        ).astype(np.uint8)
        bright_fg = cv2.bitwise_and(bright_fg, contour_region)

        seeds = cv2.bitwise_or(color_fg, sat_fg)
        seeds = cv2.bitwise_or(seeds, bright_fg)
        seeds = cv2.bitwise_or(seeds, base)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        combined = self.fill_holes(combined)
        combined = cv2.bitwise_and(combined, contour_region)
        combined = self.remove_small_components(combined)
        combined = self.keep_largest_component(combined)

        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = combined
        return full_mask

    def extract_object_mask_from_contour(
        self,
        binary: np.ndarray,
        contour: np.ndarray,
        color_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a per-object mask from contour geometry and optional color refinement."""
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)

        clipped = cv2.bitwise_and(filled, binary)
        clipped = self.remove_small_components(clipped)
        clipped = self.keep_largest_component(clipped)
        clipped_area = int(np.sum(clipped > 0))

        if color_image is not None:
            refined = self.segment_inside_contour(color_image, contour, binary)
            refined_area = int(np.sum(refined > 0))

            if (
                refined_area > int(clipped_area * 1.15)
                and not self._is_overfilled_mask(refined, contour, max_extent=0.65)
            ):
                finalized_refined = self._finalize_object_mask(color_image, refined)
                refined_final_area = int(np.sum(finalized_refined > 0))

                if refined_final_area <= int(clipped_area * 1.45):
                    return finalized_refined

                finalized_clipped = self._finalize_object_mask(color_image, clipped)
                if int(np.sum(finalized_clipped > 0)) >= int(clipped_area * 0.45):
                    return finalized_clipped

                return finalized_refined

        contour_area = cv2.contourArea(contour)
        features = self.estimate_basic_features(contour)
        edge_density = self.calculate_edge_density_from_mask(binary, filled)

        is_fragmented = (
            edge_density >= 0.08
            or features["solidity"] < 0.72
            or features["extent"] < 0.50
            or (contour_area > 0 and clipped_area < contour_area * 0.40)
        )

        if is_fragmented:
            merged = self.preserve_nearby_object_parts(
                source_mask=binary,
                main_contour=contour,
            )
            merged = self.remove_small_components(merged)
            if int(np.sum(merged > 0)) > clipped_area:
                return self._finalize_object_mask(color_image, merged)

        if clipped_area > 0:
            result = clipped
            if features["solidity"] > 0.45:
                result = self.fill_holes(result)
            return self._finalize_object_mask(color_image, result)

        return self._finalize_object_mask(color_image, self.fill_holes(filled))

    def remove_shadow_from_mask(
        self,
        image: np.ndarray,
        object_mask: np.ndarray,
    ) -> np.ndarray:
        """Drop shadow pixels that resemble the background more than the object core."""
        binary = np.where(object_mask > 0, 255, 0).astype(np.uint8)
        foreground = binary > 0
        foreground_count = int(np.sum(foreground))

        if foreground_count == 0:
            return binary

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        lightness = lab[:, :, 0]

        _, background_color, _, _ = self.background_lab_stats(image)
        background_color = np.asarray(background_color, dtype=np.float32)

        mask_lightness = lightness[foreground]
        mask_saturation = saturation[foreground]
        mask_lab = lab[foreground]
        foreground_indices = np.argwhere(foreground)

        distance_to_bg = np.linalg.norm(mask_lab - background_color, axis=1)
        core_score = mask_saturation + mask_lightness * 0.35 + distance_to_bg * 0.45
        core_threshold = float(np.percentile(core_score, 62))
        core_pixels = mask_lab[core_score >= core_threshold]

        if len(core_pixels) >= 8:
            core_color = np.median(core_pixels, axis=0)
        else:
            core_color = np.median(mask_lab, axis=0)

        core_lightness = float(core_color[0])
        background_lightness = float(background_color[0])

        distance_to_core = np.linalg.norm(mask_lab - core_color, axis=1)
        distance_to_core = np.maximum(distance_to_core, 1.0)

        ratio = self.config.shadow_bg_distance_ratio
        shadow_like = (
            (distance_to_bg < distance_to_core * ratio)
            & (
                (mask_lightness < core_lightness - self.config.shadow_l_delta)
                | (mask_saturation < self.config.shadow_max_saturation)
            )
        ) | (
            (distance_to_bg < distance_to_core * 0.72)
            & (mask_lightness < background_lightness + self.config.shadow_bg_l_margin)
            & (distance_to_bg < float(np.percentile(distance_to_bg, 35)))
        )

        core_seed = np.zeros_like(binary)
        core_indices = foreground_indices[core_score >= core_threshold]
        if len(core_indices) > 0:
            core_seed[core_indices[:, 0], core_indices[:, 1]] = 255
        else:
            core_seed = binary.copy()

        protect_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        )
        protected_core = cv2.dilate(
            core_seed,
            protect_kernel,
            iterations=self.config.shadow_core_dilate_iterations,
        )

        distance_source = np.where(
            protected_core > 0,
            0,
            np.where(foreground, 255, 255),
        ).astype(np.uint8)
        distance_from_core = cv2.distanceTransform(distance_source, cv2.DIST_L2, 5)
        core_distance_at_pixels = distance_from_core[foreground]

        shadow_like = shadow_like & (
            core_distance_at_pixels >= self.config.shadow_min_core_distance
        )

        keep = foreground.copy()
        shadow_indices = foreground_indices[shadow_like]
        keep[shadow_indices[:, 0], shadow_indices[:, 1]] = False

        cleaned = np.zeros_like(binary)
        cleaned[keep] = 255

        if int(np.sum(cleaned > 0)) < int(foreground_count * self.config.shadow_min_keep_ratio):
            return binary

        cleaned = self.keep_largest_component(cleaned)
        cleaned = self.remove_small_components(cleaned)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
        if int(np.sum(opened > 0)) >= int(np.sum(cleaned > 0) * 0.55):
            cleaned = opened

        return cleaned

    def _finalize_object_mask(
        self,
        color_image: np.ndarray | None,
        object_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply post-processing such as shadow removal to a single-object mask."""
        if color_image is None:
            return object_mask

        cleaned = self.remove_shadow_from_mask(color_image, object_mask)
        if int(np.sum(cleaned > 0)) > 0:
            return cleaned

        return object_mask

    def mask_noise_penalty(self, mask: np.ndarray) -> float:
        """Higher values mean a noisier / over-segmented candidate mask."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        frame_area = binary.size
        foreground_ratio = np.sum(binary > 0) / frame_area

        penalty = 0.0
        if foreground_ratio > 0.18:
            penalty += (foreground_ratio - 0.18) * 6.0

        component_count = cv2.connectedComponents(binary)[0] - 1
        if component_count > 6:
            penalty += min(1.5, (component_count - 6) * 0.15)

        return penalty

    def fill_by_geodesic_dilation(
        self,
        seeds: np.ndarray,
        region: np.ndarray,
        iterations: int = 28,
    ) -> np.ndarray:
        """Grow *seeds* inside *region* to fill multi-color solid objects."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        current = cv2.bitwise_and(seeds, region)

        for _ in range(iterations):
            dilated = cv2.dilate(current, kernel)
            current = cv2.bitwise_and(dilated, region)

        return self.fill_holes(current)

    def _region_from_seeds(
        self,
        seeds: np.ndarray,
        pad_ratio: float = 0.14,
    ) -> np.ndarray | None:
        """Build a compact ROI from seed pixels (convex hull + padding)."""
        points = cv2.findNonZero(seeds)
        if points is None:
            return None

        hull = cv2.convexHull(points)
        region = np.zeros(seeds.shape[:2], dtype=np.uint8)
        cv2.drawContours(region, [hull], -1, 255, cv2.FILLED)

        pad = max(8, int(pad_ratio * max(region.shape[:2])))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (pad * 2 + 1, pad * 2 + 1),
        )
        return cv2.dilate(region, kernel, iterations=1)

    def grabcut_refine_contour(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        seeds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Use GrabCut inside the contour ROI for multi-color solid objects."""
        height, width = image.shape[:2]
        x, y, box_w, box_h = cv2.boundingRect(contour)
        pad = max(12, int(0.08 * max(box_w, box_h)))

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + box_w + pad)
        y2 = min(height, y + box_h + pad)

        roi = image[y1:y2, x1:x2].copy()
        roi_height, roi_width = roi.shape[:2]

        region = np.zeros((roi_height, roi_width), dtype=np.uint8)
        if seeds is not None:
            seed_roi = seeds[y1:y2, x1:x2]
            seed_region = self._region_from_seeds(seed_roi)
            if seed_region is not None:
                region = seed_region

        if int(np.sum(region > 0)) == 0:
            shifted = contour.copy()
            shifted[:, :, 0] -= x1
            shifted[:, :, 1] -= y1
            cv2.drawContours(region, [shifted], -1, 255, cv2.FILLED)

        gc_mask = np.full((roi_height, roi_width), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[region == 0] = cv2.GC_BGD

        if seeds is not None:
            seed_roi = seeds[y1:y2, x1:x2]
            gc_mask[seed_roi > 0] = cv2.GC_FGD

        moments = cv2.moments(region)
        if moments["m00"] > 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            cv2.circle(gc_mask, (center_x, center_y), 4, cv2.GC_FGD, -1)

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                roi,
                gc_mask,
                None,
                background_model,
                foreground_model,
                5,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            return np.zeros((height, width), dtype=np.uint8)

        refined = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        refined = cv2.bitwise_and(refined, region)
        refined = self.fill_holes(refined)
        refined = self.keep_largest_component(refined)

        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = refined
        return full_mask

    def preserve_nearby_object_parts(
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
        candidate = self.remove_border_connected_components(candidate)

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
            not_large_background = not self.looks_like_background_region(
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

    def find_largest_reasonable_contour(self, mask: np.ndarray) -> np.ndarray | None:
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

    def find_best_contour(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        scored_contours = []
        for contour in contours:
            score = self.score_contour(contour, mask.shape[:2])
            if score > -0.5:
                scored_contours.append((score, contour))

        if scored_contours:
            scored_contours.sort(key=lambda item: item[0], reverse=True)
            return scored_contours[0][1]

        return self.find_largest_reasonable_contour(mask)

    def find_object_contours(
        self,
        mask: np.ndarray,
        max_objects: int = 8,
        min_iou_overlap: float = 0.45,
    ) -> list[np.ndarray]:
        """Return up to *max_objects* non-overlapping object contours, best score first."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        scored_contours: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            score = self.score_contour(contour, mask.shape[:2])
            if score > -0.5:
                scored_contours.append((score, contour))

        scored_contours.sort(key=lambda item: item[0], reverse=True)

        selected: list[np.ndarray] = []
        selected_boxes: list[tuple[int, int, int, int]] = []

        for _, contour in scored_contours:
            if len(selected) >= max_objects:
                break

            box = cv2.boundingRect(contour)
            if any(
                self._bbox_iou(box, existing) > min_iou_overlap
                for existing in selected_boxes
            ):
                continue

            selected.append(contour)
            selected_boxes.append(box)

        return selected

    @staticmethod
    def _bbox_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Intersection-over-union for axis-aligned bounding boxes (x, y, w, h)."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    def draw_contours_debug(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        max_objects: int = 8,
    ) -> np.ndarray:
        """Draw numbered bounding boxes for all scored contours (debug overlay)."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        binary = self.remove_border_connected_components(binary)
        contours = self.find_object_contours(binary, max_objects=max_objects)

        output = image.copy()
        colors = (
            (0, 255, 255),
            (255, 128, 0),
            (0, 200, 255),
            (255, 0, 255),
            (0, 255, 128),
            (128, 128, 255),
            (255, 255, 0),
            (180, 105, 255),
        )

        for index, contour in enumerate(contours, start=1):
            color = colors[(index - 1) % len(colors)]
            cv2.drawContours(output, [contour], -1, color, 2)

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
            cv2.putText(
                output,
                f"#{index}",
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            output,
            f"contours: {len(contours)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return output

    def score_contour(
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

        if self.is_obvious_artifact(
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

        border_penalty = self.calculate_border_penalty(
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

    def is_obvious_artifact(
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

    def looks_like_background_region(
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

    def calculate_border_penalty(
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

    def estimate_basic_features(self, contour: np.ndarray) -> dict[str, float]:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        return {"extent": float(extent), "solidity": float(solidity)}

    def calculate_edge_density_from_mask(
        self, source_mask: np.ndarray, object_mask: np.ndarray,
    ) -> float:
        object_pixels = object_mask > 0
        object_area = int(np.sum(object_pixels))

        if object_area == 0:
            return 0.0

        source_pixels = source_mask > 0
        active_pixels = int(np.sum(source_pixels & object_pixels))
        return active_pixels / object_area
