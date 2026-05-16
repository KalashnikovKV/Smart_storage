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
