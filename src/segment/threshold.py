"""Classical threshold / edge / color-distance segmentation."""

import cv2
import numpy as np

from src.clean.mask_ops import MaskOps
from src.config import AppConfig


class ThresholdSegmenter:
    """Multi-candidate threshold segmentation (Otsu, adaptive, edge, LAB distance)."""

    def __init__(self, config: AppConfig, mask_ops: MaskOps) -> None:
        self.config = config
        self.mask_ops = mask_ops

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

    def _prepare_candidate_mask(self, mask: np.ndarray) -> np.ndarray:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        binary = self.mask_ops.remove_border_connected_components(binary)
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
            score = self.mask_ops.score_contour(contour, mask.shape[:2])
            if score > best_score:
                best_contour = contour
                best_score = score

        if best_contour is None:
            return None, -1.0

        object_mask = np.zeros_like(mask)
        cv2.drawContours(object_mask, [best_contour], -1, 255, cv2.FILLED)

        features = self.mask_ops.estimate_basic_features(best_contour)
        edge_density = self.mask_ops.calculate_edge_density_from_mask(mask, object_mask)

        is_fragmented_or_cable_like = (
            edge_density >= 0.055
            or features["solidity"] < 0.85
            or features["extent"] < 0.60
        )

        if is_fragmented_or_cable_like:
            preserved_mask = self.mask_ops.preserve_nearby_object_parts(
                source_mask=mask,
                main_contour=best_contour,
            )
            if int(np.sum(preserved_mask > 0)) > int(np.sum(object_mask > 0)):
                object_mask = preserved_mask

        return object_mask, best_score

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
        closed = self.mask_ops.remove_border_connected_components(closed)

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
        mask = self.mask_ops.remove_border_connected_components(mask)

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
        mask = self.mask_ops.remove_border_connected_components(mask)

        return mask
