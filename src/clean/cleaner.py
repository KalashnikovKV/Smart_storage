"""Mask morphological cleaning and single-object refinement."""

import cv2
import numpy as np

from src.clean.mask_ops import MaskOps
from src.config import AppConfig


class MaskCleaner:
    """Clean a segmentation mask into a stable single-object mask."""

    def __init__(self, config: AppConfig, mask_ops: MaskOps) -> None:
        self.config = config
        self.mask_ops = mask_ops

    def clean(self, mask: np.ndarray) -> np.ndarray:
        """Clean selected object mask."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        binary = self.mask_ops.remove_border_connected_components(binary)

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

        main_contour = self.mask_ops.find_best_contour(opened)

        if main_contour is None:
            main_contour = self.mask_ops.find_best_contour(binary)

        if main_contour is None:
            return opened

        main_features = self.mask_ops.estimate_basic_features(main_contour)
        main_mask = np.zeros_like(binary)
        cv2.drawContours(main_mask, [main_contour], -1, 255, cv2.FILLED)

        edge_density = self.mask_ops.calculate_edge_density_from_mask(binary, main_mask)

        is_thin_or_fragmented_object = (
            edge_density >= 0.08
            or main_features["solidity"] < 0.78
            or main_features["extent"] < 0.55
        )

        if is_thin_or_fragmented_object:
            clean_mask = self.mask_ops.preserve_nearby_object_parts(
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
                clean_mask = self.mask_ops.fill_holes(clean_mask)

        clean_mask = self.mask_ops.remove_border_noise(clean_mask)

        if int(np.sum(clean_mask > 0)) == 0:
            return main_mask

        return clean_mask
