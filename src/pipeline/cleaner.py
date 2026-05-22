"""Mask morphological cleaning and multi-object refinement."""

import cv2
import numpy as np

from src.pipeline.mask_ops import MaskOps
from src.config import AppConfig


class MaskCleaner:
    """Clean a segmentation mask while preserving multiple foreground objects."""

    def __init__(self, config: AppConfig, mask_ops: MaskOps) -> None:
        self.config = config
        self.mask_ops = mask_ops

    def clean(self, mask: np.ndarray) -> np.ndarray:
        """Clean foreground mask; keep up to *max_objects* separate objects."""
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

        contours = self.mask_ops.find_object_contours(
            opened,
            max_objects=self.config.max_objects,
        )

        if not contours:
            contours = self.mask_ops.find_object_contours(
                binary,
                max_objects=self.config.max_objects,
            )

        if not contours:
            return opened

        combined = np.zeros_like(binary)
        for main_contour in contours:
            object_clean = self._clean_contour_region(binary, main_contour)
            combined = cv2.bitwise_or(combined, object_clean)

        combined = self.mask_ops.remove_border_noise(combined)
        combined = self.mask_ops.remove_small_components(combined)

        if int(np.sum(combined > 0)) == 0:
            return opened

        return combined

    def _clean_contour_region(
        self,
        binary: np.ndarray,
        main_contour: np.ndarray,
    ) -> np.ndarray:
        """Apply per-object hole filling and fragment preservation."""
        return self.mask_ops.extract_object_mask_from_contour(binary, main_contour)
