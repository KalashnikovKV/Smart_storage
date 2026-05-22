"""Contrast enhancement: CLAHE in LAB, gamma, light Gaussian blur."""

import cv2
import numpy as np

from src.config import AppConfig


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction via LUT."""
    if gamma <= 0:
        return image
    inverse_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inverse_gamma * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def enhance(image: np.ndarray, config: AppConfig) -> np.ndarray:
    """Enhance image using CLAHE, gamma correction and light blur."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_size,
    )
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced = apply_gamma(enhanced, config.gamma_value)
    enhanced = cv2.GaussianBlur(enhanced, config.blur_kernel, 0)

    return enhanced
