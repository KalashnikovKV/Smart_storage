"""Minimal BGR image validation for pipeline input."""

import numpy as np


def is_valid_bgr_uint8(image: np.ndarray | None) -> bool:
    """Return True if *image* is a non-empty uint8 HxWx3 array."""
    if image is None or image.size == 0:
        return False
    return (
        image.ndim == 3
        and image.shape[2] == 3
        and image.dtype == np.uint8
    )
