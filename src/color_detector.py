"""Color detection module with HSV analysis and K-means clustering."""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src import config
from src.models import ColorResult


class ColorDetector:
    """Detects dominant color of an object using two methods."""

    def __init__(self) -> None:
        self.hsv_ranges = config.HSV_RANGES
        self.kmeans_clusters = config.KMEANS_CLUSTERS
        self.kmeans_max_iter = config.KMEANS_MAX_ITER
        self.kmeans_n_init = config.KMEANS_N_INIT

    def detect_hsv(self, image: np.ndarray, mask: np.ndarray) -> ColorResult:
        """Detect dominant color using HSV range analysis.

        Args:
            image: BGR image.
            mask: Binary mask of the object region.

        Returns:
            ColorResult with detected color name and confidence.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total_pixels = int(np.sum(mask > 0))
        if total_pixels == 0:
            return ColorResult(name="unknown", confidence=0.0, method="hsv")

        # Extract H, S, V channels within mask
        h_vals = hsv[:, :, 0][mask > 0]
        s_vals = hsv[:, :, 1][mask > 0]
        v_vals = hsv[:, :, 2][mask > 0]

        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))

        # Check achromatic colors first (white, black, gray) by S/V
        if mean_v > 200 and mean_s < 50:
            count = int(np.sum((s_vals < 50) & (v_vals > 200)))
            return ColorResult(
                name="white",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )
        if mean_v < 50:
            count = int(np.sum(v_vals < 50))
            return ColorResult(
                name="black",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )
        if mean_s < 50 and 150 <= mean_v <= 200:
            count = int(np.sum((s_vals < 50) & (v_vals >= 150) & (v_vals <= 200)))
            return ColorResult(
                name="silver",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )
        if mean_s < 50:
            count = int(np.sum((s_vals < 50) & (v_vals >= 50) & (v_vals < 200)))
            return ColorResult(
                name="gray",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )

        # Chromatic colors — find dominant hue
        best_color = "unknown"
        best_count = 0
        chromatic_ranges = {
            k: v for k, v in self.hsv_ranges.items()
            if k not in ("white", "black", "gray", "silver")
        }
        for color_name, ranges in chromatic_ranges.items():
            h_min, h_max = ranges["h"]
            s_min, s_max = ranges["s"]
            v_min, v_max = ranges["v"]
            in_range = (
                (h_vals >= h_min) & (h_vals <= h_max)
                & (s_vals >= s_min) & (s_vals <= s_max)
                & (v_vals >= v_min) & (v_vals <= v_max)
            )
            count = int(np.sum(in_range))
            if count > best_count:
                best_count = count
                best_color = color_name.replace("_low", "").replace("_high", "")

        confidence = best_count / total_pixels if total_pixels > 0 else 0.0
        return ColorResult(
            name=best_color,
            confidence=confidence,
            method="hsv",
            rgb=self._mean_rgb(image, mask),
        )

    def detect_kmeans(self, image: np.ndarray, mask: np.ndarray) -> ColorResult:
        """Detect dominant color using K-means clustering.

        Args:
            image: BGR image.
            mask: Binary mask of the object region.

        Returns:
            ColorResult with detected color name and confidence.
        """
        pixels = image[mask > 0].reshape(-1, 3).astype(np.float32)
        if len(pixels) < self.kmeans_clusters:
            return ColorResult(name="unknown", confidence=0.0, method="kmeans")

        kmeans = KMeans(
            n_clusters=self.kmeans_clusters,
            max_iter=self.kmeans_max_iter,
            n_init=self.kmeans_n_init,
            random_state=42,
        )
        kmeans.fit(pixels)

        # Find the largest cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        dominant_center = kmeans.cluster_centers_[dominant_idx].astype(np.uint8)
        dominant_count = int(np.max(counts))

        # Convert BGR center to HSV for color naming
        bgr_pixel = dominant_center.reshape(1, 1, 3)
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
        color_name = self._hsv_to_name(hsv_pixel)

        rgb = (int(dominant_center[2]), int(dominant_center[1]), int(dominant_center[0]))
        confidence = dominant_count / len(pixels)

        return ColorResult(
            name=color_name,
            confidence=confidence,
            method="kmeans",
            rgb=rgb,
        )

    def detect(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """Run both detection methods and return combined result.

        Args:
            image: BGR image.
            mask: Binary mask of the object region.

        Returns:
            Dict with keys: 'hsv', 'kmeans', 'primary_color', 'primary_confidence'.
        """
        hsv_result = self.detect_hsv(image, mask)
        kmeans_result = self.detect_kmeans(image, mask)

        # Compare methods
        # Treat blue/black as equivalent (dark objects often flip between them)
        DARK_COLORS = {"black", "blue", "gray"}
        hsv_name = hsv_result.name
        kmeans_name = kmeans_result.name

        names_agree = (hsv_name == kmeans_name) or (
            hsv_name in DARK_COLORS and kmeans_name in DARK_COLORS
        )

        if names_agree:
            # Pick the more confident one as primary
            if hsv_result.confidence >= kmeans_result.confidence:
                primary_color = kmeans_name  # prefer kmeans name when dark
            else:
                primary_color = kmeans_name
            primary_confidence = max(hsv_result.confidence, kmeans_result.confidence)
            method_used = "combined"
        else:
            if hsv_result.confidence >= kmeans_result.confidence:
                primary_color = hsv_name
                primary_confidence = hsv_result.confidence * 0.85
                method_used = "hsv"
            else:
                primary_color = kmeans_name
                primary_confidence = kmeans_result.confidence * 0.85
                method_used = "kmeans"

        return {
            "hsv": hsv_result,
            "kmeans": kmeans_result,
            "primary_color": primary_color,
            "primary_confidence": primary_confidence,
            "method_used": method_used,
        }

    def _hsv_to_name(self, hsv_pixel: np.ndarray) -> str:
        """Map a single HSV pixel to a color name."""
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

        if v < 50:
            return "black"
        if s < 50 and v > 200:
            return "white"
        if s < 50 and 150 <= v <= 200:
            return "silver"
        if s < 50:
            return "gray"

        # Chromatic
        if h <= 10 or h >= 170:
            return "red"
        if 100 <= h <= 130:
            return "blue"
        if 35 <= h <= 85:
            return "green"
        return "unknown"

    @staticmethod
    def _mean_rgb(image: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
        """Calculate mean RGB of masked region."""
        pixels = image[mask > 0]
        if len(pixels) == 0:
            return (0, 0, 0)
        mean_bgr = np.mean(pixels, axis=0).astype(int)
        return (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
