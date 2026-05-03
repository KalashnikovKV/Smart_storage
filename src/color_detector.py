"""Color detection module with HSV analysis and K-means clustering."""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src import config
from src.models import ColorResult


class ColorDetector:
    """Detects dominant color of an object using HSV and K-means."""

    def __init__(self) -> None:
        self.hsv_ranges = config.HSV_RANGES
        self.kmeans_clusters = config.KMEANS_CLUSTERS
        self.kmeans_max_iter = config.KMEANS_MAX_ITER
        self.kmeans_n_init = config.KMEANS_N_INIT
        self.max_kmeans_pixels = getattr(config, "KMEANS_MAX_PIXELS", 12000)

    def detect_hsv(self, image: np.ndarray, mask: np.ndarray) -> ColorResult:
        """Detect dominant color using HSV range analysis."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        valid_pixels = mask > 0
        total_pixels = int(np.sum(valid_pixels))

        if total_pixels == 0:
            return ColorResult(name="unknown", confidence=0.0, method="hsv")

        h_vals = hsv[:, :, 0][valid_pixels]
        s_vals = hsv[:, :, 1][valid_pixels]
        v_vals = hsv[:, :, 2][valid_pixels]

        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))

        if mean_v < 85:
            count = int(np.sum(v_vals < 100))
            return ColorResult(
                name="black",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )

        if mean_v < 125 and mean_s < 150:
            count = int(np.sum((v_vals < 125) & (s_vals < 150)))
            return ColorResult(
                name="black",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )

        if mean_v > 185 and mean_s < 60:
            count = int(np.sum((v_vals > 185) & (s_vals < 60)))
            return ColorResult(
                name="white",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )

        if mean_s < 70 and 135 <= mean_v <= 225:
            count = int(np.sum((s_vals < 70) & (v_vals >= 135) & (v_vals <= 225)))
            return ColorResult(
                name="silver",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )

        if mean_s < 90 and 75 <= mean_v < 210:
            count = int(np.sum((s_vals < 90) & (v_vals >= 75) & (v_vals < 210)))
            return ColorResult(
                name="gray",
                confidence=count / total_pixels,
                method="hsv",
                rgb=self._mean_rgb(image, mask),
            )

        best_color = "unknown"
        best_count = 0

        chromatic_ranges = {
            key: value
            for key, value in self.hsv_ranges.items()
            if key not in {"white", "black", "gray", "silver"}
        }

        for color_name, ranges in chromatic_ranges.items():
            h_min, h_max = ranges["h"]
            s_min, s_max = ranges["s"]
            v_min, v_max = ranges["v"]

            in_range = (
                (h_vals >= h_min)
                & (h_vals <= h_max)
                & (s_vals >= s_min)
                & (s_vals <= s_max)
                & (v_vals >= v_min)
                & (v_vals <= v_max)
            )

            count = int(np.sum(in_range))
            normalized_name = color_name.replace("_low", "").replace("_high", "")

            if count > best_count:
                best_count = count
                best_color = normalized_name

        confidence = best_count / total_pixels if total_pixels else 0.0

        return ColorResult(
            name=best_color,
            confidence=confidence,
            method="hsv",
            rgb=self._mean_rgb(image, mask),
        )

    def detect_kmeans(self, image: np.ndarray, mask: np.ndarray) -> ColorResult:
        """Detect dominant color using K-means clustering."""
        pixels = image[mask > 0].reshape(-1, 3).astype(np.float32)

        if len(pixels) < self.kmeans_clusters:
            return ColorResult(name="unknown", confidence=0.0, method="kmeans")

        if len(pixels) > self.max_kmeans_pixels:
            random_generator = np.random.default_rng(42)
            selected_indices = random_generator.choice(
                len(pixels),
                size=self.max_kmeans_pixels,
                replace=False,
            )
            pixels = pixels[selected_indices]

        kmeans = KMeans(
            n_clusters=self.kmeans_clusters,
            max_iter=self.kmeans_max_iter,
            n_init=self.kmeans_n_init,
            random_state=42,
        )
        kmeans.fit(pixels)
        # Find the largest cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_label = labels[np.argmax(counts)]
        dominant_center = kmeans.cluster_centers_[dominant_label].astype(np.uint8)
        dominant_count = int(np.max(counts))
        
        # Convert BGR center to HSV for color naming
        bgr_pixel = dominant_center.reshape(1, 1, 3)
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
        color_name = self._hsv_to_name(hsv_pixel)

        rgb = (
            int(dominant_center[2]),
            int(dominant_center[1]),
            int(dominant_center[0]),
        )

        confidence = dominant_count / len(pixels)

        return ColorResult(
            name=color_name,
            confidence=confidence,
            method="kmeans",
            rgb=rgb,
        )

    def detect(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """Run HSV and K-means color detection and combine the result."""
        hsv_result = self.detect_hsv(image, mask)
        kmeans_result = self.detect_kmeans(image, mask)

        mean_hsv = self._mean_hsv(image, mask)
        mean_h, mean_s, mean_v = mean_hsv

        hsv_name = hsv_result.name
        kmeans_name = kmeans_result.name

        dark_names = {"black", "gray", "blue"}
        neutral_names = {"white", "gray", "silver"}

        if mean_v < 90:
            primary_color = "black"
            primary_confidence = max(hsv_result.confidence, kmeans_result.confidence)
            method_used = "combined"

        elif mean_v < 125 and mean_s < 150 and {hsv_name, kmeans_name} & dark_names:
            primary_color = "black"
            primary_confidence = max(hsv_result.confidence, kmeans_result.confidence)
            method_used = "combined"

        elif hsv_name == "black" and kmeans_name in dark_names:
            primary_color = "black"
            primary_confidence = max(hsv_result.confidence, kmeans_result.confidence)
            method_used = "combined"

        elif hsv_name == kmeans_name:
            primary_color = hsv_name
            primary_confidence = max(hsv_result.confidence, kmeans_result.confidence)
            method_used = "combined"

        elif hsv_name in neutral_names and kmeans_name in neutral_names:
            if hsv_result.confidence >= kmeans_result.confidence:
                primary_color = hsv_name
            else:
                primary_color = kmeans_name

            primary_confidence = max(hsv_result.confidence, kmeans_result.confidence)
            method_used = "combined"

        elif hsv_name in dark_names and kmeans_name in dark_names:
            if kmeans_name == "blue" and mean_s >= 150 and mean_v >= 110:
                primary_color = "blue"
            elif hsv_name == "blue" and mean_s >= 150 and mean_v >= 110:
                primary_color = "blue"
            else:
                primary_color = "black"

            primary_confidence = max(hsv_result.confidence, kmeans_result.confidence)
            method_used = "combined"

        elif hsv_result.confidence >= kmeans_result.confidence:
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

        if v < 85:
            return "black"

        if v < 125 and s < 150:
            return "black"

        if s < 60 and v > 185:
            return "white"

        if s < 70 and 135 <= v <= 225:
            return "silver"

        if s < 90 and 75 <= v < 210:
            return "gray"

        if h <= 10 or h >= 170:
            return "red"

        if 11 <= h <= 24:
            if v < 170 and s < 180:
                return "brown"
            return "orange"

        if 25 <= h <= 34:
            return "yellow"

        if 35 <= h <= 85:
            return "green"

        if 90 <= h <= 135:
            return "blue"

        if 136 <= h <= 160:
            return "purple"

        return "unknown"

    @staticmethod
    def _mean_rgb(image: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
        """Calculate mean RGB of masked region."""
        pixels = image[mask > 0]

        if len(pixels) == 0:
            return (0, 0, 0)

        mean_bgr = np.mean(pixels, axis=0).astype(int)

        return (
            int(mean_bgr[2]),
            int(mean_bgr[1]),
            int(mean_bgr[0]),
        )

    @staticmethod
    def _mean_hsv(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        """Calculate mean HSV of masked region."""
        pixels_exist = mask > 0

        if int(np.sum(pixels_exist)) == 0:
            return (0.0, 0.0, 0.0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_mean = float(np.mean(hsv[:, :, 0][pixels_exist]))
        s_mean = float(np.mean(hsv[:, :, 1][pixels_exist]))
        v_mean = float(np.mean(hsv[:, :, 2][pixels_exist]))

        return (h_mean, s_mean, v_mean)