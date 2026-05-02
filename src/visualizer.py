"""Visualization module using OpenCV highgui."""

import cv2
import numpy as np

from src.models import DetectionResult, PipelineResult


class Visualizer:
    """Displays pipeline results using OpenCV windows."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    TEXT_BG = (0, 0, 0)  # Black background for text
    PANEL_SIZE = (320, 240)  # Width x Height for each panel

    def draw_detection(
        self, image: np.ndarray, detection: DetectionResult
    ) -> np.ndarray:
        """Draw bounding box and labels on image.

        Args:
            image: BGR image to draw on.
            detection: DetectionResult with bbox and properties.

        Returns:
            Image with drawn annotations.
        """
        result = image.copy()
        x, y, w, h = detection.bbox

        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

        # Draw labels
        label = f"{detection.primary_color} | {detection.size_category}"
        label_size = cv2.getTextSize(
            label, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
        )[0]
        cv2.rectangle(
            result,
            (x, y - label_size[1] - 10),
            (x + label_size[0] + 4, y),
            self.TEXT_BG,
            cv2.FILLED,
        )
        cv2.putText(
            result, label, (x + 2, y - 5),
            self.FONT, self.FONT_SCALE, self.TEXT_COLOR, self.FONT_THICKNESS,
        )
        return result

    def create_dashboard(self, result: PipelineResult) -> np.ndarray:
        """Compose all pipeline stages into a single dashboard image.

        Layout: 2 rows x 3 columns
        Row 1: Original | Enhanced | Segmentation Mask
        Row 2: Cleaned Mask | Detection | Decision

        Args:
            result: PipelineResult with all intermediate outputs.

        Returns:
            Single BGR image with all panels.
        """
        pw, ph = self.PANEL_SIZE

        # Prepare panels
        original = cv2.resize(result.original, (pw, ph))
        enhanced = cv2.resize(result.enhanced, (pw, ph))

        # Convert masks to BGR for display
        mask_bgr = cv2.cvtColor(
            cv2.resize(result.mask, (pw, ph)), cv2.COLOR_GRAY2BGR
        )
        cleaned_bgr = cv2.cvtColor(
            cv2.resize(result.cleaned_mask, (pw, ph)), cv2.COLOR_GRAY2BGR
        )

        # Detection panel with bounding box
        detection_img = self.draw_detection(result.enhanced, result.detection)
        detection_panel = cv2.resize(detection_img, (pw, ph))

        # Decision panel
        decision_panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        self._draw_decision_text(decision_panel, result)

        # Add labels to each panel
        panels = [
            ("1. Original", original),
            ("2. Enhanced", enhanced),
            ("3. Segmentation", mask_bgr),
            ("4. Cleaned Mask", cleaned_bgr),
            ("5. Detection", detection_panel),
            ("6. Decision", decision_panel),
        ]
        labeled = []
        for title, panel in panels:
            labeled.append(self._add_title(panel, title))

        # Compose: 2 rows x 3 columns
        row1 = np.hstack(labeled[:3])
        row2 = np.hstack(labeled[3:])
        dashboard = np.vstack([row1, row2])
        return dashboard

    def _draw_decision_text(
        self, panel: np.ndarray, result: PipelineResult
    ) -> None:
        """Draw decision text on the decision panel."""
        d = result.decision
        y_offset = 30

        conf_pct = f"{d.confidence * 100:.0f}%"

        # Encode to ASCII-safe for OpenCV (replace non-ASCII with '?')
        def safe(text: str) -> str:
            return text.encode("ascii", errors="replace").decode("ascii")

        if d.is_unknown:
            line1 = "Unknown Object"
            closest = safe(d.closest_match) if d.closest_match else "N/A"
            line2 = f"Closest: {closest}"
        else:
            line1 = safe(d.category)
            line2 = f"Confidence: {conf_pct}"

        cv2.putText(
            panel, line1, (10, y_offset),
            self.FONT, 0.6, (0, 255, 0), 1,
        )
        cv2.putText(
            panel, line2, (10, y_offset + 30),
            self.FONT, 0.5, self.TEXT_COLOR, 1,
        )
        cv2.putText(
            panel, f"Color: {d.color}", (10, y_offset + 60),
            self.FONT, 0.45, self.TEXT_COLOR, 1,
        )
        cv2.putText(
            panel, f"Size: {d.size}", (10, y_offset + 85),
            self.FONT, 0.45, self.TEXT_COLOR, 1,
        )
        cv2.putText(
            panel, f"Method: {d.method_used}", (10, y_offset + 110),
            self.FONT, 0.45, self.TEXT_COLOR, 1,
        )
        cv2.putText(
            panel, f"Time: {result.processing_time_ms:.0f}ms", (10, y_offset + 135),
            self.FONT, 0.45, (0, 200, 200), 1,
        )

        # Shape features
        det = result.detection
        cv2.putText(
            panel,
            f"Shape: {det.shape_category} (circ={det.circularity:.2f})",
            (10, y_offset + 160),
            self.FONT, 0.4, (200, 150, 0), 1,
        )

        # HSV vs K-means comparison
        cv2.putText(
            panel,
            f"HSV: {det.color_hsv.name} ({det.color_hsv.confidence:.0%})",
            (10, y_offset + 185),
            self.FONT, 0.4, (200, 200, 0), 1,
        )
        cv2.putText(
            panel,
            f"KMeans: {det.color_kmeans.name} ({det.color_kmeans.confidence:.0%})",
            (10, y_offset + 205),
            self.FONT, 0.4, (200, 200, 0), 1,
        )

    def _add_title(self, panel: np.ndarray, title: str) -> np.ndarray:
        """Add a title bar to the top of a panel."""
        h, w = panel.shape[:2]
        title_bar = np.zeros((25, w, 3), dtype=np.uint8)
        cv2.putText(
            title_bar, title, (5, 18),
            self.FONT, 0.5, (0, 255, 255), 1,
        )
        return np.vstack([title_bar, panel])

    WINDOW_NAME = "Smart Storage - Pipeline Dashboard"

    def show_pipeline(self, result: PipelineResult) -> None:
        """Display the full pipeline dashboard.

        Args:
            result: PipelineResult to display.
        """
        dashboard = self.create_dashboard(result)
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 960, 530)
        cv2.imshow(self.WINDOW_NAME, dashboard)
        cv2.waitKey(1)  # Force the window to render on Linux/X11
