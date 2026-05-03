"""Visualization module for pipeline results."""

from pathlib import Path

import cv2
import numpy as np

from src import config
from src.models import DetectionResult, PipelineResult


class Visualizer:
    """Displays and saves pipeline results."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    BOX_COLOR = (0, 255, 0) # Green
    TEXT_COLOR = (255, 255, 255) # White
    TEXT_BG = (0, 0, 0) # Black background for text
    PANEL_SIZE = (360, 260)   # Width x Height for each panel

    def draw_detection(
        self,
        image: np.ndarray,
        detection: DetectionResult,
    ) -> np.ndarray:
        """Draw a single detection."""
        result = image.copy()
        self._draw_detection_on_image(result, detection)
        return result

    def draw_all_detections(self, image: np.ndarray, result: PipelineResult) -> np.ndarray:
        """Draw all detected objects and final decisions."""
        output = image.copy()

        decision_by_id = {
            decision.object_id: decision
            for decision in result.decisions
        }

        for detection in result.detections:
            decision = decision_by_id.get(detection.object_id)
            self._draw_detection_on_image(output, detection, decision)

        return output

    def create_dashboard(self, result: PipelineResult) -> np.ndarray:
        """Create a 2x3 dashboard with all pipeline stages."""
        panel_width, panel_height = self.PANEL_SIZE
        # Resize and pad all images to fit the panels
        original = self._resize_with_padding(result.original, (panel_width, panel_height))
        enhanced = self._resize_with_padding(result.enhanced, (panel_width, panel_height))
        # For masks, convert to BGR and pad to avoid distortion
        mask = self._resize_with_padding(result.mask, (panel_width, panel_height))
        cleaned = self._resize_with_padding(result.cleaned_mask, (panel_width, panel_height))
        # Convert masks to BGR for visualization
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        # Draw detections on the enhanced image for the detection panel
        detection_image = self.draw_all_detections(result.enhanced, result)
        detection_panel = self._resize_with_padding(
            detection_image,
            (panel_width, panel_height),
        )
        # Create a blank panel for the decision text
        decision_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        self._draw_decision_text(decision_panel, result)
        # Combine panels into a 2x3 grid
        panels = [
            ("1. Original", original),
            ("2. Enhanced", enhanced),
            ("3. Segmentation", mask_bgr),
            ("4. Cleaned Mask", cleaned_bgr),
            ("5. Detection", detection_panel),
            ("6. Decision", decision_panel),
        ]

        labeled_panels = [self._add_title(panel, title) for title, panel in panels]
        # Stack panels into a 2x3 grid
        row1 = np.hstack(labeled_panels[:3])
        row2 = np.hstack(labeled_panels[3:])

        return np.vstack([row1, row2])

    def save_pipeline_outputs(self, result: PipelineResult, base_name: str) -> None:
        """Save all required output images for one test image."""
        stages_dir = Path(config.STAGES_DIR)

        folders = {
            "original": stages_dir / "original",
            "enhanced": stages_dir / "enhanced",
            "mask": stages_dir / "mask",
            "cleaned_mask": stages_dir / "cleaned_mask",
            "detection": stages_dir / "detection",
            "dashboard": stages_dir / "dashboard",
        }

        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)

        detection_image = self.draw_all_detections(result.enhanced, result)
        dashboard = self.create_dashboard(result)

        cv2.imwrite(str(folders["original"] / f"{base_name}_original.jpg"), result.original)
        cv2.imwrite(str(folders["enhanced"] / f"{base_name}_enhanced.jpg"), result.enhanced)
        cv2.imwrite(str(folders["mask"] / f"{base_name}_mask.jpg"), result.mask)
        cv2.imwrite(str(folders["cleaned_mask"] / f"{base_name}_cleaned_mask.jpg"), result.cleaned_mask)
        cv2.imwrite(str(folders["detection"] / f"{base_name}_detection.jpg"), detection_image)
        cv2.imwrite(str(folders["dashboard"] / f"{base_name}_dashboard.jpg"), dashboard)

    def show_pipeline(self, result: PipelineResult) -> None:
        """Display dashboard."""
        dashboard = self.create_dashboard(result)
        cv2.imshow("Smart Storage — Pipeline Dashboard", dashboard)

    def _draw_detection_on_image(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        decision=None,
    ) -> None:
        """Draw bounding box and label directly on image."""
        x, y, w, h = detection.bbox

        cv2.rectangle(image, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

        if decision is None:
            label = (
                f"#{detection.object_id}: "
                f"{detection.primary_color}, {detection.size_category}"
            )
        else:
            label = (
                f"#{detection.object_id}: {decision.category} | "
                f"{decision.color}, {decision.size} | "
                f"{decision.confidence:.0%}"
            )

        label_size = cv2.getTextSize(
            label,
            self.FONT,
            self.FONT_SCALE,
            self.FONT_THICKNESS,
        )[0]

        label_x = x
        label_y = max(y - label_size[1] - 10, 0)

        cv2.rectangle(
            image,
            (label_x, label_y),
            (label_x + label_size[0] + 8, label_y + label_size[1] + 12),
            self.TEXT_BG,
            cv2.FILLED,
        )

        cv2.putText(
            image,
            label,
            (label_x + 4, label_y + label_size[1] + 6),
            self.FONT,
            self.FONT_SCALE,
            self.TEXT_COLOR,
            self.FONT_THICKNESS,
        )

    def _draw_decision_text(
        self, panel: np.ndarray, result: PipelineResult
    ) -> None:
        """Draw decision text on the decision panel."""
        d = result.decision
        det = result.detection
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
            panel,
            line1,
            (10, y_offset),
            self.FONT,
            0.6,
            (0, 255, 0),
            1,
        )

        cv2.putText(
            panel,
            line2,
            (10, y_offset + 30),
            self.FONT,
            0.5,
            self.TEXT_COLOR,
            1,
        )

        cv2.putText(
            panel,
            f"Color: {d.color}",
            (10, y_offset + 60),
            self.FONT,
            0.45,
            self.TEXT_COLOR,
            1,
        )

        cv2.putText(
            panel,
            f"Visual size: {d.size}",
            (10, y_offset + 85),
            self.FONT,
            0.45,
            self.TEXT_COLOR,
            1,
        )

        cv2.putText(
            panel,
            f"Area: {det.area_ratio * 100:.1f}% of image",
            (10, y_offset + 110),
            self.FONT,
            0.42,
            self.TEXT_COLOR,
            1,
        )

        cv2.putText(
            panel,
            f"BBox: {det.bbox_width_ratio * 100:.0f}% x {det.bbox_height_ratio * 100:.0f}%",
            (10, y_offset + 132),
            self.FONT,
            0.42,
            self.TEXT_COLOR,
            1,
        )

        cv2.putText(
            panel,
            f"Time: {result.processing_time_ms:.0f}ms",
            (10, y_offset + 155),
            self.FONT,
            0.45,
            (0, 200, 200),
            1,
        )

        cv2.putText(
            panel,
            f"Shape: {det.shape_category}",
            (10, y_offset + 180),
            self.FONT,
            0.4,
            (200, 150, 0),
            1,
        )

        cv2.putText(
            panel,
            f"HSV: {det.color_hsv.name} ({det.color_hsv.confidence:.0%})",
            (10, y_offset + 202),
            self.FONT,
            0.4,
            (200, 200, 0),
            1,
        )

        cv2.putText(
            panel,
            f"KMeans: {det.color_kmeans.name} ({det.color_kmeans.confidence:.0%})",
            (10, y_offset + 222),
            self.FONT,
            0.4,
            (200, 200, 0),
            1,
        )
    def _resize_with_padding(
        self,
        image: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Resize without distortion and pad to target size."""
        target_width, target_height = target_size
        image_height, image_width = image.shape[:2]

        if image_height == 0 or image_width == 0:
            if len(image.shape) == 2:
                return np.zeros((target_height, target_width), dtype=np.uint8)

            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        scale = min(target_width / image_width, target_height / image_height)

        new_width = max(1, int(image_width * scale))
        new_height = max(1, int(image_height * scale))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        if len(image.shape) == 2:
            canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        else:
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width,
        ] = resized

        return canvas

    def _add_title(self, panel: np.ndarray, title: str) -> np.ndarray:
        """Add title bar."""
        _, width = panel.shape[:2]
        title_bar = np.zeros((25, width, 3), dtype=np.uint8)

        cv2.putText(
            title_bar,
            title,
            (5, 18),
            self.FONT,
            0.5,
            (0, 255, 255),
            1,
        )

        return np.vstack([title_bar, panel])

    def _safe(self, text: str) -> str:
        """Make text safe for OpenCV rendering."""
        return text.encode("ascii", errors="replace").decode("ascii")