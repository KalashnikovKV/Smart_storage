"""Visualization module for pipeline results."""

from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig
from src.models import Decision, DetectionResult, PipelineResult


class WindowClosed(Exception):
    """Raised when the user closes the OpenCV window."""


class Visualizer:
    """Displays and saves pipeline results."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    BOX_COLOR = (0, 255, 0) # Green
    TEXT_COLOR = (255, 255, 255) # White
    TEXT_BG = (0, 0, 0) # Black background for text
    OBJECT_COLORS = (
        (0, 255, 255),
        (255, 128, 0),
        (0, 200, 255),
        (255, 0, 255),
        (0, 255, 128),
        (128, 128, 255),
        (255, 255, 0),
        (180, 105, 255),
    )
    PANEL_SIZE = (480, 360)   # Width x Height for each panel
    WINDOW_NAME = "Smart Storage — Pipeline Dashboard"

    def __init__(self) -> None:
        self._window_created = False
        self._screen_size: tuple[int, int] | None = None

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

    def build_colored_object_masks(
        self,
        object_masks: list[np.ndarray],
        detections: list[DetectionResult] | None = None,
        decisions: list | None = None,
    ) -> np.ndarray:
        """Render per-object binary masks as a single color-coded BGR image."""
        if not object_masks:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        height, width = object_masks[0].shape[:2]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        decision_by_id = {
            decision.object_id: decision
            for decision in (decisions or [])
        }

        for index, mask in enumerate(object_masks):
            color = self.OBJECT_COLORS[index % len(self.OBJECT_COLORS)]
            canvas[mask > 0] = color

            moments = cv2.moments(mask)
            if moments["m00"] <= 0:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            if detections and index < len(detections):
                object_id = detections[index].object_id
            else:
                object_id = index + 1

            decision = decision_by_id.get(object_id)
            label = f"#{object_id}: {decision.category}" if decision else f"#{object_id}"

            cv2.putText(
                canvas,
                label,
                (max(center_x - 60, 4), max(center_y, 24)),
                self.FONT,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

    def create_panel_placeholder(
        self,
        height: int,
        width: int,
        lines: tuple[str, ...],
    ) -> np.ndarray:
        """Empty panel with centered hint text."""
        panel = np.full((height, width, 3), 28, dtype=np.uint8)
        y = height // 2 - len(lines) * 16

        for line in lines:
            text_size = cv2.getTextSize(line, self.FONT, 0.65, 1)[0]
            x = max((width - text_size[0]) // 2, 8)
            cv2.putText(
                panel,
                line,
                (x, y),
                self.FONT,
                0.65,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            y += 32

        return panel

    def create_mask_placeholder(self, height: int, width: int) -> np.ndarray:
        """Empty mask panel shown before analysis completes."""
        return self.create_panel_placeholder(
            height,
            width,
            ("Object Masks", "", "Segmentation masks", "appear after", "analysis"),
        )

    def create_contour_placeholder(self, height: int, width: int) -> np.ndarray:
        """Empty contour panel shown before analysis completes."""
        return self.create_panel_placeholder(
            height,
            width,
            ("Contours", "", "Object outline on", "camera feed"),
        )

    def build_detection_panel(
        self,
        frame: np.ndarray,
        result: PipelineResult,
    ) -> np.ndarray:
        """Render the live frame with bounding boxes and category labels."""
        return self.draw_all_detections(frame, result)

    def build_contour_overlay_panel(
        self,
        frame: np.ndarray,
        result: PipelineResult,
    ) -> np.ndarray:
        """Draw object contours and labels on top of the camera frame."""
        output = frame.copy()
        decision_by_id = {
            decision.object_id: decision
            for decision in result.decisions
        }

        for detection in result.detections:
            decision = decision_by_id.get(detection.object_id)
            color = self.OBJECT_COLORS[(detection.object_id - 1) % len(self.OBJECT_COLORS)]

            if detection.contour is not None and len(detection.contour) > 0:
                cv2.drawContours(output, [detection.contour], -1, color, 2)

            if decision is None:
                label = (
                    f"#{detection.object_id}: "
                    f"{detection.primary_color}, {detection.size_category}"
                )
            else:
                label = (
                    f"#{detection.object_id}: {decision.category} | "
                    f"{decision.confidence:.0%}"
                )

            label_x, label_y = detection.bbox[0], max(detection.bbox[1] - 8, 20)
            cv2.putText(
                output,
                self._ascii_safe(label),
                (label_x, label_y),
                self.FONT,
                self.FONT_SCALE,
                color,
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return output

    def create_video_side_by_side(
        self,
        frame: np.ndarray,
        mask_panel: np.ndarray,
        *,
        left_title: str = "Video",
        right_title: str = "Object Masks",
        status_line: str = "",
    ) -> np.ndarray:
        """Combine the video frame and mask panel into one wide image."""
        left = frame.copy()
        right = mask_panel.copy()

        left_height, left_width = left.shape[:2]
        right_height, right_width = right.shape[:2]

        if right_height != left_height:
            scale = left_height / right_height
            right = cv2.resize(
                right,
                (max(1, int(right_width * scale)), left_height),
                interpolation=cv2.INTER_NEAREST,
            )

        combined = np.hstack([left, right])
        titled = self._add_split_titles(
            combined,
            left_width=left_width,
            left_title=left_title,
            right_title=right_title,
            status_line=status_line,
        )
        return titled

    def create_video_triple_view(
        self,
        frame: np.ndarray,
        mask_panel: np.ndarray,
        contour_panel: np.ndarray,
        *,
        left_title: str = "Video",
        center_title: str = "Object Masks",
        right_title: str = "Contours",
        status_line: str = "",
    ) -> np.ndarray:
        """Combine camera feed, mask, and contour overlay into one wide image."""
        target_height = frame.shape[0]
        left = frame.copy()
        center = self._match_panel_height(mask_panel, target_height)
        right = self._match_panel_height(contour_panel, target_height)

        combined = np.hstack([left, center, right])
        return self._add_triple_titles(
            combined,
            column_widths=[left.shape[1], center.shape[1], right.shape[1]],
            left_title=left_title,
            center_title=center_title,
            right_title=right_title,
            status_line=status_line,
        )

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
        object_count = len(self._paired_objects(result))
        decision_title = (
            "6. Decision"
            if object_count <= 1
            else f"6. Decision ({object_count} objects)"
        )
        # Combine panels into a 2x3 grid
        panels = [
            ("1. Original", original),
            ("2. Enhanced", enhanced),
            ("3. Segmentation", mask_bgr),
            ("4. Cleaned Mask", cleaned_bgr),
            ("5. Detection", detection_panel),
            (decision_title, decision_panel),
        ]

        labeled_panels = [self._add_title(panel, title) for title, panel in panels]
        # Stack panels into a 2x3 grid
        row1 = np.hstack(labeled_panels[:3])
        row2 = np.hstack(labeled_panels[3:])

        return np.vstack([row1, row2])

    def create_mask_review_dashboard(
        self,
        *,
        original: np.ndarray,
        enhanced: np.ndarray,
        mask: np.ndarray,
        cleaned_mask: np.ndarray,
        raw_overlay: np.ndarray,
        cleaned_overlay: np.ndarray,
        contour_overlay: np.ndarray,
        detection_panel: np.ndarray,
        metrics_lines: list[str],
        status_line: str = "",
        panel_size: tuple[int, int] = (400, 300),
    ) -> np.ndarray:
        """Eight-panel view for interactive mask/contour review (2 rows x 4 cols)."""
        panel_width, panel_height = panel_size

        def panel(image: np.ndarray, title: str) -> np.ndarray:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            fitted = self._resize_with_padding(image, (panel_width, panel_height))
            return self._add_title(fitted, title)

        row1 = np.hstack([
            panel(original, "1. Original"),
            panel(enhanced, "2. Enhanced"),
            panel(raw_overlay, "3. Raw overlay"),
            panel(cleaned_overlay, "4. Cleaned overlay"),
        ])
        row2 = np.hstack([
            panel(mask, "5. Raw mask"),
            panel(cleaned_mask, "6. Cleaned mask"),
            panel(contour_overlay, "7. Contours"),
            panel(detection_panel, "8. Detection"),
        ])
        grid = np.vstack([row1, row2])

        metrics_panel = np.zeros((120, grid.shape[1], 3), dtype=np.uint8)
        y = 22
        for line in metrics_lines[:4]:
            cv2.putText(
                metrics_panel,
                self._safe(line),
                (8, y),
                self.FONT,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            y += 24

        footer = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
        hint = (
            "n/Space=next  p=prev  m=bad mask  c=bad contours  b=both  o=OK  "
            "s=save  q=quit"
        )
        cv2.putText(
            footer,
            self._safe(status_line or hint),
            (8, 24),
            self.FONT,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return np.vstack([grid, metrics_panel, footer])

    def save_pipeline_outputs(self, result: PipelineResult, base_name: str) -> None:
        """Save all required output images for one test image."""
        stages_dir = Path(AppConfig().stages_dir)

        folders = {
            "original": stages_dir / "original",
            "enhanced": stages_dir / "enhanced",
            "mask": stages_dir / "mask",
            "cleaned_mask": stages_dir / "cleaned_mask",
            "detection": stages_dir / "detection",
            "contours": stages_dir / "contours",
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
        cv2.imwrite(str(folders["contours"] / f"{base_name}_contours.jpg"), detection_image)
        cv2.imwrite(str(folders["dashboard"] / f"{base_name}_dashboard.jpg"), dashboard)

    def show_pipeline(self, result: PipelineResult) -> None:
        """Display dashboard in a resizable window."""
        dashboard = self.create_dashboard(result)
        if not self._window_created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            h, w = dashboard.shape[:2]
            win_w, win_h = self._fit_window_size(w, h)
            cv2.resizeWindow(self.WINDOW_NAME, win_w, win_h)
            self._window_created = True
        cv2.imshow(self.WINDOW_NAME, dashboard)

    def is_window_open(self) -> bool:
        """Return False after the user closes the window with the title-bar X button."""
        if not self._window_created:
            return False
        try:
            return cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close_window(self) -> None:
        """Destroy the dashboard window and reset internal state."""
        if self._window_created:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass
            self._window_created = False

    def _raise_if_window_closed(self) -> None:
        if not self.is_window_open():
            self.close_window()
            raise WindowClosed()

    def pump_events(self, result: PipelineResult, delay_ms: int = 30) -> int:
        """Redraw the dashboard and process OpenCV GUI events (use while terminal blocks)."""
        self._raise_if_window_closed()
        self.show_pipeline(result)
        key = cv2.waitKey(delay_ms) & 0xFF
        self._raise_if_window_closed()
        return key

    def wait_until_key_or_close(self, result: PipelineResult) -> None:
        """Block until a key is pressed or the user closes the window."""
        while self.is_window_open():
            self.show_pipeline(result)
            if self._wait_key_or_window_closed(self.WINDOW_NAME):
                return
        self.close_window()
        raise WindowClosed()

    def wait_until_dismissed(
        self,
        window_name: str,
        frame: np.ndarray | None = None,
    ) -> None:
        """Block until a key is pressed or the user closes *window_name*."""
        while True:
            if frame is not None:
                cv2.imshow(window_name, frame)
            if self._wait_key_or_window_closed(window_name):
                return

    @staticmethod
    def _wait_key_or_window_closed(window_name: str) -> bool:
        """Process GUI events; return True on key press or window close."""
        key = cv2.waitKey(30)
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise WindowClosed()
        except cv2.error:
            raise WindowClosed()
        return key != -1

    def _draw_detection_on_image(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        decision=None,
    ) -> None:
        """Draw contour, bounding box and label directly on image."""
        x, y, w, h = detection.bbox
        color = self.OBJECT_COLORS[(detection.object_id - 1) % len(self.OBJECT_COLORS)]

        if detection.contour is not None and len(detection.contour) > 0:
            cv2.drawContours(image, [detection.contour], -1, color, 2)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

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
            color,
            self.FONT_THICKNESS,
        )

    def _paired_objects(
        self, result: PipelineResult,
    ) -> list[tuple[Decision, DetectionResult]]:
        """Align each decision with its detection by object_id."""
        decisions = result.decisions or [result.decision]
        detection_by_id = {
            detection.object_id: detection
            for detection in result.detections
        }

        pairs: list[tuple[Decision, DetectionResult]] = []
        for decision in decisions:
            detection = detection_by_id.get(decision.object_id, result.detection)
            pairs.append((decision, detection))
        return pairs

    @staticmethod
    def _ascii_safe(text: str) -> str:
        """Encode to ASCII-safe text for OpenCV rendering."""
        return text.encode("ascii", errors="replace").decode("ascii")

    def _draw_decision_text(
        self, panel: np.ndarray, result: PipelineResult
    ) -> None:
        """Draw decision text for one or many detected objects."""
        pairs = self._paired_objects(result)
        if len(pairs) == 1:
            self._draw_single_object_decision(
                panel,
                pairs[0][0],
                pairs[0][1],
                result.processing_time_ms,
            )
            return

        self._draw_multi_object_decision(panel, pairs, result.processing_time_ms)

    def _draw_single_object_decision(
        self,
        panel: np.ndarray,
        decision: Decision,
        detection: DetectionResult,
        processing_time_ms: float,
    ) -> None:
        """Full detail block for a single detected object."""
        safe = self._ascii_safe
        conf_pct = f"{decision.confidence * 100:.0f}%"

        if decision.is_unknown:
            line1 = "Unknown Object"
            closest = safe(decision.closest_match) if decision.closest_match else "N/A"
            line2 = f"Closest: {closest}"
        else:
            line1 = safe(decision.category)
            line2 = f"Confidence: {conf_pct}"

        rows = [
            (line1, 0.75, (0, 255, 0), 2),
            (line2, 0.60, self.TEXT_COLOR, 1),
            (f"Color: {decision.color}", 0.55, self.TEXT_COLOR, 1),
            (f"Size: {decision.size}", 0.55, self.TEXT_COLOR, 1),
            (f"Area: {detection.area_ratio * 100:.1f}% of frame", 0.50, self.TEXT_COLOR, 1),
            (
                f"BBox: {detection.bbox_width_ratio * 100:.0f}% x "
                f"{detection.bbox_height_ratio * 100:.0f}%",
                0.50,
                self.TEXT_COLOR,
                1,
            ),
            (f"Time: {processing_time_ms:.0f} ms", 0.55, (0, 200, 200), 1),
            (f"Shape: {detection.shape_category}", 0.50, (200, 150, 0), 1),
            (
                f"HSV:    {detection.color_hsv.name} "
                f"({detection.color_hsv.confidence:.0%})",
                0.48,
                (200, 200, 0),
                1,
            ),
            (
                f"KMeans: {detection.color_kmeans.name} "
                f"({detection.color_kmeans.confidence:.0%})",
                0.48,
                (200, 200, 0),
                1,
            ),
        ]
        self._draw_text_rows(panel, rows, start_y=32, line_spacing=28)

    def _draw_multi_object_decision(
        self,
        panel: np.ndarray,
        pairs: list[tuple[Decision, DetectionResult]],
        processing_time_ms: float,
    ) -> None:
        """Compact per-object blocks when multiple objects are present."""
        safe = self._ascii_safe
        scale = 0.46 if len(pairs) > 4 else 0.50
        y = 28

        for decision, detection in pairs:
            if decision.is_unknown:
                headline = (
                    f"#{decision.object_id} Unknown "
                    f"(closest: {safe(decision.closest_match or 'N/A')})"
                )
            else:
                headline = (
                    f"#{decision.object_id} {safe(decision.category)} "
                    f"{decision.confidence:.0%}"
                )

            detail = (
                f"{decision.color} | {decision.size} | "
                f"{detection.shape_category or 'n/a'}"
            )
            metrics = (
                f"area {detection.area_ratio * 100:.1f}%  "
                f"HSV {detection.color_hsv.name}  "
                f"KM {detection.color_kmeans.name}"
            )

            y = self._draw_text_rows(
                panel,
                [
                    (headline, scale + 0.05, (0, 255, 0), 2),
                    (detail, scale, self.TEXT_COLOR, 1),
                    (metrics, scale - 0.04, (180, 180, 180), 1),
                ],
                start_y=y,
                line_spacing=22,
            )
            y += 8

        self._draw_text_rows(
            panel,
            [(f"Time: {processing_time_ms:.0f} ms", 0.48, (0, 200, 200), 1)],
            start_y=min(y, panel.shape[0] - 12),
            line_spacing=22,
        )

    def _draw_text_rows(
        self,
        panel: np.ndarray,
        rows: list[tuple[str, float, tuple[int, int, int], int]],
        *,
        start_y: int,
        line_spacing: int,
    ) -> int:
        """Render text rows and return the next Y position."""
        y = start_y
        for text, scale, color, thickness in rows:
            safe_text = self._ascii_safe(text)
            cv2.putText(
                panel,
                safe_text,
                (12, y),
                self.FONT,
                scale,
                color,
                thickness,
            )
            line_h = int(
                cv2.getTextSize(safe_text, self.FONT, scale, thickness)[0][1] * 2.0
            )
            y += max(line_h, line_spacing)
        return y
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

    def _add_split_titles(
        self,
        image: np.ndarray,
        *,
        left_width: int,
        left_title: str,
        right_title: str,
        status_line: str = "",
    ) -> np.ndarray:
        """Add title bars for a side-by-side video + mask view."""
        _, width = image.shape[:2]
        title_height = 48 if status_line else 28
        title_bar = np.zeros((title_height, width, 3), dtype=np.uint8)

        cv2.putText(
            title_bar,
            left_title,
            (8, 20),
            self.FONT,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            title_bar,
            right_title,
            (left_width + 8, 20),
            self.FONT,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if status_line:
            cv2.putText(
                title_bar,
                status_line,
                (8, 40),
                self.FONT,
                0.45,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )

        return np.vstack([title_bar, image])

    def _match_panel_height(self, panel: np.ndarray, target_height: int) -> np.ndarray:
        """Resize *panel* height to *target_height* while preserving aspect ratio."""
        panel_height, panel_width = panel.shape[:2]
        if panel_height == target_height:
            return panel.copy()

        scale = target_height / panel_height
        return cv2.resize(
            panel,
            (max(1, int(panel_width * scale)), target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    def _add_triple_titles(
        self,
        image: np.ndarray,
        *,
        column_widths: list[int],
        left_title: str,
        center_title: str,
        right_title: str,
        status_line: str = "",
    ) -> np.ndarray:
        """Add title bars for a three-column video view."""
        _, width = image.shape[:2]
        left_width, center_width, _right_width = column_widths
        center_x = left_width
        right_x = left_width + center_width

        title_height = 48 if status_line else 28
        title_bar = np.zeros((title_height, width, 3), dtype=np.uint8)

        cv2.putText(
            title_bar,
            left_title,
            (8, 20),
            self.FONT,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            title_bar,
            center_title,
            (center_x + 8, 20),
            self.FONT,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            title_bar,
            right_title,
            (right_x + 8, 20),
            self.FONT,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if status_line:
            cv2.putText(
                title_bar,
                status_line,
                (8, 40),
                self.FONT,
                0.45,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )

        return np.vstack([title_bar, image])

    def _fit_window_size(self, content_w: int, content_h: int) -> tuple[int, int]:
        """Return window size that fits 90 % of the screen while preserving aspect ratio."""
        if self._screen_size is None:
            self._screen_size = self._detect_screen_size()
        max_w = int(self._screen_size[0] * 0.9)
        max_h = int(self._screen_size[1] * 0.9)
        if content_w <= max_w and content_h <= max_h:
            return content_w, content_h
        scale = min(max_w / content_w, max_h / content_h)
        return max(1, int(content_w * scale)), max(1, int(content_h * scale))

    @staticmethod
    def _detect_screen_size() -> tuple[int, int]:
        """Return (width, height) of the primary screen; falls back to 1920x1080."""
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            return 1920, 1080

    def _safe(self, text: str) -> str:
        """Make text safe for OpenCV rendering."""
        return text.encode("ascii", errors="replace").decode("ascii")