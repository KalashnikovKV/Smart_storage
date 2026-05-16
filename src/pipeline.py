"""CV Pipeline for Smart Storage — Object Sorting System.

Pipeline:
image -> enhance -> segment -> clean -> detect -> decide

--- Classification via CLASSIFICATION_RULES (config.py) ---

Rules are scored in RuleBasedDecisionEngine.score_rule() by matching color /
size / shape_hints plus optional numeric guards (aspect, solidity, circularity,
etc.). Adding a new class = one new ClassificationRule entry in AppConfig.rules.

1. Keyboard        — large dark elongated rectangular (R01)
2. Mouse           — compact dark rounded (R02)
3. Charger Adapter — compact light solid block/rect, circ >= 0.25 (R03)
4. Flash Drive     — small compact any color (R04)
5. USB-C Cable     — loop/elongated/hollow-coil, solidity < 0.62 (R05)
6. Headphones      — large/medium ring_like/irregular, solidity >= 0.62 (R06)
7. Colored Object  — chromatic fallback
8. Unknown Object

--- Calibration data (from real --debug measurements) ---

Object                  edge   solid  circ   shape       area
White charger (Image_13) 0.011  0.534  0.132  block       0.071
Silver charger (Image_6) 0.018  0.851  0.386  rectangular 0.044
Cable hollow (Image_5)   0.017  0.819  0.181  rectangular 0.045
Big headphones (Image_2) 0.076  0.640  0.161  ring_like   0.207
Small headphones (Image_4) 0.060 0.729 0.196  ring_like   0.122

Key separator — cable hollow vs charger:
  Cable hollow:     circ=0.181  (low — elongated blob from coil inner circle)
  Silver charger:   circ=0.386  (higher — compact rectangular block)
  White charger:    circ=0.132  (also low — BUT shape=block, not rectangular)

  The edge < 0.025 guard WRONGLY rejects both chargers (edge=0.011 and 0.018).
  REMOVED. New approach:

  For shape=rectangular AND edge < 0.025:
    → cable hollow if circ < 0.25  (cable: circ=0.181 → cable)
    → charger      if circ >= 0.25 (silver charger: circ=0.386 → charger)

  For shape=block:
    → always allow charger (white charger has shape=block, never a cable)
    → cable hollow never produces shape=block (it's a nearly circular blob
      which classifies as rectangular or oval, not block)
"""

import logging
import time

import numpy as np

from src.clean.cleaner import MaskCleaner
from src.clean.mask_ops import MaskOps
from src.config import AppConfig
from src.decision.protocol import Classifier
from src.decision.rule_based import RuleBasedDecisionEngine
from src.decision.yolo_category import yolo_category
from src.detect.color import ColorDetector
from src.detect.from_mask import ObjectMaskDetector
from src.enhance.ops import enhance as enhance_image
from src.image.validate import is_valid_bgr_uint8
from src.models import Decision, DetectionResult, PipelineResult
from src.segment.protocol import Segmenter
from src.segment.threshold import ThresholdSegmenter

LOGGER = logging.getLogger(__name__)


class Pipeline:
    """Main CV pipeline: enhance -> segment -> clean -> detect -> decide."""

    def __init__(
        self,
        config: AppConfig | None = None,
        classifier: Classifier | None = None,
        segmenter: Segmenter | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.mask_ops = MaskOps(self.config)
        self.color_detector = ColorDetector(self.config)
        self._threshold_segmenter = ThresholdSegmenter(self.config, self.mask_ops)
        self._mask_cleaner = MaskCleaner(self.config, self.mask_ops)
        self._object_detector = ObjectMaskDetector(
            self.config, self.mask_ops, self.color_detector,
        )
        self._rule_engine = RuleBasedDecisionEngine(self.config)
        self.classifier = classifier
        self.segmenter = segmenter
        self._current_frame: np.ndarray | None = None

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance image using CLAHE, gamma correction and light blur."""
        return enhance_image(image, self.config)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment the main object from background."""
        return self._threshold_segmenter.segment(image)

    def clean(self, mask: np.ndarray) -> np.ndarray:
        """Clean selected object mask."""
        return self._mask_cleaner.clean(mask)

    def detect(self, image: np.ndarray, mask: np.ndarray) -> DetectionResult | None:
        """Detect one object. Kept for compatibility with earlier code."""
        detections = self.detect_all(
            color_image=image,
            processing_image=image,
            mask=mask,
        )
        return detections[0] if detections else None

    def detect_all(
        self,
        color_image: np.ndarray,
        processing_image: np.ndarray,
        mask: np.ndarray,
    ) -> list[DetectionResult]:
        """Detect the main foreground object."""
        object_mask = self._object_detector.build_final_object_mask(mask)

        if object_mask is None:
            return []

        detection = self._object_detector.build_detection_from_mask(
            color_image=color_image,
            processing_image=processing_image,
            object_mask=object_mask,
            object_id=1,
        )

        return [detection] if detection is not None else []

    def _extract_roi(self, detection: DetectionResult) -> np.ndarray | None:
        """Crop the bounding-box region from the current frame for ML classifiers."""
        if self._current_frame is None:
            return None
        x, y, w, h = detection.bbox
        roi = self._current_frame[y : y + h, x : x + w]
        return roi if roi.size > 0 else None

    def decide(self, detection: DetectionResult) -> Decision:
        """Delegate to injected classifier or fall back to rule-based logic."""
        if self.classifier is not None:
            roi = self._extract_roi(detection)
            return self.classifier.classify(detection, roi)
        return self._rule_engine.decide(detection)

    def decide_all(self, detections: list[DetectionResult]) -> list[Decision]:
        """Make decisions for all detections."""
        return [self.decide(detection) for detection in detections]

    def run(self, image: np.ndarray) -> PipelineResult | None:
        """Execute the full pipeline.

        When a Segmenter (e.g. YOLOSegmenter) is configured, YOLO-Seg provides
        class + bbox + mask for each object and ColorDetector runs
        inside the YOLO mask. Pipeline.decide() is not called.

        Without a segmenter the classic threshold-based path
        (segment → clean → detect → rule-based decide) is used.
        """
        if not is_valid_bgr_uint8(image):
            return None

        start = time.time()

        enhanced = self.enhance(image)
        mask = self.segment(enhanced)
        cleaned = self.clean(mask)

        if self.segmenter is not None:
            detections, decisions = self._run_yolo_path(image, enhanced)
        else:
            detections, decisions = self._run_rule_based_path(
                image, enhanced, mask, cleaned,
            )

        if not detections:
            return None

        elapsed_ms = (time.time() - start) * 1000

        return PipelineResult(
            original=image,
            enhanced=enhanced,
            mask=mask,
            cleaned_mask=cleaned,
            detection=detections[0],
            decision=decisions[0],
            detections=detections,
            decisions=decisions,
            processing_time_ms=round(elapsed_ms, 1),
        )

    def _run_yolo_path(
        self,
        image: np.ndarray,
        enhanced: np.ndarray,
    ) -> tuple[list[DetectionResult], list[Decision]]:
        """YOLO-Seg path: class + mask from YOLO, color from ColorDetector."""
        yolo_detections = self.segmenter.segment(image)

        detections: list[DetectionResult] = []
        decisions: list[Decision] = []

        for object_id, yolo_det in enumerate(yolo_detections, start=1):
            detection = self._object_detector.build_detection_from_mask(
                color_image=image,
                processing_image=enhanced,
                object_mask=yolo_det.mask,
                object_id=object_id,
            )

            if detection is None:
                LOGGER.debug(
                    "YOLO detection %d skipped — mask geometry rejected", object_id,
                )
                continue

            category = yolo_category(yolo_det.class_name)
            conf = yolo_det.confidence

            decisions.append(
                self._decision_from_yolo(detection, category, conf, object_id)
            )
            detections.append(detection)

        return detections, decisions

    def _decision_from_yolo(
        self,
        detection: DetectionResult,
        category: str,
        conf: float,
        object_id: int,
    ) -> Decision:
        """Build Decision from YOLO class — Pipeline.decide() is not used."""
        return Decision(
            category=category,
            confidence=conf,
            color=detection.primary_color,
            size=detection.size_category,
            method_used="yolo",
            is_unknown=conf < self.config.low_confidence,
            closest_match="" if conf >= self.config.low_confidence else category,
            object_id=object_id,
        )

    def _run_rule_based_path(
        self,
        image: np.ndarray,
        enhanced: np.ndarray,
        mask: np.ndarray,
        cleaned: np.ndarray,
    ) -> tuple[list[DetectionResult], list[Decision]]:
        """Classic threshold + rule-based path (original behaviour)."""
        detections = self.detect_all(
            color_image=image,
            processing_image=enhanced,
            mask=cleaned,
        )

        if not detections:
            detections = self.detect_all(
                color_image=image,
                processing_image=enhanced,
                mask=mask,
            )

        if not detections:
            return [], []

        self._current_frame = image
        decisions = self.decide_all(detections)
        self._current_frame = None

        return detections, decisions
