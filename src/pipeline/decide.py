"""Rule-based classification (config.rules) without Pipeline dependency."""

import logging

import numpy as np

from src.config import AppConfig, ClassificationRule
from src.models import Decision, DetectionResult

LOGGER = logging.getLogger(__name__)


class RuleBasedDecisionEngine:
    """Scores AppConfig.rules and builds a Decision from a DetectionResult."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def decide(self, detection: DetectionResult) -> Decision:
        """Produce final decision by scoring all classification rules."""
        color = detection.primary_color
        size = detection.size_category

        LOGGER.debug(
            "DETECT: color=%s size=%s shape=%s area=%.3f circ=%.3f solid=%.3f "
            "extent=%.3f edge=%.3f bbox_w=%.3f bbox_h=%.3f aspect=%.2f",
            color, size, detection.shape_category,
            detection.area_ratio, detection.circularity, detection.solidity,
            detection.extent, detection.edge_density,
            detection.bbox_width_ratio, detection.bbox_height_ratio,
            detection.aspect_ratio,
        )

        color_confidence = max(
            detection.color_hsv.confidence,
            detection.color_kmeans.confidence,
        )

        chromatic_colors = {
            "red", "green", "blue", "yellow", "orange", "purple", "brown",
        }

        best_rule: ClassificationRule | None = None
        best_score = 0.0

        for rule in self.config.rules:
            score = self.score_rule(rule, detection, color)
            if score > best_score:
                best_score = score
                best_rule = rule

        category = "Unknown Object"
        confidence = 0.0
        is_unknown = False
        closest_match = ""

        if best_rule is not None:
            category = best_rule.category_en
            confidence = best_score * color_confidence
        elif color in chromatic_colors:
            category = "Colored Object"
            confidence = 0.65 * color_confidence
        else:
            confidence = max(0.30 * color_confidence, 0.15)
            is_unknown = True
            closest_match = self.guess_closest_category(detection)

        if confidence < self.config.low_confidence:
            is_unknown = True
            closest_match = closest_match or category
            category = "Unknown Object"

        return Decision(
            category=category,
            confidence=round(float(confidence), 3),
            color=color,
            size=size,
            method_used="combined",
            is_unknown=is_unknown,
            closest_match=closest_match,
            object_id=getattr(detection, "object_id", 1),
        )

    def score_rule(
        self, rule: ClassificationRule, detection: DetectionResult, color: str,
    ) -> float:
        """Return matching score for a rule; 0.0 if any hard condition fails."""
        if color not in rule.colors:
            return 0.0
        if detection.size_category not in rule.sizes:
            return 0.0
        if detection.shape_category not in rule.shape_hints:
            return 0.0

        if detection.aspect_ratio < rule.min_aspect_ratio:
            return 0.0
        if detection.aspect_ratio > rule.max_aspect_ratio:
            return 0.0
        if detection.solidity < rule.min_solidity:
            return 0.0
        if detection.solidity > rule.max_solidity:
            return 0.0
        if detection.extent < rule.min_extent:
            return 0.0
        if detection.edge_density > rule.max_edge_density:
            return 0.0
        if detection.area_ratio < rule.min_area_ratio:
            return 0.0
        if detection.area_ratio > rule.max_area_ratio:
            return 0.0

        if (
            detection.shape_category == "rectangular"
            and detection.circularity < rule.min_circularity_if_rectangular
        ):
            return 0.0

        return rule.base_confidence

    def guess_closest_category(self, detection: DetectionResult) -> str:
        """Return fallback category hint."""
        if detection.shape_category == "oval":
            return "Mouse-like object"

        if detection.shape_category == "rectangular":
            return "Rectangular object"

        if detection.shape_category in {"ring_like", "irregular"}:
            return "Cable-like or headphones-like object"

        return "Unknown visual object"


class RuleBasedClassifier:
    """Classifier that delegates to RuleBasedDecisionEngine (Classifier protocol)."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._engine = RuleBasedDecisionEngine(config or AppConfig())

    def classify(
        self,
        detection: DetectionResult,
        roi: np.ndarray | None = None,
    ) -> Decision:
        """Return a Decision using rule-based heuristics."""
        return self._engine.decide(detection)
