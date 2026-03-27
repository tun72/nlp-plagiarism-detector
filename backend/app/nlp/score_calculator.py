from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import pstdev

from app.core.config import Settings, get_settings


def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass
class LayerScores:
    nlp: float
    academic: float
    website: float
    google: float
    llm: float


@dataclass
class ConfidenceResult:
    value: float
    level: str


class ScoreCalculator:
    """Weighted multi-layer scoring and confidence estimation."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        weights = {
            "nlp": max(0.0, float(settings.plagiarism_weight_nlp)),
            "academic": max(0.0, float(settings.plagiarism_weight_academic)),
            "website": max(0.0, float(settings.plagiarism_weight_web)),
            "google": max(0.0, float(settings.plagiarism_weight_google)),
            "llm": max(0.0, float(settings.plagiarism_weight_llm)),
        }
        total = sum(weights.values())
        if total <= 0:
            total = 1.0
            weights = {"nlp": 0.4, "academic": 0.2, "website": 0.2, "google": 0.1, "llm": 0.1}
        self.weights = {k: v / total for k, v in weights.items()}

    def sentence_probability(self, scores: LayerScores) -> float:
        value = (
            (scores.nlp * self.weights["nlp"])
            + (scores.academic * self.weights["academic"])
            + (scores.website * self.weights["website"])
            + (scores.google * self.weights["google"])
            + (scores.llm * self.weights["llm"])
        )
        return round(_clamp_01(value), 4)

    def confidence(self, layer_values: list[float]) -> ConfidenceResult:
        if not layer_values:
            return ConfidenceResult(value=0.0, level="low")

        clipped = [_clamp_01(v) for v in layer_values]
        evidence_layers = sum(1 for v in clipped if v >= 0.35)
        agreement = min(1.0, evidence_layers / 5.0)

        variance_penalty = pstdev(clipped) if len(clipped) > 1 else 0.0
        consistency = 1.0 / (1.0 + variance_penalty)

        mean_score = sum(clipped) / len(clipped)
        extremity = abs(mean_score - 0.5) * 2.0

        confidence = _clamp_01((agreement * 0.50) + (consistency * 0.30) + (extremity * 0.20))
        return ConfidenceResult(value=round(confidence, 4), level=self.confidence_level(confidence))

    def overall_probability(self, probabilities: list[float], sentence_lengths: list[int] | None = None) -> float:
        if not probabilities:
            return 0.0
        if sentence_lengths and len(sentence_lengths) == len(probabilities):
            total_len = sum(max(1, n) for n in sentence_lengths)
            weighted = sum(prob * max(1, n) for prob, n in zip(probabilities, sentence_lengths))
            return round(_clamp_01(weighted / max(1, total_len)), 4)
        return round(_clamp_01(sum(probabilities) / len(probabilities)), 4)

    def confidence_level(self, value: float) -> str:
        if value >= 0.80:
            return "very_high"
        if value >= 0.60:
            return "high"
        if value >= 0.40:
            return "medium"
        return "low"

    def sigmoid_ratio(self, numerator: float, denominator: float, multiplier: float = 3.0) -> float:
        if denominator <= 0:
            return 0.0
        ratio = max(0.0, numerator) / denominator
        z = (ratio * multiplier) - (multiplier / 2)
        return _clamp_01(1.0 / (1.0 + math.exp(-z)))
