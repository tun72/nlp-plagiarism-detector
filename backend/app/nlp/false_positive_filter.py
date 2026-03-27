from __future__ import annotations

from dataclasses import dataclass
import re

COMMON_PHRASES = {
    "in conclusion",
    "on the other hand",
    "as a result",
    "according to the study",
    "the purpose of this research",
    "machine learning is a subset of artificial intelligence",
    "artificial intelligence is transforming the world",
}

COMMON_KNOWLEDGE_HINTS = {
    "earth revolves around the sun",
    "water boils at 100",
    "photosynthesis",
    "world war",
    "newton",
    "gravity",
    "mitosis",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

CITATION_PATTERN = re.compile(
    r"(\[[0-9,\-\s]+\])|(\([A-Z][A-Za-z\-\s]+,\s*(19|20)\d{2}[a-z]?\))"
)


@dataclass
class FalsePositiveDecision:
    adjusted_probability: float
    adjustment_factor: float
    suppressed: bool
    reasons: list[str]


class FalsePositiveFilter:
    """Heuristic filter to suppress trivial/cited/common-knowledge false positives."""

    def assess(
        self,
        *,
        sentence: str,
        matched_text: str,
        probability: float,
        lexical: float,
        semantic: float,
        word_coverage: float,
        char_similarity: float,
    ) -> FalsePositiveDecision:
        reasons: list[str] = []
        adjustment = 1.0

        normalized = " ".join((sentence or "").lower().split())
        tokens = [tok for tok in re.findall(r"[a-z0-9']+", normalized) if tok]
        non_stop = [tok for tok in tokens if tok not in STOPWORDS]

        if CITATION_PATTERN.search(sentence or ""):
            adjustment *= 0.70
            reasons.append("Citation-like sentence")

        if len(tokens) < 5:
            adjustment *= 0.65
            reasons.append("Very short sentence")

        if normalized in COMMON_PHRASES:
            adjustment *= 0.60
            reasons.append("Common phrase")

        if non_stop and (1.0 - (len(non_stop) / max(1, len(tokens)))) > 0.62:
            adjustment *= 0.70
            reasons.append("Mostly function words")

        if any(phrase in normalized for phrase in COMMON_KNOWLEDGE_HINTS):
            adjustment *= 0.70
            reasons.append("Common knowledge statement")

        if lexical < 0.08 and semantic < 0.55 and word_coverage < 0.30 and char_similarity < 0.30:
            adjustment *= 0.62
            reasons.append("Weak multi-signal evidence")

        if sentence and matched_text and sentence.strip().lower() == matched_text.strip().lower():
            # Preserve exact duplicate signal from over-filtering.
            adjustment = max(adjustment, 0.85)

        adjusted = max(0.0, min(1.0, probability * adjustment))
        suppressed = adjusted < 0.25 and len(reasons) >= 2

        return FalsePositiveDecision(
            adjusted_probability=round(adjusted, 4),
            adjustment_factor=round(adjustment, 4),
            suppressed=suppressed,
            reasons=reasons,
        )
