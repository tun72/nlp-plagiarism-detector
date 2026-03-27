from __future__ import annotations

from app.nlp.calibrator import SimilarityCalibrator


class AIPlagiarismDetectorModel:
    """Final plagiarism probability model.

    Uses a trained calibrator model when available, otherwise uses a tuned
    weighted fallback to stay operational in all environments.
    """

    def __init__(self, calibrator_model_path: str | None = None) -> None:
        self.calibrator = SimilarityCalibrator(calibrator_model_path)

    def predict_probability(
        self,
        lexical: float,
        semantic: float,
        overlap: float,
        containment: float,
        *,
        word_coverage: float = 0.0,
        char_similarity: float = 0.0,
    ) -> float:
        calibrated = self.calibrator.predict_probability(lexical, semantic, overlap, containment)
        # Evidence features that strengthen precision for close paraphrases and exact term retention.
        evidence_score = (word_coverage * 0.45) + (containment * 0.25) + (overlap * 0.20) + (char_similarity * 0.10)

        if calibrated is not None:
            score = (calibrated * 0.82) + (evidence_score * 0.18)
        else:
            score = (
                (lexical * 0.32)
                + (semantic * 0.28)
                + (overlap * 0.12)
                + (containment * 0.10)
                + (word_coverage * 0.12)
                + (char_similarity * 0.06)
            )

        # Suppress weak candidates that score mostly from one noisy feature.
        if lexical < 0.04 and semantic < 0.30 and evidence_score < 0.28:
            score *= 0.65

        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score
