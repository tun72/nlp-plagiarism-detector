from __future__ import annotations

from pathlib import Path

import joblib


class SimilarityCalibrator:
    def __init__(self, model_path: str | None) -> None:
        self.model = None
        if not model_path:
            return
        path = Path(model_path)
        if path.exists() and path.is_file():
            self.model = joblib.load(path)

    def predict_probability(self, lexical: float, semantic: float, overlap: float, containment: float) -> float | None:
        if self.model is None:
            return None
        features = [[lexical, semantic, overlap, containment]]
        prob = float(self.model.predict_proba(features)[0][1])
        return max(0.0, min(1.0, prob))
