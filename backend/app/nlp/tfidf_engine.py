from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TfidfMatch:
    index: int
    score: float


class TfidfEngine:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = None
        self._corpus: list[str] = []

    def fit(self, corpus: list[str]) -> None:
        self._corpus = corpus
        if not corpus:
            self._matrix = None
            return
        self._matrix = self.vectorizer.fit_transform(corpus)

    def prepare_query(self, text: str):
        if not self._corpus or self._matrix is None:
            return None
        return self.vectorizer.transform([text])

    def similarity_from_prepared(self, prepared_query, index: int) -> float:
        if prepared_query is None or self._matrix is None:
            return 0.0
        if index < 0 or index >= len(self._corpus):
            return 0.0
        return float(cosine_similarity(prepared_query, self._matrix[index]).item())

    def top_k_from_prepared(self, prepared_query, k: int = 3) -> list[TfidfMatch]:
        if prepared_query is None or self._matrix is None:
            return []
        scores = cosine_similarity(prepared_query, self._matrix).flatten()
        if scores.size == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:k]
        return [TfidfMatch(index=int(i), score=float(scores[i])) for i in top_indices]

    def top_k(self, text: str, k: int = 3) -> list[TfidfMatch]:
        prepared = self.prepare_query(text)
        return self.top_k_from_prepared(prepared, k=k)
