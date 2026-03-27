from __future__ import annotations

from functools import lru_cache
import logging
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@lru_cache
def _load_model(model_name: str, local_files_only: bool):
    if not model_name:
        return None, "No model name configured"
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        return None, f"sentence-transformers import failed: {exc}"

    try:
        return SentenceTransformer(model_name, local_files_only=local_files_only), None
    except Exception as exc:
        # Fast fallback to non-semantic mode if model is unavailable.
        return None, f"Model load failed for '{model_name}': {exc}"


class SemanticEngine:
    def __init__(self, model_name: str, local_files_only: bool = True) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.model, self.load_error = _load_model(model_name, local_files_only)
        self.using_fallback = self.model is None
        self._embeddings = None
        self._fallback_vec = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb")
        self._fallback_matrix = None
        self._corpus: list[str] = []
        if self.model_name and self.using_fallback and self.load_error:
            logger.warning("%s", self.load_error)

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    @property
    def backend_name(self) -> str:
        if self.model is not None:
            return "huggingface_sentence_transformers"
        return "char_tfidf_fallback"

    def status(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_loaded": self.is_model_loaded,
            "backend": self.backend_name,
            "local_files_only": self.local_files_only,
            "fallback_reason": self.load_error if self.using_fallback else None,
        }

    def fit(self, corpus: list[str]) -> None:
        self._corpus = corpus
        if not corpus:
            self._embeddings = None
            self._fallback_matrix = None
            return
        if self.model is None:
            self._fallback_matrix = self._fallback_vec.fit_transform(corpus)
            return
        self._embeddings = self.model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)

    def prepare_query(self, text: str):
        if self.model is None:
            if self._fallback_matrix is None:
                return None
            return self._fallback_vec.transform([text])
        return self.model.encode([text], convert_to_tensor=True, normalize_embeddings=True)

    def encode(self, text: str):
        if not text:
            return None
        if self.model is None:
            return None
        try:
            emb = self.model.encode([text], normalize_embeddings=True)
            return emb[0] if len(emb) else None
        except Exception:
            return None

    def similarity_to_index(self, text: str, index: int) -> float:
        prepared = self.prepare_query(text)
        return self.similarity_from_prepared(prepared, index)

    def similarity_from_prepared(self, prepared_query, index: int) -> float:
        if index < 0 or index >= len(self._corpus):
            return 0.0

        if self.model is None:
            if self._fallback_matrix is None or prepared_query is None:
                return 0.0
            score = cosine_similarity(prepared_query, self._fallback_matrix[index]).item()
            return float(score)

        if self._embeddings is None or prepared_query is None:
            return 0.0

        from sentence_transformers.util import cos_sim

        return float(cos_sim(prepared_query, self._embeddings[index]).item())

    def top_k_from_prepared(self, prepared_query, k: int = 3) -> list[tuple[int, float]]:
        if k <= 0 or not self._corpus:
            return []

        if self.model is None:
            if self._fallback_matrix is None or prepared_query is None:
                return []
            scores = cosine_similarity(prepared_query, self._fallback_matrix).flatten()
        else:
            if self._embeddings is None or prepared_query is None:
                return []
            from sentence_transformers.util import cos_sim

            tensor_scores = cos_sim(prepared_query, self._embeddings).squeeze(0)
            if hasattr(tensor_scores, "detach"):
                scores = tensor_scores.detach().cpu().numpy()
            else:
                scores = np.asarray(tensor_scores)

        if getattr(scores, "size", 0) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices]
