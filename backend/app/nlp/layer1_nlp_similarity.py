from __future__ import annotations

from dataclasses import dataclass

from app.nlp.features import char_ngram_similarity, containment_score, token_overlap, word_coverage_score
from app.nlp.semantic_engine import SemanticEngine
from app.nlp.tfidf_engine import TfidfEngine


def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass
class SimilarityComponents:
    tfidf: float
    lexical: float
    semantic: float
    syntactic: float
    character: float
    evidence: float
    composite: float
    overlap: float
    containment: float
    word_coverage: float


@dataclass
class Layer1PreparedQuery:
    normalized_text: str
    tfidf_query: object | None
    semantic_queries: list[object]


class Layer1NlpSimilarityEngine:
    """
    Local NLP similarity layer.

    Combines lexical, semantic, syntactic, and character-level evidence into a
    single composite score suitable for sentence-level plagiarism ranking.
    """

    def __init__(self, semantic_models: list[str], *, semantic_local_files_only: bool = True) -> None:
        models = [name for name in (semantic_models or []) if name]
        if not models:
            models = [""]

        self._tfidf = TfidfEngine()
        self._semantic_engines = [
            SemanticEngine(model_name=name, local_files_only=semantic_local_files_only)
            for name in models
        ]
        self._corpus: list[str] = []

    def fit(self, corpus: list[str]) -> None:
        self._corpus = [text for text in corpus if text]
        self._tfidf.fit(self._corpus)
        for engine in self._semantic_engines:
            engine.fit(self._corpus)

    def has_corpus(self) -> bool:
        return bool(self._corpus)

    def prepare_query(self, normalized_text: str) -> Layer1PreparedQuery:
        text = (normalized_text or "").strip()
        return Layer1PreparedQuery(
            normalized_text=text,
            tfidf_query=self._tfidf.prepare_query(text),
            semantic_queries=[engine.prepare_query(text) for engine in self._semantic_engines],
        )

    def top_candidate_indices(self, prepared_query: Layer1PreparedQuery, *, lexical_k: int, semantic_k: int) -> set[int]:
        if not self._corpus:
            return set()

        indices: set[int] = set()

        for lexical in self._tfidf.top_k_from_prepared(prepared_query.tfidf_query, k=max(1, lexical_k)):
            indices.add(lexical.index)

        for idx, engine in enumerate(self._semantic_engines):
            semantic_query = prepared_query.semantic_queries[idx] if idx < len(prepared_query.semantic_queries) else None
            for candidate_index, _ in engine.top_k_from_prepared(semantic_query, k=max(1, semantic_k)):
                indices.add(candidate_index)

        return indices

    def score_candidate(self, prepared_query: Layer1PreparedQuery, candidate_index: int) -> SimilarityComponents:
        if candidate_index < 0 or candidate_index >= len(self._corpus):
            return SimilarityComponents(
                tfidf=0.0,
                lexical=0.0,
                semantic=0.0,
                syntactic=0.0,
                character=0.0,
                evidence=0.0,
                composite=0.0,
                overlap=0.0,
                containment=0.0,
                word_coverage=0.0,
            )

        candidate_text = self._corpus[candidate_index]
        lexical = self._tfidf.similarity_from_prepared(prepared_query.tfidf_query, candidate_index)

        semantic_scores: list[float] = []
        for idx, engine in enumerate(self._semantic_engines):
            semantic_query = prepared_query.semantic_queries[idx] if idx < len(prepared_query.semantic_queries) else None
            semantic_scores.append(engine.similarity_from_prepared(semantic_query, candidate_index))
        semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0

        overlap = token_overlap(prepared_query.normalized_text, candidate_text)
        containment = containment_score(prepared_query.normalized_text, candidate_text)
        word_coverage = word_coverage_score(prepared_query.normalized_text, candidate_text)
        character = char_ngram_similarity(prepared_query.normalized_text, candidate_text)

        syntactic = (overlap * 0.60) + (containment * 0.40)
        evidence = (word_coverage * 0.45) + (containment * 0.25) + (overlap * 0.20) + (character * 0.10)

        composite = (
            (lexical * 0.25)
            + (semantic * 0.35)
            + (syntactic * 0.20)
            + (character * 0.10)
            + (evidence * 0.10)
        )

        return SimilarityComponents(
            tfidf=round(_clamp_01(lexical), 4),
            lexical=round(_clamp_01(lexical), 4),
            semantic=round(_clamp_01(semantic), 4),
            syntactic=round(_clamp_01(syntactic), 4),
            character=round(_clamp_01(character), 4),
            evidence=round(_clamp_01(evidence), 4),
            composite=round(_clamp_01(composite), 4),
            overlap=round(_clamp_01(overlap), 4),
            containment=round(_clamp_01(containment), 4),
            word_coverage=round(_clamp_01(word_coverage), 4),
        )
