from __future__ import annotations

from pydantic import BaseModel, Field


class PlagiarismRequest(BaseModel):
    text: str = Field(min_length=1, description="Input text to scan for plagiarism")


class SimilarityComponents(BaseModel):
    tfidf: float
    lexical: float
    semantic: float
    syntactic: float
    character: float
    evidence: float
    composite: float


class LayerScores(BaseModel):
    nlp: float
    academic: float
    website: float
    google: float
    llm: float


class FalsePositiveInfo(BaseModel):
    adjusted_probability: float
    adjustment_factor: float
    suppressed: bool
    reasons: list[str]


class MatchResultV2(BaseModel):
    sentence: str
    source_title: str
    source_type: str
    source_url: str
    matched_text: str

    match_type: str | None = None

    tfidf_score: float
    lexical_score: float
    semantic_score: float
    overlap_score: float
    containment_score: float
    word_coverage: float
    char_similarity: float
    evidence_score: float

    layer1_score: float
    layer2_score: float
    layer3_score: float
    layer4_score: float
    layer5_score: float

    layer_scores: LayerScores
    similarity_components: SimilarityComponents
    false_positive: FalsePositiveInfo

    llm_reasoning: str | None = None
    plagiarism_probability: float


class SentenceResultV2(BaseModel):
    sentence: str
    flagged: bool
    best_score: float
    confidence: float
    confidence_level: str
    matches: list[MatchResultV2]


class SourceAccuracyResultV2(BaseModel):
    source_title: str
    source_type: str
    source_url: str
    best_probability: float
    average_probability: float
    average_word_coverage: float
    average_evidence_score: float
    matched_sentences: int
    representative_text: str
    layer_support: list[str] = Field(default_factory=list)


class LayerBreakdown(BaseModel):
    nlp: float
    academic: float
    website: float
    google: float
    llm: float


class PlagiarismResponseV2(BaseModel):
    version: str = "v2"

    overall_similarity_percent: float
    flagged_sentences: int
    total_sentences: int
    scanned_web_sources: int
    scanned_book_sources: int
    processing_ms: int

    confidence: float
    confidence_level: str
    score_breakdown: LayerBreakdown

    top_accuracy_sources: list[SourceAccuracyResultV2]
    results: list[SentenceResultV2]
