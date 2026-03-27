from pydantic import BaseModel, Field


class PlagiarismRequest(BaseModel):
    text: str = Field(min_length=1, description="Input text to scan for plagiarism")


class MatchResult(BaseModel):
    sentence: str
    source_title: str
    source_type: str
    source_url: str
    matched_text: str
    match_type: str | None = None
    lexical_score: float
    semantic_score: float
    overlap_score: float
    containment_score: float
    word_coverage: float
    char_similarity: float
    evidence_score: float
    layer1_score: float | None = None
    layer2_score: float | None = None
    layer3_score: float | None = None
    layer4_score: float | None = None
    layer5_score: float | None = None
    plagiarism_probability: float


class SentenceResult(BaseModel):
    sentence: str
    flagged: bool
    best_score: float
    matches: list[MatchResult]


class SourceAccuracyResult(BaseModel):
    source_title: str
    source_type: str
    source_url: str
    best_probability: float
    average_probability: float
    average_word_coverage: float
    average_evidence_score: float
    matched_sentences: int
    representative_text: str


class PlagiarismResponse(BaseModel):
    overall_similarity_percent: float
    flagged_sentences: int
    total_sentences: int
    scanned_web_sources: int
    scanned_book_sources: int
    processing_ms: int
    top_accuracy_sources: list[SourceAccuracyResult]
    results: list[SentenceResult]
