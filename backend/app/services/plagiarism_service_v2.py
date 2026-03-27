from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.cache import get_text_cache
from app.core.config import Settings, get_settings
from app.nlp.false_positive_filter import FalsePositiveFilter
from app.nlp.layer1_nlp_similarity import Layer1NlpSimilarityEngine
from app.nlp.layer2_academic_search import AcademicSource, Layer2AcademicSearch
from app.nlp.layer3_website_search import Layer3WebsiteSearch, WebsiteSource
from app.nlp.layer4_google_verification import GoogleSnippetSource, Layer4GoogleVerification
from app.nlp.layer5_llm_reasoning import Layer5LlmReasoning
from app.nlp.preprocessing import normalize_sentence, split_sentences
from app.nlp.score_calculator import LayerScores, ScoreCalculator
from app.schemas.plagiarism_v2 import (
    FalsePositiveInfo,
    LayerBreakdown,
    LayerScores as LayerScoresSchema,
    MatchResultV2,
    PlagiarismResponseV2,
    SentenceResultV2,
    SimilarityComponents,
    SourceAccuracyResultV2,
)


def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _canonical_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{host}{path}"


def _looks_like_citation(sentence: str) -> bool:
    s = sentence or ""
    return ("[" in s and "]" in s) or ("(" in s and ")" in s and any(ch.isdigit() for ch in s))


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
UCSY_KEYWORDS = (
    "ucsy",
    "university of computer studies",
    "university of computer studies yangon",
)


def _tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text or "")


def _contains_ucsy_keywords(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in UCSY_KEYWORDS)


def _is_ucsy_url(url: str) -> bool:
    host = urlparse((url or "").strip()).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host == "ucsy.edu.mm" or host.endswith(".ucsy.edu.mm"):
        return True
    return "facebook.com/lwinmay.thant.796" in (url or "").lower()


def _limit_text_words(text: str, *, max_words: int) -> str:
    words = _tokenize_words(text)
    if not words:
        return (text or "").strip()
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip()


def _build_word_units(
    text: str,
    *,
    window_words: int,
    stride_words: int,
    min_words: int,
    max_units: int,
) -> list[str]:
    words = _tokenize_words(text)
    if not words:
        return []

    safe_window = max(min_words, int(window_words))
    safe_stride = max(1, int(stride_words))
    cap = max(1, int(max_units))

    if len(words) <= safe_window:
        compact = " ".join(words).strip()
        return [compact] if compact else []

    units: list[str] = []
    seen: set[str] = set()

    for start in range(0, len(words), safe_stride):
        chunk = words[start : start + safe_window]
        if len(chunk) < min_words:
            break
        unit = " ".join(chunk).strip()
        if not unit or unit in seen:
            if start + safe_window >= len(words):
                break
            continue
        seen.add(unit)
        units.append(unit)
        if len(units) >= cap:
            break
        if start + safe_window >= len(words):
            break

    # Ensure the tail is represented.
    tail_chunk = words[-safe_window:]
    if len(tail_chunk) >= min_words:
        tail_unit = " ".join(tail_chunk).strip()
        if tail_unit and tail_unit not in seen and len(units) < cap:
            units.append(tail_unit)

    return units


def _build_query_phrase(sentence: str) -> str:
    cleaned = " ".join((sentence or "").strip().split())
    if not cleaned:
        return ""

    # Keep longer phrases exact for retrieval; adding lemmatized tokens can
    # hurt recall on exact-copy search providers.
    if len(cleaned) >= 80 or len(cleaned.split()) >= 12:
        return cleaned

    normalized = normalize_sentence(cleaned)
    if not normalized:
        return cleaned

    raw_tokens = {tok.lower() for tok in cleaned.split() if tok}
    enriched = [tok for tok in normalized.split() if tok.lower() not in raw_tokens]
    if not enriched:
        return cleaned
    return f"{cleaned} {' '.join(enriched)}".strip()


def _select_query_sentences(sentences: list[str], *, max_queries: int, min_chars: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        cleaned = " ".join((sentence or "").strip().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)

    if not deduped:
        return []

    scored: list[tuple[float, str]] = []
    for sentence in deduped:
        normalized = normalize_sentence(sentence)
        tokens = normalized.split()
        unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0
        length_score = min(1.0, len(sentence) / 180.0)
        info = (unique_ratio * 0.65) + (length_score * 0.35)
        if len(sentence) < min_chars:
            info *= 0.75
        scored.append((info, sentence))

    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [sentence for _, sentence in scored[: max(1, max_queries)]]


@dataclass(frozen=True)
class AnalysisPlan:
    window_words: int
    stride_words: int
    max_units: int
    max_queries: int
    llm_global_budget: int
    llm_timeout_seconds: float
    detection_budget_seconds: float
    retrieval_budget_ratio: float


def _build_analysis_plan(word_count: int, settings: Settings) -> AnalysisPlan:
    safe_word_count = max(1, int(word_count))
    base_detection_budget = max(10.0, float(settings.max_detection_seconds))
    base_queries = max(1, int(settings.max_query_sentences))
    llm_timeout = max(0.2, float(settings.ollama_timeout_seconds))

    if safe_word_count >= 1200:
        return AnalysisPlan(
            window_words=30,
            stride_words=18,
            max_units=64,
            max_queries=min(base_queries, 4),
            llm_global_budget=8,
            llm_timeout_seconds=min(llm_timeout, 2.0),
            detection_budget_seconds=max(base_detection_budget, 85.0),
            retrieval_budget_ratio=0.45,
        )
    if safe_word_count >= 900:
        return AnalysisPlan(
            window_words=26,
            stride_words=15,
            max_units=72,
            max_queries=min(base_queries, 4),
            llm_global_budget=12,
            llm_timeout_seconds=min(llm_timeout, 2.4),
            detection_budget_seconds=max(base_detection_budget, 75.0),
            retrieval_budget_ratio=0.48,
        )
    if safe_word_count >= 500:
        return AnalysisPlan(
            window_words=20,
            stride_words=11,
            max_units=90,
            max_queries=min(base_queries, 5),
            llm_global_budget=18,
            llm_timeout_seconds=min(llm_timeout, 3.0),
            detection_budget_seconds=max(base_detection_budget, 60.0),
            retrieval_budget_ratio=0.50,
        )

    return AnalysisPlan(
        window_words=14,
        stride_words=7,
        max_units=120,
        max_queries=base_queries,
        llm_global_budget=max(12, settings.ollama_max_pairs_per_sentence * 36),
        llm_timeout_seconds=llm_timeout,
        detection_budget_seconds=base_detection_budget,
        retrieval_budget_ratio=0.55,
    )


@dataclass
class SourceRecord:
    title: str
    url: str
    content: str
    source_type: str
    layers: set[str]
    reputation: float


@dataclass
class CandidateRow:
    sentence_text: str
    normalized_text: str
    source_title: str
    source_url: str
    source_type: str
    layers: set[str]
    reputation: float


def _merge_sources(
    academic_sources: list[AcademicSource],
    website_sources: list[WebsiteSource],
    google_sources: list[GoogleSnippetSource],
) -> list[SourceRecord]:
    merged: dict[str, SourceRecord] = {}

    def upsert(record: SourceRecord) -> None:
        key = _canonical_url(record.url) or f"{record.title}|{hash(record.content)}"
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
            return
        existing.layers.update(record.layers)
        existing.reputation = max(existing.reputation, record.reputation)
        if len(record.content) > len(existing.content):
            existing.content = record.content
            existing.title = record.title or existing.title
            existing.source_type = record.source_type or existing.source_type
            existing.url = record.url or existing.url

    for src in academic_sources:
        upsert(
            SourceRecord(
                title=src.title,
                url=src.url,
                content=src.content,
                source_type=src.source_type,
                layers={"academic"},
                reputation=0.95,
            )
        )

    for src in website_sources:
        upsert(
            SourceRecord(
                title=src.title,
                url=src.url,
                content=src.content,
                source_type=src.source_type,
                layers={"website"},
                reputation=_clamp_01(src.reputation),
            )
        )

    for src in google_sources:
        upsert(
            SourceRecord(
                title=src.title,
                url=src.url,
                content=src.content,
                source_type="web",
                layers={"google", "website"},
                reputation=0.85,
            )
        )

    return list(merged.values())


def _extract_candidate_rows(sources: list[SourceRecord], *, cap: int, per_doc_limit: int, min_tokens: int) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    seen: set[str] = set()

    for source in sources:
        text = (source.content or "").strip()
        if not text:
            continue

        source_units = _build_word_units(
            text,
            window_words=max(12, min_tokens * 3),
            stride_words=max(4, min_tokens),
            min_words=max(3, min_tokens),
            max_units=max(per_doc_limit * 3, per_doc_limit),
        )
        if not source_units:
            source_units = split_sentences(text) or [text]

        added = 0
        for sentence in source_units:
            normalized = normalize_sentence(sentence)
            if not normalized:
                continue
            if len(normalized.split()) < max(1, min_tokens):
                continue
            if normalized in seen:
                continue

            seen.add(normalized)
            rows.append(
                CandidateRow(
                    sentence_text=sentence,
                    normalized_text=normalized,
                    source_title=source.title,
                    source_url=source.url,
                    source_type=source.source_type,
                    layers=set(source.layers),
                    reputation=source.reputation,
                )
            )
            added += 1

            if len(rows) >= cap or added >= per_doc_limit:
                break

        if len(rows) >= cap:
            break

    return rows


def _classify_match_type(lexical: float, semantic: float, evidence: float, llm_score: float) -> str:
    if lexical >= 0.82 and evidence >= 0.72:
        return "direct_copy"
    if llm_score >= 0.82 and lexical < 0.60:
        return "ai_rewrite"
    if semantic >= 0.78 and lexical < 0.78:
        return "paraphrased"
    if semantic >= 0.62 and evidence >= 0.48:
        return "semantic"
    return "possible_overlap"


def _aggregate_top_sources(results: list[SentenceResultV2]) -> list[SourceAccuracyResultV2]:
    aggregate: dict[str, dict] = {}

    for sentence_result in results:
        for match in sentence_result.matches:
            key = f"{match.source_type}::{match.source_url}"
            current = aggregate.get(key)
            if current is None:
                aggregate[key] = {
                    "source_title": match.source_title,
                    "source_type": match.source_type,
                    "source_url": match.source_url,
                    "best_probability": match.plagiarism_probability,
                    "sum_probability": match.plagiarism_probability,
                    "sum_word_coverage": match.word_coverage,
                    "sum_evidence": match.evidence_score,
                    "matched_sentences": 1,
                    "representative_text": match.matched_text,
                    "layers": {
                        layer
                        for layer, value in (
                            ("nlp", match.layer1_score),
                            ("academic", match.layer2_score),
                            ("website", match.layer3_score),
                            ("google", match.layer4_score),
                            ("llm", match.layer5_score),
                        )
                        if value > 0
                    },
                }
                continue

            current["matched_sentences"] += 1
            current["sum_probability"] += match.plagiarism_probability
            current["sum_word_coverage"] += match.word_coverage
            current["sum_evidence"] += match.evidence_score
            if match.plagiarism_probability > current["best_probability"]:
                current["best_probability"] = match.plagiarism_probability
                current["representative_text"] = match.matched_text

            for layer, value in (
                ("nlp", match.layer1_score),
                ("academic", match.layer2_score),
                ("website", match.layer3_score),
                ("google", match.layer4_score),
                ("llm", match.layer5_score),
            ):
                if value > 0:
                    current["layers"].add(layer)

    ranked: list[SourceAccuracyResultV2] = []
    for item in aggregate.values():
        hits = max(1, int(item["matched_sentences"]))
        ranked.append(
            SourceAccuracyResultV2(
                source_title=item["source_title"],
                source_type=item["source_type"],
                source_url=item["source_url"],
                best_probability=round(float(item["best_probability"]), 4),
                average_probability=round(float(item["sum_probability"]) / hits, 4),
                average_word_coverage=round(float(item["sum_word_coverage"]) / hits, 4),
                average_evidence_score=round(float(item["sum_evidence"]) / hits, 4),
                matched_sentences=hits,
                representative_text=item["representative_text"],
                layer_support=sorted(item["layers"]),
            )
        )

    ranked.sort(
        key=lambda src: (
            (src.average_probability * 0.60) + (src.best_probability * 0.40),
            src.average_evidence_score,
            src.average_word_coverage,
            src.matched_sentences,
        ),
        reverse=True,
    )
    return ranked[:12]


def _fallback_discovered_sources(sources: list[SourceRecord]) -> list[SourceAccuracyResultV2]:
    by_url: dict[str, SourceRecord] = {}
    for source in sources:
        key = _canonical_url(source.url) or f"{source.title}|{hash(source.content)}"
        existing = by_url.get(key)
        if existing is None or len((source.content or "")) > len((existing.content or "")):
            by_url[key] = source

    ranked = sorted(
        by_url.values(),
        key=lambda src: (
            1 if "google" in src.layers else 0,
            1 if "website" in src.layers else 0,
            1 if "academic" in src.layers else 0,
            src.reputation,
            len(src.content or ""),
        ),
        reverse=True,
    )[:12]

    fallback: list[SourceAccuracyResultV2] = []
    for src in ranked:
        preview = " ".join((src.content or "").split())[:240]
        layer_support = sorted(src.layers) if src.layers else ["discovered"]
        fallback.append(
            SourceAccuracyResultV2(
                source_title=src.title,
                source_type=src.source_type,
                source_url=src.url,
                best_probability=0.0,
                average_probability=0.0,
                average_word_coverage=0.0,
                average_evidence_score=0.0,
                matched_sentences=0,
                representative_text=preview,
                layer_support=layer_support,
            )
        )

    return fallback


def _empty_response(*, processing_ms: int, total_sentences: int = 0) -> PlagiarismResponseV2:
    return PlagiarismResponseV2(
        overall_similarity_percent=0.0,
        flagged_sentences=0,
        total_sentences=total_sentences,
        scanned_web_sources=0,
        scanned_book_sources=0,
        processing_ms=processing_ms,
        confidence=0.0,
        confidence_level="low",
        score_breakdown=LayerBreakdown(nlp=0.0, academic=0.0, website=0.0, google=0.0, llm=0.0),
        top_accuracy_sources=[],
        results=[],
    )


def _should_cache_response(response: PlagiarismResponseV2) -> bool:
    # Do not cache zero-evidence outcomes; provider/network availability can vary per request.
    if response.scanned_web_sources <= 0 and response.scanned_book_sources <= 0:
        if response.overall_similarity_percent <= 0.0 and not response.top_accuracy_sources:
            if not any(item.best_score > 0.0 or item.matches for item in response.results):
                return False
    return True


def _build_ucsy_keyword_source() -> SourceAccuracyResultV2:
    return SourceAccuracyResultV2(
        source_title="Web: ucsy.edu.mm (Keyword Triggered)",
        source_type="web",
        source_url="https://www.ucsy.edu.mm/",
        best_probability=0.0,
        average_probability=0.0,
        average_word_coverage=0.0,
        average_evidence_score=0.0,
        matched_sentences=0,
        representative_text="UCSY keyword detected in input. UCSY-targeted retrieval path was prioritized.",
        layer_support=["keyword_triggered", "website"],
    )


def detect_plagiarism_v2(_db: Session, text: str) -> PlagiarismResponseV2:
    settings = get_settings()
    started = perf_counter()
    max_input_words = max(50, int(settings.max_input_words))
    limited_text = _limit_text_words(text, max_words=max_input_words)
    input_word_count = len(_tokenize_words(limited_text))
    analysis_plan = _build_analysis_plan(input_word_count, settings)
    ucsy_keyword_detected = _contains_ucsy_keywords(limited_text)

    cache = get_text_cache()
    cache_key = f"v2.6::{limited_text}"
    cached = cache.get(cache_key)
    if cached and cached.get("version") == "v2":
        return PlagiarismResponseV2(**cached)

    # Primary analysis granularity: overlapping word windows.
    candidate_units = _build_word_units(
        limited_text,
        window_words=analysis_plan.window_words,
        stride_words=analysis_plan.stride_words,
        min_words=max(4, settings.source_min_tokens_per_sentence),
        max_units=analysis_plan.max_units,
    )
    if not candidate_units:
        raw_sentences = split_sentences(limited_text)
        candidate_units = [s.strip() for s in raw_sentences if s and s.strip()]
    if not candidate_units and limited_text.strip():
        candidate_units = [limited_text.strip()]

    if not candidate_units:
        response = _empty_response(processing_ms=int((perf_counter() - started) * 1000), total_sentences=0)
        payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        if _should_cache_response(response):
            cache.put(cache_key, payload)
        return response

    hard_deadline = started + analysis_plan.detection_budget_seconds
    retrieval_deadline = started + (analysis_plan.detection_budget_seconds * analysis_plan.retrieval_budget_ratio)

    raw_sentences = [s.strip() for s in split_sentences(limited_text) if s and s.strip()]
    query_candidates: list[str] = []
    seen_query_candidates: set[str] = set()
    for sentence in [*raw_sentences, *candidate_units]:
        cleaned = " ".join((sentence or "").strip().split())
        if not cleaned or cleaned in seen_query_candidates:
            continue
        seen_query_candidates.add(cleaned)
        query_candidates.append(cleaned)

    query_sentences = _select_query_sentences(
        query_candidates or candidate_units,
        max_queries=min(analysis_plan.max_queries, 4),
        min_chars=max(20, settings.min_sentence_length),
    )
    query_phrases = [q for q in (_build_query_phrase(s) for s in query_sentences) if q]

    layer2 = Layer2AcademicSearch(settings=settings)
    layer3 = Layer3WebsiteSearch(settings=settings)
    layer4 = Layer4GoogleVerification(settings=settings)
    layer5 = Layer5LlmReasoning(settings=settings)
    layer1 = Layer1NlpSimilarityEngine(
        semantic_models=settings.parsed_semantic_models,
        semantic_local_files_only=settings.semantic_local_files_only,
    )
    score_calculator = ScoreCalculator(settings=settings)
    false_positive_filter = FalsePositiveFilter()

    academic_sources: list[AcademicSource] = []
    website_sources: list[WebsiteSource] = []
    google_snippets: list[GoogleSnippetSource] = []
    verification_map: dict[str, set[str]] = {}

    if perf_counter() < retrieval_deadline:
        website_sources = layer3.discover(
            query_phrases,
            deadline=retrieval_deadline,
            prioritize_ucsy=ucsy_keyword_detected,
        )
    if perf_counter() < retrieval_deadline:
        google_snippets = layer4.discover_snippets(
            query_sentences,
            deadline=retrieval_deadline,
            paragraph_text=limited_text,
        )
    if perf_counter() < retrieval_deadline:
        academic_sources = layer2.discover(query_phrases, deadline=retrieval_deadline)
    if perf_counter() < retrieval_deadline:
        verification_map = layer4.verify_sentences(
            candidate_units,
            deadline=retrieval_deadline,
            paragraph_text=limited_text,
        )

    merged_sources = _merge_sources(academic_sources, website_sources, google_snippets)
    scanned_web_sources = sum(1 for source in merged_sources if source.source_type == "web")
    scanned_book_sources = sum(1 for source in merged_sources if source.source_type == "book")

    rows = _extract_candidate_rows(
        merged_sources,
        cap=max(1, max(settings.web_sentence_cap, settings.book_sentence_cap)),
        per_doc_limit=max(5, settings.source_sentences_per_doc),
        min_tokens=max(1, settings.source_min_tokens_per_sentence),
    )

    if not rows:
        empty_top_sources: list[SourceAccuracyResultV2] = []
        if ucsy_keyword_detected:
            empty_top_sources = [_build_ucsy_keyword_source()]
        response = PlagiarismResponseV2(
            overall_similarity_percent=0.0,
            flagged_sentences=0,
            total_sentences=len(candidate_units),
            scanned_web_sources=scanned_web_sources,
            scanned_book_sources=scanned_book_sources,
            processing_ms=int((perf_counter() - started) * 1000),
            confidence=0.25,
            confidence_level="low",
            score_breakdown=LayerBreakdown(nlp=0.0, academic=0.0, website=0.0, google=0.0, llm=0.0),
            top_accuracy_sources=empty_top_sources,
            results=[
                SentenceResultV2(
                    sentence=sentence,
                    flagged=False,
                    best_score=0.0,
                    confidence=0.0,
                    confidence_level="low",
                    matches=[],
                )
                for sentence in candidate_units
            ],
        )
        payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        if _should_cache_response(response):
            cache.put(cache_key, payload)
        return response

    corpus = [row.normalized_text for row in rows]
    layer1.fit(corpus)

    effective_threshold = max(0.35, min(0.9, float(settings.similarity_threshold) * 0.80))
    lexical_k = max(settings.top_matches_per_sentence * settings.lexical_candidate_multiplier, settings.top_matches_per_sentence)
    semantic_k = max(settings.semantic_candidate_k, settings.top_matches_per_sentence * 2)

    results: list[SentenceResultV2] = []
    flagged_sentences = 0

    layer1_values: list[float] = []
    layer2_values: list[float] = []
    layer3_values: list[float] = []
    layer4_values: list[float] = []
    layer5_values: list[float] = []

    sentence_probabilities: list[float] = []
    sentence_lengths: list[int] = []
    sentence_confidences: list[float] = []
    total_ollama_calls = 0

    for sentence in candidate_units:
        if perf_counter() >= hard_deadline:
            results.append(
                SentenceResultV2(
                    sentence=sentence,
                    flagged=False,
                    best_score=0.0,
                    confidence=0.0,
                    confidence_level="low",
                    matches=[],
                )
            )
            sentence_probabilities.append(0.0)
            sentence_lengths.append(max(1, len(sentence.split())))
            sentence_confidences.append(0.0)
            layer1_values.append(0.0)
            layer2_values.append(0.0)
            layer3_values.append(0.0)
            layer4_values.append(0.0)
            layer5_values.append(0.0)
            continue

        normalized = normalize_sentence(sentence)
        if not normalized:
            results.append(
                SentenceResultV2(
                    sentence=sentence,
                    flagged=False,
                    best_score=0.0,
                    confidence=0.0,
                    confidence_level="low",
                    matches=[],
                )
            )
            sentence_probabilities.append(0.0)
            sentence_lengths.append(max(1, len(sentence.split())))
            sentence_confidences.append(0.0)
            layer1_values.append(0.0)
            layer2_values.append(0.0)
            layer3_values.append(0.0)
            layer4_values.append(0.0)
            layer5_values.append(0.0)
            continue

        prepared = layer1.prepare_query(normalized)
        candidate_indices = layer1.top_candidate_indices(prepared, lexical_k=lexical_k, semantic_k=semantic_k)

        scored_matches: list[MatchResultV2] = []
        ollama_calls = 0
        verified_urls = verification_map.get(sentence, set())
        citation_like = _looks_like_citation(sentence)

        for idx in candidate_indices:
            if perf_counter() >= hard_deadline:
                break

            row = rows[idx]
            components = layer1.score_candidate(prepared, idx)

            layer2_score = 1.0 if "academic" in row.layers else 0.0
            layer3_score = row.reputation if "website" in row.layers else 0.0

            canonical = _canonical_url(row.source_url)
            google_verified = canonical in verified_urls if verified_urls else False
            if "google" in row.layers:
                layer4_score = max(0.45, (components.word_coverage * 0.55) + (components.character * 0.45))
            elif google_verified:
                layer4_score = max(0.50, components.evidence)
            else:
                layer4_score = 0.0

            llm_score = max(components.semantic, components.evidence * 0.80)
            llm_label = "weak_signal"
            llm_reasoning = "LLM layer not triggered for this pair."
            if (
                analysis_plan.llm_global_budget > 0
                and total_ollama_calls < analysis_plan.llm_global_budget
                and ollama_calls < max(0, settings.ollama_max_pairs_per_sentence)
                and (components.semantic >= 0.58 or components.lexical >= settings.ollama_min_lexical_score)
            ):
                remaining = max(0.1, hard_deadline - perf_counter())
                llm = layer5.analyze_pair(
                    sentence,
                    row.sentence_text,
                    lexical=components.lexical,
                    semantic=components.semantic,
                    evidence=components.evidence,
                    timeout_seconds=min(analysis_plan.llm_timeout_seconds, remaining),
                )
                llm_score = llm.score
                llm_label = llm.label
                llm_reasoning = llm.rationale
                ollama_calls += 1
                total_ollama_calls += 1

            layer_scores = LayerScores(
                nlp=components.composite,
                academic=layer2_score,
                website=layer3_score,
                google=layer4_score,
                llm=llm_score,
            )
            raw_probability = score_calculator.sentence_probability(layer_scores)

            fp_decision = false_positive_filter.assess(
                sentence=sentence,
                matched_text=row.sentence_text,
                probability=raw_probability,
                lexical=components.lexical,
                semantic=components.semantic,
                word_coverage=components.word_coverage,
                char_similarity=components.character,
            )

            adjusted_probability = fp_decision.adjusted_probability
            if citation_like:
                adjusted_probability = round(_clamp_01(adjusted_probability * 0.78), 4)

            match = MatchResultV2(
                sentence=sentence,
                source_title=row.source_title,
                source_type=row.source_type,
                source_url=row.source_url,
                matched_text=row.sentence_text,
                match_type=_classify_match_type(
                    lexical=components.lexical,
                    semantic=components.semantic,
                    evidence=components.evidence,
                    llm_score=llm_score,
                )
                if llm_label == "weak_signal"
                else llm_label,
                tfidf_score=components.tfidf,
                lexical_score=components.lexical,
                semantic_score=components.semantic,
                overlap_score=components.overlap,
                containment_score=components.containment,
                word_coverage=components.word_coverage,
                char_similarity=components.character,
                evidence_score=components.evidence,
                layer1_score=round(components.composite, 4),
                layer2_score=round(layer2_score, 4),
                layer3_score=round(layer3_score, 4),
                layer4_score=round(layer4_score, 4),
                layer5_score=round(llm_score, 4),
                layer_scores=LayerScoresSchema(
                    nlp=round(components.composite, 4),
                    academic=round(layer2_score, 4),
                    website=round(layer3_score, 4),
                    google=round(layer4_score, 4),
                    llm=round(llm_score, 4),
                ),
                similarity_components=SimilarityComponents(
                    tfidf=components.tfidf,
                    lexical=components.lexical,
                    semantic=components.semantic,
                    syntactic=components.syntactic,
                    character=components.character,
                    evidence=components.evidence,
                    composite=components.composite,
                ),
                false_positive=FalsePositiveInfo(
                    adjusted_probability=fp_decision.adjusted_probability,
                    adjustment_factor=fp_decision.adjustment_factor,
                    suppressed=fp_decision.suppressed,
                    reasons=fp_decision.reasons,
                ),
                llm_reasoning=llm_reasoning,
                plagiarism_probability=adjusted_probability,
            )
            scored_matches.append(match)

        scored_matches.sort(
            key=lambda m: (
                m.plagiarism_probability,
                m.layer1_score,
                m.semantic_score,
                m.evidence_score,
                m.word_coverage,
            ),
            reverse=True,
        )

        filtered = [
            m
            for m in scored_matches
            if m.plagiarism_probability >= settings.min_match_probability
            and (
                m.word_coverage >= (settings.min_word_coverage * 0.80)
                or m.lexical_score >= settings.min_lexical_similarity
                or m.semantic_score >= settings.min_semantic_similarity
                or m.char_similarity >= settings.min_char_similarity
            )
            and not m.false_positive.suppressed
        ]

        non_suppressed = [m for m in scored_matches if not m.false_positive.suppressed]
        dedupe_candidates = filtered or non_suppressed
        if not dedupe_candidates and scored_matches:
            dedupe_candidates = scored_matches[:1]
        unique_by_source: dict[str, MatchResultV2] = {}
        for match in dedupe_candidates:
            key = _canonical_url(match.source_url) or f"{match.source_title}|{match.matched_text[:120]}"
            existing = unique_by_source.get(key)
            if existing is None or match.plagiarism_probability > existing.plagiarism_probability:
                unique_by_source[key] = match

        final_matches = sorted(
            unique_by_source.values(),
            key=lambda m: (
                m.plagiarism_probability,
                m.layer1_score,
                m.semantic_score,
                m.evidence_score,
            ),
            reverse=True,
        )[: settings.top_matches_per_sentence]

        best_score = final_matches[0].plagiarism_probability if final_matches else 0.0
        flagged = best_score >= effective_threshold
        if flagged:
            flagged_sentences += 1

        if final_matches:
            best = final_matches[0]
            layer1_values.append(best.layer1_score)
            layer2_values.append(best.layer2_score)
            layer3_values.append(best.layer3_score)
            layer4_values.append(best.layer4_score)
            layer5_values.append(best.layer5_score)
            layer_vector = [
                best.layer1_score,
                best.layer2_score,
                best.layer3_score,
                best.layer4_score,
                best.layer5_score,
            ]
            sentence_conf = score_calculator.confidence(layer_vector)
            sentence_confidence = sentence_conf.value
            sentence_conf_level = sentence_conf.level
        else:
            layer1_values.append(0.0)
            layer2_values.append(0.0)
            layer3_values.append(0.0)
            layer4_values.append(0.0)
            layer5_values.append(0.0)
            sentence_confidence = 0.0
            sentence_conf_level = "low"

        results.append(
            SentenceResultV2(
                sentence=sentence,
                flagged=flagged,
                best_score=round(best_score, 4),
                confidence=round(sentence_confidence, 4),
                confidence_level=sentence_conf_level,
                matches=final_matches,
            )
        )

        sentence_probabilities.append(best_score)
        sentence_lengths.append(max(1, len(sentence.split())))
        sentence_confidences.append(sentence_confidence)

    total_sentences = len(results)
    overall_probability = score_calculator.overall_probability(sentence_probabilities, sentence_lengths)
    overall_similarity_percent = round(overall_probability * 100.0, 2)

    avg_confidence = (sum(sentence_confidences) / total_sentences) if total_sentences else 0.0
    overall_layer_confidence = score_calculator.confidence(
        [
            sum(layer1_values) / max(1, total_sentences),
            sum(layer2_values) / max(1, total_sentences),
            sum(layer3_values) / max(1, total_sentences),
            sum(layer4_values) / max(1, total_sentences),
            sum(layer5_values) / max(1, total_sentences),
        ]
    )
    confidence = round(_clamp_01((avg_confidence * 0.70) + (overall_layer_confidence.value * 0.30)), 4)

    top_sources = _aggregate_top_sources(results)
    if not top_sources and merged_sources:
        top_sources = _fallback_discovered_sources(merged_sources)
    if ucsy_keyword_detected:
        ucsy_top = [source for source in top_sources if _is_ucsy_url(source.source_url)]
        if ucsy_top:
            non_ucsy = [source for source in top_sources if not _is_ucsy_url(source.source_url)]
            top_sources = [*ucsy_top, *non_ucsy][:12]
        else:
            ucsy_from_discovery = [
                source
                for source in _fallback_discovered_sources(merged_sources)
                if _is_ucsy_url(source.source_url)
            ]
            if ucsy_from_discovery:
                top_sources = [ucsy_from_discovery[0], *top_sources][:12]
            else:
                top_sources = [_build_ucsy_keyword_source(), *top_sources][:12]
    processing_ms = int((perf_counter() - started) * 1000)

    response = PlagiarismResponseV2(
        overall_similarity_percent=overall_similarity_percent,
        flagged_sentences=flagged_sentences,
        total_sentences=total_sentences,
        scanned_web_sources=scanned_web_sources,
        scanned_book_sources=scanned_book_sources,
        processing_ms=processing_ms,
        confidence=confidence,
        confidence_level=score_calculator.confidence_level(confidence),
        score_breakdown=LayerBreakdown(
            nlp=round(sum(layer1_values) / max(1, total_sentences), 4),
            academic=round(sum(layer2_values) / max(1, total_sentences), 4),
            website=round(sum(layer3_values) / max(1, total_sentences), 4),
            google=round(sum(layer4_values) / max(1, total_sentences), 4),
            llm=round(sum(layer5_values) / max(1, total_sentences), 4),
        ),
        top_accuracy_sources=top_sources,
        results=results,
    )

    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    if _should_cache_response(response):
        cache.put(cache_key, payload)
    return response


def detect_plagiarism(_db: Session, text: str) -> PlagiarismResponseV2:
    """Backwards-compatible alias used by routes/services."""
    return detect_plagiarism_v2(_db, text)
