from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass
import math
import re
from time import perf_counter
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.cache import get_text_cache
from app.core.config import get_settings
from app.nlp.ai_plagiarism_model import AIPlagiarismDetectorModel
from app.nlp.features import char_ngram_similarity, containment_score, token_overlap, word_coverage_score
from app.nlp.ollama_engine import OllamaSemanticEngine
from app.nlp.preprocessing import normalize_sentence, split_sentences
from app.nlp.semantic_engine import SemanticEngine
from app.nlp.tfidf_engine import TfidfEngine
from app.schemas.plagiarism import MatchResult, PlagiarismResponse, SentenceResult, SourceAccuracyResult
from app.scraper.search_client import SearchClient, WebSource

ACADEMIC_HOSTS = {
    "arxiv.org",
    "openalex.org",
    "api.openalex.org",
    "crossref.org",
    "api.crossref.org",
    "semanticscholar.org",
    "api.semanticscholar.org",
    "scholar.google.com",
    "researchgate.net",
    "pubmed.ncbi.nlm.nih.gov",
    "acm.org",
    "ieeexplore.ieee.org",
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "wiley.com",
    "tandfonline.com",
    "jstor.org",
}

CITATION_PATTERN = re.compile(
    r"(\[[0-9,\-\s]+\])|(\([A-Z][A-Za-z\-\s]+,\s*(19|20)\d{2}[a-z]?\))|(\"[^\"]{10,}\")"
)


@dataclass
class LayeredSource:
    title: str
    url: str
    content: str
    source_type: str
    layers: set[str]


@dataclass
class CandidateRow:
    sentence_text: str
    normalized_text: str
    source_title: str
    source_url: str
    source_type: str
    layers: set[str]


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


def _host_from_url(url: str) -> str:
    host = urlparse((url or "").strip()).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _looks_like_citation(sentence: str) -> bool:
    return bool(CITATION_PATTERN.search(sentence or ""))


def _source_is_academic(source: WebSource) -> bool:
    if (source.source_type or "").lower() == "book":
        return True
    host = _host_from_url(source.url)
    return host in ACADEMIC_HOSTS or any(host.endswith(f".{academic}") for academic in ACADEMIC_HOSTS)


def _build_query_phrase(sentence: str) -> str:
    raw = " ".join((sentence or "").strip().split())
    if not raw:
        return ""
    normalized = normalize_sentence(raw)
    if not normalized:
        return raw

    raw_terms = {token.lower() for token in raw.split()}
    extra_terms = [token for token in normalized.split() if token.lower() not in raw_terms]
    if not extra_terms:
        return raw
    return f"{raw} {' '.join(extra_terms)}".strip()


def _select_query_sentences(sentences: list[str], max_queries: int, min_chars: int) -> list[str]:
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

    scored = []
    for sentence in deduped:
        tokens = normalize_sentence(sentence).split()
        unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0
        length_score = min(1.0, len(sentence) / 180.0)
        informational = (unique_ratio * 0.65) + (length_score * 0.35)
        if len(sentence) < min_chars:
            informational *= 0.75
        scored.append((informational, sentence))

    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    top = [s for _, s in scored[: max(1, max_queries)]]
    return top


def _normalize_layer_weights() -> dict[str, float]:
    settings = get_settings()
    weights = {
        "nlp": max(0.0, float(settings.plagiarism_weight_nlp)),
        "academic": max(0.0, float(settings.plagiarism_weight_academic)),
        "web": max(0.0, float(settings.plagiarism_weight_web)),
        "google": max(0.0, float(settings.plagiarism_weight_google)),
        "llm": max(0.0, float(settings.plagiarism_weight_llm)),
    }
    total = sum(weights.values())
    if total <= 0:
        return {"nlp": 0.4, "academic": 0.2, "web": 0.2, "google": 0.1, "llm": 0.1}
    return {k: v / total for k, v in weights.items()}


def _classify_match_type(
    lexical: float,
    semantic: float,
    evidence: float,
    char_sim: float,
    word_coverage: float,
    llm_score: float,
) -> str:
    if lexical >= 0.82 and char_sim >= 0.78 and word_coverage >= 0.74:
        return "direct_copy"
    if llm_score >= 0.78 and semantic >= 0.70 and lexical < 0.55:
        return "ai_rewrite"
    if semantic >= 0.78 and evidence >= 0.55 and lexical < 0.78:
        return "paraphrased"
    if semantic >= 0.66 and evidence >= 0.48:
        return "semantic"
    return "possible_overlap"


def _aggregate_top_sources(results: list[SentenceResult]) -> list[SourceAccuracyResult]:
    aggregate: dict[str, dict] = {}
    for sentence in results:
        for match in sentence.matches:
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
                    "sum_evidence_score": match.evidence_score,
                    "matched_sentences": 1,
                    "representative_text": match.matched_text,
                }
                continue

            current["matched_sentences"] += 1
            current["sum_probability"] += match.plagiarism_probability
            current["sum_word_coverage"] += match.word_coverage
            current["sum_evidence_score"] += match.evidence_score
            if match.plagiarism_probability > current["best_probability"]:
                current["best_probability"] = match.plagiarism_probability
                current["representative_text"] = match.matched_text

    ranked: list[SourceAccuracyResult] = []
    for data in aggregate.values():
        hits = max(1, int(data["matched_sentences"]))
        ranked.append(
            SourceAccuracyResult(
                source_title=data["source_title"],
                source_type=data["source_type"],
                source_url=data["source_url"],
                best_probability=round(float(data["best_probability"]), 4),
                average_probability=round(float(data["sum_probability"]) / hits, 4),
                average_word_coverage=round(float(data["sum_word_coverage"]) / hits, 4),
                average_evidence_score=round(float(data["sum_evidence_score"]) / hits, 4),
                matched_sentences=hits,
                representative_text=data["representative_text"],
            )
        )

    ranked.sort(
        key=lambda source: (
            (source.average_probability * 0.60) + (source.best_probability * 0.40),
            source.average_evidence_score,
            source.average_word_coverage,
            source.matched_sentences,
        ),
        reverse=True,
    )
    return ranked[:12]


def _discover_for_query(
    search_client: SearchClient,
    query: str,
    per_query_budget: float,
) -> list[LayeredSource]:
    settings = get_settings()
    discovered: list[LayeredSource] = []
    local_deadline = perf_counter() + max(0.2, per_query_budget)

    def step_timeout(default_value: float) -> float:
        remaining = local_deadline - perf_counter()
        if remaining <= 0:
            return 0.0
        return max(0.2, min(default_value, remaining))

    def append_sources(items: list[WebSource], layer_name: str) -> None:
        for src in items:
            text = (src.content or "").strip()
            if not text:
                continue
            discovered.append(
                LayeredSource(
                    title=src.title,
                    url=src.url,
                    content=text,
                    source_type=src.source_type,
                    layers={layer_name},
                )
            )

    try:
        t = step_timeout(max(1.0, settings.web_fetch_timeout))
        if t > 0:
            academic_domains = [
                "arxiv.org",
                "openalex.org",
                "crossref.org",
                "semanticscholar.org",
                "pubmed.ncbi.nlm.nih.gov",
                "acm.org",
                "ieeexplore.ieee.org",
                "springer.com",
                "nature.com",
                "sciencedirect.com",
            ]
            academic_sources = search_client.discover_sources(
                query=query,
                max_results=max(2, settings.web_max_results),
                timeout=t,
                domains=academic_domains,
                source_type="web",
                overall_timeout=t,
            )
            append_sources(academic_sources, "academic")
    except Exception:
        pass

    if settings.allow_book_search:
        try:
            t = step_timeout(max(1.0, settings.book_fetch_timeout))
            if t > 0:
                book_sources = search_client.discover_sources(
                    query=query,
                    max_results=max(2, settings.book_max_results),
                    timeout=t,
                    domains=settings.parsed_book_domains,
                    source_type="book",
                    overall_timeout=t,
                )
                append_sources(book_sources, "academic")
        except Exception:
            pass

    if settings.allow_web_search:
        try:
            t = step_timeout(max(1.0, settings.web_fetch_timeout))
            if t > 0:
                website_sources = search_client.discover_sources(
                    query=query,
                    max_results=max(2, settings.web_max_results),
                    timeout=t,
                    domains=None,
                    source_type="web",
                    overall_timeout=t,
                )
                append_sources(website_sources, "website")
        except Exception:
            pass

        try:
            t = step_timeout(max(1.0, settings.web_fetch_timeout))
            if t > 0:
                google_sources = search_client.discover_google_search_sources(
                    query=f'"{query}"',
                    timeout=t,
                    limit=max(2, settings.web_max_results),
                    domains=None,
                )
                append_sources(google_sources, "google")
        except Exception:
            pass

    return discovered


def _dedupe_layered_sources(items: list[LayeredSource]) -> list[LayeredSource]:
    layer_priority = {"google": 3, "academic": 2, "website": 1}
    by_key: dict[str, LayeredSource] = {}

    for item in items:
        canonical = _canonical_url(item.url)
        key = canonical or f"{item.title}|{hash(item.content)}"
        current = by_key.get(key)
        if current is None:
            by_key[key] = item
            continue

        current.layers.update(item.layers)
        if len(item.content) > len(current.content):
            current.content = item.content
            current.title = item.title or current.title
            current.source_type = item.source_type or current.source_type

        current_best = max((layer_priority.get(layer, 0) for layer in current.layers), default=0)
        incoming_best = max((layer_priority.get(layer, 0) for layer in item.layers), default=0)
        if incoming_best > current_best:
            current.url = item.url or current.url

        if _source_is_academic(
            WebSource(title=current.title, url=current.url, content=current.content, source_type=current.source_type)
        ):
            current.layers.add("academic")

    return list(by_key.values())


def _collect_layered_sources(candidate_sentences: list[str], deadline: float) -> tuple[list[LayeredSource], int, int]:
    settings = get_settings()
    if perf_counter() >= deadline:
        return [], 0, 0

    max_queries = settings.max_query_sentences if settings.max_query_sentences > 0 else 6
    selected_sentences = _select_query_sentences(
        candidate_sentences,
        max_queries=max_queries,
        min_chars=max(20, settings.min_sentence_length),
    )
    query_phrases = [phrase for phrase in (_build_query_phrase(sentence) for sentence in selected_sentences) if phrase]
    if not query_phrases:
        return [], 0, 0

    worker_count = max(1, min(settings.query_parallelism, len(query_phrases)))
    remaining_total = max(0.1, deadline - perf_counter())
    query_waves = max(1, math.ceil(len(query_phrases) / worker_count))
    per_query_budget = max(0.6, min(float(settings.web_fetch_timeout) + 1.5, remaining_total / query_waves))

    search_client = SearchClient()
    discovered: list[LayeredSource] = []
    futures = {}
    executor = ThreadPoolExecutor(max_workers=worker_count)
    for query in query_phrases:
        futures[executor.submit(_discover_for_query, search_client, query, per_query_budget)] = query

    try:
        wait_timeout = max(0.1, deadline - perf_counter())
        for future in as_completed(futures, timeout=wait_timeout):
            if perf_counter() >= deadline:
                break
            try:
                discovered.extend(future.result(timeout=max(0.1, deadline - perf_counter())))
            except Exception:
                continue
    except FuturesTimeoutError:
        pass
    finally:
        for future in futures:
            if not future.done():
                future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    unique_sources = _dedupe_layered_sources(discovered)
    scanned_web_sources = len([source for source in unique_sources if source.source_type == "web"])
    scanned_book_sources = len([source for source in unique_sources if source.source_type == "book"])
    return unique_sources, scanned_web_sources, scanned_book_sources


def _extract_candidate_rows(sources: list[LayeredSource], cap: int, deadline: float) -> list[CandidateRow]:
    settings = get_settings()
    rows: list[CandidateRow] = []
    seen_normalized: set[str] = set()

    for source in sources:
        if perf_counter() >= deadline and rows:
            break

        text = (source.content or "")[: settings.source_text_char_limit]
        if not text.strip():
            continue

        source_sentences = split_sentences(text)
        if not source_sentences:
            source_sentences = [text]

        per_doc_limit = settings.source_sentences_per_doc
        if "google" in source.layers:
            per_doc_limit = min(5, per_doc_limit)

        added = 0
        for sentence in source_sentences:
            if perf_counter() >= deadline and rows:
                break

            normalized = normalize_sentence(sentence)
            if not normalized:
                continue
            if len(normalized.split()) < max(1, int(settings.source_min_tokens_per_sentence)):
                continue
            if normalized in seen_normalized:
                continue

            seen_normalized.add(normalized)
            rows.append(
                CandidateRow(
                    sentence_text=sentence,
                    normalized_text=normalized,
                    source_title=source.title,
                    source_url=source.url,
                    source_type=source.source_type,
                    layers=set(source.layers),
                )
            )
            added += 1

            if added >= per_doc_limit or len(rows) >= cap:
                break
        if len(rows) >= cap:
            break

    return rows


def _build_google_verification_map(
    candidate_sentences: list[str],
    deadline: float,
) -> dict[str, set[str]]:
    settings = get_settings()
    if perf_counter() >= deadline:
        return {}
    if settings.google_verification_sentences <= 0:
        return {}

    selected = _select_query_sentences(
        candidate_sentences,
        max_queries=settings.google_verification_sentences,
        min_chars=max(20, settings.min_sentence_length),
    )
    if not selected:
        return {}

    client = SearchClient()
    verification: dict[str, set[str]] = {}
    for sentence in selected:
        if perf_counter() >= deadline:
            break
        query = f'"{sentence}"'
        remaining = max(0.2, min(float(settings.web_fetch_timeout), deadline - perf_counter()))
        if remaining <= 0:
            break
        try:
            urls = client.search_urls(query=query, max_results=4, timeout=remaining, domains=None)
        except Exception:
            urls = []
        verification[sentence] = {_canonical_url(url) for url in urls if url}

    return verification


def detect_plagiarism(_db: Session, text: str) -> PlagiarismResponse:
    started = perf_counter()
    settings = get_settings()
    text_cache = get_text_cache()

    cached = text_cache.get(text)
    if cached:
        return PlagiarismResponse(**cached)

    hard_deadline = started + float(settings.max_detection_seconds)
    retrieval_deadline = started + (float(settings.max_detection_seconds) * 0.55)

    raw_sentences = split_sentences(text)
    candidate_sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]
    if not candidate_sentences and text.strip():
        candidate_sentences = [text.strip()]

    if not candidate_sentences:
        response = PlagiarismResponse(
            overall_similarity_percent=0.0,
            flagged_sentences=0,
            total_sentences=0,
            scanned_web_sources=0,
            scanned_book_sources=0,
            processing_ms=int((perf_counter() - started) * 1000),
            top_accuracy_sources=[],
            results=[],
        )
        payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        text_cache.put(text, payload)
        return response

    layered_sources, scanned_web_sources, scanned_book_sources = _collect_layered_sources(candidate_sentences, retrieval_deadline)
    sentence_cap = max(settings.web_sentence_cap, settings.book_sentence_cap)
    rows = _extract_candidate_rows(layered_sources, cap=max(1, sentence_cap), deadline=retrieval_deadline)

    if not rows and perf_counter() < retrieval_deadline:
        fallback_client = SearchClient()
        fallback_query = _build_query_phrase(candidate_sentences[0])
        fallback_sources: list[LayeredSource] = []
        timeout_budget = max(0.8, min(float(settings.web_fetch_timeout), retrieval_deadline - perf_counter()))

        if settings.allow_web_search and timeout_budget > 0:
            try:
                for src in fallback_client.discover_sources(
                    query=fallback_query,
                    max_results=max(2, settings.web_max_results),
                    timeout=timeout_budget,
                    domains=settings.parsed_web_domains,
                    source_type="web",
                    overall_timeout=timeout_budget,
                ):
                    fallback_sources.append(
                        LayeredSource(
                            title=src.title,
                            url=src.url,
                            content=src.content,
                            source_type=src.source_type,
                            layers={"website", "google"},
                        )
                    )
            except Exception:
                pass

        if settings.allow_book_search and timeout_budget > 0:
            try:
                for src in fallback_client.discover_sources(
                    query=fallback_query,
                    max_results=max(2, settings.book_max_results),
                    timeout=min(timeout_budget, float(settings.book_fetch_timeout)),
                    domains=settings.parsed_book_domains,
                    source_type="book",
                    overall_timeout=timeout_budget,
                ):
                    fallback_sources.append(
                        LayeredSource(
                            title=src.title,
                            url=src.url,
                            content=src.content,
                            source_type=src.source_type,
                            layers={"academic"},
                        )
                    )
            except Exception:
                pass

        if fallback_sources:
            layered_sources = _dedupe_layered_sources([*layered_sources, *fallback_sources])
            scanned_web_sources = len([source for source in layered_sources if source.source_type == "web"])
            scanned_book_sources = len([source for source in layered_sources if source.source_type == "book"])
            rows = _extract_candidate_rows(layered_sources, cap=max(1, sentence_cap), deadline=retrieval_deadline)

    corpus = [row.normalized_text for row in rows]
    tfidf = TfidfEngine()
    tfidf.fit(corpus)

    semantic_engines: list[SemanticEngine] = []
    if settings.enable_semantic_model:
        for model_name in settings.parsed_semantic_models:
            semantic_engines.append(SemanticEngine(model_name, local_files_only=settings.semantic_local_files_only))
    if not semantic_engines:
        semantic_engines.append(SemanticEngine("", local_files_only=settings.semantic_local_files_only))
    for engine in semantic_engines:
        engine.fit(corpus)

    ai_model = AIPlagiarismDetectorModel(settings.calibrator_model_path)
    layer_weights = _normalize_layer_weights()

    ollama_engine: OllamaSemanticEngine | None = None
    if settings.enable_ollama_semantic:
        ollama_engine = OllamaSemanticEngine(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_model_name,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        if not ollama_engine.is_available(timeout_seconds=min(0.8, settings.ollama_timeout_seconds)):
            ollama_engine = None

    verification_map = _build_google_verification_map(candidate_sentences, deadline=retrieval_deadline)
    effective_threshold = max(0.42, min(0.9, float(settings.similarity_threshold) * 0.80))

    results: list[SentenceResult] = []
    flagged_sentences = 0

    nlp_layer_values: list[float] = []
    llm_layer_values: list[float] = []
    academic_hits = 0
    web_hits = 0
    google_hits = 0

    lexical_k = max(settings.top_matches_per_sentence * settings.lexical_candidate_multiplier, settings.top_matches_per_sentence)
    semantic_k = max(settings.semantic_candidate_k, settings.top_matches_per_sentence * 2)

    for sentence in candidate_sentences:
        if perf_counter() >= hard_deadline:
            results.append(SentenceResult(sentence=sentence, flagged=False, best_score=0.0, matches=[]))
            continue

        normalized = normalize_sentence(sentence)
        if not normalized:
            results.append(SentenceResult(sentence=sentence, flagged=False, best_score=0.0, matches=[]))
            nlp_layer_values.append(0.0)
            llm_layer_values.append(0.0)
            continue

        if not rows:
            results.append(SentenceResult(sentence=sentence, flagged=False, best_score=0.0, matches=[]))
            nlp_layer_values.append(0.0)
            llm_layer_values.append(0.0)
            continue

        prepared_tfidf = tfidf.prepare_query(normalized)
        lexical_matches = tfidf.top_k_from_prepared(prepared_tfidf, k=lexical_k)
        lexical_by_index = {match.index: match.score for match in lexical_matches}

        prepared_semantic_queries = [engine.prepare_query(normalized) for engine in semantic_engines]
        semantic_candidates: dict[int, float] = {}
        for engine_idx, engine in enumerate(semantic_engines):
            for index, score in engine.top_k_from_prepared(prepared_semantic_queries[engine_idx], k=semantic_k):
                existing = semantic_candidates.get(index, 0.0)
                if score > existing:
                    semantic_candidates[index] = score

        candidate_indices = set(lexical_by_index) | set(semantic_candidates)
        if not candidate_indices:
            results.append(SentenceResult(sentence=sentence, flagged=False, best_score=0.0, matches=[]))
            nlp_layer_values.append(0.0)
            llm_layer_values.append(0.0)
            continue

        scored_matches: list[MatchResult] = []
        ollama_calls = 0
        verified_urls = verification_map.get(sentence, set())
        citation_like = _looks_like_citation(sentence)
        token_count = len(normalized.split())

        for index in candidate_indices:
            if perf_counter() >= hard_deadline:
                break

            row = rows[index]
            lexical_score = lexical_by_index.get(index)
            if lexical_score is None:
                lexical_score = tfidf.similarity_from_prepared(prepared_tfidf, index)

            semantic_scores = []
            for engine_idx, engine in enumerate(semantic_engines):
                semantic_scores.append(engine.similarity_from_prepared(prepared_semantic_queries[engine_idx], index))
            semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
            semantic_score = max(semantic_score, semantic_candidates.get(index, 0.0))

            overlap = token_overlap(normalized, row.normalized_text)
            containment = containment_score(normalized, row.normalized_text)
            word_coverage = word_coverage_score(normalized, row.normalized_text)
            char_similarity = char_ngram_similarity(normalized, row.normalized_text)
            evidence_score = (word_coverage * 0.45) + (containment * 0.25) + (overlap * 0.20) + (char_similarity * 0.10)

            layer1_score = ai_model.predict_probability(
                lexical_score,
                semantic_score,
                overlap,
                containment,
                word_coverage=word_coverage,
                char_similarity=char_similarity,
            )

            layer2_score = 1.0 if "academic" in row.layers else 0.0
            layer3_score = 1.0 if "website" in row.layers else 0.0

            canonical_source_url = _canonical_url(row.source_url)
            is_google_verified = canonical_source_url in verified_urls if verified_urls else False
            if "google" in row.layers:
                layer4_score = max(0.45, (word_coverage * 0.55) + (char_similarity * 0.45))
            elif is_google_verified:
                layer4_score = max(0.50, evidence_score)
            else:
                layer4_score = 0.0

            layer5_score = max(semantic_score, evidence_score * 0.80)
            if (
                ollama_engine is not None
                and ollama_calls < max(0, settings.ollama_max_pairs_per_sentence)
                and (semantic_score >= 0.58 or lexical_score >= settings.ollama_min_lexical_score)
            ):
                remaining = max(0.0, hard_deadline - perf_counter())
                if remaining > 0.05:
                    ollama_score = ollama_engine.similarity(
                        sentence_a=sentence,
                        sentence_b=row.sentence_text,
                        timeout_seconds=min(settings.ollama_timeout_seconds, remaining),
                    )
                    ollama_calls += 1
                    if ollama_score is not None:
                        blend = _clamp_01(float(settings.ollama_semantic_blend))
                        layer5_score = ((1.0 - blend) * layer5_score) + (blend * ollama_score)
                        semantic_score = ((1.0 - blend) * semantic_score) + (blend * ollama_score)

            probability = (
                (layer_weights["nlp"] * layer1_score)
                + (layer_weights["academic"] * layer2_score)
                + (layer_weights["web"] * layer3_score)
                + (layer_weights["google"] * layer4_score)
                + (layer_weights["llm"] * layer5_score)
            )

            if citation_like:
                probability *= 0.72
            if token_count < max(5, settings.source_min_tokens_per_sentence + 1):
                probability *= 0.60
            if lexical_score < 0.05 and semantic_score < 0.55 and evidence_score < 0.35:
                probability *= 0.55

            probability = _clamp_01(probability)
            match_type = _classify_match_type(
                lexical=lexical_score,
                semantic=semantic_score,
                evidence=evidence_score,
                char_sim=char_similarity,
                word_coverage=word_coverage,
                llm_score=layer5_score,
            )

            scored_matches.append(
                MatchResult(
                    sentence=sentence,
                    source_title=row.source_title,
                    source_type=row.source_type,
                    source_url=row.source_url,
                    matched_text=row.sentence_text,
                    match_type=match_type,
                    lexical_score=round(lexical_score, 4),
                    semantic_score=round(semantic_score, 4),
                    overlap_score=round(overlap, 4),
                    containment_score=round(containment, 4),
                    word_coverage=round(word_coverage, 4),
                    char_similarity=round(char_similarity, 4),
                    evidence_score=round(evidence_score, 4),
                    layer1_score=round(layer1_score, 4),
                    layer2_score=round(layer2_score, 4),
                    layer3_score=round(layer3_score, 4),
                    layer4_score=round(layer4_score, 4),
                    layer5_score=round(layer5_score, 4),
                    plagiarism_probability=round(probability, 4),
                )
            )

        scored_matches.sort(
            key=lambda match: (
                match.plagiarism_probability,
                match.layer1_score or 0.0,
                match.semantic_score,
                match.evidence_score,
                match.word_coverage,
            ),
            reverse=True,
        )

        filtered_matches = [
            match
            for match in scored_matches
            if match.plagiarism_probability >= settings.min_match_probability
            and (
                match.word_coverage >= (settings.min_word_coverage * 0.80)
                or match.lexical_score >= settings.min_lexical_similarity
                or match.semantic_score >= settings.min_semantic_similarity
                or match.char_similarity >= settings.min_char_similarity
            )
        ]

        deduped_candidates = filtered_matches or scored_matches
        unique_matches: dict[str, MatchResult] = {}
        for match in deduped_candidates:
            key = _canonical_url(match.source_url) or f"{match.source_title}|{match.matched_text[:120]}"
            existing = unique_matches.get(key)
            if existing is None or match.plagiarism_probability > existing.plagiarism_probability:
                unique_matches[key] = match

        final_matches = sorted(
            unique_matches.values(),
            key=lambda match: (
                match.plagiarism_probability,
                match.layer1_score or 0.0,
                match.semantic_score,
                match.evidence_score,
            ),
            reverse=True,
        )[: settings.top_matches_per_sentence]
        best_score = final_matches[0].plagiarism_probability if final_matches else 0.0
        flagged = best_score >= effective_threshold
        if flagged:
            flagged_sentences += 1

        best_match = final_matches[0] if final_matches else None
        if best_match is not None:
            nlp_layer_values.append(best_match.layer1_score or 0.0)
            llm_layer_values.append(best_match.layer5_score or 0.0)
            if (best_match.layer2_score or 0.0) > 0:
                academic_hits += 1
            if (best_match.layer3_score or 0.0) > 0:
                web_hits += 1
            if (best_match.layer4_score or 0.0) > 0:
                google_hits += 1
        else:
            nlp_layer_values.append(0.0)
            llm_layer_values.append(0.0)

        results.append(
            SentenceResult(
                sentence=sentence,
                flagged=flagged,
                best_score=round(best_score, 4),
                matches=final_matches,
            )
        )

    total_sentences = len(results)
    if total_sentences == 0:
        overall_similarity_percent = 0.0
    else:
        nlp_avg = sum(nlp_layer_values) / total_sentences
        llm_avg = sum(llm_layer_values) / total_sentences
        academic_ratio = academic_hits / total_sentences
        web_ratio = web_hits / total_sentences
        google_ratio = google_hits / total_sentences

        weighted_score = (
            (layer_weights["nlp"] * nlp_avg)
            + (layer_weights["academic"] * academic_ratio)
            + (layer_weights["web"] * web_ratio)
            + (layer_weights["google"] * google_ratio)
            + (layer_weights["llm"] * llm_avg)
        )
        overall_similarity_percent = round(_clamp_01(weighted_score) * 100.0, 2)

    top_sources = _aggregate_top_sources(results)
    processing_ms = int((perf_counter() - started) * 1000)

    response = PlagiarismResponse(
        overall_similarity_percent=overall_similarity_percent,
        flagged_sentences=flagged_sentences,
        total_sentences=total_sentences,
        scanned_web_sources=scanned_web_sources,
        scanned_book_sources=scanned_book_sources,
        processing_ms=processing_ms,
        top_accuracy_sources=top_sources,
        results=results,
    )
    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    text_cache.put(text, payload)
    return response
