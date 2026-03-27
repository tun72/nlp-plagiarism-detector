from __future__ import annotations


def _token_set(text: str) -> set[str]:
    return {token for token in text.split() if token}


def token_overlap(a: str, b: str) -> float:
    tokens_a = _token_set(a)
    tokens_b = _token_set(b)
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union else 0.0


def containment_score(a: str, b: str) -> float:
    tokens_a = a.split()
    tokens_b = b.split()
    if not tokens_a or not tokens_b:
        return 0.0
    short, long_ = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    short_set = set(short)
    return len(short_set.intersection(long_)) / len(short_set) if short_set else 0.0


def word_coverage_score(query_text: str, candidate_text: str) -> float:
    query_tokens = _token_set(query_text)
    if not query_tokens:
        return 0.0
    candidate_tokens = _token_set(candidate_text)
    if not candidate_tokens:
        return 0.0
    return len(query_tokens.intersection(candidate_tokens)) / len(query_tokens)


def char_ngram_similarity(a: str, b: str, n: int = 4) -> float:
    text_a = (a or "").strip()
    text_b = (b or "").strip()
    if not text_a or not text_b:
        return 0.0

    size = max(2, int(n))
    if len(text_a) < size or len(text_b) < size:
        return 1.0 if text_a == text_b else 0.0

    grams_a = {text_a[i : i + size] for i in range(len(text_a) - size + 1)}
    grams_b = {text_b[i : i + size] for i in range(len(text_b) - size + 1)}
    if not grams_a or not grams_b:
        return 0.0

    inter = len(grams_a.intersection(grams_b))
    union = len(grams_a.union(grams_b))
    return inter / union if union else 0.0
