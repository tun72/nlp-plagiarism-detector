from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from urllib.parse import urlparse

from app.core.config import Settings, get_settings
from app.scraper.search_client import SearchClient


def _canonical_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{host}{path}"


@dataclass
class GoogleSnippetSource:
    title: str
    url: str
    content: str


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def _tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text or "")


def _build_paragraph_queries(text: str, *, max_queries: int) -> list[str]:
    words = _tokenize_words(text)
    if not words:
        return []

    queries: list[str] = []

    # Head/tail phrase queries improve full-paragraph evidence recall.
    head = " ".join(words[: min(28, len(words))]).strip()
    if head:
        queries.append(f'"{head}"')

    if len(words) > 28:
        tail = " ".join(words[-28:]).strip()
        if tail and tail != head:
            queries.append(f'"{tail}"')

    # Word-chunk paragraph queries (quoted + loose) to capture partial copied spans.
    chunk_words = 8 if len(words) >= 80 else 6
    stride = max(3, chunk_words // 2)
    for start in range(0, len(words), stride):
        chunk = words[start : start + chunk_words]
        if len(chunk) < 4:
            break
        joined = " ".join(chunk).strip()
        if not joined:
            continue
        queries.append(f'"{joined}"')
        queries.append(joined)
        if start + chunk_words >= len(words):
            break
        if len(queries) >= max(2, max_queries) * 2:
            break

    deduped: list[str] = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped[: max(2, max_queries)]


class Layer4GoogleVerification:
    """Google verification layer over sentence queries and paragraph word-chunk queries."""

    def __init__(self, settings: Settings | None = None, client: SearchClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or SearchClient()

    def verify_sentences(self, sentences: list[str], *, deadline: float, paragraph_text: str = "") -> dict[str, set[str]]:
        if not self.settings.allow_web_search:
            return {}

        sentence_query_limit = max(1, int(self.settings.google_verification_sentences))
        selected = [s.strip() for s in sentences if s and s.strip()][:sentence_query_limit]
        paragraph_queries = _build_paragraph_queries(
            paragraph_text,
            max_queries=max(2, sentence_query_limit),
        )

        result: dict[str, set[str]] = {}
        paragraph_urls: set[str] = set()

        for sentence in selected:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break

            timeout = max(0.3, min(float(self.settings.web_fetch_timeout), remaining))
            try:
                urls = self.client.search_urls(
                    query=sentence,
                    max_results=max(8, self.settings.web_max_results * 2),
                    timeout=timeout,
                    domains=None,
                )
            except Exception:
                urls = []
            result[sentence] = {_canonical_url(url) for url in urls if url}

        # Global paragraph verification URLs are merged into each sentence evidence set.
        for query in paragraph_queries:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            timeout = max(0.3, min(float(self.settings.web_fetch_timeout), remaining))
            try:
                urls = self.client.search_urls(
                    query=query,
                    max_results=max(8, self.settings.web_max_results * 2),
                    timeout=timeout,
                    domains=None,
                )
            except Exception:
                urls = []
            for url in urls:
                canonical = _canonical_url(url)
                if canonical:
                    paragraph_urls.add(canonical)

        if paragraph_urls:
            for sentence in selected:
                result.setdefault(sentence, set()).update(paragraph_urls)

        return result

    def discover_snippets(self, queries: list[str], *, deadline: float, paragraph_text: str = "") -> list[GoogleSnippetSource]:
        if not self.settings.allow_web_search:
            return []

        sentence_query_limit = max(1, int(self.settings.google_verification_sentences))
        paragraph_queries = _build_paragraph_queries(
            paragraph_text,
            max_queries=max(2, sentence_query_limit),
        )
        query_list: list[str] = []
        for query in [*queries, *paragraph_queries]:
            cleaned = (query or "").strip()
            if cleaned and cleaned not in query_list:
                query_list.append(cleaned)
        query_list = query_list[: max(2, sentence_query_limit * 2)]

        snippets: dict[str, GoogleSnippetSource] = {}
        for query in query_list:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            timeout = max(0.3, min(float(self.settings.web_fetch_timeout), remaining))
            try:
                sources = self.client.discover_google_search_sources(
                    query=query.strip(),
                    timeout=timeout,
                    limit=max(8, self.settings.web_max_results * 2),
                    domains=None,
                )
            except Exception:
                sources = []

            for src in sources:
                canonical = _canonical_url(src.url)
                if not canonical:
                    continue
                snippets[canonical] = GoogleSnippetSource(title=src.title, url=src.url, content=src.content)

        return list(snippets.values())
