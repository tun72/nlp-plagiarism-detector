from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from urllib.parse import urlparse

from app.core.config import Settings, get_settings
from app.scraper.search_client import SearchClient

ACADEMIC_DOMAINS = [
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
    "wiley.com",
    "tandfonline.com",
    "jstor.org",
]


@dataclass
class AcademicSource:
    title: str
    url: str
    content: str
    source_type: str
    provider: str


def _canonical_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{host}{path}"


def _provider_from_url(url: str) -> str:
    host = urlparse((url or "").strip()).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "academic"


class Layer2AcademicSearch:
    """Academic-source retrieval layer (arXiv/OpenAlex/Crossref/Semantic Scholar and books)."""

    def __init__(self, settings: Settings | None = None, client: SearchClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or SearchClient()

    def discover(self, queries: list[str], *, deadline: float) -> list[AcademicSource]:
        if not queries:
            return []
        if not self.settings.allow_web_search and not self.settings.allow_book_search:
            return []

        unique: dict[str, AcademicSource] = {}

        for query in queries:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break

            if self.settings.allow_web_search:
                timeout = max(0.4, min(float(self.settings.web_fetch_timeout), remaining))
                try:
                    for src in self.client.discover_sources(
                        query=query,
                        max_results=max(2, self.settings.web_max_results),
                        timeout=timeout,
                        domains=ACADEMIC_DOMAINS,
                        source_type="web",
                        overall_timeout=timeout,
                    ):
                        if not (src.content or "").strip():
                            continue
                        key = _canonical_url(src.url) or f"{src.title}|{hash(src.content)}"
                        unique[key] = AcademicSource(
                            title=src.title,
                            url=src.url,
                            content=src.content,
                            source_type=src.source_type,
                            provider=_provider_from_url(src.url),
                        )
                except Exception:
                    pass

            if not self.settings.allow_book_search:
                continue

            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            timeout = max(0.4, min(float(self.settings.book_fetch_timeout), remaining))
            try:
                for src in self.client.discover_sources(
                    query=query,
                    max_results=max(2, self.settings.book_max_results),
                    timeout=timeout,
                    domains=self.settings.parsed_book_domains,
                    source_type="book",
                    overall_timeout=timeout,
                ):
                    if not (src.content or "").strip():
                        continue
                    key = _canonical_url(src.url) or f"{src.title}|{hash(src.content)}"
                    unique[key] = AcademicSource(
                        title=src.title,
                        url=src.url,
                        content=src.content,
                        source_type=src.source_type,
                        provider=_provider_from_url(src.url),
                    )
            except Exception:
                pass

        return list(unique.values())
