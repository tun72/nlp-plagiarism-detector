from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from urllib.parse import urlparse

from app.core.config import Settings, get_settings
from app.scraper.search_client import SearchClient

BLOCKED_HOSTS = {
    "accounts.google.com",
    "support.google.com",
    "webcache.googleusercontent.com",
    "m.facebook.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
}


@dataclass
class WebsiteSource:
    title: str
    url: str
    content: str
    source_type: str
    reputation: float


def _canonical_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{host}{path}"


def _host(url: str) -> str:
    host = urlparse((url or "").strip()).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_reputation(host: str) -> float:
    if not host:
        return 0.0
    if host in BLOCKED_HOSTS:
        return 0.0
    if host == "ucsy.edu.mm" or host.endswith(".ucsy.edu.mm"):
        return 0.99
    if host.endswith(".edu") or host.endswith(".ac.uk"):
        return 0.95
    if host.endswith(".gov"):
        return 0.92
    if any(tag in host for tag in ("wikipedia.org", "arxiv.org", "springer.com", "nature.com", "sciencedirect.com")):
        return 0.90
    if host.endswith(".org"):
        return 0.82
    if host.endswith(".com"):
        return 0.75
    return 0.65


class Layer3WebsiteSearch:
    """General website retrieval layer with lightweight domain-quality filtering."""

    def __init__(self, settings: Settings | None = None, client: SearchClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or SearchClient()
        self._ucsy_domain_patterns = (
            "ucsy.edu.mm",
            "www.ucsy.edu.mm",
            "facebook.com/lwinmay.thant.796",
        )

    def _ucsy_domains(self) -> list[str]:
        from_settings = [d for d in self.settings.parsed_web_domains if any(p in d.lower() for p in self._ucsy_domain_patterns)]
        defaults = ["ucsy.edu.mm", "www.ucsy.edu.mm", "ucsy.edu.mm/ucsy", "www.ucsy.edu.mm/ucsy"]
        merged: list[str] = []
        for d in [*from_settings, *defaults]:
            value = (d or "").strip()
            if value and value not in merged:
                merged.append(value)
        return merged

    def discover(self, queries: list[str], *, deadline: float, prioritize_ucsy: bool = False) -> list[WebsiteSource]:
        if not self.settings.allow_web_search or not queries:
            return []

        unique: dict[str, WebsiteSource] = {}
        per_query_limit = max(10, int(self.settings.web_max_results) * 3)

        for query in queries:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break

            timeout = max(0.4, min(float(self.settings.web_fetch_timeout), remaining))
            try:
                sources = self.client.discover_sources(
                    query=query,
                    max_results=per_query_limit,
                    timeout=timeout,
                    domains=None,
                    source_type="web",
                    overall_timeout=max(timeout * 2.2, timeout + 3.0),
                    include_provider_snippets=False,
                )
            except Exception:
                continue

            for src in sources:
                text = (src.content or "").strip()
                if not text:
                    continue
                host = _host(src.url)
                score = _host_reputation(host)
                if score <= 0.0:
                    continue
                key = _canonical_url(src.url) or f"{src.title}|{hash(text)}"
                unique[key] = WebsiteSource(
                    title=src.title,
                    url=src.url,
                    content=text,
                    source_type=src.source_type,
                    reputation=score,
                )

            # Always include direct Wikipedia API retrieval in website layer.
            remaining = deadline - perf_counter()
            if remaining > 0 and self.settings.allow_wikipedia_fallback:
                wiki_timeout = max(0.4, min(float(self.settings.web_fetch_timeout), remaining))
                try:
                    wiki_sources = self.client.discover_wikipedia_sources(
                        query=query,
                        timeout=wiki_timeout,
                        limit=max(2, int(self.settings.web_max_results)),
                    )
                except Exception:
                    wiki_sources = []

                for src in wiki_sources:
                    text = (src.content or "").strip()
                    if not text:
                        continue
                    key = _canonical_url(src.url) or f"{src.title}|{hash(text)}"
                    unique[key] = WebsiteSource(
                        title=src.title,
                        url=src.url,
                        content=text,
                        source_type=src.source_type,
                        reputation=max(0.90, _host_reputation(_host(src.url))),
                    )

                # Fallback: domain-targeted crawl when Wikipedia API returns no content.
                if not wiki_sources:
                    remaining = deadline - perf_counter()
                    if remaining > 0:
                        wiki_timeout = max(0.4, min(float(self.settings.web_fetch_timeout), remaining))
                        try:
                            wiki_fallback = self.client.discover_sources(
                                query=query,
                                max_results=max(2, int(self.settings.web_max_results)),
                                timeout=wiki_timeout,
                                domains=["en.wikipedia.org", "wikipedia.org"],
                                source_type="web",
                                overall_timeout=max(wiki_timeout * 1.8, wiki_timeout + 1.5),
                                include_provider_snippets=False,
                            )
                        except Exception:
                            wiki_fallback = []

                        for src in wiki_fallback:
                            text = (src.content or "").strip()
                            if not text:
                                continue
                            if "wikipedia.org" not in _host(src.url):
                                continue
                            key = _canonical_url(src.url) or f"{src.title}|{hash(text)}"
                            unique[key] = WebsiteSource(
                                title=src.title,
                                url=src.url,
                                content=text,
                                source_type=src.source_type,
                                reputation=max(0.90, _host_reputation(_host(src.url))),
                            )

        # UCSY-targeted retrieval path when keyword trigger is active.
        if prioritize_ucsy:
            ucsy_domains = self._ucsy_domains()
            ucsy_query_limit = min(3, len(queries))
            for query in queries[:ucsy_query_limit]:
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    break

                timeout = max(0.4, min(float(self.settings.web_fetch_timeout), remaining))
                try:
                    ucsy_sources = self.client.discover_sources(
                        query=query,
                        max_results=max(6, int(self.settings.web_max_results) * 2),
                        timeout=timeout,
                        domains=ucsy_domains,
                        source_type="web",
                        overall_timeout=max(timeout * 2.0, timeout + 2.0),
                        include_provider_snippets=False,
                    )
                except Exception:
                    continue

                for src in ucsy_sources:
                    text = (src.content or "").strip()
                    if not text:
                        continue
                    host = _host(src.url)
                    if not (host == "ucsy.edu.mm" or host.endswith(".ucsy.edu.mm")):
                        continue
                    key = _canonical_url(src.url) or f"{src.title}|{hash(text)}"
                    unique[key] = WebsiteSource(
                        title=src.title,
                        url=src.url,
                        content=text,
                        source_type=src.source_type,
                        reputation=0.99,
                    )

        return list(unique.values())
