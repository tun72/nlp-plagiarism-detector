from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
import re
from time import perf_counter
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup


@dataclass
class WebSource:
    title: str
    url: str
    content: str
    source_type: str


def _clean_text(text: str) -> str:
    return " ".join(BeautifulSoup(text or "", "html.parser").get_text(" ").split())


def _looks_like_pdf_url(url: str) -> bool:
    parsed = urlparse((url or "").strip())
    path = (parsed.path or "").lower()
    return path.endswith(".pdf")


def _extract_pdf_text(raw_bytes: bytes, *, max_pages: int = 24, max_chars: int = 150000) -> str:
    if not raw_bytes:
        return ""
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    text_parts: list[str] = []
    total_chars = 0
    try:
        reader = PdfReader(BytesIO(raw_bytes))
    except Exception:
        return ""

    for page_index, page in enumerate(reader.pages):
        if page_index >= max_pages or total_chars >= max_chars:
            break
        try:
            page_text = page.extract_text() or ""
        except Exception:
            continue
        if not page_text:
            continue
        cleaned = " ".join(page_text.split())
        if not cleaned:
            continue
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(cleaned) > remaining:
            cleaned = cleaned[:remaining]
        text_parts.append(cleaned)
        total_chars += len(cleaned)

    return " ".join(text_parts).strip()


def _extract_embedded_json(text: str) -> dict:
    payload = (text or "").strip()
    if not payload:
        return {}
    start = payload.find("{")
    end = payload.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(payload[start : end + 1])
    except json.JSONDecodeError:
        return {}


@lru_cache(maxsize=256)
def _cached_fetch(url: str, timeout: float) -> str:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        }
    )

    last_error = None
    for candidate in [url, url.replace("http://", "https://")]:
        try:
            response = session.get(candidate, timeout=timeout)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
    else:
        raise last_error  # type: ignore[misc]

    content_type = (response.headers.get("Content-Type") or "").lower()
    is_pdf = "application/pdf" in content_type or _looks_like_pdf_url(response.url) or _looks_like_pdf_url(url)
    if is_pdf:
        raw = response.content or b""
        # Avoid pathological memory/time on very large PDFs.
        if len(raw) > (25 * 1024 * 1024):
            return ""
        return _extract_pdf_text(raw)

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return " ".join(soup.get_text(separator=" ").split())


class SearchClient:
    """Multi-source search + parallel fetch client for plagiarism checks."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                )
            }
        )

    def _unwrap_duckduckgo_url(self, href: str) -> str:
        if href.startswith("/l/?"):
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return qs["uddg"][0]
        return href

    def _unwrap_google_url(self, href: str) -> str:
        parsed = urlparse(href)
        host = parsed.netloc.lower()

        is_google_redirect = href.startswith("/url?") or (parsed.path == "/url" and host in {"google.com", "www.google.com"})
        if not is_google_redirect:
            return href

        qs = parse_qs(parsed.query)
        for key in ("url", "q"):
            values = qs.get(key) or []
            if values and values[0]:
                return values[0]
        return href

    def _clean_candidate_url(self, href: str) -> str:
        href = (href or "").strip()
        if not href:
            return ""
        href = self._unwrap_duckduckgo_url(href)
        href = self._unwrap_google_url(href)
        if href.startswith("http"):
            return href
        decoded = unquote(href)
        if decoded.startswith("http"):
            return decoded
        return ""

    def _normalize_site_scope(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""

        candidate = raw if "://" in raw else f"https://{raw}"
        parsed = urlparse(candidate)
        host = (parsed.netloc or "").lower().strip()
        if not host:
            return ""

        path = (parsed.path or "").strip("/")
        if not path:
            return host
        return f"{host}/{path}"

    def _expand_site_scopes(self, scopes: list[str]) -> list[str]:
        expanded: list[str] = []
        for scope in scopes:
            if scope not in expanded:
                expanded.append(scope)

            host = scope.split("/", 1)[0]
            canonical_host = host[4:] if host.startswith("www.") else host
            if canonical_host == "ucsy.edu.mm" or canonical_host.endswith(".ucsy.edu.mm"):
                for candidate in ("ucsy.edu.mm", "www.ucsy.edu.mm", "ucsy.edu.mm/ucsy", "www.ucsy.edu.mm/ucsy"):
                    if candidate not in expanded:
                        expanded.append(candidate)
        return expanded

    def _host_matches_scope(self, host: str, scope_host: str) -> bool:
        clean_host = (host or "").lower().strip(".")
        clean_scope = (scope_host or "").lower().strip(".")
        if not clean_host or not clean_scope:
            return False

        host_core = clean_host[4:] if clean_host.startswith("www.") else clean_host
        scope_core = clean_scope[4:] if clean_scope.startswith("www.") else clean_scope
        return host_core == scope_core or host_core.endswith(f".{scope_core}")

    def _url_matches_scopes(self, url: str, scopes: list[str]) -> bool:
        if not scopes:
            return True

        parsed = urlparse(url)
        host = (parsed.netloc or "").split(":", 1)[0].lower()
        path = (parsed.path or "").strip("/").lower()
        for scope in scopes:
            scope_host, _, scope_path = scope.partition("/")
            if not self._host_matches_scope(host, scope_host):
                continue
            normalized_scope_path = scope_path.strip("/").lower()
            if normalized_scope_path and not (path == normalized_scope_path or path.startswith(f"{normalized_scope_path}/")):
                continue
            return True
        return False

    def _source_label_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        host = (parsed.netloc or "source").lower()
        path = (parsed.path or "").strip("/")
        if not path:
            return host
        first_segment = path.split("/", 1)[0]
        if not first_segment:
            return host
        return f"{host}/{first_segment}"

    def _tokenize_query_terms(self, query: str) -> list[str]:
        tokens = [tok.strip().strip("'\".,;:!?()[]{}") for tok in re.split(r"\s+", query or "") if tok.strip()]
        return [tok for tok in tokens if tok]

    def _build_search_queries(self, base_query: str, domain_filter: str) -> list[str]:
        terms = self._tokenize_query_terms(base_query)
        unique_terms: list[str] = []
        seen_lower: set[str] = set()
        for term in terms:
            key = term.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            unique_terms.append(term)

        # Keep Google query strings compact enough to avoid provider rejection.
        trimmed_terms = unique_terms[:24]
        compact_query = " ".join(trimmed_terms).strip()
        strict_all_words = " ".join(f'"{term}"' for term in trimmed_terms).strip()
        plus_prefixed = " ".join(f"+{term}" for term in trimmed_terms[:14]).strip()

        candidates: list[str] = []
        cleaned_base = " ".join((base_query or "").split())
        if cleaned_base and len(cleaned_base) <= 180:
            candidates.append(f'"{cleaned_base}"{domain_filter}'.strip())

        if compact_query:
            candidates.append(f"{compact_query}{domain_filter}".strip())
            candidates.append(f"{compact_query} filetype:pdf{domain_filter}".strip())

        # Word-by-word Google strategies: chunk tokens to improve recall across many sites.
        if trimmed_terms:
            chunk_size = 5 if len(trimmed_terms) >= 5 else max(2, len(trimmed_terms))
            stride = max(1, chunk_size // 2)
            for start in range(0, len(trimmed_terms), stride):
                chunk = trimmed_terms[start : start + chunk_size]
                if len(chunk) < 2:
                    break
                phrase_chunk = " ".join(chunk)
                exact_chunk = " ".join(f'"{term}"' for term in chunk)
                candidates.append(f'"{phrase_chunk}"{domain_filter}'.strip())
                candidates.append(f"{exact_chunk}{domain_filter}".strip())
                candidates.append(f"{phrase_chunk} filetype:pdf{domain_filter}".strip())
                if start + chunk_size >= len(trimmed_terms):
                    break
                if len(candidates) >= 16:
                    break

        if strict_all_words:
            candidates.append(f"{strict_all_words}{domain_filter}".strip())
        if plus_prefixed:
            candidates.append(f"{plus_prefixed}{domain_filter}".strip())

        deduped: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped[:16]

    def _is_blocked_domain(self, netloc: str, blocked_domains: set[str]) -> bool:
        host = (netloc or "").lower()
        if not host:
            return True
        for blocked in blocked_domains:
            blocked = blocked.lower()
            if blocked.startswith("."):
                if host.endswith(blocked):
                    return True
                continue
            if host == blocked or host.endswith(f".{blocked}"):
                return True
        return False

    def _search_blocked_domains(self) -> set[str]:
        return {
            "duckduckgo.com",
            "html.duckduckgo.com",
            "lite.duckduckgo.com",
            "search.brave.com",
            "google.com",
            "www.google.com",
            "googleusercontent.com",
            "webcache.googleusercontent.com",
            "accounts.google.com",
            "support.google.com",
        }

    def _duckduckgo_html(self, query: str, timeout: float) -> str:
        for method in ("post", "get"):
            try:
                if method == "post":
                    resp = self._session.post("https://html.duckduckgo.com/html/", data={"q": query}, timeout=timeout)
                else:
                    resp = self._session.get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException:
                continue
        return ""

    def _duckduckgo_lite_html(self, query: str, timeout: float) -> str:
        try:
            resp = self._session.get("https://lite.duckduckgo.com/lite/", params={"q": query}, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            return ""

    def _brave_html(self, query: str, timeout: int) -> str:
        try:
            resp = self._session.get("https://search.brave.com/search", params={"q": query, "source": "web"}, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            return ""

    def _google_html(self, query: str, timeout: int, *, start: int = 0, num: int = 20) -> str:
        safe_start = max(0, int(start))
        safe_num = max(10, min(100, int(num)))
        try:
            resp = self._session.get(
                "https://www.google.com/search",
                params={
                    "q": query,
                    "hl": "en",
                    "num": safe_num,
                    "start": safe_start,
                    "filter": "0",
                    "safe": "off",
                    "pws": "0",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            return ""

    def _parse_google_result_items(
        self,
        html: str,
        max_results: int,
        blocked_domains: set[str],
        allowed_scopes: list[str] | None = None,
    ) -> list[tuple[str, str, str]]:
        if not html:
            return []

        scopes = allowed_scopes or []
        soup = BeautifulSoup(html, "html.parser")
        items: list[tuple[str, str, str]] = []
        seen: set[str] = set()

        for block in soup.select("div#search div.g, div#search div.MjjYud, div.g, div.MjjYud"):
            headline = block.find("h3")
            if headline is None:
                continue

            anchor = headline.find_parent("a", href=True) or block.select_one("a[href]")
            if anchor is None:
                continue

            url = self._clean_candidate_url(anchor.get("href") or "")
            if not url or url in seen:
                continue

            netloc = urlparse(url).netloc.split(":", 1)[0].lower()
            if self._is_blocked_domain(netloc, blocked_domains):
                continue
            if scopes and not self._url_matches_scopes(url, scopes):
                continue

            snippet_node = block.select_one(
                "div.VwiC3b, span.aCOpRe, div.IsZvec, div[data-sncf], div[data-content-feature='1']"
            )
            snippet = _clean_text(snippet_node.get_text(" ", strip=True) if snippet_node else "")
            title = _clean_text(headline.get_text(" ", strip=True)) or self._source_label_from_url(url)
            if not snippet:
                # Google frequently changes snippet classes; fallback to block text.
                block_text = _clean_text(block.get_text(" ", strip=True))
                if block_text and title and block_text.lower().startswith(title.lower()):
                    block_text = block_text[len(title) :].strip(" -:|")
                snippet = block_text

            seen.add(url)
            items.append((title, url, snippet))
            if len(items) >= max_results:
                break

        if items:
            return items

        # Fallback: rely on h3 -> parent anchor when class names are unavailable.
        for headline in soup.select("h3"):
            anchor = headline.find_parent("a", href=True)
            if anchor is None:
                continue

            url = self._clean_candidate_url(anchor.get("href") or "")
            if not url or url in seen:
                continue

            netloc = urlparse(url).netloc.split(":", 1)[0].lower()
            if self._is_blocked_domain(netloc, blocked_domains):
                continue
            if scopes and not self._url_matches_scopes(url, scopes):
                continue

            title = _clean_text(headline.get_text(" ", strip=True)) or self._source_label_from_url(url)
            parent_block = anchor.find_parent("div")
            snippet = ""
            if parent_block is not None:
                block_text = _clean_text(parent_block.get_text(" ", strip=True))
                if block_text and title and block_text.lower().startswith(title.lower()):
                    block_text = block_text[len(title) :].strip(" -:|")
                snippet = block_text

            seen.add(url)
            items.append((title, url, snippet))
            if len(items) >= max_results:
                break

        return items

    def _parse_brave_result_items(
        self,
        html: str,
        max_results: int,
        blocked_domains: set[str],
        allowed_scopes: list[str] | None = None,
    ) -> list[tuple[str, str, str]]:
        if not html:
            return []

        scopes = allowed_scopes or []
        soup = BeautifulSoup(html, "html.parser")
        items: list[tuple[str, str, str]] = []
        seen: set[str] = set()

        # Brave commonly renders web results inside `div.snippet` containers.
        for block in soup.select("div.snippet, div.result"):
            anchor = None
            for candidate in block.select("a[href]"):
                url = self._clean_candidate_url(candidate.get("href") or "")
                if url:
                    anchor = candidate
                    break
            if anchor is None:
                continue

            url = self._clean_candidate_url(anchor.get("href") or "")
            if not url or url in seen:
                continue

            netloc = urlparse(url).netloc.split(":", 1)[0].lower()
            if self._is_blocked_domain(netloc, blocked_domains):
                continue
            if scopes and not self._url_matches_scopes(url, scopes):
                continue

            title = _clean_text(anchor.get_text(" ", strip=True)) or self._source_label_from_url(url)
            block_text = _clean_text(block.get_text(" ", strip=True))
            snippet = block_text
            if snippet and title and snippet.lower().startswith(title.lower()):
                snippet = snippet[len(title) :].strip(" -:|")
            if len(snippet) < 20:
                snippet = block_text if len(block_text) >= 20 else self._source_label_from_url(url)

            seen.add(url)
            items.append((title, url, snippet))
            if len(items) >= max_results:
                break

        return items

    def _parse_duckduckgo_result_items(
        self,
        html: str,
        max_results: int,
        blocked_domains: set[str],
        allowed_scopes: list[str] | None = None,
    ) -> list[tuple[str, str, str]]:
        if not html:
            return []

        scopes = allowed_scopes or []
        soup = BeautifulSoup(html, "html.parser")
        items: list[tuple[str, str, str]] = []
        seen: set[str] = set()

        # `div.result` appears in html.duckduckgo.com and table rows appear in lite.duckduckgo.com.
        for block in soup.select("div.result, tr"):
            anchor = block.select_one("a.result__a[href], a.result-link[href], a[href]")
            if anchor is None:
                continue

            url = self._clean_candidate_url(anchor.get("href") or "")
            if not url or url in seen:
                continue

            netloc = urlparse(url).netloc.split(":", 1)[0].lower()
            if self._is_blocked_domain(netloc, blocked_domains):
                continue
            if scopes and not self._url_matches_scopes(url, scopes):
                continue

            title = _clean_text(anchor.get_text(" ", strip=True)) or self._source_label_from_url(url)
            snippet_node = block.select_one(".result__snippet, .snippet")
            snippet = _clean_text(snippet_node.get_text(" ", strip=True) if snippet_node else "")
            if not snippet:
                block_text = _clean_text(block.get_text(" ", strip=True))
                if block_text and title and block_text.lower().startswith(title.lower()):
                    block_text = block_text[len(title) :].strip(" -:|")
                snippet = block_text
            if len(snippet) < 20:
                snippet = _clean_text(title) or self._source_label_from_url(url)

            seen.add(url)
            items.append((title, url, snippet))
            if len(items) >= max_results:
                break

        return items

    def _duckduckgo_result_items(
        self,
        query: str,
        timeout: float,
        max_results: int,
        blocked_domains: set[str],
        allowed_scopes: list[str] | None = None,
    ) -> list[tuple[str, str, str]]:
        html = self._duckduckgo_html(query, timeout)
        items = self._parse_duckduckgo_result_items(
            html,
            max_results=max_results,
            blocked_domains=blocked_domains,
            allowed_scopes=allowed_scopes,
        )
        if items:
            return items
        lite_html = self._duckduckgo_lite_html(query, timeout)
        return self._parse_duckduckgo_result_items(
            lite_html,
            max_results=max_results,
            blocked_domains=blocked_domains,
            allowed_scopes=allowed_scopes,
        )

    def _parse_links(
        self,
        html: str,
        max_results: int,
        blocked_domains: set[str],
        allowed_scopes: list[str] | None = None,
    ) -> list[str]:
        if not html:
            return []
        scopes = allowed_scopes or []
        soup = BeautifulSoup(html, "html.parser")
        urls: list[str] = []
        for a in soup.select("a[href]"):
            url = self._clean_candidate_url(a.get("href") or "")
            if not url:
                continue
            netloc = urlparse(url).netloc.split(":", 1)[0].lower()
            if self._is_blocked_domain(netloc, blocked_domains):
                continue
            if scopes and not self._url_matches_scopes(url, scopes):
                continue
            if url in urls:
                continue
            urls.append(url)
            if len(urls) >= max_results:
                break
        return urls

    def search_urls(
        self,
        query: str,
        max_results: int = 5,
        timeout: float = 10,
        domains: list[str] | None = None,
    ) -> list[str]:
        if not query.strip():
            return []

        from app.core.config import get_settings

        settings = get_settings()
        cleaned: list[str] = []
        if domains:
            for d in domains:
                scope = self._normalize_site_scope(d)
                if scope and scope not in cleaned:
                    cleaned.append(scope)
            cleaned = self._expand_site_scopes(cleaned)

        domain_filter = ""
        if cleaned:
            filter_body = " OR ".join(f"site:{d}" for d in cleaned)
            domain_filter = f" ({filter_body})"

        base_query = query.strip()
        queries = self._build_search_queries(base_query=base_query, domain_filter=domain_filter)
        queries = queries[: max(2, min(4, max_results))]

        urls: list[str] = []
        blocked = self._search_blocked_domains()
        google_pages = max(1, min(8, int(settings.web_google_deep_pages)))
        google_results_per_page = max(10, min(100, int(settings.web_google_results_per_page)))
        skip_google = False

        for q in queries:
            # Deep Google pagination first.
            if not skip_google:
                for page in range(google_pages):
                    start = page * google_results_per_page
                    html = self._google_html(q, timeout, start=start, num=google_results_per_page)
                    if not html:
                        if page == 0:
                            skip_google = True
                        break

                    items = self._parse_google_result_items(
                        html,
                        max_results=max_results,
                        blocked_domains=blocked,
                        allowed_scopes=cleaned,
                    )
                    page_urls = [url for _, url, _ in items] or self._parse_links(
                        html,
                        max_results=max_results,
                        blocked_domains=blocked,
                        allowed_scopes=cleaned,
                    )
                    for url in page_urls:
                        if url not in urls:
                            urls.append(url)
                            if len(urls) >= max_results:
                                return urls

            # Then broaden with Brave if still short.
            for url in self._parse_links(
                self._brave_html(q, timeout),
                max_results=max_results,
                blocked_domains=blocked,
                allowed_scopes=cleaned,
            ):
                if url not in urls:
                    urls.append(url)
                    if len(urls) >= max_results:
                        return urls

            # DuckDuckGo fallback when Google/Brave are blocked or sparse.
            for _, url, _ in self._duckduckgo_result_items(
                q,
                timeout,
                max_results=max_results,
                blocked_domains=blocked,
                allowed_scopes=cleaned,
            ):
                if url not in urls:
                    urls.append(url)
                    if len(urls) >= max_results:
                        return urls

        return urls

    def discover_google_search_sources(
        self,
        query: str,
        timeout: int = 8,
        limit: int = 3,
        domains: list[str] | None = None,
    ) -> list[WebSource]:
        if not query.strip() or limit <= 0:
            return []

        from app.core.config import get_settings

        settings = get_settings()
        cleaned: list[str] = []
        if domains:
            for d in domains:
                scope = self._normalize_site_scope(d)
                if scope and scope not in cleaned:
                    cleaned.append(scope)
            cleaned = self._expand_site_scopes(cleaned)

        domain_filter = ""
        if cleaned:
            filter_body = " OR ".join(f"site:{d}" for d in cleaned)
            domain_filter = f" ({filter_body})"

        queries = self._build_search_queries(base_query=query.strip(), domain_filter=domain_filter)
        blocked = self._search_blocked_domains()
        google_pages = 1
        google_results_per_page = max(10, min(100, int(settings.web_google_results_per_page)))
        natural_queries = [q for q in queries if "+" not in q and q.count('"') <= 6]
        snippet_queries = (natural_queries or queries)[: max(1, min(2, len(natural_queries or queries)))]

        sources: list[WebSource] = []
        seen_urls: set[str] = set()

        # Primary snippets from Brave (fast fallback when Google/Wikipedia endpoints are blocked).
        for q in snippet_queries:
            items = self._parse_brave_result_items(
                self._brave_html(q, timeout),
                max_results=max(google_results_per_page, limit),
                blocked_domains=blocked,
                allowed_scopes=cleaned,
            )
            for title, url, snippet in items:
                snippet_text = _clean_text(snippet) if snippet else ""
                if len(snippet_text) < 20:
                    snippet_text = _clean_text(title) or self._source_label_from_url(url)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                sources.append(
                    WebSource(
                        title=f"Web: brave ({title})",
                        url=url,
                        content=snippet_text,
                        source_type="web",
                    )
                )
                if len(sources) >= limit:
                    return sources

        if sources:
            return sources

        # DuckDuckGo snippets as a secondary fallback.
        for q in snippet_queries:
            items = self._duckduckgo_result_items(
                q,
                timeout,
                max_results=max(google_results_per_page, limit),
                blocked_domains=blocked,
                allowed_scopes=cleaned,
            )
            for title, url, snippet in items:
                snippet_text = _clean_text(snippet) if snippet else ""
                if len(snippet_text) < 20:
                    snippet_text = _clean_text(title) or self._source_label_from_url(url)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                sources.append(
                    WebSource(
                        title=f"Web: duckduckgo ({title})",
                        url=url,
                        content=snippet_text,
                        source_type="web",
                    )
                )
                if len(sources) >= limit:
                    return sources

        if sources:
            return sources

        skip_google = False
        for q in snippet_queries:
            if skip_google:
                break
            for page in range(google_pages):
                start = page * google_results_per_page
                html = self._google_html(q, timeout, start=start, num=google_results_per_page)
                if not html:
                    if page == 0:
                        skip_google = True
                    break
                items = self._parse_google_result_items(
                    html,
                    max_results=max(google_results_per_page, limit),
                    blocked_domains=blocked,
                    allowed_scopes=cleaned,
                )
                if not items:
                    fallback_urls = self._parse_links(
                        html,
                        max_results=max(google_results_per_page, limit),
                        blocked_domains=blocked,
                        allowed_scopes=cleaned,
                    )
                    items = [
                        (
                            self._source_label_from_url(url),
                            url,
                            self._source_label_from_url(url),
                        )
                        for url in fallback_urls
                    ]

                for title, url, snippet in items:
                    snippet_text = _clean_text(snippet) if snippet else ""
                    if len(snippet_text) < 20:
                        snippet_text = _clean_text(title) or self._source_label_from_url(url)
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    sources.append(
                        WebSource(
                            title=f"Web: google ({title})",
                            url=url,
                            content=snippet_text,
                            source_type="web",
                        )
                    )
                    if len(sources) >= limit:
                        return sources

        if sources:
            return sources

        # Fallback when Google markup or connectivity blocks snippets:
        # still return URL-level evidence discovered via broader web providers.
        fallback_urls = self.search_urls(
            query=query,
            max_results=limit,
            timeout=timeout,
            domains=domains,
        )
        for url in fallback_urls:
            label = self._source_label_from_url(url)
            sources.append(
                WebSource(
                    title=f"Web: google-fallback ({label})",
                    url=url,
                    content=label,
                    source_type="web",
                )
            )
            if len(sources) >= limit:
                break

        return sources

    def discover_wikipedia_sources(self, query: str, timeout: int = 8, limit: int = 2) -> list[WebSource]:
        if not query.strip():
            return []

        def wikipedia_json(params: dict) -> dict:
            api_url = "https://en.wikipedia.org/w/api.php"
            try:
                response = self._session.get(api_url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                # Network-restricted fallback: use r.jina.ai to proxy Wikipedia API.
                from requests.models import PreparedRequest

                req = PreparedRequest()
                req.prepare_url(api_url, params)
                source_url = req.url or api_url
                normalized = source_url.split("://", 1)[1] if "://" in source_url else source_url
                proxy_url = f"https://r.jina.ai/http://{normalized}"
                try:
                    proxy_response = self._session.get(proxy_url, timeout=timeout)
                    proxy_response.raise_for_status()
                except requests.RequestException:
                    return {}
                return _extract_embedded_json(proxy_response.text)

        data = wikipedia_json(
            {
                "action": "query",
                "list": "search",
                "format": "json",
                "utf8": 1,
                "srlimit": limit,
                "srsearch": query,
            }
        )
        if not data:
            return []

        results = data.get("query", {}).get("search", [])
        sources: list[WebSource] = []
        for item in results:
            title = item.get("title", "").strip()
            if not title:
                continue
            extract = _clean_text(item.get("snippet") or "")
            if len(extract) < 20:
                ext = wikipedia_json(
                    {
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": 1,
                        "format": "json",
                        "titles": title,
                        "exintro": 0,
                    }
                )
                pages = ext.get("query", {}).get("pages", {}) if ext else {}
                for page in pages.values():
                    extract = _clean_text(page.get("extract") or "")
                    if len(extract) >= 20:
                        break
            if len(extract) < 20:
                continue
            page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            sources.append(WebSource(title=f"Web: wikipedia.org ({title})", url=page_url, content=extract, source_type="web"))
        return sources

    def discover_crossref_sources(self, query: str, timeout: int = 8, limit: int = 3) -> list[WebSource]:
        if not query.strip():
            return []
        try:
            resp = self._session.get(
                "https://api.crossref.org/works",
                params={"query.bibliographic": query, "rows": limit},
                timeout=timeout,
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
        except requests.RequestException:
            return []

        sources: list[WebSource] = []
        for item in items[:limit]:
            title = " ".join(item.get("title", [])[:1]).strip() or "Untitled"
            abstract = _clean_text(item.get("abstract") or "")
            if len(abstract) < 20:
                continue
            url = item.get("URL") or "https://api.crossref.org"
            sources.append(WebSource(title=f"Web: crossref ({title})", url=url, content=abstract, source_type="web"))
        return sources

    def discover_openalex_sources(self, query: str, timeout: int = 8, limit: int = 3) -> list[WebSource]:
        if not query.strip():
            return []
        try:
            resp = self._session.get(
                "https://api.openalex.org/works",
                params={"search": query, "per-page": limit},
                timeout=timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except requests.RequestException:
            return []

        def decode_abstract(inv_idx: dict) -> str:
            words: list[tuple[int, str]] = []
            for w, positions in (inv_idx or {}).items():
                for pos in positions:
                    words.append((int(pos), w))
            words.sort(key=lambda x: x[0])
            return " ".join(w for _, w in words)

        sources: list[WebSource] = []
        for item in results[:limit]:
            title = (item.get("display_name") or "Untitled").strip()
            abstract = _clean_text(decode_abstract(item.get("abstract_inverted_index") or {}))
            if len(abstract) < 20:
                continue
            url = item.get("id") or "https://openalex.org"
            sources.append(WebSource(title=f"Web: openalex ({title})", url=url, content=abstract, source_type="web"))
        return sources

    def discover_arxiv_sources(self, query: str, timeout: int = 8, limit: int = 3) -> list[WebSource]:
        if not query.strip():
            return []
        try:
            resp = self._session.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
                timeout=timeout,
            )
            resp.raise_for_status()
            xml = resp.text
        except requests.RequestException:
            return []

        soup = BeautifulSoup(xml, "xml")
        entries = soup.find_all("entry")[:limit]
        sources: list[WebSource] = []
        for e in entries:
            title = _clean_text((e.title.text if e.title else "Untitled"))
            summary = _clean_text((e.summary.text if e.summary else ""))
            link = (e.id.text if e.id else "https://arxiv.org")
            if len(summary) < 20:
                continue
            sources.append(WebSource(title=f"Web: arxiv ({title})", url=link, content=summary, source_type="web"))
        return sources

    def discover_semantic_scholar_sources(self, query: str, timeout: int = 8, limit: int = 3) -> list[WebSource]:
        if not query.strip():
            return []
        try:
            resp = self._session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": limit, "fields": "title,abstract,url"},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except requests.RequestException:
            return []

        sources: list[WebSource] = []
        for item in data[:limit]:
            title = _clean_text(item.get("title") or "Untitled")
            abstract = _clean_text(item.get("abstract") or "")
            if len(abstract) < 20:
                continue
            url = item.get("url") or "https://www.semanticscholar.org"
            sources.append(WebSource(title=f"Web: semanticscholar ({title})", url=url, content=abstract, source_type="web"))
        return sources

    def discover_gutendex_sources(self, query: str, timeout: int = 10, limit: int = 3) -> list[WebSource]:
        if not query.strip():
            return []
        try:
            resp = self._session.get("https://gutendex.com/books", params={"search": query}, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return []

        books = data.get("results", [])[:limit]
        sources: list[WebSource] = []
        for book in books:
            title = (book.get("title") or "Unknown title").strip()
            formats = book.get("formats") or {}
            candidate_url = (
                formats.get("text/plain; charset=utf-8")
                or formats.get("text/plain")
                or formats.get("text/html; charset=utf-8")
                or formats.get("text/html")
            )
            if not candidate_url:
                continue
            try:
                text = _cached_fetch(candidate_url, timeout)
            except Exception:
                continue
            if not text:
                continue
            sources.append(WebSource(title=f"Book: gutenberg.org ({title})", url=candidate_url, content=text, source_type="book"))
        return sources

    def discover_openlibrary_sources(self, query: str, timeout: int = 8, limit: int = 3) -> list[WebSource]:
        if not query.strip():
            return []
        try:
            resp = self._session.get("https://openlibrary.org/search.json", params={"q": query, "limit": limit}, timeout=timeout)
            resp.raise_for_status()
            docs = resp.json().get("docs", [])
        except requests.RequestException:
            return []

        sources: list[WebSource] = []
        for doc in docs[:limit]:
            title = (doc.get("title") or "Unknown").strip()
            author = ", ".join(doc.get("author_name", [])[:2])
            year = str(doc.get("first_publish_year") or "")
            snippet = _clean_text(f"{title}. {author}. {year}")
            if len(snippet) < 20:
                continue
            key = doc.get("key") or ""
            url = f"https://openlibrary.org{key}" if key else "https://openlibrary.org"
            sources.append(WebSource(title=f"Book: openlibrary.org ({title})", url=url, content=snippet, source_type="book"))
        return sources

    def discover_google_books_sources(self, query: str, timeout: int = 8, limit: int = 3) -> list[WebSource]:
        if not query.strip():
            return []
        try:
            resp = self._session.get(
                "https://www.googleapis.com/books/v1/volumes",
                params={"q": query, "maxResults": limit, "printType": "books"},
                timeout=timeout,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except requests.RequestException:
            return []

        sources: list[WebSource] = []
        for item in items[:limit]:
            volume = item.get("volumeInfo", {})
            title = (volume.get("title") or "Unknown").strip()
            text = _clean_text(f"{volume.get('description') or ''} {(item.get('searchInfo') or {}).get('textSnippet', '')}")
            if len(text) < 20:
                continue
            url = volume.get("infoLink") or "https://books.google.com"
            sources.append(WebSource(title=f"Book: books.google.com ({title})", url=url, content=text, source_type="book"))
        return sources

    def discover_archive_sources(self, query: str, timeout: int = 8, limit: int = 3) -> list[WebSource]:
        if not query.strip():
            return []
        try:
            resp = self._session.get(
                "https://archive.org/advancedsearch.php",
                params={
                    "q": query,
                    "fl[]": ["title", "description", "identifier"],
                    "rows": limit,
                    "output": "json",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
        except requests.RequestException:
            return []

        sources: list[WebSource] = []
        for d in docs[:limit]:
            title = _clean_text(d.get("title") or "Unknown")
            desc = _clean_text(d.get("description") or "")
            if len(desc) < 20:
                continue
            ident = d.get("identifier") or ""
            url = f"https://archive.org/details/{ident}" if ident else "https://archive.org"
            sources.append(WebSource(title=f"Book: archive.org ({title})", url=url, content=desc, source_type="book"))
        return sources

    def discover_sources(
        self,
        query: str,
        max_results: int = 5,
        timeout: float = 10,
        domains: list[str] | None = None,
        source_type: str = "web",
        overall_timeout: float | None = None,
        include_provider_snippets: bool = True,
    ) -> list[WebSource]:
        sources: list[WebSource] = []
        started = perf_counter()

        from app.core.config import get_settings

        settings = get_settings()

        def dedupe(items: list[WebSource]) -> list[WebSource]:
            unique: dict[str, WebSource] = {}
            for src in items:
                key = f"{src.source_type}|{src.url}|{src.title}"
                unique[key] = src
            return list(unique.values())[:max_results]

        def time_left() -> float | None:
            if overall_timeout is None:
                return None
            return overall_timeout - (perf_counter() - started)

        def timed_out() -> bool:
            remaining = time_left()
            return remaining is not None and remaining <= 0

        def bounded_timeout(default: float) -> float:
            remaining = time_left()
            if remaining is None:
                return max(0.05, float(default))
            if remaining <= 0:
                return 0.0
            return max(0.05, min(float(default), remaining))

        # 1) Fast provider snippets first (high chance to return quickly).
        if source_type == "web" and not timed_out():
            fast_quota = max(1, max_results // 2)
            wikipedia_requested = domains is None or any("wikipedia.org" in d.lower() for d in (domains or []))

            if include_provider_snippets:
                if settings.allow_wikipedia_fallback and wikipedia_requested:
                    step_timeout = bounded_timeout(timeout)
                    if step_timeout > 0:
                        wiki_limit = max(1, fast_quota // 2)
                        sources.extend(self.discover_wikipedia_sources(query=query, timeout=step_timeout, limit=wiki_limit))

                if len(sources) < fast_quota and not timed_out():
                    step_timeout = bounded_timeout(timeout)
                    if step_timeout > 0:
                        sources.extend(
                            self.discover_crossref_sources(query=query, timeout=step_timeout, limit=fast_quota - len(sources))
                        )
                if len(sources) < fast_quota and not timed_out():
                    step_timeout = bounded_timeout(timeout)
                    if step_timeout > 0:
                        sources.extend(
                            self.discover_openalex_sources(query=query, timeout=step_timeout, limit=fast_quota - len(sources))
                        )
                if len(sources) < fast_quota and not timed_out():
                    step_timeout = bounded_timeout(timeout)
                    if step_timeout > 0:
                        sources.extend(
                            self.discover_arxiv_sources(query=query, timeout=step_timeout, limit=fast_quota - len(sources))
                        )
                if len(sources) < fast_quota and not timed_out():
                    step_timeout = bounded_timeout(timeout)
                    if step_timeout > 0:
                        sources.extend(
                            self.discover_semantic_scholar_sources(
                                query=query,
                                timeout=step_timeout,
                                limit=fast_quota - len(sources),
                            )
                        )

            # Deep Google snippets add robust content when source pages are blocked/slow.
            if len(sources) < max_results and not timed_out():
                step_timeout = bounded_timeout(timeout)
                if step_timeout > 0:
                    snippet_quota = min(max(1, max_results // 2), max_results - len(sources))
                    sources.extend(
                        self.discover_google_search_sources(
                            query=query,
                            timeout=step_timeout,
                            limit=snippet_quota,
                            domains=domains,
                        )
                    )

        if source_type == "book" and not timed_out():
            fast_quota = max(1, max_results // 2)
            step_timeout = bounded_timeout(timeout)
            if step_timeout > 0:
                sources.extend(self.discover_openlibrary_sources(query=query, timeout=step_timeout, limit=fast_quota))
            if len(sources) < fast_quota and not timed_out():
                step_timeout = bounded_timeout(timeout)
                if step_timeout > 0:
                    sources.extend(
                        self.discover_google_books_sources(query=query, timeout=step_timeout, limit=fast_quota - len(sources))
                    )
            if len(sources) < fast_quota and not timed_out():
                step_timeout = bounded_timeout(timeout)
                if step_timeout > 0:
                    sources.extend(self.discover_archive_sources(query=query, timeout=step_timeout, limit=fast_quota - len(sources)))
            if len(sources) < fast_quota and not timed_out():
                step_timeout = bounded_timeout(timeout + 2)
                if step_timeout > 0:
                    sources.extend(self.discover_gutendex_sources(query=query, timeout=step_timeout, limit=fast_quota - len(sources)))

        # 2) Then fill remaining slots by crawling discovered URLs.
        remaining = max_results - len(sources)
        if remaining > 0 and not timed_out():
            # Reserve enough budget for page fetch after search discovery in web-only mode.
            search_default = timeout
            if source_type == "web" and not include_provider_snippets:
                search_default = max(0.4, timeout * 0.45)
            search_timeout = bounded_timeout(search_default)
            if search_timeout > 0:
                urls = self.search_urls(query=query, max_results=remaining, timeout=search_timeout, domains=domains)
            else:
                urls = []

            if urls:
                executor = ThreadPoolExecutor(max_workers=min(4, len(urls)))
                future_map: dict = {}
                for url in urls:
                    fetch_timeout = bounded_timeout(timeout)
                    if fetch_timeout <= 0:
                        break
                    future_map[executor.submit(_cached_fetch, url, fetch_timeout)] = url
                try:
                    wait_timeout = bounded_timeout(timeout)
                    if wait_timeout > 0 and future_map:
                        for fut in as_completed(future_map, timeout=wait_timeout):
                            if timed_out():
                                break
                            url = future_map[fut]
                            try:
                                result_timeout = bounded_timeout(timeout)
                                if result_timeout <= 0:
                                    break
                                text = fut.result(timeout=result_timeout)
                            except Exception:
                                continue
                            if not text:
                                continue
                            source_label = self._source_label_from_url(url)
                            prefix = "Book" if source_type == "book" else "Web"
                            sources.append(
                                WebSource(title=f"{prefix}: {source_label}", url=url, content=text, source_type=source_type)
                            )
                            if len(sources) >= max_results:
                                break
                except FuturesTimeoutError:
                    pass
                finally:
                    for fut in future_map:
                        if not fut.done():
                            fut.cancel()
                    # Avoid waiting for slow network futures after local deadline.
                    executor.shutdown(wait=False, cancel_futures=True)

        return dedupe(sources)

    def connectivity_report(self, timeout: int = 4) -> dict[str, bool]:
        checks = {
            "duckduckgo_html": "https://html.duckduckgo.com/html/",
            "ucsy": "https://www.ucsy.edu.mm/",
            "brave_search": "https://search.brave.com/search?q=test",
            "google_search": "https://www.google.com/search?q=test",
            "wikipedia_api": "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch=test",
            "crossref_api": "https://api.crossref.org/works?rows=1",
            "openalex_api": "https://api.openalex.org/works?search=test&per-page=1",
            "arxiv_api": "http://export.arxiv.org/api/query?search_query=all:test&max_results=1",
            "semantic_scholar_api": "https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1",
            "gutendex_api": "https://gutendex.com/books?search=test",
            "openlibrary_api": "https://openlibrary.org/search.json?q=test&limit=1",
            "google_books_api": "https://www.googleapis.com/books/v1/volumes?q=test&maxResults=1",
            "archive_api": "https://archive.org/advancedsearch.php?q=test&rows=1&output=json",
        }
        out: dict[str, bool] = {}
        for name, url in checks.items():
            try:
                r = self._session.get(url, timeout=timeout)
                out[name] = 200 <= r.status_code < 400
            except requests.RequestException:
                out[name] = False
        return out
