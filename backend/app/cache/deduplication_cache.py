"""
Request Deduplication & Caching Layer

In-memory LRU cache for:
- Identical text analysis results
- Source matches
- LLM analysis results
- Similarity computations

Reduces redundant processing and improves API response times.
"""

from functools import lru_cache
from hashlib import sha256
from typing import Optional, Dict, Any
from dataclasses import dataclass
from time import time


@dataclass
class CachedResult:
    """Cached plagiarism detection result."""
    text_hash: str
    result: Dict[str, Any]
    timestamp: float
    ttl_seconds: int = 3600  # 1 hour default TTL

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time() - self.timestamp > self.ttl_seconds


class TextDeduplicationCache:
    """
    LRU cache for input text deduplication.

    Prevents reprocessing identical texts within a time window.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for entries (default 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CachedResult] = {}
        self.access_order = []

    def compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of text for caching key."""
        return sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for text.

        Args:
            text: Input text

        Returns:
            Cached result or None if not found/expired
        """
        text_hash = self.compute_hash(text)

        if text_hash not in self.cache:
            return None

        cached = self.cache[text_hash]

        if cached.is_expired():
            # Remove expired entry
            del self.cache[text_hash]
            self.access_order.remove(text_hash)
            return None

        # Update access order for LRU
        if text_hash in self.access_order:
            self.access_order.remove(text_hash)
        self.access_order.append(text_hash)

        return cached.result

    def put(self, text: str, result: Dict[str, Any]) -> None:
        """
        Cache result for text.

        Args:
            text: Input text
            result: Detection result
        """
        text_hash = self.compute_hash(text)

        # Remove LRU item if cache is full
        if len(self.cache) >= self.max_size and text_hash not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[text_hash] = CachedResult(
            text_hash=text_hash,
            result=result,
            timestamp=time(),
            ttl_seconds=self.ttl_seconds,
        )

        if text_hash in self.access_order:
            self.access_order.remove(text_hash)
        self.access_order.append(text_hash)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        expired = sum(1 for v in self.cache.values() if v.is_expired())

        return {
            "total_entries": len(self.cache),
            "expired_entries": expired,
            "active_entries": len(self.cache) - expired,
            "max_size": self.max_size,
            "usage_percent": (len(self.cache) / self.max_size * 100) if self.max_size > 0 else 0,
        }


class SimilarityResultCache:
    """
    LRU cache for similarity computation results.

    Caches sentence pair similarity scores to avoid recomputation.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize similarity cache."""
        self.max_size = max_size
        self.cache: Dict[str, float] = {}
        self.access_order = []

    def compute_pair_key(self, text1: str, text2: str) -> str:
        """Compute cache key for text pair."""
        pair = f"{text1}||{text2}"
        return sha256(pair.encode("utf-8")).hexdigest()

    def get(self, text1: str, text2: str) -> Optional[float]:
        """Get cached similarity score."""
        key = self.compute_pair_key(text1, text2)

        if key not in self.cache:
            return None

        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        return self.cache[key]

    def put(self, text1: str, text2: str, similarity: float) -> None:
        """Cache similarity score."""
        key = self.compute_pair_key(text1, text2)

        # Remove LRU item if full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = similarity

        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


# Global cache instances
_text_cache = TextDeduplicationCache(max_size=100, ttl_seconds=3600)
_similarity_cache = SimilarityResultCache(max_size=1000)


def get_text_cache() -> TextDeduplicationCache:
    """Get global text deduplication cache."""
    return _text_cache


def get_similarity_cache() -> SimilarityResultCache:
    """Get global similarity cache."""
    return _similarity_cache


def clear_all_caches() -> None:
    """Clear all caches (for testing/maintenance)."""
    _text_cache.clear()
    _similarity_cache.clear()


def cache_stats() -> Dict[str, Any]:
    """Get combined cache statistics."""
    return {
        "text_cache": _text_cache.stats(),
        "similarity_cache": {
            "total_entries": len(_similarity_cache.cache),
            "max_size": _similarity_cache.max_size,
            "usage_percent": (
                len(_similarity_cache.cache) / _similarity_cache.max_size * 100
                if _similarity_cache.max_size > 0 else 0
            ),
        },
    }
