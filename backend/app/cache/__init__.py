"""Cache module initialization."""

from app.cache.deduplication_cache import (
    TextDeduplicationCache,
    SimilarityResultCache,
    get_text_cache,
    get_similarity_cache,
    clear_all_caches,
    cache_stats,
)

__all__ = [
    "TextDeduplicationCache",
    "SimilarityResultCache",
    "get_text_cache",
    "get_similarity_cache",
    "clear_all_caches",
    "cache_stats",
]
