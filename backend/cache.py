"""In-memory cache for embeddings and results"""
from typing import Dict, Any, Optional
from functools import lru_cache
import json


class Cache:
    """Simple in-memory cache for embeddings and LLM results.
    
    Tradeoff: In-memory is fast but lost on restart. Redis would persist
    but adds operational overhead. For MVP, in-memory is pragmatic.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if len(self._cache) >= self.max_size:
            # Simple eviction: remove first key (FIFO)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value

    def clear(self) -> None:
        """Clear all cache."""
        self._cache.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cache

    def dump(self) -> Dict[str, Any]:
        """Dump cache contents (for debugging)."""
        return self._cache.copy()
