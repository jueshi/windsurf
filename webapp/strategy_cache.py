from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class StrategyCache:
    def __init__(self, default_ttl: int = 600):
        self._store: dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl

    def _is_expired(self, entry: CacheEntry) -> bool:
        return time.time() > entry.expires_at

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        if self._is_expired(entry):
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self.default_ttl
        self._store[key] = CacheEntry(value=value, expires_at=time.time() + ttl)

    def clear(self) -> None:
        self._store.clear()


strategy_cache = StrategyCache(default_ttl=600)
