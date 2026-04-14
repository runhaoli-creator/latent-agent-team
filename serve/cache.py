"""Episode-level response cache for the latent-agent-team API.

Caches episode results keyed on ``(instruction, env_spec, comm_mode, bitrate_k, seed)``
so repeated identical requests — common during benchmark reruns and unit
debugging — return instantly without replaying the 15-step agent loop.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Protocol


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    @property
    def backend_name(self) -> str: ...


def make_key(instruction: str, env_spec: dict[str, Any], comm_mode: str,
             bitrate_k: int | None, seed: int | None) -> str:
    payload = json.dumps(
        {"i": instruction, "e": env_spec, "m": comm_mode, "k": bitrate_k, "s": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class LRUMemoryCache:
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

    @property
    def backend_name(self) -> str:
        return "memory"


class RedisCache:
    def __init__(self, url: str, ttl_seconds: int = 1800):
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise RuntimeError("redis backend requires redis-py") from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def set(self, key: str, value: str) -> None:
        self._client.set(key, value, ex=self._ttl)

    @property
    def backend_name(self) -> str:
        return "redis"


def build_cache_from_env() -> CacheBackend:
    backend = os.getenv("LAT_CACHE_BACKEND", "memory").lower()
    if backend == "redis":
        return RedisCache(
            os.getenv("LAT_REDIS_URL", "redis://localhost:6379/2"),
            ttl_seconds=int(os.getenv("LAT_REDIS_TTL", "1800")),
        )
    return LRUMemoryCache(capacity=int(os.getenv("LAT_LRU_CAPACITY", "1024")))
