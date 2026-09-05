"""Image search backends.

Every backend exposes `search(query, limit, settings) -> list[Candidate]`.
Backends are tried in `settings.sources` order and their results are merged, so
a flaky or rate-limited backend degrades the result instead of breaking it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("prompt2ref.sources")


@dataclass
class Candidate:
    url: str
    source: str
    width: int = 0
    height: int = 0
    title: str = ""
    page_url: str = ""
    credit: str = ""
    license: str = ""
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def pixels(self) -> int:
        return max(0, self.width) * max(0, self.height)


class SearchBackend:
    name = "base"

    def search(self, query: str, limit: int, settings) -> list[Candidate]:
        raise NotImplementedError

    def available(self, settings) -> bool:
        return True


def get_backend(name: str) -> SearchBackend | None:
    from . import ddg, keyed, openverse, wikimedia

    registry = {
        "ddg": ddg.DDGBackend,
        "duckduckgo": ddg.DDGBackend,
        "wikimedia": wikimedia.WikimediaBackend,
        "openverse": openverse.OpenverseBackend,
        "serpapi": keyed.SerpApiBackend,
        "google_cse": keyed.GoogleCSEBackend,
    }
    cls = registry.get(name.strip().lower())
    return cls() if cls else None


def search_all(query: str, limit: int, settings) -> list[Candidate]:
    """Merge results from every configured backend, dropping duplicate URLs."""
    seen: set[str] = set()
    merged: list[Candidate] = []
    for name in settings.sources:
        backend = get_backend(name)
        if backend is None or not backend.available(settings):
            continue
        try:
            hits = backend.search(query, limit, settings)
        except Exception as exc:  # a dead backend must not kill the run
            log.warning("source %s failed for %r: %s", name, query, exc)
            continue
        for c in hits:
            key = c.url.split("?")[0]
            if not c.url or key in seen:
                continue
            seen.add(key)
            merged.append(c)
        if len(merged) >= limit:
            break
    return merged
