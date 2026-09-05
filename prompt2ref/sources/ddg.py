"""DuckDuckGo image search - the default backend, no API key required."""

from __future__ import annotations

import random
import time

from . import Candidate, SearchBackend, log


class DDGBackend(SearchBackend):
    name = "ddg"

    def available(self, settings) -> bool:
        try:
            import ddgs  # noqa: F401
        except ImportError:
            log.warning("ddgs not installed - run: pip install ddgs")
            return False
        return True

    def search(self, query: str, limit: int, settings) -> list[Candidate]:
        from ddgs import DDGS

        wants_logo = "logo" in query.lower()
        last_exc: Exception | None = None
        hits = None
        for attempt in range(3):
            try:
                with DDGS(timeout=int(settings.timeout)) as client:
                    hits = client.images(
                        query,
                        max_results=limit,
                        safesearch="moderate",
                        type_image=None if wants_logo else "photo",
                    )
                break
            except Exception as exc:  # rate limiting is common; back off and retry
                last_exc = exc
                time.sleep(1.5 * (attempt + 1) + random.random())
        if hits is None:
            raise last_exc if last_exc else RuntimeError("ddg search failed")

        out: list[Candidate] = []
        for h in hits:
            url = h.get("image") or ""
            if not url:
                continue
            out.append(
                Candidate(
                    url=url,
                    source="ddg",
                    width=int(h.get("width") or 0),
                    height=int(h.get("height") or 0),
                    title=h.get("title") or "",
                    page_url=h.get("url") or "",
                    credit=h.get("source") or "",
                )
            )
        return out
