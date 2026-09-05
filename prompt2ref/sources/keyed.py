"""Optional keyed backends, enabled automatically when their env keys exist.

    SERPAPI_API_KEY                 -> Google Images via SerpAPI
    GOOGLE_API_KEY + GOOGLE_CSE_ID  -> Google Programmable Search (image mode)

Both jump to the front of the source order in config.load_settings, because
they return better-targeted results than the keyless backends.
"""

from __future__ import annotations

import requests

from . import Candidate, SearchBackend


class SerpApiBackend(SearchBackend):
    name = "serpapi"

    def available(self, settings) -> bool:
        return bool(settings.serpapi_key)

    def search(self, query: str, limit: int, settings) -> list[Candidate]:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_images",
                "q": query,
                "num": min(limit, 100),
                "api_key": settings.serpapi_key,
            },
            timeout=settings.timeout,
        )
        r.raise_for_status()
        out: list[Candidate] = []
        for h in r.json().get("images_results", [])[:limit]:
            url = h.get("original")
            if not url:
                continue
            out.append(
                Candidate(
                    url=url,
                    source="serpapi",
                    width=int(h.get("original_width") or 0),
                    height=int(h.get("original_height") or 0),
                    title=h.get("title") or "",
                    page_url=h.get("link") or "",
                    credit=h.get("source") or "",
                )
            )
        return out


class GoogleCSEBackend(SearchBackend):
    name = "google_cse"

    def available(self, settings) -> bool:
        return bool(settings.google_api_key and settings.google_cse_id)

    def search(self, query: str, limit: int, settings) -> list[Candidate]:
        out: list[Candidate] = []
        for start in range(1, min(limit, 30) + 1, 10):  # CSE returns 10 per page
            r = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": settings.google_api_key,
                    "cx": settings.google_cse_id,
                    "q": query,
                    "searchType": "image",
                    "num": 10,
                    "start": start,
                    "imgSize": "large",
                },
                timeout=settings.timeout,
            )
            if r.status_code != 200:
                break
            items = r.json().get("items", [])
            for h in items:
                img = h.get("image", {})
                out.append(
                    Candidate(
                        url=h.get("link", ""),
                        source="google_cse",
                        width=int(img.get("width") or 0),
                        height=int(img.get("height") or 0),
                        title=h.get("title") or "",
                        page_url=img.get("contextLink") or "",
                        credit=h.get("displayLink") or "",
                    )
                )
            if len(items) < 10:
                break
        return out[:limit]
