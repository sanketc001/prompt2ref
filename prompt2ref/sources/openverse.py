"""Openverse - keyless search over openly licensed images."""

from __future__ import annotations

import requests

from . import Candidate, SearchBackend

API = "https://api.openverse.org/v1/images/"


class OpenverseBackend(SearchBackend):
    name = "openverse"

    def search(self, query: str, limit: int, settings) -> list[Candidate]:
        r = requests.get(
            API,
            params={"q": query, "page_size": min(limit, 20), "mature": "false"},
            timeout=settings.timeout,
            headers={"User-Agent": settings.user_agent},
        )
        r.raise_for_status()
        out: list[Candidate] = []
        for h in r.json().get("results", []):
            url = h.get("url")
            if not url:
                continue
            lic = (h.get("license") or "").upper()
            ver = h.get("license_version") or ""
            out.append(
                Candidate(
                    url=url,
                    source="openverse",
                    width=int(h.get("width") or 0),
                    height=int(h.get("height") or 0),
                    title=h.get("title") or "",
                    page_url=h.get("foreign_landing_url") or "",
                    credit=h.get("creator") or "",
                    license=(lic + " " + ver).strip(),
                )
            )
        return out
