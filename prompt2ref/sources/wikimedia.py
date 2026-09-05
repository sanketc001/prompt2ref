"""Wikipedia / Wikimedia Commons.

High precision for people, landmarks and artworks, and every hit carries
license metadata - useful when the references have to be attributable.
"""

from __future__ import annotations

import re

import requests

from . import Candidate, SearchBackend

COMMONS_API = "https://commons.wikimedia.org/w/api.php"


class WikimediaBackend(SearchBackend):
    name = "wikimedia"

    def search(self, query: str, limit: int, settings) -> list[Candidate]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": "filetype:bitmap " + query,
            "gsrnamespace": 6,
            "gsrlimit": min(limit, 50),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime",
        }
        r = requests.get(
            COMMONS_API,
            params=params,
            timeout=settings.timeout,
            headers={"User-Agent": settings.user_agent},
        )
        r.raise_for_status()
        pages = (r.json().get("query") or {}).get("pages") or {}

        out: list[Candidate] = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("url")
            mime = info.get("mime", "")
            if not url or not mime.startswith("image/") or mime == "image/svg+xml":
                continue
            ext = info.get("extmetadata") or {}
            out.append(
                Candidate(
                    url=url,
                    source="wikimedia",
                    width=int(info.get("width") or 0),
                    height=int(info.get("height") or 0),
                    title=page.get("title", ""),
                    page_url=info.get("descriptionurl", ""),
                    credit=_plain(ext.get("Artist", {}).get("value", "")),
                    license=_plain(ext.get("LicenseShortName", {}).get("value", "")),
                )
            )
        return out


def _plain(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()[:200]
