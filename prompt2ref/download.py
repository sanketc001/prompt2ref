"""Fetch, validate, dedupe and save candidate images."""

from __future__ import annotations

import hashlib
import io
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from .shots import Shot
from .sources import Candidate

log = logging.getLogger("prompt2ref.download")

Image.MAX_IMAGE_PIXELS = 200_000_000  # generous, but still guards decompression bombs

BAD_URL_MARKERS = ("logo-sprite", "placeholder", "no-image", "avatar-default")

# Stock-photo comps ship with a watermark burned into the pixels. They rank well
# on resolution and would quietly poison an image-to-image reference, so they are
# rejected outright rather than merely down-ranked.
WATERMARK_HOSTS = (
    "alamy.com",
    "shutterstock.com",
    "dreamstime.com",
    "123rf.com",
    "depositphotos.com",
    "istockphoto.com",
    "gettyimages.",
    "agefotostock.com",
    "ftcdn.net",          # Adobe Stock CDN
    "stock.adobe.com",
    "vecteezy.com",
    "pond5.com",
    "bigstockphoto.com",
    "imago-images",
    "zumapress.com",
    "stockphoto",
    "canstockphoto",
    "watermark",
)

# Stock comp filenames: /1000_F_502610758_... , /preview/... , /comp/...
WATERMARK_PATH_RE = re.compile(r"/\d{3,4}_f_\d|/comp/|/preview[s]?/|_watermark")


def is_blocked(url: str) -> bool:
    u = url.lower()
    if any(m in u for m in WATERMARK_HOSTS) or any(m in u for m in BAD_URL_MARKERS):
        return True
    return bool(WATERMARK_PATH_RE.search(u))


@dataclass
class SavedImage:
    path: Path
    shot: str
    width: int
    height: int
    bytes: int
    sha256: str
    dhash: str
    source: str
    source_url: str
    page_url: str = ""
    credit: str = ""
    license: str = ""
    query: str = ""

    def to_dict(self, root: Path) -> dict:
        d = {
            "shot": self.shot,
            "file": self.path.name,
            "path": str(self.path.relative_to(root)).replace("\\", "/"),
            "abs_path": str(self.path),
            "width": self.width,
            "height": self.height,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "dhash": self.dhash,
            "source": self.source,
            "source_url": self.source_url,
            "page_url": self.page_url,
            "credit": self.credit,
            "license": self.license,
            "query": self.query,
        }
        return d


def dhash(img: Image.Image, size: int = 8) -> str:
    """Difference hash - cheap perceptual fingerprint for near-duplicate rejection."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = []
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits.append(px[base + col] < px[base + col + 1])
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return format(value, "016x")


def hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


@dataclass
class FetchContext:
    """Per-entity state so shots don't collide or re-save the same picture."""

    seen_sha: set[str] = field(default_factory=set)
    seen_dhash: list[str] = field(default_factory=list)

    def is_duplicate(self, sha: str, dh: str, max_distance: int) -> bool:
        if sha in self.seen_sha:
            return True
        return any(hamming(dh, prev) <= max_distance for prev in self.seen_dhash)

    def remember(self, sha: str, dh: str) -> None:
        self.seen_sha.add(sha)
        self.seen_dhash.append(dh)


def _session(settings) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": settings.user_agent,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def fetch_bytes(url: str, settings, session: requests.Session | None = None) -> bytes | None:
    if is_blocked(url):
        return None
    sess = session or _session(settings)
    try:
        r = sess.get(url, timeout=settings.timeout, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("Content-Type", "").lower()
        if ctype and not ctype.startswith("image/"):
            return None
        if ctype.startswith("image/svg"):
            return None
        buf = io.BytesIO()
        for chunk in r.iter_content(64 * 1024):
            buf.write(chunk)
            if buf.tell() > settings.max_download_bytes:
                return None
        return buf.getvalue()
    except Exception as exc:
        log.debug("fetch failed %s: %s", url, exc)
        return None


def _open_image(raw: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    return ImageOps.exif_transpose(img)


def score_candidate(c: Candidate, shot: Shot) -> float:
    """Rank before downloading: resolution first, then aspect fit, then source."""
    score = 0.0
    if c.pixels:
        score += min(c.pixels / 1_000_000.0, 12.0)
    if shot.prefer_aspect and c.width and c.height:
        ratio = c.width / c.height
        score -= min(abs(ratio - shot.prefer_aspect) * 3.0, 6.0)
    score += {"serpapi": 1.5, "google_cse": 1.5, "wikimedia": 1.0, "ddg": 0.5}.get(c.source, 0.0)
    if any(w in (c.url + " " + c.title).lower() for w in ("thumb", "preview", "sample")):
        score -= 1.5
    return score


def accept_image(img: Image.Image, shot: Shot, settings) -> str | None:
    """Return a rejection reason, or None when the image passes."""
    w, h = img.size
    if w < max(shot.min_width, 1) or h < max(shot.min_height, 1):
        return "too small (%dx%d)" % (w, h)
    ratio = w / h if h else 0
    lo, hi = shot.aspect
    if not (lo <= ratio <= hi):
        return "aspect %.2f outside %.2f-%.2f" % (ratio, lo, hi)
    return None


def save_image(
    raw: bytes,
    img: Image.Image,
    dest_dir: Path,
    shot: Shot,
    index: int,
) -> tuple[Path, int, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    keep_alpha = shot.allow_alpha and img.mode in ("RGBA", "LA", "P")
    if keep_alpha:
        out = img.convert("RGBA")
        path = dest_dir / ("%s_%02d.png" % (shot.key, index))
        out.save(path, "PNG", optimize=True)
    else:
        out = img.convert("RGB")
        path = dest_dir / ("%s_%02d.jpg" % (shot.key, index))
        out.save(path, "JPEG", quality=92, subsampling=1, optimize=True)
    return path, out.width, out.height


def download_shot(
    candidates: list[Candidate],
    shot: Shot,
    dest_dir: Path,
    settings,
    ctx: FetchContext,
    want: int,
    start_index: int = 1,
    query: str = "",
    on_event=None,
) -> list[SavedImage]:
    """Walk ranked candidates until `want` images are saved for this shot."""
    saved: list[SavedImage] = []
    session = _session(settings)
    ranked = sorted(candidates, key=lambda c: score_candidate(c, shot), reverse=True)
    index = start_index

    for cand in ranked:
        if len(saved) >= want:
            break
        raw = fetch_bytes(cand.url, settings, session)
        if not raw:
            continue
        sha = hashlib.sha256(raw).hexdigest()
        img = _open_image(raw)
        if img is None:
            continue
        reason = accept_image(img, shot, settings)
        if reason:
            log.debug("reject %s: %s", cand.url, reason)
            continue
        dh = dhash(img)
        if ctx.is_duplicate(sha, dh, settings.dedupe_distance):
            continue

        path, w, h = save_image(raw, img, dest_dir, shot, index)
        ctx.remember(sha, dh)
        item = SavedImage(
            path=path,
            shot=shot.key,
            width=w,
            height=h,
            bytes=path.stat().st_size,
            sha256=sha,
            dhash=dh,
            source=cand.source,
            source_url=cand.url,
            page_url=cand.page_url,
            credit=cand.credit,
            license=cand.license,
            query=query,
        )
        saved.append(item)
        index += 1
        if on_event:
            on_event(
                {
                    "type": "image",
                    "shot": shot.key,
                    "file": path.name,
                    "width": w,
                    "height": h,
                    "source": cand.source,
                }
            )
    return saved
