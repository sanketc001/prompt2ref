"""Runtime configuration. Everything is overridable via env or CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, but present in requirements.txt
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sources that never need a key come first; keyed ones are appended only when
# their credentials exist (see load_settings).
FREE_SOURCES = ("ddg", "wikimedia", "openverse")


@dataclass
class Settings:
    refs_dir: Path = PROJECT_ROOT / "refs"
    runs_dir: Path = PROJECT_ROOT / "runs"

    # Entity extraction
    model: str = "claude-opus-5"
    effort: str = "low"
    api_key: str | None = None

    # Image acceptance
    min_width: int = 640
    min_height: int = 640
    min_pixels: int = 640 * 640
    max_download_bytes: int = 25 * 1024 * 1024
    dedupe_distance: int = 6  # max dhash hamming distance to call two images equal

    # Fetching
    per_shot: int | None = None  # None -> use each shot's own default count
    candidates_multiplier: int = 10  # search hits to consider per wanted image
    min_candidates: int = 20         # floor, since blocked/broken URLs thin the pool
    workers: int = 3
    timeout: float = 20.0
    user_agent: str = (
        "Prompt2Ref/0.1 (reference-image collector; +https://localhost) "
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )

    # Visual QC: show every new frame to Claude and drop watermarked, wrong-subject
    # or wrong-shot results. Costs one extra vision call per entity.
    verify: bool = False

    # ComfyUI integration (see prompt2ref.comfy)
    comfy_root: Path | None = None

    sources: tuple[str, ...] = FREE_SOURCES
    serpapi_key: str | None = None
    google_api_key: str | None = None
    google_cse_id: str | None = None

    extra: dict = field(default_factory=dict)

    def entity_dir(self, slug: str) -> Path:
        return self.refs_dir / slug


def load_settings(**overrides) -> Settings:
    """Read .env + environment, then apply explicit overrides."""
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    s = Settings()
    s.api_key = os.getenv("ANTHROPIC_API_KEY") or None
    s.model = os.getenv("PROMPT2REF_MODEL", s.model)
    s.verify = os.getenv("PROMPT2REF_VERIFY", "").lower() in {"1", "true", "yes"}
    if os.getenv("PROMPT2REF_COMFY_ROOT"):
        s.comfy_root = Path(os.environ["PROMPT2REF_COMFY_ROOT"])
    s.serpapi_key = os.getenv("SERPAPI_API_KEY") or None
    s.google_api_key = os.getenv("GOOGLE_API_KEY") or None
    s.google_cse_id = os.getenv("GOOGLE_CSE_ID") or None

    sources = list(FREE_SOURCES)
    if s.google_api_key and s.google_cse_id:
        sources.insert(0, "google_cse")
    if s.serpapi_key:
        sources.insert(0, "serpapi")
    env_sources = os.getenv("PROMPT2REF_SOURCES")
    if env_sources:
        sources = [x.strip() for x in env_sources.split(",") if x.strip()]
    s.sources = tuple(sources)

    if os.getenv("PROMPT2REF_REFS_DIR"):
        s.refs_dir = Path(os.environ["PROMPT2REF_REFS_DIR"]).resolve()
        s.runs_dir = s.refs_dir.parent / "runs"

    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(s, k):
            s.extra[k] = v
            continue
        cur = getattr(s, k)
        setattr(s, k, Path(v) if isinstance(cur, Path) else v)

    s.min_pixels = min(s.min_pixels, s.min_width * s.min_height)
    s.refs_dir.mkdir(parents=True, exist_ok=True)
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    return s
