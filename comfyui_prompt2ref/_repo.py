"""Puts the Prompt2Ref checkout on sys.path so these nodes can import
`prompt2ref` (entities, pipeline, imagepack, ...) instead of duplicating it.

Mirrors the DEFAULT_COMFY_ROOT convention in prompt2ref/comfy/builder.py:
a hardcoded default for this machine, overridable via PROMPT2REF_REPO.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_PROMPT2REF_REPO = Path(r"C:\Users\micke\Desktop\AI\Prompt2Ref")


def ensure_on_path() -> None:
    override = os.getenv("PROMPT2REF_REPO")
    candidates = [
        Path(override) if override else None,
        Path(__file__).resolve().parent.parent,  # this file lives in <repo>/comfyui_prompt2ref/
        DEFAULT_PROMPT2REF_REPO,
    ]
    for c in candidates:
        if c and (c / "prompt2ref" / "__init__.py").is_file():
            p = str(c)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    raise ImportError(
        "Could not locate the Prompt2Ref repo (checked PROMPT2REF_REPO, this "
        "file's own parent, and %s). Set PROMPT2REF_REPO to the checkout path."
        % DEFAULT_PROMPT2REF_REPO
    )


ensure_on_path()
