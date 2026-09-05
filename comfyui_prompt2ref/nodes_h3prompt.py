"""user prompt + packed picture captions -> MiniMax H3's structured prompt,
written by a swappable LLM provider (P2RLocalProvider or
P2RAnthropicProvider). Feeds straight into MiniMaxH3ReferenceToVideo's
`prompt` STRING input.
"""

from __future__ import annotations

import json

from . import _repo  # noqa: F401
from . import local_llm


def _picture_brief(captions: list[dict]) -> str:
    lines = ["<Picture %d>: %s  [%dx%d]" % (
        c["slot"], c.get("caption", ""), c.get("width", 0), c.get("height", 0))
        for c in captions]
    return "\n".join(lines) or "(no reference pictures)"


def _fallback(user_prompt: str, captions: list[dict], seconds: float) -> dict:
    defs = ["<Picture %d> is %s." % (c["slot"], c.get("caption", "a reference")) for c in captions]
    retention = ["<Picture %d> ([Shot 1] / reference): fully_preserved - keep the exact "
                "appearance shown in the reference." % c["slot"] for c in captions]
    tags = " ".join("<Picture %d>" % c["slot"] for c in captions)
    return {
        "subject_definitions": defs or ["No image references are supplied."],
        "summary": "[reference generation] The target video is a %g-second clip: %s"
                   % (seconds, user_prompt.strip()),
        "retention_analysis": retention or ["No references to preserve."],
        "detailed_description": "[Shot 1] %s %s" % (
            user_prompt.strip(), ("Featuring " + tags + ".") if tags else ""),
        "overall_soundscape": "Natural ambient sound appropriate to the scene.",
        "non_diegetic_music": "No non-diegetic music is present.",
    }


class P2RBuildH3Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": ("P2R_PROVIDER",),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "captions_json": ("STRING", {"multiline": True, "default": "[]"}),
                "seconds": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 15.0, "step": 0.1}),
                "use_llm": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("h3_prompt",)
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, provider, user_prompt, captions_json, seconds=1.0, use_llm=True):
        from prompt2ref.comfy.promptgen import SECTIONS, assemble
        from prompt2ref.comfy.promptgen import SYSTEM as H3_SYSTEM

        captions = json.loads(captions_json or "[]")
        parts = None

        if use_llm:
            try:
                system_prompt = (
                    H3_SYSTEM + "\n\n"
                    "Output ONLY a single JSON object, no prose, no code fence, with exactly "
                    "these string/string-array keys: %s." % ", ".join(SECTIONS)
                )
                user = (
                    "USER PROMPT:\n%s\n\nDURATION: %.2f seconds\n\nREFERENCE PICTURES:\n%s"
                    % (user_prompt.strip(), seconds, _picture_brief(captions))
                )
                text = provider.generate(system_prompt, user)
                data = local_llm.extract_json(text)
                if isinstance(data, dict) and all(k in data for k in SECTIONS):
                    parts = data
            except Exception as exc:
                print("[prompt2ref] H3 rewrite failed (%r); using deterministic layout" % exc)

        if parts is None:
            parts = _fallback(user_prompt, captions, seconds)

        return (assemble(parts),)


NODE_CLASS_MAPPINGS = {"P2RBuildH3Prompt": P2RBuildH3Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {"P2RBuildH3Prompt": "P2R Build H3 Prompt"}
