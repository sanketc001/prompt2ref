"""Rewrite a free-form prompt into the MiniMax H3 structured prompt.

The H3 reference model wants a sectioned prompt where every reference input is
bound by tag - `<Picture N>` for images, `<Video N>`, `<Audio N>` - and then
referred to by that tag in the shot description. Claude does the rewrite; the
section scaffolding is assembled here so the shape is always exactly right.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger("prompt2ref.comfy.promptgen")

SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

SYSTEM = """You write prompts for the MiniMax H3 reference-to-video model.

You are given a user's short generation prompt and a list of reference pictures \
that have already been collected and wired into the workflow. Each picture has a \
tag like <Picture 1>. Rewrite the user's prompt into MiniMax H3's structured form.

Rules:
- Refer to every subject by its tag (<Picture 1>, <Picture 2>, ...) after defining \
it. Never invent a tag that is not in the supplied list, and use every tag at \
least once in detailed_description.
- subject_definitions: one line per picture, of the form \
"<Picture 1> is <short visual description of what the picture shows>." Describe \
what the reference depicts; do not restate the shot list verbatim.
- summary: one sentence, starting with "[reference generation] The target video is \
a N-second ..." using the duration you are given.
- retention_analysis: one line per picture, of the form \
"<Picture 1> ([Shot 1] / role): fully_preserved - what must stay identical." \
Use fully_preserved for identity/appearance, reference for style or environment.
- detailed_description: the actual shot, beginning with "[Shot 1]". Describe camera, \
motion, lighting and action. Keep it to what fits the given duration - a 1-second \
clip is a single continuous beat, not a sequence. Spoken lines, if any, go in \
<d>[English] ...</d> tags.
- overall_soundscape: the diegetic audio.
- non_diegetic_music: music, or "No non-diegetic music is present."
- Keep the user's creative intent. Do not add characters or products they did not ask for."""

SCHEMA = {
    "type": "object",
    "properties": {
        "subject_definitions": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "retention_analysis": {"type": "array", "items": {"type": "string"}},
        "detailed_description": {"type": "string"},
        "overall_soundscape": {"type": "string"},
        "non_diegetic_music": {"type": "string"},
    },
    "required": list(SECTIONS),
    "additionalProperties": False,
}


def assemble(parts: dict) -> str:
    """Render the section dict into the exact text layout the template expects."""
    def block(v) -> str:
        return "\n".join(v) if isinstance(v, list) else str(v).strip()

    out = []
    for name in SECTIONS:
        out.append(name)
        out.append(block(parts.get(name, "")))
        out.append("")
    return "\n".join(out).strip() + "\n"


def build_prompt(
    user_prompt: str,
    pictures: list,          # list[PackedPicture]
    seconds: float,
    settings,
    use_llm: bool = True,
) -> str:
    """Structured H3 prompt for `user_prompt` bound to the packed picture slots."""
    if use_llm and settings.api_key:
        try:
            return assemble(_rewrite(user_prompt, pictures, seconds, settings))
        except Exception as exc:
            log.warning("prompt rewrite failed (%s); using deterministic layout", exc)
    return assemble(_fallback(user_prompt, pictures, seconds))


def _picture_brief(pictures: list) -> str:
    lines = []
    for p in pictures:
        lines.append("<Picture %d>: %s  [%dx%d]" % (p.slot, p.caption, p.width, p.height))
    return "\n".join(lines) or "(no reference pictures)"


def _rewrite(user_prompt: str, pictures: list, seconds: float, settings) -> dict:
    import anthropic

    from ..llm import structured_message

    client = anthropic.Anthropic(api_key=settings.api_key)
    user = (
        "USER PROMPT:\n%s\n\nDURATION: %.2f seconds\n\nREFERENCE PICTURES:\n%s"
        % (user_prompt.strip(), seconds, _picture_brief(pictures))
    )
    response = structured_message(
        client, model=settings.model, max_tokens=8000, system=SYSTEM,
        schema=SCHEMA, effort="medium",
        messages=[{"role": "user", "content": user}],
    )
    if getattr(response, "stop_reason", None) == "refusal":
        raise RuntimeError("model declined to write the prompt")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def _fallback(user_prompt: str, pictures: list, seconds: float) -> dict:
    defs, retention = [], []
    for p in pictures:
        defs.append("<Picture %d> is %s." % (p.slot, p.caption))
        retention.append(
            "<Picture %d> ([Shot 1] / reference): fully_preserved - keep the exact "
            "appearance shown in the reference." % p.slot
        )
    tags = " ".join("<Picture %d>" % p.slot for p in pictures)
    return {
        "subject_definitions": defs or ["No image references are supplied."],
        "summary": "[reference generation] The target video is a %g-second clip: %s"
                   % (seconds, user_prompt.strip()),
        "retention_analysis": retention or ["No references to preserve."],
        "detailed_description": "[Shot 1] %s %s"
                                % (user_prompt.strip(),
                                   ("Featuring " + tags + ".") if tags else ""),
        "overall_soundscape": "Natural ambient sound appropriate to the scene.",
        "non_diegetic_music": "No non-diegetic music is present.",
    }
