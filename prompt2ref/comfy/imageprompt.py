"""Rewrite a free-form prompt into an image-model prompt bound to the references.

The video path (promptgen.py) targets MiniMax H3's sectioned form. Image models
want one flowing paragraph instead, so the shape is different, but the job is
the same: every packed reference gets named in the text so the model knows which
subject each conditioning image is, and the user's intent survives the rewrite.

Two modes:

    generate  compose a *new* image from the references
    edit      change an existing image, the references telling the model what
              the replacement subjects must look like

Qwen-Image-Edit prefixes the encoded text with "Picture 1: <|vision_start|>..."
per conditioning image, so referring to "Picture 1" in prose lands on exactly
the image we passed. Flux.2 has no such tag, and gets the same text with the
references described positionally instead - hence `tagged`.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger("prompt2ref.comfy.imageprompt")

MODES = ("generate", "edit")

DEFAULT_NEGATIVE = (
    "watermark, text overlay, logo, signature, jpeg artifacts, lowres, blurry, "
    "deformed hands, extra limbs, collage, split frame, contact sheet, border"
)

SYSTEM_GENERATE = """You write prompts for a reference-conditioned image generation model.

You are given a user's short generation prompt and a list of reference pictures \
that have already been collected and wired into the workflow as conditioning \
images. Rewrite the user's prompt into one vivid image prompt.

Rules:
- Name every reference by its tag exactly as supplied (e.g. "Picture 1") the \
first time you mention that subject, then keep describing the scene normally. \
Never invent a tag that is not in the supplied list, and mention every tag once.
- The references define *identity and appearance*, not composition: say what the \
subject must keep from the reference (face, livery, packaging, silhouette), not \
that the output should look like the reference photo.
- Describe the shot the user asked for: subject placement, camera, lens, lighting, \
time of day, mood, and the key surface/material details. One paragraph, no lists, \
no section headers, no markdown.
- Keep the user's creative intent exactly. Do not add characters or products they \
did not ask for, and do not change the setting they named.
- negative_prompt: a short comma-separated list of artefacts to avoid for this \
particular image. Do not put the subject in it.
- The reference pictures may be contact sheets of several photos of one subject. \
Never ask for a grid, collage or multi-panel output."""

SYSTEM_EDIT = """You write edit instructions for a reference-conditioned image editing model.

You are given a user's requested change, the fact that the image being edited is \
supplied as the first conditioning image, and a list of reference pictures wired \
in after it. Rewrite the request into one precise edit instruction.

Rules:
- Write it as an instruction about the existing image ("Replace ...", "Change ...", \
"Add ..."), not as a description of a new picture.
- State explicitly what must stay untouched - composition, framing, background, \
lighting, pose - so the model does not redraw the whole frame.
- Name every reference by its tag exactly as supplied (e.g. "Picture 2") when \
saying what the edited element must now look like. Never invent a tag that is not \
in the supplied list, and mention every tag once.
- One paragraph, no lists, no section headers, no markdown.
- negative_prompt: a short comma-separated list of artefacts to avoid. Do not put \
the subject in it.
- The reference pictures may be contact sheets of several photos of one subject. \
Never ask for a grid, collage or multi-panel output."""

SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
    },
    "required": ["prompt", "negative_prompt"],
    "additionalProperties": False,
}


def picture_tag(slot: int, tagged: bool = True) -> str:
    """What the prompt calls slot `slot`."""
    return "Picture %d" % slot if tagged else "reference %d" % slot


def picture_brief(pictures: list, tagged: bool = True, edit_offset: int = 0) -> str:
    """The tag -> caption list handed to the model."""
    lines = []
    if edit_offset:
        lines.append("%s: the image being edited (already supplied)"
                     % picture_tag(1, tagged))
    for p in pictures:
        lines.append("%s: %s  [%dx%d]"
                     % (picture_tag(p.slot + edit_offset, tagged), p.caption,
                        p.width, p.height))
    return "\n".join(lines) or "(no reference pictures)"


def build_image_prompt(
    user_prompt: str,
    pictures: list,          # list[PackedPicture]
    settings,
    mode: str = "generate",
    tagged: bool = True,
    edit_offset: int = 0,    # 1 when the edited image occupies slot 1
    style: str = "",
    negative: str | None = None,
    use_llm: bool = True,
) -> tuple[str, str]:
    """(prompt, negative_prompt) for `user_prompt` bound to the packed pictures."""
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (MODES,))
    if use_llm and settings.api_key:
        try:
            parts = _rewrite(user_prompt, pictures, settings, mode, tagged,
                             edit_offset, style)
            return (_with_style(parts["prompt"], style),
                    negative or parts.get("negative_prompt") or DEFAULT_NEGATIVE)
        except Exception as exc:
            log.warning("image prompt rewrite failed (%s); using deterministic layout", exc)
    text = _fallback(user_prompt, pictures, mode, tagged, edit_offset)
    return _with_style(text, style), (negative or DEFAULT_NEGATIVE)


def _with_style(text: str, style: str) -> str:
    style = (style or "").strip().strip(",")
    return "%s %s" % (text.strip(), style) if style else text.strip()


def _rewrite(user_prompt, pictures, settings, mode, tagged, edit_offset, style) -> dict:
    import anthropic

    from ..llm import structured_message

    client = anthropic.Anthropic(api_key=settings.api_key)
    user = "USER REQUEST:\n%s\n\nREFERENCE PICTURES:\n%s" % (
        user_prompt.strip(), picture_brief(pictures, tagged, edit_offset))
    if style.strip():
        user += "\n\nSTYLE THE USER ASKED FOR: %s" % style.strip()
    response = structured_message(
        client, model=settings.model, max_tokens=4000,
        system=SYSTEM_EDIT if mode == "edit" else SYSTEM_GENERATE,
        schema=SCHEMA, effort="low",
        messages=[{"role": "user", "content": user}],
    )
    if getattr(response, "stop_reason", None) == "refusal":
        raise RuntimeError("model declined to write the prompt")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def _fallback(user_prompt, pictures, mode, tagged, edit_offset) -> str:
    """Deterministic layout - no API key, or the rewrite failed."""
    binds = ["%s shows %s" % (picture_tag(p.slot + edit_offset, tagged), p.caption)
             for p in pictures]
    bind_text = ("; ".join(binds) + ". Keep those subjects exactly as shown. ") if binds else ""
    if mode == "edit":
        head = "%s is the image being edited. " % picture_tag(1, tagged) if edit_offset else ""
        return ("%s%sApply this change and leave the rest of the frame - composition, "
                "background, framing and lighting - untouched: %s"
                % (head, bind_text, user_prompt.strip()))
    return "%s%s" % (bind_text, user_prompt.strip())
