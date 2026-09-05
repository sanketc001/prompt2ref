"""Optional visual QC.

URL blocklists only catch watermark hosts we already know about, and nothing
catches "that is the wrong Chris Evans". This pass shows the downloaded frames
to Claude and drops the ones that would poison an image-to-image reference.
"""

from __future__ import annotations

import base64
import io
import json
import logging

from PIL import Image

log = logging.getLogger("prompt2ref.verify")

MAX_EDGE = 512          # QC only needs to see the subject, not the detail
MAX_PER_CALL = 8

SYSTEM = """You are checking candidate reference images for an image-to-image \
generation pipeline. For each numbered image, decide whether it is usable as a \
clean visual reference for the stated entity and shot.

Reject an image if any of these are true:
- it shows a watermark, tiled logo text, stock-agency overlay, or a site URL burned in
- it is not the stated entity (a different person, a lookalike, a different model of product)
- it is a collage, contact sheet, meme, screenshot with UI, or has heavy text/graphics
- it is a drawing/AI render when a real photo of a real entity was asked for
- the subject is tiny, cropped away, or unrecognisable
- the shot type is clearly wrong (a face closeup where a full-body frame was requested, \
or vice versa)

Accept otherwise. Be strict about watermarks and identity, lenient about lighting, \
crop and background."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "ok": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "ok", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def _thumb_b64(path) -> str | None:
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        return None
    img = img.convert("RGB")
    img.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode()


def verify_images(entity, images: list, settings) -> dict[str, tuple[bool, str]]:
    """Return {filename: (ok, reason)}. On any failure everything is accepted."""
    if not images or not settings.api_key:
        return {}
    verdicts: dict[str, tuple[bool, str]] = {}
    for chunk_start in range(0, len(images), MAX_PER_CALL):
        chunk = images[chunk_start: chunk_start + MAX_PER_CALL]
        try:
            verdicts.update(_verify_chunk(entity, chunk, settings))
        except Exception as exc:
            log.warning("verification failed for %s: %s", entity.slug, exc)
    return verdicts


def _verify_chunk(entity, chunk: list, settings) -> dict[str, tuple[bool, str]]:
    import anthropic

    content: list[dict] = []
    usable: list = []
    for item in chunk:
        b64 = _thumb_b64(item.path)
        if b64 is None:
            continue
        usable.append(item)
        content.append({
            "type": "text",
            "text": "Image %d - requested shot: %s" % (len(usable), item.shot),
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    if not usable:
        return {}

    label = entity.name + ((" (" + entity.descriptor + ")") if entity.descriptor else "")
    content.append({
        "type": "text",
        "text": "Entity: %s\nType: %s\nJudge all %d images." % (
            label, entity.type.value, len(usable)),
    })

    client = anthropic.Anthropic(api_key=settings.api_key)
    response = client.messages.create(
        model=settings.model,
        max_tokens=4000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        messages=[{"role": "user", "content": content}],
    )
    if getattr(response, "stop_reason", None) == "refusal":
        return {}
    data = json.loads(next(b.text for b in response.content if b.type == "text"))

    out: dict[str, tuple[bool, str]] = {}
    for v in data.get("verdicts", []):
        i = int(v.get("index", 0)) - 1
        if 0 <= i < len(usable):
            out[usable[i].path.name] = (bool(v.get("ok")), v.get("reason", ""))
    return out
