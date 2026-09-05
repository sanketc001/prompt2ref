"""Named-entity extraction from a free-form generation prompt.

Primary path is Claude with a strict JSON schema; the heuristic fallback keeps
the tool usable offline or without an API key (lower recall, no typing).
"""

from __future__ import annotations

import json
import logging
import re

from .entities import Entity, EntityType, dedupe_entities

log = logging.getLogger("prompt2ref.extract")

SYSTEM = """You extract named entities from prompts written for image and video generation models.

Return every entity a reference image could be collected for: real people, \
fictional characters, landmarks and buildings, cities and places, brands and \
organizations, specific products and models, vehicles, named animals or breeds, \
artworks, named dishes, and named events.

Rules:
- Only named/specific things. Skip generic nouns ("a man", "a city", "a phone", \
"sunset", "cyberpunk", "35mm lens", camera and style words).
- Use the entity's canonical full name, not the prompt's shorthand: "Tom" in a \
prompt about Top Gun is "Tom Cruise"; "the Taj" is "Taj Mahal"; "iPhone 16" \
stays "iPhone 16".
- descriptor: a short disambiguator that makes image search unambiguous \
("American actor", "mausoleum in Agra India", "Apple smartphone 2024"). Keep it \
under 40 characters. Empty if the name is already unambiguous.
- search_name: what you would literally type into an image search to get this \
exact thing.
- mention: the exact substring from the prompt that referred to it.
- confidence: 0-1, how sure you are this is a real named entity worth fetching.
- If a person and a character they play are both referenced, emit whichever the \
prompt actually depicts; emit both only if both appear visually.
- Return an empty list if the prompt names nothing specific."""

SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [t.value for t in EntityType],
                    },
                    "descriptor": {"type": "string"},
                    "search_name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "mention": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "name",
                    "type",
                    "descriptor",
                    "search_name",
                    "aliases",
                    "mention",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}


def extract_entities(prompt: str, settings, use_llm: bool = True) -> list[Entity]:
    """Prompt text -> typed entities. Falls back to heuristics on any failure."""
    prompt = (prompt or "").strip()
    if not prompt:
        return []
    if use_llm and settings.api_key:
        try:
            return dedupe_entities(_extract_with_claude(prompt, settings))
        except Exception as exc:
            log.warning("Claude extraction failed (%s); using heuristic fallback", exc)
    return dedupe_entities(heuristic_entities(prompt))


def _extract_with_claude(prompt: str, settings) -> list[Entity]:
    import anthropic

    from .llm import structured_message

    client = anthropic.Anthropic(api_key=settings.api_key)
    response = structured_message(
        client, model=settings.model, max_tokens=8000, system=SYSTEM,
        schema=SCHEMA, effort=settings.effort,
        messages=[{"role": "user", "content": "PROMPT:\n" + prompt}],
    )
    if getattr(response, "stop_reason", None) == "refusal":
        raise RuntimeError("model declined to extract entities")
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return [Entity.from_dict(d) for d in data.get("entities", [])]


# --- heuristic fallback -----------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "with", "for", "to",
    "from", "by", "as", "into", "over", "under", "near", "this", "that", "his",
    "her", "its", "their", "cinematic", "photo", "photograph", "shot", "scene",
    "style", "portrait", "closeup", "close", "up", "wide", "angle", "lens",
    "camera", "lighting", "golden", "hour", "hd", "4k", "8k", "realistic",
    "hyperrealistic", "render", "detailed", "beautiful", "background", "video",
    "image", "prompt", "man", "woman", "person", "people", "city", "street",
}

# Capitalized runs, optionally carrying digits ("iPhone 16", "Boeing 747").
NAME_RE = re.compile(
    r"\b((?:[A-Z][\w'’&.-]*|i[A-Z]\w*)(?:\s+(?:of|de|del|la|van|von|the)\s+"
    r"[A-Z][\w'’&.-]*|\s+[A-Z][\w'’&.-]*|\s+\d{1,4}(?:[A-Za-z]{1,3})?)*)\b"
)


def heuristic_entities(prompt: str) -> list[Entity]:
    """Capitalized-run NER. No typing, so everything lands as OTHER."""
    found: list[Entity] = []
    for m in NAME_RE.finditer(prompt):
        raw = m.group(1).strip(" .,-")
        words = [w for w in raw.split() if w]
        if not words:
            continue
        if all(w.lower() in STOPWORDS for w in words):
            continue
        # Drop a leading sentence-start word that is only capitalized by position.
        if m.start() == 0 or prompt[max(0, m.start() - 2): m.start()].strip() in {".", "!", "?"}:
            if len(words) == 1:
                continue
        if len(raw) < 3:
            continue
        found.append(
            Entity(name=raw, type=EntityType.OTHER, mention=raw, confidence=0.4)
        )
    return found
