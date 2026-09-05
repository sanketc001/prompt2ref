"""prompt -> named entities, via a swappable LLM provider (P2RLocalProvider
or P2RAnthropicProvider - see nodes_provider.py) instead of a hardcoded Claude
API call.
"""

from __future__ import annotations

import json

from . import _repo  # noqa: F401  (adds the Prompt2Ref repo to sys.path)
from . import local_llm


class P2RExtractEntities:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": ("P2R_PROVIDER",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "use_llm": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("entities_json", "entities_readable")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, provider, prompt, use_llm=True):
        from prompt2ref.entities import Entity, EntityType, dedupe_entities
        from prompt2ref.extract import SYSTEM, heuristic_entities

        prompt = (prompt or "").strip()
        entities: list[Entity] = []
        used_llm = False

        if prompt and use_llm:
            try:
                type_list = ", ".join(t.value for t in EntityType)
                system_prompt = (
                    SYSTEM + "\n\n"
                    "Output ONLY a single JSON object, no prose, no code fence, of the exact "
                    'shape {"entities": [{"name": str, "type": one of [%s], '
                    '"descriptor": str, "search_name": str, "aliases": [str], '
                    '"mention": str, "confidence": number 0-1}]}.' % type_list
                )
                text = provider.generate(system_prompt, "PROMPT:\n" + prompt)
                data = local_llm.extract_json(text)
                if isinstance(data, dict) and isinstance(data.get("entities"), list):
                    entities = [
                        Entity.from_dict(d) for d in data["entities"]
                        if isinstance(d, dict) and d.get("name")
                    ]
                    used_llm = True
            except Exception as exc:
                print("[prompt2ref] LLM extraction failed (%r); using heuristic fallback" % exc)

        if not entities:
            entities = heuristic_entities(prompt)
        entities = dedupe_entities(entities)

        payload = [e.to_dict() for e in entities]
        if payload:
            readable = ("[llm] " if used_llm else "[heuristic] ") + ", ".join(
                "%s (%s)" % (e["name"], e["type"]) for e in payload)
        else:
            readable = "(no entities found)"
        return (json.dumps(payload, indent=2), readable)


NODE_CLASS_MAPPINGS = {"P2RExtractEntities": P2RExtractEntities}
NODE_DISPLAY_NAME_MAPPINGS = {"P2RExtractEntities": "P2R Extract Entities"}
