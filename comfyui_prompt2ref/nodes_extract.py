"""prompt -> named entities, using a local CLIP-loaded LLM instead of Claude."""

from __future__ import annotations

import json

from . import _repo  # noqa: F401  (adds the Prompt2Ref repo to sys.path)
from . import local_llm


class P2RExtractEntities:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "use_llm": ("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 768, "min": 128, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("entities_json", "entities_readable")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, clip, prompt, use_llm=True, max_length=768, temperature=0.4, seed=0):
        from prompt2ref.entities import Entity, EntityType, dedupe_entities
        from prompt2ref.extract import SYSTEM, heuristic_entities

        prompt = (prompt or "").strip()
        entities: list[Entity] = []
        used_llm = False

        if prompt and use_llm:
            try:
                type_list = ", ".join(t.value for t in EntityType)
                instruction = (
                    SYSTEM + "\n\n"
                    "Output ONLY a single JSON object, no prose, no code fence, of the exact "
                    'shape {"entities": [{"name": str, "type": one of [%s], '
                    '"descriptor": str, "search_name": str, "aliases": [str], '
                    '"mention": str, "confidence": number 0-1}]}.'
                    "\n\nPROMPT:\n%s" % (type_list, prompt)
                )
                text = local_llm.generate_text(
                    clip, instruction, max_length=max_length, temperature=temperature, seed=seed)
                data = local_llm.extract_json(text)
                if isinstance(data, dict) and isinstance(data.get("entities"), list):
                    entities = [
                        Entity.from_dict(d) for d in data["entities"]
                        if isinstance(d, dict) and d.get("name")
                    ]
                    used_llm = True
            except Exception as exc:
                print("[prompt2ref] local extraction failed (%r); using heuristic fallback" % exc)

        if not entities:
            entities = heuristic_entities(prompt)
        entities = dedupe_entities(entities)

        payload = [e.to_dict() for e in entities]
        if payload:
            readable = ("[local llm] " if used_llm else "[heuristic] ") + ", ".join(
                "%s (%s)" % (e["name"], e["type"]) for e in payload)
        else:
            readable = "(no entities found)"
        return (json.dumps(payload, indent=2), readable)


NODE_CLASS_MAPPINGS = {"P2RExtractEntities": P2RExtractEntities}
NODE_DISPLAY_NAME_MAPPINGS = {"P2RExtractEntities": "P2R Extract Entities (local LLM)"}
