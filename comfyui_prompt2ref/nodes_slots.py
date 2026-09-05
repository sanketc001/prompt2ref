"""Up to 5 manually-editable, individually-switchable entity slots sitting
between P2RExtractEntities and P2RFetchReferences.

Each P2REntitySlot pulls entity `index` out of the auto-extracted list as a
starting point (or starts blank if the extractor didn't find that many /
enabled is off), lets you override name/type/descriptor, and can be switched
off entirely - a disabled slot contributes nothing, so <Picture N> numbering
downstream stays compact instead of leaving gaps.
"""

from __future__ import annotations

import json

from . import _repo  # noqa: F401

MAX_SLOTS = 5


class P2REntitySlot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "entities_json": ("STRING", {"multiline": True, "default": "[]"}),
                "index": ("INT", {"default": 0, "min": 0, "max": MAX_SLOTS - 1,
                                  "tooltip": "which auto-extracted entity (in order) this slot starts from"}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "name_override": ("STRING", {"default": ""}),
                "type_override": ("STRING", {"default": "",
                                             "tooltip": "e.g. person, landmark, product - blank keeps the extracted type"}),
                "descriptor_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("entity_json",)
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, entities_json, index=0, enabled=True,
            name_override="", type_override="", descriptor_override=""):
        if not enabled:
            return ("",)

        from prompt2ref.entities import Entity, EntityType

        try:
            arr = json.loads(entities_json or "[]")
        except json.JSONDecodeError:
            arr = []
        base = arr[index] if isinstance(arr, list) and 0 <= index < len(arr) else {}

        name = (name_override or "").strip() or base.get("name", "")
        if not name:
            return ("",)

        manual = bool((name_override or "").strip())
        ent = Entity(
            name=name,
            type=EntityType.coerce((type_override or "").strip() or base.get("type", "other")),
            descriptor=(descriptor_override or "").strip() or base.get("descriptor", ""),
            search_name="" if manual else base.get("search_name", ""),
            aliases=[] if manual else list(base.get("aliases", [])),
            mention="" if manual else base.get("mention", ""),
            confidence=1.0 if manual else float(base.get("confidence", 1.0)),
        )
        return (json.dumps(ent.to_dict()),)


class P2RCombineEntities:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "entity_json_1": ("STRING", {"default": ""}),
                "entity_json_2": ("STRING", {"default": ""}),
                "entity_json_3": ("STRING", {"default": ""}),
                "entity_json_4": ("STRING", {"default": ""}),
                "entity_json_5": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("entities_json", "summary")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, entity_json_1="", entity_json_2="", entity_json_3="",
            entity_json_4="", entity_json_5=""):
        from prompt2ref.entities import Entity, dedupe_entities

        entities = []
        for raw in (entity_json_1, entity_json_2, entity_json_3, entity_json_4, entity_json_5):
            if not raw or not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("name"):
                entities.append(Entity.from_dict(d))

        entities = dedupe_entities(entities)
        payload = [e.to_dict() for e in entities]
        summary = ("%d entities: " % len(payload)) + ", ".join(
            "%s (%s)" % (e["name"], e["type"]) for e in payload) if payload else "(no entities enabled)"
        return (json.dumps(payload), summary)


NODE_CLASS_MAPPINGS = {
    "P2REntitySlot": P2REntitySlot,
    "P2RCombineEntities": P2RCombineEntities,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "P2REntitySlot": "P2R Entity Slot (switchable)",
    "P2RCombineEntities": "P2R Combine Entity Slots",
}
