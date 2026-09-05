"""entities -> downloaded reference images on disk (no LLM involved).

Reuses prompt2ref.pipeline.Pipeline as-is: same web search backends, same
dedupe/accept rules, same on-disk refs/<slug>/entity.json layout as the CLI
and web UI. `settings.verify` is forced off here - visual QC is a separate
node (P2RVerifyReferences) so it can use the local model instead of Claude.
"""

from __future__ import annotations

import json

from . import _repo  # noqa: F401


class P2RFetchReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "entities_json": ("STRING", {"multiline": True, "default": "[]"}),
                "per_shot": ("INT", {"default": -1, "min": -1, "max": 20,
                                     "tooltip": "-1 = each entity type's own default shot count"}),
                "force": ("BOOLEAN", {"default": False, "tooltip": "refetch even if the folder is filled"}),
                "sources": ("STRING", {"default": "ddg,wikimedia,openverse"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_json",)
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, entities_json, per_shot=-1, force=False, sources="ddg,wikimedia,openverse"):
        from prompt2ref.config import load_settings
        from prompt2ref.entities import Entity
        from prompt2ref.pipeline import Pipeline

        try:
            raw = json.loads(entities_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("entities_json is not valid JSON: %s" % exc)

        entities = [Entity.from_dict(d) for d in raw if isinstance(d, dict) and d.get("name")]
        if not entities:
            return (json.dumps({"entities": []}),)

        src_tuple = tuple(s.strip() for s in sources.split(",") if s.strip())
        settings = load_settings(sources=src_tuple or None, verify=False)
        pipe = Pipeline(settings)

        records = [
            pipe.collect_entity(e, force=force, per_shot=(None if per_shot < 0 else per_shot))
            for e in entities
        ]
        return (json.dumps({"entities": records}),)


NODE_CLASS_MAPPINGS = {"P2RFetchReferences": P2RFetchReferences}
NODE_DISPLAY_NAME_MAPPINGS = {"P2RFetchReferences": "P2R Fetch References"}
