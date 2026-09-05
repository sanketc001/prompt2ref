"""manifest -> up to 9 packed <Picture N> sheets, wired straight to
MiniMaxH3ReferenceToVideo's ref_image_1..9 inputs. Reuses
prompt2ref.comfy.imagepack.pack_entities() - the same tiling/contact-sheet
code the CLI/web-UI path uses to build a static workflow JSON - but returns
live IMAGE tensors instead of writing a pre-baked LoadImage graph.
"""

from __future__ import annotations

import json

from . import _repo  # noqa: F401

MAX_SLOTS = 9


def _load_tensor(path):
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ...]


class P2RPackReferences:
    @classmethod
    def INPUT_TYPES(cls):
        from prompt2ref.comfy.imagepack import PACK_MODES

        return {
            "required": {
                "manifest_json": ("STRING", {"multiline": True, "default": "{}"}),
                "pack_mode": (list(PACK_MODES), {"default": "per-entity"}),
                "ref_max": ("INT", {"default": 1024, "min": 0, "max": 4096,
                                    "tooltip": "longest edge cap; 0 = native resolution"}),
                "labels": ("BOOLEAN", {"default": False, "tooltip": "draw entity names on multi-image sheets"}),
            }
        }

    RETURN_TYPES = ("IMAGE",) * MAX_SLOTS + ("STRING", "STRING")
    RETURN_NAMES = tuple("ref_image_%d" % i for i in range(1, MAX_SLOTS + 1)) + (
        "captions_json", "warnings")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, manifest_json, pack_mode="per-entity", ref_max=1024, labels=False):
        from prompt2ref.comfy.imagepack import ScalePolicy, pack_entities
        from prompt2ref.config import load_settings

        manifest = json.loads(manifest_json or "{}")
        settings = load_settings()
        out_dir = settings.refs_dir.parent / "comfy_staging" / "packed"

        policy = ScalePolicy.native() if not ref_max else ScalePolicy(max_edge=ref_max)
        packed, warnings = pack_entities(
            manifest.get("entities", []), out_dir=out_dir, slots=MAX_SLOTS,
            mode=pack_mode, policy=policy, labels=labels)

        slots = [None] * MAX_SLOTS
        captions = []
        for p in packed:
            if 1 <= p.slot <= MAX_SLOTS:
                slots[p.slot - 1] = _load_tensor(p.path)
            captions.append({"slot": p.slot, "caption": p.caption,
                             "width": p.width, "height": p.height})

        return tuple(slots) + (json.dumps(captions), "\n".join(warnings) or "(no warnings)")


NODE_CLASS_MAPPINGS = {"P2RPackReferences": P2RPackReferences}
NODE_DISPLAY_NAME_MAPPINGS = {"P2RPackReferences": "P2R Pack References (concatenate sheet)"}
