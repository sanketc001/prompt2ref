"""The image half of the node pipeline: pack references for an image model's
few conditioning slots, write the image / edit prompt with a swappable LLM
provider, and solve the output size.

    manifest_json --[P2RPackReferencesImage]--> ref_image_1..4, captions_json
    provider + captions_json --[P2RBuildImagePrompt]--> image_prompt, negative_prompt
    megapixels + aspect --[P2RImageResolution]--> width, height

`image_prompt` feeds TextEncodeQwenImageEditPlus.prompt (or CLIPTextEncode.text
for Flux.2), and `ref_image_N` feeds that same node's image1/image2/image3 - or
a VAEEncode -> ReferenceLatent chain.

Image models take far fewer references than MiniMax H3's nine slots, which is
why this is a separate packer: three or four slots have to be spent well, so
the default is one best frame per entity rather than H3's tiled sheets, and
entities past the last slot can be folded onto it instead of dropped.
"""

from __future__ import annotations

import json

from . import _repo  # noqa: F401
from . import local_llm

MAX_SLOTS = 4
ASPECTS = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9")


def _load_tensor(path):
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ...]


class P2RPackReferencesImage:
    @classmethod
    def INPUT_TYPES(cls):
        from prompt2ref.comfy.imagepack import PACK_MODES

        return {
            "required": {
                "manifest_json": ("STRING", {"multiline": True, "default": "{}"}),
                "slots": ("INT", {"default": 3, "min": 1, "max": MAX_SLOTS,
                                  "tooltip": "how many conditioning images the model takes "
                                             "(Qwen-Image-Edit: 3)"}),
                "pack_mode": (list(PACK_MODES), {"default": "primary"}),
                "overflow": (["combine", "drop"], {
                    "default": "combine",
                    "tooltip": "what to do with entities past the last slot: tile them onto "
                               "it, or leave them out"}),
                "ref_max": ("INT", {"default": 1024, "min": 0, "max": 4096,
                                    "tooltip": "longest edge cap; 0 = native resolution"}),
                "labels": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",) * MAX_SLOTS + ("STRING", "STRING")
    RETURN_NAMES = tuple("ref_image_%d" % i for i in range(1, MAX_SLOTS + 1)) + (
        "captions_json", "warnings")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, manifest_json, slots=3, pack_mode="primary", overflow="combine",
            ref_max=1024, labels=False):
        from prompt2ref.comfy.imagepack import ScalePolicy, pack_entities
        from prompt2ref.config import load_settings

        manifest = json.loads(manifest_json or "{}")
        entities = [e for e in manifest.get("entities", []) if e.get("images")]
        settings = load_settings()
        out_dir = settings.refs_dir.parent / "comfy_staging" / "packed_image"
        policy = ScalePolicy.native() if not ref_max else ScalePolicy(max_edge=ref_max)

        packed, warnings = [], []
        if overflow == "combine" and len(entities) > slots > 1:
            head, tail = entities[:slots - 1], entities[slots - 1:]
            packed, warnings = pack_entities(
                head, out_dir=out_dir, slots=slots - 1, mode=pack_mode,
                policy=policy, labels=labels)
            rest, rest_warnings = pack_entities(
                tail, out_dir=out_dir, slots=1, mode="all-in-one",
                policy=policy, labels=True, prefix="p2rx")
            warnings.extend(rest_warnings)
            for p in rest:              # the combined sheet takes the last slot
                p.slot = slots
                packed.append(p)
            warnings.append("%d entities folded onto slot %d: %s"
                            % (len(tail), slots, ", ".join(e["name"] for e in tail)))
        elif entities:
            packed, warnings = pack_entities(
                entities, out_dir=out_dir, slots=slots, mode=pack_mode,
                policy=policy, labels=labels)

        images = [None] * MAX_SLOTS
        captions = []
        for p in packed:
            if 1 <= p.slot <= MAX_SLOTS:
                images[p.slot - 1] = _load_tensor(p.path)
            captions.append({"slot": p.slot, "caption": p.caption,
                             "width": p.width, "height": p.height})

        return tuple(images) + (json.dumps(captions),
                                "\n".join(warnings) or "(no warnings)")


class P2RBuildImagePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        from prompt2ref.comfy.imageprompt import MODES

        return {
            "required": {
                "provider": ("P2R_PROVIDER",),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "captions_json": ("STRING", {"multiline": True, "default": "[]"}),
                "mode": (list(MODES), {"default": "generate"}),
                "tagged_references": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "on for Qwen-Image-Edit, whose encoder tags each conditioning "
                               "image as 'Picture N'; off for models without such tags"}),
                "editing_image_first": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "the frame being edited occupies the first conditioning slot, "
                               "so references start at 2"}),
                "style": ("STRING", {"default": "", "multiline": False}),
                "use_llm": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompt", "negative_prompt")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, provider, user_prompt, captions_json, mode="generate",
            tagged_references=True, editing_image_first=False, style="", use_llm=True):
        from prompt2ref.comfy.imageprompt import DEFAULT_NEGATIVE, SYSTEM_EDIT, SYSTEM_GENERATE

        captions = json.loads(captions_json or "[]")
        offset = 1 if editing_image_first else 0
        parts = None

        if use_llm:
            try:
                system_prompt = (
                    (SYSTEM_EDIT if mode == "edit" else SYSTEM_GENERATE) + "\n\n"
                    "Output ONLY a single JSON object, no prose, no code fence, with exactly "
                    "the string keys prompt and negative_prompt."
                )
                user = "USER REQUEST:\n%s\n\nREFERENCE PICTURES:\n%s" % (
                    user_prompt.strip(), self._brief(captions, tagged_references, offset))
                if style.strip():
                    user += "\n\nSTYLE THE USER ASKED FOR: %s" % style.strip()
                data = local_llm.extract_json(provider.generate(system_prompt, user))
                if isinstance(data, dict) and data.get("prompt"):
                    parts = data
            except Exception as exc:
                print("[prompt2ref] image prompt rewrite failed (%r); "
                      "using deterministic layout" % exc)

        if parts is None:
            parts = {"prompt": self._fallback(user_prompt, captions, mode,
                                              tagged_references, offset),
                     "negative_prompt": DEFAULT_NEGATIVE}

        text = parts["prompt"].strip()
        if style.strip():
            text = "%s %s" % (text, style.strip().strip(","))
        return (text, parts.get("negative_prompt") or DEFAULT_NEGATIVE)

    @staticmethod
    def _brief(captions, tagged, offset) -> str:
        from prompt2ref.comfy.imageprompt import picture_tag

        lines = []
        if offset:
            lines.append("%s: the image being edited (already supplied)" % picture_tag(1, tagged))
        for c in captions:
            lines.append("%s: %s  [%dx%d]" % (
                picture_tag(c["slot"] + offset, tagged), c.get("caption", ""),
                c.get("width", 0), c.get("height", 0)))
        return "\n".join(lines) or "(no reference pictures)"

    @staticmethod
    def _fallback(user_prompt, captions, mode, tagged, offset) -> str:
        from prompt2ref.comfy.imageprompt import picture_tag

        binds = ["%s shows %s" % (picture_tag(c["slot"] + offset, tagged),
                                  c.get("caption", "a reference"))
                 for c in captions]
        bind_text = ("; ".join(binds) + ". Keep those subjects exactly as shown. ") if binds else ""
        if mode == "edit":
            head = "%s is the image being edited. " % picture_tag(1, tagged) if offset else ""
            return ("%s%sApply this change and leave the rest of the frame - composition, "
                    "background, framing and lighting - untouched: %s"
                    % (head, bind_text, user_prompt.strip()))
        return "%s%s" % (bind_text, user_prompt.strip())


class P2RImageResolution:
    """A megapixel budget and an aspect ratio -> a width/height the model likes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05}),
                "aspect": (list(ASPECTS), {"default": "1:1"}),
                "multiple": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8,
                                     "tooltip": "round each side up to this; 16 suits the "
                                                "Qwen-Image and Flux.2 latents"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "readout")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, megapixels=1.0, aspect="1:1", multiple=16):
        from prompt2ref.comfy.template import solve_resolution

        w, h = solve_resolution(megapixels, aspect, multiple=multiple)
        return (w, h, "%dx%d (%.2f MP, %s)" % (w, h, w * h / 1e6, aspect))


NODE_CLASS_MAPPINGS = {
    "P2RPackReferencesImage": P2RPackReferencesImage,
    "P2RBuildImagePrompt": P2RBuildImagePrompt,
    "P2RImageResolution": P2RImageResolution,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "P2RPackReferencesImage": "P2R Pack References (image slots)",
    "P2RBuildImagePrompt": "P2R Build Image / Edit Prompt",
    "P2RImageResolution": "P2R Image Resolution",
}
