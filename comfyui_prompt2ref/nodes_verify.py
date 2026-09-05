"""Visual QC pass: show each entity's downloaded frames to the local model
and drop the ones that would poison an image-to-image reference, same idea
as prompt2ref.verify but judged by the local CLIP-loaded LLM instead of Claude.
"""

from __future__ import annotations

import json

from . import _repo  # noqa: F401
from . import local_llm

MAX_PER_CALL = 6
QC_EDGE = 448  # square resize so multiple images can share one tokenize() call


def _load_square(path, edge=QC_EDGE):
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path)
    img.load()
    img = img.convert("RGB").resize((edge, edge), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ...]  # [1, H, W, C]


class P2RVerifyReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "manifest_json": ("STRING", {"multiline": True, "default": "{}"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_json", "report")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, clip, manifest_json, enabled=True, max_length=512, temperature=0.2, seed=0):
        manifest = json.loads(manifest_json or "{}")
        entities = manifest.get("entities", [])
        report_lines = []

        if not enabled:
            return (json.dumps(manifest), "visual QC disabled - all images kept")

        for entity in entities:
            images = entity.get("images", [])
            if not images:
                continue
            kept, rejected = [], []
            for start in range(0, len(images), MAX_PER_CALL):
                chunk = images[start:start + MAX_PER_CALL]
                verdicts = self._verify_chunk(clip, entity, chunk, max_length, temperature, seed)
                for i, img in enumerate(chunk):
                    ok, reason = verdicts.get(i + 1, (True, ""))
                    (kept if ok else rejected).append((img, reason))

            if rejected:
                import os
                for img, reason in rejected:
                    path = img.get("abs_path")
                    if path and os.path.exists(path):
                        os.unlink(path)
                    report_lines.append("  - %s/%s rejected - %s" % (
                        entity.get("slug", entity.get("name", "?")), img.get("file", ""), reason))

            entity["images"] = [img for img, _ in kept]
            if entity.get("primary") and not any(
                entity["primary"] == img.get("path") for img, _ in kept
            ):
                entity["primary"] = kept[0][0]["path"] if kept else None
            report_lines.append("%s: kept %d, rejected %d" % (
                entity.get("name", "?"), len(kept), len(rejected)))

        return (json.dumps(manifest), "\n".join(report_lines) or "(no images to check)")

    @staticmethod
    def _verify_chunk(clip, entity, chunk, max_length, temperature, seed) -> dict[int, tuple[bool, str]]:
        from prompt2ref.verify import SYSTEM

        images = []
        lines = []
        for i, img in enumerate(chunk):
            path = img.get("abs_path")
            if not path:
                continue
            try:
                images.append(_load_square(path))
            except Exception:
                continue
            lines.append("Image %d - requested shot: %s" % (i + 1, img.get("shot", "reference")))
        if not images:
            return {}

        label = entity.get("name", "") + (
            " (%s)" % entity["descriptor"] if entity.get("descriptor") else "")
        instruction = (
            SYSTEM + "\n\n"
            + "\n".join(lines)
            + "\n\nEntity: %s\nType: %s\nJudge all %d images.\n\n" % (
                label, entity.get("type", "other"), len(images))
            + 'Output ONLY a single JSON object, no prose, of the exact shape '
              '{"verdicts": [{"index": int, "ok": bool, "reason": str}]}, one entry per image.'
        )
        try:
            text = local_llm.generate_text(
                clip, instruction, max_length=max_length, temperature=temperature,
                seed=seed, images=images)
            data = local_llm.extract_json(text)
        except Exception as exc:
            print("[prompt2ref] local verify failed (%r); keeping all images in this chunk" % exc)
            return {}

        out: dict[int, tuple[bool, str]] = {}
        if isinstance(data, dict):
            for v in data.get("verdicts", []):
                if isinstance(v, dict) and "index" in v:
                    out[int(v["index"])] = (bool(v.get("ok", True)), str(v.get("reason", "")))
        return out


NODE_CLASS_MAPPINGS = {"P2RVerifyReferences": P2RVerifyReferences}
NODE_DISPLAY_NAME_MAPPINGS = {"P2RVerifyReferences": "P2R Verify References (local LLM)"}
