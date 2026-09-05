"""Visual QC pass: show each entity's downloaded frames to an LLM provider
(P2RLocalProvider or P2RAnthropicProvider) and drop the ones that would
poison an image-to-image reference - same idea as prompt2ref.verify, judged
by whichever provider is wired in instead of a hardcoded Claude API call.
"""

from __future__ import annotations

import json

from . import _repo  # noqa: F401
from . import local_llm

MAX_PER_CALL = 6
QC_EDGE = 448  # square resize so multiple images can share one call


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
                "provider": ("P2R_PROVIDER",),
                "manifest_json": ("STRING", {"multiline": True, "default": "{}"}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_json", "report")
    FUNCTION = "run"
    CATEGORY = "prompt2ref"

    def run(self, provider, manifest_json, enabled=True):
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
                verdicts = self._verify_chunk(provider, entity, chunk)
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
    def _verify_chunk(provider, entity, chunk) -> dict[int, tuple[bool, str]]:
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
        user_prompt = (
            "\n".join(lines)
            + "\n\nEntity: %s\nType: %s\nJudge all %d images.\n\n" % (
                label, entity.get("type", "other"), len(images))
            + 'Output ONLY a single JSON object, no prose, of the exact shape '
              '{"verdicts": [{"index": int, "ok": bool, "reason": str}]}, one entry per image.'
        )
        try:
            text = provider.generate(SYSTEM, user_prompt, images=images)
            data = local_llm.extract_json(text)
        except Exception as exc:
            print("[prompt2ref] verify failed (%r); keeping all images in this chunk" % exc)
            return {}

        out: dict[int, tuple[bool, str]] = {}
        if isinstance(data, dict):
            for v in data.get("verdicts", []):
                if isinstance(v, dict) and "index" in v:
                    out[int(v["index"])] = (bool(v.get("ok", True)), str(v.get("reason", "")))
        return out


NODE_CLASS_MAPPINGS = {"P2RVerifyReferences": P2RVerifyReferences}
NODE_DISPLAY_NAME_MAPPINGS = {"P2RVerifyReferences": "P2R Verify References"}
