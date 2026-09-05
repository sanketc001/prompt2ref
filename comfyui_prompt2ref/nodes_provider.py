"""LLM providers: P2RExtractEntities / P2RVerifyReferences / P2RBuildH3Prompt
each take a `provider` input (custom type "P2R_PROVIDER") instead of hard-
wiring the local CLIP - swap in whichever of these two feeds it:

    P2RLocalProvider     - the local CLIP-loaded model (e.g. Qwen3-VL-8B),
                           zero cost, runs on this machine.
    P2RAnthropicProvider - a hosted Claude model over the API (e.g. Haiku
                           4.5), for when you want a stronger/faster model
                           for a given step without spending local GPU time.

Both expose the same interface: `.generate(system_prompt, user_prompt,
images=None) -> str`, so the three consumer nodes don't care which one they
got. `images`, when given, is a list of ComfyUI IMAGE tensors ([1,H,W,C]).
"""

from __future__ import annotations

from . import _repo  # noqa: F401
from . import local_llm


def _b64_jpeg(image_tensor) -> str:
    import base64
    import io

    import numpy as np
    from PIL import Image

    arr = (image_tensor[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


class LocalCLIPProvider:
    def __init__(self, clip, max_length=768, temperature=0.4, top_p=0.9,
                 top_k=40, seed=0):
        self.clip = clip
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed

    def generate(self, system_prompt: str, user_prompt: str, images=None) -> str:
        # The local model has no separate system-role slot at this call site
        # (see local_llm.generate_text / ComfyUI's own TextGenerate node) -
        # fold it into the same turn.
        full_prompt = (system_prompt.strip() + "\n\n" + user_prompt) if system_prompt else user_prompt
        kwargs = {"images": images} if images else {}
        return local_llm.generate_text(
            self.clip, full_prompt, max_length=self.max_length,
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            seed=self.seed, **kwargs,
        )


class AnthropicProvider:
    def __init__(self, model="claude-haiku-4-5", max_tokens=1024):
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str, images=None) -> str:
        from prompt2ref.config import load_settings

        settings = load_settings()
        if not settings.api_key:
            raise RuntimeError("no ANTHROPIC_API_KEY configured")

        import anthropic

        content = [{"type": "text", "text": user_prompt}]
        for img in (images or []):
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": _b64_jpeg(img)}})

        client = anthropic.Anthropic(api_key=settings.api_key)
        response = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system_prompt or None,
            messages=[{"role": "user", "content": content}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("model declined to respond")
        return next((b.text for b in response.content if b.type == "text"), "")


class P2RLocalProvider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "max_length": ("INT", {"default": 768, "min": 128, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 500}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("P2R_PROVIDER",)
    RETURN_NAMES = ("provider",)
    FUNCTION = "run"
    CATEGORY = "prompt2ref/providers"

    def run(self, clip, max_length=768, temperature=0.4, top_p=0.9, top_k=40, seed=0):
        return (LocalCLIPProvider(clip, max_length, temperature, top_p, top_k, seed),)


class P2RAnthropicProvider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "claude-haiku-4-5",
                                     "tooltip": "any model id your ANTHROPIC_API_KEY can use"}),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 8192}),
            }
        }

    RETURN_TYPES = ("P2R_PROVIDER",)
    RETURN_NAMES = ("provider",)
    FUNCTION = "run"
    CATEGORY = "prompt2ref/providers"

    def run(self, model="claude-haiku-4-5", max_tokens=1024):
        return (AnthropicProvider(model, max_tokens),)


NODE_CLASS_MAPPINGS = {
    "P2RLocalProvider": P2RLocalProvider,
    "P2RAnthropicProvider": P2RAnthropicProvider,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "P2RLocalProvider": "P2R Provider: Local CLIP",
    "P2RAnthropicProvider": "P2R Provider: Anthropic API",
}
