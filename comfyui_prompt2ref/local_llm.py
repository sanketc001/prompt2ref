"""Local-model call helper shared by the P2R nodes.

Mirrors ComfyUI's own `TextGenerate` node (comfy_extras/nodes_textgen.py):
clip.tokenize -> clip.generate -> clip.decode. Load the CLIP with CLIPLoader
using a `type` that is NOT one of the QWEN3VL-specialized values (ideogram4,
boogu, krea2, mage, minimax) - e.g. "stable_diffusion" - so an 8B Qwen3-VL
checkpoint gets the generic chat template instead of another model's
conditioning-only template.
"""

from __future__ import annotations

import json
import re


def generate_text(clip, prompt: str, max_length: int = 768,
                   temperature: float = 0.4, top_p: float = 0.9, top_k: int = 40,
                   min_p: float = 0.0, repetition_penalty: float = 1.05,
                   seed: int = 0, do_sample: bool = True, **tokenize_kwargs) -> str:
    tokens = clip.tokenize(prompt, **tokenize_kwargs)
    generated_ids = clip.generate(
        tokens, do_sample=do_sample, max_length=max_length, temperature=temperature,
        top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
        seed=seed,
    )
    text = clip.decode(generated_ids)
    # Qwen3 emits an (empty, since we don't ask it to think) <think>...</think> block first.
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def extract_json(text: str):
    """Find the first balanced {...} or [...] block in `text` and parse it.

    Local models don't get Claude's json_schema-constrained output, so they
    sometimes wrap the JSON in prose or a code fence - scan for a balanced
    block by bracket depth rather than trusting a naive regex.
    """
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_ch:
                    depth += 1
                elif text[i] == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find(open_ch, start + 1)
    return None
