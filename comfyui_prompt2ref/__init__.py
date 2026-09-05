"""Prompt2Ref as ComfyUI nodes - the same prompt->entities->references->H3-prompt
pipeline as the standalone tool, but driven by a locally-loaded CLIP/LLM
(e.g. Qwen3-VL-8B via CLIPLoader) instead of the Anthropic API.

    prompt --[P2RExtractEntities]--> entities_json
    entities_json --[P2RFetchReferences]--> manifest_json (images on disk)
    manifest_json --[P2RVerifyReferences]--> manifest_json (rejects dropped)
    manifest_json --[P2RPackReferences]--> ref_image_1..9, captions_json
    captions_json --[P2RBuildH3Prompt]--> h3_prompt (-> MiniMaxH3ReferenceToVideo.prompt)
"""

from .nodes_extract import NODE_CLASS_MAPPINGS as _EXTRACT_CLASSES
from .nodes_extract import NODE_DISPLAY_NAME_MAPPINGS as _EXTRACT_NAMES
from .nodes_slots import NODE_CLASS_MAPPINGS as _SLOTS_CLASSES
from .nodes_slots import NODE_DISPLAY_NAME_MAPPINGS as _SLOTS_NAMES
from .nodes_fetch import NODE_CLASS_MAPPINGS as _FETCH_CLASSES
from .nodes_fetch import NODE_DISPLAY_NAME_MAPPINGS as _FETCH_NAMES
from .nodes_verify import NODE_CLASS_MAPPINGS as _VERIFY_CLASSES
from .nodes_verify import NODE_DISPLAY_NAME_MAPPINGS as _VERIFY_NAMES
from .nodes_pack import NODE_CLASS_MAPPINGS as _PACK_CLASSES
from .nodes_pack import NODE_DISPLAY_NAME_MAPPINGS as _PACK_NAMES
from .nodes_h3prompt import NODE_CLASS_MAPPINGS as _H3PROMPT_CLASSES
from .nodes_h3prompt import NODE_DISPLAY_NAME_MAPPINGS as _H3PROMPT_NAMES

NODE_CLASS_MAPPINGS = {
    **_EXTRACT_CLASSES, **_SLOTS_CLASSES, **_FETCH_CLASSES, **_VERIFY_CLASSES,
    **_PACK_CLASSES, **_H3PROMPT_CLASSES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_EXTRACT_NAMES, **_SLOTS_NAMES, **_FETCH_NAMES, **_VERIFY_NAMES,
    **_PACK_NAMES, **_H3PROMPT_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
