"""Prompt2Ref as ComfyUI nodes - the same prompt->entities->references->H3-prompt
pipeline as the standalone tool, driven by a swappable LLM provider instead of
a hardcoded Claude API call: P2RLocalProvider (a locally-loaded CLIP, e.g.
Qwen3-VL-8B) or P2RAnthropicProvider (a hosted Claude model over the API)
both feed the same `provider` input on the three consumer nodes.

    P2RLocalProvider / P2RAnthropicProvider --> provider
    provider + prompt --[P2RExtractEntities]--> entities_json
    entities_json --[P2REntitySlot x5 + P2RCombineEntities]--> entities_json (switchable)
    entities_json --[P2RFetchReferences]--> manifest_json (images on disk)
    provider + manifest_json --[P2RVerifyReferences]--> manifest_json (rejects dropped)

Everything above is shared. The tail depends on what you are making:

    video  manifest_json --[P2RPackReferences]--> ref_image_1..9, captions_json
           provider + captions_json --[P2RBuildH3Prompt]--> h3_prompt
                                       (-> MiniMaxH3ReferenceToVideo.prompt)

    image  manifest_json --[P2RPackReferencesImage]--> ref_image_1..4, captions_json
           provider + captions_json --[P2RBuildImagePrompt]--> image_prompt, negative_prompt
                                       (-> TextEncodeQwenImageEditPlus / CLIPTextEncode)
           megapixels + aspect --[P2RImageResolution]--> width, height
"""

from .nodes_provider import NODE_CLASS_MAPPINGS as _PROVIDER_CLASSES
from .nodes_provider import NODE_DISPLAY_NAME_MAPPINGS as _PROVIDER_NAMES
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
from .nodes_image import NODE_CLASS_MAPPINGS as _IMAGE_CLASSES
from .nodes_image import NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_NAMES

NODE_CLASS_MAPPINGS = {
    **_PROVIDER_CLASSES, **_EXTRACT_CLASSES, **_SLOTS_CLASSES, **_FETCH_CLASSES,
    **_VERIFY_CLASSES, **_PACK_CLASSES, **_H3PROMPT_CLASSES, **_IMAGE_CLASSES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_PROVIDER_NAMES, **_EXTRACT_NAMES, **_SLOTS_NAMES, **_FETCH_NAMES,
    **_VERIFY_NAMES, **_PACK_NAMES, **_H3PROMPT_NAMES, **_IMAGE_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
