"""ComfyUI bridge: Prompt2Ref references -> video and image workflows.

    video   MiniMax H3 Ref2VA, by filling in the shipped template
    image   Qwen-Image-Edit / Flux.2 Klein / Qwen-Image, built from scratch
            (generation from references, and editing an existing frame)
"""

from .builder import BuildOptions, BuildResult, build_workflow, load_manifest
from .imagebuild import ImageBuildOptions, ImageBuildResult, build_image_workflow
from .imagegraph import BACKENDS, DEFAULT_BACKEND, ImageBackend, solve_image_resolution
from .imagepack import PACK_MODES, ScalePolicy, pack_entities
from .imageprompt import MODES as IMAGE_MODES
from .template import Ref2VATemplate, solve_resolution
from .uigraph import api_to_ui, save_ui_workflow

__all__ = [
    "BuildOptions",
    "BuildResult",
    "build_workflow",
    "load_manifest",
    "ImageBuildOptions",
    "ImageBuildResult",
    "build_image_workflow",
    "BACKENDS",
    "DEFAULT_BACKEND",
    "ImageBackend",
    "IMAGE_MODES",
    "solve_image_resolution",
    "PACK_MODES",
    "ScalePolicy",
    "pack_entities",
    "Ref2VATemplate",
    "solve_resolution",
    "api_to_ui",
    "save_ui_workflow",
]
