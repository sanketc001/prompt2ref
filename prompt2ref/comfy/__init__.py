"""ComfyUI bridge: Prompt2Ref references -> MiniMax H3 Ref2VA workflow."""

from .builder import BuildOptions, BuildResult, build_workflow, load_manifest
from .imagepack import PACK_MODES, ScalePolicy, pack_entities
from .template import Ref2VATemplate, solve_resolution

__all__ = [
    "BuildOptions",
    "BuildResult",
    "build_workflow",
    "load_manifest",
    "PACK_MODES",
    "ScalePolicy",
    "pack_entities",
    "Ref2VATemplate",
    "solve_resolution",
]
