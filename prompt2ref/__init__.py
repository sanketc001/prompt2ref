"""Prompt2Ref - turn a text prompt into a local library of reference images.

Pipeline: prompt -> named entities -> per-type shot plan -> image search ->
download/validate/dedupe -> refs/<slug>/ folders + a manifest for downstream
image-to-image / image-to-video workflows.
"""

from .entities import Entity, EntityType
from .pipeline import Pipeline, build_references
from .config import Settings, load_settings

__all__ = [
    "Entity",
    "EntityType",
    "Pipeline",
    "build_references",
    "Settings",
    "load_settings",
]
__version__ = "0.1.0"
