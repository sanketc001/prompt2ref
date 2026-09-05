"""Per-entity-type shot plans.

A "shot" is one reference angle we want on disk for an entity. The plan is what
turns "Tom Cruise" into a closeup portrait *and* a full-body frame, and
"iPhone 16" into a clean product shot *and* an in-use shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .entities import EntityType


@dataclass(frozen=True)
class Shot:
    key: str                      # file prefix, e.g. "portrait"
    label: str                    # human label for the UI
    queries: tuple[str, ...]      # "{q}" is replaced with the entity query base
    count: int = 2                # how many images to keep
    min_width: int = 640
    min_height: int = 640
    aspect: tuple[float, float] = (0.3, 3.0)   # accepted w/h range
    prefer_aspect: float | None = None         # ranking target, w/h
    allow_alpha: bool = False     # keep PNG transparency (logos)
    required: bool = True         # a plan fails loudly only if a required shot is empty


PERSON_SHOTS = (
    Shot(
        key="portrait",
        label="Closeup portrait",
        queries=(
            "{q} closeup portrait face photo",
            "{q} headshot high resolution",
            "{q} face close up photograph",
        ),
        count=2,
        prefer_aspect=0.8,
        aspect=(0.5, 1.6),
    ),
    Shot(
        key="fullbody",
        label="Full body",
        queries=(
            "{q} full body photo standing full length",
            "{q} full length red carpet photo",
            "{q} full body outfit photo head to toe",
        ),
        count=2,
        prefer_aspect=0.62,
        aspect=(0.3, 1.1),
    ),
    Shot(
        key="profile",
        label="Three-quarter / profile",
        queries=("{q} side profile photo", "{q} three quarter view portrait"),
        count=1,
        aspect=(0.5, 1.6),
        required=False,
    ),
)

PLACE_SHOTS = (
    Shot(
        key="establishing",
        label="Wide establishing",
        queries=(
            "{q} wide exterior photograph high resolution",
            "{q} landscape photo daytime",
        ),
        count=2,
        prefer_aspect=1.6,
        aspect=(0.8, 3.0),
    ),
    Shot(
        key="detail",
        label="Detail",
        queries=("{q} architectural detail closeup photo", "{q} close up photo"),
        count=1,
        required=False,
    ),
)

PRODUCT_SHOTS = (
    Shot(
        key="product",
        label="Product shot",
        queries=(
            "{q} product photo white background official",
            "{q} press render transparent background",
            "{q} studio product photography",
        ),
        count=2,
    ),
    Shot(
        key="context",
        label="In use / lifestyle",
        queries=("{q} in hand real life photo", "{q} lifestyle photo being used"),
        count=1,
        required=False,
    ),
)

ORG_SHOTS = (
    Shot(
        key="logo",
        label="Logo",
        queries=("{q} logo png transparent background", "{q} official logo high resolution"),
        count=2,
        min_width=400,
        min_height=200,
        aspect=(0.2, 5.0),
        allow_alpha=True,
    ),
    Shot(
        key="context",
        label="Context",
        queries=("{q} headquarters building photo", "{q} storefront photo"),
        count=1,
        required=False,
    ),
)

VEHICLE_SHOTS = (
    Shot(
        key="exterior",
        label="Exterior 3/4",
        queries=("{q} three quarter front exterior photo", "{q} official press photo side view"),
        count=2,
        prefer_aspect=1.5,
    ),
    Shot(
        key="interior",
        label="Interior",
        queries=("{q} interior dashboard photo",),
        count=1,
        required=False,
    ),
)

ANIMAL_SHOTS = (
    Shot(
        key="fullbody",
        label="Full body",
        queries=("{q} full body photo", "{q} standing side view photo"),
        count=2,
    ),
    Shot(
        key="closeup",
        label="Head closeup",
        queries=("{q} head closeup photo", "{q} face close up"),
        count=1,
    ),
)

ARTWORK_SHOTS = (
    Shot(
        key="artwork",
        label="The work",
        queries=("{q} high resolution image of the artwork", "{q} full painting scan"),
        count=2,
        aspect=(0.3, 3.0),
    ),
)

FOOD_SHOTS = (
    Shot(
        key="plated",
        label="Plated closeup",
        queries=("{q} plated dish closeup food photography", "{q} authentic photo of the dish"),
        count=2,
    ),
    Shot(
        key="overhead",
        label="Overhead",
        queries=("{q} overhead flat lay food photo",),
        count=1,
        required=False,
    ),
)

CHARACTER_SHOTS = (
    Shot(
        key="fullbody",
        label="Full body",
        queries=("{q} character full body official art", "{q} full body costume reference"),
        count=2,
        prefer_aspect=0.7,
        aspect=(0.3, 1.4),
    ),
    Shot(
        key="portrait",
        label="Closeup",
        queries=("{q} face closeup", "{q} character portrait"),
        count=2,
        aspect=(0.5, 1.6),
    ),
)

GENERIC_SHOTS = (
    Shot(
        key="reference",
        label="Reference",
        queries=("{q} photo high resolution", "{q}"),
        count=3,
    ),
)

SHOT_PLANS: dict[EntityType, tuple[Shot, ...]] = {
    EntityType.PERSON: PERSON_SHOTS,
    EntityType.FICTIONAL_CHARACTER: CHARACTER_SHOTS,
    EntityType.LANDMARK: PLACE_SHOTS,
    EntityType.PLACE: PLACE_SHOTS,
    EntityType.ORGANIZATION: ORG_SHOTS,
    EntityType.PRODUCT: PRODUCT_SHOTS,
    EntityType.VEHICLE: VEHICLE_SHOTS,
    EntityType.ANIMAL: ANIMAL_SHOTS,
    EntityType.ARTWORK: ARTWORK_SHOTS,
    EntityType.FOOD: FOOD_SHOTS,
    EntityType.EVENT: GENERIC_SHOTS,
    EntityType.OTHER: GENERIC_SHOTS,
}


def plan_for(entity_type: EntityType, per_shot: int | None = None) -> tuple[Shot, ...]:
    """Shot plan for a type, optionally forcing a uniform image count."""
    shots = SHOT_PLANS.get(EntityType.coerce(entity_type), GENERIC_SHOTS)
    if per_shot is None:
        return shots
    return tuple(
        Shot(
            key=s.key, label=s.label, queries=s.queries, count=max(1, per_shot),
            min_width=s.min_width, min_height=s.min_height, aspect=s.aspect,
            prefer_aspect=s.prefer_aspect, allow_alpha=s.allow_alpha, required=s.required,
        )
        for s in shots
    )


def expand_queries(shot: Shot, query_base: str) -> list[str]:
    return [q.replace("{q}", query_base) for q in shot.queries]
