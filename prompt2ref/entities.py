"""Entity model shared by the extractor, the shot planner and the manifest."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum


class EntityType(str, Enum):
    PERSON = "person"
    FICTIONAL_CHARACTER = "fictional_character"
    LANDMARK = "landmark"          # Taj Mahal, Eiffel Tower
    PLACE = "place"                # Kyoto, Times Square
    ORGANIZATION = "organization"  # brands, teams, studios -> logo + context
    PRODUCT = "product"            # iPhone 16, Nike Air Max
    VEHICLE = "vehicle"            # Porsche 911, USS Enterprise
    ANIMAL = "animal"              # named breeds/species/individuals
    ARTWORK = "artwork"            # paintings, sculptures, album covers
    FOOD = "food"                  # named dishes
    EVENT = "event"                # Coachella, Diwali
    OTHER = "other"

    @classmethod
    def coerce(cls, value) -> "EntityType":
        if isinstance(value, cls):  # already typed - str() on a str-Enum is "EntityType.X"
            return value
        if not value:
            return cls.OTHER
        v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "celebrity": cls.PERSON,
            "actor": cls.PERSON,
            "human": cls.PERSON,
            "character": cls.FICTIONAL_CHARACTER,
            "monument": cls.LANDMARK,
            "building": cls.LANDMARK,
            "location": cls.PLACE,
            "city": cls.PLACE,
            "country": cls.PLACE,
            "brand": cls.ORGANIZATION,
            "company": cls.ORGANIZATION,
            "org": cls.ORGANIZATION,
            "device": cls.PRODUCT,
            "object": cls.PRODUCT,
            "car": cls.VEHICLE,
            "creature": cls.ANIMAL,
            "artwork": cls.ARTWORK,
            "painting": cls.ARTWORK,
            "dish": cls.FOOD,
        }
        if v in aliases:
            return aliases[v]
        try:
            return cls(v)
        except ValueError:
            return cls.OTHER


def slugify(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = re.sub(r"[^\w\s-]", "", n).strip().lower()
    n = re.sub(r"[\s_-]+", "-", n)
    return n or "entity"


@dataclass
class Entity:
    name: str
    type: EntityType = EntityType.OTHER
    descriptor: str = ""        # disambiguator, e.g. "American actor, born 1962"
    search_name: str = ""       # what to actually type into image search
    aliases: list[str] = field(default_factory=list)
    mention: str = ""           # the surface form found in the prompt
    confidence: float = 1.0

    def __post_init__(self):
        self.type = EntityType.coerce(self.type)
        self.name = self.name.strip()
        if not self.search_name:
            self.search_name = self.name

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def query_base(self) -> str:
        """Search name plus a short disambiguator when one is useful."""
        base = self.search_name.strip()
        d = (self.descriptor or "").strip().rstrip(".")
        if d and len(d) <= 40 and d.lower() not in base.lower():
            return f"{base} {d}"
        return base

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        d["slug"] = self.slug
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            name=d.get("name", ""),
            type=EntityType.coerce(d.get("type")),
            descriptor=d.get("descriptor", "") or "",
            search_name=d.get("search_name", "") or "",
            aliases=list(d.get("aliases") or []),
            mention=d.get("mention", "") or "",
            confidence=float(d.get("confidence", 1.0) or 1.0),
        )


def dedupe_entities(entities: list[Entity]) -> list[Entity]:
    """Collapse duplicates by slug, keeping the richest record."""
    out: dict[str, Entity] = {}
    for e in entities:
        if not e.name:
            continue
        key = e.slug
        cur = out.get(key)
        if cur is None:
            out[key] = e
            continue
        if cur.type is EntityType.OTHER and e.type is not EntityType.OTHER:
            cur.type = e.type
        if not cur.descriptor and e.descriptor:
            cur.descriptor = e.descriptor
        for a in e.aliases:
            if a not in cur.aliases:
                cur.aliases.append(a)
    return list(out.values())
