"""Turn an entity's reference images into the file(s) a <Picture N> slot loads.

Two independent knobs:

* **scale** - `max_edge` (default 1024), `megapixels`, or native (no resampling).
* **pack**  - how entities map onto the nine picture slots:

    per-entity   one slot per entity; that entity's shots tiled into one sheet
    primary      one slot per entity, best single frame only
    per-image    every image gets its own slot (highest fidelity, fills fast)
    all-in-one   every entity tiled into a single sheet in <Picture 1>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PACK_MODES = ("per-entity", "primary", "per-image", "all-in-one")

SHEET_BG = (18, 18, 20)
SHEET_GAP = 12
LABEL_H = 34


@dataclass
class ScalePolicy:
    """How a reference image is resized before it reaches ComfyUI."""

    max_edge: int | None = 1024      # longest side cap; None/0 = no cap
    megapixels: float | None = None  # if set, overrides max_edge
    allow_upscale: bool = False

    @classmethod
    def native(cls) -> "ScalePolicy":
        return cls(max_edge=None, megapixels=None)

    def describe(self) -> str:
        if self.megapixels:
            return "%.2f MP" % self.megapixels
        if not self.max_edge:
            return "native"
        return "max %dpx" % self.max_edge

    def target_size(self, w: int, h: int) -> tuple[int, int]:
        if self.megapixels:
            scale = math.sqrt((self.megapixels * 1_000_000) / max(1, w * h))
        elif self.max_edge:
            scale = self.max_edge / max(w, h)
        else:
            return w, h
        if scale >= 1.0 and not self.allow_upscale:
            return w, h
        return max(8, int(round(w * scale))), max(8, int(round(h * scale)))

    def apply(self, img: Image.Image) -> Image.Image:
        tw, th = self.target_size(img.width, img.height)
        if (tw, th) == (img.width, img.height):
            return img
        return img.resize((tw, th), Image.Resampling.LANCZOS)


@dataclass
class PackedPicture:
    """One rendered file destined for one <Picture N> slot."""

    slot: int
    filename: str                    # relative to ComfyUI/input
    path: Path
    width: int
    height: int
    entities: list[str] = field(default_factory=list)   # entity names on this sheet
    sources: list[str] = field(default_factory=list)    # source image paths
    caption: str = ""                # what the prompt should say this picture shows


def _load(path: str | Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        return None
    return img.convert("RGB")


def _font(size: int):
    for name in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(
    images: list[tuple[str, Image.Image]],
    tile_policy: ScalePolicy,
    labels: bool = False,
) -> Image.Image:
    """Tile images into one sheet, each cell pre-capped by `tile_policy`.

    A grid is used rather than a strip so a 4-shot entity stays roughly square
    instead of becoming a 4000px letterbox the model has to squint at. The final
    size cap is applied to the assembled sheet by the caller, not here.
    """
    tiles = [(name, tile_policy.apply(im)) for name, im in images]
    if len(tiles) == 1 and not labels:
        return tiles[0][1]

    # Justified layout: normalise to a common row height, greedily fill rows to a
    # target width, then stretch each row to that width. A fixed grid wastes half
    # the sheet on background when portrait and landscape shots are mixed, and the
    # model only ever sees the pixels the subjects occupy.
    base_h = sorted(im.height for _, im in tiles)[len(tiles) // 2]
    norm = [(n, im, max(1, round(im.width * base_h / im.height))) for n, im in tiles]
    total_w = sum(w for _, _, w in norm)
    rows_n = max(1, round(math.sqrt(total_w / (base_h * 1.5))))
    row_target = max(base_h, total_w / rows_n)

    rows: list[list] = [[]]
    acc = 0.0
    for item in norm:
        if acc and acc + item[2] > row_target * 1.25:
            rows.append([])
            acc = 0.0
        rows[-1].append(item)
        acc += item[2]
    rows = [r for r in rows if r]

    laid: list[list[tuple[str, Image.Image]]] = []
    for row in rows:
        gaps = SHEET_GAP * (len(row) + 1)
        scale = (row_target - gaps) / max(1, sum(w for _, _, w in row))
        h = max(16, int(base_h * scale))
        laid.append([
            (n, im.resize((max(8, int(w * scale)), h), Image.Resampling.LANCZOS))
            for n, im, w in row
        ])

    label_h = LABEL_H if labels else 0
    sheet_w = int(row_target)
    sheet_h = sum(r[0][1].height + label_h for r in laid) + SHEET_GAP * (len(laid) + 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), SHEET_BG)
    draw = ImageDraw.Draw(sheet)
    font = _font(max(13, label_h - 14)) if labels else None

    y = SHEET_GAP
    for row in laid:
        x = SHEET_GAP
        for name, im in row:
            sheet.paste(im, (x, y))
            if font is not None:
                draw.text((x + 2, y + im.height + 4), name[:44],
                          fill=(232, 232, 236), font=font)
            x += im.width + SHEET_GAP
        y += row[0][1].height + label_h + SHEET_GAP
    return sheet


def pack_entities(
    entities: list[dict],
    out_dir: Path,
    slots: int = 9,
    mode: str = "per-entity",
    policy: ScalePolicy | None = None,
    labels: bool | None = None,
    prefix: str = "p2r",
    scale_per_tile: bool = False,
) -> tuple[list[PackedPicture], list[str]]:
    """Render the picture files. Returns (packed, warnings).

    `entities` are manifest entries: {name, slug, type, primary, images:[{abs_path, shot}]}
    """
    policy = policy or ScalePolicy()
    if mode not in PACK_MODES:
        raise ValueError("pack mode must be one of %s" % (PACK_MODES,))
    if labels is None:
        labels = mode == "all-in-one"   # a combined sheet needs names to bind to
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    packed: list[PackedPicture] = []

    # Tiles are pre-capped only to bound the intermediate; the delivered sheet is
    # what has to honour the cap, so the policy is applied again after assembly.
    tile_policy = policy if scale_per_tile else ScalePolicy(
        max_edge=policy.max_edge, megapixels=policy.megapixels)

    def finish(sheet: Image.Image) -> Image.Image:
        return sheet if scale_per_tile else policy.apply(sheet)

    def save(img: Image.Image, stem: str, slot: int, names, sources, caption) -> None:
        fname = "%s_%02d_%s.png" % (prefix, slot, stem)
        path = out_dir / fname
        img.save(path, "PNG", optimize=True)
        packed.append(PackedPicture(
            slot=slot, filename=fname, path=path, width=img.width, height=img.height,
            entities=list(names), sources=list(sources), caption=caption,
        ))

    usable = [e for e in entities if e.get("images")]
    for e in entities:
        if not e.get("images"):
            warnings.append("%s has no reference images - skipped" % e.get("name", "?"))

    if mode == "all-in-one":
        items, names, sources = [], [], []
        for e in usable:
            best = _ordered_images(e)[0]
            im = _load(best["abs_path"])
            if im is None:
                continue
            items.append((e["name"], im))
            names.append(e["name"])
            sources.append(best["abs_path"])
        if not items:
            return [], warnings + ["nothing to pack"]
        sheet = finish(contact_sheet(items, tile_policy, labels=labels))
        save(sheet, "all", 1, names, sources,
             "a reference sheet showing " + ", ".join(names))
        return packed, warnings

    slot = 1
    for e in usable:
        if slot > slots:
            warnings.append(
                "%s dropped - the workflow only has %d picture slots" % (e["name"], slots))
            continue
        imgs = _ordered_images(e)

        if mode == "per-image":
            for img_rec in imgs:
                if slot > slots:
                    warnings.append("%s/%s dropped - out of picture slots"
                                    % (e["name"], img_rec.get("shot", "")))
                    break
                im = _load(img_rec["abs_path"])
                if im is None:
                    continue
                save(policy.apply(im), "%s-%s" % (e["slug"], img_rec.get("shot", "ref")), slot,
                     [e["name"]], [img_rec["abs_path"]],
                     _caption(e, [img_rec]))
                slot += 1
            continue

        chosen = imgs[:1] if mode == "primary" else imgs
        loaded = [(_shot_label(e, r), _load(r["abs_path"])) for r in chosen]
        loaded = [(n, im) for n, im in loaded if im is not None]
        if not loaded:
            warnings.append("%s images could not be opened - skipped" % e["name"])
            continue
        sheet = finish(contact_sheet(loaded, tile_policy, labels=labels and len(loaded) > 1))
        save(sheet, e["slug"], slot, [e["name"]], [r["abs_path"] for r in chosen],
             _caption(e, chosen))
        slot += 1

    return packed, warnings


def _ordered_images(entity: dict) -> list[dict]:
    """Primary first, then the rest in plan order."""
    imgs = list(entity.get("images", []))
    primary = entity.get("primary")
    if primary:
        imgs.sort(key=lambda i: 0 if i.get("path") == primary else 1)
    return imgs


def _shot_label(entity: dict, rec: dict) -> str:
    return "%s - %s" % (entity["name"], rec.get("shot", "reference"))


SHOT_PHRASES = {
    "portrait": "a closeup of the face",
    "fullbody": "a full-body view",
    "profile": "a three-quarter/profile view",
    "establishing": "a wide establishing view",
    "detail": "a detail view",
    "product": "a clean product view",
    "context": "an in-context view",
    "logo": "the logo",
    "exterior": "the exterior",
    "interior": "the interior",
    "closeup": "a closeup",
    "artwork": "the work itself",
    "plated": "a plated closeup",
    "overhead": "an overhead view",
    "reference": "a reference view",
}


def _caption(entity: dict, records: list[dict]) -> str:
    shots = [r.get("shot", "reference") for r in records]
    phrases = [SHOT_PHRASES.get(s, s) for s in dict.fromkeys(shots)]
    desc = entity.get("descriptor") or ""
    who = entity["name"] + (" (%s)" % desc if desc else "")
    if len(phrases) == 1:
        return "%s, shown as %s" % (who, phrases[0])
    return "%s, shown as %s" % (who, ", ".join(phrases[:-1]) + " and " + phrases[-1])
