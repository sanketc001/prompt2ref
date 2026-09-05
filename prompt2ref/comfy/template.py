"""Read/write surface for the MiniMax H3 Ref2VA ComfyUI workflow.

The template ships in pure text-to-video mode: every `<Picture N>`, `<Video N>`
and `<Audio N>` group is bypassed (mode 4). Filling a picture slot therefore
means two things - point its LoadImage at a file *and* un-bypass the group.

Nodes are located by group title / node title rather than by id, so the module
survives the template being re-saved from ComfyUI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

BYPASS = 4
ACTIVE = 0

PICTURE_RE = re.compile(r"^<Picture (\d+)>$")

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "minimax_h3_ref2va.json"


@dataclass
class PictureSlot:
    number: int              # 1-based, matches the <Picture N> tag in the prompt
    group_title: str
    load_image_id: int
    member_ids: list[int]    # every node in the group, toggled together

    @property
    def tag(self) -> str:
        return "<Picture %d>" % self.number


def _inside(pos, bounding) -> bool:
    x, y = pos[0], pos[1]
    bx, by, bw, bh = bounding
    return bx <= x <= bx + bw and by <= y <= by + bh


class Ref2VATemplate:
    """Thin, explicit editor over the workflow JSON."""

    def __init__(self, path: str | Path = DEFAULT_TEMPLATE):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self._by_id = {n["id"]: n for n in self.data["nodes"]}
        self.pictures = self._find_picture_slots()

    # -- discovery ----------------------------------------------------------
    def _group_members(self, group: dict) -> list[dict]:
        return [n for n in self.data["nodes"] if _inside(n.get("pos", [0, 0]), group["bounding"])]

    def _find_picture_slots(self) -> list[PictureSlot]:
        slots: list[PictureSlot] = []
        for g in self.data.get("groups", []):
            m = PICTURE_RE.match(str(g.get("title", "")))
            if not m:
                continue
            members = self._group_members(g)
            loaders = [n for n in members if n["type"] == "LoadImage"]
            if not loaders:
                continue
            slots.append(PictureSlot(
                number=int(m.group(1)),
                group_title=g["title"],
                load_image_id=loaders[0]["id"],
                member_ids=[n["id"] for n in members],
            ))
        slots.sort(key=lambda s: s.number)
        if not slots:
            raise ValueError("no <Picture N> groups found in " + str(self.path))
        return slots

    def _node_by_title(self, title: str, node_type: str | None = None) -> dict:
        for n in self.data["nodes"]:
            if n.get("title") == title and (node_type is None or n["type"] == node_type):
                return n
        raise KeyError("node titled %r not found" % title)

    def _first_of_type(self, node_type: str) -> dict:
        for n in self.data["nodes"]:
            if n["type"] == node_type:
                return n
        raise KeyError("no node of type %r" % node_type)

    def group_titled(self, title: str) -> dict | None:
        for g in self.data.get("groups", []):
            if g.get("title") == title:
                return g
        return None

    # -- picture slots ------------------------------------------------------
    def set_picture(self, number: int, filename: str) -> None:
        """Point slot `number` at `filename` (relative to ComfyUI/input) and enable it."""
        slot = next((s for s in self.pictures if s.number == number), None)
        if slot is None:
            raise IndexError("template has no <Picture %d>" % number)
        loader = self._by_id[slot.load_image_id]
        loader["widgets_values"][0] = filename
        for nid in slot.member_ids:
            self._by_id[nid]["mode"] = ACTIVE

    def clear_picture(self, number: int) -> None:
        slot = next((s for s in self.pictures if s.number == number), None)
        if slot is None:
            return
        for nid in slot.member_ids:
            self._by_id[nid]["mode"] = BYPASS

    def clear_all_pictures(self) -> None:
        for s in self.pictures:
            self.clear_picture(s.number)

    # -- prompt -------------------------------------------------------------
    def set_prompt(self, text: str) -> None:
        node = self._node_by_title("BASE PROMPT", "PrimitiveStringMultiline")
        node["widgets_values"][0] = text

    def get_prompt(self) -> str:
        return self._node_by_title("BASE PROMPT", "PrimitiveStringMultiline")["widgets_values"][0]

    # -- video settings -----------------------------------------------------
    def set_duration_seconds(self, seconds: float) -> None:
        node = self._node_by_title("VIDEO DURATION (seconds)", "PrimitiveFloat")
        node["widgets_values"][0] = float(seconds)

    @staticmethod
    def frames_for(seconds: float, fps: int = 24) -> int:
        """Mirror of the template's ComfyMathExpression node.

        `max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17`
        - the sampler needs a frame count congruent to 5 (mod 17).
        """
        base = max(5, round(seconds * fps))
        return base + (5 - (base % 17)) % 17

    def set_frames(self, frames: int) -> None:
        """Advanced override: pin VIDEO_LENGTH_F instead of deriving it from seconds.

        Replaces the frame-count expression with a literal, so the duration
        widget stops driving the length. Must still be congruent to 5 (mod 17).
        """
        for n in self.data["nodes"]:
            if n["type"] == "ComfyMathExpression" and "% 17" in str(n.get("widgets_values", [""])[0]):
                n["widgets_values"][0] = str(int(frames))
                return
        raise KeyError("frame-count expression node not found")

    def set_resolution(self, width: int, height: int) -> None:
        """Set the manual resolution on the VIDEO RESOLUTION ResolutionMaster node."""
        node = self._node_by_title("VIDEO RESOLUTION", "ResolutionMaster")
        node["widgets_values"][2] = int(width)
        node["widgets_values"][3] = int(height)

    def get_resolution(self) -> tuple[int, int]:
        w = self._node_by_title("VIDEO RESOLUTION", "ResolutionMaster")["widgets_values"]
        return int(w[2]), int(w[3])

    # The "Half resolution" group holds one ImageScaleBy(area, 0.5) between the
    # resolution node and the latent, so the rendered video is half of whatever
    # ResolutionMaster says whenever that group is active.
    def _half_res_node(self) -> dict | None:
        g = self.group_titled("Half resolution")
        if not g:
            return None
        for n in self._group_members(g):
            if n["type"] == "ImageScaleBy":
                return n
        return None

    @property
    def half_res_active(self) -> bool:
        n = self._half_res_node()
        return bool(n) and n.get("mode", ACTIVE) == ACTIVE

    @property
    def half_res_factor(self) -> float:
        n = self._half_res_node()
        if not n or n.get("mode", ACTIVE) != ACTIVE:
            return 1.0
        try:
            return float(n["widgets_values"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            return 0.5

    def set_half_res(self, enabled: bool) -> None:
        n = self._half_res_node()
        if n is None:
            return
        # Bypassing ImageScaleBy makes it a pass-through, which is exactly the
        # "full resolution" behaviour; the rest of the group is inert.
        n["mode"] = ACTIVE if enabled else BYPASS

    # -- other reference kinds ----------------------------------------------
    def disable_group(self, title: str) -> None:
        g = self.group_titled(title)
        if not g:
            return
        for n in self._group_members(g):
            n["mode"] = BYPASS

    def disable_av_references(self) -> None:
        for i in (1, 2, 3):
            self.disable_group("<Video %d>" % i)
            self.disable_group("<Audio %d>" % i)

    # -- output -------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        return p


def parse_aspect(aspect: str) -> float:
    a = str(aspect).strip().lower()
    named = {"landscape": "16:9", "portrait": "9:16", "square": "1:1", "vertical": "9:16"}
    a = named.get(a, a)
    if ":" in a:
        w, h = a.split(":", 1)
        return float(w) / float(h)
    if "x" in a:
        w, h = a.split("x", 1)
        return float(w) / float(h)
    return float(a)


def solve_resolution(megapixels: float, aspect: str = "16:9", multiple: int = 32) -> tuple[int, int]:
    """w x h for a target megapixel budget, each side rounded up to `multiple`.

    Rounding both sides up is what reproduces the template's own size table -
    0.2 -> 608x352, 0.3 -> 736x416, 0.4 -> 864x480, 0.5 -> 960x544, 1.5 -> 1664x928 -
    and 864x480 is also the MiniMax node's built-in default.
    """
    ratio = parse_aspect(aspect)
    target = megapixels * 1_000_000
    import math

    w = math.ceil(math.sqrt(target * ratio) / multiple) * multiple
    h = math.ceil(math.sqrt(target / ratio) / multiple) * multiple
    return max(multiple, w), max(multiple, h)
