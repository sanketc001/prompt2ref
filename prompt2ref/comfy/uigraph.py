"""API-format graph -> a workflow .json you can open in ComfyUI.

The video path edits a shipped UI template, so it never needs this. The image
backends build their graphs from scratch in API format (that is what `/prompt`
takes), and a graph you cannot open and inspect is a bad demo - so this turns
one back into the UI format, which means knowing each node class's declared
input order, which inputs are widgets and which are sockets, and its outputs.

Three sources for that, in order: node classes importable in this process (the
P2R nodes themselves), a running ComfyUI's `/object_info`, and the snapshot in
`schemas.json` next to this file. The snapshot keeps `--comfy` working with no
server up; the live lookup keeps it working for nodes added since.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

SCHEMA_SNAPSHOT = Path(__file__).with_name("schemas.json")

# The only input types ComfyUI draws as a widget. Everything else - IMAGE,
# MODEL, and every custom type a node pack invents - is a socket, so testing
# for these beats trying to enumerate the sockets.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# Nodes that serialise their widgets as a {name: value} dict instead of the
# positional list everything else uses.
DICT_WIDGET_NODES = {"VHS_VideoCombine"}

COL_W, ROW_H = 400, 230


class SchemaSource:
    """Resolves class_type -> {inputs:[[name,type,is_widget,default]], outputs:[...]}."""

    def __init__(self, server: str | None = None, extra: dict | None = None):
        self.server = server
        self.cache: dict[str, dict] = dict(extra or {})
        try:
            self.snapshot = json.loads(SCHEMA_SNAPSHOT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.snapshot = {}

    def get(self, class_type: str) -> dict:
        if class_type in self.cache:
            return self.cache[class_type]
        for loader in (self._from_local, self._from_server, self._from_snapshot):
            schema = loader(class_type)
            if schema:
                self.cache[class_type] = schema
                return schema
        raise KeyError(
            "no schema for %r - start ComfyUI (or pass its address) so the node's "
            "input order can be read from /object_info" % class_type)

    # -- sources ------------------------------------------------------------
    def _from_local(self, class_type: str) -> dict | None:
        if not class_type.startswith("P2R"):
            return None
        try:
            import comfyui_prompt2ref as pkg

            cls = pkg.NODE_CLASS_MAPPINGS[class_type]
        except Exception:
            return None
        return from_node_class(cls)

    def _from_server(self, class_type: str) -> dict | None:
        if not self.server:
            return None
        try:
            with urllib.request.urlopen(
                "%s/object_info/%s" % (self.server.rstrip("/"), class_type), timeout=20
            ) as r:
                info = json.load(r)[class_type]
        except Exception:
            return None
        return from_object_info(info)

    def _from_snapshot(self, class_type: str) -> dict | None:
        return self.snapshot.get(class_type)


def _field(name, type_spec, opts) -> list:
    if isinstance(type_spec, list):
        type_str, options = "COMBO", type_spec
    elif type_spec == "COMBO":
        type_str, options = "COMBO", opts.get("options", [])
    else:
        type_str, options = type_spec, None
    default = opts["default"] if "default" in opts else (options[0] if options else "")
    return [name, type_str, type_str in WIDGET_TYPES, default]


def from_object_info(info: dict) -> dict:
    fields = []
    for section in ("required", "optional"):
        for name, spec in info["input"].get(section, {}).items():
            fields.append(_field(name, spec[0], spec[1] if len(spec) > 1 else {}))
    return {"inputs": fields, "outputs": list(info.get("output", [])),
            "output_names": list(info.get("output_name", info.get("output", [])))}


def from_node_class(cls) -> dict:
    """Same shape, read straight off a ComfyUI node class in this process."""
    spec = cls.INPUT_TYPES()
    fields = []
    for section in ("required", "optional"):
        for name, entry in (spec.get(section) or {}).items():
            fields.append(_field(name, entry[0], entry[1] if len(entry) > 1 else {}))
    outputs = list(getattr(cls, "RETURN_TYPES", ()))
    return {"inputs": fields, "outputs": outputs,
            "output_names": list(getattr(cls, "RETURN_NAMES", outputs))}


def _layers(graph: dict) -> dict[str, int]:
    """Longest-path depth per node, so the drawing reads left to right."""
    depth: dict[str, int] = {}

    def walk(key: str, seen: frozenset) -> int:
        if key in depth:
            return depth[key]
        if key in seen:            # cycles don't happen here, but don't hang on one
            return 0
        d = 0
        for value in graph[key]["inputs"].values():
            for link in _links_in(value):
                if link[0] in graph:
                    d = max(d, walk(link[0], seen | {key}) + 1)
        depth[key] = d
        return d

    for key in graph:
        walk(key, frozenset())
    return depth


def _links_in(value):
    """Every [node, slot] link inside an input value (autogrow inputs are dicts)."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return [value]
    if isinstance(value, dict):
        return [v for v in value.values()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
    return []


def api_to_ui(
    graph: dict,
    title: str = "prompt2ref",
    server: str | None = None,
    schemas: SchemaSource | None = None,
    extra: dict | None = None,
    titles: dict | None = None,
) -> dict:
    """Convert an API-format graph into a UI workflow.

    Node keys become titles, so keep them readable; `titles` overrides that for
    the nodes where a sentence beats an identifier.
    """
    titles = titles or {}
    src = schemas or SchemaSource(server=server)
    depth = _layers(graph)
    keys = sorted(graph, key=lambda k: (depth[k], k))

    node_id = {k: 100 + i for i, k in enumerate(keys)}
    row = {}
    ui_nodes, links = [], []
    next_link = [1]

    def new_link(src_key, src_slot, dst_id, dst_slot, type_str):
        lid = next_link[0]
        next_link[0] += 1
        links.append([lid, node_id[src_key], src_slot, dst_id, dst_slot, type_str])
        return lid

    for order, key in enumerate(keys):
        spec = graph[key]
        class_type = spec["class_type"]
        raw = spec.get("inputs", {})
        schema = src.get(class_type)

        ui_inputs, widgets = [], []
        dict_widgets = {} if class_type in DICT_WIDGET_NODES else None

        for name, type_str, is_widget, default in schema["inputs"]:
            value = raw.get(name)

            # Autogrow inputs (e.g. MiniMaxH3ReferenceToVideo.ref_images) arrive as
            # a {slot_name: link} dict and are addressed by their flattened id.
            if isinstance(value, dict) and _links_in(value):
                for slot_name, link in value.items():
                    lid = new_link(link[0], link[1], node_id[key], len(ui_inputs), "IMAGE")
                    ui_inputs.append({"name": "%s.%s" % (name, slot_name),
                                      "type": "IMAGE", "link": lid})
                continue

            is_link = isinstance(value, list) and len(value) == 2 and value[0] in graph
            if dict_widgets is not None and is_widget and not is_link:
                dict_widgets[name] = default if value is None else value
                continue

            if is_link:
                lid = new_link(value[0], value[1], node_id[key], len(ui_inputs), type_str)
                ui_inputs.append({"name": name, "type": type_str, "link": lid})
                # A widget converted to an input still holds a slot in widgets_values.
                if is_widget:
                    widgets.append("" if type_str == "STRING" else default)
            elif is_widget:
                widgets.append(default if value is None else value)
            elif value is not None:
                raise ValueError("%s.%s: socket-only input got the literal %r"
                                 % (key, name, value))

        # VHS_VideoCombine and friends grow widgets the schema does not declare
        # (pix_fmt only exists once a format is chosen). Carry those through
        # rather than silently dropping the value the graph asked for.
        if dict_widgets is not None:
            declared = {f[0] for f in schema["inputs"]}
            for name, value in raw.items():
                if name not in declared and not _links_in(value):
                    dict_widgets[name] = value

        ui_outputs = [{"name": n, "type": t, "links": None}
                      for n, t in zip(schema["output_names"], schema["outputs"])]

        col = depth[key]
        row[col] = row.get(col, -1) + 1
        ui_nodes.append({
            "id": node_id[key], "type": class_type,
            "pos": [col * COL_W, row[col] * ROW_H],
            "size": [340, 120], "flags": {}, "order": order, "mode": 0,
            "inputs": ui_inputs, "outputs": ui_outputs, "title": titles.get(key, key),
            "properties": {"Node name for S&R": class_type},
            "widgets_values": dict_widgets if dict_widgets is not None else widgets,
        })

    by_id = {n["id"]: n for n in ui_nodes}
    for lid, src_id, src_slot, _dst_id, _dst_slot, _t in links:
        outs = by_id[src_id]["outputs"]
        if src_slot < len(outs):
            outs[src_slot]["links"] = (outs[src_slot]["links"] or []) + [lid]

    return {
        "id": title,
        "revision": 0,
        "last_node_id": max(node_id.values()) if node_id else 0,
        "last_link_id": next_link[0] - 1,
        "nodes": ui_nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": dict(extra or {}),
        "version": 0.4,
    }


def save_ui_workflow(graph: dict, path: str | Path, title: str = "prompt2ref",
                     server: str | None = None, extra: dict | None = None,
                     titles: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(api_to_ui(graph, title, server=server, extra=extra, titles=titles),
                   indent=1),
        encoding="utf-8")
    return p
