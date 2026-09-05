"""Convert the validated API-format test graph (test_workflow.py) into a
UI-loadable ComfyUI workflow .json, using live /object_info to get each node
class's exact input/output order and types. One-off tool, not a node.
"""

import json
import urllib.request

from test_workflow import graph, TEST_PROMPT

SERVER = "http://127.0.0.1:8188"

# Types ComfyUI never renders as a widget - always a socket.
CONNECTION_ONLY = {"CLIP", "VAE", "IMAGE", "CONDITIONING", "LATENT", "AUDIO",
                    "MODEL", "MASK", "CONTROL_NET", "COMFY_AUTOGROW_V3"}


def fetch_object_info(class_types):
    info = {}
    for ct in class_types:
        with urllib.request.urlopen(f"{SERVER}/object_info/{ct}") as resp:
            info[ct] = json.load(resp)[ct]
    return info


def ordered_fields(node_info):
    """[(name, type_str, is_widget, default)] in the class's declared schema order."""
    out = []
    for section in ("required", "optional"):
        for name, spec in node_info["input"].get(section, {}).items():
            type_spec, opts = spec[0], (spec[1] if len(spec) > 1 else {})
            type_str = type_spec if isinstance(type_spec, str) else "COMBO"
            is_widget = type_str not in CONNECTION_ONLY
            if "default" in opts:
                default = opts["default"]
            elif isinstance(type_spec, list) and type_spec:
                default = type_spec[0]
            else:
                default = ""
            out.append((name, type_str, is_widget, default))
    return out


def main():
    class_types = sorted({n["class_type"] for n in graph.values()})
    info = fetch_object_info(class_types)

    keys = list(graph.keys())
    node_id = {k: 1000 + i for i, k in enumerate(keys)}

    ui_nodes = []
    links = []
    next_link_id = [1]

    def new_link(src_key, src_slot, dst_id, dst_slot, type_str):
        lid = next_link_id[0]
        next_link_id[0] += 1
        links.append([lid, node_id[src_key], src_slot, dst_id, dst_slot, type_str])
        return lid

    for col, key in enumerate(keys):
        spec = graph[key]
        class_type = spec["class_type"]
        raw_inputs = spec["inputs"]
        node_info = info[class_type]
        fields = ordered_fields(node_info)

        ui_inputs = []
        widgets_values = []

        for name, type_str, is_widget, default in fields:
            value = raw_inputs.get(name)

            if name == "ref_images" and isinstance(value, dict):
                # Autogrow: one dotted-name input entry per slot (matches the
                # exported-template convention: "ref_images.ref_image_0").
                for slot_name, link_val in value.items():
                    src_key, src_slot = link_val
                    lid = new_link(src_key, src_slot, node_id[key], len(ui_inputs), "IMAGE")
                    ui_inputs.append({"name": "ref_images." + slot_name, "type": "IMAGE", "link": lid})
                continue

            is_link = isinstance(value, list) and len(value) == 2 and value[0] in node_id
            if is_link:
                src_key, src_slot = value
                lid = new_link(src_key, src_slot, node_id[key], len(ui_inputs), type_str)
                ui_inputs.append({"name": name, "type": type_str, "link": lid})
                if is_widget:
                    widgets_values.append("" if type_str == "STRING" else default)
                continue

            if is_widget:
                widgets_values.append(value if value is not None else default)
            elif value is not None:
                # a connection-only field given a literal - shouldn't happen here
                raise ValueError(f"{key}.{name}: connection-only field got a literal {value!r}")

        ui_outputs = []
        for out_name, out_type in zip(
            node_info.get("output_name", node_info.get("output", [])),
            node_info.get("output", []),
        ):
            link_ids = [lid for (lid, s, ss, d, ds, t) in links if s == node_id[key]]
            ui_outputs.append({"name": out_name, "type": out_type, "links": None})

        ui_nodes.append({
            "id": node_id[key],
            "type": class_type,
            "pos": [col * 380, (hash(key) % 5) * 260],
            "size": [340, 200],
            "flags": {},
            "order": col,
            "mode": 0,
            "inputs": ui_inputs,
            "outputs": ui_outputs,
            "title": key,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets_values,
        })

    # second pass: fill in each output's "links" list now that all links exist
    by_node = {n["id"]: n for n in ui_nodes}
    for lid, src_id, src_slot, dst_id, dst_slot, type_str in links:
        out = by_node[src_id]["outputs"][src_slot]
        out["links"] = (out["links"] or []) + [lid]

    workflow = {
        "id": "prompt2ref-local-llm-test",
        "revision": 0,
        "last_node_id": max(node_id.values()),
        "last_link_id": next_link_id[0] - 1,
        "nodes": ui_nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"prompt2ref_test_prompt": TEST_PROMPT},
        "version": 0.4,
    }

    out_path = r"C:\Users\micke\Desktop\AI\Prompt2Ref\workflows\prompt2ref_local_llm_node.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
