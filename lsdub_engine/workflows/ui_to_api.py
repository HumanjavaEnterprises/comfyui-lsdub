#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Convert ComfyUI UI-format workflow JSON to API-format JSON.

UI format:  {"nodes": [...], "links": [...]}
API format: {"node_id": {"class_type": "X", "inputs": {field: value, ...}}}

The API format is what ComfyUI's /prompt endpoint accepts.
"""
import json
import sys


def ui_to_api(ui_graph: dict, node_info: dict) -> dict:
    """Convert a UI-format workflow graph to API format.

    Args:
        ui_graph: UI-format graph with 'nodes' and 'links'.
        node_info: ComfyUI /object_info response, mapping class_type -> {input, input_order, ...}

    Returns:
        API-format dict: {node_id_str: {"class_type": ..., "inputs": {...}}}
    """
    link_map = {}
    for link in ui_graph.get("links", []):
        link_id, src_node, src_output = link[0], link[1], link[2]
        # ComfyUI expects link source node IDs as strings
        link_map[link_id] = (str(src_node), src_output)

    api = {}
    for node in ui_graph.get("nodes", []):
        node_id = str(node["id"])
        node_type = node["type"]
        widgets_values = node.get("widgets_values", [])
        ui_inputs = node.get("inputs", [])

        info = node_info.get(node_type, {})
        field_names = info.get("input_order", {}).get("required", [])
        field_types = {
            f: v[0]
            for f, v in info.get("input", {}).get("required", {}).items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str)
        }

        # Map socket TYPE (not name) -> link id for linked inputs
        link_by_type = {}
        for inp in ui_inputs:
            socket_type = inp.get("type")
            if inp.get("link") is not None:
                link_by_type[socket_type] = inp["link"]

        inputs = {}
        actual_values = widgets_values[1:] if len(widgets_values) > 0 else []
        widget_idx = 0

        for field in field_names:
            field_type = field_types.get(field)
            matched = False
            for socket_type, link_id in link_by_type.items():
                if field_type == socket_type:
                    src_node, src_output = link_map[link_id]
                    inputs[field] = [src_node, src_output]
                    matched = True
                    break
            if not matched:
                if widget_idx < len(actual_values):
                    inputs[field] = actual_values[widget_idx]
                else:
                    inputs[field] = None
                widget_idx += 1

        api[node_id] = {"class_type": node_type, "inputs": inputs}

    return api


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ui_to_api.py <ui_workflow.json> <output.json>")
        sys.exit(1)
    ui_path = sys.argv[1]
    out_path = sys.argv[2]
    node_info_path = sys.argv[3] if len(sys.argv) > 3 else None

    if node_info_path:
        node_info = json.load(open(node_info_path))
    else:
        import requests
        r = requests.get("http://localhost:8188/object_info", timeout=120)
        node_info = r.json()

    ui_graph = json.load(open(ui_path))
    api = ui_to_api(ui_graph, node_info)
    json.dump(api, open(out_path, "w"), indent=2)
    print(f"Converted {ui_path} -> {out_path}")
    print(f"  Nodes: {len(api)}")
    print(f"  Node types: {sorted(set(v['class_type'] for v in api.values()))}")
