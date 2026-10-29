# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""comfy_backend.py — one ComfyClient interface, selectable backend.

LSDub runs the SAME static workflow JSON three ways; only the execution target differs:

  backend="local"  stdlib HTTP straight to ComfyUI's own /prompt API (via /upload/image for files).
  backend="cloud"  Comfy Python SDK (comfy-sdk) -> https://cloud.comfy.org (needs COMFY_API_KEY).
  backend="direct" alias for local (kept for backward compat).

Select with --backend / LSDUB_BACKEND.
"""
import json
import os
import re

CLOUD_URL = "https://cloud.comfy.org"
WORKFLOWS_API = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "workflows", "api"))


def resolve_backend(name=None):
    return (name or os.environ.get("LSDUB_BACKEND") or "local").lower()


def _substitute_str(s, subs):
    """Substitute {KEY} / {key} placeholders in a string."""
    def repl(m):
        key = m.group(1)
        for k, v in subs.items():
            if k == key or k.lower() == key:
                return str(v)
        return m.group(0)
    return re.sub(r'\{(\w+)\}', repl, s)


def _substitute(obj, subs):
    """Recursively apply {KEY} substitutions in string values of an API-format graph."""
    if isinstance(obj, dict):
        return {_substitute(k, subs): _substitute(v, subs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(item, subs) for item in obj]
    if isinstance(obj, str):
        return _substitute_str(obj, subs)
    return obj


def _resolve_wf_path(path):
    """Resolve a workflow path: absolute, or relative to workflows/api/."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(WORKFLOWS_API, path))


def _local_run(url, graph, out_node_id, out_dir, upload_paths=None):
    """Run a pre-substituted API-format graph against a local ComfyUI via HTTP.

    upload_paths: optional list of local file paths to upload via /upload/image.
    The graph must contain {UPLOAD_N} placeholders (N = 0-indexed) for the
    server-side file names to be substituted.
    """
    from . import comfy as cf

    # Upload files and replace {UPLOAD_N} placeholders
    for i, up in enumerate(upload_paths or []):
        server_name = cf.upload(url, up)
        graph = _substitute(graph, {f"UPLOAD_{i}": server_name,
                                     f"upload_{i}": server_name})

    pid = cf.queue(url, graph, client_id="lsdub")
    entry = cf.wait(url, pid, timeout=1800)

    rows = cf.files(entry)
    paths = []
    for row in rows:
        dest = os.path.join(out_dir, row["filename"])
        d = os.path.dirname(dest)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(cf.fetch(url, row))
        paths.append(dest)
    return paths


def make_client(backend=None):
    """Return (run_fn, label).

    run_fn(path, out_node_id, out_dir='.', upload_paths=None, **subs) -> list[local output paths]

    path:          path to the static API-format workflow JSON (absolute, or
                   relative to workflows/api/ inside the LSDub package)
    out_node_id:   ComfyUI node id whose output we want (e.g. '60' for SaveImage)
    out_dir:       directory to write output files to
    upload_paths:  optional list of local file paths to upload to ComfyUI
                   before running (for local backend; for cloud these become
                   Asset handles automatically)
    **subs:        dict of placeholder substitutions applied to the template
                   e.g. {'VISEME': 'aa', 'SEED': 42, 'PROMPT': 'close-up...'}

    For local backend: upload_paths files are uploaded via /upload/image and their
    server-side names are substituted into {UPLOAD_N} placeholders.
    """
    backend = resolve_backend(backend)

    if backend in ("direct", "local"):
        label = f"local (http -> ComfyUI {os.environ.get('COMFY_URL', '127.0.0.1:8188')})"

        def run(path, out_node_id, out_dir=".", upload_paths=None, **subs):
            url = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
            wf_path = _resolve_wf_path(path)
            with open(wf_path) as f:
                graph = json.load(f)
            graph = _substitute(graph, subs)
            return _local_run(url, graph, out_node_id, out_dir, upload_paths)

        return run, label

    # cloud
    if backend == "cloud":
        key = os.environ.get("COMFY_API_KEY")
        if not key:
            raise RuntimeError("backend=cloud needs COMFY_API_KEY")
        base = os.environ.get("COMFY_BASE_URL", CLOUD_URL)
        label = f"cloud ({base})"

        def run(path, out_node_id, out_dir=".", upload_paths=None, **subs):
            from comfy_sdk import Comfy

            import inspect
            kw = {'base_url': base} if 'base_url' in inspect.signature(Comfy).parameters else {}
            c = Comfy(api_key=key, **kw)   # never writes os.environ: this runs inside the ComfyUI process
            try:
                wf_path = _resolve_wf_path(path)
                with open(wf_path) as f:
                    graph = json.load(f)
                graph = _substitute(graph, subs)
                wf = c.workflows.from_json(graph)

                # Upload files as Asset handles and set them in the graph
                for i, up in enumerate(upload_paths or []):
                    asset = c.assets.from_file(up)
                    # Find the first node with an empty 'image' or 'video' field
                    # and set it to this asset
                    for node_id, node in wf.json.items():
                        inputs = node.get("inputs", {})
                        if any(v == "" or v == None for k, v in inputs.items()):
                            for field_name, v in inputs.items():
                                if v == "" or v is None:
                                    wf.set_input(node_id, field_name, asset)
                                    break
                            break

                job = c.run(wf)
                paths = []
                for o in job.get_outputs(out_node_id):
                    dest = os.path.join(out_dir, o.name)
                    d = os.path.dirname(dest)
                    if d:
                        os.makedirs(d, exist_ok=True)
                    o.to_file(dest)
                    paths.append(dest)
                return paths
            finally:
                c.close()

        return run, label
