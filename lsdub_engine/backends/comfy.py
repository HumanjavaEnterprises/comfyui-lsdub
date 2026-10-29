# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""comfy.py — minimal ComfyUI HTTP client (stdlib only).

  base_url()        $COMFY_URL, else http://127.0.0.1:8188
  ready(url)        GET /system_stats answers
  upload(url, p)    POST /upload/image -> the server-side name
  queue(url, g)     POST /prompt -> prompt_id
  wait(url, pid)    poll /history until done -> the history entry; raises on error / timeout
  files(entry)      every output file row of a history entry
  fetch(url, row)   GET /view -> bytes
"""
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import uuid


def base_url():
    return os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


def _call(url, path, data=None, ctype="application/json", timeout=60):
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": ctype} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ready(url, timeout=5):
    try:
        _call(url, "/system_stats", timeout=timeout)
        return True
    except Exception:                                                  # noqa: BLE001
        return False


def free(url, timeout=10):
    try:
        _call(url, "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
              timeout=timeout)
        return True
    except Exception:                                                  # noqa: BLE001
        return False


def upload(url, path):
    b = uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        payload = fh.read()
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + payload + \
        (f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
         f"--{b}--\r\n").encode()
    return json.loads(_call(url, "/upload/image", body, f"multipart/form-data; boundary={b}",
                            timeout=120))["name"]


def queue(url, graph, client_id="lsdub"):
    body = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    return json.loads(_call(url, "/prompt", body, timeout=60))["prompt_id"]


def wait(url, pid, timeout=900, poll=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.loads(_call(url, f"/history/{pid}", timeout=30))
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                errs = [m[1] for m in st.get("messages", []) if m and m[0] == "execution_error"]
                raise RuntimeError("; ".join(
                    f"{e.get('node_type')} {e.get('exception_type')}: "
                    f"{str(e.get('exception_message'))[:400]}" for e in errs) or json.dumps(st)[:800])
            return h[pid]
        time.sleep(poll)
    raise TimeoutError(f"ComfyUI prompt {pid} not done after {timeout}s")


def files(entry):
    return [row for node in (entry.get("outputs") or {}).values() for rows in node.values()
            if isinstance(rows, list) for row in rows if isinstance(row, dict) and "filename" in row]


def fetch(url, row):
    q = urllib.parse.urlencode({"filename": row["filename"], "subfolder": row.get("subfolder", ""),
                                "type": row.get("type", "output")})
    return _call(url, f"/view?{q}", timeout=120)
