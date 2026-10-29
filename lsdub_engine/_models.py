# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Where model files live. Nothing is downloaded at import time; `lsdub fetch-models` fetches them."""
import hashlib
import os
import urllib.request

CACHE = os.environ.get("LIPS_CACHE", os.path.expanduser("~/.cache/lsdub"))
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
            "face_landmarker.task")   # MediaPipe Face Landmarker, Apache-2.0
FACE_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def face_landmarker_path():
    return os.environ.get("LIPS_FACE_MODEL", os.path.join(CACHE, "face_landmarker.task"))


def fetch_face_landmarker(force=False):
    dst = face_landmarker_path()
    if os.path.exists(dst) and not force:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(FACE_URL, dst + ".part")
    with open(dst + ".part", "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != FACE_SHA256:
        os.remove(dst + ".part")
        raise RuntimeError(f"face_landmarker.task checksum mismatch (got {got}); refusing to use it")
    os.replace(dst + ".part", dst)
    return dst
