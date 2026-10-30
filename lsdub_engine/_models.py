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
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")   # YuNet face detector, OpenCV Zoo, MIT
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def set_cache(path):
    """where the small face models live (the ComfyUI pack points this at models/lsdub). Config only: no download."""
    global CACHE
    CACHE = path


def yunet_path():
    return os.environ.get("LIPS_YUNET", os.path.join(CACHE, "face_detection_yunet_2023mar.onnx"))


def _fetch(url, dst, sha):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dst + ".part")  # noqa: S310 (fixed https URL, checksum below)
    except OSError as e:                                   # no network, a proxy, a 404: say what to do by hand
        if os.path.exists(dst + ".part"): os.remove(dst + ".part")
        raise RuntimeError(f"could not download {os.path.basename(dst)} ({e}). Download it from {url} and place it at "
                           f"{dst}") from e
    with open(dst + ".part", "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != sha:
        os.remove(dst + ".part")
        raise RuntimeError(f"{os.path.basename(dst)} checksum mismatch (got {got}); refusing to use it")
    os.replace(dst + ".part", dst)
    return dst


def fetch_yunet(force=False):
    dst = yunet_path()
    return dst if os.path.exists(dst) and not force else _fetch(YUNET_URL, dst, YUNET_SHA256)


def face_landmarker_path():
    return os.environ.get("LIPS_FACE_MODEL", os.path.join(CACHE, "face_landmarker.task"))


def fetch_face_landmarker(force=False):
    dst = face_landmarker_path()
    return dst if os.path.exists(dst) and not force else _fetch(FACE_URL, dst, FACE_SHA256)


def hf_snapshot(cache_dir, repo_id, need=("config.json",)):
    """the local folder of a Hugging Face model under cache_dir, so it loads by PATH (no hub call, works offline):
    the cache layout's snapshot named by refs/main, else any snapshot holding every file in `need` (newest first; a
    build that places only snapshots/<rev>/ files has no refs/main), else a flat local_dir layout <cache_dir>/<name>/.
    None when the files are not there (the caller then loads by repo id, which may download)."""
    if not cache_dir: return None
    root = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))
    snaps, cands = os.path.join(root, "snapshots"), []
    try:
        with open(os.path.join(root, "refs", "main"), encoding="utf-8") as f:
            cands.append(os.path.join(snaps, f.read().strip()))
    except OSError:
        pass
    try:
        cands += sorted((os.path.join(snaps, d) for d in os.listdir(snaps)), key=os.path.getmtime, reverse=True)
    except OSError:
        pass
    cands.append(os.path.join(cache_dir, repo_id.split("/")[-1]))
    for c in cands:
        if os.path.isdir(c) and all(os.path.isfile(os.path.join(c, n)) for n in need): return c
    return None
