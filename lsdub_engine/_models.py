# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Where model files live. Nothing is downloaded at import time; `lsdub fetch-models` fetches them."""

import hashlib
import os
import urllib.request

CACHE = os.environ.get("LIPS_CACHE", os.path.expanduser("~/.cache/lsdub"))
FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)  # MediaPipe Face Landmarker
FACE_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)  # YuNet face detector, OpenCV Zoo
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


# repo -> the revision LSDub is pinned to and the sha256 of the files it loads. comfy-build.yaml agrees (pack test).
KNOWN = {
    "facebook/dinov2-base": {
        "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "files": {
            "config.json": "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
            "model.safetensors": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
        },
    },
    "Systran/faster-whisper-large-v3": {
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        "files": {
            "config.json": "a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9",
            "model.bin": "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1",
            "preprocessor_config.json": "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
            "tokenizer.json": "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca",
            "vocabulary.json": "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1",
        },
    },
    "florence-community/Florence-2-large": {
        "revision": "4271c66b88cdbc05735372ec13b2360108de5317",
        "files": {"model.safetensors": "7715423d6549bf1e71188bdd84f4ac960cc0597886af24a5ef7b66f128660685"},
    },
    "facebook/sam2.1-hiera-base-plus": {
        "revision": "b7320756a13354e7530a63935656d35b2f91a290",
        "files": {"sam2.1_hiera_base_plus.pt": "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"},
    },
}
# a file made here, not downloaded: `lsdub cast export-onnx` from this repo at its pinned revision, its hash in a sidecar (exports
# differ by torch version, so no table can name the hash)
CAST_ONNX = {"source": "facebook/dinov2-base", "sidecar": ".sha256"}


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
    except OSError as e:  # no network, a proxy, a 404: say what to do by hand
        if os.path.exists(dst + ".part"):
            os.remove(dst + ".part")
        raise RuntimeError(f"could not download {os.path.basename(dst)} ({e}). Download it from {url} and place it at {dst}") from e
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_file(path, sha):
    """path, or RuntimeError when its bytes are not sha."""
    got = sha256_file(path)
    if got != sha:
        raise RuntimeError(f"{os.path.basename(path)} checksum mismatch (got {got}); refusing to use it")
    return path


def stamp_export(path):
    """after an export: record its hash beside it."""
    with open(path + CAST_ONNX["sidecar"], "w", encoding="utf-8") as f:
        f.write(sha256_file(path))


def check_export(path):
    """an export with a recorded hash matches it; an older export without one loads as it is."""
    try:
        with open(path + CAST_ONNX["sidecar"], encoding="utf-8") as f:
            return check_file(path, f.read().strip())
    except FileNotFoundError:
        return path


def verify_snapshot(path, repo_id):
    """the files of the pinned snapshot match KNOWN, once: a .verified stamp in the folder skips the next read."""
    k = KNOWN.get(repo_id)
    stamp = os.path.join(path, ".verified")
    if not k or os.path.basename(path) != k["revision"] or os.path.exists(stamp):
        return path
    for name, sha in k["files"].items():
        if os.path.isfile(os.path.join(path, name)):
            check_file(os.path.join(path, name), sha)
    try:
        with open(stamp, "w", encoding="utf-8") as f:
            f.write(k["revision"])
    except OSError:  # a read-only model folder: it hashes again next time
        pass
    return path


def hf_snapshot(cache_dir, repo_id, need=("config.json",), revision=None):
    """the local folder of a Hugging Face model under cache_dir, so it loads by PATH (no hub call, works offline):
    the cache layout's snapshot named by refs/main, else any snapshot holding every file in `need` (newest first; a
    build that places only snapshots/<rev>/ files has no refs/main), else a flat local_dir layout <cache_dir>/<name>/. revision (default: the pinned one in KNOWN) is
    tried first, and a pinned snapshot is checked against KNOWN.
    None when the files are not there (the caller then loads by repo id, which may download)."""
    if not cache_dir:
        return None
    root = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))
    snaps, cands = os.path.join(root, "snapshots"), []
    revision = revision or KNOWN.get(repo_id, {}).get("revision")
    if revision:
        cands.append(os.path.join(snaps, revision))
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
        if os.path.isdir(c) and all(os.path.isfile(os.path.join(c, n)) for n in need):
            return verify_snapshot(c, repo_id)
    return None
