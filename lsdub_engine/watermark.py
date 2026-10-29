# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""watermark.py — default-on invisible watermark for lsdub output, plus a verifier.

Method: additive spread-spectrum on luma. A +/-1 pseudo-random pattern on a fixed 240x135 cell grid (seeded by a public
key) is stretched to the frame size, so the mark survives rescaling; its amplitude follows local texture, so flat skin
gets less. Detection high-passes the luma and correlates it with the keyed pattern; the null distribution comes from
the same correlation against decoy keys, so the result is a z-score (present when the mean z over the sampled frames
clears --threshold). It is a disclosure marker, not DRM: anyone who knows the key can verify it, and a determined
adversary can remove it.

The mp4 also carries a metadata tag (comment = TAG) on every file we write.
"""
import argparse
from .io import ff
import hashlib
import json
import sys

import numpy as np
import cv2

KEY = "humanjava-lips-v1"
TAG = "synthetic lip-sync: lsdub (Humanjava Enterprises); watermark key " + KEY
GRID = (135, 240)  # rows, cols of pattern cells (16:9)


def _seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def pattern(key, h, w):
    rng = np.random.default_rng(_seed(key))
    p = rng.choice(np.array([-1.0, 1.0], np.float32), size=GRID)
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST)


def _hp(y):
    return y - cv2.GaussianBlur(y, (0, 0), 3)


def embed_frame(bgr, pat, strength=3.0):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[..., 0]
    tex = cv2.GaussianBlur(np.abs(_hp(y)), (0, 0), 4)
    amp = strength * np.clip(0.5 + tex / 8.0, 0.5, 1.5)
    ycc[..., 0] = np.clip(y + amp * pat, 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def _corr(hy, pat):
    hp = _hp(pat)
    return float((hy * hp).sum() / (np.linalg.norm(hy) * np.linalg.norm(hp) + 1e-9))


def detect_frame(bgr, key=KEY, decoys=24):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    h, w = y.shape
    hy = _hp(y)
    c = _corr(hy, pattern(key, h, w))
    null = np.array([_corr(hy, pattern(f"{key}#decoy{i}", h, w)) for i in range(decoys)])
    return (c - null.mean()) / (null.std() + 1e-9)


def _probe(path):
    out = ff.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(n) / float(d)


def apply(src, dst, key=KEY, strength=3.0, crf=14):
    """Watermark every frame of src -> dst (audio copied, metadata tag added)."""
    w, h, fps = _probe(src)
    pat = pattern(key, h, w)
    cap = cv2.VideoCapture(src)
    p = ff.popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-i", src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-c:v", "libx264", "-crf", str(crf),
                          "-pix_fmt", "yuv420p", "-metadata", f"comment={TAG}", "-shortest", dst], stdin=ff.PIPE)
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        p.stdin.write(embed_frame(fr, pat, strength).tobytes())
        n += 1
    cap.release()
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed writing {dst}")
    return n


def apply_inplace(path, **kw):
    import os
    tmp = path + ".wm.mp4"
    apply(path, tmp, **kw)
    os.replace(tmp, path)


def verify(path, key=KEY, frames=24, threshold=4.0):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or frames
    idx = set(np.linspace(0, max(0, total - 1), min(frames, total)).astype(int).tolist())
    zs, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idx:
            zs.append(detect_frame(fr, key))
        i += 1
    cap.release()
    tag = ff.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True).stdout.strip()
    z = float(np.mean(zs)) if zs else 0.0
    return {"file": path, "frames_checked": len(zs), "z_mean": round(z, 2), "z_min": round(float(np.min(zs)), 2) if zs else None,
            "watermark": z >= threshold, "metadata_tag": tag or None}


NO_WM_WARNING = ("WARNING: writing WITHOUT the invisible watermark. You are responsible for disclosing that this video's lip "
                 "movement is synthetic, and for having consent from anyone whose likeness appears. See RESPONSIBLE_USE.md.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lsdub watermark")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a_ = sp.add_parser("apply")
    a_.add_argument("src")
    a_.add_argument("dst")
    a_.add_argument("--strength", type=float, default=3.0)
    v_ = sp.add_parser("verify")
    v_.add_argument("video")
    v_.add_argument("--threshold", type=float, default=4.0)
    v_.add_argument("--frames", type=int, default=24)
    a = ap.parse_args(argv)
    if a.cmd == "apply":
        print(f"{apply(a.src, a.dst, strength=a.strength)} frames -> {a.dst}")
    else:
        r = verify(a.video, frames=a.frames, threshold=a.threshold)
        print(json.dumps(r))
        sys.exit(0 if r["watermark"] else 1)


if __name__ == "__main__":
    main()
