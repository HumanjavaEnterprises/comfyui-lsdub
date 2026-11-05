# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""drive_shape.py: the drive's shape over time. One stage of the drive chain, behind one paint flag, --warp-shape (off | closures | asym).

  drive (the track's per-frame `open`) -> [shape] -> the aperture stage -> the warp's targets

closures  every bilabial the alignment put in the track (a run of frames on one P, B or M) keeps a shut frame: where neither the run nor
          the frame either side of it is a PP frame, its middle frame becomes one (vis PP, open 0). The closure comes from the alignment's
          phones, not from the vocal's sound.
asym      closures, then each frame's opening is the least of itself and the next ADVANCE_S of frames: the mouth starts closing that much
          before the drive falls, and opens on time, so it closes faster than it opens. Phrase-start pre-roll (opening before a line's
          first sound) is not done.

Pure: track frames in, new track frames out; nothing else is read."""

import numpy as np

MODES = ("off", "closures", "asym")
BILABIAL = ("P", "B", "M")
ADVANCE_S = 0.02  # s: a closing starts this much ahead of the drive's fall (one internal frame at 50 fps)


def _phone(f):
    return "".join(c for c in str(f.get("phone", "")).upper() if c.isalpha())


def closures(frames):
    """-> (new frames, frame indices made PP). The input is not changed."""
    out = [dict(f) for f in frames]
    n, added, i = len(out), [], 0
    while i < n:
        if _phone(out[i]) not in BILABIAL:
            i += 1
            continue
        j = i
        while j + 1 < n and out[j + 1].get("phone") == out[i].get("phone") and out[j + 1].get("word") == out[i].get("word"):
            j += 1
        if not any(out[k]["vis"] == "PP" for k in range(max(0, i - 1), min(n, j + 2))):
            k = (i + j) // 2
            out[k] = dict(out[k], vis2=out[k]["vis"], vis="PP", mix=0.0, open=0.0)
            added.append(k)
        i = j + 1
    return out, added


def advance_closing(opens, k):
    """each value the least of itself and the next k: falls come k frames earlier, rises keep their time."""
    o = np.asarray(opens, float)
    if k <= 0 or not len(o):
        return o.copy()
    p = np.concatenate([o, np.full(k, o[-1])])
    return np.min(np.stack([p[d : d + len(o)] for d in range(k + 1)]), 0)


def shaped(frames, fps, mode, advance_s=ADVANCE_S):
    """-> (new frames, info). mode off: the frames as they are."""
    if mode not in MODES:
        raise ValueError(f"--warp-shape {mode!r}")
    if mode == "off":
        return list(frames), {"mode": "off"}
    out, added = closures(frames)
    info = {"mode": mode, "closures_added": added}
    if mode == "asym":
        k = int(round(advance_s * fps))
        o = advance_closing([f.get("open", 0.0) for f in out], k)
        out = [dict(f, open=round(float(v), 3)) for f, v in zip(out, o)]
        info["advance_frames"] = k
    return out, info
