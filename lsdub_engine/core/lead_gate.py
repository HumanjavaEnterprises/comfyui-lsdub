# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lead_gate.py: drive only what the singer on screen sings. One stage of the drive chain, behind one paint flag, --warp-lead-gate.

  stem -> [lead gate] -> drive (the track's per-frame `open`) -> the shape and aperture stages -> the warp's targets

Where the take's own mouth stays shut and still for at least HOLD_S while the drive asks it to open, the voice in the stem is most
likely not this singer's: a backing line, an ad-lib, another voice bleeding into the lead stem. The drive is held down there, eased
over EASE frames at each end. The take decides, not the vocal: the gate reads the take's own opening only (in units of its range).

Pure: the drive's openings and the take's openings in, the gated openings and the gate out. A better lead / backing split of the stem is
the user's to make before the dub (a lead stem given to the dub is what drives it); none is bundled."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

SHUT = 0.15  # the take's mouth, in units of its own range (g_lo..g_hi), at or under this is shut
STILL = 0.08  # ... and still when it moves less than this (p90 - p10) across the run
HOLD_S = 0.4  # s: the shortest run that counts (a breath between words is not a voice the singer did not sing)
ASK = 0.3  # the drive must ask for at least this opening somewhere in the run, or there is nothing to gate
EASE = 1.0  # frames: zero-phase ease of the gate's edges


def runs(mask):
    """[(start, end inclusive)] of the True runs."""
    m = np.r_[False, np.asarray(mask, bool), False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def gate(opens, take_u, fps, shut=SHUT, still=STILL, hold_s=HOLD_S, ask=ASK, ease=EASE):
    """-> (gated openings, gate 0..1 per frame, [(start, end)] of the gated runs)."""
    o = np.asarray(opens, float)
    u = np.asarray(take_u, float)[: len(o)]
    if len(u) < len(o):
        u = np.r_[u, np.full(len(o) - len(u), np.inf)]  # past the take's frames: never shut
    hold = max(1, int(round(hold_s * fps)))
    g = np.zeros(len(o))
    kept = []
    for a, b in runs(u <= shut):
        if b - a + 1 < hold:
            continue
        seg = u[a : b + 1]
        if np.percentile(seg, 90) - np.percentile(seg, 10) > still or o[a : b + 1].max() < ask:
            continue
        g[a : b + 1] = 1.0
        kept.append((int(a), int(b)))
    if ease > 0 and kept:
        g = np.maximum(g, gaussian_filter1d(g, ease, mode="nearest"))
    return o * (1.0 - np.clip(g, 0.0, 1.0)), g, kept


def gated(frames, take_u, fps, **kw):
    """track frames -> (new frames with the gated 'open', info). The input is not changed."""
    o, g, kept = gate([f.get("open", 0.0) for f in frames], take_u, fps, **kw)
    return [dict(f, open=round(float(v), 3)) for f, v in zip(frames, o)], {"runs": kept, "frames": int((g > 0.5).sum())}
