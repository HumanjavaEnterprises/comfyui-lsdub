# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""nose.py: does the paint move the nose. The nose base against the take, frame by frame: the take's own head motion is in both, so what is left is the warp's.

  shifts(take, render, track) -> dx, dy per frame in px (a template of the take's nose base found in the render)
  stillness(dx, dy, iod)      -> reading: how far it sits off the take, how it steps, the frames it jumps, the verdict
  overpull(gap_take, gap_render) -> the frames a wide take was pulled nearly shut in one reach: what folds the skin

Reported by Verify and the ruler; a shot past NOSE_FAIL or with a jump past JUMP_FAIL fails its `nose` reading."""

import cv2
import numpy as np

SCALE, PAD = 3, 8  # upsampling for sub-pixel shifts, search range px
JUMP = 0.012  # eye distances a nose base may move against the take from one frame to the next: past it, a jump
JUMP_FAIL = 0.03  # a single step this big fails the shot
NOSE_FAIL = 6  # jumps a shot may hold
PULL_OPEN, PULL_DROP = (
    0.2,
    0.17,
)  # a take at least this open, rendered this far under it (eye distances): the warp asked more than the skin gives


def grey(f):
    return cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)


def box(p):
    xs, ys = p[[98, 327], 0], p[[1, 4], 1]
    return int(xs.min()) - 2, int(xs.max()) + 2, int(ys.min()) + 2, int(p[2, 1]) + 3


def shift(a, b, m=PAD):
    """where a's middle sits in b, px from where it was."""
    A = cv2.resize(a[m:-m, m:-m], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    B = cv2.resize(b, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    _, _, _, loc = cv2.minMaxLoc(cv2.matchTemplate(B, A, cv2.TM_CCOEFF_NORMED))
    return loc[0] / SCALE - m, loc[1] / SCALE - m


def shifts(take, render, track):
    n = min(len(take), len(render), len(track))
    dx, dy = np.full(n, np.nan), np.full(n, np.nan)
    for f in range(n):
        if not np.isfinite(track[f]).all():
            continue
        x0, x1, y0, y1 = box(track[f])
        H, W = take[f].shape[:2]
        if x0 - PAD < 0 or y0 - PAD < 0 or y1 + PAD >= H or x1 + PAD >= W or y1 - y0 < 8:
            continue
        s = np.s_[y0 - PAD : y1 + PAD, x0 - PAD : x1 + PAD]
        dx[f], dy[f] = shift(grey(take[f])[s], grey(render[f])[s])
    return dx, dy


def stillness(dx, dy, iod):
    """-> dict: mean offset and steps in eye distances, the frames that jump, ok."""
    d = np.hypot(dx, dy) / iod
    st = np.hypot(np.diff(dx), np.diff(dy)) / iod[1:]
    jumps = np.flatnonzero(st > JUMP) + 1
    top = float(np.nanmax(st)) if np.isfinite(st).any() else 0.0
    return dict(
        nose_off=round(float(np.nanmean(d)), 4) if np.isfinite(d).any() else None,
        nose_step=round(top, 4),
        nose_jumps=[int(k) for k in jumps],
        nose_ok=bool(top <= JUMP_FAIL and len(jumps) <= NOSE_FAIL),
    )


def overpull(gap_take, gap_render):
    """frames where a wide take was pulled nearly shut: the reach that folds the skin."""
    t, r = np.asarray(gap_take, float), np.asarray(gap_render, float)
    return [int(k) for k in np.flatnonzero((t >= PULL_OPEN) & (t - r >= PULL_DROP))]
