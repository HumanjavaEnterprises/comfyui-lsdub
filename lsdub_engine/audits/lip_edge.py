# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lip_edge.py: lip-edge counting per frame (only our animated lips may ever be visible).

Lips are redder than skin, so walk down three columns (philtrum line and +-25 % of the mouth width) from the inner
lower lip toward the chin, in Lab a* (redness) relative to the frame's own cheek skin: the real lower lip is the first
red band; a SECOND red band (or a second strong luminance edge) below it, before the chin, is a stray lip edge. Same
walk above the upper lip toward the nose. A frame counts when >= 2 of the 3 columns see a second band.

`frame_ghost` = two LUMINANCE valleys along the column (it also counts a teeth row or the tongue over the lower lip);
`frame_ghost_lip` = the lip-aware check: the first red run from the inner lip is the lip itself, the teeth / tongue /
cavity region is excluded, and only a SECOND red run after a run of skin counts. Used by the per-frame gate."""
import cv2, numpy as np

CHEEK = [187, 207, 205, 50, 411, 427, 425, 280]
LIP_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIP_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def valleys(prof, prom=12.0):
    from scipy.signal import find_peaks
    pk, _ = find_peaks(-np.asarray(prof, float), prominence=prom, distance=3)
    return len(pk)


def _cols(p):
    """the six walks: (which, fx, start, end) in frame px."""
    for fx in (0.0, -0.2, 0.2):
        for which in ('lo', 'up'):
            if which == 'lo':
                s0 = p[14] + fx * (p[291] - p[61])
                s1 = s0 + 0.7 * (p[152] - p[14])
            else:
                s0 = p[13] + fx * (p[291] - p[61])
                s1 = s0 + 0.75 * (p[2] - p[13])
            yield which, fx, s0, s1


def _line(img, s0, s1, interp=cv2.INTER_LINEAR):
    L = max(8, int(np.linalg.norm(s1 - s0)))
    xs = np.linspace(s0[0], s1[0], L)
    ys = np.linspace(s0[1], s1[1], L)
    return cv2.remap(img, xs.astype(np.float32)[None], ys.astype(np.float32)[None], interp)[0]


def frame_ghost(f, p):
    Lc = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 0.8)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    grey = ((hsv[..., 1] < 45) & (hsv[..., 2] > 60)).astype(np.uint8)
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # mic mesh / metal, not thin teeth lines
    hits_lo, hits_up = 0, 0
    centre = False
    for which, fx, s0, s1 in _cols(p):
        prof = _line(Lc, s0, s1)
        # stop at a mic / foreign object (grey, unsaturated) crossing the column: its edges are not lips
        gm = _line(grey, s0, s1, cv2.INTER_NEAREST)
        hit = np.nonzero(gm[3:] > 0)[0]
        if len(hit): prof = prof[: 3 + hit[0]]
        if len(prof) >= 8 and valleys(prof) >= 2:
            if which == 'lo': hits_lo += 1
            else: hits_up += 1
            if fx == 0.0: centre = True
    # the ghost sits on the midline (where the old lip hangs lowest): the centre column, confirmed by one more column
    return (centre and (hits_lo >= 2 or hits_up >= 2)) or hits_lo >= 3 or hits_up >= 3, hits_lo, hits_up


def red_runs(a_rel, thr, min_len=3, min_gap=3):
    """runs of a* above thr along a column (start, end) with gaps < min_gap bridged and runs < min_len dropped."""
    r = np.asarray(a_rel, float) > thr
    runs = []
    i = 0
    n = len(r)
    while i < n:
        if not r[i]:
            i += 1
            continue
        j = i
        while True:
            while j + 1 < n and r[j + 1]: j += 1                 # the red run
            nxt = j + 1
            while nxt < n and not r[nxt]: nxt += 1               # the gap after it
            if nxt < n and nxt - j - 1 < min_gap: j = nxt        # a short gap is bridged
            else: break
        if j - i + 1 >= min_len: runs.append((i, j))
        i = j + 1
    return runs


def frame_ghost_lip(f, p):
    """lip-aware: along each walk, a* (Lab redness) relative to the cheek skin; the lip's own a* sets the threshold
    (35 % of the way from skin to lip). The first red run is the lip; a second red run after >= 3 px of skin is a second
    lip. The teeth / tongue / cavity region is excluded: pixels that are not red (teeth, cavity) never form a run, and the
    walk starts at the inner lip so the interior is behind it. Returns (ghost, hits_lo, hits_up)."""
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = cv2.GaussianBlur(lab[..., 1], (0, 0), 0.8)
    H, W = a.shape
    def hull(idx, k=1.0):
        q = p[idx, :2]
        c = q.mean(0)
        q = c + (q - c) * k
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(q).astype(np.int32)), 1)
        return m > 0
    cheek = hull(CHEEK, 0.9)
    lipm = hull(LIP_OUT, 0.95) & ~hull(LIP_IN, 1.05)
    if cheek.sum() < 20 or lipm.sum() < 20: return False, 0, 0
    a_skin = float(np.median(a[cheek]))
    a_lip = float(np.median(a[lipm]))
    d = a_lip - a_skin
    if d < 3: return False, 0, 0                                               # no red lips to find a second one of
    thr = max(4.0, 0.35 * d)
    hits_lo, hits_up = 0, 0
    centre = False
    for which, fx, s0, s1 in _cols(p):
        prof = _line(a, s0, s1) - a_skin
        if len(prof) < 8: continue
        # a lip's own darker lower half may dip under the threshold: gaps shorter than 12 % of the walk stay one lip
        runs = red_runs(prof, thr, min_gap=max(3, int(0.12 * len(prof))))
        # the first run must be the lip itself: it starts within the first 40 % of the walk (the inner lip is where the
        # walk starts; a lip that begins later than that is a lip the track missed, still the first lip); the second
        # must be clearly lip-coloured (its peak at least half way from skin to lip)
        if len(runs) >= 2 and runs[0][0] <= 0.4 * len(prof) and runs[1][0] - runs[0][1] >= 3 and prof[runs[1][0]:runs[1][1] + 1].max() >= 0.5 * d:
            if which == 'lo': hits_lo += 1
            else: hits_up += 1
            if fx == 0.0: centre = True
    return (centre and (hits_lo >= 2 or hits_up >= 2)) or hits_lo >= 3 or hits_up >= 3, hits_lo, hits_up


