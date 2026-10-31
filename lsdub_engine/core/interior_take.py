# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""interior_take.py: rung 1 of the real interior (--warp-interior-source take). Each dubbed frame's opening is filled from the take's
own closest frame by shape and pose (MouthBank.nearest): its tongue and cavity pixels, carried by an affine fit of the two inner lip
contours onto the planned contour. Teeth pixels are not harvested: they are replaced by the interior colour around them, the arch
layers keep the teeth. Per frame the pixels take a Lab match from the skin ring around the mouth (donor frame to this frame); the
interior's lightness and the pink share of the visible opening are held to the take's own at that aperture, never above."""

import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

from . import lip_layers as ll

LIP_IN, LIP_OUT = ll.LIP_IN, ll.LIP_OUT
OPEN_MIN = 0.06  # a donor below this aperture (x mouth width) shows no interior
HOLD = 0.50  # keep the running donor while its cost is within this of the best one
DWELL = 10  # frames a new donor stays, unless it stops being open enough
STILL_PCT = 60  # the frames below this percentile of the plan's gap speed count as held: the donor does not change on them
SOFT = 0.012  # the interior's softening, x the box's side, at least 1 px
FADE = 5  # frames of cross-fade around a donor switch
OPEN_KEEP = 0.70  # the running donor may stay down to this share of the planned gap
OPEN_FIT = (
    0.85  # a donor opens at least this share of the planned gap: a narrower one is stretched, and its teeth and lip rim fill the mouth
)


def aperture(P):
    return float(np.linalg.norm(P[13, :2] - P[14, :2]) / max(1e-6, np.linalg.norm(P[61, :2] - P[291, :2])))


def poly(pts, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, float)).astype(np.int32)], 1)
    return m > 0


def pink(lab):
    """the mouth-ruler's pink test on an OpenCV Lab image (float or uint8)."""
    L = lab[..., 0] * 100.0 / 255.0
    return (L > 25) & (L <= 75) & (lab[..., 1] - 128.0 > 14)


def teeth(lab):
    L = lab[..., 0] * 100.0 / 255.0
    return (L > 55) & (np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0) < 22)


def box_of(P, H, W, pad):
    x0, y0 = np.floor(P[:, :2].min(0) - pad).astype(int)
    x1, y1 = np.ceil(P[:, :2].max(0) + pad).astype(int)
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def skin_ring(frame, P):
    """mean Lab of the skin ring around the lips."""
    P = np.asarray(P, float)
    w = np.linalg.norm(P[61, :2] - P[291, :2])
    x0, y0, x1, y1 = box_of(P[LIP_OUT], *frame.shape[:2], 0.45 * w)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.full(3, np.nan)
    sh = (y1 - y0, x1 - x0)
    m = poly(P[LIP_OUT][:, :2] - [x0, y0], sh).astype(np.uint8)
    k = max(3, int(0.12 * w) | 1)
    ko = max(k + 2, int(0.40 * w) | 1)
    ring = cv2.dilate(m, np.ones((ko, ko), np.uint8)) & ~cv2.dilate(m, np.ones((k, k), np.uint8)) & 1
    if ring.sum() < 16:
        return np.full(3, np.nan)
    return cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[ring > 0].astype(np.float64).mean(0)


class TakeInterior:
    def __init__(self, bank, frames, S, Fe, scale, pose_w=2.0):
        self.frames, self.S = frames, S
        n = len(frames)
        self.ap = np.array([aperture(np.asarray(S[i])) for i in range(n)])
        self.share, self.dark = np.zeros(n), np.full(n, np.nan)
        for i in range(n):
            self.share[i], self.dark[i] = self.frame_interior(i)
        seen = self.ap >= OPEN_MIN
        # pink share by aperture: the take's median per bin, the cap of the fill
        self.bins = np.linspace(0.0, max(0.1, float(self.ap.max())) + 1e-6, 9)
        mid = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.curve = self.binned(self.share, seen, mid)
        self.lcurve = self.binned(self.dark, seen, mid)
        self.curve, self.lcurve = np.nan_to_num(self.curve, nan=0.0), np.nan_to_num(self.lcurve, nan=255.0)
        self.j = self.pick(bank, Fe, scale, pose_w)
        ring = np.array([skin_ring(f, s) for f, s in zip(frames, S)])
        good = np.isfinite(ring[:, 0])
        idx = np.arange(n)
        for c in range(3):
            ring[:, c] = np.interp(idx, idx[good], ring[good, c]) if good.any() else 0.0
        self.ring = gaussian_filter1d(ring, 2.0, axis=0, mode="nearest")
        self.log, self.gain = {}, {}

    def binned(self, v, seen, mid):
        """v's median per aperture bin, interpolated over the empty ones."""
        c = np.full(len(mid), np.nan)
        for b in range(len(mid)):
            hit = (self.ap >= self.bins[b]) & (self.ap < self.bins[b + 1]) & seen & np.isfinite(v)
            if hit.any():
                c[b] = float(np.median(v[hit]))
        ok = np.isfinite(c)
        return np.interp(mid, mid[ok], c[ok]) if ok.any() else np.full(len(mid), np.nan)

    def pick(self, bank, Fe, scale, pose_w):
        """the donor per frame, with hysteresis: open enough for the planned gap (or the widest the take has), closest by shape and
        pose; the running donor stays while it is still open enough and either the mouth is held, it is within HOLD of the best, or it
        changed under DWELL frames ago."""
        n, gap = len(self.frames), bank.F[:, 0]
        usable = self.ap >= OPEN_MIN
        top = float(gap[usable].max()) if usable.any() else 0.0
        js, prev, since = np.full(n, -1), -1, 0
        speed = np.abs(np.diff(Fe[:, 0], prepend=Fe[0, 0]))
        still = speed < np.percentile(speed, STILL_PCT)  # a change of pixels shows on a held mouth and hides in a moving one
        for i in range(n):
            fit = usable & (gap >= min(OPEN_FIT * Fe[i, 0], 0.9 * top))
            cost = np.max(np.abs(bank.F - Fe[i]) / scale, axis=1) + pose_w * np.linalg.norm(bank.pose - bank.pose[i], axis=1)
            cost = np.where(fit, cost, np.inf)
            j = int(np.argmin(cost)) if fit.any() else prev
            if (
                prev >= 0
                and usable[prev]
                and gap[prev] >= min(OPEN_KEEP * Fe[i, 0], 0.9 * top)
                and (still[i] or since < DWELL or cost[prev] <= cost[j] + HOLD)
            ):
                j = prev
            since = since + 1 if j == prev else 0
            js[i] = prev = j
        return js

    def frame_interior(self, i):
        """the take's frame: (pink share of the opening, median L of the opening without its teeth), L in OpenCV's 0-255."""
        P = np.asarray(self.S[i])[:, :2]
        x0, y0, x1, y1 = box_of(P[LIP_IN], *self.frames[i].shape[:2], 2)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return 0.0, np.nan
        m = poly(P[LIP_IN] - [x0, y0], (y1 - y0, x1 - x0))
        if m.sum() < 8:
            return 0.0, np.nan
        lab = cv2.cvtColor(self.frames[i][y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
        rest = m & ~teeth(lab)
        return float(pink(lab)[m].mean()), float(np.median(lab[..., 0][rest])) if rest.sum() >= 8 else np.nan

    def target(self, ap, curve=None):
        return float(np.interp(ap, 0.5 * (self.bins[:-1] + self.bins[1:]), self.curve if curve is None else curve))

    def donor_lab(self, j, Q, h, w):
        """donor j carried onto the planned contour: its teeth gone, in Lab, with the mask of what it covers; None if it shows nothing."""
        if j < 0 or self.ap[j] < OPEN_MIN:
            return None
        src = np.asarray(self.S[j])[LIP_IN][:, :2]
        A = np.c_[src, np.ones(len(src))]
        M = np.linalg.lstsq(A, np.asarray(Q, float)[:, :2], rcond=None)[0].T  # donor frame px -> box px, 2 x 3
        img = cv2.warpAffine(self.frames[j], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        have = poly((A @ M.T).astype(np.float32), (h, w))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        keep = have & ~cv2.dilate(teeth(lab).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if keep.sum() < 8:
            return None
        # teeth pixels and anything the donor's contour does not reach: the interior colour around them
        wt = keep.astype(np.float32)
        sg = max(2.0, 0.08 * np.sqrt(have.sum()))
        fill = cv2.GaussianBlur(lab * wt[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(wt, (0, 0), sg), 1e-3)[..., None]
        lab = np.where(keep[..., None], lab, fill)
        return lab, self.ring[j]

    def matched(self, j, i, Q, h, w):
        got = self.donor_lab(j, Q, h, w)
        if got is None:
            return None
        lab, d = got
        c = self.ring[i]
        lab[..., 0] *= float(np.clip(c[0] / max(1.0, d[0]), 0.6, 1.6))
        lab[..., 1:] += (c[1:] - d[1:]).astype(np.float32)
        return lab

    def paint(self, i, Q, box, ap_out, over_a, over_rgb=None):
        """the box's interior: (BGR float, alpha). Q = the planned inner contour in box px, over_a / over_rgb = the layer that sits
        above it (the upper arch): its pink counts towards the visible share. Around a donor switch the two donors cross-fade."""
        x0, y0, w, h = box
        j = int(self.j[i])
        z = np.zeros((h, w), np.float32)
        lab = self.matched(j, i, Q, h, w)
        if lab is None:
            return np.zeros((h, w, 3), np.float32), z
        n, r = len(self.j), FADE // 2
        for s in range(max(1, i - r), min(n, i + r + 1)):  # s = a frame where the donor changed
            if self.j[s] != self.j[s - 1] and self.j[s - 1] >= 0 and self.j[s] >= 0:
                t = (i - s + r + 1) / (2 * r + 1)  # 0 -> 1 across the switch: the old donor, then the new
                a, b = self.matched(int(self.j[s - 1]), i, Q, h, w), self.matched(int(self.j[s]), i, Q, h, w)
                if a is not None and b is not None:
                    lab = a * (1 - t) + b * t
                break
        lab = np.clip(lab, 0, 255)
        opening = poly(np.asarray(Q)[:, :2], (h, w))
        # a warp's sub-pixel wander in fine texture reads as boil. Inside the opening only: the lips around it must not bleed in as pink
        sg, m = max(1.0, SOFT * np.sqrt(h * w)), opening.astype(np.float32)
        lab = np.where(
            opening[..., None],
            cv2.GaussianBlur(lab * m[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(m, (0, 0), sg), 1e-3)[..., None],
            lab,
        )
        vis = opening & (over_a < 0.5)
        if vis.sum() >= 8:  # never lighter than the take's interior at this aperture
            g = float(np.clip(self.target(ap_out, self.lcurve) / max(1.0, float(np.median(lab[..., 0][vis]))), 0.3, 1.0))
            g = 0.75 * self.gain.get(i - 1, g) + 0.25 * g  # no flicker from the median of one frame
            self.gain[i] = g
            lab[..., 0] *= g
        over, area = 0, max(1, int(opening.sum()))
        if over_rgb is not None and (opening & ~vis).any():
            ol = cv2.cvtColor(np.clip(over_rgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            over = int(pink(ol)[opening & ~vis].sum())
        got = (float(pink(lab)[vis].sum()) + over) / area
        tgt = self.target(ap_out)
        k = 0.0
        if got > tgt:  # never pinker than the take at this aperture: the excess sinks into the cavity's colour
            k = 1.0 - tgt / got
            pk = pink(lab)
            pm = cv2.GaussianBlur(pk.astype(np.float32), (0, 0), 1.0)[..., None]
            nonp = ~pk & opening
            base = (np.median(lab[nonp], 0) if nonp.sum() > 8 else np.array([50.0, 138.0, 134.0])).astype(np.float32)
            for _ in range(6):  # a blend only leaves the pink test late: raise it until the share is held
                sunk = lab * (1 - pm * k) + base * pm * k
                if (float(pink(sunk)[vis].sum()) + over) / area <= tgt or k >= 1.0:
                    break
                k = min(1.0, k * 1.4 + 0.1)
            lab = sunk
        self.log[int(i)] = dict(donor=j, pink=round(got, 4), target=round(tgt, 4), sank=round(k, 3))
        bgr = cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        a = cv2.GaussianBlur(cv2.dilate(opening.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 0.6)
        return bgr, a
