# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""interior_take.py: rung 1 of the real interior (--warp-interior-source take). Each dubbed frame's opening is filled from the take's
own closest frame by shape and pose (MouthBank.nearest): its tongue and cavity pixels, carried by an affine fit of the two inner lip
contours onto the planned contour. Teeth pixels are not harvested: they are replaced by the interior colour around them, the arch
layers keep the teeth. Per frame the pixels take a Lab match from the skin ring around the mouth (donor frame to this frame), and the
pink share of the visible opening is held to the take's own share at that aperture, never above it."""

import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

from . import lip_layers as ll

LIP_IN, LIP_OUT = ll.LIP_IN, ll.LIP_OUT
OPEN_MIN = 0.06  # a donor below this aperture (x mouth width) shows no interior
HOLD = 0.10  # keep the running donor while its shape distance is within this of the best one


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
        self.share = np.zeros(n)
        for i in range(n):
            self.share[i] = self.frame_pink(i)
        seen = self.ap >= OPEN_MIN
        # pink share by aperture: the take's median per bin, the cap of the fill
        self.bins = np.linspace(0.0, max(0.1, float(self.ap.max())) + 1e-6, 9)
        self.curve = np.full(len(self.bins) - 1, np.nan)
        for b in range(len(self.curve)):
            hit = (self.ap >= self.bins[b]) & (self.ap < self.bins[b + 1]) & seen
            if hit.any():
                self.curve[b] = float(np.median(self.share[hit]))
        ok = np.isfinite(self.curve)
        mid = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.curve = np.interp(mid, mid[ok], self.curve[ok]) if ok.any() else np.zeros_like(mid)
        # the donor per frame: a held run where the running donor still fits (fewer switches, less boil)
        self.j = np.full(n, -1)
        prev = -1
        for i in range(n):
            j, d = bank.nearest(i, Fe[i], scale, pose_w, exclude=-1, prev=prev)
            if prev >= 0 and float(np.max(np.abs(bank.F[prev] - Fe[i]) / scale)) <= d + HOLD:
                j = prev
            self.j[i] = prev = j
        ring = np.array([skin_ring(f, s) for f, s in zip(frames, S)])
        good = np.isfinite(ring[:, 0])
        idx = np.arange(n)
        for c in range(3):
            ring[:, c] = np.interp(idx, idx[good], ring[good, c]) if good.any() else 0.0
        self.ring = gaussian_filter1d(ring, 2.0, axis=0, mode="nearest")
        self.log = {}

    def frame_pink(self, i):
        P = np.asarray(self.S[i])[:, :2]
        x0, y0, x1, y1 = box_of(P[LIP_IN], *self.frames[i].shape[:2], 2)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return 0.0
        m = poly(P[LIP_IN] - [x0, y0], (y1 - y0, x1 - x0))
        if m.sum() < 8:
            return 0.0
        return float(pink(cv2.cvtColor(self.frames[i][y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32))[m].mean())

    def target(self, ap):
        return float(self.curve[int(np.clip(np.searchsorted(self.bins, ap) - 1, 0, len(self.curve) - 1))])

    def paint(self, i, Q, box, ap_out, over_a):
        """the box's interior: (BGR float, alpha). Q = the planned inner contour in box px, over_a = the alpha that sits above it
        (the upper arch), for the visible share."""
        x0, y0, w, h = box
        j = int(self.j[i])
        z = np.zeros((h, w), np.float32)
        if j < 0 or self.ap[j] < OPEN_MIN:
            return np.zeros((h, w, 3), np.float32), z
        src = np.asarray(self.S[j])[LIP_IN][:, :2]
        A = np.c_[src, np.ones(len(src))]
        M = np.linalg.lstsq(A, np.asarray(Q, float)[:, :2], rcond=None)[0].T  # donor frame px -> box px, 2 x 3
        img = cv2.warpAffine(self.frames[j], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        pts = (A @ M.T).astype(np.float32)
        have = poly(pts, (h, w))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        keep = have & ~cv2.dilate(teeth(lab).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if keep.sum() < 8:
            return np.zeros((h, w, 3), np.float32), z
        # teeth pixels and anything the donor's contour does not reach: the interior colour around them
        wt = keep.astype(np.float32)
        sg = max(2.0, 0.08 * np.sqrt(have.sum()))
        fill = cv2.GaussianBlur(lab * wt[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(wt, (0, 0), sg), 1e-3)[..., None]
        lab = np.where(keep[..., None], lab, fill)
        # the Lab match: donor light -> this frame's light, from the skin ring
        d, c = self.ring[j], self.ring[i]
        lab[..., 0] *= float(np.clip(c[0] / max(1.0, d[0]), 0.6, 1.6))
        lab[..., 1:] += (c[1:] - d[1:]).astype(np.float32)
        lab = np.clip(lab, 0, 255)
        opening = poly(np.asarray(Q)[:, :2], (h, w))
        vis = opening & (over_a < 0.5)
        got = float(pink(lab)[vis].sum()) / max(1, int(opening.sum()))
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
                if float(pink(sunk)[vis].sum()) / max(1, int(opening.sum())) <= tgt or k >= 1.0:
                    break
                k = min(1.0, k * 1.4 + 0.1)
            lab = sunk
        self.log[int(i)] = dict(donor=j, pink=round(got, 4), target=round(tgt, 4), sank=round(k, 3))
        bgr = cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        a = cv2.GaussianBlur(cv2.dilate(opening.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 0.6)
        return bgr, a
