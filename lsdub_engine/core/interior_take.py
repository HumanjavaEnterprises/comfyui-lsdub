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
PINK_ROOM = 0.94  # the fill's core pink stops at this share of the take's core share at that aperture
CURVE_SIGMA = 1.0  # the pink curve's smoothing across aperture bins
CORE = 7  # the take's pink share is read this many px inside its contour (kernel side): the rim is lip, not interior
RELEASE = 0.8  # the pink sink's share kept per frame after the last frame that needed it
SANK_MIN = 0.005  # a sink below this is off
SANK_STEPS = 8  # bisection steps on the sink: a blend to 1/256
OPEN_MIN = 0.06  # a donor below this aperture (x mouth width) shows no interior
HOLD = 0.50  # keep the running donor while its cost is within this of the best one
DWELL = 10  # frames a new donor stays, unless it stops being open enough
STILL_PCT = 60  # the frames below this percentile of the plan's gap speed count as held: the donor does not change on them
FILL_REACH = 0.02  # --warp-interior-mend: under this share of kept pixels nearby, the fill's reading is noise
SINK_DARK = 15  # --warp-interior-mend: the percentile of the opening's non-pink lightness the sink aims at
SOFT = 0.02  # the interior's softening, x the box's side, at least 1 px
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


def least_sink(lab, pm, base, vis, over, area, tgt):
    """the smallest blend toward `base` that brings the pink share to tgt, by bisection: the share falls off a cliff against the
    blend, so stepping up overshoots far under the target. 1.0 when even that does not hold it."""
    held = lambda v: (float(pink(lab * (1 - pm * v) + base * pm * v)[vis].sum()) + over) / area <= tgt
    if not held(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(SANK_STEPS):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if held(mid) else (mid, hi)
    return hi


def inner(m):
    """the opening clear of the lip's own rim, as the ruler's pink_core; the whole opening when too little is left."""
    core = cv2.erode(m.astype(np.uint8), np.ones((CORE, CORE), np.uint8)) > 0
    return core if core.sum() > 30 else m


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
    def __init__(self, bank, frames, S, Fe, scale, pose_w=2.0, moved=None, open_floor=0.0, mend=False):
        """moved: frames per frame the plan was moved earlier (--coart-sync-vocal): the donors are picked on the plan as it stood and
        move with it, so moving the plan does not change which of the take's frames fill it. open_floor: donors open at least this
        percentile (0..1) of the take's open frames: a narrow planned mouth is filled from a frame with an interior to show, since a
        narrow donor's opening is lip rim, which the pink sink turns into a flat slot. mend: the fill around a donor's teeth never reads
        where it has no pixels (those read as Lab zero, blue), and the sink takes pink into the cavity's dark, not its median."""
        self.frames, self.S = frames, S
        self.mend = bool(mend)
        self.open_floor = float(open_floor)
        n = len(frames)
        self.ap = np.array([aperture(np.asarray(S[i])) for i in range(n)])
        self.share, self.dark = np.zeros(n), np.full(n, np.nan)
        for i in range(n):
            self.share[i], self.dark[i] = self.frame_interior(i)
        seen = self.ap >= OPEN_MIN
        # pink share by aperture: the take's median per bin, the cap of the fill
        self.bins = np.linspace(0.0, max(0.1, float(self.ap.max())) + 1e-6, 9)
        mid = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.curve = self.pink_curve(self.binned(self.share, seen, mid, drop_zero=True))
        self.lcurve = np.nan_to_num(self.binned(self.dark, seen, mid), nan=255.0)
        if moved is None:
            self.j = self.pick(bank, Fe, scale, pose_w)
        else:
            k, d = np.arange(n), np.asarray(moved, int)
            self.j = self.pick(bank, Fe[np.clip(k - d, 0, n - 1)], scale, pose_w)[np.clip(k + d, 0, n - 1)]
        ring = np.array([skin_ring(f, s) for f, s in zip(frames, S)])
        good = np.isfinite(ring[:, 0])
        idx = np.arange(n)
        for c in range(3):
            ring[:, c] = np.interp(idx, idx[good], ring[good, c]) if good.any() else 0.0
        self.ring = gaussian_filter1d(ring, 2.0, axis=0, mode="nearest")
        self.log, self.gain, self.sank = {}, {}, {}

    EMPTY = (None, None, None)
    open_floor = 0.0
    mend = False

    def state(self, i):
        """what paint(i) left behind: a gate candidate's copy, to keep it only if the candidate is kept."""
        return self.log.get(int(i)), self.gain.get(i), self.sank.get(i)

    def put(self, i, st):
        """state(i) back; None = frame i has none (a candidate that paints no interior)."""
        for d, k, v in zip((self.log, self.gain, self.sank), (int(i), i, i), st):
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v

    @staticmethod
    def pink_curve(c):
        """the cap per aperture, smooth across bins so a frame's target cannot jump with its bin. No reading anywhere: no cap."""
        return gaussian_filter1d(c, CURVE_SIGMA, mode="nearest") if np.isfinite(c).all() else np.ones(len(c))

    def binned(self, v, seen, mid, drop_zero=False):
        """v's median per aperture bin, interpolated over the empty ones. drop_zero: a bin reading exactly 0 is empty too."""
        c = np.full(len(mid), np.nan)
        for b in range(len(mid)):
            hit = (self.ap >= self.bins[b]) & (self.ap < self.bins[b + 1]) & seen & np.isfinite(v)
            if hit.any():
                c[b] = float(np.median(v[hit]))
        ok = np.isfinite(c) & (c > 0 if drop_zero else True)
        return np.interp(mid, mid[ok], c[ok]) if ok.any() else np.full(len(mid), np.nan)

    def pick(self, bank, Fe, scale, pose_w):
        """the donor per frame, with hysteresis: open enough for the planned gap (or the widest the take has), closest by shape and
        pose; the running donor stays while it is still open enough and either the mouth is held, it is within HOLD of the best, or it
        changed under DWELL frames ago."""
        n, gap = len(self.frames), bank.F[:, 0]
        usable = self.ap >= OPEN_MIN
        top = float(gap[usable].max()) if usable.any() else 0.0
        floor = float(np.percentile(gap[usable], 100 * self.open_floor)) if self.open_floor > 0 and usable.any() else 0.0
        js, prev, since = np.full(n, -1), -1, 0
        speed = np.abs(np.diff(Fe[:, 0], prepend=Fe[0, 0]))
        still = speed < np.percentile(speed, STILL_PCT)  # a change of pixels shows on a held mouth and hides in a moving one
        for i in range(n):
            fit = usable & (gap >= max(floor, min(OPEN_FIT * Fe[i, 0], 0.9 * top)))
            cost = np.max(np.abs(bank.F - Fe[i]) / scale, axis=1) + pose_w * np.linalg.norm(bank.pose - bank.pose[i], axis=1)
            cost = np.where(fit, cost, np.inf)
            j = int(np.argmin(cost)) if fit.any() else prev
            if (
                prev >= 0
                and usable[prev]
                and gap[prev] >= max(floor, min(OPEN_KEEP * Fe[i, 0], 0.9 * top))
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
        share = float(pink(lab)[inner(m)].mean())
        return share, float(np.median(lab[..., 0][rest])) if rest.sum() >= 8 else np.nan

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
        reach = cv2.GaussianBlur(wt, (0, 0), sg)
        fill = cv2.GaussianBlur(lab * wt[..., None], (0, 0), sg) / np.maximum(reach, 1e-3)[..., None]
        if self.mend:
            fill = np.where(reach[..., None] < FILL_REACH, lab[keep].mean(0), fill)
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
        core = inner(opening)  # the pink cap reads the opening as the take's curve was read: clear of the rim
        if vis.sum() >= 8:  # never lighter than the take's interior at this aperture
            g = float(np.clip(self.target(ap_out, self.lcurve) / max(1.0, float(np.median(lab[..., 0][vis]))), 0.3, 1.0))
            g = 0.75 * self.gain.get(i - 1, g) + 0.25 * g  # no flicker from the median of one frame
            self.gain[i] = g
            lab[..., 0] *= g
        seen = core & vis
        over, area = 0, max(1, int(core.sum()))
        if over_rgb is not None and (core & ~vis).any():
            ol = cv2.cvtColor(np.clip(over_rgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            over = int(pink(ol)[core & ~vis].sum())
        got = (float(pink(lab)[seen].sum()) + over) / area
        tgt = PINK_ROOM * self.target(ap_out)  # both are core shares
        need, tail = 0.0, RELEASE * self.sank.get(i - 1, 0.0)
        if got > tgt or tail >= SANK_MIN:  # never pinker than the take at this aperture: the excess sinks into the cavity's colour
            pk = pink(lab)
            pm = cv2.GaussianBlur(pk.astype(np.float32), (0, 0), 1.0)[..., None]
            nonp = ~pk & opening
            base = (np.median(lab[nonp], 0) if nonp.sum() > 8 else np.array([50.0, 138.0, 134.0])).astype(np.float32)
            if self.mend and nonp.sum() > 8:
                base[0] = min(base[0], float(np.percentile(lab[nonp][:, 0], SINK_DARK)))
            if got > tgt:
                need = least_sink(lab, pm, base, seen, over, area, tgt)
        k = max(need, tail)  # the cap holds from the first frame it is needed; letting go is what is eased
        k = 0.0 if k < SANK_MIN else k
        if k > 0:
            lab = lab * (1 - pm * k) + base * pm * k
        self.sank[i] = k
        held = (float(pink(lab)[seen].sum()) + over) / area
        self.log[int(i)] = dict(donor=j, pink=round(got, 4), held=round(held, 4), target=round(tgt, 4), sank=round(k, 3))
        bgr = cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        a = cv2.GaussianBlur(cv2.dilate(opening.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 0.6)
        return bgr, a


# --warp-teeth-fit take
LINE_DEPTH = 0.06  # a row this much darker than the crown above it is the take's bite line: the atlas below it is the other arch
COMB = 0.004  # the atlas's softening along the arch, x IOD: the column grille of the harvest and the hard crown lines
FIT_MAX = 1.35  # the most the arch may be lifted
LINE_SPAN = 0.30  # the bite line is looked for in this share of the crown nearest the edge


def clean_atlas(arch, iod, upper=True, th=None):
    """one arch, cleaned once in atlas space (so it cannot boil): dark specks gone, the harvest's strip past the bite line cut
    (one arch, no second row under the first), the column grille and crown lines softened along the arch. In place, returns the log."""
    T = arch.tex.astype(np.float32).copy()
    Th, Tw = T.shape[:2]
    r_e = (0.0 - arch.h0) * arch.texel - 0.5
    th = max(4.0, (0.09 * iod if th is None else th) * arch.texel)
    rows = np.arange(Th, dtype=np.float32)
    d = r_e - rows if upper else rows - r_e  # 0 at the incisal edge, + toward the gum
    crown = (d >= -0.5) & (d <= th)
    rgb = np.clip(T[..., :3], 0, 255).astype(np.uint8)
    k = max(3, int(0.012 * iod * arch.texel) | 1)
    T[..., :3] = np.where(crown[:, None, None] & (T[..., 3:4] > 0.5), cv2.medianBlur(rgb, min(k, 5)).astype(np.float32), T[..., :3])
    L = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    sm = cv2.GaussianBlur(L, (0, 0), max(1.0, 0.02 * iod * arch.texel), sigmaY=0.1)  # along the arch: the line, not the teeth
    near = np.nonzero((d >= 0) & (d <= LINE_SPAN * th))[0]
    above = np.nonzero((d > LINE_SPAN * th) & (d <= 0.7 * th))[0]
    cut = np.full(Tw, -1, np.float32)
    if len(near) >= 3 and len(above) >= 2:
        body = np.median(sm[above], 0)
        rw = near[np.argmin(sm[near], 0)]
        dip = sm[rw, np.arange(Tw)]
        ok = (dip < (1 - LINE_DEPTH) * body) & (T[above, :, 3].mean(0) > 0.5)
        kk = max(3, int(0.08 * iod * arch.texel) | 1)
        rw = cv2.medianBlur(rw.astype(np.float32)[None, :], min(kk, 5)).ravel()
        cut = np.where(ok, rw, -1.0)
        for c in np.nonzero(ok)[0]:
            s = d - (r_e - cut[c] if upper else cut[c] - r_e)  # + = crown side of the line
            T[:, c, 3] *= np.clip(0.5 + s / max(1.0, 0.25 * arch.texel), 0, 1)
    sg = max(0.8, COMB * iod * arch.texel)
    pm = np.dstack([T[..., :3] * T[..., 3:4], T[..., 3]])
    bl = cv2.GaussianBlur(pm, (0, 0), sg, sigmaY=0.1)
    m = crown[:, None, None].astype(np.float32)
    pm = pm * (1 - m) + bl * m
    T = np.dstack([pm[..., :3] / np.maximum(pm[..., 3:4], 1e-3), pm[..., 3]])
    arch.tex = T.astype(np.float32)
    return dict(cut_cols=int((cut >= 0).sum()), comb_sigma=round(float(sg), 2))


class TeethFit:
    """--warp-teeth-fit take: the arch's lightness held to the take's own teeth at this aperture. Never above the take's."""

    def __init__(self, frames, S):
        n = len(frames)
        self.ap = np.array([aperture(np.asarray(s)) for s in S])
        L = np.full(n, np.nan)
        for i in range(n):
            P = np.asarray(S[i])[:, :2]
            x0, y0, x1, y1 = box_of(P[LIP_IN], *frames[i].shape[:2], 2)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            m = poly(P[LIP_IN] - [x0, y0], (y1 - y0, x1 - x0))
            lab = cv2.cvtColor(frames[i][y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
            t = m & teeth(lab)
            if t.sum() >= 25:
                L[i] = float(lab[..., 0][t].mean())
        ok = np.isfinite(L)
        self.seen = int(ok.sum())
        self.bins = np.linspace(float(self.ap.min()), float(self.ap.max()) + 1e-6, 6)
        mid, c = 0.5 * (self.bins[:-1] + self.bins[1:]), []
        for b in range(5):
            hit = ok & (self.ap >= self.bins[b]) & (self.ap < self.bins[b + 1])
            c.append(float(np.median(L[hit])) if hit.any() else np.nan)
        c = np.array(c)
        g = np.isfinite(c)
        self.mid, self.curve = mid[g], c[g]
        self.gain, self.log = {}, {}

    EMPTY = (None, None)

    def state(self, i):
        """what apply(i) left behind: a gate candidate's copy, as TakeInterior.state."""
        return self.log.get(int(i)), self.gain.get(i)

    def put(self, i, st):
        for d, k, v in zip((self.log, self.gain), (int(i), i), st):
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v

    def apply(self, i, rgb, a, ap):
        """the arch (BGR float, alpha) -> BGR, lifted to the take at aperture ap; a smoothed gain, so no flicker."""
        vis = a > 0.5
        if not len(self.curve) or vis.sum() < 12:
            return rgb
        lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        t = vis & teeth(lab)
        if t.sum() < 12:
            return rgb
        tgt = float(np.interp(ap, self.mid, self.curve))
        g = float(np.clip(tgt / max(1.0, float(lab[..., 0][t].mean())), 1.0, FIT_MAX))
        g = 0.8 * self.gain.get(i - 1, g) + 0.2 * g
        self.gain[i] = g
        self.log[int(i)] = dict(gain=round(g, 3), target=round(tgt * 100 / 255, 1))
        lab[..., 0] = np.clip(lab[..., 0] * g, 0, 255)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        return np.where(vis[..., None], out, rgb)


def seal_edge(a):
    """the arch's alpha tail: its column quads overlap in the half-transparent rows, and that double "over" is the comb under the
    biting edge. Averaged along the row, then a short ramp, the edge has no columns left."""
    s = cv2.GaussianBlur(a, (0, 0), 2.0, sigmaY=0.1)
    t = np.clip((s - 0.12) / 0.76, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def corner_guard(rgb5, outer, opening, io, lipc):
    """the take's dark corner crease the warp stretches into the lip band: those pixels take the lip colour around them.
    -> (rgb, guarded px)."""
    lab = cv2.cvtColor(np.clip(rgb5, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    band = outer & ~opening
    if band.sum() < 30:
        return rgb5, 0
    cols = np.nonzero(opening.any(0))[0]
    if not len(cols):
        return rgb5, 0
    span = max(4.0, 0.22 * (cols[-1] - cols[0]))
    xx = np.arange(rgb5.shape[1], dtype=np.float32)[None, :]
    corner = (xx < cols[0] + span) | (xx > cols[-1] - span)
    dark = band & corner & (lab[..., 0] < 0.62 * float(np.median(lab[..., 0][band])))
    if not dark.any():
        return rgb5, 0
    m = cv2.GaussianBlur(cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.0)[..., None]
    return rgb5 * (1 - m) + lipc * m, int(dark.sum())
