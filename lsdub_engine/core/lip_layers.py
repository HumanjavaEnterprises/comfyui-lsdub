#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lip_layers.py: v4 mouth layering helpers, shared by --mode warp and --mode morph (all CPU, opt-in by flags).

  SIZE     the new mouth is calibrated to the TAKE's own mouth: its rest width / lip thickness and its natural opening
           range per face. size_cap() keeps a target from going past the take's own p95 (per dimension) unless the
           take itself is already there; mouth_scale() scales a library mouth (paint) about its centre to the take's
           rest width / lip thickness.
  ERASE    "blank out the lips, then put ours in front": erase_lips() removes a face's lips from ONE image (the take's
           most-closed frame) and fills the hole with the take's own skin: a harmonic (Laplace) membrane from the hole's
           boundary (the frame's own skin shading, no smear streaks) plus the take's own fine skin texture (high-pass)
           lifted from the chin below. The bald frame is then carried into every frame by the head pose (the clean
           plate machinery), so the fill is one static texture that rides the skull (no boil), Lab-matched per frame.
  COLOUR   LipLock: one reference lip colour for the character. Per frame the lips' a*/b* are shifted to the reference
           (L follows the shot's lighting), inside a feathered lip mask weighted by per-pixel lip-ness (a chroma key
           between the frame's skin and lip colour, so the lip edge blends and skin / teeth / tongue are not tinted);
           the per-frame shift is eased zero-phase over time.
"""
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

LIP_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIP_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


# ----------------------------------------------------------------------------------------------------------- SIZE --
def mouth_dims(P, iod=None):
    """(width, open, height, thick) of one face's mouth from 478 landmarks P (x, y[, z]), in units of the inter-ocular
    distance (outer eye corners 33-263) unless iod is given. width 61-291, open 13-14 (inner gap), height 0-17 (outer),
    thick = upper lip 0-13 + lower lip 14-17."""
    P = np.asarray(P, float)[..., :2]
    d = lambda i, j: np.linalg.norm(P[..., i, :] - P[..., j, :], axis=-1)
    io = d(33, 263) if iod is None else iod
    return np.stack([d(61, 291) / io, d(13, 14) / io, d(0, 17) / io, (d(0, 13) + d(14, 17)) / io], -1)


def take_size(P_track, rest_q=20):
    """the take's own mouth scale from its track (T x 478 x 2|3): rest = median over its rest_q % most-closed frames,
    plus p5 / p95 per dimension. -> dict(rest=[w, o, h, t], p5=[...], p95=[...]) in inter-ocular units."""
    D = mouth_dims(P_track)
    o = D[:, 1]
    rest = o <= np.percentile(o, rest_q)
    return dict(rest=np.median(D[rest], 0), p5=np.percentile(D, 5, 0), p95=np.percentile(D, 95, 0))


def size_cap(target, own, hi, lo=None, sigma=1.0):
    """cap a per-frame target series (T,) at hi: never above max(hi, own) (the take's own frame is never shrunk by the
    cap: it IS the face's own mouth), and with lo given never below min(lo, own). The cut is eased zero-phase
    (gaussian sigma frames on the removed amount, max-filtered first so a peak is not let back through)."""
    from scipy.ndimage import maximum_filter1d
    def ease(x):
        if sigma <= 0 or not x.any(): return x
        return np.maximum(x, gaussian_filter1d(maximum_filter1d(x, 3, mode='nearest'), sigma, mode='nearest'))
    t = np.asarray(target, float).copy()
    own = np.asarray(own, float)
    top = np.maximum(hi, own)
    t = np.minimum(t - ease(np.maximum(0.0, t - top)), top)
    if lo is not None:
        bot = np.minimum(lo, own)
        t = np.maximum(t + ease(np.maximum(0.0, bot - t)), bot)
    return t


def mouth_scale(L, base, sx, sy, r_in=1.15, r_out=2.2):
    """scale a mouth shape about its centre: landmarks L (... x 478 x D, library space) and the library base face
    `base` (478 x >=2). Points within r_in x the lip half-extents (measured on the base) are scaled by (sx, sy) about
    the base mouth centre; the scale falls off smoothly (C2) to none at r_out, so the mesh never tears. The scale acts on
    each frame's own shape (L - c), so openings and widths shrink with the lips. Returns a scaled copy."""
    b = np.asarray(base, float)[:, :2]
    c = b[[13, 14, 61, 291]].mean(0)
    hx = max(1e-6, np.abs(b[LIP_OUT, 0] - c[0]).max())
    hy = max(1e-6, np.abs(b[LIP_OUT, 1] - c[1]).max())
    rho = np.sqrt(((b[:, 0] - c[0]) / hx) ** 2 + ((b[:, 1] - c[1]) / hy) ** 2)
    w = 1.0 - smootherstep((rho - r_in) / max(1e-6, r_out - r_in))
    L = np.array(L, float, copy=True)
    d = L[..., :2] - c
    L[..., 0] = L[..., 0] + w * (sx - 1.0) * d[..., 0]
    L[..., 1] = L[..., 1] + w * (sy - 1.0) * d[..., 1]
    return L


# ---------------------------------------------------------------------------------------------------------- ERASE --
def harmonic_fill(img, mask):
    """fill mask (bool H x W) of img (H x W x C float/uint8) with the harmonic (Laplace) interpolation of the pixels on
    its boundary: the smoothest surface that meets the surrounding skin exactly (a membrane; no streaks like Telea)."""
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import splu
    img = np.asarray(img, np.float64)
    out = img.copy()
    m = np.asarray(mask, bool).copy()
    m[0, :] = m[-1, :] = False
    m[:, 0] = m[:, -1] = False
    ys, xs = np.nonzero(m)
    if len(ys) == 0: return out
    idx = -np.ones(m.shape, int)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)
    A = lil_matrix((n, n))
    C = img.shape[2] if img.ndim == 3 else 1
    B = np.zeros((n, C))
    flat = img.reshape(img.shape[0], img.shape[1], C)
    for k, (y, x) in enumerate(zip(ys, xs)):
        A[k, k] = 4.0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j = idx[y + dy, x + dx]
            if j >= 0: A[k, j] = -1.0
            else: B[k] += flat[y + dy, x + dx]
    sol = splu(csr_matrix(A).tocsc()).solve(B)
    o = out.reshape(img.shape[0], img.shape[1], C)
    o[ys, xs] = sol
    return o.reshape(img.shape)


def corner_points(q, c, corner):
    """v11: the outer-lip polygon q (about its centre c) with its two mouth CORNERS pushed outward along the corner line
    by corner = (left px, right px), tapering to 0 toward the lips' middle (|u|^2, u = the position along the line)."""
    q = np.asarray(q, float)
    e = q[LIP_OUT.index(291)] - q[LIP_OUT.index(61)]
    L = float(np.linalg.norm(e))
    if L < 1e-6: return q
    e = e / L
    u = ((q - c) @ e) / (0.5 * L)
    g = np.where(u < 0, corner[0], corner[1]) * np.clip(np.abs(u), 0, 1) ** 2
    return q + (np.sign(u) * g)[:, None] * e


def lip_erase_mask(img, P, grow=0.22, key=True, lip_ab=None, skin_ab=None, corner=None):
    """bool mask of everything lip in img: the outer-lip hull of landmarks P grown by `grow` (fraction of its size,
    about its centre) + a few px, and (key) any lip-coloured pixel within a wider hull (the anti-aliased red edge and
    lipstick that sits just outside the landmark outline). v11 corner = (left px, right px): the hull is also grown at the
    two mouth corners (corner_points), as far as the plan moves them (mouth_layers' --warp-cover-corners)."""
    P = np.asarray(P, float)[:, :2]
    H, W = img.shape[:2]
    q = P[LIP_OUT]
    c = q.mean(0)
    if corner is not None and max(corner) > 0: q = corner_points(q, c, corner)
    def hull(g):
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(c + (q - c) * (1 + g)).astype(np.int32)), 1)
        return m > 0
    m = hull(grow)
    if key:
        wide = hull(grow + 0.25)
        lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lipm = hull(0.0)
        ring = wide & ~hull(grow + 0.1)
        if lipm.sum() > 20 and ring.sum() > 20:
            la, sa = np.median(lab[..., 1][lipm]), np.median(lab[..., 1][ring])
            if la - sa > 4: m |= wide & (lab[..., 1] > sa + 0.35 * (la - sa))
    if lip_ab is not None and skin_ab is not None:
        # v8: any lip-coloured pixel (vs the SHOT's lip / skin colour) within a wide hull joins the hole, so the membrane's
        # boundary never sits on lip and the fill never turns pink (a wide-open or lipsticked mouth outgrows the landmark hull)
        lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lp = lipness(lab, lip_ab, skin_ab) > 0.4
        wide = hull(grow + 0.6)
        add = (lp & wide).astype(np.uint8)
        add = cv2.morphologyEx(add, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nl, lbl, cc, _ = cv2.connectedComponentsWithStats(add)
        touch = set(np.unique(lbl[cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0])) - {0}
        m |= np.isin(lbl, list(touch))          # only blobs connected to the lips, not a rosy cheek
    k = max(3, int(round(0.04 * np.ptp(q[:, 0]))) | 1)
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def erase_lips(img, P, grow=0.22, texture=1.0, seed=7, source='grain', lip_ab=None, skin_ab=None, corner=None):
    """the face in img (BGR) with its lips removed: the lip region (lip_erase_mask) is filled with the harmonic membrane
    of the surrounding skin plus grain: source 'grain' (default) = static gaussian grain matched to the std of the
    take's own skin high-pass around the hole (the bald frame is ONE image carried by the pose, so the grain rides the
    skull, no boil); 'patch' = the take's own high-pass skin texture from the same-shaped patch below (chin) or above,
    creases clipped (it can still print a faint chin contour). Returns (bald image uint8, mask bool)."""
    from . import mouth_morph as mm
    img = np.asarray(img)
    m = lip_erase_mask(img, P, grow, lip_ab=lip_ab, skin_ab=skin_ab, corner=corner)
    ys, xs = np.nonzero(m)
    if len(ys) == 0: return img.copy(), m
    y0, y1, x0, x1 = max(0, ys.min() - 3), min(img.shape[0], ys.max() + 4), max(0, xs.min() - 3), min(img.shape[1], xs.max() + 4)
    sub = img[y0:y1, x0:x1].astype(np.float64)
    ms = m[y0:y1, x0:x1]
    fill = harmonic_fill(sub, ms)
    if texture > 0:
        f32 = img.astype(np.float32)
        hp = f32 - cv2.GaussianBlur(f32, (0, 0), 1.5)
        hh = y1 - y0
        tex = None
        skin = mm.skin_px(img)
        for dy in ((int(hh * 1.05), -int(hh * 1.05), int(hh * 0.8)) if source == 'patch' else ()):
            a0, a1 = y0 + dy, y1 + dy
            if a0 < 0 or a1 > img.shape[0]: continue
            if skin[a0:a1, x0:x1][ms].mean() > 0.7 and not lip_erase_mask(img, P, grow)[a0:a1, x0:x1][ms].any():
                tex = hp[a0:a1, x0:x1].astype(np.float64)
                break
        if tex is None:   # matched grain from the skin ring around the hole
            ring = cv2.dilate(ms.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~ms
            sd = float(hp[y0:y1, x0:x1][ring].std()) if ring.sum() > 20 else 1.0
            g = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, sub.shape[:2]).astype(np.float32), (0, 0), 0.6)
            tex = (g / (g.std() + 1e-6) * sd)[..., None] * np.ones((1, 1, 3))
        # only the fine grain travels: creases / contour edges in the source patch are clipped out (robust +-1.5 sd of
        # the hole's own surrounding skin texture), so no chin line is printed onto the philtrum
        ring = cv2.dilate(ms.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~ms
        rs = hp[y0:y1, x0:x1][ring] if ring.sum() > 20 else hp[y0:y1, x0:x1].reshape(-1, 3)
        k = 1.5 * 1.4826 * float(np.median(np.abs(rs - np.median(rs)))) + 1e-3
        tex = np.clip(tex, -k, k)
        fe = cv2.GaussianBlur(ms.astype(np.float32), (0, 0), 1.0)[..., None]
        fill = fill + texture * tex * fe
    out = img.copy()
    reg = out[y0:y1, x0:x1]
    reg[ms] = np.clip(fill[ms] + 0.5, 0, 255).astype(np.uint8)
    return out, m


# --------------------------------------------------------------------------------------------------------- COLOUR --
def parse_lip_ref(s):
    """'L,a,b' (OpenCV 8-bit Lab, a/b offset 128) -> np.array(3); a path to a PNG -> its lip-tissue mean (MediaPipe)."""
    try:
        v = [float(x) for x in str(s).split(',')]
        if len(v) == 3: return np.array(v)
    except ValueError: pass
    from . import mouth_morph as mm
    img = cv2.imread(s)
    return mm.lip_ref_stats(img)[:, 0]


def lipness(lab, lip_ab, skin_ab):
    """per-pixel 0..1: how lip-coloured a pixel is, by its (a*, b*) position along the line from the frame's skin colour
    to its lip colour (projection, smootherstep from 25 % to 75 % of the way), times a chroma floor (greys: teeth)."""
    v = np.asarray(lip_ab, float) - np.asarray(skin_ab, float)
    n2 = max(1e-6, float(v @ v))
    t = ((lab[..., 1] - skin_ab[0]) * v[0] + (lab[..., 2] - skin_ab[1]) * v[1]) / n2
    return smootherstep((t - 0.25) / 0.5).astype(np.float32)


def lip_masks(P, shape, grow=0.12):
    """(region, inner) for 478 landmarks P: the outer-lip hull grown by `grow`, feathered (float 0..1), and the inner-lip
    hull slightly shrunk (bool: teeth / tongue / cavity, never recoloured)."""
    from . import mouth_paint as mp_
    P = np.asarray(P, float)[:, :2]
    lw = np.linalg.norm(P[61] - P[291])
    reg = mp_.hull_mask(P[LIP_OUT], shape, grow, max(3, lw * 0.08))
    inner = mp_.hull_mask(P[LIP_IN], shape, -0.08, 3) > 0.5
    return reg, inner


def lip_stats(img, P):
    """(lip mean L, a, b; skin mean a, b) of one frame: lips = lip tissue (outer minus inner hull), skin = a ring just
    outside the lips, skin-coloured pixels only. None when too few pixels."""
    from . import mouth_morph as mm, mouth_paint as mp_
    P = np.asarray(P, float)[:, :2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    w = mm.lip_tissue_mask(P, img.shape)
    ring = (mp_.hull_mask(P[LIP_OUT], img.shape, 0.6, 3) > 0.5) & ~(mp_.hull_mask(P[LIP_OUT], img.shape, 0.25, 3) > 0.5) & mm.skin_px(img)
    if w.sum() < 30 or ring.sum() < 30: return None
    return np.array([lab[..., c][w].mean() for c in range(3)] + [lab[..., c][ring].mean() for c in (1, 2)])


def lip_fit(img, P, lip_ab, skin_ab):
    """0..1: how well landmarks P sit on img's lips = mean lip-ness (vs the SHOT's median lip / skin colour) over the lip
    tissue mask. A mistracked frame (mask on skin) scores low; layers that paste pixels at the landmarks skip it."""
    from . import mouth_morph as mm
    w = mm.lip_tissue_mask(np.asarray(P, float)[:, :2], img.shape)
    if w.sum() < 30: return 0.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(lipness(lab, lip_ab, skin_ab)[w].mean())


class LipLock:
    """one lip colour for the character across shots. ref = (L, a, b), OpenCV 8-bit Lab. a*/b* are locked fully to the
    reference (per-frame chroma gain + rotation about neutral that takes the lips' mean onto it, eased zero-phase over
    time, sigma frames); L is left to the shot's
    lighting (l_share 0) or moved toward the reference by l_share. The shift acts only on lip-coloured pixels inside the
    feathered lip region (lipness x region); inside the inner lip only clearly-lip pixels (teeth / tongue stay)."""

    def __init__(self, ref, sigma=4.0, l_share=0.0, grow=0.12, clip=3.0):
        self.ref = np.asarray(ref, float)
        self.clip = clip
        self.sigma, self.l_share, self.grow = sigma, l_share, grow
        self.st = None

    def fit(self, stats):
        """stats: per frame lip_stats() (or None). Fills gaps, eases the per-frame lip / skin colour zero-phase."""
        n = len(stats)
        ok = np.array([s is not None for s in stats])
        if not ok.any():
            self.st = None
            return self
        A = np.array([s if s is not None else np.full(5, np.nan) for s in stats], float)
        ii = np.arange(n)
        for c in range(5): A[:, c] = np.interp(ii, ii[ok], A[ok, c])
        # a mistracked frame (mask on skin) must not set its own lip colour: per-frame stats stay within +-clip of the
        # shot's median lip / skin colour (the take's lipstick does not change inside a shot)
        med = np.median(A[ok], 0)
        A = med + np.clip(A - med, -self.clip, self.clip)
        if self.sigma > 0: A = gaussian_filter1d(A, self.sigma, axis=0, mode='nearest')
        self.st = A
        return self

    def gain(self, i):
        """the lips' chroma map for frame i as a complex factor on (a*-128) + j(b*-128): it takes the frame's mean lip
        chroma onto the reference's. A gain + rotation about neutral keeps the lips' own hue relations (an additive
        a*/b* shift turns the lighter, less saturated lip pixels pink / purple)."""
        if self.st is None: return 1.0 + 0j
        s = self.st[min(i, len(self.st) - 1)]
        m = complex(s[1] - 128, s[2] - 128)
        r = complex(self.ref[1] - 128, self.ref[2] - 128)
        return r / m if abs(m) > 1e-3 else 1.0 + 0j

    def shift(self, i):
        """(dL, da, db) at a fully lip pixel of frame i's MEAN lip colour (for the log)."""
        if self.st is None: return np.zeros(3)
        s = self.st[min(i, len(self.st) - 1)]
        return np.array([self.l_share * (self.ref[0] - s[0]), self.ref[1] - s[1], self.ref[2] - s[2]])

    def weight(self, img, P, i):
        if self.st is None: return None
        s = self.st[min(i, len(self.st) - 1)]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        reg, inner = lip_masks(P, img.shape, self.grow)
        lp = lipness(lab, s[1:3], s[3:5])
        # inside the inner lip only clearly-lip pixels move (lp^3): teeth stay grey and the tongue keeps most of its pink,
        # but a closure's lips laid across the inner loop are still locked
        return (reg * np.where(inner, lp ** 3, lp)).astype(np.float32), lab

    def apply(self, img, P, i):
        """img (BGR uint8) with its lips moved to the reference colour; P = this frame's 478 landmarks (where the lips
        ARE in img). Returns (image, weight map)."""
        r = self.weight(img, P, i)
        if r is None: return img, None
        w, lab = r
        d = self.shift(i)
        gn = self.gain(i)
        if not (w > 0.002).any(): return img, w
        ys, xs = np.nonzero(w > 0.002)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = lab[y0:y1, x0:x1].copy()
        ww = w[y0:y1, x0:x1]
        z = (sub[..., 1] - 128) + 1j * (sub[..., 2] - 128)
        z = z * (1 + ww * (gn - 1))
        sub[..., 1], sub[..., 2] = z.real + 128, z.imag + 128
        sub[..., 0] += ww * d[0]
        out = img.copy()
        out[y0:y1, x0:x1] = cv2.cvtColor(np.clip(sub + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        keep = w[y0:y1, x0:x1] <= 0.002
        out[y0:y1, x0:x1][keep] = img[y0:y1, x0:x1][keep]   # bit-identical outside
        return out, w
