#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""teeth_layer.py: v5 rigid teeth for --mode warp (--warp-teeth rigid). All CPU.

Teeth are bone: they never stretch, squash or change size. Lips may move and resize; teeth may only move as the skull
and the jaw move.

  ANATOMY   upper teeth are fixed to the skull: placed by the strict rigid head pose only (RigidPlacement, the same pose
            the warp's falloff rides), i.e. no scaling beyond the pose's own (perspective) scale. Lower teeth ride the
            jaw: the skull pose + a rotation about a hinge near the condyle (behind and below the outer canthi in the
            canonical head frame, --warp-teeth-hinge), by the angle that brings the harvest frame's lower lip line onto the
            TARGET lower lip (the warped 14), measured along the head's down axis. A 3D rotation: no scaling.
  PLATES    harvested once per face from the take's own frames (the warp's donors / stretched pixels are no longer used
            for the interior): the upper arch from the frame showing the most upper teeth with the arches apart (sharp,
            near-frontal); the lower arch + the front of the tongue (a 0.12 IOD band above the lower lip line) from the
            frame with the clearest dark open bite. A plate is a source frame + its pose; it is rendered into frame i by
            ONE affine (frame h -> canonical plane -> frame i, + jaw for the lower), so it is resampled once.
  RENDER    per frame, behind the lips: the dark cavity (the take's own interior tone), the lower plate at skull + jaw,
            the upper plate at the skull; the warped lips occlude them (the target inner-lip contour is the mask, and
            in a band around it any non-lip tooth / cavity pixel is replaced too, lip pixels kept). Under the lower
            plate (the lower lip's inner surface) the warped take is kept with its teeth-like pixels filled from their
            neighbours.
  LIGHT     the plates' and cavity's L are scaled by a low-frequency gain: the frame's own mouth-surround brightness /
            the harvest frame's, eased zero-phase over time.

v6 (opt-in; founder: "sized right but floating around like loose dentures"). Measured on the v5 output by DIS flow:
the MediaPipe skull pose misses the real upper-lip-root skin motion by ~2x the flow floor and lags it 0.1-0.7 frame, so
pose-placed teeth swam against flow-true lips; the per-frame jaw from the noisy target lip jittered (15-21 direction
flips per shot). So:
  PIN       skin_pin / set_pin: both plates take the upper-lip root's own motion (philtrum + nose base DIS flow in the
            warped frames, high-passed onto the pose), one motion source for teeth and upper lip.
  JAW PLAN  plan_jaw / plan: one least-squares trajectory over the clip (target, 2nd/3rd-difference smoothness, speed and
            acceleration limits, holds on sustained vowels, lower plate hidden below the lower lip on near-closed frames).
  PLATES    tongue_drop (the tongue rides lower behind the lower teeth), throat (soft cavity shading on wide opens),
            mirror (missing arch columns from the mirror image, ends recede and feather; no rectangular cut), plates > 1
            (per-head-angle upper harvests, only blended when registration proves they are the same teeth).
"""
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

from . import lip_layers as ll

LIP_IN, LIP_OUT = ll.LIP_IN, ll.LIP_OUT


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def poly_mask(pts, shape, grow=0.0, px=0.0):
    """bool mask of the polygon pts (N x 2) scaled about its centroid by (1 + grow), then dilated (px > 0) or eroded."""
    H, W = shape[:2]
    p = np.asarray(pts, float)[:, :2]
    c = p.mean(0)
    q = c + (p - c) * (1 + grow)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    k = int(round(abs(px)))
    if k > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        m = cv2.dilate(m, ker) if px > 0 else cv2.erode(m, ker)
    return m > 0


def skin_ring_L(img, P):
    """mean L (OpenCV 8-bit Lab) of the skin ring around the lips of landmarks P: the frame's local light at the mouth."""
    from . import mouth_paint as mp_, mouth_morph as mm
    P = np.asarray(P, float)[:, :2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ring = (mp_.hull_mask(P[LIP_OUT], img.shape, 0.6, 3) > 0.5) & ~(mp_.hull_mask(P[LIP_OUT], img.shape, 0.25, 3) > 0.5)
    sk = ring & mm.skin_px(img)
    if sk.sum() < 30: sk = ring
    return float(lab[..., 0][sk].mean()) if sk.sum() else np.nan


def teeth_key(lab, skin_L, l_rel=0.95, a_max=142.0, b_max=158.0):
    """bool: teeth-coloured pixels (bright vs the shot's skin, low a* = not lip / tongue / gum)."""
    return (lab[..., 0] > l_rel * skin_L) & (lab[..., 1] < a_max) & (lab[..., 2] < b_max)


def lipness_map(img, lip_ab, skin_ab):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return ll.lipness(lab, lip_ab, skin_ab)


# ------------------------------------------------------------------------------------------------- geometry --
def head_axes(Cm):
    """canonical head axes from the median canonical face Cm (478 x 3): u lateral (33 -> 263), d down (nose bridge ->
    chin, orthogonalised), b = u x d (back, into the screen for MediaPipe z)."""
    u = Cm[263] - Cm[33]
    u = u / np.linalg.norm(u)
    d = Cm[152] - Cm[168]
    d = d - u * (d @ u)
    d = d / np.linalg.norm(d)
    return u, d, np.cross(u, d)


def rot_axis(u, th):
    """3x3 rotation by th (rad) about the unit axis u (Rodrigues)."""
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def hinge_point(Cm, hy=0.25, hz=0.6):
    """the jaw hinge (condyle) in the canonical frame: hy x IOD below and hz x IOD behind the mid-point of the outer
    canthi (33 / 263)."""
    u, d, b = head_axes(Cm)
    iod = np.linalg.norm(Cm[263, :2] - Cm[33, :2])
    return (Cm[33] + Cm[263]) / 2 + hy * iod * d + hz * iod * b


def jaw_angle(T, H, u, d, delta, iters=4):
    """angle th (rad) of the rotation about axis u through H that moves the point T by `delta` along d (canonical px;
    > 0 = down = opening). Exact for the vertical component (Newton on dy cos th + lever sin th = dy + delta)."""
    r = T - H
    dy = r @ d
    lev = (np.cross(u, r)) @ d
    if abs(lev) < 1e-6: return 0.0
    th = delta / lev
    for _ in range(iters):
        f = dy * np.cos(th) + lev * np.sin(th) - (dy + delta)
        fp = -dy * np.sin(th) + lev * np.cos(th)
        if abs(fp) < 1e-9: break
        th -= f / fp
    return float(th)


def plane_affine(pl, i, z0, R=None, H=None, ref=(0.0, 0.0), span=100.0):
    """2x3 affine: canonical (x, y) on the plane z = z0 -> frame i pixels, optionally after the 3D rotation R about the
    point H (the jaw). Built from three plane points through the rigid pose (exact: the map is affine)."""
    cx, cy = ref
    X = np.array([[cx, cy, z0], [cx + span, cy, z0], [cx, cy + span, z0]], float)
    Y = X if R is None else (X - H) @ R.T + H
    Q = np.asarray(pl.rigid(i, Y), float)
    L = np.stack([(Q[1] - Q[0]) / span, (Q[2] - Q[0]) / span], 1)
    return np.hstack([L, (Q[0] - L @ np.array([cx, cy]))[:, None]])


def compose(A, B):
    """affine A after B (2x3 each)."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def inv_aff(A):
    return cv2.invertAffineTransform(np.asarray(A, np.float64))


def shift_aff(A, c):
    """the affine A followed by a frame-px translation c."""
    A = np.array(A, np.float64)
    A[:, 2] += np.asarray(c, float)[:2]
    return A


# ------------------------------------------------------------------------------------------------- v6: pin --
ROOT = [2, 97, 326, 167, 393, 164, 165, 391]      # the upper-lip ROOT: philtrum + nose base (the upper teeth hang here)
NOSE = [1, 4, 5, 195, 197, 6, 45, 275, 48, 278, 64, 294, 98, 327]   # v12: the nose (tip, bridge, alae): bone + cartilage, no lip


def _hull_mask(P, idx, shape, k=0.9):
    q = np.asarray(P, float)[idx, :2]
    c = q.mean(0)
    q = c + (q - c) * k
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(q.astype(np.int32)), 1)
    return m > 0


def region_flow(a, b, mask, box, dis=None, fb=0.7):
    """median DIS flow a -> b (gray) over the forward-backward consistent pixels of mask, inside box (x0, y0, x1, y1);
    nan when fewer than 12 consistent pixels."""
    dis = dis or cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    x0, y0, x1, y1 = box
    A, B = np.ascontiguousarray(a[y0:y1, x0:x1]), np.ascontiguousarray(b[y0:y1, x0:x1])
    if A.shape[0] < 8 or A.shape[1] < 8: return np.full(2, np.nan)
    fw = dis.calc(A, B, None)
    bk = dis.calc(B, A, None)
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
    bw = cv2.remap(bk, xx + fw[..., 0], yy + fw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ok = (np.linalg.norm(fw + bw, axis=2) < fb) & mask[y0:y1, x0:x1]
    return np.median(fw[ok], 0) if ok.sum() >= 12 else np.full(2, np.nan)


def skin_pin(frames, S, pose_pts, ref, idx=ROOT, sigma=4.0, noise=0.5, iod=None):
    """per-frame 2D correction (frame px) that makes a pose-carried point move like the SKIN at idx actually moves.

    frames: the frames the teeth go into (BGR); S: their track (T x 478 x >=2, places the skin region); pose_pts: where
    the pose puts that skin region's centre per frame (T x 2). The skin is followed by DIS flow chained frame to frame
    from `ref` (forward-backward checked medians over the region), so the teeth and the lip root share ONE motion source.
    Chained flow random-walks at low frequency, the pose does not (it is fitted per frame), so: correction =
    high-pass(sigma frames) of (flow track - pose track) -- the pose keeps the slow placement, the skin gives every
    frame-to-frame step -- then a zero-phase ease (noise frames) for the flow's own noise. sigma <= 0: no high-pass."""
    n = len(frames)
    G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    H, W = G[0].shape
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    pose_pts = np.asarray(pose_pts, float)
    if iod is None: iod = np.linalg.norm(np.asarray(S)[:, 33, :2] - np.asarray(S)[:, 263, :2], axis=1)
    def box(t):
        c = np.asarray(S[t], float)[idx, :2].mean(0)
        r = 0.5 * float(np.atleast_1d(iod)[min(t, len(np.atleast_1d(iod)) - 1)])
        return (max(0, int(c[0] - r)), max(0, int(c[1] - r)), min(W, int(c[0] + r)), min(H, int(c[1] + r)))
    def step(t, t2):
        f = region_flow(G[t], G[t2], _hull_mask(S[t], idx, G[t].shape), box(t), dis)
        return f if np.isfinite(f).all() else (pose_pts[t2] - pose_pts[t])     # unmeasured: the pose's own step
    tr = np.zeros((n, 2))
    tr[ref] = pose_pts[ref]
    for t in range(ref, n - 1): tr[t + 1] = tr[t] + step(t, t + 1)
    for t in range(ref, 0, -1): tr[t - 1] = tr[t] + step(t, t - 1)
    c = tr - pose_pts
    if sigma > 0: c = c - gaussian_filter1d(c, sigma, axis=0, mode='nearest')
    if noise > 0: c = gaussian_filter1d(c, noise, axis=0, mode='nearest')
    return c


# --------------------------------------------------------------------------------------------- v6: jaw plan --
def plan_jaw(th_t, w=None, hide=None, hide_w=None, hold=None, lead=0.0, vmax=np.radians(3.0), amax=np.radians(1.0),
             lam_acc=1.5, lam_jerk=0.65, lam_hold=2.0, mu=30.0, lim=(-25.0, 30.0), floor=None):
    """the jaw trajectory over the WHOLE clip, solved once (least squares), not per frame. Radians per frame.

    th_t: per-frame target angle (from the target lower lip; noisy); w: its confidence (0..1); lead: frames of extra
    anticipation (the target is read `lead` frames ahead; the target already carries the lips' lead, so 0 = the jaw
    leads the audio exactly as the lips do); terms:
      target   sqrt(w) (th - th_t(i + lead))
      smooth   lam_acc * second difference + lam_jerk * third difference (zero-phase by construction: no lag)
      hold     lam_hold * first difference on `hold` frames (a sustained vowel: the jaw rests)
      hide     mu * hide_w * max(0, hide - th): on near-closed frames the lower teeth sit hidden below the lower lip
               (hide = the angle that puts the lower incisal edge at the target lower lip line)
      limits   mu * max(0, |d th| - vmax), mu * max(0, |d2 th| - amax): a real jaw's speed / acceleration
      floor    (v7, optional; scalar or per frame) mu * max(0, floor - th): the bite (lower incisors never rise through
               the upper ones) and the exposure limit (never more than a sliver above the target lower lip)
    Returns th (n,), clipped to lim degrees."""
    from scipy.optimize import least_squares
    th_t = np.asarray(th_t, float)
    n = len(th_t)
    if n < 4: return np.clip(th_t, *np.radians(lim))
    w = np.ones(n) if w is None else np.clip(np.asarray(w, float), 0, 1)
    if lead: th_t = np.interp(np.arange(n) + lead, np.arange(n), th_t)
    hide = None if hide is None else np.asarray(hide, float)
    hide_w = np.zeros(n) if hide_w is None else np.asarray(hide_w, float)
    hold = np.zeros(n, bool) if hold is None else np.asarray(hold, bool)
    sw = np.sqrt(np.maximum(w, 1e-3))
    hm = hold[1:] & hold[:-1]
    def res(x):
        d1, d2, d3 = np.diff(x), np.diff(x, 2), np.diff(x, 3)
        r = [sw * (x - th_t), lam_acc * d2, lam_jerk * d3, lam_hold * d1 * hm,
             mu * np.maximum(0, np.abs(d1) - vmax), mu * np.maximum(0, np.abs(d2) - amax)]
        if hide is not None: r.append(mu * hide_w * np.maximum(0, hide - x))
        if floor is not None: r.append(mu * np.maximum(0, np.asarray(floor, float) - x))
        return np.concatenate(r)
    x0 = gaussian_filter1d(th_t if hide is None else np.where(hide_w > 0.5, np.maximum(th_t, hide), th_t), 1.0, mode='nearest')
    if floor is not None: x0 = np.maximum(x0, floor)
    x = least_squares(res, x0, method='trf', x_scale=0.05, max_nfev=4000).x
    return np.clip(x, *np.radians(lim))


def sustained(vis, vowels=('aa', 'E', 'ih', 'oh', 'ou')):
    """bool per frame: a vowel held over this frame and both neighbours."""
    v = list(vis)
    n = len(v)
    out = np.zeros(n, bool)
    for i in range(1, n - 1): out[i] = v[i] in vowels and v[i - 1] == v[i] == v[i + 1]
    return out


# ---------------------------------------------------------------------------------------------- the layer --
def arch_split(t, inner, top_tol, tol=2, dark=None):
    """split the teeth pixels t (bool) inside the inner-lip region `inner` (bool) into the UPPER arch, per column: the
    first run of teeth (gaps <= tol px bridged) that starts within top_tol px of the region's top in that column (the
    upper teeth hang from the upper lip). Returns (upper mask = region top .. the run's end, i.e. teeth + the dark gaps
    between them; per-column length of the non-teeth run below the arch (nan: no upper teeth in that column))."""
    H, W = t.shape
    up = np.zeros_like(t)
    gap = np.full(W, np.nan)
    for x in np.nonzero(inner.any(0))[0]:
        ys = np.nonzero(inner[:, x])[0]
        y0, y1 = ys.min(), ys.max()
        col = t[y0:y1 + 1, x]
        tt = np.nonzero(col)[0]
        if len(tt) == 0 or tt[0] > top_tol: continue
        e = tt[0]
        for k in tt[1:]:
            if k - e > tol + 1: break
            e = k
        up[y0:y0 + e + 1, x] = True
        nxt = tt[tt > e]
        b = nxt[0] if len(nxt) else len(col)
        gap[x] = (b - e - 1) if dark is None else int(dark[y0 + e + 1:y0 + b, x].sum())
    return up, gap


class TeethLayer:
    """rigid teeth per face. frames: the take's frames (BGR), S: its smoothed track (T x 478 x 3, frame px), pl: the
    strict rigid pose (RigidPlacement, canonical -> frame), C: the canonical track (T x 478 x 3), F: features (gap,
    width) per frame (canonical px). hinge = (hy, hz) in IOD (see hinge_point); jaw_gain: lower-teeth drop per unit of
    target gap change; fit: per-frame lip-fit (0..1) of the track (mistracked frames are never harvested)."""

    def __init__(self, frames, S, pl, C, F, hinge=(0.25, 0.6), jaw_gain=1.0, fit=None, light_sigma=2.0, up_idx=None, lo_idx=None, lo_band=0.12,
                 plates=1, plate_min_deg=12.0, tongue_drop=0.0, throat=0.0, mirror=False, build='v11'):
        self.frames, self.S, self.pl, self.C, self.F = frames, S, pl, C, F
        n = len(frames)
        self.lo_band = lo_band
        self.tongue_drop, self.throat = tongue_drop, throat
        self.v6_plate = mirror
        self.pin = np.zeros((n, 2))
        self.jaw_plan = None
        self.plate_w = None
        self.g_hi = float(np.percentile(F[:, 0], 95))
        self.Cm = np.median(C, 0)
        self.iod = float(np.linalg.norm(self.Cm[263, :2] - self.Cm[33, :2]))
        self.u, self.d, self.b = head_axes(self.Cm)
        self.H = hinge_point(self.Cm, *hinge)
        self.jaw_gain = jaw_gain
        self.z_up = float(self.Cm[13, 2] + 0.05 * self.iod)
        self.z_lo = float(self.Cm[14, 2] + 0.05 * self.iod)
        self.ref = tuple(self.Cm[[13, 14], :2].mean(0))
        self.skinL = np.array([skin_ring_L(frames[i], S[i]) for i in range(n)])
        sL = np.where(np.isfinite(self.skinL), self.skinL, np.nanmedian(self.skinL))
        self.skin_med = float(np.median(sL))
        self.light = gaussian_filter1d(sL, light_sigma, mode='nearest') if light_sigma > 0 else sL
        fit = np.ones(n) if fit is None else np.asarray(fit, float)
        self.stats = [self.frame_teeth(i) for i in range(n)]
        pose = pl.par[:, 1:4]
        dev = np.linalg.norm(pose - np.median(pose, 0), axis=1)
        front = np.exp(-(dev / 0.25) ** 2)
        sharp = np.array([s['sharp'] for s in self.stats])
        sharp = np.clip(sharp / max(1e-6, np.median(sharp[sharp > 0]) if (sharp > 0).any() else 1), 0, 2)
        ok = (fit > 0.7) if (fit > 0.7).any() else np.ones(n, bool)
        sc = pl.s ** 2
        # an OPEN bite only: a dark run of >= ~0.05 IOD below the upper arch (a clenched / near-closed bite would carry
        # both arches on one plate; the lip gap alone does not tell, a smile parts the lips over clenched teeth)
        apart = np.array([s['apart'] for s in self.stats])
        okl = ok * 1.0
        ok = ok * smootherstep((apart - 0.02) / 0.05)
        apart_d = np.array([s['apart_dark'] for s in self.stats])
        up = np.array([s['up_area'] for s in self.stats]) / sc * front * np.sqrt(sharp) * ok
        # the lower arch + tongue ride the jaw from the frame with the clearest open bite: the largest dark run below the
        # upper arch (a tongue resting high against the upper teeth would ride up in front of the cavity), then its
        # lower teeth / interior
        inn = np.array([s['in_area'] for s in self.stats]) / sc
        lo = apart_d ** 2 * front * np.sqrt(sharp) * okl * (1 + np.array([s['lo_area'] for s in self.stats]) / sc / np.maximum(inn, 1))
        self.h_up = int(np.argmax(up)) if up_idx is None else int(up_idx)
        self.h_lo = int(np.argmax(lo)) if lo_idx is None else int(lo_idx)
        self._up_score, self._sharp, self._ok = up / np.maximum(front, 1e-6), sharp, ok
        self.build, self.build_log = build, None
        if build == 'v12' and up_idx is None: self._harvest_v12(front, sharp, okl, apart_d)     # v12: no clenched harvest
        self._build_plates()
        self.cavity = self._cavity_colour()
        if plates > 1: self._pose_plates(plates, plate_min_deg)

    def exposed_up(self, i):
        """v12: the exposed upper tooth height of frame i (IOD): per central column (|x - the mouth centre| <= 0.12 IOD) the
        upper arch's run from the opening's top to its incisal edge (arch_split: the teeth + the dark gaps between them),
        the median over the columns that have one (0 when fewer than 3 do)."""
        img = self.frames[i]
        P = np.asarray(self.S[i], float)[:, :2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        inner = poly_mask(P[LIP_IN], img.shape, 0.05)
        io = self.iod * self.pl.s[i]
        upm, _ = arch_split(inner & teeth_key(lab, self.skin_med), inner, 0.05 * io)
        cx = P[[13, 14], 0].mean()
        cols = np.arange(int(cx - 0.12 * io), int(cx + 0.12 * io) + 1)
        cols = cols[(cols >= 0) & (cols < img.shape[1])]
        hh = upm[:, cols].sum(0)
        hh = hh[hh > 0]
        return float(np.median(hh)) / io if len(hh) >= 3 else 0.0

    def _harvest_v12(self, front, sharp, okl, apart_d):
        """v12 build (--warp-arch-build v12): the upper plate is harvested from the frame with the most EXPOSED upper tooth
        height (exposed_up) whose arches are clearly APART (a DARK run of >= ~0.04 IOD below the upper incisal edge,
        stats apart_dark), near the median head pose and sharp -- not the largest tooth area: a smile over clenched teeth
        has a large area but carries the lower row on the same plate under a dark bite line, so the plate showed two
        stacked rows of incisors ('teeth inside teeth', mv30 S3 s1234 f25). The extra harvest frames (_upper_candidates)
        are scored the same way. Logged: build_log."""
        n = len(self.frames)
        h = np.array([self.exposed_up(i) for i in range(n)])
        gap = smootherstep((apart_d - 0.02) / 0.04)
        sc_ = h * gap * front * np.sqrt(sharp) * okl
        old = int(self.h_up)
        if sc_.max() <= 0:
            self.build_log = dict(up=dict(h_up=old, fallback='v11 (no frame shows the upper teeth with the arches apart)'))
            return
        self.h_up = int(np.argmax(sc_))
        self._up_score = h * np.sqrt(sharp)
        self._ok = okl * gap
        r4 = lambda x: round(float(x), 4)
        self.build_log = dict(up=dict(h_up=self.h_up, exposed_iod=r4(h[self.h_up]), apart_dark_iod=r4(apart_d[self.h_up]), v11_h_up=old,
                                      v11_exposed_iod=r4(h[old]), v11_apart_dark_iod=r4(apart_d[old]),
                                      top=[dict(i=int(i), exposed=r4(h[i]), apart_dark=r4(apart_d[i]), score=r4(sc_[i])) for i in np.argsort(-sc_)[:5]]),
                              exposed_max_iod=r4(h[gap > 0.5].max()) if (gap > 0.5).any() else 0.0)

    # per-frame teeth stats in the frame itself (harvest choice + the measurement tool)
    def frame_teeth(self, i, img=None, P=None):
        img = self.frames[i] if img is None else img
        P = self.S[i] if P is None else P
        P = np.asarray(P, float)[:, :2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        inner = poly_mask(P[LIP_IN], img.shape, 0.05)
        t = inner & teeth_key(lab, self.skin_med)
        io = self.iod * self.pl.s[i]
        upm, _ = arch_split(t, inner, 0.05 * io)
        up = upm & t
        lo = t & ~upm
        # arches apart: below the upper arch, the DARK (cavity) pixels inside the SHRUNK inner contour (no lip pixels: a
        # clenched smile parts the lips but its arches touch -> none; a tongue raised against the upper teeth -> few)
        shr = poly_mask(P[LIP_IN], img.shape, -0.08)
        cx = P[[13, 14], 0].mean()
        cols = np.arange(int(cx - 0.12 * io), int(cx + 0.12 * io))
        cols = cols[(cols >= 0) & (cols < img.shape[1])]
        def run(dark):
            _, gap = arch_split(t, shr, 0.08 * io, dark=dark)
            gc = gap[cols]
            gc = gc[np.isfinite(gc)]
            return float(np.median(gc)) / io if len(gc) >= 3 else 0.0
        apart = run(None)                                                     # any non-teeth run (upper arch gate)
        apart_dark = run(lab[..., 0] < 0.6 * self.skin_med)                   # dark (cavity) run (lower arch gate)
        g = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
        sharp = float(g[t].var()) if t.sum() > 30 else 0.0
        return dict(up_area=float(up.sum()), lo_area=float(lo.sum()), in_area=float(inner.sum()), sharp=sharp, apart=apart, apart_dark=apart_dark)

    def _region(self, h):
        img = self.frames[h]
        P = self.S[h][:, :2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        pin = poly_mask(P[LIP_IN], img.shape, 0.05)
        io = self.iod * self.pl.s[h]
        t = pin & teeth_key(lab, self.skin_med, 0.8)
        t = cv2.morphologyEx(t.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        topy = np.where(pin.any(0), pin.argmax(0), 10 ** 6)
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(t.astype(np.uint8))
        ku = np.zeros_like(t)
        for k in range(1, nl):
            if st[k, cv2.CC_STAT_AREA] < max(12, 0.02 * t.sum()): continue
            ys, xs = np.nonzero(lbl == k)
            if np.median([ys[xs == x].min() - topy[x] for x in np.unique(xs)]) <= 0.08 * io: ku |= lbl == k
        upper = np.zeros_like(t)
        for x in np.nonzero(ku.any(0))[0]: upper[:np.nonzero(ku[:, x])[0].max() + 1, x] = True
        rows = np.arange(img.shape[0])[:, None]
        ramp = np.clip((rows - topy[None, :]) / max(1.0, 0.025 * io), 0, 1).astype(np.float32)
        return img, pin, upper & pin, ramp, topy, io

    def _build_plates(self):
        """per arch, from its harvest frame. Teeth = bright, low-a* pixels (shaded lateral teeth included) inside the
        inner-lip contour; a component belongs to the upper arch when it hangs from the upper lip (its top within 0.08 IOD
        of the contour's top in its own columns).
        Upper plate: per column from the contour's top down to the upper arch's lowest pixel (the incisal edge), so the
        teeth AND the dark gaps between them travel together; its top (the harvest frame's lip line) fades over 0.025 IOD.
        Lower plate: everything inside the contour below the upper arch (lower teeth, tongue: they ride the jaw), minus
        the harvest frame's upper-lip line; its own dark cavity is not carried (alpha ramps in with lightness from the
        region's darkest tenth to its 40th percentile), so the dark space above it is the shared cavity tone."""
        region = self._region
        f = max(1.0, 0.006 * self.iod)
        self.up_img, self.up_alpha = self._upper_plate(self.h_up)
        img_l, pin_l, upper_l, _, topy_l, io_l = region(self.h_lo)
        kd = max(1, int(round(0.03 * io_l)))                                    # no grey tooth fringe on the lower plate
        up_d = cv2.dilate(upper_l.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd + 1, 2 * kd + 1))) > 0
        lo = pin_l & ~up_d & (np.arange(img_l.shape[0])[:, None] >= (topy_l + int(0.05 * io_l))[None, :])
        # no lip tissue rides the jaw: the region's sides are pulled in 0.03 IOD (the lip corners) and in its edge band
        # (0.06 IOD) lip-coloured pixels are dropped (they would print as red strands inside the target mouth)
        kx = max(1, int(round(0.03 * io_l)))
        lo = cv2.erode(lo.astype(np.uint8), np.ones((1, 2 * kx + 1), np.uint8)) > 0
        st = ll.lip_stats(img_l, self.S[self.h_lo][:, :2])
        if st is not None:
            ke = max(1, int(round(0.06 * io_l)))
            band = pin_l & ~(cv2.erode(pin_l.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ke + 1, 2 * ke + 1))) > 0)
            lp = lipness_map(img_l, st[1:3], st[3:5])
            lo &= ~(band & (lp > 0.5) & ~teeth_key(cv2.cvtColor(img_l, cv2.COLOR_BGR2LAB), self.skin_med, 0.8))
        geo = cv2.GaussianBlur(lo.astype(np.float32), (0, 0), 2.5 * f) * (1 - cv2.GaussianBlur(upper_l.astype(np.float32), (0, 0), 2.5 * f))
        # only the band just above the lower lip line rides the jaw (lower teeth + the tongue's front, <= lo_band IOD): a
        # wide-open harvest frame's tongue reaches up to the upper teeth and would ride up in front of the cavity
        boty = np.where(pin_l.any(0), img_l.shape[0] - 1 - pin_l[::-1].argmax(0), -1)
        hgt = (boty[None, :] - np.arange(img_l.shape[0])[:, None]) / max(1.0, io_l)
        geo = geo * (1 - smootherstep((hgt - self.lo_band) / 0.03))
        Ll = cv2.GaussianBlur(cv2.cvtColor(img_l, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), f)
        Lc = float(np.percentile(Ll[lo], 10)) if lo.sum() > 20 else 0.0
        Lt = float(np.percentile(Ll[lo], 40)) if lo.sum() > 20 else 1.0
        lit = smootherstep((Ll - Lc) / max(8.0, Lt - Lc)).astype(np.float32)
        if self.tongue_drop > 0:
            # v6 lower tongue: the plate's non-teeth part (the tongue's front) is carried tongue_drop IOD DOWN behind the
            # lower teeth (which stay); the space it leaves is the cavity, so the tongue no longer rides up toward the
            # upper teeth / shows in a near-closed slit
            tk = cv2.GaussianBlur((teeth_key(cv2.cvtColor(img_l, cv2.COLOR_BGR2LAB), self.skin_med, 0.8) & lo).astype(np.float32), (0, 0), f)
            tk = np.clip(tk * 1.5, 0, 1)
            dy = self.tongue_drop * io_l
            M = np.float32([[1, 0, 0], [0, 1, dy]])
            sz = img_l.shape[1::-1]
            img_s = cv2.warpAffine(img_l, M, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            g_s = cv2.warpAffine(geo * (1 - tk), M, sz, flags=cv2.INTER_LINEAR, borderValue=0)
            lit_s = cv2.warpAffine(lit, M, sz, flags=cv2.INTER_LINEAR, borderValue=0)
            g_t = geo * tk
            geo2 = g_t + (1 - g_t) * g_s
            img_l = np.clip((img_l.astype(np.float32) * g_t[..., None] + img_s.astype(np.float32) * ((1 - g_t) * g_s)[..., None])
                            / np.maximum(geo2, 1e-3)[..., None], 0, 255).astype(np.uint8)
            lit = ((lit * g_t + lit_s * (1 - g_t) * g_s) / np.maximum(geo2, 1e-3)).astype(np.float32)
            geo = geo2 * (boty[None, :] >= np.arange(img_l.shape[0])[:, None])          # nothing below the harvest lip line
        if self.v6_plate:
            # v6: the lower plate's lateral ends feather out (no straight-edged cut where it stops; the columns it does not
            # reach are cavity). Not mirrored: a mirrored tongue prints its shading creases as dark V lines
            cov = geo.max(0) > 0.3
            if cov.any():
                xs = np.arange(len(cov))
                l0, l1 = np.nonzero(cov)[0][[0, -1]]
                dist = np.minimum(xs - l0, l1 - xs).astype(np.float32)
                geo = geo * smootherstep(dist / max(1.0, 0.05 * io_l))[None, :]
        self.lo_img = img_l
        self.lo_geo = np.clip(geo, 0, 1).astype(np.float32)
        self.lo_lit = lit.astype(np.float32)
        self.A_up_h = plane_affine(self.pl, self.h_up, self.z_up, ref=self.ref)
        self.A_lo_h = plane_affine(self.pl, self.h_lo, self.z_lo, ref=self.ref)
        self.T_lo = self.C[self.h_lo, 14].copy()
        self.up_plates = [dict(h=self.h_up, img=self.up_img, alpha=self.up_alpha, A_h=self.A_up_h)]
        # how far the lower plate's top sits ABOVE the harvest lower lip line (14), along the head's down axis (canonical
        # px): the extra drop that hides the whole lower plate below the target lower lip (the near-closed constraint)
        ys, xs = np.nonzero(self.lo_geo > 0.5)
        if len(ys):
            Ai = inv_aff(self.A_lo_h)
            q = np.stack([xs, ys], 1) @ Ai[:, :2].T + Ai[:, 2]
            dd = self.d[:2] / max(1e-6, np.linalg.norm(self.d[:2]))
            self.e_lo = float(max(0.0, np.percentile((self.T_lo[:2] - q) @ dd, 98)))
        else: self.e_lo = 0.0

    def _upper_plate(self, h, mirror=None):
        """the upper plate of harvest frame h: (image, alpha). v6 (mirror; --warp-teeth-mirror): columns the harvest frame does not show (the far side of a turned head, a lip corner) are filled from
        the arch's mirror image about the face midline on the canonical plane (teeth are symmetric), and the arch's
        lateral ends recede into shadow and feather out (no rectangular cut where the plate ends)."""
        img_u, _, upper, ramp, _, io = self._region(h)
        f = max(1.0, 0.006 * self.iod)
        al = (cv2.GaussianBlur(upper.astype(np.float32), (0, 0), f) * ramp).astype(np.float32)
        if mirror is None: mirror = self.v6_plate
        if not mirror or al.max() <= 0: return img_u, al
        A_h = plane_affine(self.pl, h, self.z_up, ref=self.ref)
        xm = float(self.ref[0])
        Mm = compose(A_h, compose(np.array([[-1.0, 0, 2 * xm], [0, 1.0, 0]]), inv_aff(A_h)))
        sz = img_u.shape[1::-1]
        img_m = cv2.warpAffine(img_u, Mm, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        al_m = cv2.warpAffine(al, Mm, sz, flags=cv2.INTER_LINEAR, borderValue=0)
        own = cv2.GaussianBlur((al.max(0) > 0.1).astype(np.float32)[None, :], (0, 0), max(1.0, 0.01 * io))[0]
        add = al_m * (1 - own)[None, :]
        a2 = al + (1 - al) * add
        img = (img_u.astype(np.float32) * al[..., None] + img_m.astype(np.float32) * ((1 - al) * add)[..., None]) / np.maximum(a2, 1e-4)[..., None]
        cov = a2.max(0) > 0.3
        xs = np.arange(len(cov))
        if cov.any():
            l0, l1 = np.nonzero(cov)[0][[0, -1]]
            dist = np.minimum(xs - l0, l1 - xs).astype(np.float32)
            shade = 0.6 + 0.4 * smootherstep(dist / max(1.0, 0.1 * io))
            fade = smootherstep(dist / max(1.0, 0.03 * io))
            img = img * shade[None, :, None]
            a2 = a2 * fade[None, :]
        return np.clip(img, 0, 255).astype(np.uint8), a2.astype(np.float32)

    def _pose_plates(self, k, min_deg=12.0, sigma_t=2.0):
        """v6 per-head-angle upper plates: a frontal harvest cannot show what a far-turned head shows. Over the head's
        out-of-plane pose (pitch, yaw; roll is in the image plane and the affine carries it), up to k harvests: h_up,
        then greedily the best-scoring open-bite frame >= min_deg from the plates already chosen. Each is registered to
        the main plate on the canonical plane (phase correlation of the teeth) so all share one position and size; per
        frame they are blended by a gaussian in pose distance (weights eased over time), never swapped."""
        op = self.pl.par[:, 1:3]
        len(op)
        sc = self._up_score * self._ok
        s0 = max(1e-9, sc[self.h_up])
        md = np.radians(min_deg)
        self.plate_dbg = [dict(pose_range_deg=np.round(np.degrees(np.ptp(op, 0)), 1).tolist())]
        hs = [self.h_up]
        for _ in range(k - 1):
            dist = np.min([np.linalg.norm(op - op[h], axis=1) for h in hs], 0)
            need = (dist >= md).mean()                                           # share of frames far from every plate
            s_ = sc / s0 * np.minimum(1.0, dist / (2 * md)) * (dist >= md)
            j = int(np.argmax(s_))
            self.plate_dbg.append(dict(best=j, score=round(float(s_[j]), 3), far_share=round(float(need), 2),
                                                               pose_deg=np.round(np.degrees(op[j]), 1).tolist()))
            if s_[j] < 0.03 or need < 0.08: break
            hs.append(j)
        if len(hs) < 2: return
        # register each extra plate to the main one on the canonical plane
        io = self.iod
        c = np.array(self.ref)
        x0, y0 = c - [0.8 * io, 0.5 * io]
        w, hgt = int(1.6 * io), int(1.0 * io)
        def canon(img, alpha, A_h):
            M = compose(inv_aff(A_h), np.eye(3)[:2])
            M = compose(np.array([[1, 0, -x0], [0, 1, -y0]], float), M)
            L = cv2.cvtColor(cv2.warpAffine(img, M, (w, hgt), flags=cv2.INTER_LINEAR), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            a = cv2.warpAffine(alpha, M, (w, hgt), flags=cv2.INTER_LINEAR)
            return L * a, a
        L0, a0 = canon(self.up_img, self.up_alpha, self.A_up_h)
        win = cv2.createHanningWindow((w, hgt), cv2.CV_32F)
        for h in hs[1:]:
            img, al = self._upper_plate(h)
            A_h = plane_affine(self.pl, h, self.z_up, ref=self.ref)
            L1, a1 = canon(img, al, A_h)
            (sx, sy), resp = cv2.phaseCorrelate(L0, L1, win)                    # L1 = L0 shifted by (sx, sy)
            ok_s = abs(sx) < 0.06 * io and abs(sy) < 0.06 * io
            M = np.float32([[1, 0, -sx], [0, 1, -sy]]) if ok_s else np.float32([[1, 0, 0], [0, 1, 0]])
            L1s = cv2.warpAffine(L1, M, (w, hgt))
            a1s = cv2.warpAffine(a1, M, (w, hgt))
            ov = (a0 > 0.5) & (a1s > 0.5)
            ncc = float(np.corrcoef(L0[ov], L1s[ov])[0, 1]) if ov.sum() > 50 else -1.0
            rec = dict(h=h, shift=(round(float(sx), 2), round(float(sy), 2)), ncc=round(ncc, 3))
            # a second harvest is only blended when it IS the same teeth after registration (the take regenerates its
            # teeth per frame; two different arches cross-faded read as grey ghost teeth)
            if not ok_s or ncc < 0.6:
                self.plate_dbg.append(dict(rejected=rec))
                continue
            A_h = compose(A_h, np.array([[1, 0, sx], [0, 1, sy]], float))      # canonical - (sx, sy): onto the main plate
            self.up_plates.append(dict(img=img, alpha=al, A_h=A_h, **rec))
        if len(self.up_plates) < 2: return
        Pk = np.array([op[p['h']] for p in self.up_plates])
        sp = 0.6 * md
        D = np.linalg.norm(op[:, None, :] - Pk[None], axis=2)
        Wt = np.exp(-0.5 * (D / sp) ** 2) + 1e-6
        Wt = Wt / Wt.sum(1, keepdims=True)
        if sigma_t > 0:
            Wt = gaussian_filter1d(Wt, sigma_t, axis=0, mode='nearest')
            Wt = Wt / Wt.sum(1, keepdims=True)
        self.plate_w = Wt

    # ---------------------------------------------------------------------------------------------- v6 anchoring --
    def set_pin(self, frames, S=None, sigma=4.0, noise=0.5, idx=None):
        """teeth share the upper-lip root's motion: the per-frame correction (frame px) from the DIS flow of the root
        skin (philtrum + nose base) in `frames` (the warped output, before the teeth). Applied to both plates (the lower
        through the jaw hinge), so nothing moves relative to the upper lip root except by the jaw. idx (v12): another
        skin region (e.g. NOSE: the head's own fast motion the skull pose misses, without the lip's)."""
        idx = ROOT if idx is None else list(idx)
        S = self.S if S is None else S
        X = self.Cm[idx]
        pp = np.array([self.pl.rigid(i, X).mean(0) for i in range(len(frames))])
        iod = self.iod * self.pl.s[:len(frames)]
        self.pin = skin_pin(frames, S, pp, self.h_up, idx, sigma, noise, iod)
        return self.pin

    def hide_angle(self, i, p14):
        """jaw angle (rad) that puts the lower plate's top at the target lower lip line (all of it hidden behind it)."""
        return self.jaw(None, i, p14, lim=(-60, 60), extra=self.e_lo + 0.01 * self.iod)

    def plan(self, P14, open_u, vis=None, close_thr=0.2, lead=0.0, vmax=3.0, amax=1.0, lam_acc=1.5, lam_jerk=0.65, w=None, floor=None):
        """the whole clip's jaw at once (plan_jaw). P14: the target lower lip per frame (frame px, where the warped lower
        lip is); open_u: the target opening per frame in the take's range (0 shut .. 1 its p95); vis: viseme per frame
        (sustained vowels hold). Near-closed frames (open_u below close_thr) keep the lower plate hidden."""
        n = len(P14)
        th = np.array([self.jaw(None, i, P14[i], lim=(-60, 60)) for i in range(n)])
        hide = np.array([self.hide_angle(i, P14[i]) for i in range(n)]) if close_thr > 0 else None
        hw = smootherstep((close_thr - np.asarray(open_u, float)) / max(1e-6, 0.5 * close_thr)) if close_thr > 0 else None
        hold = sustained(vis) if vis is not None else None
        self.jaw_target, self.jaw_hide, self.jaw_hide_w = th, hide, hw
        self.jaw_plan = plan_jaw(th, w, hide, hw, hold, lead, np.radians(vmax), np.radians(amax), lam_acc, lam_jerk, floor=floor)
        return self.jaw_plan

    def _cavity_colour(self):
        """the take's own dark interior tone (BGR): median of the darker-half non-teeth inner-lip pixels of its open
        frames, at the shot's median light."""
        g = self.F[:, 0]
        open_ = np.argsort(-g)[:max(3, len(g) // 5)]
        px = []
        for i in open_:
            img = self.frames[i]
            P = self.S[i][:, :2]
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            m = poly_mask(P[LIP_IN], img.shape, -0.1) & ~teeth_key(lab, self.skin_med)
            if m.sum() < 20: continue
            v = img[m].astype(float)
            L = lab[..., 0][m]
            px.append(v[(L >= np.percentile(L, 20)) & (L <= np.percentile(L, 50))])
        if not px: return np.array([20.0, 15.0, 30.0])
        return np.median(np.vstack(px), 0)

    def jaw(self, gap_t, i=None, p14=None, lim=(-25.0, 30.0), extra=0.0):
        """jaw angle (rad), relative to the lower harvest frame. With the target lower-lip point p14 (frame i px, where
        the warped lower lip IS): the drop that brings the harvest frame's lower lip (14, carried by the jaw) onto it,
        measured in the canonical head frame along the head's down axis (after the skin pin of frame i is taken out).
        Otherwise from the target opening gap_t (canonical px) vs the harvest frame's. jaw_gain scales the drop; extra
        (canonical px) is added to it; clamped to lim (degrees)."""
        if p14 is not None and i is not None:
            Ai = inv_aff(plane_affine(self.pl, i, self.z_lo, ref=self.ref))
            q = Ai[:, :2] @ (np.asarray(p14, float)[:2] - self.pin[i]) + Ai[:, 2]
            dd = self.d[:2] / max(1e-6, np.linalg.norm(self.d[:2]))
            delta = float((q - self.T_lo[:2]) @ dd)
        else: delta = gap_t - self.F[self.h_lo, 0]
        th = jaw_angle(self.T_lo, self.H, self.u, self.d, self.jaw_gain * delta + extra)
        return float(np.clip(th, np.radians(lim[0]), np.radians(lim[1])))

    def jaw_at(self, i, gap_t=None, p14=None):
        """the jaw used for frame i: the clip plan when one was solved (plan), else per frame (v5)."""
        return float(self.jaw_plan[i]) if self.jaw_plan is not None else self.jaw(gap_t, i, p14)

    def affines(self, i, gap_t, p14=None, A_h=None):
        """(upper, lower) 2x3 maps: harvest frame px -> frame i px (both carry the skin pin of frame i)."""
        A_up = shift_aff(compose(plane_affine(self.pl, i, self.z_up, ref=self.ref), inv_aff(self.A_up_h if A_h is None else A_h)), self.pin[i])
        R = rot_axis(self.u, self.jaw_at(i, gap_t, p14))
        A_lo = shift_aff(compose(plane_affine(self.pl, i, self.z_lo, R, self.H, ref=self.ref), inv_aff(self.A_lo_h)), self.pin[i])
        return A_up, A_lo

    def cavity_field(self, box, Q, gap_t, cav):
        """v6 soft throat: on wide-open targets the flat cavity tone gets depth -- darker toward the throat (centre,
        slightly low), a little lighter and warmer toward the cheeks / palate at the edge of the opening. Strength ~
        throat x how wide open the target is (0 below 45 % of the take's p95 opening, full at 100 %)."""
        o = float(smootherstep((gap_t / max(1e-6, self.g_hi) - 0.45) / 0.55)) * self.throat
        if o <= 1e-3: return np.broadcast_to(np.asarray(cav, np.float32), box + (3,))
        c = Q.mean(0)
        rx = max(2.0, np.ptp(Q[:, 0]) / 2)
        ry = max(2.0, np.ptp(Q[:, 1]) / 2)
        yy, xx = np.mgrid[0:box[0], 0:box[1]].astype(np.float32)
        r2 = ((xx - c[0]) / rx) ** 2 + ((yy - c[1] - 0.15 * ry) / ry) ** 2
        dark = 1 - 0.55 * o * np.exp(-r2 / 0.3)
        warm = 0.5 * np.asarray(cav, np.float32) + np.array([18, 24, 66], np.float32)      # dark red-brown (cheek / palate)
        e = (0.5 * o * smootherstep((r2 - 0.25) / 0.75))[..., None]
        return np.clip((np.asarray(cav, np.float32) * (1 - e) + warm * e) * dark[..., None], 0, 255)

    def render(self, out, i, P_in, gap_t, lip_ab=None, skin_ab=None, band=0.04, interp=cv2.INTER_LANCZOS4, jaw_from='lip', pose_fade=None):
        """composite the teeth layer into out (frame i) inside the inner-lip contour P_in (20 x 2, where the lips ARE
        in out). Returns (image, mask float 0..1)."""
        H, W = out.shape[:2]
        P_in = np.asarray(P_in, float)[:, :2]
        if cv2.contourArea(P_in.astype(np.float32)) < 4: return out, np.zeros((H, W), np.float32)
        e = band * self.iod * self.pl.s[i]
        x0, y0 = np.maximum(0, np.floor(P_in.min(0) - 2 * e - 4).astype(int))
        x1, y1 = np.minimum([W, H], np.ceil(P_in.max(0) + 2 * e + 5).astype(int))
        if x1 - x0 < 4 or y1 - y0 < 4: return out, np.zeros((H, W), np.float32)
        box = (y1 - y0, x1 - x0)
        sh = lambda A: A - np.array([[0, 0, x0], [0, 0, y0]], float)
        A_up, A_lo = self.affines(i, gap_t, P_in[LIP_IN.index(14)] if jaw_from == 'lip' else None)
        g = float(self.light[i])
        def plate(img, alpha, A, h):
            A = sh(A)
            p = cv2.warpAffine(img, A, box[::-1], flags=interp, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
            a = cv2.warpAffine(alpha, A, box[::-1], flags=cv2.INTER_LINEAR, borderValue=0)
            k = g / max(1e-3, self.light[h])
            lab = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[..., 0] = np.clip(lab[..., 0] * k, 0, 255)
            return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32), np.clip(a, 0, 1)[..., None]
        cav = np.clip(self.cavity * g / max(1e-3, self.skin_med), 0, 255)
        io = self.iod * self.pl.s[i]
        reg = out[y0:y1, x0:x1]
        labr = cv2.cvtColor(reg, cv2.COLOR_BGR2LAB)
        Q = P_in - [x0, y0]
        core = poly_mask(Q, box, 0.0)
        rows = np.arange(box[0])[:, None]
        if self.throat > 0: cav = self.cavity_field(box, Q, gap_t, cav)
        if self.plate_w is None or len(self.up_plates) < 2:
            up, aup = plate(self.up_img, self.up_alpha, A_up, self.h_up)
        else:      # v6 per-head-angle upper plates, blended by yaw (premultiplied)
            acc, aacc = 0.0, 0.0
            for k, pk in enumerate(self.up_plates):
                wk = float(self.plate_w[i, k])
                if wk < 0.02: continue
                Ak = A_up if k == 0 else self.affines(i, gap_t, None, pk['A_h'])[0]
                pi_, ai_ = plate(pk['img'], pk['alpha'], Ak, pk['h'])
                acc = acc + wk * pi_ * ai_
                aacc = aacc + wk * ai_
            aup = np.clip(aacc, 0, 1)
            up = acc / np.maximum(aacc, 1e-4)
        lo, geo = plate(self.lo_img, self.lo_geo, A_lo, self.h_lo)
        lit = cv2.warpAffine(self.lo_lit, sh(A_lo), box[::-1], flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        cov = geo[..., 0] > 0.97
        ca = cov.any(0)
        first = np.where(ca, cov.argmax(0), box[0])
        last = np.where(ca, box[0] - 1 - cov[::-1].argmax(0), -1)
        if not ca.all() and self.v6_plate:          # (also when the plate is fully hidden: a near-closed frame)
            # v6: in columns the lower plate does not reach, the cavity / take boundary follows the opening's own shape
            # (half way between the target lips; a thin corner or a closed slit stays the take's lips) instead of a
            # straight line carried across from the plate (that printed a rectangular notch); softened horizontally below
            cb = core.any(0)
            bot = np.where(cb, box[0] - 1 - core[::-1].argmax(0), box[0])
            top = np.where(cb, core.argmax(0), box[0])
            frac = smootherstep(((bot - top) / max(1.0, io) - 0.02) / 0.05)          # thin corner / slit: no cavity
            mid = np.round(top + frac * (bot - top) / 2).astype(int)
            first = np.where(ca, first, np.where(cb, mid, box[0]))
            last = np.where(ca, last, np.where(cb, mid - 1, -1))
        elif ca.any() and not ca.all():    # columns the plate does not reach: its top / bottom lines carried across (no
            xs = np.arange(box[1])         # vertical step between covered and uncovered columns)
            first = np.round(np.interp(xs, xs[ca], first[ca])).astype(int)
            last = np.round(np.interp(xs, xs[ca], last[ca])).astype(int)
        # base: the warped frame, kept only under the lower plate (the lower lip's inner surface); there any teeth-like
        # pixel (a take tooth, its fringe; a looser key) and whatever the plate only half covers is the local non-teeth
        # colour (normalised convolution), so no stray take tooth or seam survives. Above the lower plate's top: the
        # cavity tone (the space the jaw opened behind the upper teeth); columns the plate does not reach: cavity
        kk = max(1, int(round(0.008 * io)))
        below = rows > last[None, :]
        tk = cv2.dilate(teeth_key(labr, self.skin_med, 0.8).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kk + 1, 2 * kk + 1)))
        tk = np.maximum(tk, cv2.dilate((below & teeth_key(labr, self.skin_med, 0.6, 150)).astype(np.uint8),
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * kk + 1, 4 * kk + 1))))
        w_ok = (1 - tk).astype(np.float32)
        sg = max(2.0, 0.03 * io)
        soft = cv2.GaussianBlur(reg.astype(np.float32) * w_ok[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(w_ok, (0, 0), sg), 1e-3)[..., None]
        tk = np.maximum(tk, ((below | (rows >= first[None, :])) & (geo[..., 0] < 0.98)).astype(np.uint8))
        tkf = cv2.GaussianBlur(tk.astype(np.float32), (0, 0), 1.0)[..., None]
        base = reg.astype(np.float32) * (1 - tkf) + soft * tkf
        above = (rows < first[None, :]).astype(np.float32)
        if self.v6_plate: above = cv2.GaussianBlur(above, (0, 0), sigmaX=max(1.0, 0.03 * io), sigmaY=max(1.0, 0.006 * io))[..., None]
        else: above = cv2.GaussianBlur(above, (0, 0), max(1.0, 0.006 * io))[..., None] * (rows <= np.maximum(last, first)[None, :])[..., None]
        layer = base * (1 - above) + cav * above
        layer = layer * (1 - geo) + (lo * lit + cav * (1 - lit)) * geo        # lower arch + tongue (jaw)
        layer = layer * (1 - aup) + up * aup                                    # upper arch (skull), in front
        # mask: the inner-lip polygon; in a band around it, also any NON-lip pixel of out (a stretched tooth / cavity
        # the warp left at the lip edge), lip pixels kept
        ring = poly_mask(Q, box, 0.0, e) & ~poly_mask(Q, box, 0.0, -e)
        if lip_ab is not None:
            lp = lipness_map(reg, lip_ab, skin_ab)
            lab = cv2.cvtColor(reg, cv2.COLOR_BGR2LAB)
            stray = ring & (lp < 0.35) & (teeth_key(lab, self.skin_med) | (lab[..., 0] < 0.5 * self.skin_med))
            m = core | stray
        else: m = core
        mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(0.7, 0.004 * self.iod * self.pl.s[i]))
        mf = np.where(core, np.maximum(mf, poly_mask(Q, box, 0.0, -1.5).astype(np.float32)), mf)[..., None] * self.pose_alpha(i, pose_fade)
        o = out.copy()
        o[y0:y1, x0:x1] = np.clip(reg.astype(np.float32) * (1 - mf) + layer * mf + 0.5, 0, 255).astype(np.uint8)
        M = np.zeros((H, W), np.float32)
        M[y0:y1, x0:x1] = mf[..., 0]
        return o, M

    def pose_alpha(self, i, fade=None):
        """1 near the harvest poses, easing to 0 between fade[0] and fade[1] degrees of head rotation away from them: a
        frontal plate cannot show what a far-turned head shows (the back teeth of the near side), so there the take's
        own interior is kept. fade None: always 1."""
        if fade is None: return 1.0
        p = self.pl.par[:, 1:4]
        dev = min(np.linalg.norm(p[i] - p[self.h_up]), np.linalg.norm(p[i] - p[self.h_lo]))
        return float(1.0 - smootherstep((np.degrees(dev) - fade[0]) / max(1e-6, fade[1] - fade[0])))

    def info(self):
        return dict(h_up=self.h_up, h_lo=self.h_lo, hinge=np.round(self.H, 1).tolist(), z_up=round(self.z_up, 1), z_lo=round(self.z_lo, 1),
                    cavity=np.round(self.cavity, 1).tolist(), iod=round(self.iod, 1),
                    plates=[dict(h=p['h'], shift=p.get('shift')) for p in self.up_plates], plate_dbg=getattr(self, 'plate_dbg', None), e_lo=round(self.e_lo, 2),
                    pin_px_p95=round(float(np.percentile(np.linalg.norm(self.pin, axis=1), 95)), 2), tongue_drop=self.tongue_drop, throat=self.throat,
                    jaw_plan=self.jaw_plan is not None)
