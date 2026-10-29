#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_warp.py: the --mode warp engine of lsdub paint ("warp, don't paste"). Called through mouth_paint.py.

No library mouth is painted. The take's OWN lips are moved, per frame, toward the shape the audio asks for, by a smooth
dense warp of the take's own pixels (lighting, texture and skin stay the render's by construction).

  1. MOUTH MODEL (per take): MediaPipe 3D track -> one rigid skull pose per frame (RigidPlacement on MESH_ANCH, pose
     smoothed zero-phase) -> every frame's lip/chin points in the canonical head frame. Features: inner-lip gap (13-14)
     and corner width (61-291). The face's own structure = a linear map (least squares over the take's frames) from
     (gap, width) to all control points in 3D, plus a PCA of the observed shapes (reported, for the log).
  2. TARGET from the viseme track (lead already in it): per-viseme ACCEPTABLE bands of openness / spread-round, mixed by
     vis/vis2 (+ the track's openness on vowels), in THIS take's observed range (shut .. p95 gap, p10 .. p90 width). The
     target is the observed shape clipped into the band, so only what is wrong moves. Residual zero-phase smoothed.
  3. RESIDUAL only: delta = J (target - observed) in the canonical frame, rotated into the frame by the pose (s R), with a
     confidence dead-band (|residual| small -> warp scale 0: where the take already matches, it is untouched).
  4. WARP: backward map x -> x + w(x) d(x), d = TPS (regularised) through the control points (target pos -> observed pos)
     and a fixed ring of skin anchors (zero), w = 1 on the lips/chin, smooth falloff to exactly 0 at the ring hull.
     Beyond the ring the output pixel IS the take pixel (no seam, pixel fidelity 1).
  5. INTERIOR: target more open than the take -> the opened gap is filled from the take's OWN frame nearest in
     (gap, head pose) (its inner-lip region piecewise-affine warped onto the target inner-lip polygon), or 'dark'
     (darkened stretched pixels), or 'stretch' (the warp alone). More closed -> the warp compresses.

  r2 (defaults; every r2 change has a flag that restores r1, see the --warp-* help):
     - temporal: the residual is taken against the zero-phase SMOOTHED observed shape (--warp-feat-smooth), and the
       residual itself (= the warp field, linear in it) is eased zero-phase (--warp-temporal); landmark noise no longer
       becomes pixel slosh
     - falloff: an ellipse FIXED in the canonical head frame, carried by the strict rigid pose (HeadEllipse), C2
       smootherstep, ring at 2.6x the mouth envelope (r1: a per-frame hull that breathed with the track, a visible cut)
     - fold guard: per frame, the residual is scaled down until the backward map does not fold (fold_det / fold_scale)
     - jaw: the opening is re-referenced to the upper lip (it rides the skull; the lower lip + chin carry the gap)
     - targets: past the band edge toward its centre (--warp-centre); spread / round classes floored (--warp-class-width)
     - bank (mode b) off by default (another frame's texture = swap or double mouth); sources / donors cross-faded when
       used; ONE interior donor per opening segment; donor fill own-or-donor (interior_alpha), tight feather
     - Lanczos resampling

  v5 (opt-in): --warp-teeth rigid -- the interior inside the target inner lip comes from rigid teeth plates (teeth_layer.py):
     upper arch at the skull pose, lower arch at skull + jaw rotation about the condyle hinge; no donors, no stretched teeth.
  v7 (opt-in): --warp-compose layers -- the mouth as an explicit back-to-front layer stack (mouth_layers.py): frame,
     mouth cover (the take's mouth erased to its skin), back of mouth, lower arch + tongue, upper arch, lips, face; full
     curved 3D arches; lower teeth pinned to the chin skin (--warp-teeth-chin-pin); a bite floor in the jaw plan.
  v6 (opt-in, on top of v5): --warp-teeth-anchor skin (teeth ride the upper-lip root skin's own flow, not the pose alone),
     --warp-teeth-jaw plan (one smooth, speed-limited jaw trajectory for the whole clip; lower teeth hidden on near-closed
     frames), --warp-teeth-tongue-drop, --warp-teeth-plates N (per-head-angle upper plates), --warp-teeth-throat.

  lsdub paint --mode warp --video take.mp4 --track vis.json --out warped.mp4 [--mask-out m.npz] [--warp-interior donor]
"""
import json
from ..io import ff
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

from . import mouth_paint as mp_
from . import mouth_morph as mm
from . import lip_layers as ll

LIP_OUT = mp_.LIP_OUT                       # outer lip loop (20)
LIP_IN = mm.LIP_IN                          # inner lip loop (20), ordered
CHIN_C = [18, 200, 199, 175]                # soft chin under the lower lip: rides the jaw with the opening
CTRL = LIP_OUT + LIP_IN + CHIN_C
# upper lip (outer + inner, corners excluded): rides the skull. --warp-jaw re-references the opening to it, so a gap change
# moves the lower lip + chin (the jaw) and leaves the upper lip where the skull holds it
UPPER = [185, 40, 39, 37, 0, 267, 269, 270, 409, 191, 80, 81, 82, 13, 312, 311, 310, 415]
# per-viseme ACCEPTABLE bands (the residual is the distance to the band, so a take already inside it is left alone):
# openness in units of the take's own range (0 = shut, 1 = its p95 gap); width in -1 (p10, round) .. +1 (p90, spread)
V_OPEN = {'aa': (0.65, 1.05), 'oh': (0.5, 0.9), 'E': (0.42, 0.85), 'ih': (0.22, 0.6), 'ou': (0.12, 0.45), 'RR': (0.1, 0.45),
          'CH': (0.08, 0.4), 'DD': (0.08, 0.45), 'kk': (0.08, 0.45), 'nn': (0.08, 0.45), 'TH': (0.08, 0.4),
          'FF': (0.03, 0.3), 'SS': (0.05, 0.35), 'PP': (0.0, 0.03), 'sil': (0.0, 0.45)}
INF = 9.0
V_WIDTH = {'ih': (0.25, INF), 'E': (0.2, INF), 'SS': (0.1, INF), 'ou': (-INF, -0.3), 'oh': (-INF, -0.15),
           'CH': (-INF, -0.1), 'RR': (-INF, -0.1)}


def _enamel_params(strength, mix=None):
    """v14 (--warp-arch-enamel): the enamel material-pass dict from the strength S and an optional "spec,transl,ao,warm" mix
    (default preset 1.0 / 0.8 / 0.7 / 1.0, each x S). a_target 128 = Lab neutral (the warm pull holds a* low)."""
    s = float(strength)
    r = [1.0, 0.8, 0.7, 1.0]
    if mix:
        try: r = [float(x) for x in str(mix).split(',')][:4] + r[len([float(x) for x in str(mix).split(',')][:4]):]
        except ValueError: pass
    return dict(spec=s * r[0], transl=s * r[1], ao=s * r[2], warm=s * r[3], a_target=128.0)


def library_bands(lib, half=0.15):
    """--lib given: the TARGET GEOMETRY comes from the LSDub mouth library (each viseme's landmark shape), expressed in
    library units (openness 0 = PP .. 1 = aa, width -1 = roundest .. +1 = widest, 0 = sil) -> per-viseme bands. Only the
    numbers are taken; no library pixel is used."""
    import glob, os
    G = {}
    for f in glob.glob(os.path.join(lib, '*_lm.npy')):
        v = os.path.basename(f)[:-7]
        if v == 'base': continue
        L = np.load(f)
        G[v] = (float(np.linalg.norm(L[13, :2] - L[14, :2])), float(np.linalg.norm(L[61, :2] - L[291, :2])))
    if 'aa' not in G or 'sil' not in G: return dict(V_OPEN), dict(V_WIDTH)
    g0 = G.get('PP', G['sil'])[0]
    g1 = max(1e-6, G['aa'][0] - g0)
    w0 = G['sil'][1]
    wmax = max(1e-6, max(w for _, w in G.values()) - w0)
    wmin = max(1e-6, w0 - min(w for _, w in G.values()))
    bo, bw = {}, {}
    for v, (g, w) in G.items():
        u = (g - g0) / g1
        bo[v] = (max(0.0, u - half), u + half if v != 'aa' else max(1.05, u + half))
        uw = (w - w0) / (wmax if w >= w0 else wmin)
        if uw > 0.15: bw[v] = (uw - 0.25, INF)
        elif uw < -0.15: bw[v] = (-INF, uw + 0.25)
    bo['PP'] = (0.0, 0.03)
    bo.setdefault('sil', V_OPEN['sil'])
    bo['sil'] = V_OPEN['sil']
    for k, v in mp_.LIBMAP.items(): bo.setdefault(k, bo.get(v, V_OPEN[k]))
    return bo, bw


SPREAD_V, ROUND_V = ('ih', 'E', 'SS'), ('ou', 'oh', 'CH', 'RR')


def class_floor(bands, wmin, rng=None, ref='half'):
    """make the spread / round classes unambiguous: a spread viseme's width band starts at least wmin ABOVE the take's
    median width, a round one ends at least wmin BELOW it (the library's own E / oh bands sit almost on the median, so
    an edge target there reads as neutral). wmin <= 0: bands unchanged."""
    bo, bw = bands or (dict(V_OPEN), dict(V_WIDTH))
    bw = dict(bw)
    if wmin <= 0: return bo, bw
    ws, wr = wmin, wmin
    if ref == 'wide' and rng is not None:
        # v3: the floor in units of the take's WIDER half-range. A take that sits near its p90 width most of the time
        # (p90 - median tiny) otherwise gets a 'spread' floor that is barely above its own median, and a 'round' floor
        # the same the other way; both halves are expressed in their own units, so convert the wide half's distance.
        up, dn = max(1e-6, rng['w_hi'] - rng['w_med']), max(1e-6, rng['w_med'] - rng['w_lo'])
        h = max(up, dn)
        ws, wr = wmin * h / up, wmin * h / dn
    for v in SPREAD_V:
        lo, hi = bw.get(v, (-INF, INF))
        bw[v] = (max(lo, ws), INF)
    for v in ROUND_V:
        lo, hi = bw.get(v, (-INF, INF))
        bw[v] = (-INF, min(hi, -wr))
    return bo, bw


def features(C):
    """C: (..., K, 3) points in canonical (head) frame, K indexed like the 478 mesh -> (gap, width)."""
    return np.stack([np.linalg.norm(C[..., 13, :2] - C[..., 14, :2], axis=-1),
                     np.linalg.norm(C[..., 61, :2] - C[..., 291, :2], axis=-1)], -1)


class MouthModel:
    """the character's own mouth: canonical control points X (T x K x 3) vs features F (T x 2)."""

    def __init__(self, Xc, F):
        T, K, _ = Xc.shape
        self.F = F
        A = np.hstack([np.ones((T, 1)), F])
        Y = Xc.reshape(T, -1)
        self.coef, *_ = np.linalg.lstsq(A, Y, rcond=None)           # (3, K*3)
        self.J = self.coef[1:].reshape(2, K, 3)                      # d(points)/d(gap, width)
        Z = Y - Y.mean(0)
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        ev = s ** 2 / max(1e-9, (s ** 2).sum())
        self.pca_var = ev[:4].round(3).tolist()
        self.pca = Vt[:4]
        g, w = F[:, 0], F[:, 1]
        self.rng = dict(g_lo=float(min(np.percentile(g, 3), 0.12 * np.median(g))), g_hi=float(np.percentile(g, 95)),
                        g_max=float(g.max()), w_lo=float(np.percentile(w, 10)), w_med=float(np.median(w)),
                        w_hi=float(np.percentile(w, 90)), w_p5=float(np.percentile(w, 5)), w_p95=float(np.percentile(w, 95)))

    def jaw_reference(self, upper_idx):
        """opening as a JAW move: subtract the upper lip's mean response to the gap from every point, so (in the canonical
        head frame) the upper lip stays with the skull and the lower lip + soft chin carry the whole opening / closing.
        Width is left as fitted (corners move symmetrically)."""
        self.J = self.J.copy()
        self.J[0] = self.J[0] - self.J[0][upper_idx].mean(0)

    def delta(self, dF):
        """canonical displacement (K x 3) of the control points for a feature change dF = (dgap, dwidth)."""
        return np.tensordot(dF, self.J, axes=(0, 0))


def _to_band(u, lo, hi, centre):
    """clip u into [lo, hi]; with centre > 0 a value OUTSIDE the band goes past the edge toward the band centre (a
    one-sided band, e.g. width (0.2, INF), uses edge +- 0.5 as its centre). Inside the band: untouched."""
    if lo <= u <= hi: return u
    mid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (lo + 0.5 if np.isfinite(lo) else hi - 0.5)
    e = lo if u < lo else hi
    return e + centre * (mid - e)


def target_features(fr_t, F, rng, open_gain=1.0, width_gain=1.0, sigma=1.0, bands=None, centre=0.0, width_mix='switch',
                    close_depth=0.0):
    """per-frame target (gap, width): the observed features F (T x 2) clipped into the viseme's band (vis/vis2 mixed),
    bands in the take's own range; vowels' lower bound is raised by the audio's openness (A2F jaw / envelope).
    centre (0..1): a shape outside its band is moved past the band edge toward the band centre (0 = edge, r1 behaviour).
    width_mix: 'switch' (r2) = the width band of vis2 once mix >= 0.5, even when vis2 has none (a spread E whose vis2 is
    sil / aa then loses its width target mid-vowel); 'primary' (v3) = vis's band unless vis2 has its own band and mix > 0.5.
    close_depth: PP targets this far BELOW the take's shut gap (units of its range): a press, not a touch, so the closure
    survives the field's smoothing and reads shut."""
    n = len(F)
    go, gs = rng['g_lo'], max(1e-6, rng['g_hi'] - rng['g_lo'])
    uo = (F[:, 0] - go) / gs
    uw = np.where(F[:, 1] >= rng['w_med'], (F[:, 1] - rng['w_med']) / max(1e-6, rng['w_hi'] - rng['w_med']),
                                           (F[:, 1] - rng['w_med']) / max(1e-6, rng['w_med'] - rng['w_lo']))
    to, tw = uo.copy(), uw.copy()
    VO, VW = bands or (V_OPEN, V_WIDTH)
    for i in range(n):
        f = fr_t[min(i, len(fr_t) - 1)]
        v1, v2 = f['vis'], f['vis2']
        mx = f['mix'] if v2 != v1 else 0.0
        b1, b2 = VO.get(v1, (0.0, 0.6)), VO.get(v2, (0.0, 0.6))
        lo, hi = (1 - mx) * b1[0] + mx * b2[0], (1 - mx) * b1[1] + mx * b2[1]
        if v1 in ('aa', 'E', 'oh', 'ih'): lo = max(lo, min(hi, 0.8 * float(f.get('open', 0))))
        if v1 == 'PP': lo, hi = -close_depth, 0.03 - close_depth
        to[i] = _to_band(uo[i], lo * open_gain if lo > 0 else lo, hi * max(1.0, open_gain) if hi > 0 else hi, centre)
        w1, w2 = VW.get(v1, (-INF, INF)), VW.get(v2, (-INF, INF))
        if width_mix == 'primary': wl, wh = w2 if (mx > 0.5 and w2 != (-INF, INF)) or w1 == (-INF, INF) else w1
        elif mx < 0.5: wl, wh = w1
        else: wl, wh = w2
        tw[i] = _to_band(uw[i], wl * width_gain if wl > -INF else -np.inf, wh * width_gain if wh < INF else np.inf, centre) \
            if (wl, wh) != (-INF, INF) else uw[i]
    if sigma > 0:      # zero-phase ease of the residual; PP closures keep their depth
        pp = np.array([fr_t[min(i, len(fr_t) - 1)]['vis'] == 'PP' for i in range(n)])
        r = gaussian_filter1d(to - uo, sigma, mode='nearest')
        to = np.where(pp, np.minimum(to, uo + r), uo + r)
        tw = uw + gaussian_filter1d(tw - uw, sigma, mode='nearest')
    g = go + to * gs
    w = rng['w_med'] + tw * np.where(tw >= 0, rng['w_hi'] - rng['w_med'], rng['w_med'] - rng['w_lo'])
    return np.stack([g, w], 1)


def confidence(dF, scale, d0=0.04, d1=0.12):
    """warp scale 0..1 per frame: 0 where the residual is inside the dead-band (the take already matches)."""
    r = np.max(np.abs(dF) / scale, axis=1)
    x = np.clip((r - d0) / max(1e-6, d1 - d0), 0, 1)
    return x * x * (3 - 2 * x)


def tps_field(src, disp, pts, lam):
    """regularised TPS through src -> disp, evaluated at pts (N x 2). lam in normalised units."""
    m = mm.tps_fit(src, disp, lam=lam)
    return mm.tps_apply(m, pts)


def ring_anchors(ctrl_xy, centre, factor=1.6, n=28):
    """fixed skin ring: the control hull pushed out radially by `factor` about the mouth centre, resampled to n points."""
    hull = cv2.convexHull(ctrl_xy.astype(np.float32))[:, 0]
    ang = np.arctan2(hull[:, 1] - centre[1], hull[:, 0] - centre[0])
    rad = np.linalg.norm(hull - centre, axis=1)
    o = np.argsort(ang)
    ang, rad = ang[o], rad[o]
    a = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = np.interp(a, ang, rad, period=2 * np.pi) * factor
    return centre + np.stack([np.cos(a), np.sin(a)], 1) * r[:, None]


def warp_frame(img, src, dst, anchors, lam=1e-3, inner=1.15, src_img=None, anchors_src=None, weight=None,
               interp=cv2.INTER_CUBIC):
    """move the pixels at src (K x 2) to dst, ring `anchors` fixed. Returns (warped image, weight map w in [0,1]).
    w = 1 inside the control hull grown by `inner` (about its centre), falls to exactly 0 at the anchor hull.
    src_img / anchors_src (mode b): sample another frame of the clip (src = its points, anchors_src = where this frame's
    ring sits in it); the result is blended onto img by w, so the mask edge is img's own skin.
    weight: a precomputed full-frame falloff (e.g. ellipse_weight, fixed in the head frame); default = the r1 per-frame
    hull falloff (core = control hull x inner, smoothstep of the distance ratio to the anchor hull)."""
    H, W = img.shape[:2]
    c = dst.mean(0)
    hull_out = cv2.convexHull(anchors.astype(np.float32))[:, 0]
    if weight is not None:
        ys, xs = np.nonzero(weight > 0)
        if len(xs) == 0: return img.copy(), np.zeros((H, W), np.float32)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    else:
        x0, y0 = np.floor(hull_out.min(0)).astype(int) - 2
        x1, y1 = np.ceil(hull_out.max(0)).astype(int) + 3
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4: return img.copy(), np.zeros((H, W), np.float32)
    bw, bh = x1 - x0, y1 - y0
    if weight is not None:
        wgt = weight[y0:y1, x0:x1].astype(np.float32)
    else:
        outer = np.zeros((bh, bw), np.uint8)
        cv2.fillConvexPoly(outer, np.round(hull_out - [x0, y0]).astype(np.int32), 255)
        allp = np.vstack([src, dst])
        hin = cv2.convexHull(((allp - c) * inner + c).astype(np.float32))[:, 0]
        core = np.zeros((bh, bw), np.uint8)
        cv2.fillConvexPoly(core, np.round(hin - [x0, y0]).astype(np.int32), 255)
        core &= outer
        d_out = cv2.distanceTransform(outer, cv2.DIST_L2, 5)                       # distance to the outer edge (inside)
        d_in = cv2.distanceTransform(255 - core, cv2.DIST_L2, 5)                   # distance to the core (outside it)
        t = np.where(core > 0, 1.0, d_out / np.maximum(d_out + d_in, 1e-6))
        t = np.where(outer > 0, t, 0.0)
        wgt = (t * t * (3 - 2 * t)).astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    P = np.stack([xx.ravel(), yy.ravel()], 1)
    a_src = anchors if anchors_src is None else anchors_src
    ctrl = np.vstack([dst, anchors])
    disp = np.vstack([src - dst, a_src - anchors])
    D = tps_field(ctrl, disp, P, lam).reshape(bh, bw, 2).astype(np.float32)
    mx = xx + wgt * D[..., 0]
    my = yy + wgt * D[..., 1]
    if src_img is None:
        patch = cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REFLECT)
    else:   # mode b: full displacement into the other frame, then cross-blend by w (edge = this frame's skin)
        P2 = cv2.remap(src_img, xx + D[..., 0], yy + D[..., 1], interp, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        patch = np.clip(P2 * wgt[..., None] + img[y0:y1, x0:x1].astype(np.float32) * (1 - wgt[..., None]), 0, 255).astype(np.uint8)
    out = img.copy()
    sub = out[y0:y1, x0:x1]
    m = wgt > 0
    sub[m] = patch[m]
    w_full = np.zeros((H, W), np.float32)
    w_full[y0:y1, x0:x1] = wgt
    return out, w_full


def fan_tris(n):
    return np.array([[n, k, (k + 1) % n] for k in range(n)])


def interior_fill(out, donor_img, donor_in, dst_in, alpha, mode='donor', erode=0.18, feather=0.35):
    """fill the opened mouth: donor = the donor frame's inner-lip region warped onto dst_in (fan around the centroid);
    dark = darken what the warp stretched into the gap. alpha (0..1) scales the blend."""
    if alpha <= 0.01: return out, None
    H, W = out.shape[:2]
    c = dst_in.mean(0)
    poly = ((dst_in - c) * (1 - erode * 0.3) + c)
    x0, y0 = np.floor(dst_in.min(0)).astype(int) - 4
    x1, y1 = np.ceil(dst_in.max(0)).astype(int) + 5
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 3 or y1 - y0 < 3: return out, None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(poly - [x0, y0]).astype(np.int32)], 255)
    gap = max(1.0, np.ptp(dst_in[:, 1]))
    k = max(3, int(gap * feather) | 1)
    mf = cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255 * alpha
    reg = out[y0:y1, x0:x1].astype(np.float32)
    if mode == 'donor' and donor_img is not None:
        sp = np.vstack([donor_in, donor_in.mean(0)])
        dp = np.vstack([dst_in, c])
        fill, ok = mm.pwa(donor_img, sp, dp, fan_tris(len(dst_in)), (x0, y0, x1 - x0, y1 - y0))
        fill = fill.astype(np.float32)
        mf = mf * ok
    else:   # dark cavity: the stretched pixels, darkened toward the take's own darkest mouth tone
        fill = reg * 0.35
    reg = reg * (1 - mf[..., None]) + fill * mf[..., None]
    out = out.copy()
    out[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
    return out, (x0, y0, mf)


def seal_fill(out, dst_in, u, thr, grow=1.6):
    """v3 closure seal: when the target opening u (x the take's range above its shut gap) is below thr, the slit the
    compressed mouth still shows (a line of teeth / cavity between the moved lips) is inpainted from the SAME frame's
    lips around it (cv2 Telea), blended in by how shut the target is. No other frame's texture is used (no swap).
    Returns (image, blend alpha)."""
    if thr <= 0 or u >= thr: return out, 0.0
    al = float(smootherstep((thr - u) / thr * 2.0))
    if al <= 0.01: return out, 0.0
    H, W = out.shape[:2]
    c = dst_in.mean(0)
    q = (dst_in - c) * [1.0, grow] + c
    q[:, 1] = c[1] + np.sign(q[:, 1] - c[1]) * np.maximum(np.abs(q[:, 1] - c[1]), 1.5)    # at least a 3 px tall slit
    x0, y0 = np.floor(q.min(0)).astype(int) - 8
    x1, y1 = np.ceil(q.max(0)).astype(int) + 9
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4: return out, 0.0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(q - [x0, y0]).astype(np.int32)], 255)
    reg = out[y0:y1, x0:x1]
    fill = cv2.inpaint(reg, m, 3, cv2.INPAINT_TELEA).astype(np.float32)
    mf = cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32)[..., None] / 255 * al
    out = out.copy()
    out[y0:y1, x0:x1] = np.clip(reg * (1 - mf) + fill * mf + 0.5, 0, 255).astype(np.uint8)
    return out, al


def smootherstep(t):
    """C2 ramp 0..1 (6t^5 - 15t^4 + 10t^3): zero first AND second derivative at both ends, so the falloff has no
    visible crease where it meets the untouched skin or the fully-moved lips."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


class HeadEllipse:
    """the warp's falloff + skin ring, FIXED in the canonical (stabilised) head frame and carried into each frame by the
    strict rigid skull pose, so the ring and the falloff ride the skull instead of being rebuilt per frame from noisy
    lip points (r1: the hull of src+dst, re-grown every frame -> the falloff band itself breathed with the track).
    Four half-axes (left/right, up/down) from the envelope of the mouth's canonical points over the whole take (observed
    AND targets); rho = elliptical radius in those units; w = 1 inside rho <= r_in, smootherstep to exactly 0 at r_out.
    The quadrant switch is C1 (d rho / d q is 0 across the axes)."""

    def __init__(self, Xc, centre, r_in=1.0, r_out=1.8, q=98):
        d = Xc[..., :2].reshape(-1, 2) - centre[:2]
        def ext(v):
            v = v[v > 0]
            return float(np.percentile(v, q)) if len(v) else 1.0
        self.c = np.asarray(centre, float)
        self.ax = (ext(-d[:, 0]), ext(d[:, 0]))
        self.ay = (ext(-d[:, 1]), ext(d[:, 1]))
        self.r_in, self.r_out = r_in, r_out

    def rho(self, q):
        ax = np.where(q[..., 0] < 0, self.ax[0], self.ax[1])
        ay = np.where(q[..., 1] < 0, self.ay[0], self.ay[1])
        return np.sqrt((q[..., 0] / ax) ** 2 + (q[..., 1] / ay) ** 2)

    def _affine(self, pl, i):
        return pl.s[i] * pl.R[i][:2, :2], pl.rigid(i, self.c[None])[0]

    def anchors(self, pl, i, n=28, r=None):
        """n skin-ring points on rho = r (default r_out), canonical plane through the centre, projected into frame i."""
        r = self.r_out if r is None else r
        a = np.linspace(-np.pi, np.pi, n, endpoint=False)
        ca, sa = np.cos(a), np.sin(a)
        q = np.stack([r * ca * np.where(ca < 0, self.ax[0], self.ax[1]), r * sa * np.where(sa < 0, self.ay[0], self.ay[1])], 1)
        X = np.hstack([self.c[:2] + q, np.full((n, 1), self.c[2])])
        return pl.rigid(i, X)

    def weight(self, pl, i, shape):
        H, W = shape[:2]
        A, o = self._affine(pl, i)
        ring = self.anchors(pl, i, 64, self.r_out * 1.02)
        x0, y0 = np.maximum(0, np.floor(ring.min(0)).astype(int) - 1)
        x1, y1 = np.ceil(ring.max(0)).astype(int) + 2
        x1, y1 = min(W, x1), min(H, y1)
        w = np.zeros((H, W), np.float32)
        if x1 - x0 < 2 or y1 - y0 < 2: return w
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        q = np.stack([xx - o[0], yy - o[1]], -1) @ np.linalg.inv(A).T
        w[y0:y1, x0:x1] = smootherstep((self.r_out - self.rho(q)) / max(1e-6, self.r_out - self.r_in))
        return w


def fold_det(src, dst, anchors, wmap, lam, step=4, a_src=None, inner=None, grow=1.4):
    """minimum Jacobian determinant of the backward map x -> x + w(x) D(x) on a coarse grid (step px). < 0 = the map
    folds: content repeats (a smeared second lip on the chin, a doubled edge) and swims as the fold moves. A large
    displacement D across a falloff narrower than ~2 |D| folds even though D and w are each smooth.
    inner: the target inner-lip polygon (frame px); it is skipped (grown x grow, i.e. the lips themselves too): opening a shut take maps the whole cavity
    onto the lip line by construction, and that area is the interior fill's job."""
    ys, xs = np.nonzero(wmap > 0)
    if len(xs) == 0: return 1.0
    yy, xx = np.mgrid[ys.min():ys.max() + 1:step, xs.min():xs.max() + 1:step].astype(np.float64)
    if yy.shape[0] < 3 or yy.shape[1] < 3: return 1.0
    P = np.stack([xx.ravel(), yy.ravel()], 1)
    a_s = anchors if a_src is None else a_src
    D = tps_field(np.vstack([dst, anchors]), np.vstack([src - dst, a_s - anchors]), P, lam).reshape(yy.shape + (2,))
    w = wmap[yy.astype(int), xx.astype(int)]
    mx, my = w * D[..., 0], w * D[..., 1]
    dxx, dxy = np.gradient(mx, step, axis=1), np.gradient(mx, step, axis=0)
    dyx, dyy = np.gradient(my, step, axis=1), np.gradient(my, step, axis=0)
    det = (1 + dxx) * (1 + dyy) - dxy * dyx
    if inner is not None:
        c = inner.mean(0)
        poly = ((inner - c) * grow + c).astype(np.float32).reshape(-1, 1, 2)
        keep = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) < 0 for x, y in P]).reshape(det.shape)
        det = det[keep] if keep.any() else det
    return float(det.min())


def fold_scale(det_at, thr=0.3, steps=(1.0, 0.85, 0.7, 0.55, 0.4, 0.25)):
    """largest share s of the residual whose warp keeps the Jacobian determinant >= thr (det_at(s) -> min det)."""
    for k in steps:
        if det_at(k) >= thr: return k
    return steps[-1]


def soft_sources(idx, sigma, floor=0.05):
    """per-frame source choice (ints, e.g. -1 = the frame's own mouth, j = bank entry / donor frame j) -> per-frame
    {source: weight}: one-hot per source, zero-phase gaussian over time, weights < floor dropped, renormalised. A switch
    of source becomes a cross-fade over ~4 sigma frames instead of a one-frame swap (sigma <= 0: hard switch)."""
    idx = np.asarray(idx, int)
    if sigma <= 0: return [{int(k): 1.0} for k in idx]
    keys = np.unique(idx)
    Wt = np.stack([gaussian_filter1d((idx == k).astype(float), sigma, mode='nearest') for k in keys], 1)
    out = []
    for i in range(len(idx)):
        w = np.where(Wt[i] >= floor, Wt[i], 0.0)
        if w.sum() <= 0: w = (keys == idx[i]).astype(float)
        w = w / w.sum()
        out.append({int(keys[k]): float(w[k]) for k in np.nonzero(w)[0]})
    return out


def smooth_residual(dF, sigma, keep_close=None):
    """zero-phase gaussian (frames) of the per-frame residual. The control displacement is linear in it (J dF, canonical
    head frame), so this smooths the WARP FIELD itself over time, not the landmarks. keep_close (bool per frame): those
    frames keep the full depth of a closing (PP)."""
    if sigma <= 0: return np.array(dF, float)
    r = gaussian_filter1d(np.asarray(dF, float), sigma, axis=0, mode='nearest')
    if keep_close is not None and np.any(keep_close):
        k = np.asarray(keep_close, bool)
        r[k, 0] = np.minimum(r[k, 0], np.asarray(dF)[k, 0])
    return r


def interior_alpha(gap, egap, more, mode='invented'):
    """how much of the opened mouth's interior is filled from the donor. 'invented' (r2): by the share of the target
    opening the take does NOT already show (1 - gap / target gap) -> a take already open keeps its own (stretched) teeth
    and cavity, a shut take gets the donor's; no double teeth from a donor laid over the take's own. 'more' (r1): by
    the opening added, in units of the take's range."""
    if mode == 'more': return float(np.clip((more - 0.08) / 0.3, 0, 1))
    inv = 1.0 - max(0.0, gap) / max(1e-6, egap)        # a narrow ramp: mostly own OR donor, rarely a see-through mix
    return float(smootherstep((inv - 0.3) / 0.15))      # (a half-blended dark cavity over the take's teeth reads grey)


class MouthBank:
    """the clip's own mouth states, harvested per face (keyed by face id): per entry the frame index, its features
    (gap, width), canonical control shape (stabilised head frame) and head pose. Pixels stay in the decoded clip and are
    re-sampled on demand (an entry = frame index + its frame-space points). One bank per face: today lsdub's tracker
    follows one face (num_faces=1 -> face id 0); a multi-face clip gets one MouthBank per tracked id."""

    def __init__(self, face_id, F, Xc, pose, S2):
        self.face_id, self.F, self.Xc, self.pose, self.S2 = face_id, F, Xc, pose, S2

    def nearest(self, i, ft, scale, pose_w=2.0, exclude=2, prev=-1):
        cost = np.max(np.abs(self.F - ft) / scale, axis=1) + pose_w * np.linalg.norm(self.pose - self.pose[i], axis=1)
        cost = cost + 9.0 * (np.abs(np.arange(len(self.F)) - i) <= exclude)
        if prev >= 0: cost[prev] -= 0.05                 # stickiness: fewer donor switches
        j = int(np.argmin(cost))
        return j, float(np.max(np.abs(self.F[j] - ft) / scale))


def al_kind(gap, egap, rng, thr):
    """True when the target is a closure (target opening below thr of the take's range)."""
    return (egap - rng['g_lo']) / max(1e-6, rng['g_hi'] - rng['g_lo']) < thr


def erase_alpha(gap, egap, rng, thr=0.12, reductions=False, width=None, ewidth=None, sigma=0.7):
    """per-frame strength (0..1) of the v4 erase layer. Closures: the target opening u is below thr (x the take's range)
    while the take itself shows more (a gap the warp would only compress: teeth line). reductions=True also layers
    frames whose target is clearly SMALLER than the take (gap or width), so the old lips never show around the new ones.
    Eased zero-phase (sigma frames)."""
    sc = max(1e-6, rng['g_hi'] - rng['g_lo'])
    u_t = (np.asarray(egap) - rng['g_lo']) / sc
    u_f = (np.asarray(gap) - rng['g_lo']) / sc
    a = smootherstep((thr - u_t) / max(1e-6, thr) * 2.0) * smootherstep((u_f - u_t - 0.05) / 0.15)
    if reductions:
        ws = max(1e-6, rng['w_hi'] - rng['w_lo'])
        red = np.maximum((u_f - u_t), 0.0 if width is None else (np.asarray(width) - np.asarray(ewidth)) / ws)
        a = np.maximum(a, smootherstep((red - 0.1) / 0.2))
    if sigma > 0: a = np.clip(gaussian_filter1d(a, sigma, mode='nearest'), 0, 1)
    return a


def most_closed(S, gap, pose, fit=None):
    """the take's most-closed frame among its well-tracked (fit > 0.7), near-median-pose (deviation <= its 40th
    percentile) frames: a turned head's foreshortened thin lips would be laid onto a frontal mouth."""
    pz = np.linalg.norm(pose - np.median(pose, 0), axis=1)
    g = np.asarray(gap, float)
    ok = pz <= np.percentile(pz, 40)
    if fit is not None and (ok & (np.asarray(fit) > 0.7)).any(): ok &= np.asarray(fit) > 0.7
    return int(np.flatnonzero(ok)[np.argmin(g[ok])])


IN_PAIRS = [(95, 191), (88, 80), (178, 81), (87, 82), (14, 13), (317, 312), (402, 311), (318, 310), (324, 415)]


def lip_frame(P):
    """outer lip loop (20) + the lips' contact line (corners 78 / 308 + the midpoints of the 9 paired inner points)."""
    P = np.asarray(P, float)[:, :2]
    mid = np.array([P[78]] + [(P[a] + P[b]) / 2 for a, b in IN_PAIRS] + [P[308]])
    return np.vstack([P[ll.LIP_OUT], mid])


class EraseLayer:
    """v4 "blank out the lips, then put ours in front" for the warp. ONE bald image (the take's most-closed frame with
    its lips erased to its own skin, lip_layers.erase_lips) and that frame's own closed lips are carried into each frame
    by a mesh: an outer ring that follows the rigid skull pose, and the lip points that land on the TARGET lip shape
    (dst). Layer = closed lips (soft lip-shaped mask) over the bald skin; it replaces the region where either the take's
    own lips or the target lips are (both hulls grown), blended in by the per-frame alpha. A per-frame Lab match on the
    skin band around the region (eased zero-phase at finish) keeps the layer on the frame's light."""

    def __init__(self, ref_img, ref_P, pl, r, ref3, grow=0.22, ring_r=1.9):
        self.lips_img = ref_img
        self.bald, _ = ll.erase_lips(ref_img, ref_P, grow)
        self.P = np.asarray(ref_P, float)
        self.pl, self.r, self.grow = pl, r, grow
        self.Cr = pl.to_lib(r, np.asarray(ref3, float))                       # the reference frame in the canonical frame
        c = self.P[ll.LIP_OUT].mean(0)
        hx = np.ptp(self.P[ll.LIP_OUT][:, 0]) / 2
        hy = max(np.ptp(self.P[ll.LIP_OUT][:, 1]) / 2, 0.5 * hx)
        a = np.linspace(-np.pi, np.pi, 20, endpoint=False)
        self.ring = c + np.stack([np.cos(a) * hx * ring_r, np.sin(a) * hy * ring_r * 1.7], 1)
        from scipy.spatial import Delaunay
        # the mesh: outer lip loop + the lips' CONTACT line (midpoints of the paired upper / lower inner-lip points) + the
        # ring. The inner loop itself is never a vertex set: on a closed mouth its upper and lower points nearly coincide,
        # and those sliver triangles flip under any change of shape (lips sampled from the wrong place)
        sp = np.vstack([lip_frame(self.P), self.ring])
        self.tris = Delaunay(sp).simplices
        self.lab, self.items = [], {}
        # the reference ring points in the canonical frame (orthographic lift on the lower-face points' plane)
        low = [k for k in range(478) if self.P[k, 1] > self.P[2, 1]]
        self.ring_c = pl.to_lib(r, mm.plane_lift(self.ring, np.asarray(ref3, float)[low]))

    def composite(self, out, i, P_take, P_dst, alpha, kind='closed'):
        H, W = out.shape[:2]
        ring_i = self.pl.rigid(i, self.ring_c)
        sp = np.vstack([lip_frame(self.P), self.ring])
        dp = np.vstack([lip_frame(P_dst), ring_i])
        x0, y0 = np.maximum(0, np.floor(dp.min(0)).astype(int) - 2)
        x1, y1 = np.minimum([W, H], np.ceil(dp.max(0)).astype(int) + 3)
        box = (x0, y0, x1 - x0, y1 - y0)
        if box[2] < 4 or box[3] < 4: return out, np.zeros((H, W), np.float32)
        bald, okm = mm.pwa(self.bald, sp, dp, self.tris, box)
        lw = np.linalg.norm(P_dst[61] - P_dst[291])
        rel = lambda Q: np.asarray(Q, float)[ll.LIP_OUT] - [x0, y0]
        shp = (box[3], box[2])
        if kind == 'closed':
            lips, _ = mm.pwa(self.lips_img, sp, dp, self.tris, box)
            fg = mp_.hull_mask(rel(P_dst), shp, 0.04, max(3, lw * 0.06))
        else:   # 'own': the warp output's own lips (and interior) at the target shape stay on top
            lips = out[y0:y1, x0:x1]
            fg = mp_.hull_mask(rel(P_dst), shp, 0.06, max(3, lw * 0.06))
        E = np.maximum(mp_.hull_mask(rel(P_dst), shp, self.grow + 0.1, 3), mp_.hull_mask(rel(P_take), shp, self.grow + 0.1, 3))
        E = cv2.GaussianBlur((E > 0.5).astype(np.float32), (0, 0), max(2.0, lw * 0.06)) * okm
        layer = fg[..., None] * lips.astype(np.float32) + (1 - fg[..., None]) * bald.astype(np.float32)
        reg = out[y0:y1, x0:x1]
        band = (E > 0.05) & (E < 0.6) & mm.skin_px(reg) & mm.skin_px(np.clip(layer, 0, 255).astype(np.uint8)) & (fg < 0.05)
        self.lab.append((i, mm.lab_stats(layer, reg.astype(np.float32), band)))
        self.items[i] = (box, layer, E * float(alpha))
        m = np.zeros((H, W), np.float32)
        m[y0:y1, x0:x1] = E * float(alpha)
        return out, m

    def eased_stats(self):
        """the per-frame Lab match stats of the layered frames, eased zero-phase over them: {frame: stats (9,)}."""
        if not self.items: return {}
        idx = np.array([k for k, _ in self.lab])
        st = np.array([s for _, s in self.lab], float)
        good = st[:, 2] > 0
        if good.sum() >= 2:
            for c in range(9): st[:, c] = gaussian_filter1d(np.interp(idx, idx[good], st[good, c]), 1.5, mode='nearest')
        return {i: s for (i, _), s in zip(self.lab, st)}

    def stat_for(self, i):
        """v9 (re-made frame): the round-0 eased stats of frame i; a frame that was not layered in round 0 takes them
        interpolated from the layered frames around it."""
        ez = getattr(self, 'eased', {})
        if i in ez: return ez[i]
        ks = np.array(sorted(ez))
        V = np.array([ez[k] for k in ks], float)
        return np.array([np.interp(i, ks, V[:, c]) for c in range(V.shape[1])]) if len(ks) else np.zeros(9)

    def apply_item(self, i, img, s):
        """the Lab-matched layer of frame i over img -> a new image."""
        (x0, y0, w, h), layer, E = self.items[i]
        lay = mm.apply_lab(layer, s) if s[2] > 0 else layer
        reg = img[y0:y0 + h, x0:x0 + w].astype(np.float32)
        o = img.copy()
        o[y0:y0 + h, x0:x0 + w] = np.clip(lay * E[..., None] + reg * (1 - E[..., None]) + 0.5, 0, 255).astype(np.uint8)
        return o

    def finish(self, outs):
        """Lab match (eased over the layered frames) and composite into outs, in place."""
        if not self.items: return
        self.eased = self.eased_stats()
        for (i, _) in self.lab:
            outs[i] = self.apply_item(i, outs[i], self.eased[i])


LOWER_FACE_REF = mm.ANCH_STRICT + [2, 98, 327, 205, 425, 50, 280, 132, 361, 172, 397, 152, 148, 377]


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]       # MediaPipe's face outline, in order


def face_region(img, P, io, box):
    """v12: bool mask over box (x0, y0, x1, y1) of where the take's FACE is at this frame: the skull pose's face (the projected
    face mesh's outline, FACE_OVAL, filled) and, beyond it, the take's own pixels: skin (mouth_morph.skin_px) or its own mouth
    (outer lip + 0.03 IOD), specks opened (0.012 IOD), gaps closed (0.03 IOD), only the piece that holds the mouth, holes
    filled; all within the mesh hull grown 0.3 IOD (a skin-coloured background beside the head never counts as face).
    (Lipness is NOT a face cue here: on mv30 NC the auburn hair reads as lip-coloured, lipness >= 0.35, over most of it.)"""
    from . import teeth_layer as tl
    x0, y0, x1, y1 = box
    sub = img[y0:y1, x0:x1]
    h, w = sub.shape[:2]
    Pl = np.asarray(P, float)[:, :2] - [x0, y0]
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    face = mm.skin_px(sub).astype(np.uint8)
    face = cv2.morphologyEx(face, cv2.MORPH_OPEN, ker(max(1, int(round(0.012 * io)))))
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, ker(max(1, int(round(0.03 * io)))))
    own = tl.poly_mask(Pl[LIP_OUT], (h, w), 0.0, max(1.0, 0.03 * io))
    face = (face > 0) | own
    nl, lbl = cv2.connectedComponents(face.astype(np.uint8))
    face = np.isin(lbl, [k for k in np.unique(lbl[own]) if k > 0])
    inv = (~face).astype(np.uint8)
    nl, lbl = cv2.connectedComponents(inv)
    edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
    face |= (inv > 0) & ~np.isin(lbl, edge)                                    # holes (nostril, teeth, a dark crease) filled
    hull = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.round(Pl).astype(np.int32)), 1)
    grown = cv2.dilate(hull, ker(max(1, int(round(0.3 * io)))))
    # the skull pose's own face (the projected mesh's outline, FACE_OVAL) is face whatever its pixels read (a cheek in shadow
    # fails skin_px); the pixels extend it to the real silhouette where the turned head's mesh ends inside the cheek (NC)
    mesh = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mesh, [np.round(Pl[FACE_OVAL]).astype(np.int32)], 1)
    return (face | (mesh > 0)) & (grown > 0)


class YawClamp:
    """v12 (--warp-yaw-clamp M, opt-in): the PLAN keeps the mouth inside the face under yaw. mv30 NC f74-76: the turned
    head's FAR corner of a wider, more open planned mouth crossed the cheek's silhouette onto the hair; the lip layer then
    drew the warped hair there and its non-lip cleanup painted it over with skin (a mottled patch with dark strokes).
    Per frame an allowed region = the take's own face (face_region: the skull pose's face mesh outline, extended by the take's
    skin pixels and its own mouth). The far half of the mouth = the side of the corner nearer the nose tip in the picture. A
    planned outer-lip point there (the control polygon, densified x4; t >= 0.15 of the way from the centre line to the far
    corner) closer than the lip layer's own margin (0.02 IOD) + M IOD to the region's edge is out; the far half is then
    CLAMPED toward the take's own outline (which is face by construction): dst += b w(t) (src - dst), w = smoothstep(t /
    0.6) (0 on the centre line, 1 from 0.6 of the way to the corner), with the least b in 0..1 that brings every point in
    (bisection; b = 1 when even the take's own outline only touches the margin). b is eased over the clip (max with a
    gaussian of its 3-frame max: never less than a frame needs, no one-frame pop); a re-made frame (the gate) takes the
    eased b or what its own outline needs, whichever is larger. On the frames it clamps, the layer stack's corner cover
    (--warp-cover-corners: the residue fill and the old-lip blobs, the corner face guard) takes the take's own skin / lips as
    its face test (face_box; v11's lipness test reads the auburn hair as face and painted skin over it). A frame whose plan
    stays inside the face is untouched (a frontal shot renders bit-for-bit as without the flag)."""

    def __init__(self, clip, margin, src, dst):
        self.c, self.m = clip, float(margin)
        n = clip.n
        ix = {k: j for j, k in enumerate(CTRL)}
        self.ix = ix
        self.out = [ix[k] for k in LIP_OUT]
        self.reg = {}
        need = np.array([self.solve(i, src[i], dst[i]) for i in range(n)])
        from scipy.ndimage import maximum_filter1d
        self.b = np.clip(np.maximum(need, gaussian_filter1d(maximum_filter1d(need, 3, mode='nearest'), 1.0, mode='nearest')), 0, 1)
        self.b[self.b < 0.005] = 0.0
        self.need = need

    def io(self, i):
        return float(np.linalg.norm(self.c.S[i][33, :2] - self.c.S[i][263, :2]))

    def region(self, i):
        """(x0, y0, distance-to-the-edge map px) of frame i's allowed region, cached."""
        if i not in self.reg:
            P = self.c.S[i][:, :2]
            io = self.io(i)
            c = P[[13, 14, 61, 291]].mean(0)
            H, W = self.c.H, self.c.W
            x0, y0 = max(0, int(c[0] - 1.3 * io)), max(0, int(c[1] - 0.9 * io))
            x1, y1 = min(W, int(c[0] + 1.3 * io) + 1), min(H, int(c[1] + 1.1 * io) + 1)
            f = face_region(self.c.frames[i], P, io, (x0, y0, x1, y1)).astype(np.uint8)
            d = cv2.distanceTransform(cv2.copyMakeBorder(f, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=1), cv2.DIST_L2, 5)[2:-2, 2:-2]
            self.reg[i] = (x0, y0, d.astype(np.float16))
        return self.reg[i]

    def far_t(self, i, dst):
        """per control point: t = 0 on the mouth's centre line (outer-lip mids 0 / 17) .. 1 at the far corner, along the
        corner line (< 0: the near half); the far corner = the one nearer the nose tip (1) in the take's frame i."""
        P = self.c.S[i][:, :2]
        ix = self.ix
        e0 = P[291] - P[61]
        sg = 1.0 if abs(float((P[291] - P[1]) @ e0)) < abs(float((P[61] - P[1]) @ e0)) else -1.0
        c = 0.5 * (dst[ix[0]] + dst[ix[17]])
        e = dst[ix[291]] - dst[ix[61]]
        e = e / max(1e-6, np.linalg.norm(e))
        s = sg * ((dst - c) @ e)
        return s / max(1e-6, float(s[ix[291] if sg > 0 else ix[61]])), c, e, sg

    def pull(self, i, src, dst, b):
        if b <= 0: return dst
        t = np.clip(self.far_t(i, dst)[0] / 0.6, 0.0, 1.0)
        w = t * t * (3 - 2 * t)
        return dst + (b * w)[:, None] * (src - dst)

    def clearance(self, i, dst, t_ref=None):
        """min over the far half's outer-lip samples (t >= 0.15, t of the unclamped plan) of the distance to the region's
        edge minus the margin, px (>= 0: inside)."""
        x0, y0, d = self.region(i)
        io = self.io(i)
        Q = dst[self.out]
        t = (self.far_t(i, dst)[0] if t_ref is None else t_ref)[self.out]
        u = np.linspace(0, 1, 4, endpoint=False)[:, None]
        R = np.concatenate([Q * (1 - x) + np.roll(Q, -1, 0) * x for x in u[:, 0]])
        tt = np.concatenate([t * (1 - x) + np.roll(t, -1) * x for x in u[:, 0]])
        R = R[tt >= 0.15]
        if not len(R): return 1e9
        q = np.round(R - [x0, y0]).astype(int)
        inb = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < d.shape[1]) & (q[:, 1] < d.shape[0])
        if not inb.any(): return 1e9
        return float(d[q[inb, 1], q[inb, 0]].astype(np.float32).min()) - (0.02 + self.m) * io

    def solve(self, i, src, dst):
        """the least b in 0..1 that puts the far half inside (0: already inside; 1: the take's own far half)."""
        t0 = self.far_t(i, dst)[0]
        if self.clearance(i, dst, t0) >= 0: return 0.0
        if self.clearance(i, self.pull(i, src, dst, 1.0), t0) < 0: return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(7):
            mid = 0.5 * (lo + hi)
            if self.clearance(i, self.pull(i, src, dst, mid), t0) >= 0: hi = mid
            else: lo = mid
        return float(hi)

    def face_box(self, i, box):
        """the corner cover's face test (LayerCompositor.face_of) over a layer box (x0, y0, w, h) of frame i, on the frames the
        clamp acts on (b > 0): the take's own pixels only, strictly -- skin (mouth_morph.skin_px) or its own lips (outer lip
        + 0.01 IOD); no mesh, no closing (the dark hair band beside the lips is not face, so nothing paints skin over it).
        None elsewhere (the layer stack keeps v11's own face test there)."""
        if self.b[i] <= 0: return None
        from . import teeth_layer as tl
        bx, by, w, h = box
        img = self.c.frames[i][by:by + h, bx:bx + w]
        P = self.c.S[i][:, :2] - [bx, by]
        return mm.skin_px(img) | tl.poly_mask(P[LIP_OUT], (h, w), 0.0, max(1.0, 0.01 * self.io(i)))

    def apply(self, i, src, dst, ent=None):
        """frame i's planned control points (CTRL, frame px) from src (the take's), clamped as needed; ent: log entry."""
        b = float(self.b[i])
        out = self.pull(i, src, dst, b)
        if self.clearance(i, out, self.far_t(i, dst)[0]) < 0:
            b = max(b, self.solve(i, src, dst))
            out = self.pull(i, src, dst, b)
        if ent is not None and b > 0:       # (+ the clamped control points: the offline checker's replay of warp.json uses them)
            ent['yaw_clamp'] = round(b, 3)
            ent['yaw_clamp_px'] = round(float(np.abs(out - dst).max()), 2)
            ent['yaw_clamp_dst'] = np.round(out, 2).tolist()
        return out


def _track(a, n):
    """MediaPipe 3D track of the take; --warp-track-cache reuses a saved one (same video, same frame count)."""
    import os
    cp = getattr(a, 'warp_track_cache', None)
    if cp and os.path.exists(cp):
        z = np.load(cp)
        if int(z['n']) == n and str(z['video']) == a.video and str(z.get('mode', 'full')) == getattr(a, 'warp_track_mode', 'full'): return z['S3'], float(z['fps']), float(z['found'])
    S3, fps, found = mp_.track(a.video, n, z=True, smooth='none', mode=getattr(a, 'warp_track_mode', 'full'))
    if cp: np.savez(cp, S3=np.asarray(S3), fps=fps, found=found, n=n, video=a.video, mode=getattr(a, 'warp_track_mode', 'full'))
    return np.asarray(S3), fps, found


class WarpClip:
    """v9: the --mode warp pipeline as an object, so one frame can be re-made without re-running the clip.

      __init__      the whole-clip analysis up to the per-frame loop (track, pose, mouth model, targets, size cap, fold
                    guard, donors)
      run_batch()   round 0 = the v8 batch: every frame warped, then the teeth / layer stack, the erase layer and the lip
                    lock over all frames, in the same order and with the same arithmetic as the pre-v9 run()
                    (bit-identical output)
      render_frame(i, ov)   frame i through the whole chain again (warp -> layers -> erase -> lip lock) with per-frame
                    parameter OVERRIDES ov; the clip-level series (pins, jaw plan, eased layer parameters, Lab and
                    lip-lock fits) stay as round 0 computed them unless an override replaces frame i's own value
      write()       the video, the masks and warp.json
    Overrides (all optional, frame i only): layers_off (v11: the frame without the layer stack -- the plain own-pixel
    warp, then the erase layer and the lip lock as usual), S (478 x 3 landmarks of the take at i), dFe (the applied residual), ell
    (another HeadEllipse: falloff / ring), lam (TPS regularisation), jaw (planned jaw angle, rad), dth (the chin pin's
    extra jaw angle, rad), pin (upper-lip-root pin, px), open_s / tg (eased opening share / tongue pose), light (the
    frame's mouth-surround light), cover_grow (the L1 cover's erase growth), erase_al (erase alpha), lock_st (the lip
    lock's per-frame colour state)."""

    def __init__(self, a):
        g = lambda k, d: getattr(a, k, d)
        self.a, self.g = a, g
        T = json.load(open(a.track))
        fr_t = T['frames']
        n = a.frames or len(fr_t)
        S3raw, fps, found = _track(a, n)
        n = min(n, len(S3raw))
        S3raw = S3raw[:n]
        S = mp_.smooth_track(S3raw, g('track_smooth', 'k5'))
        ref = S3raw[int(np.argmin(np.abs(np.arange(n) - n // 2)))]
        pl = mm.RigidPlacement(ref, S3raw, mm.MESH_ANCH, smooth=a.warp_pose_smooth, lam=0.0)
        C = np.array([pl.to_lib(i, S[i]) for i in range(n)])                        # canonical head frame, all 478
        F = features(C)
        model = MouthModel(C[:, CTRL], F)
        ix = {k: j for j, k in enumerate(CTRL)}
        iin = [ix[k] for k in LIP_IN]
        if g('warp_jaw', True): model.jaw_reference([ix[k] for k in UPPER])
        if g('warp_jaw_gain', 1.0) != 1.0:      # the soft chin's share of an opening / closing (jaw drop / raise)
            model.J = model.J.copy()
            model.J[0, [ix[k] for k in CHIN_C]] *= a.warp_jaw_gain
        bands = library_bands(a.lib) if a.lib else None
        if g('warp_class_width', 0.0) > 0: bands = class_floor(bands, a.warp_class_width, model.rng, g('warp_class_ref', 'half'))
        # the residual is taken against the zero-phase SMOOTHED observed shape: landmark noise in F is measurement noise, not
        # pixel motion, and a residual computed from raw F would re-inject it into the field (the r1 slosh). The take's own
        # small motion stays in its pixels; only the smooth (target - shape) correction is added on top.
        Fs = gaussian_filter1d(F, g('warp_feat_smooth', 0.0), axis=0, mode='nearest') if g('warp_feat_smooth', 0.0) > 0 else F
        pp = np.array([fr_t[min(i, len(fr_t) - 1)]['vis'] == 'PP' for i in range(n)])
        # ---- v10c (opt-in): --warp-target coart = the coarticulation planner (lsdub/core/coart.py) instead of the per-frame
        # step targets: one planned trajectory (dominance-blended phone targets, duration undershoot, fitted to the take's own
        # motion statistics and shape manifold, closures hard; the jaw / lip split set on model.J); pp = its closure frames
        if g('warp_target', 'v9') == 'coart':
            from . import coart
            Ft, pp = coart.plan_clip(a, fr_t, Fs, model, bands, pp, F, S3raw)
        else:
            Ft = target_features(fr_t, Fs, model.rng, a.warp_open_gain, a.warp_width_gain, a.warp_sigma, bands, g('warp_centre', 0.0),
                                 g('warp_width_mix', 'switch'), g('warp_close_depth', 0.0))
        # ---- end v10c
        dl = np.array([np.linalg.norm(S[i][33, :2] - S[i][168, :2]) for i in range(n)])
        dr = np.array([np.linalg.norm(S[i][263, :2] - S[i][168, :2]) for i in range(n)])
        self.fs = gaussian_filter1d(2 * np.minimum(dl, dr) / np.maximum(dl + dr, 1e-6), 1.0, mode='nearest')   # v18: 1 frontal, -> 0 profile
        if g('warp_yaw_width', 0.0) > 0:
            # v18: the planned width follows the head turn. A turned head shows its mouth foreshortened; the plan's width
            # comes from the take's whole-clip range, so on a near-profile frame it asked for the frontal width and the far
            # corner went onto the cheek (sky B f61-63). Per frame the foreshortening = the eyes' balance about the nose
            # bridge (1 frontal, -> 0 in profile), relative to the clip's median; the width target is scaled by it.
            fs = self.fs
            k = np.clip(fs / np.median(fs), 0.6, 1.0) ** g('warp_yaw_width', 1.0)
            Ft = Ft.copy()
            Ft[:, 1] = np.minimum(Ft[:, 1], np.maximum(Fs[:, 1], Ft[:, 1] * k))
            print(f'yaw width: {int((k < 0.97).sum())} frames narrowed (min x{k.min():.2f}) {np.where(k < 0.97)[0].tolist()}')
        dF = Ft - Fs
        scale = np.array([max(1e-6, model.rng['g_hi'] - model.rng['g_lo']), max(1e-6, model.rng['w_hi'] - model.rng['w_lo'])])
        conf = confidence(dF, scale, a.warp_deadband, a.warp_deadband + g('warp_ramp', 0.08)) if a.warp_deadband >= 0 else np.ones(n)
        conf = gaussian_filter1d(conf, 1.0, mode='nearest')
        dFe = dF * conf[:, None] * a.warp_gain
        dFe = smooth_residual(dFe, g('warp_temporal', 0.0), pp)                     # the field itself, over time
        tw = g('warp_temporal_width', 0.0)                                           # width (corners -> cheeks): slower
        if tw > g('warp_temporal', 0.0): dFe[:, 1] = smooth_residual(dFe[:, 1:], tw)[:, 0] if g('warp_temporal', 0.0) <= 0 \
            else gaussian_filter1d(dFe[:, 1], float(np.sqrt(tw ** 2 - g('warp_temporal', 0.0) ** 2)), mode='nearest')
        cap_k = g('warp_size_cap', 0.0)
        if cap_k > 0:
            # v4 size cap: the new mouth is never bigger than this face's own mouth would be. Per dimension the target stays
            # within the take's own p95 opening and p5..p95 width (x cap_k), unless the take's own frame is already past it
            # (that frame IS the face's own mouth and is never shrunk by the cap); the cut is eased zero-phase
            Fe0 = F + dFe
            rr = model.rng
            Fe0[:, 0] = ll.size_cap(Fe0[:, 0], F[:, 0], cap_k * rr['g_hi'])
            Fe0[:, 1] = ll.size_cap(Fe0[:, 1], F[:, 1], cap_k * rr['w_p95'], rr['w_p5'] / cap_k)
            ncap = int((np.abs(Fe0 - F - dFe) > 0.5).any(1).sum())
            dFe = Fe0 - F
            print(f'size cap x{cap_k}: gap <= {cap_k * rr["g_hi"]:.1f}, width {rr["w_p5"] / cap_k:.1f}..{cap_k * rr["w_p95"]:.1f} '
                  f'(canonical px); {ncap} frames cut by > 0.5 px')
        Fe = F + dFe                                                                 # effective target features
        rot = np.array([pl.s[i] * pl.R[i] for i in range(n)])                        # canonical -> frame linear part
        pose = pl.par[:, 1:4]
        interp = cv2.INTER_LANCZOS4 if g('warp_interp', 'cubic') == 'lanczos' else cv2.INTER_CUBIC
        ell = None
        if g('warp_falloff', 'hull') == 'ellipse':
            Xall = np.concatenate([C[:, CTRL], C[:, CTRL] + np.einsum('tf,fkd->tkd', dFe, model.J)])
            Xall = Xall[:, [ix[k] for k in LIP_OUT + LIP_IN + [18, 200]]]            # the chin-tip points stay in the falloff
            ell = HeadEllipse(Xall, C[:, [13, 14, 61, 291]].mean((0, 1)), g('warp_core', 1.0), a.warp_ring or 2.6)
        ring_f = a.warp_ring or 1.6
        cap = cv2.VideoCapture(a.video)
        W, H = int(cap.get(3)), int(cap.get(4))
        frames = []
        for i in range(n):
            ok, fr = cap.read()
            if not ok: break
            frames.append(fr)
        cap.release()
        n = len(frames)
        self.fr_t, self.n, self.S3raw, self.fps, self.found, self.S, self.pl, self.C, self.F, self.model = fr_t, n, S3raw, fps, found, S, pl, C, F, model
        self.ix, self.iin, self.Ft, self.scale, self.conf, self.Fe, self.rot, self.pose = ix, iin, Ft, scale, conf, Fe, rot, pose
        self.interp, self.ell, self.ring_f, self.W, self.H, self.frames = interp, ell, ring_f, W, H, frames
        self.dFe_pre_fold = dFe.copy()                                               # (v9: the residual before the fold guard)
        fscale = np.ones(n)
        if g('warp_fold_guard', 0.0) > 0 and ell is not None:
            # fold guard: per frame, the largest share of the residual whose warp does not fold; min-filtered (a fold's
            # neighbours give way too) and eased zero-phase so the guard itself does not pop
            from scipy.ndimage import minimum_filter1d
            for i in range(n):
                src = S[i][CTRL, :2]
                if np.abs(dFe[i] / scale).max() < 0.05: continue
                def det_at(k, i=i, src=src):
                    dst = src + (model.delta(k * dFe[i]) @ rot[i].T)[:, :2]
                    an, wm = self.geometry(i, src, dst)
                    return fold_det(src, dst, an, wm, a.warp_lam, inner=dst[iin])
                fscale[i] = fold_scale(det_at, a.warp_fold_guard)
            fscale = np.minimum(fscale, gaussian_filter1d(minimum_filter1d(fscale, 3, mode='nearest'), 1.0, mode='nearest'))
            dFe = dFe * fscale[:, None]
            Fe = F + dFe
        self.fscale, self.dFe, self.Fe = fscale, dFe, Fe
        self.yc = None
        if g('warp_yaw_clamp', 0.0) > 0:
            # v12 (opt-in): the planned outline kept inside the take's face under yaw (YawClamp: the far half clamped)
            self.yc = YawClamp(self, a.warp_yaw_clamp, [S[i][CTRL, :2] for i in range(n)],
                               [S[i][CTRL, :2] + (model.delta(dFe[i]) @ rot[i].T)[:, :2] for i in range(n)])
            fz = np.flatnonzero(self.yc.b > 0)
            print(f'yaw clamp (margin 0.02 + {a.warp_yaw_clamp} IOD): {len(fz)} frames clamped {fz.tolist()}; share of the take\'s far '
                  f'half up to {self.yc.b.max():.2f} (needed {self.yc.need.max():.2f})')
        bank = MouthBank(0, F, C[:, CTRL], pose, S[:, :, :2])
        self.bank = bank
        # per frame: (a) correct the frame's own mouth, or (b) pull the bank entry nearest the target (shape, then pose)
        # when it is much closer to the target than this frame is (and this frame needs a big correction)
        choice = np.array(['a'] * n, object)
        src_j = np.full(n, -1)
        prevb = -1
        for i in range(n):
            need = float(np.max(np.abs(dFe[i]) / scale))
            if a.warp_bank and need > a.warp_bank_thr:
                j, dj = bank.nearest(i, Fe[i], scale, a.warp_pose_w, prev=prevb)
                if dj < need - 0.15:
                    choice[i] = 'b'
                    src_j[i] = prevb = j
        # donor choice for the interior: nearest in (gap, pose) among the take's frames. r2: ONE donor per opening segment
        # (a run of frames that need a fill), the one with the least summed cost over the run, so the teeth never swap
        # mid-opening and no two donors' teeth are ever cross-faded (that reads as grey teeth); r1 ('frame'): per frame
        need_fill = Fe[:, 0] > F[:, 0] + 0.08 * scale[0]
        def dcost(i):
            return np.abs(F[:, 0] - Fe[i, 0]) / scale[0] + a.warp_pose_w * np.linalg.norm(pose - pose[i], axis=1) \
                + 0.5 * (np.abs(np.arange(n) - i) < 2)                               # not itself / its immediate neighbours
        donors = np.full(n, -1)
        if g('warp_donor_hold', 'frame') == 'segment':
            i = 0
            while i < n:
                if not need_fill[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and need_fill[j + 1]: j += 1
                tot = sum(dcost(k) for k in range(i, j + 1))
                tot[i:j + 1] += 9.0     # not a frame of the run itself
                donors[i:j + 1] = int(np.argmin(tot))
                i = j + 1
        else:
            prev = -1
            for i in range(n):
                if not need_fill[i]: continue
                cost = dcost(i)
                if prev >= 0: cost[prev] -= g('warp_donor_stick', 0.05)
                donors[i] = prev = int(np.argmin(cost))
        xf = g('warp_xfade', 0.0)
        msrc, mdon = soft_sources(src_j, xf), soft_sources(donors, xf if g('warp_donor_hold', 'frame') == 'frame' else 0.0)               # cross-fades between sources / donors
        self.choice, self.src_j, self.donors, self.msrc, self.mdon = choice, src_j, donors, msrc, mdon
        self.rigid_teeth = g('warp_teeth', 'off') == 'rigid'      # v5: the interior comes from the rigid teeth layer, no donors
        self.teeth = self.comp = self.erase = self.lock = None
        self.er_al = None
        self.md = None
        self.fit_t = None
        self.warped, self.outs, self.masks, self.log, self.dsts, self.P_out = [None] * n, [None] * n, [None] * n, [None] * n, [None] * n, [None] * n

    # ------------------------------------------------------------------------------------------------ geometry --
    def geometry(self, i, src, dst, ell=None):
        ell = self.ell if ell is None else ell
        if ell is not None: return ell.anchors(self.pl, i), ell.weight(self.pl, i, (self.H, self.W))
        mc = self.S[i][[13, 14, 61, 291]].mean(0)[:2]
        return ring_anchors(np.vstack([src, dst]), mc, self.ring_f), None

    def dst_of(self, i, dFe_i=None, Si=None):
        """the target control points of frame i (frame px) for the applied residual dFe_i, from landmarks Si."""
        Si = self.S[i] if Si is None else Si
        dFe_i = self.dFe[i] if dFe_i is None else dFe_i
        src = Si[CTRL, :2]
        dst = src + (self.model.delta(dFe_i) @ self.rot[i].T)[:, :2]
        return src, (dst if getattr(self, 'yc', None) is None else self.yc.apply(i, src, dst))      # v12: --warp-yaw-clamp

    def fold_det_of(self, i, dFe_i=None, Si=None, ell=None, lam=None):
        """v9 (the gate's fold ruler): the minimum Jacobian determinant of frame i's applied warp (1.0: no move)."""
        src, dst = self.dst_of(i, dFe_i, Si)
        if np.abs(dst - src).max() < 0.25: return 1.0
        an, wm = self.geometry(i, src, dst, ell)
        if wm is None: return 1.0
        return fold_det(src, dst, an, wm, self.a.warp_lam if lam is None else lam, inner=dst[self.iin])

    # ---------------------------------------------------------------------------------------------------- warp --
    def warp_one(self, i, ov=None):
        """frame i's warp (mode a: its own mouth moved to the target; mode b: a bank entry) -> (image, weight map, log
        entry, dst control points or None)."""
        ov = ov or {}
        a, g = self.a, self.g
        S, C, F, model, pl = self.S, self.C, self.F, self.model, self.pl
        H, W = self.H, self.W
        Si = ov.get('S', S[i])
        dFe_i = ov.get('dFe', self.dFe[i])
        Fe_i = F[i] + dFe_i
        ell = ov.get('ell', None)
        lam = ov.get('lam', a.warp_lam)
        scale = self.scale
        iin = self.iin
        frames = self.frames
        msrc, mdon = self.msrc, self.mdon
        fr = frames[i]
        src = Si[CTRL, :2]
        acc = np.zeros(fr.shape, np.float32)
        wacc = np.zeros((H, W), np.float32)
        ent = dict(i=i, mode=self.choice[i], gap=round(float(F[i, 0]), 2), width=round(float(F[i, 1]), 2), tgap=round(float(self.Ft[i, 0]), 2),
                   twidth=round(float(self.Ft[i, 1]), 2), egap=round(float(Fe_i[0]), 2), ewidth=round(float(Fe_i[1]), 2), conf=round(float(self.conf[i]), 3),
                   src=int(self.src_j[i]), donor=int(self.donors[i]), fold_scale=round(float(self.fscale[i]), 2),
                   donors={str(k): round(v, 2) for k, v in mdon[i].items()}, mix={str(k): round(v, 2) for k, v in msrc[i].items()}, move_px=0.0)
        dst_i = None
        for k, wk in msrc[i].items():
            if k >= 0:   # bank entry k: its canonical shape corrected to the target by J, projected with THIS frame's pose
                Xt = C[k, CTRL] + model.delta(Fe_i - F[k])
                dst = pl.rigid(i, Xt) - pl.rigid(i, C[i, CTRL]) + src
                if getattr(self, 'yc', None) is not None: dst = self.yc.apply(i, src, dst, ent)        # v12
                anchors, wmap = self.geometry(i, src, dst, ell)
                M, _ = cv2.estimateAffinePartial2D(Si[LOWER_FACE_REF, :2].astype(np.float32), S[k][LOWER_FACE_REF, :2].astype(np.float32))
                a_src = anchors @ M[:, :2].T + M[:, 2]
                out, w = warp_frame(fr, S[k][CTRL, :2], dst, anchors, lam=lam, src_img=frames[k], anchors_src=a_src,
                                    weight=wmap, interp=self.interp)
            else:
                dl = model.delta(dFe_i)
                dst = src + (dl @ self.rot[i].T)[:, :2]
                if getattr(self, 'yc', None) is not None: dst = self.yc.apply(i, src, dst, ent)        # v12: inside the face
                if np.abs(dst - src).max() < 0.25:
                    out, w = fr, np.zeros((H, W), np.float32)
                else:
                    anchors, wmap = self.geometry(i, src, dst, ell)
                    out, w = warp_frame(fr, src, dst, anchors, lam=lam, weight=wmap, interp=self.interp)
                    more = (Fe_i[0] - F[i, 0]) / scale[0]
                    small = (Fe_i[0] - model.rng['g_lo']) / scale[0] < g('warp_donor_min', 0.0)
                    if a.warp_interior != 'stretch' and not self.rigid_teeth and more > 0.08 and not small and any(d >= 0 for d in mdon[i]):
                        al = interior_alpha(F[i, 0], Fe_i[0], more, g('warp_interior_mode', 'invented'))
                        base = out.astype(np.float32)
                        o2 = np.zeros_like(base)
                        for d, wd in ov.get('donors', mdon[i]).items():        # cross-faded donors (d = -1: no fill)
                            o2 += wd * (base if d < 0 else interior_fill(out, frames[d], S[d][LIP_IN, :2], dst[iin], al,
                                                                         a.warp_interior, feather=g('warp_feather', 0.35))[0].astype(np.float32))
                        out = np.clip(o2 + 0.5, 0, 255).astype(np.uint8)
                    if g('warp_seal', 0.0) > 0 and more < 0:          # a closing: seal the slit
                        out, sa = seal_fill(out, dst[iin], (Fe_i[0] - model.rng['g_lo']) / scale[0], a.warp_seal, g('warp_seal_grow', 1.6))
                        ent['seal'] = round(sa, 2)
            ent['move_px'] = max(ent['move_px'], round(float(np.abs(dst - src).max()), 2))
            if k < 0: dst_i = dst
            acc += wk * out.astype(np.float32)
            wacc += wk * w
        out = fr if not wacc.any() else np.clip(acc + 0.5, 0, 255).astype(np.uint8)
        if len(msrc[i]) == 1 and wacc.any(): out = np.where((wacc > 0)[..., None], out, fr)
        return out, wacc, ent, dst_i

    def p_out_of(self, i, dst, Si=None):
        """where the lips ARE in the warped frame i: the landmarks with the control points at dst."""
        P = (self.S[i] if Si is None else Si)[:, :2].copy()
        if dst is not None: P[CTRL] = dst
        return P

    # -------------------------------------------------------------------------------------------- layer stack --
    def setup_layers(self):
        """v5..v8: the rigid teeth / layer compositor, its pins (DIS flow of the round-0 warped frames), the jaw plan and
        the eased per-frame layer parameters."""
        _a, g, n = self.a, self.g, self.n
        frames, S, pl, C, F, Fe, model = self.frames, self.S, self.pl, self.C, self.F, self.Fe, self.model
        fr_t, P_out, outs = self.fr_t, self.P_out, self.warped
        # v5 rigid teeth: teeth are bone. The interior (inside the target inner-lip contour) is replaced by the rigid
        # teeth layer: upper plate at the skull pose, lower plate at skull + jaw angle (from the target opening), a dark
        # cavity behind; the warped lips occlude them. No stretched or per-segment donor teeth reach the output.
        from . import teeth_layer as tl_
        st_ = [ll.lip_stats(frames[i], S[i][:, :2]) for i in range(n)]
        okst = [x for x in st_ if x is not None]
        md = np.median(okst, 0) if okst else None
        fit_t = np.array([ll.lip_fit(frames[i], S[i][:, :2], md[1:3], md[3:5]) for i in range(n)]) if md is not None else None
        hy, hz = [float(x) for x in str(g('warp_teeth_hinge', '0.25,0.6')).split(',')]
        teeth = tl_.TeethLayer(frames, S, pl, C, F, (hy, hz), g('warp_teeth_jaw_gain', 1.0), fit_t, plates=g('warp_teeth_plates', 1),
                               tongue_drop=g('warp_teeth_tongue_drop', 0.0), throat=g('warp_teeth_throat', 0.0),
                               mirror=g('warp_teeth_mirror', False), **({'build': 'v12'} if g('warp_arch_build', 'v11') == 'v12' else {}))
        layered = g('warp_compose', 'plates') == 'layers'
        comp = None
        if layered:
            # v7: the mouth as an explicit layer stack (mouth_layers.py): frame, cover, cavity, lower arch + tongue, upper
            # arch, lips, face; full curved arches in head space, the lower one pinned to the chin skin
            from . import mouth_layers as ml_
            self.dbg_f = [int(x) for x in str(g('warp_layers_debug_frames', '') or '').split(',') if x.strip()]
            rim = [float(x) for x in str(g('warp_lip_rim', '0.22,0.012')).split(',')]
            comp = ml_.LayerCompositor(teeth, frames, S, None if md is None else md[1:3], None if md is None else md[3:5],
                                       g('warp_arch_k', 3.2), g('warp_arch_upper', 0.40), g('warp_arch_lower', 0.36),
                                       g('warp_arch_overjet', 0.03), g('warp_arch_overbite', 0.03), g('warp_tongue', 'on'),
                                       g('warp_cover', 'on'), bool(g('warp_layers_debug', None)), None,
                                       g('warp_teeth_pin_sigma', 4.0), g('warp_arch_expose', 0.03),
                                       # v8 (all default to v7's behaviour)
                                       pin_mode=g('warp_teeth_chin_pin', 'skin'), pin_gain=g('warp_chin_pin_gain', 0.4), pin_split=g('warp_chin_pin_split', 1.5),
                                       harvest_n=g('warp_arch_harvest', 1), texel=g('warp_arch_texel', 1.0), synth=bool(g('warp_arch_synth', False)),
                                       desat=g('warp_arch_desat', 0.0), shade=g('warp_arch_shade', 0.5), tex_from=g('warp_arch_tex_from', None), tex_out=g('warp_arch_tex_out', None),
                                       tex_min_h=g('warp_arch_tex_min_h', 0.05), tex_share=bool(g('warp_arch_tex_share', False)), rim=(rim[0], rim[1]), wet=g('warp_lip_wet', 0.0), smooth=g('warp_layers_smooth', 0.0),
                                       facing=g('warp_arch_facing', 0.0), tongue_lit=g('warp_tongue_lit', 1.0), harvest_lo=g('warp_arch_harvest_lower', None),
                                       # v11 (opt-in): the interior only through the warped lips; the debug view's stabilisation
                                       interior_clip=bool(g('warp_interior_clip', False)), dbg_stab=g('warp_layers_debug_stab', 'pin'),
                                       cover_corners=g('warp_cover_corners', 0.0), **({'build': 'v12'} if g('warp_arch_build', 'v11') == 'v12' else {}),
                                       # v13 (opt-in): the anthropometric slots, the buccal corridor
                                       **({'layout': 'anthro', 'anthro_unit': g('warp_arch_anthro_unit', 'ipd')} if g('warp_arch_layout', 'harvest') == 'anthro' else {}),
                                       **({'corridor': g('warp_arch_corridor', 0.0), 'corridor_floor': g('warp_arch_corridor_floor', 0.25)}
                                          if g('warp_arch_corridor', 0.0) > 0 else {}),
                                       # v14 (opt-in): area-correct minification of the arch atlas, the enamel material pass
                                       **({'mipmap': True} if g('warp_arch_mipmap', False) else {}),
                                       **({'enamel': _enamel_params(g('warp_arch_enamel', 0.0), g('warp_arch_enamel_mix', None))}
                                          if g('warp_arch_enamel', 0.0) > 0 else {}),
                                       **({'detail': g('warp_arch_detail', 0.0)} if g('warp_arch_detail', 0.0) > 0 else {}),
                                       # v15 (opt-in): pink gingiva above / below the crowns, and the tongue covering the lower crowns at mid-open
                                       **({'gums': g('warp_arch_gums', 0.0)} if g('warp_arch_gums', 0.0) > 0 else {}),
                                       **({'tongue_cover': g('warp_tongue_cover', 0.0)} if g('warp_tongue_cover', 0.0) > 0 else {}),
                                       # v16 (opt-in): the lower teeth as the dim, half-hidden, soft-bottomed sibling of the uppers
                                       **({'lower_real': g('warp_lower_real', 0.0), 'lower_mix': g('warp_lower_mix', None),
                                           'lower_ratio': g('warp_lower_ratio', 'auto')} if g('warp_lower_real', 0.0) > 0 else {}),
                                       # v17 (opt-in): the opening filled (tongue + lower row) instead of a dark void
                                       **({'cavity_fill': g('warp_cavity_strength', 1.0)} if g('warp_cavity', 'off') == 'fill' else {}))
        if comp is not None and g('warp_arch_band_log', False): comp.band_log = True      # v12: the lip-band ruler, logging only
        if comp is not None and getattr(self, 'yc', None) is not None: comp.face_of = self.yc.face_box     # v12: the cover's face test
        single = comp is not None and g('warp_reg', 'v11') == 'single'          # v12: one reference (mouth_layers.plan_single)
        gpin = g('warp_reg_pin', 'none') if single else 'none'                 # v12: + ONE skin residual for the whole interior
        if single and gpin == 'nose':                                            # (nose: the take's own head motion the pose misses)
            teeth.set_pin(frames, S, g('warp_teeth_pin_sigma', 4.0), idx=tl_.NOSE)
        elif g('warp_teeth_anchor', 'pose') == 'skin' and (not single or gpin == 'global'):
            # v6: the teeth take the upper-lip ROOT skin's own motion (DIS flow of the philtrum / nose base in the warped
            # frames), not the MediaPipe skull pose alone (that pose misses the skin by ~2x the flow floor and lags it)
            teeth.set_pin(outs, S, g('warp_teeth_pin_sigma', 4.0))
        if g('warp_teeth_jaw', 'frame') == 'plan':
            # v6: the jaw is planned over the whole clip at once (smooth, speed / acceleration limited, holds on sustained
            # vowels, lower teeth hidden below the lower lip on near-closed frames) instead of per frame from the target lip
            u_open = (Fe[:, 0] - model.rng['g_lo']) / max(1e-6, model.rng['g_hi'] - model.rng['g_lo'])
            teeth.plan([P_out[i][14] for i in range(n)], u_open, [fr_t[min(i, len(fr_t) - 1)]['vis'] for i in range(n)],
                       g('warp_teeth_hide', 0.2), g('warp_teeth_jaw_lead', 0.0), g('warp_teeth_jaw_vmax', 3.0), g('warp_teeth_jaw_amax', 1.0),
                       g('warp_teeth_jaw_smooth', 1.5), 0.43 * g('warp_teeth_jaw_smooth', 1.5), fit_t,
                       floor=comp.jaw_floor([P_out[i][14] for i in range(n)]) if comp is not None else None)
        if comp is not None and g('warp_teeth_chin_pin', 'skin') in ('skin', 'arc') and not single:
            comp.set_chin_pin(outs, P_out)          # v7: lower teeth take the chin skin's frame-to-frame motion (v8 arc: on the hinge arc)
        if comp is not None and (g('warp_layers_smooth', 0.0) > 0 or g('warp_tongue', 'on') == 'wedge'):
            comp.prepare(Fe[:, 0], [fr_t[min(i, len(fr_t) - 1)]['vis'] for i in range(n)])   # v8: eased per-frame layer parameters
        if single: comp.plan_single(P_out, g('warp_reg_jaw_sigma', 1.0), g('warp_reg_tongue_sigma', 2.0), g('warp_reg_lo_limit', 0.09), gpin)
        self.teeth, self.comp, self.md, self.fit_t = teeth, comp, md, fit_t

    def _patch(self, i, ov):
        """apply frame i's parameter overrides in place (pose, layer parameters, lip-lock state); returns the undo list."""
        undo = []
        t, c = self.teeth, self.comp
        def put(obj, name, val):
            arr = getattr(obj, name, None)
            if arr is None: return
            undo.append((arr, i, np.array(arr[i], copy=True)))
            arr[i] = val
        if 'pose' in ov:            # v9 lm_hold (re-detect): the skull pose of frame i from its repaired landmarks
            s_, R_, t3 = ov['pose']
            put(self.pl, 's', s_)
            put(self.pl, 'R', R_)
            put(self.pl, 't', t3)
            put(self, 'rot', s_ * np.asarray(R_))
        if 'jaw' in ov and t is not None and t.jaw_plan is not None: put(t, 'jaw_plan', ov['jaw'])
        if 'pin' in ov and t is not None: put(t, 'pin', ov['pin'])
        if 'light' in ov and t is not None: put(t, 'light', ov['light'])
        if c is not None:
            if 'dth' in ov: put(c, 'dth', ov['dth'])
            if 'open_s' in ov and c.open_s is not None: put(c, 'open_s', ov['open_s'])
            if 'tg' in ov and c.tg is not None: put(c, 'tg', ov['tg'])
        if 'lock_st' in ov and self.lock is not None and self.lock.st is not None: put(self.lock, 'st', ov['lock_st'])
        return undo

    @staticmethod
    def _unpatch(undo):
        for arr, i, v in reversed(undo): arr[i] = v

    def patched(self, i, ov):
        """context: frame i's overrides applied (e.g. a fold check under an overridden pose)."""
        import contextlib
        @contextlib.contextmanager
        def cm():
            undo = self._patch(i, ov)
            try: yield self
            finally: self._unpatch(undo)
        return cm()

    def layer_one(self, i, out, ov=None, rerender=False):
        """frame i through the teeth / layer stack -> (image, mask, log fields)."""
        ov = ov or {}
        teeth, comp, Fe, md = self.teeth, self.comp, self.Fe, self.md
        P_take = ov.get('S_take', ov.get('S', self.S[i]))[:, :2]
        P_out = ov.get('P_out', self.P_out[i])
        gap = float(self.F[i, 0] + ov['dFe'][0]) if 'dFe' in ov else Fe[i, 0]
        lg = {}
        if comp is not None and rerender:
            comp.cover_lip = [x for x in comp.cover_lip if x[0] != i]
            ncl = len(comp.clamps)
        if comp is not None and 'cover_grow' in ov: comp.cover_grow = ov['cover_grow']
        try:
            if comp is not None:
                out, tm = comp.composite(out, i, P_take, P_out, gap)
                lg['chin_pin'] = [round(float(x), 2) for x in comp.pin_log(i)]
                lg['lower_clamp_px'] = round(comp.clamp, 2)
                lg['jaw_dth_deg'] = round(float(np.degrees(comp.dth[i])), 3)
                lg['cover_lip'] = comp.cover_lip_of(i)
                lg['chin_pin_raw'] = [round(float(x), 2) for x in comp.pin_raw[i]]
                lg['arc_tan'] = [round(float(x), 3) for x in comp.arc_tangent(i, comp.jaw_R(i, gap, P_out[14], extra=False))]
                if i in comp.reg: lg['reg'] = comp.reg[i]          # v11: the inter-layer registration (lsdub layer-reg)
                if i in comp.corner_log: lg['corner_px'] = comp.corner_log[i]      # v11: corner residue px filled
            else:
                out, tm = teeth.render(out, i, P_out[LIP_IN], gap, None if md is None else md[1:3], None if md is None else md[3:5])
            lg['jaw_deg'] = round(float(np.degrees(teeth.jaw_at(i, gap, P_out[14]))), 2)
            lg['teeth_pose_alpha'] = round(teeth.pose_alpha(i), 2)
            lg['teeth_pin'] = [round(float(x), 2) for x in teeth.pin[i]]
            if teeth.jaw_plan is not None:
                lg['jaw_target_deg'] = round(float(np.degrees(teeth.jaw_target[i])), 2)
                if teeth.jaw_hide is not None:
                    lg['jaw_hide_deg'] = round(float(np.degrees(teeth.jaw_hide[i])), 2)
                    lg['jaw_hide_w'] = round(float(teeth.jaw_hide_w[i]), 3)
            if teeth.plate_w is not None: lg['plate_w'] = [round(float(x), 2) for x in teeth.plate_w[i]]
        finally:
            if comp is not None and 'cover_grow' in ov: comp.cover_grow = None
        if comp is not None and rerender and len(comp.clamps) > ncl and i < ncl:
            comp.clamps[i] = comp.clamps.pop()
        return out, tm, lg

    # --------------------------------------------------------------------------------------------- erase layer --
    def setup_erase(self):
        g, n, frames, S, F, Fe, model, pl = self.g, self.n, self.frames, self.S, self.F, self.Fe, self.model, self.pl
        er = g('warp_erase', 'off')
        al = erase_alpha(F[:, 0], Fe[:, 0], model.rng, g('warp_erase_thr', 0.12), er == 'all', F[:, 1], Fe[:, 1])
        # never paste at a mistracked frame: the take's lip landmarks must sit on lip-coloured pixels
        st_ = [ll.lip_stats(frames[i], S[i][:, :2]) for i in range(n)]
        okst = [x for x in st_ if x is not None]
        if okst:
            md = np.median(okst, 0)
            fit = np.array([ll.lip_fit(frames[i], S[i][:, :2], md[1:3], md[3:5]) for i in range(n)])
            al = al * smootherstep((fit - 0.6) / 0.2)
        else: fit = np.ones(n)
        r_ = most_closed(S, F[:, 0], self.pose, fit)
        self.erase = EraseLayer(frames[r_], S[r_][:, :2], pl, r_, self.S3raw[r_], grow=g('warp_erase_grow', 0.22))
        self.er_al, self.er_fit, self.er_ref = al, fit, r_

    def erase_one(self, i, out, ov=None, rerender=False):
        """frame i's erase layer (composite only; the eased Lab match is applied by the caller) -> (image, mask, log)."""
        ov = ov or {}
        al_i = float(ov.get('erase_al', self.er_al[i]))
        if al_i <= 0.01: return out, None, {}
        g, F = self.g, self.F
        gap_t = float(F[i, 0] + ov['dFe'][0]) if 'dFe' in ov else self.Fe[i, 0]
        P_take = ov.get('S_take', ov.get('S', self.S[i]))[:, :2]
        P_out = ov.get('P_out', self.P_out[i])
        if rerender: self.erase.lab = [x for x in self.erase.lab if x[0] != i]
        out, em = self.erase.composite(out, i, P_take, P_out, al_i, 'closed' if al_kind(F[i, 0], gap_t, self.model.rng, g('warp_erase_thr', 0.12)) else 'own')
        return out, em, dict(erase=round(al_i, 2), lip_fit=round(float(self.er_fit[i]), 2), erase_dst=np.round(lip_frame(P_out), 1).tolist())

    # ------------------------------------------------------------------------------------------------ lip lock --
    def setup_lock(self):
        g, n = self.g, self.n
        lk = ll.LipLock(ll.parse_lip_ref(self.a.warp_lip_lock), g('warp_lip_lock_sigma', 4.0), g('warp_lip_lock_l', 0.0))
        lk.fit([ll.lip_stats(self.frames[i], self.S[i][:, :2]) for i in range(n)])     # the take's own lips (the warp keeps them)
        self.lock = lk

    def lock_one(self, i, out, ov=None):
        ov = ov or {}
        lk, comp = self.lock, self.comp
        P_out = ov.get('P_out', self.P_out[i])
        undo = self._patch(i, {'lock_st': ov['lock_st']}) if 'lock_st' in ov else []
        try:
            pre = out
            out, lw = lk.apply(out, P_out, i)
            if comp is not None and i in comp.interior:     # v7: the lip colour lock never tints the teeth / tongue
                (x0_, y0_, w_, h_), im_ = comp.interior[i]
                o_ = out.copy()
                sub = o_[y0_:y0_ + h_, x0_:x0_ + w_]
                sub[im_] = pre[y0_:y0_ + h_, x0_:x0_ + w_][im_]
                out = o_
            lg = dict(lip_shift=[round(float(x), 2) for x in lk.shift(i)])
        finally:
            self._unpatch(undo)
        return out, lw, lg

    # ------------------------------------------------------------------------------------------- whole frames --
    def render_frame(self, i, ov=None, fold=False):
        """v9: frame i re-made through the whole chain with overrides ov (see the class doc) -> (image, mask, log entry,
        P_out). With ov = {} after run_batch() this reproduces the batch frame bit-for-bit. fold: also log the applied
        warp's min Jacobian determinant (ent['fold_det'])."""
        ov = dict(ov or {})
        Si = ov.get('S', self.S[i])
        undo = self._patch(i, ov)
        try:
            out, mask, ent, dst = self.warp_one(i, ov)
            P_out = self.p_out_of(i, dst, Si)
            ov['P_out'] = P_out
            if fold: ent['fold_det'] = round(float(self.fold_det_of(i, ov.get('dFe'), ov.get('S'), ov.get('ell'), ov.get('lam'))), 4)
            if self.rigid_teeth and self.profile(i): ov['layers_off'] = True
            if self.rigid_teeth and not ov.get('layers_off'):
                out, tm, lg = self.layer_one(i, out, ov, rerender=True)
                mask = np.maximum(mask, tm)
                ent.update(lg)
            elif self.rigid_teeth:
                ent['layers_off'] = True          # v11 gate (h): the plain own-pixel warp, no layer stack
            if self.erase is not None:
                o2, em, lg = self.erase_one(i, out, ov, rerender=True)
                if em is not None:
                    out = self.erase.apply_item(i, out, self.erase.stat_for(i))
                    mask = np.maximum(mask, em)
                    ent.update(lg)
            if self.lock is not None:
                out, lw, lg = self.lock_one(i, out, ov)
                if lw is not None: mask = np.maximum(mask, lw)
                ent.update(lg)
        finally:
            self._unpatch(undo)
        return out, mask, ent, P_out

    def profile(self, i):
        """v18 (--warp-profile-plain T): a frame turned past T (eye balance about the nose bridge) keeps the plain
        own-pixel warp, no synthetic teeth / cavity: in near-profile their far end was a hard cut past the lip (sky B f61-63)."""
        t = self.g('warp_profile_plain', 0.0)
        return t > 0 and getattr(self, 'fs', None) is not None and self.fs[i] < t

    def run_batch(self):
        """round 0 (the v8 batch, bit-identical to the pre-v9 run())."""
        g, n = self.g, self.n
        for i in range(n):
            self.warped[i], self.masks[i], self.log[i], self.dsts[i] = self.warp_one(i)
        # v4 layers (opt-in): lip erase + the take's own closed lips at closures, then the lip colour lock
        for i in range(n): self.P_out[i] = self.p_out_of(i, self.dsts[i])            # where the lips ARE in each output
        outs, masks, log = self.outs, self.masks, self.log
        for i in range(n): outs[i] = self.warped[i]
        if self.rigid_teeth:
            self.setup_layers()
            teeth = self.teeth
            for i in range(n):
                if self.profile(i):
                    log[i]['layers_off'] = True
                    log[i]['profile'] = True
                    continue
                outs[i], tm, lg = self.layer_one(i, outs[i])
                masks[i] = np.maximum(masks[i], tm)
                log[i].update(lg)
            npf = [i for i in range(n) if self.profile(i)]
            if npf: print(f'profile plain (turn < {g("warp_profile_plain", 0.0)}): {len(npf)} frames own-pixel warp only {npf}')
            print(f'rigid teeth: upper plate from frame {teeth.h_up}, lower from {teeth.h_lo}; hinge {teeth.info()["hinge"]}; '
                  f'jaw {min(l.get("jaw_deg", 0.0) for l in log):.1f}..{max(l.get("jaw_deg", 0.0) for l in log):.1f} deg')
        er = g('warp_erase', 'off')
        if er != 'off':
            self.setup_erase()
            al = self.er_al
            for i in range(n):
                if al[i] <= 0.01: continue
                outs[i], em, lg = self.erase_one(i, outs[i])
                masks[i] = np.maximum(masks[i], em)
                log[i].update(lg)
            self.erase.finish(outs)
            print(f'lip erase ({er}): reference frame {self.er_ref}; {int((al > 0.01).sum())} frames layered')
        if g('warp_lip_lock', None):
            self.setup_lock()
            lk = self.lock
            for i in range(n):
                outs[i], lw, lg = self.lock_one(i, outs[i])
                if lw is not None: masks[i] = np.maximum(masks[i], lw)
                log[i].update(lg)
            print(f'lip lock -> Lab {lk.ref.round(1).tolist()}: shift a*/b* median {np.median([lk.shift(i)[1:] for i in range(n)], 0).round(1).tolist()}')
        return self

    def teeth_info(self):
        if not self.rigid_teeth or self.teeth is None: return None
        info = self.teeth.info()
        if self.comp is not None:
            info['layers'] = self.comp.info()
            info['layers']['bite_overlap'] = self.comp.bite_overlap(self.P_out, self.Fe[:, 0])
            if 'corridor' in info['layers']:      # v13: the corridor's summary from the delivered frames' own entries
                info['layers']['corridor'] = self.comp.corridor_info([((l or {}).get('reg') or {}).get('corridor') for l in self.log])
        return info

    def write(self, extra_json=None):
        a, g, n, W, H = self.a, self.g, self.n, self.W, self.H
        outs, masks, log = self.outs, self.masks, self.log
        if self.comp is not None and g('warp_layers_debug', None): self.comp.write_debug(a.warp_layers_debug, self.fps, self.dbg_f)
        pr = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(self.fps),
                               '-i', '-', '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', a.out], stdin=ff.PIPE)
        mk = []
        for i in range(n):
            pr.stdin.write(np.ascontiguousarray(outs[i]).tobytes())
            if a.mask_out: mk.append(cv2.resize(masks[i], None, fx=0.5, fy=0.5))
        pr.stdin.close()
        pr.wait()
        if a.mask_out: np.savez_compressed(a.mask_out, masks=(np.array(mk) * 255).astype(np.uint8))
        if getattr(a, 'warp_json', None):
            ell = self.ell
            model = self.model
            d = dict(face_id=self.bank.face_id, range=model.rng, pca_var=model.pca_var, J=model.J.round(4).tolist(),
                     ellipse=None if ell is None else dict(ax=ell.ax, ay=ell.ay, r_in=ell.r_in, r_out=ell.r_out), teeth=self.teeth_info(), frames=log)
            if extra_json: d.update(extra_json)
            json.dump(d, open(a.warp_json, 'w'))
        act = np.mean([l['move_px'] >= 0.25 for l in log])
        nb = sum(l['mode'] == 'b' for l in log)
        print(f'mode a (own mouth) {n - nb} frames, mode b (bank entry) {nb} frames: {[l["i"] for l in log if l["mode"] == "b"]}')
        print(f'face tracked {self.found:.0%}; warp active {act:.0%} of {n} frames; median move '
              f'{np.median([l["move_px"] for l in log]):.1f}px; pca var {model.pca_var}; -> {a.out}')


def run(a):
    # --- v10a: --vocalisations skip | rest (lsdub/core/vocalisations.py). vx is None with --vocalisations mouth or a
    # track without vocalisations: then this is exactly the v9 run.
    from . import vocalisations
    vx = vocalisations.for_warp(a)
    clip = WarpClip(vx.args if vx else a)
    if vx: vx.before_batch(clip)
    clip.run_batch()
    if vx: vx.after_batch(clip)
    # --- (v10a end)
    extra = None
    if getattr(a, 'warp_gate', 'off') not in (None, 'off'):
        # v9: the per-frame maker / checker / confirmation gate (lsdub/core/frame_gate.py)
        from . import frame_gate
        extra = frame_gate.gate(clip, a)
    if vx: extra = vx.finish(clip, extra)
    if getattr(a, 'occluder_mask', None):
        # an object in front of the mouth (mic, hand) keeps the take's own pixels (lsdub/core/occluders.py)
        from . import occluders
        occ = occluders.load(a.occluder_mask, clip.n, (clip.H, clip.W))
        cov = occluders.restore(clip.outs, clip.frames, occ, int(getattr(a, 'occluder_grow', 3)))
        for i, c in enumerate(cov):
            if clip.log[i] is not None: clip.log[i]['occluded_px'] = c
        print(f'occluders: take pixels kept on {sum(c > 0 for c in cov)}/{clip.n} frames')
    clip.write(extra)
