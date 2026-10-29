#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""layer_reg.py: the inter-layer REGISTRATION ruler (v11). CPU.

Founder on v10: "the sloppiness in the mouth has gone from the lips to the registration of the sub-elements of the mouth
... it doesn't show on any one frame, it's across frames where all those registrations have a weird sloshing to them."
Every layer of the v7 / v8 stack carries its own per-frame correction -- the upper teeth the lip-root skin pin, the lower
teeth the chin pin on the jaw arc and the exposure kick, the tongue its vowel pose, the cavity the planned inner-lip
contour -- so the sub-elements can move against each other and against the lips from frame to frame. This audit
MEASURES that, from the anchors the layer stack logs per frame (warp.json frames[i].reg, mouth_layers.reg_entry: each
anchor where it was rendered, in the skull-stabilised frame = the canonical plane of the upper teeth through the skull
pose only, IOD units; x along the head's left-right axis, y down).

  offsets (per frame, 2-D):
    up_lip    upper incisal edge (centre) - upper inner lip (landmark 13)
    lo_lip    lower incisal edge (centre) - lower inner lip (14)
    cav_open  cavity centroid - the opening's centroid (the warped inner-lip polygon)
    tip_lo    tongue tip - lower incisal edge
    bite      upper incisal edge - lower incisal edge
  for each offset:
    (a) jaw residual: each component regressed on the jaw angle used (+ an intercept, least squares; the jaw is the only
        thing these offsets should depend on); RMS and p95 of |residual| (IOD)
    (b) jitter: |o[t] - o[t-1]| p50 / p95 (IOD / frame)
    (c) slosh: the v9 gate's windowed ruler (W = 4, audits/frame_gate.Checker._slosh's window and arithmetic) on the
        offset's velocity: per window centre, sum over both components of (sum |v| - |sum v|) / steps -- back-and-forth
        motion minus the net -- mean / p95 / max (IOD / frame)
  (d) tex_drift: the visible upper-teeth TEXTURE's shift vs the previous frame in the skull-stabilised crop (NCC over
      integer shifts + a parabolic sub-pixel fit; visible = inside the warped opening and above the incisal-edge line),
      with (b) and (c) on its cumulative position; and its RMS difference from the logged upper-pin change (what a tooth
      riding the skull + pin should do).
  summary: the sum of the five offsets' mean slosh (IOD / frame).
The gate (--warp-gate-reg) runs gate_report on the frames it finished with: windows whose summed slosh is >= FLOOR are
reported in gate.json (not repaired).

v12: the PIXEL ruler (--pix; pix_measure). The ruler above reads the layer stack's own log, so it cannot score the take.
The pixel ruler measures the picture of the TAKE and of a RENDER alike: in the skull-stabilised crop (the warp's own
skull pose and teeth plane, skull_frame / stab_crop) the upper-teeth patch (and the lower-arch patch when visible: masks
from the video's own IMAGE-mode landmarks + the teeth key) is template-tracked from frame to frame (patch_shift: masked
NCC + sub-pixel); per frame a velocity in IOD / frame. Rulers (pix_stats): jitter |v| p50 / p95 and the low-frequency
wander (W = 4, 8: the v9 gate's window, back-and-forth energy per finite velocity), mean / p95. Plus a nose-base
rectangle (the stabilisation's own floor: skin the warp does not move) and the lower arch split along / across the jaw
arc of the hinge model (across: what no jaw angle explains; along - the logged jaw of a render). The take's own numbers
on its open-mouth frames are the floor.

  lsdub layer-reg --render <dir> [--video full.mp4] [--json out.json] [--md out.md] [--w 4] [--no-texture] [--label x]
  lsdub layer-reg --pix --video V.mp4 --take-trk TRK.npz --frames N [--take TAKE.mp4] [--lm CACHE.npz] [--render DIR]
      [--ppi 200] [--json out.json] [--md out.md] [--label x]
"""
import argparse, json, os, sys
import numpy as np, cv2

OFFSETS = (('up_lip', 'upper incisal edge - upper inner lip'), ('lo_lip', 'lower incisal edge - lower inner lip'),
           ('cav_open', 'cavity centroid - opening centroid'), ('tip_lo', 'tongue tip - lower incisal edge'),
           ('bite', 'upper - lower incisal edge'))
FLOOR = 0.0528          # gate_report: a window whose summed slosh (5 offsets, IOD / frame) is >= this is reported: v9's own
                        # p95 of the per-window sum over its 7 mv30 shots (re-rendered bit for bit with the reg log)


def _pt(r, k, j=None):
    v = r.get(k) if r else None
    if v is None: return np.full(2, np.nan)
    v = np.asarray(v, float)
    return v[j] if j is not None else v


def series(frames):
    """warp.json frames -> dict: the five offsets (n x 2, nan where a frame has no reg entry), jaw (deg), pin_up_iod (n x 2),
    the logged anchors (up / lo: n x 3 x 2, contour: n x 20 x 2, stab: n x 2 x 3)."""
    n = len(frames)
    # a frame shown as the take's own (a skipped vocalisation at strength 0, v10a) shows none of the layers: not measured
    R = [f.get('reg') if float(f.get('vocal_s', 1.0) if f.get('vocal_s') is not None else 1.0) > 0 else None for f in frames]
    up = np.array([_pt(r, 'up') if r else np.full((3, 2), np.nan) for r in R], float).reshape(n, 3, 2)
    lo = np.array([_pt(r, 'lo') if r else np.full((3, 2), np.nan) for r in R], float).reshape(n, 3, 2)
    inner = np.array([_pt(r, 'inner') if r else np.full((4, 2), np.nan) for r in R], float).reshape(n, 4, 2)   # 13, 14, 78, 308
    cav = np.array([_pt(r, 'cav') for r in R])
    op = np.array([_pt(r, 'open') for r in R])
    tip = np.array([_pt(r, 'tip') for r in R])
    O = dict(up_lip=up[:, 1] - inner[:, 0], lo_lip=lo[:, 1] - inner[:, 1], cav_open=cav - op, tip_lo=tip - lo[:, 1], bite=up[:, 1] - lo[:, 1])
    jaw = np.array([r['jaw'] if r and r.get('jaw') is not None else np.nan for r in R], float)
    pin = np.array([_pt(r, 'pin_up_iod') for r in R])
    stab = np.array([np.asarray(r['stab'], float) if r and r.get('stab') else np.full((2, 3), np.nan) for r in R])
    contour = np.array([np.asarray(r['contour'], float) if r and r.get('contour') else np.full((20, 2), np.nan) for r in R])
    return dict(off=O, jaw=jaw, pin=pin, up=up, lo=lo, inner=inner, stab=stab, contour=contour, n=n,
                layers=np.array([bool(r) for r in R]))


def jaw_residual(o, jaw):
    """o (n x 2) regressed per component on the jaw angle (+ intercept) -> residual (n x 2, nan where missing)."""
    o = np.asarray(o, float)
    res = np.full_like(o, np.nan)
    ok = np.isfinite(o).all(1) & np.isfinite(jaw)
    if ok.sum() < 3: return res
    X = np.stack([np.ones(ok.sum()), jaw[ok]], 1)
    for k in range(2):
        b, *_ = np.linalg.lstsq(X, o[ok, k], rcond=None)
        res[ok, k] = o[ok, k] - X @ b
    return res


def osc(v):
    """oscillatory energy of a velocity sequence: sum |v| - |sum v| (audits/frame_gate.osc)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.abs(v).sum() - abs(v.sum())) if len(v) else 0.0


def slosh(o, W=4):
    """per window centre c: the v9 gate's window (velocities of frames max(1, c - W + 1) .. min(n - 1, c + W)), sum over the
    components of osc(v) / steps; nan where the window has < max(3, W) steps or < 3 finite velocities."""
    o = np.asarray(o, float)
    n = len(o)
    E = np.full(n, np.nan)
    if o.ndim == 1: o = o[:, None]
    v = np.full_like(o, np.nan)
    v[1:] = np.diff(o, axis=0)
    for c in range(n):
        a0, a1 = max(1, c - W + 1), min(n - 1, c + W)
        if a1 - a0 + 1 < max(3, W): continue
        seg = v[a0:a1 + 1]
        if (np.isfinite(seg).all(1)).sum() < 3: continue
        E[c] = sum(osc(seg[:, k]) for k in range(seg.shape[1])) / (a1 - a0 + 1)
    return E


def jitter(o):
    o = np.asarray(o, float)
    d = np.linalg.norm(np.diff(o, axis=0), axis=1) if o.ndim == 2 else np.abs(np.diff(o))
    return d[np.isfinite(d)]


def _q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else float('nan')


def stats(o, jaw, W=4):
    res = jaw_residual(o, jaw)
    rn = np.linalg.norm(res, axis=1)
    j = jitter(o)
    E = slosh(o, W)
    return dict(n=int(np.isfinite(o).all(1).sum()), resid_rms=float(np.sqrt(np.nanmean(rn ** 2))) if np.isfinite(rn).any() else float('nan'),
                resid_p95=_q(rn, 95), jit_p50=_q(j, 50), jit_p95=_q(j, 95), slosh_mean=float(np.nanmean(E)) if np.isfinite(E).any() else float('nan'),
                slosh_p95=_q(E, 95), slosh_max=float(np.nanmax(E)) if np.isfinite(E).any() else float('nan'),
                slosh_argmax=int(np.nanargmax(E)) if np.isfinite(E).any() else -1, resid_argmax=int(np.nanargmax(rn)) if np.isfinite(rn).any() else -1), E, res


# --------------------------------------------------------------------------------------------- texture drift --
def stab_crops(video, S, meta, ppi=300.0, win=(-0.5, 0.5, -0.3, 0.25)):
    """skull-stabilised grey crops of every frame (the logged plane affine of each frame; window in IOD around the reg
    origin, ppi px per IOD) -> (list of float32 crops or None, the crop's origin offset in IOD)."""
    org, io = np.asarray(meta['origin'], float), float(meta['iod'])
    x0, x1, y0, y1 = win
    w, h = int(round((x1 - x0) * ppi)), int(round((y1 - y0) * ppi))
    C = np.array([[io / ppi, 0, org[0] + x0 * io], [0, io / ppi, org[1] + y0 * io]], np.float64)      # crop px -> canonical
    cap = cv2.VideoCapture(video)
    out = []
    for t in range(S['n']):
        ok, fr = cap.read()
        if not ok: break
        A = S['stab'][t]
        if not np.isfinite(A).all():
            out.append(None)
            continue
        M = np.vstack([A, [0, 0, 1]]) @ np.vstack([C, [0, 0, 1]])                                     # crop px -> frame px
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out.append(cv2.warpAffine(g, M[:2], (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT))
    return out, (x0, y0), ppi


def teeth_mask(S, t, shape, off, ppi, margin=0.004):
    """visible upper teeth of frame t in the crop: inside the warped opening (the logged contour) and above the incisal
    edge line through the logged ends (+ margin IOD)."""
    con = S['contour'][t]
    up = S['up'][t]
    if not (np.isfinite(con).all() and np.isfinite(up).all()): return None
    px = lambda p: (np.asarray(p, float) - np.asarray(off, float)) * ppi
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(px(con)).astype(np.int32)], 1)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    a, c = px(up[0]), px(up[2])
    k = (c[1] - a[1]) / (c[0] - a[0]) if abs(c[0] - a[0]) > 1e-6 else 0.0
    line = a[1] + k * (xx - a[0])
    return (m > 0) & (yy <= line + margin * ppi)


def ncc_shift(a, b, mask, r):
    """the shift (dx, dy) px that maps b onto a best (NCC over mask, integer shifts within r, parabolic sub-pixel)."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 40: return None
    best = np.full((2 * r + 1, 2 * r + 1), -2.0)
    A = a[ys, xs]
    A = A - A.mean()
    na = np.linalg.norm(A)
    if na < 1e-3: return None
    H, W_ = b.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y2, x2 = ys - dy, xs - dx
            ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W_)
            if ok.sum() < 0.8 * len(ys): continue
            B = b[y2[ok], x2[ok]]
            B = B - B.mean()
            Aa = a[ys[ok], xs[ok]]
            Aa = Aa - Aa.mean()
            nb = np.linalg.norm(B) * np.linalg.norm(Aa)
            if nb > 1e-6: best[dy + r, dx + r] = float(Aa @ B) / nb
    iy, ix = np.unravel_index(np.argmax(best), best.shape)
    if best[iy, ix] < 0.3: return None
    def sub(c_, p, m_):              # the parabola's vertex through (-1, m_), (0, c_), (+1, p)
        d = p - 2 * c_ + m_
        return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (m_ - p) / d, -0.5, 0.5))
    sx = sub(best[iy, ix], best[iy, ix + 1], best[iy, ix - 1]) if 0 < ix < 2 * r else 0.0
    sy = sub(best[iy, ix], best[iy + 1, ix], best[iy - 1, ix]) if 0 < iy < 2 * r else 0.0
    return np.array([ix - r + sx, iy - r + sy]), float(best[iy, ix])


def tex_drift(video, S, meta, ppi=300.0, r_iod=0.03):
    """per frame t: the visible upper-teeth texture's shift t-1 -> t in the skull-stabilised crop (IOD, n x 2, nan where
    fewer than 40 visible tooth px or no confident match) and its NCC."""
    crops, off, ppi = stab_crops(video, S, meta, ppi)
    n = len(crops)
    d = np.full((S['n'], 2), np.nan)
    q = np.full(S['n'], np.nan)
    r = max(2, int(round(r_iod * ppi)))
    for t in range(1, n):
        if crops[t] is None or crops[t - 1] is None: continue
        m = teeth_mask(S, t, crops[t].shape, off, ppi)
        if m is None: continue
        got = ncc_shift(crops[t], crops[t - 1], m, r)
        if got is None: continue
        d[t] = got[0] / ppi
        q[t] = got[1]
    return d, q


# ---------------------------------------------------------------------------------------------------- report --
def measure(frames, video=None, meta=None, W=4, texture=True):
    S = series(frames)
    out = dict(n=S['n'], frames_with_layers=int(S['layers'].sum()), W=W, offsets={}, per_frame={})
    E_sum = np.zeros(S['n'])
    any_ = np.zeros(S['n'], bool)
    for k, _ in OFFSETS:
        st, E, res = stats(S['off'][k], S['jaw'], W)
        out['offsets'][k] = st
        out['per_frame'][k] = dict(off=np.round(S['off'][k], 5).tolist(), resid=np.round(res, 5).tolist(), slosh=np.round(E, 5).tolist())
        ok = np.isfinite(E)
        E_sum[ok] += E[ok]
        any_ |= ok
    E_sum[~any_] = np.nan
    out['per_frame']['slosh_sum'] = np.round(E_sum, 5).tolist()
    out['summary'] = dict(slosh_sum=float(np.nansum([out['offsets'][k]['slosh_mean'] for k, _ in OFFSETS if np.isfinite(out['offsets'][k]['slosh_mean'])])),
                          resid_rms_sum=float(np.nansum([out['offsets'][k]['resid_rms'] for k, _ in OFFSETS if np.isfinite(out['offsets'][k]['resid_rms'])])))
    if texture and video and meta:
        d, q = tex_drift(video, S, meta)
        pos = np.full_like(d, np.nan)
        acc = np.zeros(2)
        seen = False
        for t in range(len(d)):                      # the texture's cumulative position (a gap restarts it)
            if np.isfinite(d[t]).all():
                acc = acc + d[t]
                pos[t] = acc
                seen = True
            elif seen:
                acc = np.zeros(2)
                seen = False
        dp = np.full_like(d, np.nan)
        dp[1:] = np.diff(S['pin'], axis=0)
        dd = np.linalg.norm(d - dp, axis=1)
        jn = np.linalg.norm(d, axis=1)
        E = slosh(pos, W)
        out['tex_drift'] = dict(n=int(np.isfinite(jn).sum()), jit_p50=_q(jn, 50), jit_p95=_q(jn, 95), slosh_mean=float(np.nanmean(E)) if np.isfinite(E).any() else float('nan'),
                                slosh_p95=_q(E, 95), slosh_max=float(np.nanmax(E)) if np.isfinite(E).any() else float('nan'),
                                vs_pin_rms=float(np.sqrt(np.nanmean(dd ** 2))) if np.isfinite(dd).any() else float('nan'), ncc_p50=_q(q, 50))
        out['per_frame']['tex_drift'] = np.round(d, 5).tolist()
        out['per_frame']['tex_ncc'] = np.round(q, 3).tolist()
    return out


def gate_report(frames, W=4, floor=None):
    """the gate's view (--warp-gate-reg): the summed offset slosh per window centre and the windows >= floor (reported,
    not repaired)."""
    floor = FLOOR if floor is None else floor
    r = measure(frames, W=W, texture=False)
    E = np.asarray(r['per_frame']['slosh_sum'], float)
    win = []
    for c in np.nonzero(np.isfinite(E) & (E >= floor))[0]:
        a0, a1 = max(0, int(c) - W), min(len(E) - 1, int(c) + W)
        top = max(OFFSETS, key=lambda kv: np.nan_to_num(r['per_frame'][kv[0]]['slosh'][c]))[0]
        if win and a0 <= win[-1]['frames'][1]:
            win[-1]['frames'][1] = a1
            win[-1]['peak'] = max(win[-1]['peak'], round(float(E[c]), 4))
            continue
        win.append(dict(frames=[a0, a1], peak=round(float(E[c]), 4), top=top))
    return dict(floor=floor, W=W, summary=r['summary'], offsets={k: {x: (round(v, 5) if isinstance(v, float) else v) for x, v in st.items()} for k, st in r['offsets'].items()},
                windows=win, repaired=False)


def table_md(label, r):
    L = [f'**{label}** ({r["frames_with_layers"]} of {r["n"]} frames with the layer stack; W = {r["W"]})', '',
         '| offset | n | (a) jaw resid RMS / p95 | (b) jitter p50 / p95 | (c) slosh mean / p95 / max (frame) |', '|---|---|---|---|---|']
    f = lambda x: 'n/a' if x is None or not np.isfinite(x) else f'{x:.4f}'
    for k, name in OFFSETS:
        s = r['offsets'][k]
        L.append(f"| {k} ({name}) | {s['n']} | {f(s['resid_rms'])} / {f(s['resid_p95'])} | {f(s['jit_p50'])} / {f(s['jit_p95'])} | "
                 f"{f(s['slosh_mean'])} / {f(s['slosh_p95'])} / {f(s['slosh_max'])} (f{s['slosh_argmax']}) |")
    if 'tex_drift' in r:
        s = r['tex_drift']
        L.append(f"| tex_drift (upper-teeth texture vs t-1) | {s['n']} | vs pin change RMS {f(s['vs_pin_rms'])} | {f(s['jit_p50'])} / {f(s['jit_p95'])} | "
                 f"{f(s['slosh_mean'])} / {f(s['slosh_p95'])} / {f(s['slosh_max'])} |")
    L += ['', f"summary: slosh sum {r['summary']['slosh_sum']:.4f} IOD/frame; jaw-residual RMS sum {r['summary']['resid_rms_sum']:.4f} IOD", '']
    return '\n'.join(L)


def line(label, r):
    t = r.get('tex_drift', {})
    return (f"layer-reg {label}: slosh sum {r['summary']['slosh_sum']:.4f} IOD/frame (" +
                                        ', '.join(f"{k} {r['offsets'][k]['slosh_mean']:.4f}" for k, _ in OFFSETS) +
                                        (f"; tex drift jitter p95 {t['jit_p95']:.4f}, slosh {t['slosh_mean']:.4f}" if t else '') + ')')


# ----------------------------------------------------------------------------------------- v12: the pixel ruler --
PIX_WIN = (-0.6, 0.6, -0.5, 0.45)        # the stabilised crop: canonical window (IOD about ref, the inner-lip centre): x0, x1, y0, y1
PIX_NOSE = (-0.22, 0.22, -0.42, -0.22)   # the nose-base reference rectangle (IOD about ref; canonical lm 1 / 2 / 98 / 327 sit at
                                         # y ~ -0.375 / -0.29 / -0.30 / -0.30 IOD, the nostrils at x ~ +-0.17): skin the warp does not move
PIX_MIN_PX = 150                         # a patch is measured with >= this many mask px at ppi 200 (scaled with ppi^2)
PIX_R_IOD = 0.05                         # search range +- (IOD / frame; v11's renders move up to ~0.035)
PIX_MIN_SCORE = 0.5                      # a match is used when its NCC peak is >= this (and not on the search border)


def _core():
    from ..core import teeth_layer as tl, lip_layers as ll, mouth_morph as mm, mouth_paint as mp_
    return tl, ll, mm, mp_


def skull_frame(S3raw, n=None, pose_smooth='sg7'):
    """the warp's own skull pose and teeth plane, rebuilt as mouth_warp.WarpClip + teeth_layer.TeethLayer do: the rigid
    pose on MESH_ANCH (pose_smooth), the k5 track S, the canonical median face Cm, IOD, z_up, ref (the inner-lip centre),
    per frame A[t] = plane_affine(pl, t, z_up, ref) (2x3: canonical (x, y) on the plane -> frame px; = warp.json
    teeth.layers.reg_meta / frames[i].reg.stab where the gate did not re-pose frame i), and the hinge geometry (u, d, b,
    H: hinge_point(Cm, 0.25, 0.6))."""
    tl, _, mm, mp_ = _core()
    S3raw = np.asarray(S3raw, float)
    n = len(S3raw) if n is None else min(int(n), len(S3raw))
    S3 = S3raw[:n]
    pl = mm.RigidPlacement(S3[int(np.argmin(np.abs(np.arange(n) - n // 2)))], S3, mm.MESH_ANCH, smooth=pose_smooth, lam=0.0)
    S = mp_.smooth_track(S3, 'k5')
    C = np.array([pl.to_lib(i, S[i]) for i in range(n)])
    Cm = np.median(C, 0)
    iod = float(np.linalg.norm(Cm[263, :2] - Cm[33, :2]))
    z_up = float(Cm[13, 2] + 0.05 * iod)
    ref = Cm[[13, 14], :2].mean(0)
    A = np.array([tl.plane_affine(pl, t, z_up, ref=tuple(ref)) for t in range(n)])
    u, d, b = tl.head_axes(Cm)
    H = tl.hinge_point(Cm, 0.25, 0.6)
    return dict(n=n, A=A, ref=ref, iod=iod, z_up=z_up, Cm=Cm, u=u, d=d, b=b, H=H, S=S, pl=pl)


def _crop_B(ref, iod, ppi=200.0, win=PIX_WIN):
    """2x3: crop px -> canonical (x, y); and the crop size (w, h)."""
    x0, x1, y0, y1 = win
    k = float(iod) / float(ppi)
    B = np.array([[k, 0.0, ref[0] + x0 * iod], [0.0, k, ref[1] + y0 * iod]], np.float64)
    return B, (int(round((x1 - x0) * ppi)), int(round((y1 - y0) * ppi)))


def stab_crop(img, A_t, ref, iod, ppi=200.0, win=PIX_WIN):
    """the skull-stabilised crop of one frame: crop px (cx, cy) <-> canonical (ref_x + (cx / ppi + win[0]) iod, ref_y +
    (cy / ppi + win[2]) iod) -> frame px through A_t (bilinear, replicated border). -> (grey float32, BGR uint8)."""
    B, (w, h) = _crop_B(ref, iod, ppi, win)
    M = (np.vstack([np.asarray(A_t, np.float64), [0, 0, 1]]) @ np.vstack([B, [0, 0, 1]]))[:2]
    fl = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.warpAffine(g.astype(np.float32), M, (w, h), flags=fl, borderMode=cv2.BORDER_REPLICATE)
    bgr = cv2.warpAffine(img, M, (w, h), flags=fl, borderMode=cv2.BORDER_REPLICATE) if img.ndim == 3 else None
    return grey, bgr


def to_crop(P, A_t, ref, iod, ppi=200.0, win=PIX_WIN):
    """frame-px points (... x >= 2) -> crop px (... x 2) (the inverse of stab_crop's map)."""
    Ai = cv2.invertAffineTransform(np.asarray(A_t, np.float64))
    P = np.asarray(P, float)[..., :2]
    c = P @ Ai[:, :2].T + Ai[:, 2]
    return np.stack([((c[..., 0] - ref[0]) / iod - win[0]) * ppi, ((c[..., 1] - ref[1]) / iod - win[2]) * ppi], -1)


def nose_mask(shape, ppi=200.0, win=PIX_WIN, box=PIX_NOSE):
    """the FIXED nose-base reference rectangle in crop px (bool)."""
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    xa, xb = int(round((box[0] - win[0]) * ppi)), int(round((box[1] - win[0]) * ppi))
    ya, yb = int(round((box[2] - win[2]) * ppi)), int(round((box[3] - win[2]) * ppi))
    m[max(0, ya):min(h, yb), max(0, xa):min(w, xb)] = True
    return m


def _extend(m, inner, k, down=True):
    """m grown k px down (or up) per column, kept inside inner."""
    out = m.copy()
    for s in range(1, k + 1):
        if down: out[s:] |= m[:-s]
        else: out[:-s] |= m[s:]
    return out & inner


def arch_masks(bgr, Pc, skin_L, ppi=200.0):
    """one frame's crop-px masks from its landmarks Pc (478 x 2, crop px): inner = the inner-lip polygon (LIP_IN) eroded
    by 0.025 IOD (no lip pixels); teeth = inner & teeth_key(Lab, the take's skin L, 0.8); up = the upper incisal patch
    (arch_split from the opening's top, top_tol 0.06 IOD: the top .. the end of the first teeth run, extended DOWN 0.02 IOD
    inside inner so the incisal edge -> cavity transition carries the vertical registration); lo = the same from the
    bottom (the last teeth run above the lower lip, extended UP 0.02 IOD, minus up); both only |x - cx| <= 0.25 IOD
    (cx = the mean x of lm 13 / 14)."""
    tl, ll, _, _ = _core()
    h, w = bgr.shape[:2]
    k = lambda f: max(1, int(round(f * ppi)))
    inner = tl.poly_mask(np.asarray(Pc, float)[ll.LIP_IN, :2], (h, w), 0.0, -k(0.025))
    teeth = inner & tl.teeth_key(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB), skin_L, 0.8)
    cx = float(np.asarray(Pc, float)[[13, 14], 0].mean())
    cols = (np.abs(np.arange(w) - cx) <= 0.25 * ppi)[None, :]
    ext, tol, top = k(0.02), max(1, int(round(2 * ppi / 200.0))), 0.06 * ppi
    up, _ = tl.arch_split(teeth, inner, top, tol)
    up = _extend(up, inner, ext, down=True) & cols
    lo, _ = tl.arch_split(teeth[::-1], inner[::-1], top, tol)
    lo = _extend(np.ascontiguousarray(lo[::-1]), inner, ext, down=False) & ~up & cols
    return dict(inner=inner, teeth=teeth, up=up, lo=lo)


_MASKED_CCOEFF = None


def _ncc_np(S, T, Tm):
    """masked NCC of T (mask Tm) at every placement in S, from plain correlations (exact; the fallback when this cv2
    has no masked TM_CCOEFF_NORMED)."""
    m = Tm.astype(np.float32)
    k = max(1.0, float(m.sum()))
    S = S - float(S.mean())
    Tz = ((T - float((T * m).sum()) / k) * m).astype(np.float32)
    num = cv2.matchTemplate(S, Tz, cv2.TM_CCORR)
    s1 = cv2.matchTemplate(S, m, cv2.TM_CCORR)
    s2 = cv2.matchTemplate(S * S, m, cv2.TM_CCORR)
    den = np.sqrt(max(float((Tz * Tz).sum()), 1e-12) * np.maximum(s2 - s1 * s1 / k, 1e-12))
    return num / den


def _ncc_map(S, T, Tm):
    global _MASKED_CCOEFF
    if _MASKED_CCOEFF is None:
        try:
            z = np.random.default_rng(0).random((12, 12)).astype(np.float32)
            cv2.matchTemplate(z, np.ascontiguousarray(z[3:9, 3:9]), cv2.TM_CCOEFF_NORMED, mask=np.ones((6, 6), np.uint8))
            _MASKED_CCOEFF = True
        except cv2.error:
            _MASKED_CCOEFF = False
    return cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED, mask=Tm) if _MASKED_CCOEFF else _ncc_np(S, T, Tm)


def patch_shift(a, b, mask, r_px, min_score=PIX_MIN_SCORE):
    """the motion (dx, dy) crop px of the masked patch from frame b (t-1) to frame a (t): content that moved by +s px
    between t-1 and t returns +s. Template = a over the mask's bounding box (the mask as the template mask), searched in
    b within +-r_px (the box expanded by r_px, clipped to the crop): masked NCC (cv2 TM_CCOEFF_NORMED), integer argmax +
    a parabolic sub-pixel fit per axis (clipped to +-0.5 px). -> (d (2,) or None, peak score); None when the peak is
    < min_score or the argmax sits on the search border."""
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    if len(ys) < 16: return None, float('nan')
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    H, W_ = b.shape[:2]
    r = int(r_px)
    sy0, sy1, sx0, sx1 = max(0, y0 - r), min(H, y1 + r), max(0, x0 - r), min(W_, x1 + r)
    T = np.ascontiguousarray(a[y0:y1, x0:x1], np.float32)
    Tm = np.ascontiguousarray(m[y0:y1, x0:x1], np.uint8)
    R = _ncc_map(np.ascontiguousarray(b[sy0:sy1, sx0:sx1], np.float32), T, Tm)
    R = np.where(np.isfinite(R), R, -1.0)
    iy, ix = np.unravel_index(int(np.argmax(R)), R.shape)
    pk = float(R[iy, ix])
    if pk < min_score or iy in (0, R.shape[0] - 1) or ix in (0, R.shape[1] - 1): return None, pk
    def sub(c_, p, m_):              # the parabola's vertex through (-1, m_), (0, c_), (+1, p)
        d = p - 2 * c_ + m_
        return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (m_ - p) / d, -0.5, 0.5))
    fx = sub(R[iy, ix], R[iy, ix + 1], R[iy, ix - 1])
    fy = sub(R[iy, ix], R[iy + 1, ix], R[iy - 1, ix])
    return np.array([x0 - (sx0 + ix + fx), y0 - (sy0 + iy + fy)], float), pk


def arch_track(frames, lm, sf, skin_L, ppi=200.0, skip=None, win=PIX_WIN, r_iod=PIX_R_IOD, keep=None):
    """per-frame velocities (n x 2, IOD / frame, nan = not measured) of the upper ('up') / lower ('lo') incisal patches and
    the nose reference ('nose') in the skull-stabilised crop, plus the per-frame peak scores and mask px counts.
    frames: the video's BGR frames; lm: ITS landmarks (n x 478 x >= 2, frame px, nan rows = none); sf: skull_frame (or
    any dict with A, ref, iod); skin_L: the take's skin L. Frame t is measured when the patch is valid in frame t
    (>= PIX_MIN_PX mask px, scaled with ppi^2), frames t-1 and t have landmarks, and neither is skipped (skip: bool per
    frame, e.g. a vocalisation shown as the take's own frame in a render). keep: frame indices whose crops and masks are
    returned under 'debug'."""
    A, ref, iod = np.asarray(sf['A'], float), np.asarray(sf['ref'], float), float(sf['iod'])
    n = min(len(frames), len(lm), len(A))
    lm = np.asarray(lm, float)[:n]
    okl = np.isfinite(lm[:, :, :2]).all((1, 2))
    sk = np.zeros(n, bool)
    if skip is not None:
        s_ = np.asarray(skip, bool)[:n]
        sk[:len(s_)] = s_
    r = max(2, int(round(r_iod * ppi)))
    minpx = PIX_MIN_PX * (ppi / 200.0) ** 2
    V = {k: np.full((n, 2), np.nan) for k in ('up', 'lo', 'nose')}
    Q = {k: np.full(n, np.nan) for k in V}
    PX = {k: np.zeros(n, int) for k in ('up', 'lo')}
    dbg = {}
    prev = None
    nose = None
    for t in range(n):
        g, bgr = stab_crop(frames[t], A[t], ref, iod, ppi, win)
        m = None
        if nose is None: nose = nose_mask(g.shape, ppi, win)
        if okl[t]:
            m = arch_masks(bgr, to_crop(lm[t], A[t], ref, iod, ppi, win), skin_L, ppi)
            PX['up'][t], PX['lo'][t] = int(m['up'].sum()), int(m['lo'].sum())
        if keep is not None and t in keep: dbg[t] = dict(grey=g, bgr=bgr, masks=m)
        if t > 0 and prev is not None and okl[t] and okl[t - 1] and not sk[t] and not sk[t - 1]:
            for k, mk in (('up', m['up']), ('lo', m['lo']), ('nose', nose)):
                if k != 'nose' and mk.sum() < minpx: continue
                d, q = patch_shift(g, prev, mk, r)
                Q[k][t] = q
                if d is not None: V[k][t] = d / ppi
        prev = g
    out = dict(up=V['up'], lo=V['lo'], nose=V['nose'], score_up=Q['up'], score_lo=Q['lo'], score_nose=Q['nose'], px_up=PX['up'],
               px_lo=PX['lo'], n=n, ppi=float(ppi), r_px=r, lm_ok=okl, skip=sk, nose_mask=nose)
    if keep is not None: out['debug'] = dbg
    return out


def wander(v, W=4):
    """per window centre c: the velocities of frames max(1, c - W + 1) .. min(n - 1, c + W) (the v9 gate's window, as
    slosh), sum over the components of osc over the FINITE velocities, / the number of finite velocities in the window;
    nan where the window has < max(3, W) steps or < 3 finite velocities (the teeth are invisible on closed frames)."""
    v = np.asarray(v, float)
    if v.ndim == 1: v = v[:, None]
    n = len(v)
    E = np.full(n, np.nan)
    ok = np.isfinite(v).all(1)
    for c in range(n):
        a0, a1 = max(1, c - W + 1), min(n - 1, c + W)
        if a1 - a0 + 1 < max(3, W): continue
        seg = v[a0:a1 + 1][ok[a0:a1 + 1]]
        if len(seg) < 3: continue
        E[c] = sum(osc(seg[:, k]) for k in range(seg.shape[1])) / len(seg)
    return E


def pix_stats(v, W=(4, 8)):
    """a velocity series (n x 2 or n, IOD / frame, nan = gap) -> n (finite), jitter |v| p50 / p95, wander_W mean / p95
    (and the number of windows) for each W."""
    v = np.asarray(v, float)
    if v.ndim == 1: v = v[:, None]
    ok = np.isfinite(v).all(1)
    j = np.linalg.norm(v[ok], axis=1)
    out = dict(n=int(ok.sum()), jit_p50=_q(j, 50), jit_p95=_q(j, 95))
    for w in W:
        E = wander(v, w)
        f = np.isfinite(E)
        out[f'w{w}_mean'] = float(E[f].mean()) if f.any() else float('nan')
        out[f'w{w}_p95'] = _q(E, 95)
        out[f'w{w}_n'] = int(f.sum())
    return out


def hinge_split(v_lo, sf, jaw=None):
    """the lower patch's velocity (n x 2, IOD / frame) vs the hinge model: the jaw rotates about u through H, so the lower
    incisors (T = Cm[14]) move along a = normalise(cross(u, T - H)[:2]) in the canonical plane. across = v - (v . a) a
    (no jaw angle explains it); along = v . a. With a per-frame jaw angle (degrees, e.g. a render's frames[i].reg.jaw):
    along_resid = along - radians(jaw[t] - jaw[t-1]) * lever, lever = |cross(u, T - H)[:2]| / IOD (IOD per rad: the arc
    speed as seen in the plane; lever_3d = the full 3-D lever, for reference)."""
    v = np.asarray(v_lo, float)
    u, H, T, iod = (np.asarray(sf[k], float) for k in ('u', 'H', 'Cm', 'iod'))
    T = T[14]
    iod = float(iod)
    c3 = np.cross(u, T - H)
    a = c3[:2] / max(1e-12, float(np.linalg.norm(c3[:2])))
    along = v @ a
    across = v - along[:, None] * a[None, :]
    lever = float(np.linalg.norm(c3[:2])) / iod
    out = dict(a=a, lever=lever, lever_3d=float(np.linalg.norm(c3)) / iod, along=along, across=across, along_resid=None, pred=None)
    if jaw is not None:
        jw = np.full(len(v), np.nan)
        j_ = np.asarray(jaw, float)[:len(v)]
        jw[:len(j_)] = j_
        pred = np.full(len(v), np.nan)
        pred[1:] = np.radians(np.diff(jw)) * lever
        out['pred'] = pred
        out['along_resid'] = along - pred
    return out


def take_skin_L(frames, S):
    """the TAKE's skin L: the median over its frames of teeth_layer.skin_ring_L(frame, S[t]) (S: its k5 track)."""
    tl = _core()[0]
    n = min(len(frames), len(S))
    return float(np.nanmedian([tl.skin_ring_L(frames[t], S[t]) for t in range(n)]))


def _rl(x, k=5):
    """nan-safe rounded list (nan -> None)."""
    x = np.asarray(x, float)
    return [None if not np.isfinite(y) else round(float(y), k) for y in x] if x.ndim == 1 else [_rl(r, k) for r in x]


def _jsonable(o):
    if isinstance(o, dict): return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.ndarray): return _jsonable(o.tolist())
    return o


def pix_measure(video, S3raw, n, lm, skin_L=None, jaw=None, skip=None, ppi=200.0, sf=None, W=(4, 8), take_fallback=False, keep=None):
    """the pixel ruler on one video (a path or a list of BGR frames) -> JSON-able dict: n, ppi, iod, up / lo / nose
    (pix_stats + n_px median + score p50), lo_hinge (across / along / along_resid: pix_stats; along_resid None without a
    jaw series), per_frame (velocities rounded 5, scores, mask px). lm: the video's own IMAGE-mode landmarks (nan rows =
    not measured; take_fallback: those rows take the take's k5 track S -- only for the TAKE, a render's lips are
    elsewhere). skin_L None -> from the video's own frames with S (fine for the take; for a render pass the take's). sf:
    skull_frame(S3raw, n) when None. keep: frame indices whose crops / masks are returned (not JSON-able) under 'debug'."""
    from .frame_gate import read_frames
    frames = read_frames(video, n) if isinstance(video, str) else list(video)
    sf = skull_frame(S3raw, n) if sf is None else sf
    n = min(len(frames), len(sf['A']), len(lm))
    frames = frames[:n]
    lm = np.array(lm, float)[:n]
    bad = ~np.isfinite(lm[:, :, :2]).all((1, 2))
    fb = 0
    if take_fallback and 'S' in sf and bad.any():
        S_ = np.asarray(sf['S'], float)[:n]
        d = min(lm.shape[2], S_.shape[2])
        lm[bad] = np.nan
        lm[np.nonzero(bad)[0], :, :d] = S_[bad][:, :, :d]
        fb = int(bad.sum())
    if skin_L is None: skin_L = take_skin_L(frames, sf['S'])
    tr = arch_track(frames, lm, sf, skin_L, ppi, skip, keep=keep)
    hg = hinge_split(tr['lo'], sf, jaw)
    A = np.asarray(sf['A'], float)[:n]
    sc = np.sqrt(np.abs(np.linalg.det(A[:, :, :2])))
    def blk(k):
        s = pix_stats(tr[k], W)
        ok = np.isfinite(tr[k]).all(1)
        if k != 'nose':
            px = tr['px_' + k][ok]
            s['n_px'] = float(np.median(px)) if len(px) else float('nan')
        s['score_p50'] = _q(tr['score_' + k][ok], 50)
        s['n_tried'] = int(np.isfinite(tr['score_' + k]).sum())
        return s
    out = dict(n=n, ppi=float(ppi), iod=float(sf['iod']), iod_px=float(np.median(sc) * sf['iod']), skin_L=float(skin_L), r_px=tr['r_px'],
               lm_ok=int(tr['lm_ok'].sum()), take_fallback=fb, skipped=int(tr['skip'].sum()), W=list(W),
               up=blk('up'), lo=blk('lo'), nose=blk('nose'),
               lo_hinge=dict(a=[round(float(x), 5) for x in hg['a']], lever=hg['lever'], lever_3d=hg['lever_3d'], across=pix_stats(hg['across'], W),
                             along=pix_stats(hg['along'], W), along_resid=None if hg['along_resid'] is None else pix_stats(hg['along_resid'], W)),
               per_frame=dict(up=_rl(tr['up']), lo=_rl(tr['lo']), nose=_rl(tr['nose']), score_up=_rl(tr['score_up'], 3), score_lo=_rl(tr['score_lo'], 3),
                              score_nose=_rl(tr['score_nose'], 3), px_up=tr['px_up'].tolist(), px_lo=tr['px_lo'].tolist(), along=_rl(hg['along']),
                              along_resid=None if hg['along_resid'] is None else _rl(hg['along_resid']), skip=tr['skip'].astype(int).tolist()))
    out = _jsonable(out)
    if keep is not None: out['debug'] = tr['debug']
    return out


PIX_ROWS = (('up', 'up', 'upper incisal patch'), ('lo', 'lo', 'lower incisal patch'), ('nose', 'nose', 'nose base (the stabilisation floor)'),
            ('lo_across', ('lo_hinge', 'across'), 'lower: across the jaw arc'), ('lo_along_resid', ('lo_hinge', 'along_resid'), 'lower: along the arc - logged jaw'))


def _pix_get(r, path):
    return r.get(path) if isinstance(path, str) else (r.get(path[0]) or {}).get(path[1])


def pix_table_md(label, r):
    f = lambda x: 'n/a' if x is None or not np.isfinite(x) else f'{x:.4f}'
    L = [f'**{label}** pixel registration, IOD / frame ({r["n"]} frames, {r["lm_ok"]} with landmarks, {r["skipped"]} skipped; ppi '
         f'{r["ppi"]:g}, IOD = {r["iod_px"]:.0f} frame px)', '',
         '| patch | n | jitter p50 | jitter p95 | W4 mean | W4 p95 | W8 mean | W8 p95 |', '|---|---|---|---|---|---|---|---|']
    for _, path, name in PIX_ROWS:
        s = _pix_get(r, path)
        if s is None:
            L.append(f'| {name} | - | n/a | n/a | n/a | n/a | n/a | n/a |')
            continue
        L.append(f"| {name} | {s['n']} | {f(s['jit_p50'])} | {f(s['jit_p95'])} | {f(s['w4_mean'])} | {f(s['w4_p95'])} | {f(s['w8_mean'])} | {f(s['w8_p95'])} |")
    return '\n'.join(L + [''])


def pix_line(label, r):
    f = lambda x: 'n/a' if x is None or not np.isfinite(x) else f'{x:.4f}'
    seg = lambda s: 'n/a' if s is None else f"jit p95 {f(s['jit_p95'])} W4 {f(s['w4_mean'])} W8 {f(s['w8_mean'])} (n {s['n']})"
    return (f"layer-reg --pix {label}: up {seg(r['up'])}; lo across {seg(r['lo_hinge']['across'])}; "
            f"lo along-resid {seg(r['lo_hinge']['along_resid'])}; nose {seg(r['nose'])} [IOD/frame]")


def pix_landmarks(frames, path, S3raw):
    """the video's IMAGE-mode landmarks (frame_gate.landmark_cache: face-box crops from the take's track, implausible
    re-detections nan); an existing cache is never overwritten: one longer than the clip is cut, a shorter one is an
    error (None)."""
    from .frame_gate import landmark_cache, filter_redet
    n = len(frames)
    S3raw = np.asarray(S3raw, float)[:n]
    if path and os.path.exists(path):
        z = np.load(path)['S3']
        if len(z) < n:
            print(f'layer-reg --pix: {path} has {len(z)} frames < {n}: not used (delete it or pass another --lm)', file=sys.stderr)
            return None
        if len(z) > n:
            lm, rej = filter_redet(np.asarray(z[:n], float), S3raw)
            if rej: print(f"layer-reg --pix: {len(rej)} implausible re-detection(s) not used: {[x['i'] for x in rej]}")
            return lm
    return landmark_cache(frames, path, None, S3raw)


def pix_main(a):
    from .frame_gate import read_frames
    if not a.take_trk:
        print('layer-reg --pix: --take-trk is required', file=sys.stderr)
        return 2
    video = a.video or (os.path.join(a.render, 'full.mp4') if a.render and not a.render.endswith('.json') else None)
    if not video or not os.path.exists(video):
        print(f'layer-reg --pix: no video ({video})', file=sys.stderr)
        return 2
    S3raw = np.load(a.take_trk)['S3']
    n = min(a.frames or len(S3raw), len(S3raw))
    frames = read_frames(video, n)
    n = min(n, len(frames))
    frames = frames[:n]
    take = a.take or video
    is_take = os.path.abspath(take) == os.path.abspath(video)
    sf = skull_frame(S3raw, n)
    tf = frames if is_take else read_frames(take, n)
    skin_L = take_skin_L(tf[:n], sf['S'])
    del tf
    lm = pix_landmarks(frames, a.lm, S3raw[:n])
    if lm is None: return 2
    jaw = skip = None
    if a.render:
        wj = a.render if a.render.endswith('.json') else os.path.join(a.render, 'warp.json')
        fr = json.load(open(wj))['frames'][:n]
        jaw = [f['reg']['jaw'] if f.get('reg') and f['reg'].get('jaw') is not None else np.nan for f in fr]
        skip = [f.get('vocal_s') is not None and float(f['vocal_s']) <= 0.0 for f in fr]
    r = pix_measure(frames, S3raw, n, lm, skin_L, jaw, skip, a.ppi, sf=sf, take_fallback=is_take)
    label = a.label or os.path.basename(os.path.dirname(os.path.abspath(video))) or os.path.basename(video)
    r.update(label=label, video=os.path.abspath(video), take=os.path.abspath(take), take_trk=os.path.abspath(a.take_trk), lm_cache=a.lm,
             render=a.render, is_take=is_take)
    if a.json: json.dump(r, open(a.json, 'w'))
    md = pix_table_md(label, r)
    if a.md: open(a.md, 'w').write(md)
    print(md)
    print(pix_line(label, r))
    return 0


# ------------------------------------------------------------------------------------ v12: the double-edge ruler --
def plate_edges(tex, upper=True, iod=1.0, texel=1.0, h0=0.0, centre_iod=0.2, cav=30.0, drop=0.2, max_run=0.05, sigma=0.004):
    """v12 'double edge' ruler on a BUILT arch plate (mouth_layers.ArchBand.tex: BGRA, rows = h from h0 at texel per
    canonical px, h along the head's down axis, the incisal edge at h = 0; columns = arc length a from -A at texel).
    Per column, composited over a dark cavity grey (cav, Lab L), the luminance profile from the root (gum) toward the incisal
    edge and on into the cavity (the lower plate is read bottom-up: its edge is on top), smoothed (sigma IOD); an
    incisal-edge-like transition = a dark horizontal line under a bright band: a fall from a tooth-bright local maximum (at
    least half way from the cavity to the column's tooth L, its p90) by >= drop x (tooth L - cav) to the next local minimum
    within max_run IOD. A clean plate has exactly 1 per tooth column (its incisal edge onto the cavity); a plate carrying a
    second row (the lower teeth of a clenched bite under its bite line) or a repeated edge band has 2 or more.
    -> counts over the tooth columns (opaque, tooth-bright): n, the share with exactly 1, with >= 2, the mean; for the
    central columns (|a| <= centre_iod IOD: the harvested incisors) and for all."""
    from scipy.ndimage import gaussian_filter1d
    T = np.asarray(tex, np.float32)
    Th, Tw = T.shape[:2]
    al = np.clip(T[..., 3], 0, 1)
    L = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    Lc = L * al + cav * (1 - al)
    if not upper: Lc, al = Lc[::-1], al[::-1]
    sg = max(0.5, sigma * iod * texel)
    run_max = max(2, int(round(max_run * iod * texel)))
    a = (np.arange(Tw) + 0.5) / texel - Tw / (2.0 * texel)
    counts = np.full(Tw, -1, int)
    for c in range(Tw):
        op = al[:, c] > 0.5
        if op.sum() < 3: continue
        top = float(np.percentile(Lc[op, c], 90))
        if top < cav + 40: continue
        p = gaussian_filter1d(Lc[:, c], sg, mode='nearest')
        k = 0
        i = 1
        while i < Th - 1:
            if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] >= cav + 0.5 * (top - cav):
                q = i
                while q + 1 < Th and p[q + 1] <= p[q]: q += 1
                j = i + int(np.argmax(p[i:q + 1] <= p[q] + 0.1 * (p[i] - p[q])))     # where the fall is 90 % done
                if p[i] - p[q] >= drop * (top - cav) and j - i <= run_max: k += 1
                i = q
            i += 1
        counts[c] = k
    def summ(m):
        x = counts[m & (counts >= 0)]
        return dict(n=int(len(x)), one=round(float((x == 1).mean()), 3) if len(x) else None, two_plus=round(float((x >= 2).mean()), 3) if len(x) else None,
                    mean=round(float(x.mean()), 3) if len(x) else None)
    return dict(centre=summ(np.abs(a) <= centre_iod * iod), all=summ(np.ones(Tw, bool)), counts=counts.tolist())


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub layer-reg', description=__doc__.split('\n')[0])
    ap.add_argument('--render', help='a render dir (warp.json + full.mp4) or a warp.json path (required without --pix; with '
                                     '--pix optional: the render\'s logged jaw and its skipped frames)')
    ap.add_argument('--video', help='the rendered video (default <dir>/full.mp4)')
    ap.add_argument('--json')
    ap.add_argument('--md')
    ap.add_argument('--w', type=int, default=4)
    ap.add_argument('--no-texture', action='store_true')
    ap.add_argument('--label')
    ap.add_argument('--pix', action='store_true', help='v12: the pixel registration ruler (template-tracked teeth patches in '
                                                        'the skull-stabilised crop) on --video: a take or a render')
    ap.add_argument('--take-trk', help='--pix: npz with S3 = the take\'s raw MediaPipe 3D track (the warp\'s input)')
    ap.add_argument('--frames', type=int, help='--pix: frame count (default: the track\'s)')
    ap.add_argument('--take', help='--pix: the take video (its skin L keys the teeth; default = --video)')
    ap.add_argument('--lm', help='--pix: IMAGE-mode landmark cache of --video (computed and saved when missing)')
    ap.add_argument('--ppi', type=float, default=200.0, help='--pix: crop px per IOD')
    a = ap.parse_args(argv)
    if a.pix: return pix_main(a)
    if not a.render: ap.error('--render is required (or --pix)')
    wj = a.render if a.render.endswith('.json') else os.path.join(a.render, 'warp.json')
    d = os.path.dirname(os.path.abspath(wj))
    W = json.load(open(wj))
    frames = W['frames']
    meta = ((W.get('teeth') or {}).get('layers') or {}).get('reg_meta')
    if not any(f.get('reg') for f in frames):
        print(f'layer-reg: {wj} has no per-frame reg entries (rendered before v11, or without the layer stack)', file=sys.stderr)
        return 2
    video = a.video or os.path.join(d, 'full.mp4')
    r = measure(frames, video if os.path.exists(video) else None, meta, a.w, not a.no_texture)
    label = a.label or os.path.basename(d)
    if a.json: json.dump(r, open(a.json, 'w'))
    md = table_md(label, r)
    if a.md: open(a.md, 'w').write(md)
    print(md)
    print(line(label, r))
    return 0


if __name__ == '__main__':
    sys.exit(main())
