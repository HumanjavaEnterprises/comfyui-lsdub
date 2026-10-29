#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""coart.py: the COARTICULATION target planner of --mode warp (mv30 v10c: --warp-target coart; the default v9 keeps
mouth_warp.target_features, the per-frame step targets).

v9 aims every frame at its own viseme's band (the take's shape clipped into the band, pushed toward the class centre):
every phone reaches its shape, the shapes step from frame to frame, the openings sit in a compressed middle of the take's
range and the mouth keeps moving through the pauses. Here the targets are PLANNED over the whole clip, in the face's own
feature space (inner-lip gap and mouth width, in units of the take's own range):

  1. SEGMENTS: the phones of the alignment the track was built from (on the track's own timing: its lead is detected by
     rebuilding it), short gaps as weak 'sil', aligner silences >= --coart-rest-ms as REST, `skip` spans as free segments
     (with v10a's --vocalisations rest: rests; --vocalisations mouth: planned as words).
  2. TARGETS: v9's band rule (library bands + class floor, the centre aim, the audio's openness) on the take's mean shape
     over each segment, + the take's own small drift inside sustained segments; a rest = the face's own rest mouth (the
     take's closed mode, at most the library's resting opening).
  3. DOMINANCE BLENDING (Cohen & Massaro), per dimension: a plateau over the segment, rising from 10 % over
     --coart-dom-width x a per-class factor before it and decaying as fast after it. Closures PP / FF and sibilants strong
     and narrow, vowels broad and soft, DD / kk / nn / TH weak (they inherit the vowel around them), lip rounding
     (ou / oh / RR / CH) anticipated over --coart-anticipate ms; a rest enters / leaves at the take's own closing /
     opening times (the naturalness ruler's fall / rise), its plateau inside the pause. DURATION-DEPENDENT UNDERSHOOT: a
     segment shorter than its class's reach time (its excursion / the take's own transition speed) keeps
     1 - U (1 - dur / reach) of its dominance (U = --coart-undershoot).
  4. (optional) an EXTERNAL DYNAMICS source (--coart-dyn csv, e.g. v10b's Audio2Face-3D run on the song timeline):
     percentile-mapped onto the take's range and mixed into the target by --coart-dyn-mix outside pauses / skip spans.
  5. THE TAKE'S OWN RANGE (--coart-range): the plan's p5 / p95 mapped onto the take's own, the neutral level fixed (shut
     stays shut, round stays round), class targets far outside the face's range clipped first; before and after the fit.
  6. CLOSURES (P / B / M, F / V phones' core frames + the track's own closure runs) are HARD: the opening <= min(the class
     target, the take's own p5 opening), pressed through mouth_warp's dead-band (press_closures).
  7. THE MANIFOLD: targets outside the take's own (gap, width) hull are projected onto it (a closure frame keeps its press).
  8. THE DYNAMICS PRIOR (the "retraining" on the face's own motion): per dimension a whole-clip least-squares smoother
     (acceleration + jerk, zero-phase by construction, closures as hard hinges, a soft speed limit at the take's own p99
     speed) whose weight moves the plan's velocity / acceleration p95 --coart-prior of the way toward the take's own.
  9. JAW / LIP SPLIT (--coart-jaw-share auto | 0..1): the share of an opening carried by the jaw (lower lip + chin; the rest
     by the upper lip) set on the mouth model's J. v9 = 1; auto = the take's own share as the naturalness ruler measures it,
     moved 20 % toward the jaw for singing. The jaw part reaches the teeth hinge plan through the planned lower lip.

plan_clip() returns the planned target features (T x 2, canonical px), which then go through mouth_warp's residual,
dead-band confidence, temporal ease, size cap and fold guard unchanged, and the frames whose closure depth the field
keeps. The plan's log goes to <warp-json stem>_coart.json.
"""
import json, os
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import mouth_warp as mw
from . import viseme_track as vt

# per class: (dominance of the OPENING, dominance of the WIDTH, rise width x --coart-dom-width)
DOM = {'PP': (4.0, 0.3, 0.6), 'FF': (3.0, 0.4, 0.6), 'SS': (1.6, 0.6, 0.7), 'CH': (1.4, 1.3, 0.7), 'TH': (0.6, 0.3, 0.8),
       'DD': (0.45, 0.25, 0.8), 'kk': (0.4, 0.25, 0.8), 'nn': (0.45, 0.25, 0.8), 'RR': (0.6, 1.0, 1.0),
       'aa': (1.0, 0.8, 1.3), 'E': (1.0, 1.0, 1.3), 'ih': (1.0, 1.0, 1.3), 'oh': (1.0, 1.4, 1.3), 'ou': (1.0, 1.6, 1.3),
       'sil': (0.5, 0.5, 1.5), 'skip': (1.0, 1.0, 1.3), 'rest': (1.5, 1.2, 1.5)}
ROUNDED = ('ou', 'oh', 'RR', 'CH')
CLOSURES = ('PP', 'FF')
OPEN_FROM_AUDIO = ('aa', 'E', 'oh', 'ih')          # v9: the track's openness raises these vowels' lower bound
NEUTRAL_DOM = 0.1                                  # the face's neutral shape: a weak pull everywhere
DRIFT_CAP = 0.1                                    # the take's own drift kept inside a sustained segment, +- (units of its range)
DRIFT_MIN = 0.15                                   # s: a segment this long starts keeping the take's drift (all of it from 2x)
SING_BIAS = 0.2                                    # auto jaw share: the take's own share moved this far toward 1
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 95, 88, 178, 87, 14, 317, 402, 318, 324]
SHAPE_P = 1.5                                      # dominance decay exponent: 2^-(tau / half)^p
RISE = float(np.log2(10.0) ** (1.0 / SHAPE_P))    # tau / h at 10 % dominance: the rise width = RISE x h
VMAX_FLOOR = 0.35                                  # the speed limit is never below this (take units / frame: shut -> open in 3)
PRIOR_FLOOR = ((0.08, 0.03), (0.15, 0.06))        # the prior's take velocity / acceleration p95 floors (opening, width), /frame
REST_OPEN_MAX = 0.15                               # the rest mouth's opening is at most this (the library's resting opening, take units)
REST_CLOSED_Q = 20.0                               # the take's closed mode = its frames below this percentile of opening
RANGE_ANCHORS = 'neutral'                          # the plan's range match: its p5 / p95 -> the take's, the neutral level (0) fixed


# --------------------------------------------------------------------------------------------- units / bands --
def norm(F, rng):
    """canonical (gap, width) -> the take's own units: opening 0 = shut .. 1 = its p95; width -1 = its p10 .. 0 =
    median .. +1 = its p90 (as mouth_warp.target_features)."""
    F = np.asarray(F, float)
    go, gs = rng['g_lo'], max(1e-6, rng['g_hi'] - rng['g_lo'])
    up, dn = max(1e-6, rng['w_hi'] - rng['w_med']), max(1e-6, rng['w_med'] - rng['w_lo'])
    uw = np.where(F[..., 1] >= rng['w_med'], (F[..., 1] - rng['w_med']) / up, (F[..., 1] - rng['w_med']) / dn)
    return np.stack([(F[..., 0] - go) / gs, uw], -1)


def denorm(U, rng):
    U = np.asarray(U, float)
    go, gs = rng['g_lo'], max(1e-6, rng['g_hi'] - rng['g_lo'])
    w = rng['w_med'] + U[..., 1] * np.where(U[..., 1] >= 0, rng['w_hi'] - rng['w_med'], rng['w_med'] - rng['w_lo'])
    return np.stack([go + U[..., 0] * gs, w], -1)


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def to_band_soft(u, lo, hi, centre, ramp=0.1):
    """v9's band rule (mouth_warp._to_band: inside the band untouched; outside, past the edge toward the band centre by
    `centre`), made continuous: the push grows from 0 at the edge to its full size `ramp` outside it."""
    if lo <= u <= hi: return float(u)
    e = lo if u < lo else hi
    mid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (lo + 0.5 if np.isfinite(lo) else hi - 0.5)
    return float(e + centre * smootherstep(abs(u - e) / ramp) * (mid - e))


def class_bands(v, bands, open_audio, a):
    """(open lo, hi), (width lo, hi) of viseme v in the take's units: the library bands (+ class floor) as v9 uses
    them, the open-gain / width-gain flags, the audio's openness on vowels, the PP press."""
    VO, VW = bands or (mw.V_OPEN, mw.V_WIDTH)
    og, wg = getattr(a, 'warp_open_gain', 1.0), getattr(a, 'warp_width_gain', 1.0)
    if v == 'skip': return (-mw.INF, mw.INF), (-mw.INF, mw.INF)
    lo, hi = VO.get(v, (0.0, 0.6))
    if v in OPEN_FROM_AUDIO: lo = max(lo, min(hi, 0.8 * float(open_audio)))
    cd = getattr(a, 'warp_close_depth', 0.0) or 0.0
    if v == 'PP': lo, hi = -cd, 0.03 - cd
    lo = lo * og if lo > 0 else lo
    hi = hi * max(1.0, og) if hi > 0 else hi
    wl, wh = VW.get(v, (-mw.INF, mw.INF))
    wl = wl * wg if wl > -mw.INF else -mw.INF
    wh = wh * wg if wh < mw.INF else mw.INF
    return (lo, hi), (wl, wh)


# ---------------------------------------------------------------------------------------------- the phones --
def load_align(path, track_path=None):
    """the alignment the track was built from (its `align` field): as given, else relative to the track's folder and
    its parents. None when not found."""
    if not path: return None
    cands = [path] if os.path.isabs(path) else [path] + ([os.path.join(d, path) for d in _parents(track_path)] if track_path else [])
    for c in cands:
        if os.path.exists(c):
            try: return json.load(open(c))
            except Exception: return None
    return None


def _parents(p, k=5):
    d = os.path.dirname(os.path.abspath(p))
    out = []
    for _ in range(k):
        out.append(d)
        d2 = os.path.dirname(d)
        if d2 == d: break
        d = d2
    return out


def detect_lead(align, fr_t, t0, n, fps, grid=np.arange(0.0, 0.2051, 0.01)):
    """the lead the track was built with: the track rebuilt (viseme_track.build) for each lead on the grid, the visemes
    compared; the median of the best-matching leads. -> (lead s, match share)."""
    vis = [f['vis'] for f in fr_t[:n]]
    res = []
    for L in grid:
        fr = vt.build(align, t0, n, fps, lead=float(L))
        res.append((float(L), float(np.mean([x['vis'] == y for x, y in zip(fr, vis)]))))
    best = max(m for _, m in res)
    Ls = [L for L, m in res if m >= best - 1e-9]
    return float(np.median(Ls)), best


def segments(fr_t, n, fps, t0, align=None, shift=0.0, rest_s=0.0, skip_kind='free'):
    """phone segments covering the clip: dicts (v viseme, a / b seconds, p phone, w word, frames: the frames whose
    centre is inside). From the alignment when given (each phone's start moved `shift` s earlier: a track built with
    a longer lead than the planner's own anticipation starts its shapes that much earlier; the ends stay), else from the track's own viseme runs;
    gaps -> sil, a gap of rest_s s or more (> 0) -> 'rest' (a pause: the mouth goes to the face's rest shape); frames
    with a truthy `skip` -> one segment per span (phones centred inside it dropped): 'skip' (free: the take's own shape;
    skip_kind 'free', v10a's --vocalisations skip / auto), 'rest' (v10a's --vocalisations rest), or the flags are ignored
    and the phones planned as words (skip_kind 'ignore': --vocalisations mouth)."""
    tc = t0 + (np.arange(n) + 0.5) / fps
    lo_t, hi_t = t0 - 1.0, t0 + n / fps + 1.0
    if align is not None:
        ph = [dict(v=p['vis'], a=p['start'] - shift, b=p['end'], p=p['p'], w=p['word']) for p in vt.phone_list(align)]
    else:
        ph = []
        lab = [fr_t[min(i, len(fr_t) - 1)]['vis'] for i in range(n)]
        i = 0
        while i < n:            # no alignment: the track's own viseme runs (its lead and minimum hold included)
            j = i
            while j + 1 < n and lab[j + 1] == lab[i]: j += 1
            if lab[i] != 'sil': ph.append(dict(v=lab[i], a=t0 + i / fps, b=t0 + (j + 1) / fps, p='', w=''))
            i = j + 1
    sk = np.array([bool(fr_t[min(i, len(fr_t) - 1)].get('skip')) and skip_kind != 'ignore' for i in range(n)])
    spans = []
    i = 0
    while i < n:
        if not sk[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and sk[j + 1]: j += 1
        spans.append((t0 + i / fps, t0 + (j + 1) / fps))
        i = j + 1
    inside = lambda t: any(a <= t < b for a, b in spans)
    ph = sorted([p for p in ph if p['b'] > lo_t and p['a'] < hi_t and not inside((p['a'] + p['b']) / 2)], key=lambda p: p['a'])
    ph += [dict(v='rest' if skip_kind == 'rest' else 'skip', a=a, b=b, p='', w='') for a, b in spans]
    ph.sort(key=lambda p: p['a'])
    out, cur = [], lo_t
    gap = lambda a, b: 'rest' if rest_s > 0 and b - a >= rest_s else 'sil'
    for p in ph:
        if p['a'] - cur >= 0.5 / fps: out.append(dict(v=gap(cur, p['a']), a=cur, b=p['a'], p='', w=''))
        a = max(p['a'], cur)
        if p['b'] > a:
            out.append(dict(p, a=a))
            cur = max(cur, p['b'])
    if hi_t - cur > 1e-6: out.append(dict(v=gap(cur, hi_t), a=cur, b=hi_t, p='', w=''))
    for s in out:
        s['frames'] = [int(k) for k in np.flatnonzero((tc >= s['a']) & (tc < s['b']))]
    return out, tc


def core_frames(s, tc, pad=0.02, min_n=2, reach=0.06):
    """the frames a closure must reach: centres within the segment +- pad, at least min_n (the nearest to it, among
    frames whose centre lies within `reach` s of the segment, so a closure just before the clip still shuts its first
    frame and one far outside shuts nothing)."""
    len(tc)
    idx = [int(k) for k in np.flatnonzero((tc >= s['a'] - pad) & (tc < s['b'] + pad))]
    dist = np.maximum(0.0, np.maximum(s['a'] - tc, tc - s['b']))
    for k in np.argsort(dist + 1e-6 * np.abs(tc - 0.5 * (s['a'] + s['b']))):
        if len(idx) >= min_n or dist[k] > reach: break
        if int(k) not in idx: idx.append(int(k))
    return sorted(idx)


# ---------------------------------------------------------------------------------------------- dominance --
def dominance(segs, tc, dom_ms, anticipate_ms, reach=None, undershoot=0.0):
    """D (S x n x 2): per segment, frame and dimension (opening, width); plateau 1 over the segment, 2^-(tau / h)^p
    outside: the dominance rises from 10 % to its plateau over the class's rise width = dom width x the class factor
    (h = that / RISE); a rounded class's width rises over max(that, anticipate): its rounding starts that early; x the
    class dominance; x the undershoot factor r = 1 - U (1 - min(1, dur / reach)). Two equal neighbours hand over (10 ->
    90 %) in about 2 x the rise width."""
    S, n = len(segs), len(tc)
    D = np.zeros((S, n, 2))
    R = np.ones(S)
    for k, s in enumerate(segs):
        ao, aw, f = DOM.get(s['v'], DOM['aa'])
        h = max(1e-3, f * dom_ms / 1000.0 / RISE)
        h_ant, h_car = s.get('h', (h, h))          # a rest segment: the take's own closing / opening times
        ca, cb = s.get('core', (s['a'], s['b']))   # ... and its plateau starts / ends inside the pause
        tb = np.maximum(0.0, ca - tc)
        ta = np.maximum(0.0, tc - cb)
        h_ant_w = max(h_ant, anticipate_ms / 1000.0 / RISE) if s['v'] in ROUNDED else h_ant
        dec = lambda tau, hh: np.power(2.0, -np.power(tau / hh, SHAPE_P))
        D[k, :, 0] = ao * np.where(tb > 0, dec(tb, h_ant), np.where(ta > 0, dec(ta, h_car), 1.0))
        D[k, :, 1] = aw * np.where(tb > 0, dec(tb, h_ant_w), np.where(ta > 0, dec(ta, h_car), 1.0))
        if reach is not None and undershoot > 0 and s['v'] not in ('sil', 'skip', 'rest'):
            R[k] = 1.0 - undershoot * (1.0 - min(1.0, (s['b'] - s['a']) / max(1e-3, reach.get(s['v'], 0.1))))
            D[k] *= R[k]
    return D, R


def reach_times(u, neutral, bands, a, fps, centre, lo=0.04, hi=0.3):
    """per class, the time (s) the take needs for the class's excursion from its neutral shape: excursion (take units)
    / the take's own transition speed per dimension (0.8 x amplitude / 10-90 % time, median over its transitions,
    measured with the naturalness ruler's transition detector). -> ({class: reach s}, {dim: speed / s})."""
    from ..audits.naturalness import transitions
    sp = []
    for d in range(2):
        tr = transitions(u[:, d], fps)[3]
        v = [0.8 * A / max(1e-3, t / 1000.0) for A, t, _ in tr if t > 0]
        sp.append(float(np.median(v)) if v else 2.0)
    out = {}
    for v in DOM:
        if v in ('sil', 'skip', 'rest'): continue
        (ol, oh), (wl, wh) = class_bands(v, bands, 0.0, a)
        ex = [abs(to_band_soft(neutral[0], ol, oh, centre) - neutral[0]) / sp[0],
              abs(to_band_soft(neutral[1], wl, wh, centre) - neutral[1]) / sp[1] if (wl, wh) != (-mw.INF, mw.INF) else 0.0]
        out[v] = float(np.clip(max(ex), lo, hi))
    return out, dict(open=sp[0], width=sp[1])


# ----------------------------------------------------------------------------------------------- targets --
def segment_targets(segs, u, fr_t, bands, a, centre, rest=None):
    """per segment: the band per dimension and its target = v9's band rule on the take's mean shape over the segment
    (a segment shorter than a frame: the frame whose centre is nearest, s['near']); a 'rest' segment: the face's rest
    shape `rest` (no band, no drift)."""
    for s in segs:
        idx = s['frames'] or [s['near']]
        if s['v'] == 'rest':
            s['band'] = ((-mw.INF, mw.INF), (-mw.INF, mw.INF))
            s['mean'] = u[idx].mean(0)
            s['T'] = (float(rest[0]), float(rest[1])) if rest is not None else (float(s['mean'][0]), float(s['mean'][1]))
            continue
        oa = float(np.mean([fr_t[min(i, len(fr_t) - 1)].get('open', 0) for i in idx]))
        (ol, oh), (wl, wh) = class_bands(s['v'], bands, oa, a)
        s['band'] = ((ol, oh), (wl, wh))
        um = u[idx].mean(0)
        s['mean'] = um
        s['T'] = (to_band_soft(um[0], ol, oh, centre) if s['v'] != 'skip' else float(um[0]),
                  to_band_soft(um[1], wl, wh, centre) if (wl, wh) != (-mw.INF, mw.INF) else float(um[1]))
    return segs


def frame_targets(s, u):
    """segment s's target at every frame (n x 2): its target + the take's own drift around the segment's mean (clipped
    to +- DRIFT_CAP; on sustained segments only: none below DRIFT_MIN s, all of it from 2 x DRIFT_MIN; none on closures),
    kept inside the band; an unbanded dimension (or a skip span) = the take's own."""
    (ol, oh), (wl, wh) = s['band']
    out = np.empty_like(u)
    if s['v'] == 'skip': return u.copy()
    if s['v'] == 'rest': return np.tile(np.asarray(s['T'], float), (len(u), 1))
    cap = 0.0 if s['v'] in CLOSURES else DRIFT_CAP * float(smootherstep(((s['b'] - s['a']) - DRIFT_MIN) / DRIFT_MIN))
    out[:, 0] = np.clip(s['T'][0] + np.clip(u[:, 0] - s['mean'][0], -cap, cap), ol, oh)
    if (wl, wh) == (-mw.INF, mw.INF): out[:, 1] = u[:, 1]
    else: out[:, 1] = np.clip(s['T'][1] + np.clip(u[:, 1] - s['mean'][1], -cap, cap), wl, wh)
    return out


def blend(segs, D, u, neutral):
    """the dominance-weighted blend of every segment's frame targets + the neutral pull -> B (n x 2), the segments'
    weights W (S x n x 2) and the neutral's w0 (n x 2); W.sum(0) + w0 = 1 everywhere."""
    num = NEUTRAL_DOM * np.asarray(neutral, float)[None, :] * np.ones_like(u)
    den = NEUTRAL_DOM * np.ones_like(u)
    for k, s in enumerate(segs):
        num += D[k] * frame_targets(s, u)
        den += D[k]
    return num / den, D / den[None], NEUTRAL_DOM / den


# ------------------------------------------------------------------------------------- the dynamics prior --
LAM_A, LAM_J = 1.0, 0.4             # acceleration : jerk in the smoother


def _smoother(n, lam):
    """lam (LAM_A D2'D2 + LAM_J D3'D3): the smoother's quadratic form."""
    I = np.eye(n)
    Q = np.zeros((n, n))
    if lam > 0 and n > 3:
        D2, D3 = np.diff(I, 2, 0), np.diff(I, 3, 0)
        Q = lam * (LAM_A * D2.T @ D2 + LAM_J * D3.T @ D3)
    return Q


def quad_fit(B, w, lam):
    """the plain fit (no bounds): argmin sum w (x - B)^2 + lam (LAM_A |d2 x|^2 + LAM_J |d3 x|^2)."""
    B = np.asarray(B, float)
    W = np.diag(np.asarray(w, float))
    return np.linalg.solve(W + _smoother(len(B), lam) + 1e-9 * np.eye(len(B)), W @ B)


def smooth_fit(B, w, lam, caps=None, vmax=None, mu=30.0, big=1e4, iters=40):
    """x = argmin sum w (x - B)^2 + lam (LAM_A |d2 x|^2 + LAM_J |d3 x|^2) + mu sum hinge(|d1 x| - vmax)^2
    + big sum hinge(x_k - cap_k)^2 (one series; caps: {frame: upper bound}, clamped exactly at the end; vmax: a soft
    speed limit per frame). Semismooth Newton on the hinges (the active set re-read from each solution; convex, so it
    settles; should it cycle, the visited solution with the lowest objective is kept). Zero-phase by construction."""
    B = np.asarray(B, float)
    n = len(B)
    w = np.asarray(w, float)
    I = np.eye(n)
    D1 = np.diff(I, 1, 0)
    Q = _smoother(n, lam)
    H0 = np.diag(w) + Q + 1e-9 * I
    g0 = w * B
    ck = sorted(caps or {})
    cv = np.array([caps[k] for k in ck]) if ck else np.zeros(0)
    def obj(z):
        o = float(np.sum(w * (z - B) ** 2) + z @ Q @ z)
        if vmax is not None: o += mu * float(np.sum(np.maximum(0.0, np.abs(np.diff(z)) - vmax) ** 2))
        if ck: o += big * float(np.sum(np.maximum(0.0, z[ck] - cv) ** 2))
        return o
    x = np.linalg.solve(H0, g0)
    best = (obj(x), x)
    seen = set()
    for _ in range(iters):
        ac = [k for k, c in zip(ck, cv) if x[k] > c]
        dv = np.diff(x)
        av = [int(k) for k in np.flatnonzero(np.abs(dv) > vmax)] if vmax is not None else []
        key = (tuple(ac), tuple(av), tuple(int(np.sign(dv[k])) for k in av))
        if key in seen: break
        seen.add(key)
        H = H0.copy()
        g = g0.copy()
        for k in ac:
            H[k, k] += big
            g[k] += big * caps[k]
        for k in av:
            d = D1[k]
            H += mu * np.outer(d, d)
            g += mu * d * np.sign(dv[k]) * vmax
        x = np.linalg.solve(H, g)
        o = obj(x)
        if o < best[0]: best = (o, x)
    x = best[1].copy()
    for k, c in zip(ck, cv): x[k] = min(x[k], c)
    return x


def p95s(x):
    x = np.asarray(x, float)
    return float(np.percentile(np.abs(np.diff(x)), 95)), float(np.percentile(np.abs(np.diff(x, 2)), 95))


def prior_lambda(fit, take, prior, floor=(0.0, 0.0), lo=1e-3, hi=1e4, steps=30):
    """the smoother weight whose fit (fit(lam) -> the series, closures and the speed limit included) has velocity /
    acceleration p95 at the plan's own moved toward the take's own by `prior` (log space: target = plan^(1 - prior) x
    take^prior, the two log ratios averaged); 0 when the unsmoothed plan is already as calm as that. The take's
    statistics are floored (a take that hardly moves must not freeze the plan). -> (lam, the statistics)."""
    vb, ab = p95s(fit(0.0))
    vt_, at_ = p95s(take)
    vt_, at_ = max(vt_, floor[0]), max(at_, floor[1])
    st = dict(v95=(vb, vt_), a95=(ab, at_))
    if prior <= 0: return 0.0, st
    tv = np.exp((1 - prior) * np.log(max(vb, 1e-9)) + prior * np.log(max(vt_, 1e-9)))
    ta = np.exp((1 - prior) * np.log(max(ab, 1e-9)) + prior * np.log(max(at_, 1e-9)))
    st['target'] = (float(tv), float(ta))
    def goal(lam):
        v, a_ = p95s(fit(lam))
        return 0.5 * (np.log(max(v, 1e-9) / tv) + np.log(max(a_, 1e-9) / ta))
    if goal(lo) <= 0: return 0.0, st
    if goal(hi) > 0: return hi, st
    a_, b_ = np.log(lo), np.log(hi)
    for _ in range(steps):
        m = 0.5 * (a_ + b_)
        if goal(np.exp(m)) > 0: a_ = m
        else: b_ = m
    return float(np.exp(0.5 * (a_ + b_))), st


# -------------------------------------------------------------------------------------------- the manifold --
class Hull:
    """the shapes this face makes: the convex hull of the take's own (opening, width) in its units (frames beyond its
    98 % Mahalanobis radius dropped, grown x grow about the centroid). inside() / project() (nearest boundary point).
    A degenerate take (no spread in one dimension: no hull) constrains nothing."""

    def __init__(self, U, q=98.0, grow=1.05):
        from scipy.spatial import ConvexHull, Delaunay
        U = np.asarray(U, float)
        self.ok = False
        self.V = U[:1].copy()
        try:
            mu = U.mean(0)
            C = np.cov(U.T)
            if len(U) < 4 or np.linalg.eigvalsh(C).min() < 1e-6: return
            d = np.einsum('ij,jk,ik->i', U - mu, np.linalg.inv(C), U - mu)
            P = U[d <= np.percentile(d, q)]
            c = P.mean(0)
            P = c + (P - c) * grow
            h = ConvexHull(P)
            self.V = P[h.vertices]
            self.tri = Delaunay(self.V)
            self.c = c
            self.ok = True
        except Exception:
            self.ok = False

    def inside(self, X):
        X = np.atleast_2d(X)
        return self.tri.find_simplex(X) >= 0 if self.ok else np.ones(len(X), bool)

    def project(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        out = X.copy()
        if not self.ok: return out
        V = self.V
        E = np.roll(V, -1, 0) - V
        for i, p in enumerate(X):
            if self.inside(p)[0]: continue
            t = np.clip(((p - V) * E).sum(1) / np.maximum((E * E).sum(1), 1e-12), 0, 1)
            Q = V + t[:, None] * E
            out[i] = Q[np.argmin(((Q - p) ** 2).sum(1))]
        return out


# ------------------------------------------------------------------------------------------------- jaw split --
def take_jaw_share(S3raw):
    """the take's own jaw share, as the naturalness ruler measures it (naturalness.shares on its mouth series): the share
    of a gap change carried by the lower lip (1 - the upper lip's), on the pipeline's own track of the take."""
    from ..audits.naturalness import mouth_series, shares
    try:
        sh = shares(mouth_series(np.asarray(S3raw, float))[0])
        up = sh['up']
        return float(np.clip(1.0 - up, 0.0, 1.0)) if np.isfinite(up) else None
    except Exception:
        return None


def jaw_split(model, share, ctrl=None, jaw_gain=1.0, s_take=None):
    """set the gap column of the mouth model's J for a jaw share s (lower lip + chin carry s of an opening, the upper
    lip 1 - s): J_gap = J_fit - (s U + (1 - s) L), U / L = the upper / lower lip's mean response in the take's own fit
    (s = 1: mouth_warp's jaw_reference, v9). share: 'auto' (the take's own share s_take -- take_jaw_share(), else its
    lip-point regression -- moved SING_BIAS of the way toward 1: singing opens with the jaw) or 0..1.
    -> (s, the take's own share)."""
    ctrl = ctrl or mw.CTRL
    ix = {k: j for j, k in enumerate(ctrl)}
    K = len(ctrl)
    Jf = model.coef[1].reshape(K, 3)
    U = Jf[[ix[k] for k in mw.UPPER]].mean(0)
    L = Jf[[ix[k] for k in LOWER_LIP]].mean(0)
    span = L[1] - U[1]
    if s_take is None: s_take = float(np.clip(L[1] / span, 0.0, 1.0)) if span > 1e-6 else 1.0
    s = float(np.clip(s_take + SING_BIAS * (1.0 - s_take), 0.0, 1.0)) if str(share) == 'auto' else float(np.clip(float(share), 0.0, 1.0))
    model.J = model.J.copy()
    model.J[0] = Jf - (s * U + (1.0 - s) * L)
    if jaw_gain != 1.0: model.J[0, [ix[k] for k in mw.CHIN_C]] *= jaw_gain
    return s, s_take


# ---------------------------------------------------------------------------------- the take's own range --
def qmap_fn(src, ref, anchors=None):
    """the percentile map of src's distribution onto ref's: a monotone function f(x) = ref's value at x's quantile in
    src. anchors None: the full map (every quantile; ties share their mean quantile); anchors = percentiles (e.g. 5, 50,
    95) or 'neutral' (src p5 / p95 -> ref p5 / p95 with the neutral level 0 -> 0): a piecewise-linear map through those
    points only (extended linearly past the end anchors, clipped to ref's range +- 10 %), which keeps the plan's own
    relative levels between its anchors. None when either distribution is degenerate."""
    s_ = np.asarray(src, float)
    r = np.asarray(ref, float)
    if len(s_) < 3 or len(r) < 3 or np.ptp(s_) < 1e-6 or np.percentile(r, 95) - np.percentile(r, 5) < 0.05: return None
    if anchors is None:
        o = np.sort(s_)
        q = np.linspace(0.0, 1.0, len(o))
        us, inv = np.unique(o, return_inverse=True)
        qu = np.bincount(inv, q) / np.bincount(inv)
        return lambda x: np.quantile(r, np.clip(np.interp(np.asarray(x, float), us, qu), 0.0, 1.0))
    if anchors == 'neutral':
        # the plan's p5 / p95 onto the take's, the neutral level fixed (gap: shut 0 -> 0; width: the take's median 0 -> 0),
        # so a closure stays shut, a round shape round and a spread one spread: only the extent is the take's own
        a5, a95, b5, b95 = np.percentile(s_, 5), np.percentile(s_, 95), np.percentile(r, 5), np.percentile(r, 95)
        pairs = sorted({(0.0, 0.0)} | {(float(a), float(b)) for a, b in ((a5, b5), (a95, b95)) if abs(a) > 1e-3 and np.sign(a) == np.sign(b)})
        ps, pr = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        if len(ps) < 2 or np.any(np.diff(pr) < 0): return None
    else:
        ps, pr = np.percentile(s_, anchors), np.percentile(r, anchors)
    if np.any(np.diff(ps) <= 1e-6): return None
    lo_s = (pr[1] - pr[0]) / (ps[1] - ps[0])
    hi_s = (pr[-1] - pr[-2]) / (ps[-1] - ps[-2])
    m = 0.1 * np.ptp(r)
    def f(x):
        x = np.asarray(x, float)
        y = np.interp(x, ps, pr)
        y = np.where(x < ps[0], pr[0] + lo_s * (x - ps[0]), np.where(x > ps[-1], pr[-1] + hi_s * (x - ps[-1]), y))
        return np.clip(y, r.min() - m, r.max() + m)
    return f


def range_match(X, ref, strength=1.0, weight=None, src_mask=None, anchors=None):
    """the planned trajectory's distribution matched to the take's own, per dimension: X (n x 2) -> X + strength x
    weight x (f(X) - X), f = qmap_fn(X[src_mask], ref, anchors) (the planned active frames onto the take's active frames:
    a rest in a pause keeps its rest shape, weight = 1 - its rest share). -> (X', [mapped?] per dimension)."""
    X = np.asarray(X, float).copy()
    n = len(X)
    wgt = np.ones((n, 2)) if weight is None else np.asarray(weight, float)
    m = np.ones(n, bool) if src_mask is None else np.asarray(src_mask, bool)
    done = []
    for d in range(2):
        f = qmap_fn(X[m, d], ref[:, d], anchors) if strength > 0 and m.sum() >= 3 else None
        if f is not None: X[:, d] = X[:, d] + strength * wgt[:, d] * (f(X[:, d]) - X[:, d])
        done.append(f is not None)
    return X, done


def rest_shape(Uref):
    """the face's own rest mouth: the take's closed mode (its frames below the REST_CLOSED_Q percentile of opening):
    median opening, capped at the library's resting opening REST_OPEN_MAX, and median width (take units)."""
    U = np.asarray(Uref, float)
    k = U[:, 0] <= np.percentile(U[:, 0], REST_CLOSED_Q)
    return np.array([float(np.clip(np.median(U[k, 0]), 0.0, REST_OPEN_MAX)), float(np.median(U[k, 1]))])


def take_times(uo, fps):
    """the take's own closing / opening times (s): the naturalness ruler's 10-90 % fall / rise medians of its opening,
    clipped to 60..300 ms (defaults 120 / 140 ms when it has too few transitions)."""
    from ..audits.naturalness import dynamics_shape
    d = dynamics_shape(np.asarray(uo, float), fps)
    f = d['fall_ms'] if np.isfinite(d['fall_ms']) else 120.0
    r = d['rise_ms'] if np.isfinite(d['rise_ms']) else 140.0
    return float(np.clip(f, 60, 300)) / 1000.0, float(np.clip(r, 60, 300)) / 1000.0


# --------------------------------------------------------------------------------- an external dynamics source --
def load_dyn(path, t0, n, fps, gap='geo_gap_cm', width='geo_width_cm'):
    """a model's mouth trajectory on the song timeline (e.g. Audio2Face-3D's 52 ARKit weights + its mesh's lip gap /
    width, v10b `a2f_cut_25fps.csv`: one row per `time_s` = the start of a 1/fps exposure interval, each row the mean over
    it) -> per frame of this clip (frame k = song t0 + k / fps, its exposure centre t0 + (k + 0.5) / fps): (gap, width)
    in the model's units, or None."""
    import csv
    try:
        with open(path) as fh:
            r = csv.reader(fh)
            h = next(r)
            rows = np.array([[float(x) for x in row] for row in r if row])
        t = rows[:, h.index('time_s')]
        G = rows[:, h.index(gap)]
        Wd = rows[:, h.index(width)]
    except Exception:
        return None
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / fps
    tc = t0 + (np.arange(n) + 0.5) / fps
    if tc[0] < t[0] - dt or tc[-1] > t[-1] + 2 * dt: return None
    return np.stack([np.interp(tc, t + 0.5 * dt, G), np.interp(tc, t + 0.5 * dt, Wd)], 1)


# ------------------------------------------------------------------------------------------------- the plan --
def plan(u, fr_t, fps, t0, neutral=None, bands=None, a=None, align=None, lead_shift=0.0, F_take_u=None, held=None, dyn=None):
    """the planner in the take's own units. u: the take's smoothed shape (n x 2: mouth_warp's Fs through norm());
    neutral: the face's neutral shape (default: its median opening, median width); align: the alignment (None: the
    track's viseme runs); lead_shift: s the phone starts move earlier; F_take_u: the take's per-frame shapes (its range,
    rest shape, manifold and motion statistics; default u); held: frames of the take's track that hold a stale detection
    (left out of its statistics); dyn: an external dynamics source (n x 2, the model's own units: load_dyn), mapped onto
    the take's range and mixed in by --coart-dyn-mix. -> (x n x 2 planned targets in take units, closure frames (PP)
    bool n, info dict)."""
    a = a or _Args()
    n = len(u)
    centre = float(getattr(a, 'warp_centre', 0.8))
    dom_ms = float(getattr(a, 'coart_dom_width', 80.0))
    U_ = float(getattr(a, 'coart_undershoot', 0.6))
    prior = float(getattr(a, 'coart_prior', 0.3))
    ant = float(getattr(a, 'coart_anticipate', 100.0))
    rest_s = float(getattr(a, 'coart_rest_ms', 120.0) or 0.0) / 1000.0
    rho = float(getattr(a, 'coart_range', 1.0))
    mix = float(getattr(a, 'coart_dyn_mix', 0.0) or 0.0) if dyn is not None else 0.0
    Ft_u = u if F_take_u is None else np.asarray(F_take_u, float)
    keep = np.ones(n, bool) if held is None else ~np.asarray(held, bool)[:n]
    if keep.sum() < 8: keep = np.ones(n, bool)
    neutral = np.array([float(np.median(u[keep, 0])), 0.0]) if neutral is None else np.asarray(neutral, float)
    # a vocalisation span (track frames with `skip`): free by default; with v10a's --vocalisations rest a rest, with
    # --vocalisations mouth mouthed as words (the flags ignored)
    skip_kind = {'rest': 'rest', 'mouth': 'ignore'}.get(str(getattr(a, 'vocalisations', None)), 'free')
    segs, tc = segments(fr_t, n, fps, t0, align, lead_shift, rest_s, skip_kind)
    for s in segs:
        if not s['frames']:
            m = 0.5 * (s['a'] + s['b'])
            s['near'] = int(np.clip(np.argmin(np.abs(tc - m)), 0, n - 1))
    # the pauses: the face's own rest mouth, reached and left at the take's own closing / opening speed
    rest = rest_shape(Ft_u[keep])
    t_fall, t_rise = take_times(Ft_u[keep, 0], fps)
    for s in segs:
        if s['v'] == 'rest':
            # the mouth closes INTO the pause over the take's own closing time and opens out of it over its opening time
            # (2 neighbours hand over 10 -> 90 % in ~4.3 h): the rest's plateau starts half a closing time after the last
            # phone and ends half an opening time before the next, so the phones keep their own shapes to their edges
            s['h'] = (t_fall / 4.3, t_rise / 4.3)
            din, dout = 0.5 * t_fall, 0.5 * t_rise
            L = s['b'] - s['a']
            k = min(1.0, max(0.0, L - 1.0 / fps) / max(1e-6, din + dout))
            s['core'] = (s['a'] + k * din, s['b'] - k * dout)
    segment_targets(segs, u, fr_t, bands, a, centre, rest)
    reach, speed = reach_times(u, neutral, bands, a, fps, centre)
    D, R = dominance(segs, tc, dom_ms, ant, reach, U_)
    B, Wt, w0 = blend(segs, D, u, neutral)
    is_rest = np.array([s['v'] == 'rest' for s in segs])
    is_skip = np.array([s['v'] == 'skip' for s in segs])
    r_rest = Wt[is_rest].sum(0) if is_rest.any() else np.zeros((n, 2))
    r_skip = Wt[is_skip].sum(0) if is_skip.any() else np.zeros((n, 2))
    pause = np.zeros(n, bool)
    for s in segs:
        if s['v'] == 'rest': pause[s['frames']] = True
    # the external dynamics (e.g. Audio2Face-3D): its trajectory mapped onto the take's own range (percentile map, the
    # take's active frames), mixed into the target outside the pauses (the silence rule) and the skipped spans
    Dm = None
    if mix > 0:
        Dm, okd = range_match(np.asarray(dyn, float), Ft_u[keep & ~pause] if (keep & ~pause).sum() >= 8 else Ft_u[keep], 1.0)
        if not all(okd):
            Dm = None
            mix = 0.0
    T0 = B.copy()
    if Dm is not None:
        md = mix * np.clip(1.0 - r_rest - r_skip, 0.0, 1.0)
        T0 = (1.0 - md) * B + md * Dm
    # the face's own range: the planned active frames' distribution matched to the take's active frames' (pauses keep
    # their rest; a skipped span is the take's own already)
    act = (r_rest[:, 0] < 0.5) & (r_skip[:, 0] < 0.5)
    wact = np.clip(1.0 - r_rest - r_skip, 0.0, 1.0)
    # a class target far outside the face's own range (a library round / spread floor) must not set the scale of the
    # whole map (it would flatten every other class on its side): targets clipped to the take's p1..p99 +- 10 % first
    ref = Ft_u[keep & ~pause] if (keep & ~pause).sum() >= 8 else Ft_u[keep]
    lo1, hi99 = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
    mg = 0.1 * (hi99 - lo1)
    if rho > 0:
        for d in range(2):
            if hi99[d] - lo1[d] >= 0.05: T0[:, d] = np.clip(T0[:, d], lo1[d] - mg[d], hi99[d] + mg[d])   # (a degenerate take: none)
    T1, rm_done = range_match(T0, ref, rho, wact, act, RANGE_ANCHORS)
    # hard closures: the opening at a P / B / M (F / V) segment's core frames <= its target and <= the take's own closure
    # depth (its p5 opening); the track's own closure runs too (its lead / minimum hold: a closure lasts as long as v9's)
    p5 = float(np.percentile(Ft_u[keep, 0], 5))
    caps, pp = {}, np.zeros(n, bool)
    tvis = [fr_t[min(i, len(fr_t) - 1)]['vis'] for i in range(n)]
    for s in segs:
        if s['v'] in CLOSURES:
            for k in core_frames(s, tc):
                caps[k] = min(caps.get(k, np.inf), s['T'][0], p5)
                pp[k] |= s['v'] == 'PP'
    i = 0
    while i < n:
        j = i
        while j + 1 < n and tvis[j + 1] == tvis[i]: j += 1
        if tvis[i] in CLOSURES:
            (ol, oh), _ = class_bands(tvis[i], bands, 0.0, a)
            c = min(to_band_soft(float(u[i:j + 1, 0].mean()), ol, oh, centre), p5)
            for k in range(i, j + 1):
                caps[k] = min(caps.get(k, np.inf), c)
                pp[k] |= tvis[i] == 'PP'
        i = j + 1
    w = np.ones(n)
    sk = np.array([bool(fr_t[min(i, len(fr_t) - 1)].get('skip')) for i in range(n)]) & (skip_kind == 'free')
    w[sk] = 0.3
    take = Ft_u[keep]                                  # the take's own per-frame motion: the prior's reference
    vs = float(getattr(a, 'coart_vmax_scale', 1.15) or 1.15)    # the take's own p99 speed x this (a sluggish take caps the mouth)
    vmax = [max(VMAX_FLOOR, float(np.percentile(np.abs(np.diff(take[:, d])), 99)) * vs) for d in range(2)]
    # the manifold: targets moved inside the shapes this face makes before the fit; a closure frame's opening may go
    # below the face's own shut lips (the press), so only its width is held inside the face's width range
    hull = Hull(Ft_u[keep], grow=float(getattr(a, 'coart_hull_grow', 1.05) or 1.05))
    capd = np.zeros(n, bool)
    capd[list(caps)] = True
    free = ~capd
    outB = free & ~hull.inside(T1)
    Bm = T1.copy()
    if outB.any(): Bm[outB] = hull.project(T1[outB])
    if hull.ok: Bm[capd, 1] = np.clip(Bm[capd, 1], hull.V[:, 1].min(), hull.V[:, 1].max())
    lam, pri = [], []
    for d in range(2):
        cd = caps if d == 0 else None
        l, st = prior_lambda(lambda L, d=d, cd=cd: smooth_fit(Bm[:, d], w, L, cd, vmax[d]), take[:, d], prior, PRIOR_FLOOR[d])
        lam.append(l)
        pri.append(st)
    fit = lambda BB, ww: np.stack([smooth_fit(BB[:, d], ww, lam[d], caps if d == 0 else None, vmax[d]) for d in range(2)], 1)
    x = fit(Bm, w)
    nproj = int(outB.sum())
    nfin = 0
    # the fit eases the peaks: the range is matched again on the fitted plan (monotone, so the timing and the order of
    # every shape stay), then anything carried outside the face's shapes is pinned at its projection and refitted
    x, _ = range_match(x, ref, rho, wact, act, RANGE_ANCHORS)
    for _ in range(2):
        out = free & ~hull.inside(x)
        if not out.any(): break
        nfin += int(out.sum())
        B2 = x.copy()
        B2[out] = hull.project(x[out])
        w2 = w.copy()
        w2[out] = 20.0
        x = np.stack([smooth_fit(B2[:, d], w2, 0.05 * lam[d], caps if d == 0 else None, vmax[d]) for d in range(2)], 1)
    if hull.ok: x[capd, 1] = np.clip(x[capd, 1], hull.V[:, 1].min(), hull.V[:, 1].max())
    for k, c in caps.items(): x[k, 0] = min(x[k, 0], c)
    rng_ratio = lambda X: [float(np.ptp(np.percentile(X[:, d], [5, 95])) / max(1e-6, np.ptp(np.percentile(Ft_u[keep, d], [5, 95])))) for d in range(2)]
    info = dict(n_segments=len(segs), dom_ms=dom_ms, undershoot=U_, prior=prior, anticipate_ms=ant, lambda_open=lam[0], lambda_width=lam[1],
                prior_stats=pri, speed_take=speed, reach_s=reach, vmax=vmax, hull_projected=nproj, hull_refit=nfin,
                closures=sorted(int(k) for k in caps), closure_cap_p5=p5, neutral=neutral.tolist(), rest=rest.tolist(),
                rest_ms=rest_s * 1000, take_fall_rise_ms=[t_fall * 1000, t_rise * 1000], pauses=[int(k) for k in np.flatnonzero(pause)],
                range_strength=rho, range_matched=rm_done, range_ratio_blend=rng_ratio(B), range_ratio_plan=rng_ratio(x),
                dyn_mix=mix, dyn=None if Dm is None else np.round(Dm, 3).tolist(), held=[int(k) for k in np.flatnonzero(~keep)],
                segments=[dict(v=s['v'], p=s['p'], w=s['w'], a=round(s['a'], 3), b=round(s['b'], 3), frames=s['frames'][:1] + s['frames'][-1:],
                               T=[round(float(t), 3) for t in s['T']], r=round(float(R[k]), 3)) for k, s in enumerate(segs) if s['b'] > t0 and s['a'] < t0 + n / fps],
                stats=dict(plan=p95s(x[:, 0]) + p95s(x[:, 1]), blend=p95s(B[:, 0]) + p95s(B[:, 1]), take=p95s(take[:, 0]) + p95s(take[:, 1])))
    return x, pp, info


class _Args:
    pass


def held_frames(S3raw):
    """frames where the take's VIDEO-mode track holds a stale detection (a row identical to its neighbour's: the shot
    starts where MediaPipe found no face and held its first detection)."""
    S = np.asarray(S3raw, float)
    n = len(S)
    h = np.zeros(n, bool)
    for i in range(1, n):
        if np.array_equal(S[i], S[i - 1]): h[i] = h[i - 1] = True
    return h


def press_closures(Ft, Fs, closures, caps_px, scale, d0, d1, max_press, iters=3):
    """a closure the warp's dead-band would soften: mouth_warp scales the residual (Ft - Fs) by a confidence that is 0
    inside the dead-band, ramps to 1 by d1 x the take's range and is then eased over +-1 frame (so an isolated closure
    frame next to frames the take already matches gets ~0.6 of it). Each closure frame's planned gap is pressed so that
    the residual the warp applies reaches the closure's cap: Ft = Fs + (cap - Fs) / conf, at most max_press px below the
    cap (the planner's closures are hard; mouth_warp's own arithmetic is unchanged)."""
    Ft = np.asarray(Ft, float).copy()
    ks = [k for k in closures if Fs[k, 0] > caps_px[k]]
    for _ in range(iters):
        conf = gaussian_filter1d(mw.confidence(Ft - Fs, scale, d0, d1), 1.0, mode='nearest') if d0 >= 0 else np.ones(len(Ft))
        for k in ks:
            Ft[k, 0] = max(Fs[k, 0] + (caps_px[k] - Fs[k, 0]) / max(0.25, float(conf[k])), caps_px[k] - max_press)
    return Ft


def plan_clip(a, fr_t, Fs, model, bands, pp_track=None, F=None, S3raw=None):
    """mouth_warp's entry (--warp-target coart): the track's alignment and lead, the jaw split on the model (the take's
    own share measured on S3raw, its raw track), the optional dynamics source (--coart-dyn), the plan;
    -> (Ft canonical px n x 2, closure frames for the residual's keep-close). Writes <warp-json stem>_coart.json."""
    n = len(Fs)
    T = json.load(open(a.track))
    fps = float(T.get('fps', 25))
    t0 = float(T.get('t0', fr_t[0].get('t', 0.0)))
    align = load_align(T.get('align'), a.track)
    lead, match, shift = None, None, 0.0
    if align is not None:
        lead, match = detect_lead(align, fr_t, t0, min(n, len(fr_t)), fps)
        if match < 0.9: lead = None
        else: shift = lead - 0.06        # the timing v9 had: the 0.06 s lead is the planner's own anticipation
    s, s_take = jaw_split(model, getattr(a, 'coart_jaw_share', 'auto'), jaw_gain=getattr(a, 'warp_jaw_gain', 1.0),
                          s_take=take_jaw_share(S3raw) if S3raw is not None else None)
    u = norm(Fs, model.rng)
    Fu = norm(F, model.rng) if F is not None else None
    held = held_frames(S3raw[:n]) if S3raw is not None else None
    dyn = None
    if getattr(a, 'coart_dyn', None) and float(getattr(a, 'coart_dyn_mix', 0.0) or 0.0) > 0:
        dyn = load_dyn(a.coart_dyn, t0, n, fps)
        if dyn is None: print(f'coart: --coart-dyn {a.coart_dyn} unreadable or does not cover this clip; no dynamics mix')
    x, pp, info = plan(u, fr_t, fps, t0, None, bands, a, align, shift, Fu, held, dyn)
    Ft = denorm(x, model.rng)
    rng = model.rng
    gs = max(1e-6, rng['g_hi'] - rng['g_lo'])
    scale = np.array([gs, max(1e-6, rng['w_hi'] - rng['w_lo'])])
    d0 = float(getattr(a, 'warp_deadband', 0.04))
    caps_px = {k: float(rng['g_lo'] + x[k, 0] * gs) for k in info['closures']}
    Ft = press_closures(Ft, Fs, info['closures'], caps_px, scale, d0, d0 + float(getattr(a, 'warp_ramp', 0.08)), 0.15 * gs)
    info['closure_press_px'] = {int(k): round(float(caps_px[k] - Ft[k, 0]), 2) for k in info['closures'] if caps_px[k] - Ft[k, 0] > 0.01}
    info.update(align='alignment' if align is not None else 'track viseme runs', lead=lead, lead_match=match, lead_shift=shift,
                jaw_share=s, jaw_share_take=s_take, dyn_source=getattr(a, 'coart_dyn', None), Ft=np.round(Ft, 3).tolist())
    rr = info['range_ratio_plan']
    print(f"coart plan: {info['n_segments']} segments ({info['align']}, lead {lead}), dom {info['dom_ms']:.0f} ms, undershoot "
          f"{info['undershoot']}, prior {info['prior']} (lambda {info['lambda_open']:.3g} / {info['lambda_width']:.3g}), "
          f"{len(info['closures'])} closure frames (<= take p5 {info['closure_cap_p5']:.3f}), {len(info['pauses'])} pause frames at rest "
          f"{np.round(info['rest'], 2).tolist()}, range p5-p95 vs take {rr[0]:.2f} / {rr[1]:.2f}, hull projected {info['hull_projected']}, "
          f"jaw share {s:.2f} (take {s_take:.2f}), dyn mix {info['dyn_mix']}")
    if getattr(a, 'warp_json', None):
        try: json.dump(info, open(os.path.splitext(a.warp_json)[0] + '_coart.json', 'w'), default=_js)
        except Exception as e: print('coart log not written:', e)
    return Ft, pp


def _js(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
