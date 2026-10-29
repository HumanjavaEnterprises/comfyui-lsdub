#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""naturalness.py: the NATURALNESS RULER (mv30 v10c). How does a render's mouth MOTION differ, statistically, from the
take's own? The take is the same face moving the way it really moves; a render whose mouth reaches every phone's shape,
steps between shapes and makes shape combinations the face never makes reads over-articulated even when every shape is
"right". CPU.

Measured outside-in and identically on the render and on the take: IMAGE-mode MediaPipe on every frame (the v9 gate's
instrument: a face-box crop from the take's track, so every version is measured in the same place), one rigid skull pose
per frame fitted on the video's OWN strict skull + brow points (zero-phase, as the pipeline), every point in that head
frame, in IOD units. Per frame: inner-lip gap (mean of 3 vertical pairs), mouth width (outer + inner corners), corner
heights (left / right), the upper lip's height, the lower lip's and the chin's drop.

Statistics per video (series_stats), then the render vs the take (compare):
  1. kinematics: |velocity| p50 / p95, |acceleration| p95, |jerk| p95 of gap and width (IOD/s, /s^2, /s^3) -> log ratios;
     power spectrum (Welch over the series and its reversal, 0.4-12.5 Hz) of gap and width, each normalised to its
     in-band power -> spectral distance = mean |log10 share_out - log10 share_take| over 5 broad bands (0.4-2, 2-4,
     4-6.5, 6.5-9, 9-12.5 Hz): where the motion's energy sits, not how much of it
  2. over-articulation: the (gap, width) cloud this face makes (the take's frames; both series smoothed by 1 frame,
     whitened by the take's covariance): the share of the render's frames outside the take's 95 % hull (grown x 1.1, so
     a re-encode of the take reads as the take) beyond the take's own share, and the KDE log-density deficit (nats; the
     take's own frames leave-one-out); per-phone peak shapes (open vowels: max gap, closures: min gap,
     round: min width, spread: max width; phones from the track) as percentile ranks in the take's own distribution:
     how extreme (0 = the take's median, 1 = its extreme) beyond the take's own in the same windows, and how much less
     varied (std of the ranks) than half the take's ("does every phone hit the class centre?"; the take does not sing
     these phones, so its own ranks are near random and only a far more uniform output is penalised)
  3. dynamics shape: 10-90 % rise / fall times of the gap's openings / closings (ms; pooled for the score, plus the
     size of their asymmetry), overshoot (share of a peak's swing given back within 3 frames, both sides), dwell at
     the extremes (ms within 10 % of the swing) -> log ratios / differences
  4. asymmetry: left - right corner height (IOD): its std (log ratio) and median shift
  5. jaw / lip split: the shares of the gap change carried by the upper lip, the lower lip and the chin (regression of
     their central differences on the gap's) -> absolute differences
  6. range_match: the gap's p5-p95 range, render / take (a compressed middle: vowels under-opened, closures shallow)
     -> |log ratio|
  7. rest_in_silence: in the aligner's pauses (no phone for >= 3 frames, 120 ms), the share of frames whose gap is below
     the take's own p10 (the mouth at rest), render vs take -> how much less often the render rests (mouthing in silence)
Score (one line per shot): S = 1.0 kin + 1.0 spec + 1.5 manifold + 0.75 dynshape + 0.5 asym + 0.5 jawlip + 0.25 peaks
+ 1.0 range + 1.0 rest (each term defined in compare(); 0 = the take's own statistics; documented in WEIGHTS). Calibration (tests): the take vs
itself = 0, the take vs its own time reversal stays natural (only the rise / fall swap and the phone windows move), a
step-target series reads unnatural.

  lsdub naturalness --video out.mp4 --take take.mp4 --take-trk trk.npz [--track vis.json] [--frames N]
      [--lm-cache out_lm.npz] [--take-lm take_lm.npz] [--json nat.json]
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

FPS = 25.0
GAP_PAIRS = [(13, 14), (82, 87), (312, 317)]
UP_IN, LO_IN, CHIN = [13, 82, 312], [14, 87, 317], [152, 175]
BAND = (0.4, 12.6)                                   # Hz, the spectral comparison band (Nyquist 12.5 at 25 fps)
SPEC_BANDS = [(0.4, 2.0), (2.0, 4.0), (4.0, 6.5), (6.5, 9.0), (9.0, 12.6)]   # broad bands of the spectral distance
WEIGHTS = dict(kin=1.0, spec=1.0, manifold=1.5, dynshape=0.75, asym=0.5, jawlip=0.5, peaks=0.25, range=1.0, rest=1.0)
# per-phone peak kinds (the lipsync checker's classes: open vowels, closures, round, spread)
OPEN_V, CLOSE_V, ROUND_V, SPREAD_V = ('aa', 'E', 'oh'), ('PP',), ('ou', 'oh', 'CH', 'RR'), ('ih', 'E', 'SS')


# ------------------------------------------------------------------------------------------------ measurement --
def fill_nan(L):
    """landmarks (n x K x D) with nan rows (no face) -> linearly interpolated (edge-held); found mask."""
    L = np.asarray(L, float).copy()
    ok = np.isfinite(L).all(axis=tuple(range(1, L.ndim)))
    if ok.all() or not ok.any(): return L, ok
    i = np.arange(len(L))
    flat = L.reshape(len(L), -1)
    for c in range(flat.shape[1]): flat[~ok, c] = np.interp(i[~ok], i[ok], flat[ok, c])
    return flat.reshape(L.shape), ok


def mouth_series(L, pose_smooth='sg7'):
    """IMAGE-mode landmarks of one video (n x 478 x 3, frame px, nan = no face) -> per-frame mouth series in the
    video's own skull frame, IOD units (dict of arrays) + found share + IOD (px). Up = positive for heights."""
    from ..core import mouth_morph as mm
    L, ok = fill_nan(L)
    n = len(L)
    ref = L[n // 2]
    pl = mm.RigidPlacement(ref, L, mm.MESH_ANCH, smooth=pose_smooth, lam=0.0)
    C = np.array([pl.to_lib(i, L[i]) for i in range(n)])[..., :2]
    iod = float(np.median(np.linalg.norm(C[:, 33] - C[:, 263], axis=1)))
    C = C / iod
    gap = np.mean([np.linalg.norm(C[:, a] - C[:, b], axis=1) for a, b in GAP_PAIRS], 0)
    width = 0.5 * (np.linalg.norm(C[:, 61] - C[:, 291], axis=1) + np.linalg.norm(C[:, 78] - C[:, 308], axis=1))
    cl, cr = -C[:, 61, 1], -C[:, 291, 1]
    return dict(gap=gap, width=width, up=-C[:, UP_IN, 1].mean(1), lo=C[:, LO_IN, 1].mean(1), chin=C[:, CHIN, 1].mean(1),
                corner=0.5 * (cl + cr), asym=cl - cr), float(ok.mean()), iod * float(np.median(pl.s))


def series_from_video(video, take_trk, n=None, cache=None, lmk=None):
    """IMAGE-mode landmarks of a video (the v9 gate's Landmarker on a face-box crop from the take's raw track; cached
    in `cache`) -> (mouth_series dict, found share, IOD px, landmarks)."""
    from .frame_gate import landmark_cache, read_frames
    S3 = np.load(take_trk)['S3']
    n = n or len(S3)
    frames = read_frames(video, n)
    n = min(n, len(frames), len(S3))
    L = landmark_cache(frames[:n], cache, lmk, S3[:n])
    s, found, iod = mouth_series(L)
    return s, found, iod, L


# ------------------------------------------------------------------------------------------------- statistics --
def kin(x, fps=FPS):
    """|velocity| p50 / p95, |acceleration| p50 / p95, |jerk| p95 of a series (units / s^k)."""
    x = np.asarray(x, float)
    v = np.abs(np.diff(x)) * fps
    a = np.abs(np.diff(x, 2)) * fps ** 2
    j = np.abs(np.diff(x, 3)) * fps ** 3
    q = lambda y, p: float(np.percentile(y, p)) if len(y) else 0.0
    return dict(v50=q(v, 50), v95=q(v, 95), a50=q(a, 50), a95=q(a, 95), j95=q(j, 95))


def psd(x, fps=FPS, nper=48):
    """Welch power spectrum (Hann, half overlap, linear detrend) averaged over the series AND its time reversal (the
    segment grid then covers both ends alike, so a time-reversed copy has exactly the same spectrum), normalised to its
    power inside BAND -> (f, P) over the band's bins, and the power shares of the broad bands SPEC_BANDS."""
    from scipy.signal import welch
    x = np.asarray(x, float)
    nper = int(min(nper, len(x)))
    f, P1 = welch(x, fs=fps, nperseg=nper, noverlap=nper // 2, detrend='linear')
    _, P2 = welch(x[::-1].copy(), fs=fps, nperseg=nper, noverlap=nper // 2, detrend='linear')
    P = 0.5 * (P1 + P2)
    b = (f >= BAND[0]) & (f <= BAND[1])
    P = np.maximum(P[b], 1e-12)
    f = f[b]
    P = P / P.sum()
    sh = np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in SPEC_BANDS])
    return f, P, np.maximum(sh, 1e-6)


def extrema(x, prom):
    """alternating local extrema of x with prominence >= prom: list of (frame, +1 max / -1 min)."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(x, prominence=prom)
    tr, _ = find_peaks(-x, prominence=prom)
    ev = sorted([(int(i), 1) for i in pk] + [(int(i), -1) for i in tr])
    out = []
    for e in ev:                                     # keep alternation (two maxima in a row: the higher one)
        if out and out[-1][1] == e[1]:
            if (e[1] > 0 and x[e[0]] > x[out[-1][0]]) or (e[1] < 0 and x[e[0]] < x[out[-1][0]]): out[-1] = e
        else: out.append(e)
    return out


def _cross(x, i0, i1, level, rising):
    """first sub-frame crossing of `level` between frames i0 .. i1 (linear interpolation)."""
    for k in range(i0, i1):
        a, b = x[k], x[k + 1]
        if (rising and a < level <= b) or (not rising and a > level >= b):
            return k + (level - a) / (b - a) if b != a else float(k)
    return float(i1)


def transitions(x, fps=FPS, prom_frac=0.2, sigma=0.7):
    """the openings / closings of a series: (smoothed series, alternating extrema, prominence, list of (amplitude,
    10-90 % time in ms, rising)). Smoothed by sigma frames first so the instrument's single-frame jitter does not split
    events; amplitude threshold prom_frac x (p95 - p5)."""
    x = gaussian_filter1d(np.asarray(x, float), sigma, mode='nearest') if sigma > 0 else np.asarray(x, float)
    rng = float(np.percentile(x, 95) - np.percentile(x, 5))
    prom = max(1e-6, prom_frac * rng)
    ev = extrema(x, prom)
    tr = []
    for (i0, s0), (i1, s1) in zip(ev, ev[1:]):
        A = x[i1] - x[i0]
        if abs(A) < prom: continue
        up = A > 0
        t10 = _cross(x, i0, i1, x[i0] + 0.1 * A, up)
        t90 = _cross(x, i0, i1, x[i0] + 0.9 * A, up)
        tr.append((float(abs(A)), max(0.0, t90 - t10) * 1000.0 / fps, bool(up)))
    return x, ev, prom, tr


def dynamics_shape(x, fps=FPS, prom_frac=0.2):
    """the shape of the motion around openings / closings of a series (transitions(): smoothed 0.7 frame, amplitude
    threshold prom_frac x (p95 - p5)):
      rise_ms / fall_ms   10-90 % times of the rises / falls (medians); trans_ms = all transitions pooled
      overshoot           at each prominent maximum, the share of its swing given back within 3 frames, both sides
                          averaged (an overshoot before a hold, or a hold before a sharp turn; reversal-symmetric)
      dwell_ms            time within 10 % of the swing around each prominent extremum (the hold at a target)."""
    x, ev, prom, tr = transitions(x, fps, prom_frac)
    n = len(x)
    rise = [d for A, d, up in tr if up]
    fall = [d for A, d, up in tr if not up]
    ovs, dwell = [], []
    for k, (i, s) in enumerate(ev):
        A = max(abs(x[i] - x[ev[k - 1][0]]) if k > 0 else 0.0, abs(x[i] - x[ev[k + 1][0]]) if k + 1 < len(ev) else 0.0)
        if A < prom: continue
        j0 = i
        while j0 - 1 >= 0 and abs(x[j0 - 1] - x[i]) <= 0.1 * A: j0 -= 1
        j1 = i
        while j1 + 1 < n and abs(x[j1 + 1] - x[i]) <= 0.1 * A: j1 += 1
        dwell.append((j1 - j0 + 1) * 1000.0 / fps)
        if s > 0 and 3 <= i < n - 3:
            ovs.append(0.5 * (max(0.0, float(x[i] - x[i + 1:i + 4].mean())) + max(0.0, float(x[i] - x[i - 3:i].mean()))) / A)
    med = lambda v: float(np.median(v)) if v else float('nan')
    return dict(rise_ms=med(rise), fall_ms=med(fall), trans_ms=med(rise + fall), overshoot=med(ovs), dwell_ms=med(dwell),
                n_rise=len(rise), n_fall=len(fall))


def shares(s):
    """jaw / lip split: regression slopes of the upper lip's raise, the lower lip's and the chin's drop on the gap
    (central differences of the series smoothed by 1 frame): the share of a gap change each carries."""
    d = lambda k: np.gradient(gaussian_filter1d(np.asarray(s[k], float), 1.0, mode='nearest'))
    g = d('gap')
    vg = float(np.dot(g - g.mean(), g - g.mean()))
    if vg <= 1e-12: return dict(up=float('nan'), lo=float('nan'), chin=float('nan'))
    sl = lambda y: float(np.dot(y - y.mean(), g - g.mean()) / vg)
    return dict(up=sl(d('up')), lo=sl(d('lo')), chin=sl(d('chin')))


def phone_runs(track_frames, n):
    """phone segments from a viseme track's per-frame `phone` labels: list of (first frame, last frame, viseme)."""
    from ..core.viseme_track import ARPA
    lab = [(track_frames[min(i, len(track_frames) - 1)].get('phone') or '') for i in range(n)]
    out = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]: j += 1
        if lab[i]:
            v = ARPA.get(lab[i]) or 'aa'
            out.append((i, j, v))
        i = j + 1
    return out


def pause_frames(track_frames, n, min_frames=3):
    """the aligner's pauses: runs of at least min_frames frames with no phone label (120 ms at 25 fps) -> bool n."""
    lab = [bool(track_frames[min(i, len(track_frames) - 1)].get('phone')) for i in range(n)]
    out = np.zeros(n, bool)
    i = 0
    while i < n:
        if lab[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not lab[j + 1]: j += 1
        if j - i + 1 >= min_frames: out[i:j + 1] = True
        i = j + 1
    return out


def phone_peaks(s, runs):
    """per phone (+-1 frame): the class's defining extreme -> list of (kind, feature, value)."""
    out = []
    n = len(s['gap'])
    for i0, i1, v in runs:
        a, b = max(0, i0 - 1), min(n, i1 + 2)
        if v in OPEN_V: out.append(('open', 'gap', float(np.max(s['gap'][a:b]))))
        if v in CLOSE_V: out.append(('close', 'gap', float(np.min(s['gap'][a:b]))))
        if v in ROUND_V: out.append(('round', 'width', float(np.min(s['width'][a:b]))))
        if v in SPREAD_V: out.append(('spread', 'width', float(np.max(s['width'][a:b]))))
    return out


def series_stats(s, fps=FPS, runs=None, pauses=None):
    """all of one video's statistics (the per-series part; the cloud comparisons need both videos: compare())."""
    st = dict(n=len(s['gap']), fps=fps, series={k: np.asarray(v, float) for k, v in s.items()}, pauses=pauses)
    g = np.asarray(s['gap'], float)
    st['gap_range'] = float(np.percentile(g, 95) - np.percentile(g, 5))
    st['gap_p10'] = float(np.percentile(g, 10))
    for k in ('gap', 'width'):
        st[k] = kin(s[k], fps)
        f, P, sh = psd(s[k], fps)
        st[k]['psd_f'] = f
        st[k]['psd'] = P
        st[k]['bands'] = sh
        st[k]['centroid_hz'] = float((f * P).sum())
    st['dyn'] = dynamics_shape(s['gap'], fps)
    st['dyn_width'] = dynamics_shape(s['width'], fps)
    a = np.asarray(s['asym'], float)
    st['asym'] = dict(std=float(np.std(a)), median=float(np.median(a)))
    st['corner_std'] = float(np.std(s['corner']))
    st['shares'] = shares(s)
    st['peaks'] = phone_peaks(s, runs) if runs else []
    return st


# ------------------------------------------------------------------------------------------------- comparison --
def _lr(a, b, eps=1e-9):
    """|log ratio| (nan-safe: a missing statistic contributes 0)."""
    if not (np.isfinite(a) and np.isfinite(b)) or a <= eps or b <= eps: return 0.0
    return float(abs(np.log(a / b)))


def cloud(ref_s, out_s, q=95.0, excl=2, sigma=1.0, grow=1.1):
    """the over-articulation numbers of out vs the reference cloud of (gap, width) shapes (both series smoothed by
    sigma frames first, whitened by the reference's covariance): the share outside the ref's q % hull (the hull of
    its Mahalanobis-central q % of frames, grown x grow about its centre: with sigma 1 / grow 1.1 a re-encode of the
    take reads as the take itself, i.e. the instrument's own jitter does not count), the mean KDE log-density (Scott
    bandwidth); the ref's own numbers on itself (KDE leave-one-out, +-excl frames excluded) for the excess."""
    from scipy.spatial import Delaunay
    sm = lambda x: gaussian_filter1d(np.asarray(x, float), sigma, mode='nearest') if sigma > 0 else np.asarray(x, float)
    R = np.stack([sm(ref_s['gap']), sm(ref_s['width'])], 1)
    O = np.stack([sm(out_s['gap']), sm(out_s['width'])], 1)
    mu = R.mean(0)
    cov = np.cov(R.T) + 1e-9 * np.eye(2)
    Wm = np.linalg.cholesky(np.linalg.inv(cov))
    Rz, Oz = (R - mu) @ Wm, (O - mu) @ Wm
    d2 = (Rz ** 2).sum(1)
    core = Rz[d2 <= np.percentile(d2, q)]
    cc = core.mean(0)
    core = cc + (core - cc) * grow
    try:
        hull = Delaunay(core)
        out_frac = float((hull.find_simplex(Oz) < 0).mean())
        ref_frac = float((hull.find_simplex(Rz) < 0).mean())
    except Exception:
        out_frac = ref_frac = float('nan')
    h = len(R) ** (-1.0 / 6.0)                          # Scott's factor in 2-D (whitened: unit covariance)
    def logp(P, loo):
        D = ((P[:, None, :] - Rz[None]) ** 2).sum(-1) / (2 * h * h)
        K = np.exp(-D)
        if loo:
            idx = np.arange(len(P))
            K[np.abs(idx[:, None] - idx[None]) <= excl] = 0.0
        return np.log(np.maximum(K.sum(1) / max(1, K.shape[1]) / (2 * np.pi * h * h), 1e-12))
    return dict(hull_out=out_frac, hull_out_ref=ref_frac, kde=float(logp(Oz, False).mean()), kde_ref=float(logp(Rz, True).mean()),
                mahal_p95=float(np.percentile(np.sqrt((Oz ** 2).sum(1)), 95)), mahal_p95_ref=float(np.percentile(np.sqrt(d2), 95)))


def peak_numbers(peaks, ref_series):
    """percentile ranks of per-phone peaks in the reference's own distribution of that feature: extremeness (mean of
    |2 rank - 1|) and spread (mean over kinds of the ranks' std)."""
    if not peaks: return dict(extreme=float('nan'), spread=float('nan'), n=0)
    rk = {}
    for kind, feat, v in peaks:
        ref = np.sort(np.asarray(ref_series[feat], float))
        r = np.searchsorted(ref, v) / max(1, len(ref))
        rk.setdefault(kind, []).append(r)
    ext = float(np.mean([abs(2 * r - 1) for v in rk.values() for r in v]))
    sp = [float(np.std(v)) for v in rk.values() if len(v) >= 2]
    return dict(extreme=ext, spread=float(np.mean(sp)) if sp else float('nan'), n=sum(len(v) for v in rk.values()))


def compare(out, ref):
    """the render's statistics vs the take's (series_stats of each) -> components, terms, score. 0 = the take."""
    T = {}
    kin_t = []
    for k in ('gap', 'width'):
        o, r = out[k], ref[k]
        kin_t.append(np.mean([_lr(o['v95'], r['v95']), _lr(o['a95'], r['a95']), _lr(o['j95'], r['j95']), _lr(o['v50'], r['v50'])]))
    T['kin'] = float(np.mean(kin_t))
    T['spec'] = float(np.mean([np.mean(np.abs(np.log10(out[k]['bands']) - np.log10(ref[k]['bands']))) for k in ('gap', 'width')]))
    cl = cloud(ref['series'], out['series'])
    T['manifold'] = 2.0 * max(0.0, (cl['hull_out'] - cl['hull_out_ref']) if np.isfinite(cl['hull_out']) else 0.0) + \
        0.5 * max(0.0, cl['kde_ref'] - cl['kde'])
    do, dr = out['dyn'], ref['dyn']
    ovs = abs(do['overshoot'] - dr['overshoot']) if np.isfinite(do['overshoot']) and np.isfinite(dr['overshoot']) else 0.0
    ra = lambda d: abs(np.log(d['rise_ms'] / d['fall_ms'])) if np.isfinite(d['rise_ms']) and np.isfinite(d['fall_ms']) and min(d['rise_ms'], d['fall_ms']) > 0 else float('nan')
    asy = abs(ra(do) - ra(dr)) if np.isfinite(ra(do)) and np.isfinite(ra(dr)) else 0.0
    T['dynshape'] = float(np.mean([_lr(do['trans_ms'], dr['trans_ms']), _lr(do['dwell_ms'], dr['dwell_ms'])]) + ovs + 0.5 * asy)
    T['asym'] = _lr(out['asym']['std'], ref['asym']['std']) + 10.0 * abs(out['asym']['median'] - ref['asym']['median'])
    so, sr = out['shares'], ref['shares']
    T['jawlip'] = float(sum(abs(so[k] - sr[k]) for k in ('up', 'chin') if np.isfinite(so[k]) and np.isfinite(sr[k])))
    po, pr = peak_numbers(out['peaks'], ref['series']), peak_numbers(ref['peaks'], ref['series'])
    pe = max(0.0, po['extreme'] - pr['extreme']) if np.isfinite(po['extreme']) and np.isfinite(pr['extreme']) else 0.0
    # "does every phone hit the class centre?": peaks more extreme than the take's own in the same windows, and peak
    # ranks less varied than HALF the take's spread (the take does not sing these phones, so its ranks are close to
    # random; a synced mouth is more consistent than that, a robotic one far more)
    sp = max(0.0, float(np.log(0.5 * pr['spread'] / po['spread']))) if np.isfinite(po['spread']) and np.isfinite(pr['spread']) and po['spread'] > 0 else 0.0
    T['peaks'] = pe + sp
    # the founder's two: the range of openings (compressed middle) and resting in the pauses (mouthing in silence)
    T['range'] = _lr(out['gap_range'], ref['gap_range'])
    pz = ref.get('pauses')
    rest_o = rest_r = None
    if pz is not None and np.asarray(pz, bool).any():
        pz = np.asarray(pz, bool)[:min(out['n'], ref['n'])]
        thr = ref['gap_p10']
        rest_o = float(np.mean(out['series']['gap'][:len(pz)][pz] < thr))
        rest_r = float(np.mean(ref['series']['gap'][:len(pz)][pz] < thr))
    T['rest'] = max(0.0, rest_r - rest_o) if rest_o is not None else 0.0
    score = float(sum(WEIGHTS[k] * T[k] for k in WEIGHTS))
    comp = dict(gap={k: (round(out['gap'][k], 4), round(ref['gap'][k], 4)) for k in ('v50', 'v95', 'a95', 'j95', 'centroid_hz')},
                width={k: (round(out['width'][k], 4), round(ref['width'][k], 4)) for k in ('v50', 'v95', 'a95', 'j95', 'centroid_hz')},
                cloud={k: round(v, 4) for k, v in cl.items()},
                dyn={k: (_r(do[k]), _r(dr[k])) for k in do}, dyn_width={k: (_r(out['dyn_width'][k]), _r(ref['dyn_width'][k])) for k in do},
                asym={k: (round(out['asym'][k], 4), round(ref['asym'][k], 4)) for k in ('std', 'median')},
                shares={k: (_r(so[k]), _r(sr[k])) for k in so}, peaks=dict(out={k: _r(v) for k, v in po.items()}, take={k: _r(v) for k, v in pr.items()}),
                range_match=_r(out['gap_range'] / max(1e-9, ref['gap_range'])), gap_range=(_r(out['gap_range']), _r(ref['gap_range'])),
                rest_in_silence=(_r(rest_o) if rest_o is not None else None, _r(rest_r) if rest_r is not None else None),
                n_pause_frames=int(np.asarray(pz, bool).sum()) if pz is not None else 0)
    return dict(score=round(score, 4), terms={k: round(v, 4) for k, v in T.items()}, weights=WEIGHTS, components=comp)


def _r(v, d=4):
    return round(float(v), d) if v is not None and np.isfinite(v) else None


def line(name, res):
    """the one-line score."""
    t = res['terms']
    c = res['components']['cloud']
    rs = res['components'].get('rest_in_silence') or (None, None)
    rtxt = f"{rs[0]:.2f} vs take {rs[1]:.2f}" if rs[0] is not None else '-'
    return (f"{name}: naturalness {res['score']:.3f} (kin {t['kin']:.2f} spec {t['spec']:.2f} manifold {t['manifold']:.2f} "
            f"[out of hull {c['hull_out']:.2f} vs take {c['hull_out_ref']:.2f}] dyn {t['dynshape']:.2f} asym {t['asym']:.2f} "
            f"jaw/lip {t['jawlip']:.2f} peaks {t['peaks']:.2f} range {t['range']:.2f} [x{res['components']['range_match']:.2f}] "
            f"rest {t['rest']:.2f} [{rtxt}])")


# --------------------------------------------------------------------------------------------- synthetic tracks --
def step_version(s, runs, keep_noise=True):
    """a STEP-TARGET copy of a series dict: every phone run (and every gap between runs) held at its mean, switched
    in one frame (the per-frame step-target motion the ruler must call unnatural); keep_noise adds back the series'
    own single-frame residual (x - gaussian 1 frame), so the instrument's noise floor is the same as the original's."""
    n = len(s['gap'])
    cuts = sorted({0, n} | {i0 for i0, _, _ in runs} | {i1 + 1 for _, i1, _ in runs})
    out = {}
    for k, x in s.items():
        x = np.asarray(x, float)
        y = np.empty(n)
        for a, b in zip(cuts, cuts[1:]):
            if b > a: y[a:b] = x[a:b].mean()
        if keep_noise: y = y + (x - gaussian_filter1d(x, 1.0, mode='nearest'))
        out[k] = y
    return out


def reversed_series(s):
    return {k: np.asarray(v, float)[::-1].copy() for k, v in s.items()}


# ------------------------------------------------------------------------------------------------------- CLI --
def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub naturalness', description=__doc__.split('\n')[0])
    ap.add_argument('--video', required=True, help='the render')
    ap.add_argument('--take', required=True)
    ap.add_argument('--take-trk', required=True, help='npz with S3 = the take\'s raw MediaPipe track (face boxes for both videos)')
    ap.add_argument('--track', help='viseme track JSON (per-phone peaks)')
    ap.add_argument('--frames', type=int)
    ap.add_argument('--lm-cache', help='cache of the render\'s IMAGE-mode landmarks')
    ap.add_argument('--take-lm', help='cache of the take\'s')
    ap.add_argument('--fps', type=float, default=FPS)
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    from .frame_gate import Landmarker
    lmk = Landmarker()
    st, fo, _, _ = series_from_video(a.video, a.take_trk, a.frames, a.lm_cache, lmk)
    tt, ft, _, _ = series_from_video(a.take, a.take_trk, a.frames, a.take_lm, lmk)
    n = min(len(st['gap']), len(tt['gap']))
    st = {k: v[:n] for k, v in st.items()}
    tt = {k: v[:n] for k, v in tt.items()}
    trf = json.load(open(a.track))['frames'] if a.track else None
    runs = phone_runs(trf, n) if trf else None
    pz = pause_frames(trf, n) if trf else None
    so, sr = series_stats(st, a.fps, runs, pz), series_stats(tt, a.fps, runs, pz)
    res = compare(so, sr)
    res.update(video=a.video, take=a.take, found=round(fo, 3), found_take=round(ft, 3), n=n)
    rev = compare(series_stats(reversed_series(tt), a.fps, runs, pz), sr)
    res['take_reversed_score'] = rev['score']
    print(line(os.path.basename(os.path.dirname(os.path.abspath(a.video))) or a.video, res))
    if a.json: json.dump(res, open(a.json, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
