#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""tooth_scale.py: the TOOTH-SCALE ruler (v13). CPU.

Founder on v12: the rendered teeth look about 1/5 too wide relative to the head on the close-ups. Teeth are bone: their
size is fixed by the face, so the ruler measures them against the face's own landmarks, never against the mouth opening.

  span      per frame, the VISIBLE TOOTH SPAN of a video, read on the TAKE's raw MediaPipe track (the same landmarks for the
            take and for the render): in the axis-aligned mouth box x in [min(61, 291) - 0.1 IPD, max(61, 291) + 0.1 IPD],
            y in [0 - 0.03 IPD, 17 + 0.03 IPD] (floor / ceil, clipped to the frame), the columns holding >= 2 tooth-key
            pixels; span = (p98 - p2 of their x) / IPD, 0 when there are none. IPD = |473 - 468| (the iris centres, per
            frame; a frame whose irises collapsed, IPD / OCD < 0.3 (--ipd-guard), takes the median ratio x its OCD); every
            span is also given in lsdub's own IOD unit, OCD = |263 - 33| (the outer eye corners, ~1.45 IPD).
            Two tooth keys (OpenCV 8-bit Lab): `tl` = teeth_layer.teeth_key(lab, skin_L), skin_L = the median over the
            take's frames of teeth_layer.skin_ring_L (finite values); `lab` = L > 150, a - 128 < 10, b - 128 < 22.
  summary   per key, over the frames where BOTH spans exceed 0.12 IPD: take p50 / p90, render p50 / p90, the median of the
            per-frame ratio render / take, n; and p50 / p90 over all frames.
  teeth     the VISIBLE UPPER TEETH at the render's widest frame (argmax of its `tl` span; also for `lab`, and the take's
            own widest frame as the reference): the upper row tracked in the mouth box (dim and lip-tinted teeth
            included), the interdental minima of the L profile along a band just below its top (count_teeth); teeth =
            minima more than 0.03 OCD inside the row's ends + 1. --stills writes a by-eye check of each count.
  plate     with --tex, the arch plates' interdental lines (plate_teeth): the midline, the two central incisor widths and
            the median tooth spacing, in IOD and (x OCD / IPD of the track) in IPD.
  accept    per key PASS iff render span p90 <= 1.05 x take span p90 and the ratio median <= 1.05; the upper central
            width in [0.12, 0.15] IPD; visible upper teeth at the render's widest frame <= 8.

  lsdub tooth-scale TAKE.mp4 RENDER.mp4 --track TRK.npz|json [--frames N] [--tex TEX.npz] [--json out.json] [--md out.md]
      [--label L] [--stills DIR] [--ipd-guard 0.3]
"""
import argparse, json, os, sys
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

THR = 0.12              # a frame enters the summary when both spans exceed this (IPD)
KEYS = ('tl', 'lab')
ACCEPT = dict(span=1.05, ratio=1.05, central=(0.12, 0.15), teeth=8)


# --------------------------------------------------------------------------------------------------------- inputs --
def read_frames(path, n=None):
    """BGR frames of a video (the first n)."""
    cap = cv2.VideoCapture(path)
    F = []
    while n is None or len(F) < n:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    cap.release()
    return F


def load_track(path):
    """(n, 478, 3) float landmarks in frame px: an npz (key S3, else its only 3-D array) or a JSON list of per-frame
    478 x 2|3 lists (or a dict with S3). 2-D landmarks get z = 0."""
    if str(path).endswith('.json'):
        J = json.load(open(path))
        J = J['S3'] if isinstance(J, dict) else J
        P = np.asarray(J, float)
    else:
        Z = np.load(path, allow_pickle=False)
        if 'S3' in Z.files: P = np.asarray(Z['S3'], float)
        else:
            arr = [k for k in Z.files if np.asarray(Z[k]).ndim == 3]
            if len(arr) != 1: raise ValueError(f'{path}: no S3 key and {len(arr)} 3-D arrays')
            P = np.asarray(Z[arr[0]], float)
    if P.ndim != 3 or P.shape[1] < 478: raise ValueError(f'{path}: landmarks of shape {P.shape}, want (n, 478, 2|3)')
    if P.shape[2] == 2: P = np.dstack([P, np.zeros(P.shape[:2])])
    return P[:, :478, :3]


def units(P):
    """per frame: IPD = |473 - 468| (iris centres) and OCD = |263 - 33| (outer eye corners, lsdub's IOD), frame px."""
    P = np.asarray(P, float)
    return np.linalg.norm(P[:, 473, :2] - P[:, 468, :2], axis=1), np.linalg.norm(P[:, 263, :2] - P[:, 33, :2], axis=1)


def guard_ipd(ipd, ocd, lo=0.3, hi=1.0):
    """frames whose iris landmarks collapsed (IPD / OCD < lo or > hi: no face found, the track's fill-in) take IPD = the
    median IPD / OCD of the other frames x their OCD. -> (ipd, indices fixed). lo = 0 turns it off. Blinks (IPD / OCD ~0.4-0.55)
    are kept by default, as the reference ruler kept them; pass lo = 0.55 to guard them too."""
    ipd, ocd = np.asarray(ipd, float).copy(), np.asarray(ocd, float)
    r = ipd / np.maximum(ocd, 1e-6)
    bad = ~np.isfinite(r) | (r < lo) | (r > hi) if lo > 0 else ~np.isfinite(r)
    if bad.any():
        rm = float(np.median(r[~bad])) if (~bad).any() else 0.69
        ipd[bad] = rm * ocd[bad]
    return ipd, np.nonzero(bad)[0].tolist()


def take_skin_L(frames, P):
    """the take's skin light at the mouth: the median over its frames of teeth_layer.skin_ring_L (finite values)."""
    from ..core import teeth_layer as tl
    s = np.array([tl.skin_ring_L(frames[i], P[i]) for i in range(min(len(frames), len(P)))], float)
    s = s[np.isfinite(s)]
    return float(np.median(s)) if len(s) else float('nan')


# ----------------------------------------------------------------------------------------------------------- span --
def mouth_box(p, ipd, shape):
    """(x0, y0, x1, y1): the axis-aligned mouth box of landmarks p (frame px), floor / ceil, clipped to the frame."""
    x0 = int(np.floor(min(p[61, 0], p[291, 0]) - 0.1 * ipd))
    x1 = int(np.ceil(max(p[61, 0], p[291, 0]) + 0.1 * ipd))
    y0 = int(np.floor(p[0, 1] - 0.03 * ipd))
    y1 = int(np.ceil(p[17, 1] + 0.03 * ipd))
    return max(0, x0), max(0, y0), min(shape[1], x1), min(shape[0], y1)


def tooth_key(lab, key='tl', skin_L=None):
    """bool tooth-key mask of an OpenCV 8-bit Lab image: 'tl' = teeth_layer.teeth_key(lab, skin_L); 'lab' = the fixed
    L > 150, a - 128 < 10, b - 128 < 22 (integer arithmetic)."""
    if key == 'tl':
        from ..core import teeth_layer as tl
        return tl.teeth_key(lab, skin_L)
    if key == 'lab':
        return (lab[..., 0] > 150) & (lab[..., 1].astype(int) - 128 < 10) & (lab[..., 2].astype(int) - 128 < 22)
    raise ValueError(f'unknown key {key!r}')


def span_px(k):
    """the visible tooth span (px) of a key mask: p98 - p2 of the x of the columns holding >= 2 key pixels; 0 if none."""
    cols = np.nonzero(np.asarray(k).sum(0) >= 2)[0]
    return float(np.percentile(cols, 98) - np.percentile(cols, 2)) if len(cols) else 0.0


def box_key(img, p, ipd, key='tl', skin_L=None):
    """(key mask, Lab crop, box) of a frame's mouth box; (None, None, box) for an empty box."""
    x0, y0, x1, y1 = mouth_box(p, ipd, img.shape)
    if x1 <= x0 or y1 <= y0: return None, None, (x0, y0, x1, y1)
    lab = cv2.cvtColor(np.ascontiguousarray(img[y0:y1, x0:x1]), cv2.COLOR_BGR2LAB)
    return tooth_key(lab, key, skin_L), lab, (x0, y0, x1, y1)


def spans(frames, P, key='tl', skin_L=None, ipd=None):
    """per-frame visible tooth span (frame px) of a video on landmarks P (ipd: the per-frame unit, default units(P))."""
    ipd = units(P)[0] if ipd is None else np.asarray(ipd, float)
    out = np.zeros(min(len(frames), len(P)))
    for i in range(len(out)):
        k, _, _ = box_key(frames[i], np.asarray(P[i], float)[:, :2], ipd[i], key, skin_L)
        out[i] = span_px(k) if k is not None else 0.0
    return out


def _r(x, d=5):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def summarize(st, sr, ipd, ocd=None, thr=THR):
    """the span summary of a take / render pair of per-frame spans (px): over the frames where BOTH exceed thr IPD
    (the frame set is fixed in IPD and reused for the OCD numbers), take / render p50 / p90, the median per-frame
    ratio render / take, n; and take / render p50 / p90 over all frames. -> {'ipd': {...}, 'ocd': {...}, 'ok': bool[]}"""
    st, sr = np.asarray(st, float), np.asarray(sr, float)
    m = min(len(st), len(sr))
    st, sr = st[:m], sr[:m]
    ipd = np.asarray(ipd, float)[:m]
    ok = (st / ipd > thr) & (sr / ipd > thr)
    res = dict(ok=ok)
    for nm, u in (('ipd', ipd), ('ocd', None if ocd is None else np.asarray(ocd, float)[:m])):
        if u is None: continue
        a, b = st / u, sr / u
        pc = lambda x, q: float(np.percentile(x, q)) if len(x) else None
        res[nm] = dict(n=int(ok.sum()), take_p50=_r(pc(a[ok], 50)), take_p90=_r(pc(a[ok], 90)), render_p50=_r(pc(b[ok], 50)),
                       render_p90=_r(pc(b[ok], 90)), ratio_med=_r(float(np.median(b[ok] / a[ok])) if ok.any() else None),
                       all=dict(take_p50=_r(pc(a, 50)), take_p90=_r(pc(a, 90)), render_p50=_r(pc(b, 50)), render_p90=_r(pc(b, 90))))
    return res


# ---------------------------------------------------------------------------------------------- visible upper teeth --
def _runs(col, max_gap=2):
    """the runs of a bool column, gaps <= max_gap px bridged: [(first, last)]."""
    r = np.nonzero(col)[0]
    if not len(r): return []
    cut = np.nonzero(np.diff(r) > max_gap + 1)[0]
    return [(int(r[s]), int(r[e])) for s, e in zip(np.r_[0, cut + 1], np.r_[cut, len(r) - 1])]


def upper_row(k, max_gap=2, min_len=2):
    """per column of a key mask the topmost run of key pixels (gaps <= max_gap px bridged; a run shorter than min_len px
    is skipped as a speck): (top, bottom) rows, -1 where the column has none."""
    k = np.asarray(k, bool)
    H, W = k.shape
    top = np.full(W, -1)
    bot = np.full(W, -1)
    for c in range(W):
        for t, b in _runs(k[:, c], max_gap):
            if b - t + 1 >= min_len:
                top[c], bot[c] = t, b
                break
    return top, bot


def track_row(kk, lab, ocd, skin_L, da=12, db=14, l_floor=0.3, min_w=0.1, min_h=0.012):
    """the UPPER tooth row of a mouth box: M = key | (a <= a_key + da, b <= b_key + db, L >= l_floor x skin_L; a_key, b_key =
    the key pixels' medians: the teeth however dim or lip-tinted, never the lips, the skin or the dark cavity); of M's
    8-connected components that hold a key pixel, are >= min_w OCD wide with a median column run >= min_h OCD (no lip-edge
    hairlines) and do not touch the box's top / left / right edge (the skin beyond the lips), the HIGHEST (smallest median
    top) is the upper row (a clenched bite is one component: its topmost runs are the upper teeth); per column its topmost
    run (gaps <= 2 px bridged). -> (top, bottom) rows per column (-1 outside the row), M."""
    from scipy.ndimage import label
    L, A, B = (lab[..., j].astype(np.int32) for j in range(3))
    H, W = kk.shape
    ak, bk = (np.median(A[kk]), np.median(B[kk])) if kk.sum() >= 10 else (134.0, 142.0)
    sk = float(skin_L) if skin_L is not None and np.isfinite(skin_L) else 150.0
    M = kk | ((A <= ak + da) & (B <= bk + db) & (L >= l_floor * sk))
    top = np.full(W, -1)
    bot = np.full(W, -1)
    lb, n = label(M, structure=np.ones((3, 3)))
    if n == 0: return top, bot, M
    edge = set(np.unique(lb[0])) | set(np.unique(lb[:, 0])) | set(np.unique(lb[:, -1]))
    kc = np.bincount(lb[kk], minlength=n + 1)
    best = None
    for q in range(1, n + 1):
        if q in edge or kc[q] == 0: continue
        C = lb == q
        t, b = upper_row(C, 2, 1)
        v = np.nonzero(t >= 0)[0]
        if len(v) < max(3, min_w * ocd) or np.median(b[v] - t[v] + 1) < max(2, min_h * ocd): continue
        mt = float(np.median(t[v]))
        if best is None or mt < best[0]: best = (mt, t, b)
    if best is not None: top, bot = best[1], best[2]
    return top, bot, M


def count_teeth(img, p, ipd, ocd, key='tl', skin_L=None, band=(0.01, 0.045), sigma=0.006, hp_sigma=0.05, dist=0.03, prom=2.5,
                rel=0.15, margin=0.03):
    """the VISIBLE UPPER TEETH of a frame, on the take's landmarks p (frame px). In the mouth box, the upper row (track_row);
    along it the L profile of the band [row top + band[0], row top + band[1]] OCD (the row's own pixels; the top line median-
    filtered over 0.02 OCD), smoothed (sigma OCD), high-passed (minus a gaussian of hp_sigma OCD); the interdental minima =
    find_peaks(-hp, distance = dist OCD, prominence >= max(prom, rel x std)) more than `margin` OCD inside the row's ends (a
    corner sliver narrower than that is no tooth); teeth = minima + 1 (0 without a row). -> dict(n, minima_x (frame px), row
    (x0, x1 frame px), row_ocd, row_ipd, box, top / band lines for drawing)."""
    from scipy.signal import find_peaks
    from scipy.ndimage import median_filter
    kk, lab, box = box_key(img, p, ipd, key, skin_L)
    x0b, y0b = box[0], box[1]
    res = dict(n=0, minima_x=[], row=None, row_ocd=0.0, row_ipd=0.0, box=list(box), key=key)
    if kk is None or kk.sum() < 10: return res
    L = lab[..., 0].astype(np.float32)
    H, W = kk.shape
    top, bot, M = track_row(kk, lab, ocd, skin_L)
    v = np.nonzero(top >= 0)[0]
    if len(v) < 5: return res
    cols = np.arange(v.min(), v.max() + 1)
    tp = np.interp(cols, v, median_filter(top[v].astype(float), size=max(3, int(0.02 * ocd) | 1), mode='nearest'))
    bt = np.interp(cols, v, bot[v].astype(float))
    prof = np.full(len(cols), np.nan)
    for j, c in enumerate(cols):
        r0 = max(0, int(round(tp[j] + band[0] * ocd)))
        r1 = min(H, max(r0 + 1, min(int(round(tp[j] + band[1] * ocd)), int(round(bt[j]))) + 1))
        m = M[r0:r1, c]
        if m.sum() >= 2: prof[j] = L[r0:r1, c][m].mean()
    ok = np.isfinite(prof)
    if ok.sum() < 5: return res
    prof = np.interp(np.arange(len(prof)), np.nonzero(ok)[0], prof[ok])
    pr = gaussian_filter1d(prof, max(0.5, sigma * ocd))
    hp = pr - gaussian_filter1d(pr, hp_sigma * ocd, mode='nearest')
    pk, _ = find_peaks(-hp, distance=max(2, int(round(dist * ocd))), prominence=max(prom, rel * float(hp.std())))
    mg = margin * ocd
    pk = pk[(pk > mg) & (pk < len(cols) - 1 - mg)]
    res.update(n=int(len(pk) + 1), minima_x=[round(float(x0b + cols[k] + 0.5), 1) for k in pk],
               row=[float(x0b + cols[0]), float(x0b + cols[-1] + 1)], row_ocd=round(len(cols) / ocd, 4), row_ipd=round(len(cols) / ipd, 4),
               _top=(x0b + cols + 0.5, y0b + tp), _band=(y0b + tp + band[0] * ocd, y0b + tp + band[1] * ocd))
    return res


def teeth_still(img, res, ocd, title, width=None):
    """a by-eye check of count_teeth: the mouth box at 3-4x, clean | with the counted minima as thin vertical ticks over the
    upper row (row top - 0.015 OCD to its band's bottom), the row's ends as short red ticks in a strip below, the title and
    the count in a band below the pictures (never over the face)."""
    x0, y0, x1, y1 = res['box']
    cr = img[y0:y1, x0:x1]
    s = float(np.clip((width or 640) / max(1, cr.shape[1]), 3.0, 4.0)) if width is None else width / max(1, cr.shape[1])
    big = cv2.resize(cr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    tk = big.copy()
    xs, ts_ = res.get('_top', (np.zeros(0), np.zeros(0)))
    _, bb = res.get('_band', (None, np.zeros(0)))
    for x in res['minima_x']:
        j = int(np.argmin(np.abs(xs - x))) if len(xs) else None
        ya = (ts_[j] - 0.015 * ocd - y0) * s if j is not None else 0
        yb = (bb[j] - y0) * s if j is not None else big.shape[0]
        X = int(round((x - x0) * s))
        cv2.line(tk, (X, int(max(0, ya))), (X, int(min(big.shape[0] - 1, yb))), (80, 255, 80), 1, cv2.LINE_AA)
    pic = np.hstack([big, np.zeros((big.shape[0], 6, 3), np.uint8), tk])
    strip = np.zeros((14, pic.shape[1], 3), np.uint8)
    off = big.shape[1] + 6
    for x in res['minima_x']:
        X = off + int(round((x - x0) * s))
        cv2.line(strip, (X, 0), (X, 10), (80, 255, 80), 1)
    if res.get('row'):
        for x in res['row']:
            X = off + int(round((x - x0) * s))
            cv2.line(strip, (X, 0), (X, 6), (60, 60, 255), 2)
    band = np.zeros((26, pic.shape[1], 3), np.uint8)
    cv2.putText(band, f"{title}: {res['n']} upper teeth ({len(res['minima_x'])} interdental minima), row {res['row_ocd']:.3f} OCD",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([pic, strip, band])


# ------------------------------------------------------------------------------------------------- plate teeth --
def plate_teeth(tex, upper=True, iod=1.0, texel=1.0, h0=0.0, unit=None, a=None, dist=0.04, sigma=0.006, hp_sigma=0.05):
    """the interdental lines of a BUILT arch plate (mouth_layers.ArchBand.tex: BGRA float, Th x Tw; rows = h from h0 at
    `texel` texels per canonical px, h along the head's down axis, the incisal edge at h = 0, the upper teeth above it
    (h < 0), the lower plate's teeth below it (h > 0); columns = arc length a = (col + 0.5) / texel - Tw / (2 texel)
    canonical px, 0 = the midline, or the per-column `a` given, canonical px; iod = canonical px per lsdub IOD).
    As mouth_layers.tooth_period: the L profile along the band just inside the incisal edge (upper: edge - 0.05 to edge -
    0.012 IOD; lower: the mirror below it), the mean over the opaque (alpha > 0.5) texels of each column, over the run of
    tooth-bright columns (band median L >= 0.35 x its p90), gaps interpolated; smoothed (sigma IOD), high-passed (minus a
    gaussian of hp_sigma IOD); the interdental lines = find_peaks(-hp, distance = dist IOD, prominence = max(1.5, 0.35 std)).
    -> lines (a, IOD), the midline (the line nearest a = 0 when within +-0.08 IOD), the two central widths (midline to
    the next line each side; None when missing), their mean, the median spacing; with unit = the face's IPD / OCD, each
    also in IPD (value / unit)."""
    from scipy.signal import find_peaks
    T = np.asarray(tex, np.float32)
    Th, Tw = T.shape[:2]
    io = float(iod) * float(texel)
    al = T[..., 3] if T.shape[2] > 3 else np.ones((Th, Tw), np.float32)
    A = (np.arange(Tw) + 0.5) / texel - Tw / (2.0 * texel) if a is None else np.asarray(a, float)
    e = int(round((0.0 - h0) * texel - 0.5))                                   # the incisal edge row (h = 0)
    r0, r1 = (max(0, e - int(0.05 * io)), max(1, e - int(0.012 * io))) if upper else (min(Th - 1, e + int(0.012 * io)), min(Th, e + int(0.05 * io)))
    res = dict(upper=bool(upper), lines=[], midline=None, central=[None, None], central_mean=None, spacing_med=None, n_lines=0,
               extent=None, band_rows=[int(r0), int(r1)], unit=None if unit is None else round(float(unit), 4))
    if r1 - r0 < 1: return _plate_ipd(res, unit)
    L = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    band, op = L[r0:r1], al[r0:r1] > 0.5
    cnt = op.sum(0)
    prof = np.where(cnt >= 1, (band * op).sum(0) / np.maximum(cnt, 1), np.nan)
    with np.errstate(all='ignore'):
        med = np.array([np.median(band[op[:, c], c]) if cnt[c] >= 1 else np.nan for c in range(Tw)])
    fin = np.isfinite(med)
    if fin.sum() < max(5, 0.1 * io): return _plate_ipd(res, unit)
    bright = fin & (np.nan_to_num(med) >= 0.35 * np.percentile(med[fin], 90))
    xs = np.nonzero(bright)[0]
    c0, c1 = int(xs.min()), int(xs.max())
    pr = prof[c0:c1 + 1]
    ok = np.isfinite(pr)
    if ok.sum() < 5: return _plate_ipd(res, unit)
    pr = np.interp(np.arange(len(pr)), np.nonzero(ok)[0], pr[ok])
    pr = gaussian_filter1d(pr, max(0.5, sigma * io))
    hp = pr - gaussian_filter1d(pr, hp_sigma * io, mode='nearest')
    pk, _ = find_peaks(-hp, distance=max(2, int(round(dist * io))), prominence=max(1.5, 0.35 * hp.std()))
    lines = sorted(float(A[c0 + k]) / iod for k in pk)
    res.update(lines=[round(x, 4) for x in lines], n_lines=len(lines), extent=[round(float(A[c0]) / iod, 4), round(float(A[c1]) / iod, 4)])
    if len(lines) >= 2: res['spacing_med'] = round(float(np.median(np.diff(lines))), 4)
    if lines:
        j = int(np.argmin(np.abs(lines)))
        if abs(lines[j]) <= 0.08:
            res['midline'] = round(lines[j], 4)
            left = lines[j] - lines[j - 1] if j >= 1 else None
            right = lines[j + 1] - lines[j] if j + 1 < len(lines) else None
            res['central'] = [None if left is None else round(left, 4), None if right is None else round(right, 4)]
            ws = [w for w in (left, right) if w is not None]
            res['central_mean'] = round(float(np.mean(ws)), 4) if ws else None
    return _plate_ipd(res, unit)


def _plate_ipd(res, unit):
    """the IPD copies of a plate_teeth result (value_ipd = value_iod / unit, unit = IPD / OCD)."""
    if unit:
        f = lambda x: None if x is None else round(float(x) / float(unit), 4)
        res.update(lines_ipd=[f(x) for x in res['lines']], midline_ipd=f(res['midline']), central_ipd=[f(x) for x in res['central']],
                   central_mean_ipd=f(res['central_mean']), spacing_med_ipd=f(res['spacing_med']))
    return res


def load_tex(path):
    """the upper and lower plates of a tex npz: paint --warp-arch-tex-out (up_tex, up_a, up_h in IOD units, lo_*, iod) or
    v12/reg/tools/platediag.py (up, lo, up_h0, lo_h0, texel, iod) -> {'up': kw, 'lo': kw} for plate_teeth(**kw)."""
    Z = np.load(path)
    iod = float(Z['iod'])
    out = {}
    for k in ('up', 'lo'):
        if f'{k}_tex' in Z.files:
            a, h = np.asarray(Z[f'{k}_a'], float) * iod, np.asarray(Z[f'{k}_h'], float) * iod
            texel = 1.0 / float(np.median(np.diff(a)))
            out[k] = dict(tex=Z[f'{k}_tex'], iod=iod, texel=texel, h0=float(h[0] - 0.5 / texel), a=a)
        elif k in Z.files:
            out[k] = dict(tex=Z[k], iod=iod, texel=float(Z['texel']), h0=float(Z[f'{k}_h0']))
    if not out: raise ValueError(f'{path}: neither paint --warp-arch-tex-out nor platediag keys')
    return out


# ----------------------------------------------------------------------------------------------------- the ruler --
def _pub(d):
    """a result dict without its drawing-only keys."""
    return {k: v for k, v in d.items() if not k.startswith('_')}


def measure(take, render, track, frames=None, tex=None, label=None, stills=None, ipd_guard=0.3):
    """the whole ruler for one take / render pair (see the module doc) -> the summary dict (json-able)."""
    P = load_track(track)
    n = len(P) if not frames else min(int(frames), len(P))
    T = read_frames(take, n)
    R = read_frames(render, n)
    n = min(n, len(T), len(R))
    if n == 0: raise ValueError('no frames to measure')
    P, T, R = P[:n], T[:n], R[:n]
    ipd, ocd = units(P)
    ipd, fixed = guard_ipd(ipd, ocd, ipd_guard)
    sk = take_skin_L(T, P)
    unit = float(np.median(ipd / ocd))
    label = label or (os.path.basename(os.path.dirname(os.path.abspath(render))) if os.path.basename(render) == 'full.mp4'
                      else os.path.splitext(os.path.basename(render))[0])
    out = dict(label=label, take=take, render=render, track=track, frames=n, skin_L=_r(sk, 2), ipd_ocd=round(unit, 4),
               ipd_px_med=_r(float(np.median(ipd)), 2), ocd_px_med=_r(float(np.median(ocd)), 2), thr_ipd=THR, ipd_guard=ipd_guard,
               ipd_fixed_frames=fixed, keys={}, series={}, teeth={})
    sp = {}
    for key in KEYS:
        st, sr = spans(T, P, key, sk, ipd), spans(R, P, key, sk, ipd)
        sp[key] = (st, sr)
        s = summarize(st, sr, ipd, ocd)
        out['keys'][key] = dict(ipd=s['ipd'], ocd=s['ocd'], both_frames=np.nonzero(s['ok'])[0].tolist())
        out['series'][key] = dict(take_ipd=[_r(x) for x in st / ipd], render_ipd=[_r(x) for x in sr / ipd],
                                  take_ocd=[_r(x) for x in st / ocd], render_ocd=[_r(x) for x in sr / ocd])
    jobs = (('render_tl', R, int(np.argmax(sp['tl'][1])), 'tl'), ('render_lab', R, int(np.argmax(sp['lab'][1])), 'lab'),
            ('take_tl', T, int(np.argmax(sp['tl'][0])), 'tl'))
    for nm, F, i, key in jobs:
        c = count_teeth(F[i], P[i, :, :2], ipd[i], ocd[i], key, sk)
        src = 'render' if nm.startswith('render') else 'take'
        out['teeth'][nm] = dict(frame=i, span_ipd=_r(sp[key][1 if src == 'render' else 0][i] / ipd[i]), **_pub(c))
        if stills:
            os.makedirs(stills, exist_ok=True)
            img = teeth_still(F[i], c, ocd[i], f'{label} {src} f{i} key {key}')
            cv2.imwrite(os.path.join(stills, f'{label}_{nm}_f{i}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if tex:
        D = load_tex(tex)
        out['plate'] = dict(tex=tex, **{k: plate_teeth(upper=k == 'up', unit=unit, **D[k]) for k in D})
    out['accept'] = acceptance(out)
    return out


def acceptance(summary):
    """the v13 tooth-scale acceptance lines of a measure() summary: per key PASS iff render span p90 <= 1.05 x take span p90
    AND the ratio median <= 1.05 (IPD, over the frames where both spans > 0.12); the upper central width (plate, IPD) in
    [0.12, 0.15]; the visible upper teeth at the render's widest frame (key tl) <= 8. -> list of dict(name, status
    'PASS' | 'FAIL' | 'n/a', text); n/a = not measurable here (no frame with both spans, no --tex)."""
    lines = []
    for key in KEYS:
        s = (summary.get('keys', {}).get(key) or {}).get('ipd') or {}
        tp, rp, rm = s.get('take_p90'), s.get('render_p90'), s.get('ratio_med')
        if not s.get('n') or None in (tp, rp, rm):
            lines.append(dict(name=f'span_{key}', status='n/a', text=f'span {key}: no frame with both spans > {THR} IPD'))
            continue
        ok = rp <= ACCEPT['span'] * tp and rm <= ACCEPT['ratio']
        lines.append(dict(name=f'span_{key}', status='PASS' if ok else 'FAIL',
                          text=f"span {key}: render p90 {rp:.3f} vs {ACCEPT['span']} x take p90 {tp:.3f} = {ACCEPT['span'] * tp:.3f}, "
                               f"ratio median {rm:.3f} vs {ACCEPT['ratio']} (n {s['n']})"))
    pl = (summary.get('plate') or {}).get('up')
    if pl is None:
        lines.append(dict(name='central', status='n/a', text='central width: no --tex'))
    else:
        cw = pl.get('central_mean_ipd')
        lo, hi = ACCEPT['central']
        lines.append(dict(name='central', status='PASS' if cw is not None and lo <= cw <= hi else 'FAIL',
                          text='central width (upper plate): ' + ('no central incisor (midline / its neighbour lines not found)' if cw is None else
                               f"{cw:.3f} IPD ({pl['central_mean']:.3f} IOD; L {pl['central'][0]} R {pl['central'][1]} IOD) in [{lo}, {hi}]")))
    t = (summary.get('teeth') or {}).get('render_tl')
    if t is None: lines.append(dict(name='teeth', status='n/a', text='visible upper teeth: not counted'))
    else:
        lines.append(dict(name='teeth', status='PASS' if t['n'] <= ACCEPT['teeth'] else 'FAIL',
                          text=f"visible upper teeth at the render's widest frame f{t['frame']}: {t['n']} <= {ACCEPT['teeth']}"))
    return lines


def _f(x, d=3):
    return '-' if x is None else f'{x:.{d}f}'


def markdown(S):
    """the short markdown report of a measure() summary."""
    L = [f"## tooth-scale {S['label']}", '', f"take `{S['take']}`  ", f"render `{S['render']}`  ",
         f"{S['frames']} frames, skin L {S['skin_L']}, IPD {S['ipd_px_med']} px, OCD {S['ocd_px_med']} px (IPD / OCD {S['ipd_ocd']})", '',
         f"visible tooth span (frames where both > {S['thr_ipd']} IPD; all = every frame)", '',
         '| key | unit | n | take p50 | take p90 | render p50 | render p90 | ratio med | all: take p90 | all: render p90 |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for key in KEYS:
        for u in ('ipd', 'ocd'):
            s = S['keys'][key][u]
            L.append(f"| {key} | {u.upper()} | {s['n']} | {_f(s['take_p50'])} | {_f(s['take_p90'])} | {_f(s['render_p50'])} | {_f(s['render_p90'])} | "
                     f"{_f(s['ratio_med'])} | {_f(s['all']['take_p90'])} | {_f(s['all']['render_p90'])} |")
    L += ['', '| visible upper teeth | frame | key | span IPD | teeth | row OCD |', '|---|---|---|---|---|---|']
    for nm, t in S['teeth'].items():
        L.append(f"| {nm.replace('_', ' ')} | {t['frame']} | {t['key']} | {_f(t['span_ipd'])} | {t['n']} | {_f(t['row_ocd'])} |")
    if S.get('plate'):
        L += ['', f"plate `{S['plate']['tex']}` (IPD = IOD / {S['ipd_ocd']})", '', '| arch | lines | midline IOD | central L / R IOD | mean IOD | mean IPD | spacing IOD |',
              '|---|---|---|---|---|---|---|']
        for k in ('up', 'lo'):
            p = S['plate'].get(k)
            if p: L.append(f"| {k} | {p['n_lines']} | {_f(p['midline'])} | {_f(p['central'][0])} / {_f(p['central'][1])} | {_f(p['central_mean'])} | "
                           f"{_f(p.get('central_mean_ipd'))} | {_f(p['spacing_med'])} |")
    L += ['', 'acceptance', ''] + [f"- {a['status']} {a['text']}" for a in S['accept']]
    return '\n'.join(L) + '\n'


def line(S):
    """the one stdout line of a measure() summary."""
    k = S['keys']
    t = S['teeth']
    p = (S.get('plate') or {}).get('up') or {}
    st = 'PASS' if all(a['status'] != 'FAIL' for a in S['accept']) else 'FAIL'
    seg = ' | '.join(f"{key} span p90 take {_f(k[key]['ipd']['take_p90'])} render {_f(k[key]['ipd']['render_p90'])} ratio {_f(k[key]['ipd']['ratio_med'])} "
                     f"n {k[key]['ipd']['n']}" for key in KEYS)
    return (f"{S['label']}: {seg} | teeth render {t['render_tl']['n']} (f{t['render_tl']['frame']}) take {t['take_tl']['n']} "
            f"(f{t['take_tl']['frame']}) | central {_f(p.get('central_mean_ipd'))} IPD | {st}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub tooth-scale', description=__doc__.split('\n')[0])
    ap.add_argument('take', help='the take (the video the render was painted on)')
    ap.add_argument('render', help='the render')
    ap.add_argument('--track', required=True, help="the take's raw MediaPipe track: npz (key S3, else its only 3-D array) or json")
    ap.add_argument('--frames', type=int, help='at most this many frames (default: all the three inputs share)')
    ap.add_argument('--tex', help='arch plates (npz of paint --warp-arch-tex-out or platediag / platebuild): the central widths')
    ap.add_argument('--json')
    ap.add_argument('--md')
    ap.add_argument('--label', help='name in the reports (default: the render dir)')
    ap.add_argument('--stills', help='write the by-eye check stills of the teeth counts (render / take widest frames) to this dir')
    ap.add_argument('--ipd-guard', type=float, default=0.3, help='frames with IPD / OCD below this (collapsed irises: no face found) '
                    'take the median ratio x OCD as their IPD (default 0.3; 0.55 also guards blinks; 0 = off)')
    a = ap.parse_args(argv)
    S = measure(a.take, a.render, a.track, a.frames, a.tex, a.label, a.stills, a.ipd_guard)
    if a.json:
        with open(a.json, 'w') as fh: json.dump(S, fh, indent=1)
    if a.md:
        with open(a.md, 'w') as fh: fh.write(markdown(S))
    print(line(S))
    for x in S['accept']: print(f"  {x['status']:4s} {x['text']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
