# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""skull_audit.py — does the mouth line up with the skull?

Per frame, in the face's own frame (eye line = x axis, nose bridge→nose base = midline):
  mid    upper-lip centre (philtrum, lm 0) offset from the nose midline, / inter-ocular width, MINUS the take's own
         value on the same frame (the take's upper lip is real anatomy, so its offset is the target)
  axis   mouth-corner line angle relative to the eye line, MINUS the take's (deg)
  teeth  centre of the visible upper teeth (bright, low-saturation pixels in the upper inner mouth) offset from the
         nose midline, / inter-ocular width, minus the take's philtrum offset (teeth sit under the philtrum)
  width  mouth width / inter-ocular width, ratio to the take's (yaw foreshortening consistency; 1.0 = consistent)

  lsdub skull-audit --take body.mp4 --frames N [--json out.json] name=video.mp4 [name=video.mp4 ...]"""
import argparse, json
import cv2, numpy as np
from ..core.mouth_paint import track

EYE_L, EYE_R, BRIDGE, NOSE_BASE = 33, 263, 168, 2
PHIL, CORNER_L, CORNER_R = 0, 61, 291
INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


def face_frame(p):
    """origin on the nose midline, x along the eye line, unit = inter-ocular width"""
    ex = p[EYE_R] - p[EYE_L]
    iod = np.linalg.norm(ex)
    ux = ex / max(iod, 1e-6)
    return ux, iod


def mid_offset(p, q):
    """signed offset of point q from the midline (bridge→nose base), measured along the eye line, / iod"""
    ux, iod = face_frame(p)
    a, b = p[BRIDGE], p[NOSE_BASE]
    d = b - a
    # point on the midline at q's height (along d): project q onto the midline, take the eye-line component of the residual
    t = np.dot(q - a, d) / max(np.dot(d, d), 1e-6)
    r = q - (a + t * d)
    return float(np.dot(r, ux) / iod)


def axis_angle(p):
    ux, _ = face_frame(p)
    m = p[CORNER_R] - p[CORNER_L]
    return float(np.degrees(np.arctan2(ux[0] * m[1] - ux[1] * m[0], np.dot(ux, m))))


def teeth_offset(f, p):
    m = np.zeros(f.shape[:2], np.uint8)
    cv2.fillPoly(m, [p[INNER].astype(np.int32)], 1)
    up = p[13]
    lo = p[14]
    if np.linalg.norm(lo - up) < 4 or m.sum() < 60: return np.nan
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    yy, xx = np.nonzero(m)
    # upper half of the opening (teeth hang from the upper lip)
    keep = (yy - up[1]) < 0.6 * (lo[1] - up[1]) + 2
    yy, xx = yy[keep], xx[keep]
    s, v = hsv[yy, xx, 1].astype(float), hsv[yy, xx, 2].astype(float)
    tooth = (v > np.percentile(v, 70)) & (s < 90) & (v > 120)
    if tooth.sum() < 15: return np.nan
    c = np.array([xx[tooth].mean(), yy[tooth].mean()])
    return mid_offset(p, c)


def gap_axis(f, p):
    """pixel-measured axis of the visible mouth interior (dark gap + teeth) vs the eye line (deg), and its centre's
    midline offset (/iod). The lip landmarks are a fitted face model; the interior is what the eye actually reads."""
    ux, iod = face_frame(p)
    uy = np.array([-ux[1], ux[0]])
    m = np.zeros(f.shape[:2], np.uint8)
    cv2.fillPoly(m, [p[INNER].astype(np.int32)], 1)
    if np.linalg.norm(p[14] - p[13]) < 5 or m.sum() < 80: return np.nan, np.nan
    yy, xx = np.nonzero(m)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[yy, xx].astype(float)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[yy, xx]
    # interior = dark cavity or teeth (low saturation); excludes lip-coloured pixels inside the fitted polygon
    inter = (g < np.percentile(g, 35)) | ((hsv[:, 1] < 80) & (hsv[:, 2] > 110))
    if inter.sum() < 40: return np.nan, np.nan
    pts = np.stack([xx[inter], yy[inter]], 1).astype(float)
    c = pts.mean(0)
    # principal axis of the interior blob, in face coordinates
    q = pts - c
    u = q @ ux
    v = q @ uy
    cov = np.cov(np.stack([u, v]))
    w, V = np.linalg.eigh(cov)
    ax = V[:, 1]
    if w[1] < 2 * w[0]: return np.nan, mid_offset(p, c)       # blob too round to have an axis
    ang = np.degrees(np.arctan2(ax[1], ax[0]))
    ang = (ang + 90) % 180 - 90
    return float(ang), mid_offset(p, c)


def measure(video, n):
    cap = cv2.VideoCapture(video)
    F = []
    while len(F) < n:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    P, _, _ = track(video, len(F))
    return F, P


def smooth(x, k=9):
    x = np.asarray(x, float)
    pad = np.pad(x, k // 2, mode='edge')
    return np.convolve(pad, np.ones(k) / k, 'valid')


def take_targets(Pt):
    phil = np.array([mid_offset(p, p[PHIL]) for p in Pt])
    ang = np.array([axis_angle(p) for p in Pt])
    wid = np.array([np.linalg.norm(p[CORNER_R] - p[CORNER_L]) / face_frame(p)[1] for p in Pt])
    # the take's mouth corners move with its own (different) singing; the skull-relative axis is the slow part
    return smooth(phil, 5), smooth(ang, 15), smooth(wid, 15)


def audit(F, P, tgt):
    phil_t, ang_t, wid_t = tgt
    n = min(len(F), len(P), len(phil_t))
    r = {'mid': [], 'axis': [], 'teeth': [], 'width': [], 'gap_axis': [], 'gap_mid': []}
    for t in range(n):
        p = P[t]
        r['mid'].append(mid_offset(p, p[PHIL]) - phil_t[t])
        r['axis'].append(axis_angle(p) - ang_t[t])
        r['teeth'].append(teeth_offset(F[t], p) - phil_t[t])
        ga, gm = gap_axis(F[t], p)
        r['gap_axis'].append(ga - ang_t[t])
        r['gap_mid'].append(gm - phil_t[t])
        r['width'].append(np.linalg.norm(p[CORNER_R] - p[CORNER_L]) / face_frame(p)[1] / wid_t[t])
    return {k: np.array(v) for k, v in r.items()}


def summary(r):
    a = lambda x: np.abs(x[np.isfinite(x)])
    s = {'mid_med': float(np.median(a(r['mid']))), 'mid_p95': float(np.percentile(a(r['mid']), 95)),
         'axis_med': float(np.median(a(r['axis']))), 'axis_p95': float(np.percentile(a(r['axis']), 95)),
         'width_p95': float(np.percentile(np.abs(r['width'] - 1), 95))}
    for k in ('gap_axis', 'gap_mid'):
        x = a(r[k])
        s[k + '_med'] = float(np.median(x)) if x.size else float('nan')
        s[k + '_p95'] = float(np.percentile(x, 95)) if x.size else float('nan')
    s['gap_frames'] = int(np.isfinite(r['gap_axis']).sum())
    te = a(r['teeth'])
    s['teeth_med'] = float(np.median(te)) if te.size else float('nan')
    s['teeth_p95'] = float(np.percentile(te, 95)) if te.size else float('nan')
    s['teeth_frames'] = int(te.size)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--json')
    ap.add_argument('videos', nargs='+')
    a = ap.parse_args(argv)
    Ft, Pt = measure(a.take, a.frames)
    tgt = take_targets(Pt)
    out = {}
    for spec in a.videos:
        name, v = spec.split('=', 1)
        F, P = measure(v, a.frames)
        r = audit(F, P, tgt)
        s = summary(r)
        out[name] = {'summary': s, 'per_frame': {k: [None if not np.isfinite(x) else round(float(x), 4) for x in r[k]] for k in r}}
        print(f"{name:10s} mid {s['mid_med']:.3f}/{s['mid_p95']:.3f}  axis {s['axis_med']:.2f}/{s['axis_p95']:.2f} deg  "
              f"gap-axis {s['gap_axis_med']:.1f}/{s['gap_axis_p95']:.1f} deg  gap-mid {s['gap_mid_med']:.3f}/{s['gap_mid_p95']:.3f} (n={s['gap_frames']})  "
              f"width p95 {s['width_p95']:.3f}")
    if a.json: json.dump(out, open(a.json, 'w'))


if __name__ == '__main__':
    main()
