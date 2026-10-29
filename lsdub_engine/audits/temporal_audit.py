# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""temporal_audit.py — in-motion artifact metrics for a lip-sync render (stills can look fine while motion boils).

Per video, landmarks tracked on a REFERENCE take (so all variants are measured in the same regions):
  boil   flow-compensated residual |I_t - warp(I_{t-1})| inside the mouth, divided by the same on the cheeks
         (1.0 = the mouth changes no more than the skin around it; texture swapping / dissolves push it up)
  ghost  double-edge score: vertical lip-edge energy at a 2-6 px offset relative to the main edge (doubled lower lip)
  slide  mouth-centre motion left after removing the rigid face motion (nose/eye anchors), 2nd-difference p95, px
  step   share of frames whose mouth change is > 2x both neighbours (alternate-frame stutter)

  lsdub motion-audit --ref take.mp4 --frames N --json out.json name=video.mp4 [name=video.mp4 ...]
  lsdub motion-audit ... --crops sheet.jpg --at F0   # 12 consecutive enlarged mouth crops per video from F0"""
import argparse, json
import cv2, numpy as np

from ..core.mouth_paint import track

MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
CHEEK_L, CHEEK_R = [187, 207, 205, 50], [411, 427, 425, 280]
RIGID = [1, 4, 33, 133, 263, 362, 168, 6]


def read(v, n):
    cap = cv2.VideoCapture(v)
    F = []
    while len(F) < n:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F


def poly_mask(pts, shape, grow=0.0):
    m = np.zeros(shape[:2], np.uint8)
    c = pts.mean(0)
    q = c + (pts - c) * (1 + grow)
    cv2.fillConvexPoly(m, cv2.convexHull(q.astype(np.int32)), 1)
    return m.astype(bool)


def rigid_residual(P):
    """mouth-centre position after removing a per-frame similarity fit of the rigid anchors to frame 0"""
    a0 = P[0, RIGID]
    out = []
    for t in range(len(P)):
        M, _ = cv2.estimateAffinePartial2D(P[t, RIGID].astype(np.float32), a0.astype(np.float32))
        mc = P[t, MOUTH].mean(0)
        out.append(M[:, :2] @ mc + M[:, 2])
    return np.array(out)


def audit(F, P, name, Pown=None):
    N = min(len(F), len(P))
    G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in F[:N]]
    boil, ghost, dm = [], [], []
    for t in range(1, N):
        mm = poly_mask(P[t, MOUTH], G[t].shape, 0.25)
        ck = poly_mask(P[t, CHEEK_L], G[t].shape, 0.3) | poly_mask(P[t, CHEEK_R], G[t].shape, 0.3)
        ys, xs = np.where(mm | ck)
        y0, y1, x0, x1 = max(ys.min() - 20, 0), ys.max() + 20, max(xs.min() - 20, 0), xs.max() + 20
        a, b = G[t - 1][y0:y1, x0:x1], G[t][y0:y1, x0:x1]
        fl = cv2.calcOpticalFlowFarneback(a.astype(np.uint8), b.astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
        wa = cv2.remap(a, xx - fl[..., 0], yy - fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        r = np.abs(b - wa)
        mmc, ckc = mm[y0:y1, x0:x1], ck[y0:y1, x0:x1]
        boil.append(r[mmc].mean() / max(r[ckc].mean(), 1e-3))
        dm.append(np.abs(b - a)[mmc].mean())
        # doubled edge: vertical derivative profile through the lower lip, autocorrelation at 2..6 px vs 0
        lo = P[t, 17]
        hi = P[t, 14]
        cx = int(round((lo[0] + hi[0]) / 2))
        ya, yb = int(hi[1]) - 4, int(lo[1]) + 12
        col = G[t][max(ya, 0):yb, max(cx - 15, 0):cx + 15].mean(1)
        d = np.abs(np.diff(col))
        if len(d) > 8:
            d = d - d.mean()
            c0 = (d * d).sum() + 1e-6
            ghost.append(max((d[k:] * d[:-k]).sum() / c0 for k in range(2, 7)))
    Q = Pown if Pown is not None else P
    R = rigid_residual(Q[:N])
    acc = np.linalg.norm(np.diff(R, 2, axis=0), axis=1)
    dm = np.array(dm)
    step = np.mean((dm[1:-1] > 2 * dm[:-2]) & (dm[1:-1] > 2 * dm[2:])) if len(dm) > 2 else 0.0
    return dict(name=name, boil_med=float(np.median(boil)), boil_p95=float(np.percentile(boil, 95)),
                ghost_med=float(np.median(ghost)) if ghost else None, slide_p95=float(np.percentile(acc, 95)),
                step=float(step), boil_series=[round(float(x), 3) for x in boil])


def crops(vids, P, at, out, n=12, scale=3):
    rows = []
    for name, F in vids:
        tiles = []
        for t in range(at, min(at + n, len(F), len(P))):
            c = P[t, MOUTH].mean(0)
            w = np.linalg.norm(P[t, 61] - P[t, 291]) * 1.1
            x0, y0 = int(c[0] - w), int(c[1] - w * 0.7)
            tile = F[t][max(y0, 0):y0 + int(w * 1.6), max(x0, 0):x0 + int(w * 2)]
            tile = cv2.resize(tile, (int(w * 2) * scale // 2, int(w * 1.6) * scale // 2), interpolation=cv2.INTER_CUBIC)
            cv2.putText(tile, f'{t}', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(tile)
        h = min(x.shape[0] for x in tiles)
        tiles = [cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h)) for x in tiles]
        row = np.hstack(tiles)
        lab = np.zeros((30, row.shape[1], 3), np.uint8)
        cv2.putText(lab, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.vstack([lab, row]))
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--json')
    ap.add_argument('--crops')
    ap.add_argument('--at', type=int, default=0)
    ap.add_argument('vids', nargs='+')
    a = ap.parse_args(argv)
    P, _, _ = track(a.ref, a.frames)
    V = [(s.split('=', 1)[0], read(s.split('=', 1)[1], a.frames)) for s in a.vids]
    paths = dict(s.split('=', 1) for s in a.vids)
    res = [audit(F, P, n, track(paths[n], a.frames)[0]) for n, F in V]
    for r in res:
        print(f"{r['name']:<14} boil med {r['boil_med']:.2f} p95 {r['boil_p95']:.2f}  ghost {r['ghost_med'] or 0:.2f}  "
              f"slide p95 {r['slide_p95']:.2f}  step {r['step']:.2f}")
    if a.json: json.dump(res, open(a.json, 'w'), indent=1)
    if a.crops: crops(V, P, a.at, a.crops)


if __name__ == '__main__':
    main()
