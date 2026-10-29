#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lipsync_temporal.py: temporal/composite metrics for a painted lip-sync video (supporting evidence; the eye decides).

  jerk  = SNAPPING. MediaPipe lip landmarks (outer+inner) relative to the nose tip, normalised by face height; the third
          finite difference per frame (jerk), RMS over the points, then the median and p95 over frames. Lower = smoother.
  seam  = SEAMS. Gradient magnitude of the video on the paint-mask boundary band (mask 0.35-0.65) divided by the
          gradient on the adjacent inside/outside bands (0.08-0.3 and 0.7-0.92), averaged over frames. About 1.0 = no
          visible edge; >1 = a seam line. Needs the mask npz from mouth_paint.py --mask-out.
  lsdub jerk --video v.mp4 [--masks m.npz] [--json out.json]
"""
import argparse, json
import numpy as np, cv2
from ..core import mouth_paint as mp_
LIPS = mp_.LIP_OUT + [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def jerk(video, n=0):
    cap = cv2.VideoCapture(video)
    N = n or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    S, fps, found = mp_.track(video, N)       # same gap-fill + light smoothing as the painter, for every video
    fh = np.linalg.norm(S[:, 10] - S[:, 152], axis=1)[:, None, None]
    rel = (S[:, LIPS] - S[:, [1]]) / fh
    j = np.diff(rel, 3, axis=0)
    r = np.sqrt((j ** 2).sum(2).mean(1)) * 1000
    return {'jerk_med': round(float(np.median(r)), 3), 'jerk_p95': round(float(np.percentile(r, 95)), 3), 'found': found}


def seam(video, masks):
    M = np.load(masks)['masks'].astype(np.float32) / 255
    cap = cv2.VideoCapture(video)
    vals = []
    for m in M:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        band = (m > 0.35) & (m < 0.65)
        nb = ((m > 0.08) & (m < 0.3)) | ((m > 0.7) & (m < 0.92))
        if band.sum() > 20 and nb.sum() > 20: vals.append(gm[band].mean() / (gm[nb].mean() + 1e-6))
    return {'seam': round(float(np.mean(vals)), 3), 'seam_p95': round(float(np.percentile(vals, 95)), 3)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--masks')
    ap.add_argument('--json')
    ap.add_argument('--frames', type=int, default=200)
    a = ap.parse_args(argv)
    r = jerk(a.video, a.frames)
    if a.masks: r.update(seam(a.video, a.masks))
    print(json.dumps(r))
    if a.json: json.dump(r, open(a.json, 'w'), indent=1)
    return r


if __name__ == '__main__':
    main()
