#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""cadence.py: the CHECKER for motion cadence (held / skipped frames). CPU. The maker is lsdub/core/retime.py.

frame-gate measures the mouth as EXCESS over the take, so a hitch the take already has (a generator stall: the picture
slows to a crawl for a few frames, then catches up) is "allowed" there and passes. This ruler is absolute: it measures
the shot's own motion speed (median DIS flow magnitude, px / frame, whole frame and optionally a ROI) and flags
  HOLD   a run of 1-4 steps whose speed is < hold_ratio x the speed around it (the motion it interrupts): reads as a
         dropped / frozen frame
  SKIP   one step > skip_ratio x both neighbours: reads as a jump
Only where the surrounding motion is real (> min_speed px / frame); a still shot holding still is not a hitch.

  lsdub cadence --video plate.mp4 [--roi y0,y1,x0,x1] [--json out.json]
"""
import argparse, json
import cv2, numpy as np


def speeds(video, roi=None, size=512):
    """median optical-flow magnitude per step (len = frames - 1), in px of the analysed image."""
    cap = cv2.VideoCapture(video)
    g = []
    while True:
        ok, f = cap.read()
        if not ok: break
        if roi: f = f[roi[0]:roi[1], roi[2]:roi[3]]
        elif size: f = cv2.resize(f, (size, size * f.shape[0] // f.shape[1]), interpolation=cv2.INTER_AREA)
        g.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return np.array([np.median(np.linalg.norm(dis.calc(g[i], g[i + 1], None), axis=2)) for i in range(len(g) - 1)])


def find(sp, hold_ratio=0.45, skip_ratio=2.2, min_speed=0.5, max_run=4, ctx=3):
    """-> list of dict(kind, a, b, speed, around): steps a..b-1 (frames a..b)."""
    out = []
    n = len(sp)
    i = 0
    while i < n:
        hold = 0
        for L in range(1, max_run + 1):                              # shortest run that is a hold
            pre = sp[max(0, i - ctx):i]
            post = sp[i + L:i + L + ctx]
            if len(pre) == 0 or len(post) == 0: continue
            around = min(np.median(pre), np.median(post))
            run = sp[i:i + L]
            if around > min_speed and run.max() < hold_ratio * around:
                while i + L < n and L < max_run and sp[i + L] < hold_ratio * around: L += 1     # take the whole stall
                run = sp[i:i + L]
                out.append(dict(kind='HOLD', a=i, b=i + L, speed=round(float(run.mean()), 3), around=round(float(around), 3)))
                hold = L
                break
        if hold: i += hold
        else:
            if 0 < i < n - 1 and sp[i] > min_speed and sp[i] > skip_ratio * max(sp[i - 1], sp[i + 1]):
                out.append(dict(kind='SKIP', a=i, b=i + 1, speed=round(float(sp[i]), 3),
                                around=round(float(max(sp[i - 1], sp[i + 1])), 3)))
            i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--roi', help='y0,y1,x0,x1 (default: whole frame)')
    ap.add_argument('--hold-ratio', type=float, default=0.45)
    ap.add_argument('--skip-ratio', type=float, default=2.2)
    ap.add_argument('--min-speed', type=float, default=0.5)
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    roi = [int(v) for v in a.roi.split(',')] if a.roi else None
    sp = speeds(a.video, roi)
    ev = find(sp, a.hold_ratio, a.skip_ratio, a.min_speed)
    print(f"cadence {a.video}: {len(sp) + 1} frames, median speed {np.median(sp):.2f} px/f, "
          f"{'PASS' if not ev else 'FLAG'} {len(ev)}")
    for e in ev: print(f"  {e['kind']} frames {e['a']}-{e['b']}  speed {e['speed']} vs {e['around']} around")
    if a.json: json.dump(dict(video=a.video, speeds=[round(float(s), 3) for s in sp], events=ev), open(a.json, 'w'), indent=1)
    return ev


if __name__ == '__main__':
    main()
