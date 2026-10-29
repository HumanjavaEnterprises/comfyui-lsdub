#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""retime.py: the MAKER for cadence hitches (lsdub/audits/cadence.py is the checker). CPU.

A generator stall is a stretch where the picture crawls for a few frames, then catches up. The frames all exist; they
are spaced wrongly in time. Around each flagged hitch (padded by --pad frames) the motion is re-spaced so its speed is
even: cumulative motion s(t) is measured (flow speed per step), a straight line from the window's first to last frame
is the target, and each frame inside is re-made at the source time where s reaches the target, by flow-based
interpolation between the two source frames around it. Frame count, the first and last frame of each window, and
everything outside the windows are unchanged, so audio sync and the track timing are kept.

  lsdub retime --video plate.mp4 --out plate_rt.mp4 [--pad 3]   (detects with the cadence checker, then fixes)
"""
import argparse
from ..io import ff
import cv2, numpy as np
from ..audits import cadence


def _read(video):
    cap = cv2.VideoCapture(video)
    F = []
    while True:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F, cap.get(cv2.CAP_PROP_FPS)


def interp(f0, f1, a, dis):
    """frame at fraction a between f0 and f1: both warped along the flow toward time a, then blended."""
    if a <= 1e-3: return f0.copy()
    if a >= 1 - 1e-3: return f1.copy()
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    fw = dis.calc(g0, g1, None)
    bw = dis.calc(g1, g0, None)
    h, w = g0.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    w0 = cv2.remap(f0, xx + a * bw[..., 0], yy + a * bw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(f1, xx + (1 - a) * fw[..., 0], yy + (1 - a) * fw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0, 1 - a, w1, a, 0)


def retime(F, sp, windows):
    """F frames, sp per-step speed, windows [(a, b)] frame ranges -> new frames (same count)."""
    out = list(F)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for a, b in windows:
        s = np.r_[0, np.cumsum(sp[a:b])]                             # cumulative motion at frames a..b
        tgt = np.linspace(0, s[-1], b - a + 1)
        for k in range(1, b - a):
            j = int(np.clip(np.searchsorted(s, tgt[k]) - 1, 0, b - a - 1))
            fr = (tgt[k] - s[j]) / max(1e-6, s[j + 1] - s[j])
            out[a + k] = interp(F[a + j], F[a + j + 1], float(np.clip(fr, 0, 1)), dis)
    return out


def windows_for(events, n, pad=3):
    W = []
    for e in events:
        a, b = max(0, e['a'] - pad), min(n - 1, e['b'] + pad)
        if W and a <= W[-1][1]: W[-1] = (W[-1][0], max(W[-1][1], b))
        else: W.append((a, b))
    return W


def write(frames, fps, out, src):
    h, w = frames[0].shape[:2]
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps),
                          '-i', '-', '-i', src, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-crf', '12',
                          '-pix_fmt', 'yuv420p', '-c:a', 'copy', out], stdin=ff.PIPE)
    for f in frames: p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pad', type=int, default=3)
    ap.add_argument('--hold-ratio', type=float, default=0.45)
    ap.add_argument('--min-speed', type=float, default=0.5)
    a = ap.parse_args(argv)
    F, fps = _read(a.video)
    sp = cadence.speeds(a.video)
    ev = cadence.find(sp, a.hold_ratio, min_speed=a.min_speed)
    W = windows_for([e for e in ev if e['kind'] == 'HOLD'], len(F), a.pad)
    print(f"retime {a.video}: {len(ev)} events, windows {W}")
    write(retime(F, sp, W), fps, a.out, a.video)
    sp2 = cadence.speeds(a.out)
    ev2 = cadence.find(sp2, a.hold_ratio, min_speed=a.min_speed)
    print(f"confirm {a.out}: {'PASS' if not ev2 else 'FLAG ' + str(ev2)}")


if __name__ == '__main__':
    main()
