#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""pose_mix.py — cross-fade the paints of a shot made from SEVERAL pose libraries (one per lsdub scan cluster),
weighted per frame by the head pose: each frame leans on the library shot closest to its own yaw/pitch, with a
smoothstep hand-over between cluster centres (eased in time, so no pop at the switch). Masks are mixed the same way.

  lsdub posemix --scan scan_shot.json --paints a.mp4 b.mp4 --masks a.npz b.npz --out s3.mp4 --mask-out s3.npz
(--paints are in --scan's cluster order, i.e. the order of the libraries built from its keyframes)
"""
import argparse, json
from ..io import ff
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', required=True)
    ap.add_argument('--paints', nargs='+', required=True)
    ap.add_argument('--masks', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mask-out')
    ap.add_argument('--soft-deg', type=float, default=4.0, help='softness of the hand-over (deg); smaller = sharper')
    a = ap.parse_args(argv)
    sc = json.load(open(a.scan))
    C = np.array([c['centre'] for c in sc['clusters']])[:len(a.paints)]
    rows = sc['frames']
    n = len(rows)
    ok = np.array([r['yaw'] is not None for r in rows])
    ii = np.arange(n)
    pose = np.stack([np.interp(ii, ii[ok], [r[k] for r in rows if r['yaw'] is not None]) for k in ('yaw', 'pitch')], 1)
    d = np.linalg.norm(pose[:, None] - C[None], axis=2)                      # n x L distance to each cluster centre
    w = np.exp(-(d - d.min(1, keepdims=True)) / a.soft_deg)
    w = gaussian_filter1d(w, 2.0, axis=0, mode='nearest')
    w /= w.sum(1, keepdims=True)
    caps = [cv2.VideoCapture(p) for p in a.paints]
    Wd, Hd = int(caps[0].get(3)), int(caps[0].get(4))
    fps = caps[0].get(5) or 25
    pw = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{Wd}x{Hd}', '-r', str(fps), '-i', '-',
                           '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', a.out], stdin=ff.PIPE)
    for i in range(n):
        fr = [c.read()[1] for c in caps]
        if any(f is None for f in fr): break
        pw.stdin.write(np.clip(sum(w[i, j] * fr[j].astype(np.float32) for j in range(len(fr))), 0, 255).astype(np.uint8).tobytes())
    pw.stdin.close()
    pw.wait()
    if a.masks and a.mask_out:
        M = [np.load(m)['masks'].astype(np.float32) for m in a.masks]
        k = min(len(m) for m in M)
        out = sum(w[:k, j, None, None] * M[j][:k] for j in range(len(M)))
        np.savez_compressed(a.mask_out, masks=np.clip(out, 0, 255).astype(np.uint8))
    print('weights per frame (first lib):', np.round(w[:, 0], 2).tolist())


if __name__ == '__main__':
    main()
