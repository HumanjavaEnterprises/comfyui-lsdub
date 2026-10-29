#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_pin.py — pin the painted lower face to the REAL skin motion of the take (founder 2026-10-15, v11 05C:
"the mouth is swimming behind the locked-in spot on the face").

Why it swims: every paint stage places the mouth from MediaPipe landmarks, and on this take those landmarks miss the
real skin motion by 2-8 px (worst on the fast head swing, f13-19; slip_audit: landmark-vs-flow err p95 ~7 px). The
painted philtrum/lip/chin therefore moves LESS than the nose on fast moves and drifts back after — it swims.

Fix = decouple position from shape:
  - shape stays whatever the paint made (lead-shifted track, pose-native mouths, skull/jaw locks)
  - position comes from the take's REAL skin motion: dense DIS flow tracks the philtrum and the nose base in the take
    and in the output; the painted philtrum must move like the output's nose plus the take's own philtrum-vs-nose
    motion. The difference is the slip, removed per frame. (Registering the output's philtrum straight onto the take's
    was tried and fails: the take sings different mouth shapes, so cross-texture matching drifts.)
  - offsets are cleaned with a CENTERED (zero-phase) gaussian, sigma ~0.6 frame — noise only, no look-ahead/behind
  - the lower face (philtrum -> chin, corner to corner) is translated by the offset inside a feathered mask that sits
    at the take's true position; the nose/eyes/cheekbones are untouched
  - a second pass re-measures and removes what is left

  lsdub pin --video v11.mp4 --take body.mp4 --frames 47 --out v11_pin.mp4 [--json pin.json] [--passes 2]
"""
import argparse, json
from ..io import ff
import cv2, numpy as np
from scipy.ndimage import gaussian_filter1d

from ..audits import slip_audit as sa

LOWER = [164, 92, 322, 206, 426, 216, 436, 212, 432, 202, 422, 210, 430, 169, 394, 170, 395, 140, 369, 171, 396,
         175, 152, 148, 377, 176, 400]   # philtrum -> chin, INSIDE the jawline: never the earrings/hair/neck edge


def lower_mask(p0, shape, D, t, grow=0.04, feather=13):
    q = p0[LOWER] + D[t]
    c = q.mean(0)
    q = c + (q - c) * (1 + grow)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(q.astype(np.int32)), 255)
    k = feather * 2 + 1
    m = cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255
    # keep the nose: nothing above the subnasale line moves (cut AFTER the feather, with a short ramp below it)
    sn_y = p0[2, 1] + D[t][1]
    ys = np.arange(shape[0], dtype=np.float32)[:, None]
    m *= np.clip((ys - sn_y) / 4.0, 0, 1)
    return m[..., None]


def pin_pass(O, T, p0, Dtake, Atake, sigma, max_px):
    """Correction = what the painted philtrum SHOULD have moved (output nose + the take's own philtrum-vs-nose motion)
    minus what it DID move (dense flow on the output). Robust: flow, not cross-texture matching (the take's mouth is
    singing different shapes, so absolute registration take<->output is ill-posed; the relative motion is not)."""
    Aout = sa.audit(O, p0)
    want = Aout['nose'][0] + (Atake['phil'][0] - Atake['nose'][0])
    S = want - Aout['phil'][0]                                  # (n,2) residual slip, frame 0 = 0 by construction
    C = np.stack([gaussian_filter1d(S[:, d], sigma, mode='nearest') for d in range(2)], 1) if sigma > 0 else S
    n = np.linalg.norm(C, axis=1, keepdims=True)
    C = np.where(n > max_px, C * max_px / np.maximum(n, 1e-6), C)
    out = []
    for t, f in enumerate(O):
        M = np.float32([[1, 0, C[t, 0]], [0, 1, C[t, 1]]])
        sh = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        m = lower_mask(p0, f.shape, Dtake, t)
        out.append((f * (1 - m) + sh * m).clip(0, 255).astype(np.uint8))
    return out, S, C


def write(frames, fps, path, audio_src):
    H, W = frames[0].shape[:2]
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps),
                          '-i', '-', '-i', audio_src, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-crf', '14',
                          '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', path], stdin=ff.PIPE)
    for f in frames: p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub pin')
    ap.add_argument('--video', required=True)
    ap.add_argument('--take', required=True)
    ap.add_argument('--frames', type=int, default=10**6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json')
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--sigma', type=float, default=0.6)
    ap.add_argument('--max-px', type=float, default=16.0)
    a = ap.parse_args(argv)
    T = sa.read(a.take, a.frames)
    O = sa.read(a.video, len(T))
    n = min(len(T), len(O))
    T, O = T[:n], O[:n]
    p0 = sa.landmarks_first(T)
    A = sa.audit(T, p0)
    Dtake = A['phil'][0]
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 25
    log = []
    for k in range(a.passes):
        O, S, C = pin_pass(O, T, p0, Dtake, A, a.sigma, a.max_px)
        log.append(dict(pass_=k + 1, shift_med=float(np.median(np.linalg.norm(S, axis=1))),
                        shift_p95=float(np.percentile(np.linalg.norm(S, axis=1), 95)), applied=C.round(2).tolist()))
        print(f"pin pass {k + 1}: residual slip med {log[-1]['shift_med']:.2f} p95 {log[-1]['shift_p95']:.2f} px")
    write(O, fps, a.out, a.video)
    if a.json: json.dump(log, open(a.json, 'w'))
    print('pinned ->', a.out)


if __name__ == '__main__':
    main()
