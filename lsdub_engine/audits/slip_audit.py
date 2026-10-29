#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""slip_audit.py — does the painted mouth stay PINNED to the face, or swim behind it? (founder 2026-10-15, v11 05C:
"the mouth is swimming behind the locked-in spot on the face ... it's mouthing accurately, just positioning")

Shape changes are allowed; POSITION is not. So we measure motion, not shape:
  - LK-track skin features (forward-backward checked) from frame 0 in each region of each video:
      nose  = nose base / alae (rigid skull reference)      cheek = upper cheeks (rigid)
      phil  = philtrum skin between nose base and lip       ulip  = upper-lip vermilion border
  - region displacement D_R(t) = median feature displacement from frame 0
  - slip_R(t) = (D_R - D_nose)_video - (D_R - D_nose)_take   (relative motion vs the nose, minus what the real face did)
  - lag: cross-correlation of the region's velocity against the nose velocity (sub-frame parabolic peak); +ve = behind
The philtrum and the upper-lip border are rigid with the skull (the upper teeth hang there), so in a pinned paint their
slip is ~0 and their lag is 0.

  lsdub slip-audit --take body.mp4 --frames 47 [--json out.json] name=video.mp4 ...
"""
import argparse, json
import cv2, numpy as np


REG = {  # MediaPipe 478 indices outlining each region (convex hull, shrunk/grown below)
    'nose':  [1, 2, 4, 5, 45, 275, 48, 278, 64, 294, 98, 327, 129, 358, 195, 197],
    'cheek': [50, 280, 101, 330, 116, 345, 117, 346, 118, 347, 123, 352, 205, 425],
    'phil':  [2, 97, 326, 167, 393, 164, 165, 391, 92, 322, 37, 267, 0],
    'ulip':  [0, 37, 267, 39, 269, 40, 270, 185, 409, 13, 82, 312, 81, 311],
}


def read(video, n):
    cap = cv2.VideoCapture(video)
    F = []
    while len(F) < n:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F


def landmarks0(frame):
    """Landmarks on ONE frame, IMAGE mode (no tracker smoothing) — only used to place the regions at frame 0."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    from ..core.mouth_paint import MODEL
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))
    H, W = frame.shape[:2]
    big = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(big[:, :, ::-1])))
    if not r.face_landmarks: raise SystemExit('no face on frame 0')
    return np.array([[q.x * W, q.y * H] for q in r.face_landmarks[0]])

def landmarks_first(frames):
    """Frame-0 landmarks even when frame 0 has no detectable face (e.g. 06C: motion blur / small face on the first
    frame): detect on the first frame that has one, then carry those points back to frame 0 along dense flow."""
    for k, f in enumerate(frames):
        try:
            p = landmarks0(f)
        except SystemExit:
            continue
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        for i in range(k, 0, -1):
            a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            fl = dis.calc(a, b, None)
            H, W = a.shape
            xi = np.clip(p[:, 0].round().astype(int), 0, W - 1)
            yi = np.clip(p[:, 1].round().astype(int), 0, H - 1)
            p = p + fl[yi, xi]
        return p
    raise SystemExit('no face in any frame')



def region_mask(p, idx, shape, grow=0.0):
    q = p[idx]
    c = q.mean(0)
    q = c + (q - c) * (1 + grow)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(q.astype(np.int32)), 255)
    return m


DIS = None


def flows(F):
    """Dense DIS optical flow t-1 -> t (computed once per video, shared by all regions)."""
    global DIS
    DIS = DIS or cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in F]
    return [DIS.calc(g[t - 1], g[t], None) for t in range(1, len(g))]


def track_region(Fl, mask):
    """Region displacement from frame 0: each step = median dense flow inside the region mask, the mask carried along
    by the accumulated displacement (sub-pixel). Returns (D[t] (T,2), mask pixel count)."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    D = [np.zeros(2)]
    H, W = mask.shape
    for fl in Fl:
        q = pts + D[-1]
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
        D.append(D[-1] + np.median(fl[yi, xi], 0))
    return np.array(D), len(pts)


def lag_frames(v_ref, v):
    """Sub-frame lag of v behind v_ref (velocity series), +ve = behind; searches +-4 frames."""
    a = v_ref - v_ref.mean()
    b = v - v.mean()
    cs = {}
    for L in range(-4, 5):
        x = a[max(0, -L):len(a) - max(0, L)]
        y = b[max(0, L):len(b) - max(0, -L)]
        cs[L] = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    L0 = max(cs, key=cs.get)
    if -4 < L0 < 4:
        y0, y1, y2 = cs[L0 - 1], cs[L0], cs[L0 + 1]
        den = y0 - 2 * y1 + y2
        return L0 + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0), y1
    return float(L0), cs[L0]


def audit(F, p0):
    out = {}
    Fl = flows(F)
    for k, idx in REG.items():
        grow = 0.15 if k in ('nose', 'cheek') else 0.0
        if k == 'cheek':   # two cheeks: union of left/right halves
            m = region_mask(p0, idx[0::2], F[0].shape, 0.2) | region_mask(p0, idx[1::2], F[0].shape, 0.2)
        else:
            m = region_mask(p0, idx, F[0].shape, grow)
        out[k] = track_region(Fl, m)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub slip-audit')
    ap.add_argument('--take', required=True)
    ap.add_argument('--frames', type=int, default=10**6)
    ap.add_argument('--json')
    ap.add_argument('videos', nargs='+')
    a = ap.parse_args(argv)
    T = read(a.take, a.frames)
    p0 = landmarks_first(T)
    ref = audit(T, p0)
    rel_take = {k: ref[k][0] - ref['nose'][0] for k in REG}
    res = {}
    print(f"{'video':10s} {'region':6s} {'slip med px':>11s} {'slip p95':>8s} {'slip max':>8s} {'lag fr':>7s} {'r':>5s} {'pts':>4s}")
    for spec in [f'take={a.take}'] + a.videos:
        name, path = spec.split('=', 1)
        F = read(path, len(T))
        n = min(len(F), len(T))
        A = audit(F[:n], p0)
        res[name] = {}
        vn = np.diff(A['nose'][0][:n], axis=0)
        for k in ('cheek', 'phil', 'ulip'):
            rel = A[k][0][:n] - A['nose'][0][:n]
            s = np.linalg.norm(rel - rel_take[k][:n], axis=1)
            vk = np.diff(A[k][0][:n], axis=0)
            # lag on the dominant motion axis (larger nose velocity variance)
            ax = int(np.argmax(vn.var(0)))
            L, r = lag_frames(vn[:, ax], vk[:, ax])
            res[name][k] = dict(slip_med=float(np.median(s)), slip_p95=float(np.percentile(s, 95)), slip_max=float(s.max()),
                                lag=float(L), r=float(r), pts=A[k][1], slip=s.round(2).tolist())
            print(f"{name:10s} {k:6s} {np.median(s):11.2f} {np.percentile(s, 95):8.2f} {s.max():8.2f} {L:7.2f} {r:5.2f} {A[k][1]:4d}")
    if a.json: json.dump(res, open(a.json, 'w'))


if __name__ == '__main__':
    main()
