#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""plane_fit.py — does the mouth sit ON the face's 3D surface, or is it a flat sticker? (founder 2026-10-15: "more
accurate to the face and the facial plane").

Reference = the take's own most-closed, well-tracked frame: its 3D landmarks (MediaPipe x, y px; z relative depth).
For every frame of a video, the reference is moved by the 3D similarity fitted on head-rigid anchors (forehead, nose,
eye corners, temples) — i.e. the take's real mouth rotated into THIS frame's head pose. That predicts, from head pose
alone, where the UPPER-LIP ARC must be (Cupid's bow + outer upper-lip contour, corners excluded): the upper lip rides
the skull (moves <= 2-3 mm with speech), so its curve has to wrap the dental arch and foreshorten with yaw/pitch.
  arc   mean distance of the arc landmarks from that prediction, in reference mouth widths (placement + plane)
  sag   error of the arc's sagitta (how much it bows along the face's up axis; pitch / cylinder curvature)
  asym  error of the near/far half-length balance about the philtrum (yaw foreshortening)
The lower lip / jaw / corners are not used: they move with the viseme, the upper arc should not.

  lsdub plane-fit --take body.mp4 --frames 47 [--json out.json] take=body.mp4 v10=v10.mp4 v11=v11.mp4
"""
import argparse, json
import numpy as np
from ..core.mouth_paint import track as _track
from ..core.mouth_morph import ANCH

RIG = [i for i in ANCH if i not in (205, 425, 98, 327, 2, 50, 280, 187, 411)]   # head-rigid only (no cheek/nostril points)


def sim3(src, dst):
    """Umeyama similarity src -> dst (N x 3): returns (sR, t)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S_, Vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = np.trace(np.diag(S_) @ D) / (a ** 2).sum()
    return s * R, md - s * R @ ms


def frame_axes(P):
    """unit right (eye line) and up axes in the image, from the outer eye corners."""
    x = P[263, :2] - P[33, :2]
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.array([x[1], -x[0]])
    return x, (y if y[1] < 0 else -y)


ARC = [40, 39, 37, 0, 267, 269, 270]   # outer upper lip, centre third-to-third (no corners)


def arc_shape(P, w):
    """sagitta of the arc (centre vs its ends, along the face up axis) and near/far half-length balance, / width w."""
    x, up = frame_axes(P)
    c = P[0, :2]
    L, R = P[40, :2], P[270, :2]
    sag = ((c - (L + R) / 2) @ up) / w
    return sag, (abs((R - c) @ x) - abs((L - c) @ x)) / w


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--json')
    ap.add_argument('videos', nargs='+', help='name=path')
    a = ap.parse_args(argv)
    T, _, _ = _track(a.take, a.frames, z=True)
    op = np.linalg.norm(T[:, 13, :2] - T[:, 14, :2], axis=1) / np.linalg.norm(T[:, 10, :2] - T[:, 152, :2], axis=1)
    r = int(np.argmin(op))
    Xr = T[r]
    out = {'ref_frame': r}
    for nv in a.videos:
        name, path = nv.split('=', 1)
        S, _, found = _track(path, a.frames, z=True)
        n = len(S)
        E = {'arc': [], 'sag': [], 'asym': []}
        for i in range(n):
            sR, t = sim3(Xr[RIG], S[i][RIG])
            pred = Xr @ sR.T + t
            w = np.linalg.norm(pred[291, :2] - pred[61, :2]) + 1e-9
            E['arc'].append(float(np.linalg.norm(S[i][ARC, :2] - pred[ARC, :2], axis=1).mean() / w))
            ps, pa = arc_shape(pred, w)
            qs, qa = arc_shape(S[i], w)
            E['sag'].append(abs(qs - ps))
            E['asym'].append(abs(qa - pa))
        o = {'found': found}
        for k, v in E.items():
            v = np.array(v)
            o[k + '_med'] = float(np.median(v))
            o[k + '_p95'] = float(np.percentile(v, 95))
        o['per_frame_arc'] = [round(v, 4) for v in E['arc']]
        out[name] = o
        print(f"{name:8s} arc {o['arc_med']:.3f}/{o['arc_p95']:.3f}  sag {o['sag_med']:.3f}/{o['sag_p95']:.3f}"
              f"  asym {o['asym_med']:.3f}/{o['asym_p95']:.3f}   (median/p95, mouth widths)")
    if a.json: json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
