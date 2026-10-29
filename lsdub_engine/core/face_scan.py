#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""face_scan.py — scan a take's face: per-frame 3D face mesh + head pose, then pick the pose keyframe(s) for a
PER-SHOT mouth library (lsdub mouths prep --frame N), so the library mouths are native to the shot's own
angle, light and lens instead of a front view borrowed from another shot.

MediaPipe Face Landmarker (Apache-2.0): 478 landmarks with relative depth + the facial transformation matrix
(canonical face -> camera). No FLAME/BFM/SMPL or any other non-commercial face model.

  lsdub scan --video zoom_shot.mp4 --out scan_shot.json [--npz scan_shot.npz] [--cluster-deg 12]

Output: per-frame yaw/pitch/roll (deg), mouth open; pose clusters (greedy, --cluster-deg wide in yaw/pitch); for each
cluster the keyframe = the most-closed mouth among frames near the cluster centre (a closed mouth is the cleanest
base for Qwen to edit). The npz keeps the 3D landmarks (x*W, y*H, z*W) for lsdub paint --pose3d.
"""
import argparse, json
import numpy as np, cv2

from .._models import face_landmarker_path
MODEL = face_landmarker_path()


def detector():
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    return mp, vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.2, min_face_presence_confidence=0.2, min_tracking_confidence=0.2))


def euler(M):
    """yaw (left/right), pitch (up/down), roll in degrees from the 4x4 canonical->camera matrix."""
    R = M[:3, :3] / np.linalg.norm(M[:3, 0])
    pitch = np.degrees(np.arctan2(-R[1, 2], R[2, 2]))
    yaw = np.degrees(np.arcsin(np.clip(R[0, 2], -1, 1)))
    roll = np.degrees(np.arctan2(-R[0, 1], R[0, 0]))
    return float(yaw), float(pitch), float(roll)


def scan(video, upscale=2.0):
    mp, l = detector()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    rows, P3 = [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        H, W = fr.shape[:2]
        big = cv2.resize(fr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        r = l.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(big[:, :, ::-1])), int(i * 1000 / fps))
        if r.face_landmarks:
            p = np.array([[q.x * W, q.y * H, q.z * W] for q in r.face_landmarks[0]])
            M = np.array(r.facial_transformation_matrixes[0])
            fh = np.linalg.norm(p[10, :2] - p[152, :2])
            y, pt, ro = euler(M)
            rows.append({'f': i, 'yaw': round(y, 2), 'pitch': round(pt, 2), 'roll': round(ro, 2),
                         'open': round(float(np.linalg.norm(p[13, :2] - p[14, :2]) / fh), 4), 'face_h': round(float(fh), 1)})
            P3.append(p)
        else:
            rows.append({'f': i, 'yaw': None})
            P3.append(np.full((478, 3), np.nan))
        i += 1
    return rows, np.array(P3), fps


def clusters(rows, deg):
    ok = [r for r in rows if r['yaw'] is not None]
    pts = np.array([[r['yaw'], r['pitch']] for r in ok])
    left = list(range(len(ok)))
    out = []
    while left:   # greedy: seed at the densest remaining pose, take everything within deg
        sub = pts[left]
        d = np.linalg.norm(sub[:, None] - sub[None], axis=2)
        seed = int(np.argmax((d < deg / 2).sum(1)))
        mem = [left[j] for j in np.where(d[seed] < deg / 2)[0]]
        c = pts[mem].mean(0)
        near = [m for m in mem if np.linalg.norm(pts[m] - c) < deg / 4] or mem
        key = min(near, key=lambda m: ok[m]['open'])
        out.append({'centre': [round(float(c[0]), 2), round(float(c[1]), 2)], 'frames': [ok[m]['f'] for m in mem],
                    'keyframe': ok[key]['f'], 'key_open': ok[key]['open']})
        left = [x for x in left if x not in mem]
    return sorted(out, key=lambda c: -len(c['frames']))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--npz')
    ap.add_argument('--cluster-deg', type=float, default=12.0)
    ap.add_argument('--upscale', type=float, default=2.0)
    a = ap.parse_args(argv)
    rows, P3, fps = scan(a.video, a.upscale)
    cl = clusters(rows, a.cluster_deg)
    ok = [r for r in rows if r['yaw'] is not None]
    rng = {k: [round(min(r[k] for r in ok), 1), round(max(r[k] for r in ok), 1)] for k in ('yaw', 'pitch', 'roll')}
    json.dump({'video': a.video, 'fps': fps, 'found': len(ok) / max(1, len(rows)), 'range': rng, 'clusters': cl, 'frames': rows},
              open(a.out, 'w'), indent=1)
    if a.npz: np.savez_compressed(a.npz, lm3d=P3.astype(np.float32))
    print(f'{a.video}: {len(ok)}/{len(rows)} frames, range {rng}')
    for c in cl: print(f"  cluster yaw/pitch {c['centre']}  n={len(c['frames'])}  keyframe {c['keyframe']} (open {c['key_open']})")


if __name__ == '__main__':
    main()
