# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_interior.py — one consistent mouth interior per shot (teeth + dark cavity), anchored to the upper lip.

A per-frame generated mouth re-invents its inside every frame. On fast moves it can paint a pale pink mass under the upper
teeth that reads as a second lower lip in motion. Real anatomy: the upper teeth are fixed to the skull and hang from
the upper lip line; below them is dark cavity and tongue. So: pick the shot's cleanest open frame (dark cavity,
bright upper teeth), and in every open frame replace the interior with that reference, aligned to the upper inner
lip (scaled by mouth width, cropped by the current inner-lip shape). Closed frames are left alone.

--skull-lock: the upper teeth are part
of the skull (maxilla), so the reference interior is placed by the RIGID skull (nose, eyes, irises: a full affine, which
carries roll and the yaw/pitch foreshortening) instead of by the deforming inner lip, and each open phrase takes the
reference frame whose head pose is closest (no teeth from a different angle).

  lsdub interior --video in.mp4 --frames N --out out.mp4 [--ref-frame K] [--strength 0.85] [--skull-lock]"""

from ..io import ff
import argparse
import cv2, numpy as np
from .mouth_paint import track

INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_IN = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]


def gap(P, t):  # inner-lip opening / mouth width
    return np.linalg.norm(P[t, 13] - P[t, 14]) / max(np.linalg.norm(P[t, 78] - P[t, 308]), 1e-3)


def inner_mask(pts, shape, shrink=0.06, blur=5):
    m = np.zeros(shape[:2], np.float32)
    c = pts.mean(0)
    q = c + (pts - c) * (1 - shrink)
    cv2.fillPoly(m, [q.astype(np.int32)], 1.0)
    k = max(3, blur | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def quality(f, P, t):
    m = inner_mask(P[t, INNER], f.shape) > 0.5
    if m.sum() < 40:
        return -1
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[m].astype(np.float32)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[m]
    pink = np.mean((hsv[:, 1] > 60) & (hsv[:, 2] > 120))  # pale pink blob share (bad)
    return (np.percentile(g, 95) - np.percentile(g, 10)) / 255 - 1.5 * pink + 0.5 * min(gap(P, t), 0.5)


SKULL = [168, 6, 197, 195, 5, 4, 1, 2, 94, 98, 327, 133, 362, 33, 263, 468, 473, 50, 280]


def skull_to_ref(P, t, r):
    """full affine mapping frame t's rigid skull points onto reference frame r's (teeth ride the skull)"""
    M, _ = cv2.estimateAffine2D(P[t, SKULL].astype(np.float32), P[r, SKULL].astype(np.float32), method=cv2.LMEDS)
    return M


def pose_dist(P, t, r):
    """residual of the best similarity fit between two frames' rigid skull points (/ inter-pupil): head-pose mismatch"""
    a, b = P[t, SKULL], P[r, SKULL]
    M, _ = cv2.estimateAffinePartial2D(a.astype(np.float32), b.astype(np.float32))
    q = a @ M[:, :2].T + M[:, 2]
    return float(np.linalg.norm(q - b, axis=1).mean() / max(np.linalg.norm(b[-4] - b[-3]), 1e-3))


def segment_refs(F, P, N, open_min, cands):
    """one reference per contiguous open run: the candidate with the closest head pose over the run"""
    ref = {}
    t = 0
    while t < N:
        if gap(P, t) < open_min:
            t += 1
            continue
        u = t
        while u < N and gap(P, u) >= open_min:
            u += 1
        best = min(cands, key=lambda c: np.mean([pose_dist(P, k, c) for k in range(t, u)]))
        for k in range(t, u):
            ref[k] = best
        t = u
    return ref


def teeth_offset_px(f, P, t):
    """lateral offset (px, along the pupil line) of the visible upper teeth's centre from the upper-lip midline (13)"""
    m = inner_mask(P[t, INNER], f.shape) > 0.5
    if m.sum() < 60:
        return None
    yy, xx = np.nonzero(m)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[yy, xx]
    tooth = (hsv[:, 1] < 80) & (hsv[:, 2] > 130)
    if tooth.sum() < 25:
        return None
    ex = P[t, 473] - P[t, 468]
    ux = ex / max(np.linalg.norm(ex), 1e-6)
    return float((np.array([xx[tooth].mean(), yy[tooth].mean()]) - P[t, 13]) @ ux), ux


def frame_to_ref(P, t, r):
    """similarity transform mapping frame t's upper inner lip onto the reference frame r's"""
    M, _ = cv2.estimateAffinePartial2D(P[t, UPPER_IN].astype(np.float32), P[r, UPPER_IN].astype(np.float32))
    return M


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-frame", type=int)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--open-min", type=float, default=0.08)
    ap.add_argument("--skull-lock", action="store_true", help="place the interior by the rigid skull + pose-matched reference")
    a = ap.parse_args(argv)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    F = []
    while len(F) < a.frames:
        ok, f = cap.read()
        if not ok:
            break
        F.append(f)
    P, _, _ = track(a.video, len(F))
    N = len(F)
    r = a.ref_frame if a.ref_frame is not None else max(range(N), key=lambda t: quality(F[t], P, t) if gap(P, t) > 0.25 else -9)
    H, W = F[r].shape[:2]
    out = []
    touched = 0
    refs = {}
    if a.skull_lock and a.ref_frame is None:
        scored = sorted((t for t in range(N) if gap(P, t) > 0.25), key=lambda t: -quality(F[t], P, t))
        cands = [c for c in scored if quality(F[c], P, c) >= quality(F[r], P, r) - 0.15][:6] or [r]
        refs = segment_refs(F, P, N, a.open_min, cands)
    for t in range(N):
        g = gap(P, t)
        if g < a.open_min:
            out.append(F[t])
            continue
        rr = refs.get(t, r)
        R = F[rr]
        M = skull_to_ref(P, t, rr) if a.skull_lock else frame_to_ref(P, t, rr)
        if M is None:
            M = frame_to_ref(P, t, rr)
        Minv = cv2.invertAffineTransform(M)
        if a.skull_lock:
            # teeth sit under the philtrum: cancel the reference's own lateral teeth offset (clamped to ~2 mm)
            to = teeth_offset_px(R, P, rr)
            if to is not None:
                off, uxr = to
                ipd = np.linalg.norm(P[t, 473] - P[t, 468])
                lim = ipd * 2.0 / 62.0
                ux_t = P[t, 473] - P[t, 468]
                ux_t = ux_t / max(np.linalg.norm(ux_t), 1e-6)
                sc = np.linalg.norm(Minv[:, 0])
                sh = -np.clip(off * sc, -lim, lim) * ux_t
                Minv = Minv.copy()
                Minv[:, 2] += sh
        ref_in_t = cv2.warpAffine(R, Minv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = inner_mask(P[t, INNER], F[t].shape, blur=max(3, int(np.linalg.norm(P[t, 78] - P[t, 308]) * 0.08)))
        # fade in with opening so a barely-open mouth isn't stamped
        w = (m * a.strength * np.clip((g - a.open_min) / 0.08, 0, 1))[..., None]
        # match brightness of the reference interior to this frame's lip light
        o = F[t].astype(np.float32) * (1 - w) + ref_in_t.astype(np.float32) * w
        out.append(np.clip(o, 0, 255).astype(np.uint8))
        touched += 1
    p = ff.popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            a.video,
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            a.out,
        ],
        stdin=ff.PIPE,
    )
    for o in out:
        p.stdin.write(o.tobytes())
    p.stdin.close()
    p.wait()
    print(
        f"interior ref frame {r}{' (skull-lock, refs ' + str(sorted(set(refs.values()))) + ')' if refs else ''}, replaced in {touched}/{N} frames"
    )


if __name__ == "__main__":
    main()
