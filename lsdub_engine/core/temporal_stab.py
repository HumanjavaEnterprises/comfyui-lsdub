# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""temporal_stab.py — calm the mouth "boil" of a lip-sync render (texture re-invented every frame).

The mouth of a painted frame is generated per frame, so its fine texture (lip lines, teeth edges, grain)
changes even where the shape barely moves: stills look fine, motion looks suspect. This pass carries the previous
stabilized frame forward along optical flow and blends it into the current one inside the mouth mask:
  - two bands: fine detail (where boil lives) is carried strongly, coarse shape only lightly, so the mouth still
    opens and closes on time (no lag, no ghost)
  - confidence = forward/backward flow agreement x photometric match; fast moves and new content (teeth appearing,
    lips parting) get low confidence and pass through untouched, so nothing smears across a real change
  - a shot-cut / big-move guard resets the history

  lsdub calm --video in.mp4 --frames N --out out.mp4 [--hf 0.65 --lf 0.25 --ref take.mp4]"""

from ..io import ff
import argparse
import cv2, numpy as np

from .mouth_paint import track

MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def flow(a, b):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 21, 4, 5, 1.2, 0)


def warp(img, fl):
    h, w = fl.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def stabilize(F, P, hf=0.65, lf=0.25, sig=3.0, grow=0.45):
    out = [F[0].copy()]
    H, W = F[0].shape[:2]
    conf_log = []
    for t in range(1, len(F)):
        pts = P[t, MOUTH]
        c = pts.mean(0)
        r = np.linalg.norm(P[t, 61] - P[t, 291])
        x0, y0 = int(max(c[0] - 1.6 * r, 0)), int(max(c[1] - 1.4 * r, 0))
        x1, y1 = int(min(c[0] + 1.6 * r, W)), int(min(c[1] + 1.6 * r, H))
        cur, prv_raw, prv = F[t][y0:y1, x0:x1], F[t - 1][y0:y1, x0:x1], out[t - 1][y0:y1, x0:x1]
        fb = flow(cur, prv_raw)  # where each current pixel came from in the previous frame
        bf = flow(prv_raw, cur)
        cyc = np.linalg.norm(fb + warp(bf, fb), axis=2)  # forward-backward disagreement, px
        wp = warp(prv, fb).astype(np.float32)
        cf = cur.astype(np.float32)
        photo = np.abs(warp(prv_raw, fb).astype(np.float32) - cf).mean(2)  # raw-to-raw match (content change)
        conf = np.exp(-((cyc / 1.0) ** 2)) * np.exp(-((photo / 14.0) ** 2))
        m = np.zeros(cur.shape[:2], np.float32)
        q = (pts - [x0, y0] - (pts.mean(0) - [x0, y0])) * (1 + grow) + (pts.mean(0) - [x0, y0])
        cv2.fillConvexPoly(m, cv2.convexHull(q.astype(np.int32)), 1.0)
        k = max(3, int(r * 0.35) | 1)
        m = cv2.GaussianBlur(m, (k, k), 0)
        if np.median(np.linalg.norm(fb, axis=2)[m > 0.5]) > 0.25 * r:  # big move: reset history
            out.append(F[t].copy())
            conf_log.append(0.0)
            continue
        lo_c, lo_w = cv2.GaussianBlur(cf, (0, 0), sig), cv2.GaussianBlur(wp, (0, 0), sig)
        a_hf, a_lf = (m * conf * hf)[..., None], (m * conf * lf)[..., None]
        res = lo_c + a_lf * (lo_w - lo_c) + (cf - lo_c) + a_hf * ((wp - lo_w) - (cf - lo_c))
        o = F[t].copy()
        o[y0:y1, x0:x1] = np.clip(res, 0, 255).astype(np.uint8)
        out.append(o)
        conf_log.append(float((conf * m).sum() / max(m.sum(), 1)))
    return out, conf_log


def stabilize_bidir(F, P, hf=0.65, lf=0.25):
    """zero-phase calm: the carry run forward and on the reversed clip, averaged, so no edge is pulled late (forward
    only) or early (backward only) — the causal pass alone can trail a fast open/close by a fraction of a frame."""
    fw, cf = stabilize(F, P, hf, lf)
    bw, cb = stabilize(F[::-1], P[::-1], hf, lf)
    out = [((a.astype(np.float32) + b.astype(np.float32)) / 2).astype(np.uint8) for a, b in zip(fw, bw[::-1])]
    return out, [(x + y) / 2 for x, y in zip(cf, cb[::-1][1:])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf", type=float, default=0.65)
    ap.add_argument("--lf", type=float, default=0.25)
    ap.add_argument("--ref", help="track landmarks on this video instead (e.g. the take) for steadier masks")
    ap.add_argument("--bidir", action="store_true", help="zero-phase: forward + backward carry averaged (no lag)")
    a = ap.parse_args(argv)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    F = []
    while len(F) < a.frames:
        ok, f = cap.read()
        if not ok:
            break
        F.append(f)
    P, _, _ = track(a.ref or a.video, len(F))
    out, cl = (stabilize_bidir if a.bidir else stabilize)(F, P, a.hf, a.lf)
    H, W = F[0].shape[:2]
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
    print(f"stabilized {len(out)} frames, mean mouth confidence {np.mean(cl):.2f}, resets {sum(1 for c in cl if c == 0)}")


if __name__ == "__main__":
    main()
