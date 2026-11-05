# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""aperture.py: how far the mouth opens. One stage of the drive chain, behind one paint flag, --warp-aperture (off | map | loop).

  drive (the track's per-frame `open`) -> [map] -> aperture target -> render -> [loop] -> output

map   the drive's own range per shot: its q-th percentile becomes a full opening (1.0 of the take's range, g_lo..g_hi), the rest
      scaled with it and clipped to 0..1. A drive that never opens past 0.6 then reaches the most this face opens in its own take, and
      a shot keeps its singer's travel instead of 60 % of it. Only ever a gain >= 1: a drive already reaching full opening is unchanged.
loop  map, then the rendered mouth measured and corrected: per frame, the opening the warp achieved on the rendered pixels (inner-lip
      gap / eye distance, re-detected) against the opening it was asked for before its fold guard, and the frame rendered again with
      the residual scaled toward the request, never past FOLD_THR's fold limit (the warp's own Jacobian check) and never past HI x the
      request. Frames the vocalisation pass hands back to the take are not touched.

The pure parts (mapped, loop_residual, fold_limited) take arrays and return arrays; close_loop is the glue that measures a WarpClip's
frames and re-renders them. Nothing here reads the vocal, the alignment or another stage's output: its inputs are the track frames, the
warp's targets and the rendered pixels."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

MODES = ("off", "map", "loop")
Q = 95.0  # the drive's percentile that becomes a full opening
FLOOR = 0.2  # a drive whose q-th percentile is under this is not scaled past 1 / FLOOR (a near-silent shot is not blown up)
LO, HI = 0.8, 1.4  # the applied residual stays within this x the requested one
TOL = 0.1  # a frame within this share of its request is left as rendered
MIN_REQ = 0.12  # a request under this share of the take's range is not a move to correct (the warp leaves the take there)
SIGMA = 2.0  # frames: zero-phase ease of the correction, so neighbours do not step against each other
MEAS_SIGMA = 2.0  # frames: zero-phase ease of the re-detected openings: a single frame's landmark noise is not a move to correct
ITERS = 2


# ------------------------------------------------------------------------------------------------------------------------- map --
def range_gain(opens, q=Q, floor=FLOOR):
    """the gain that sets the drive's q-th percentile to 1.0, never under 1 and never over 1 / floor."""
    o = np.asarray(opens, float)
    if not len(o):
        return 1.0
    return float(np.clip(1.0 / max(float(np.percentile(o, q)), floor), 1.0, 1.0 / floor))


def mapped(frames, q=Q, floor=FLOOR):
    """track frames (dicts with 'open') -> (new frames with 'open' scaled by range_gain and clipped to 0..1, the gain). The input is not
    changed; every other key is kept."""
    k = range_gain([f.get("open", 0.0) for f in frames], q, floor)
    return [dict(f, open=round(float(np.clip(float(f.get("open", 0.0)) * k, 0.0, 1.0)), 3)) for f in frames], k


# ------------------------------------------------------------------------------------------------------------------------ loop --
def loop_residual(req, ach, applied, valid, lo=LO, hi=HI, tol=TOL, sigma=SIGMA):
    """per frame: the residual to apply next (canonical gap units). req = the opening asked of the warp, ach = what the rendered pixels
    show, applied = what was applied to get them, all as residuals against the take. On a valid frame whose achieved move has the
    request's sign and is off by more than tol, the applied residual is scaled by req / ach, held within lo..hi x req; elsewhere it
    stays. The change is eased zero-phase over sigma frames (only valid frames move)."""
    req, ach, applied = (np.asarray(v, float) for v in (req, ach, applied))
    valid = np.asarray(valid, bool) & np.isfinite(ach) & (np.abs(req) > 1e-9)
    same = valid & (np.sign(ach) == np.sign(req)) & (np.abs(ach) > 0.25 * np.abs(req))
    off = same & (np.abs(ach / np.where(same, req, 1.0) - 1.0) > tol)
    want = applied.copy()
    want[off] = applied[off] * req[off] / ach[off]
    a, b = np.minimum(lo * req, hi * req), np.maximum(lo * req, hi * req)
    want = np.where(off, np.clip(want, a, b), applied)
    d = want - applied
    if sigma > 0 and off.any():
        d = gaussian_filter1d(d, sigma, mode="nearest") * valid
    return applied + d


def steady(meas, sigma=MEAS_SIGMA):
    """a re-detected opening series eased zero-phase over sigma frames, its unread (NaN) frames filled from their neighbours for the
    ease only and NaN again after it."""
    m = np.asarray(meas, float)
    ok = np.isfinite(m)
    if ok.sum() < 2 or sigma <= 0:
        return m.copy()
    f = np.interp(np.arange(len(m)), np.flatnonzero(ok), m[ok])
    return np.where(ok, gaussian_filter1d(f, sigma, mode="nearest"), np.nan)


def fold_limited(det_at, base, want, thr, iters=4):
    """the share s in [0, 1] of the step base -> want that keeps det_at(base + s (want - base)) >= thr, by bisection (1.0 when the whole
    step is clear). det_at: a residual -> the warp's minimum Jacobian determinant."""
    if det_at(want) >= thr:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if det_at(base + m * (want - base)) >= thr:
            lo = m
        else:
            hi = m
    return lo


def measure(frames, which=None):
    """per frame: inner-lip gap (13-14) / eye distance (33-263) re-detected on the pixels (MediaPipe, image mode); NaN with no face, and
    on the frames not in `which` (indices; None = all)."""
    import cv2

    from . import landmarker

    mp = landmarker.mp()
    det = landmarker.get("image", conf=0.2)
    out = np.full(len(frames), np.nan)
    for i in range(len(frames)) if which is None else which:
        f = frames[i]
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
        if r.face_landmarks:
            L = r.face_landmarks[0]
            p = lambda j: np.array([L[j].x * f.shape[1], L[j].y * f.shape[0]])
            out[i] = np.linalg.norm(p(13) - p(14)) / (np.linalg.norm(p(33) - p(263)) + 1e-9)
    return out


def calibration(take_meas, gap):
    """canonical gap -> gap / eye distance on the pixels: the median ratio over the take's own more-open frames (its closed ones are noise)."""
    m, g = np.asarray(take_meas, float), np.asarray(gap, float)
    ok = np.isfinite(m) & (g > np.percentile(g, 30)) & (g > 1e-6)
    return float(np.median(m[ok] / g[ok])) if ok.sum() >= 5 else None


def close_loop(clip, iters=ITERS, thr=None, log=print):
    """the loop on a rendered WarpClip (after run_batch and the vocalisation pass): measure, correct, render again. -> a summary dict
    (also under 'aperture' in each touched frame's log entry)."""
    n, F, rng = clip.n, clip.F, clip.model.rng
    scale = max(1e-6, rng["g_hi"] - rng["g_lo"])
    thr = float(clip.a.warp_fold_guard) if thr is None else thr
    keep = np.ones(n) if getattr(clip, "vocal_s", None) is None else np.asarray(clip.vocal_s[:n], float)
    import time

    t_ = time.perf_counter()
    req = clip.dFe_pre_fold[:n, 0] * keep  # what the warp was asked to open before its fold guard
    asked = np.flatnonzero((keep >= 1.0) & (np.abs(req) >= MIN_REQ * scale))
    cal_at = np.flatnonzero(F[:n, 0] > np.percentile(F[:n, 0], 30))[::3]  # a third of the take's open frames calibrate
    take = measure(clip.frames[:n], sorted(set(asked.tolist()) | set(cal_at.tolist())))
    cal = calibration(take, F[:n, 0])
    summary = {"iters": 0, "cal": cal, "rerendered": 0}
    if cal is None:
        log("aperture loop: the take's mouth was not read; left as rendered")
        return summary
    take_s = steady(take)
    valid = np.zeros(n, bool)
    valid[asked] = True
    valid &= np.isfinite(take)
    ratio = lambda ach: float(np.median(ach[valid] / req[valid])) if valid.any() else None
    tm = {"measure": 0.0, "fold": 0.0, "render": 0.0}
    for it in range(iters):
        t1 = time.perf_counter()
        ach = (steady(measure(clip.outs[:n], np.flatnonzero(valid))) - take_s) / cal
        tm["measure"] += time.perf_counter() - t1
        if it == 0:
            summary["ratio_before"] = ratio(ach)
        applied = clip.dFe[:n, 0].copy()
        want = loop_residual(req, ach, applied, valid)
        moved = np.flatnonzero(np.abs(want - applied) > 0.02 * scale)
        for i in moved:
            base = clip.dFe[i].copy()
            new = base.copy()
            new[0] = want[i]
            if abs(new[0]) > abs(base[0]) and thr > 0:
                t1 = time.perf_counter()
                s = fold_limited(lambda v, i=i: clip.fold_det_of(i, v), base, new, thr)
                new = base + s * (new - base)
                tm["fold"] += time.perf_counter() - t1
            if np.abs(new - base).max() < 1e-6:
                continue
            t1 = time.perf_counter()
            out, mask, ent, P_out = clip.render_frame(int(i), {"dFe": new})
            tm["render"] += time.perf_counter() - t1
            ent["aperture"] = dict(req=round(float(req[i]), 2), ach=round(float(ach[i]), 2), applied=round(float(new[0]), 2))
            clip.outs[i], clip.masks[i], clip.P_out[i] = out, mask, P_out
            clip.log[i].update(ent)
            clip.dFe[i] = new
            clip.Fe[i] = F[i] + new
            summary["rerendered"] += 1
        summary["iters"] = it + 1
        if not len(moved):
            break
    summary["ratio_after"] = ratio((steady(measure(clip.outs[:n], np.flatnonzero(valid))) - take_s) / cal)
    summary["frames_valid"] = int(valid.sum())
    summary["seconds"] = dict({k: round(v, 1) for k, v in tm.items()}, total=round(time.perf_counter() - t_, 1))
    log(f"aperture loop: achieved / asked {summary.get('ratio_before')} -> {summary['ratio_after']} on {int(valid.sum())} frames; "
        f"{summary['rerendered']} renders over {summary['iters']} rounds")  # fmt: skip
    return summary
