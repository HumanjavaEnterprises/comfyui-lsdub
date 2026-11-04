#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sync_truth.py: the rendered mouth read against the VOCAL ITSELF, with no alignment, no plan and no lyric in between. CPU, no models.

The sing gate's `corr` and verify's `sync corr` read the mouth against the opening built from the alignment, the curve the dub was
painted from: they say the render followed its plan and cannot see a plan that is off the song. This reads the same per-frame mouth
opening (sync_qa.measured_opening) against two curves taken from the vocal stem alone:

  loud   the vocal's level in dB under its p95, floored at max(p10, -45 dB) and clipped 0..1: a held sung note sits at the top for as long
         as it is held, so a mouth that stays open on it reads as following it (a raw RMS correlation punishes the note's swells instead)
  open   open voiced sound: voicing x (1 - the level dip / nasal reading of vocal_retime.features). It falls on closures and hums inside a
         phrase, where the level alone may not

Both are averaged over each video frame's own interval [k/fps, (k+1)/fps) and low-passed at MOUTH_HZ (zero phase): a mouth does not
follow vibrato. Four normalised cross-correlations over lags -max_lag..+max_lag frames, mouth against loud and against open, on the
levels and on the first differences (the opening and closing EVENTS: an attack, a release, a gap between words), averaged into one
curve:

  truth          the curve at lag 0: how well the mouth follows the song, on time. 0 = no relation, 1 = moves exactly with it
  truth_lag_ms   the curve's peak, parabolic sub-frame (+ = the mouth trails the vocal)
  truth_level / truth_events   the level and event halves at lag 0, reported
  truth_pass     truth >= TRUTH_MIN and -EARLY_MS <= truth_lag_ms <= LATE_MS

keep: a bool per frame (vocalisation spans a dub hands to the take): read() counts only pairs of kept frames; read_wav() always reads
every frame and reports the kept-frames reading beside it (truth_kept). A vocal with no level change to
read (silent, or a stem with no dips at all) or a shot under MIN_FRAMES reads None: the caller falls back to what it had.

  lsdub sync-truth --video render.mp4 --vocal work/vox.wav [--work work] [--warp-json work/crop_painted_warp.json] [--json out.json]
  (every frame of the video given: the delivered shot is the one to give; --work reads on the gate's box)
"""

import argparse, json, os
import numpy as np

TRUTH_MIN = 0.30  # a mouth that follows the vocal reads from about here up; one moving on its own reads near 0
TRUTH_FAIL = 0.15  # under this the mouth follows the song no better than a mouth that ignores it (verify: FAIL)
LATE_MS = 50.0  # the mouth may trail the vocal this much (about one frame at 25 fps, with the sub-frame estimate's half-frame play)
EARLY_MS = 80.0  # and lead it this much: lips set up before the sound
MAX_LAG = 8  # frames each way
MOUTH_HZ = 6.0  # the vocal curves are low-passed here: the mouth does not follow faster than this
FLOOR_DB = -45.0  # the loud curve's floor under the p95, unless the clip's p10 sits higher
MIN_FRAMES = 12
KEYS = ("truth", "truth_lag_ms", "truth_best", "truth_level", "truth_events", "truth_pass")
KEPT_KEYS = ("truth_kept", "truth_kept_lag_ms")  # the same reading on the frames a dub did not hand back to the take: reported only


def lowpass(x, rate, fc):
    """forward + backward one-pole low-pass at fc Hz (zero phase)."""
    y = np.array(x, float)
    if fc <= 0 or len(y) < 2:
        return y
    al = 1.0 - np.exp(-2 * np.pi * fc / rate)
    for i in range(1, len(y)):
        y[i] = y[i - 1] + al * (y[i] - y[i - 1])
    for i in range(len(y) - 2, -1, -1):
        y[i] = y[i + 1] + al * (y[i] - y[i + 1])
    return y


def per_frame(a, t, n, fps):
    """mean of a feature series (centres t, s) over each video frame's interval [k/fps, (k+1)/fps); a frame with no centre in it
    takes the nearest one."""
    a, t = np.asarray(a, float), np.asarray(t, float)
    k = np.floor(t * fps).astype(int)
    ok = (k >= 0) & (k < n)
    s, c = np.bincount(k[ok], a[ok], n), np.bincount(k[ok], minlength=n)
    out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    if np.isnan(out).any():
        mid = (np.arange(n) + 0.5) / fps
        j = np.clip(np.searchsorted(t, mid), 0, len(t) - 1)
        out = np.where(np.isnan(out), a[j], out)
    return out


def vocal_curves(x, n, fps):
    """(loud, open) per video frame from the vocal's samples at vocal_retime.SR, frame 0 at sample 0."""
    from ..core import vocal_retime as VR

    fx = VR.features(np.asarray(x, float))
    t = np.arange(len(fx["db"])) * VR.HOP + VR.WIN / 2
    db = fx["db"]
    floor = max(float(np.percentile(db, 10)), FLOOR_DB)
    loud = np.clip((db - floor) / max(1e-6, -floor), 0.0, 1.0)
    opn = fx["voiced"] * (1.0 - np.maximum(fx["quiet"], fx["nasal"]))
    return tuple(lowpass(per_frame(c, t, n, fps), fps, MOUTH_HZ) for c in (loud, opn))


def xcurve(a, b, max_lag, keep=None):
    """normalised cross-correlation of a against b at lags -max_lag..max_lag: lag L pairs a[i + L] with b[i] (L > 0 = a trails b). Only
    pairs of two kept frames count; a lag with fewer than MIN_FRAMES pairs, or a flat side, reads 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    keep = np.ones(n, bool) if keep is None else np.asarray(keep, bool)[:n]
    out = np.zeros(2 * max_lag + 1)
    for j, L in enumerate(range(-max_lag, max_lag + 1)):
        i = np.arange(max(0, -L), n - max(0, L))
        i = i[keep[i] & keep[i + L]]
        if len(i) < MIN_FRAMES:
            continue
        u, v = a[i + L], b[i]
        su, sv = u.std(), v.std()
        if su < 1e-9 or sv < 1e-9:
            continue
        out[j] = float(np.mean((u - u.mean()) * (v - v.mean())) / (su * sv))
    return out


PEAK_TIE = 0.02  # peaks this close to the highest are a tie (a rhythmic line repeats): the one nearest lag 0 is the lag


def peak(c):
    """(lag in frames, sub-frame by a parabola through the top three, value) of a curve over -m..m. Of the local maxima within
    PEAK_TIE of the highest, the one nearest lag 0."""
    c = np.asarray(c, float)
    m = (len(c) - 1) // 2
    top = float(np.max(c))
    tops = [j for j in range(len(c)) if c[j] >= top - PEAK_TIE and (j == 0 or c[j] >= c[j - 1]) and (j == len(c) - 1 or c[j] >= c[j + 1])]
    i = min(tops, key=lambda j: (abs(j - m), -c[j]))
    off = 0.0
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        d = y0 - 2 * y1 + y2
        off = float(np.clip(0.5 * (y0 - y2) / d, -0.5, 0.5)) if d != 0 else 0.0
    return i - m + off, float(c[i])


def read(op, x, fps, keep=None, max_lag=MAX_LAG):
    """the truth reading of a mouth opening series op (one per video frame) against the vocal samples x (vocal_retime.SR, starting at
    frame 0). -> dict of KEYS; every value None when there is nothing to read."""
    none = dict.fromkeys(KEYS)
    if op is None:
        return none
    op = np.asarray(op, float)
    n = len(op)
    if n < MIN_FRAMES or not np.isfinite(op).all():
        return none
    loud, opn = vocal_curves(x, n, fps)
    if loud.std() < 1e-3:
        return none
    k = None if keep is None else np.asarray(keep, bool)[:n]
    kd = None if k is None else k[:-1] & k[1:]
    m = min(max_lag, max(1, (n - MIN_FRAMES) // 2))
    lv = [xcurve(op, r, m, k) for r in (loud, opn)]
    ev = [xcurve(np.diff(op), np.diff(r), m, kd) for r in (loud, opn)]
    c = np.mean(lv + ev, 0)
    lag, best = peak(c)
    truth = float(c[m])
    lag_ms = 1000.0 * lag / fps
    r = dict(
        truth=round(truth, 3),
        truth_lag_ms=round(lag_ms, 1),
        truth_best=round(best, 3),
        truth_level=round(float(np.mean([x_[m] for x_ in lv])), 3),
        truth_events=round(float(np.mean([x_[m] for x_ in ev])), 3),
    )
    r["truth_pass"] = passes(r)
    return r


def passes(r):
    """the truth rule on a reading: following the song and on time. None when unread."""
    if not r or r.get("truth") is None:
        return None
    return bool(r["truth"] >= TRUTH_MIN and -EARLY_MS <= r["truth_lag_ms"] <= LATE_MS)


def why(r):
    """the reason a reading does not pass, in words; '' when it passes or is unread."""
    if not r or r.get("truth") is None or passes(r):
        return ""
    if r["truth"] < TRUTH_MIN:
        return f"mouth does not follow the vocal (audio sync {r['truth']:.2f} < {TRUTH_MIN})"
    side = "trails" if r["truth_lag_ms"] > 0 else "leads"
    return f"mouth {side} the vocal by {abs(r['truth_lag_ms']):.0f} ms"


def read_wav(op, wav, fps, keep=None):
    """read() on a vocal file, on EVERY frame: what the viewer sees, the frames a dub handed back to the take (vocalisation spans)
    included, so a shot cannot pass by handing more of itself back. keep (those spans' complement), when given, adds the same reading
    on the kept frames as truth_kept / truth_kept_lag_ms: reported, it decides nothing. No file to read: every value None."""
    import os

    from ..core.vocal_retime import read_mono

    if not wav or not os.path.exists(wav):
        return dict.fromkeys(KEYS + (KEPT_KEYS if keep is not None else ()))
    x = read_mono(wav)
    r = read(op, x, fps)
    if keep is not None:
        k = read(op, x, fps, keep)
        r.update(truth_kept=k["truth"], truth_kept_lag_ms=k["truth_lag_ms"])
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lsdub sync-truth")
    ap.add_argument("--video", help="the delivered shot (its mouth is read with sync_qa.measured_opening)")
    ap.add_argument("--open", help="a saved opening series (.npy) instead of --video")
    ap.add_argument("--vocal", required=True, help="the vocal stem, cut to start with the shot's first frame (a sing work dir's vox.wav)")
    ap.add_argument("--work", help="a sing work dir: the mouth is read on the gate's box (its medium.json), as the gate reads it")
    ap.add_argument("--warp-json", help="the shot's warp log: adds truth_kept, the reading without the frames handed to the take")
    ap.add_argument("--frames", type=int, help="read this many frames (default: all)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.open:
        op = np.load(a.open)
    else:
        from . import sync_qa as Q
        from ..core import medium as MED

        F, fps = MED.read(a.video, a.frames or 0)
        a.fps = fps or a.fps
        h, w = F[0].shape[:2]
        if a.work:
            from ..core.sing import face_roi

            roi = face_roi(json.load(open(os.path.join(a.work, "medium.json"))), F[0].shape)
        else:
            roi = [0, h, 0, w]
        op, _ = Q.measured_opening(F, roi, a.fps)
    if a.frames:
        op = op[: a.frames]
    keep = None
    if a.warp_json:
        from ..core.vocalisations import scored_frames

        keep = scored_frames(a.warp_json, len(op))[0]
        keep = np.ones(len(op), bool) if keep is None else keep
    r = read_wav(op, a.vocal, a.fps, keep)
    print(json.dumps(r))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    return 0 if r.get("truth_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
