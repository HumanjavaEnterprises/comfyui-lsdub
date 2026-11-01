#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""snap.py: how SNAPPY a mouth is, measured on the delivered pixels (a per-frame opening series, e.g.
sync_qa.measured_opening). CPU, no models. Three numbers:

  rise_ms         per vocal onset (the dry vocal's RMS crossing up out of silence): the time from the onset until
                  the mouth has made half of its local rise (the opening between the onset and +300 ms); the median.
                  Small = a snappy attack; negative = the mouth leads the sound (anticipation, fine down to ~-80 ms)
  closure_hit     share of p / b / m phones (from the alignment) where the mouth actually closes: within +-1 frame of
                  the phone's middle the opening falls into the bottom CLOSE_FRAC of the shot's p5..p95 range
  open_in_silence the mean opening (p10..p90 scale) over silent frames (RMS < SIL_RMS, +-1 frame): the mouth should rest
                  (the round-4 gate's definition: <= 0.35 passes)

  python -m lsdub_engine.audits.snap --open open.npy --vox vox.wav --align align.json --t0 64.3 [--fps 25]
"""

from ..io import ff
import argparse, json
import numpy as np

SR = 16000
SIL_RMS = 0.02
CLOSE_FRAC = 0.25
RISE_WIN = 0.30
CLOSERS = {"P", "B", "M"}
CLOSED_IOD = 0.06  # lips shut: the inner gap under this share of the eye distance. Shut lips read 0.05 on the tracker, a slit about 0.09


def envelope(wav, n, fps=25.0):
    a = np.frombuffer(
        ff.run(
            ["ffmpeg", "-v", "error", "-i", wav, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True
        ).stdout,
        np.float32,
    )
    hop = int(SR / fps)
    return np.array([np.sqrt(np.mean(a[i * hop : (i + 1) * hop] ** 2)) if i * hop < len(a) else 0.0 for i in range(n)])


def norm(op, lo=5, hi=95):
    a, b = np.nanpercentile(op, lo), np.nanpercentile(op, hi)
    return (op - a) / (b - a + 1e-9)


def onsets(env, thr=SIL_RMS, min_sil=2):
    """frames where the vocal comes out of >= min_sil silent frames."""
    s = env < thr
    out = []
    for i in range(min_sil, len(env)):
        if not s[i] and s[i - min_sil : i].all():
            out.append(i)
    return out


def rise_ms(op, env, fps=25.0, win=RISE_WIN):
    """median ms from each vocal onset to half the local rise; None without onsets."""
    x = norm(op)
    w = max(2, int(round(win * fps)))
    R = []
    for o in onsets(env):
        a = max(0, o - 3)
        seg = x[a : min(len(x), o + w + 1)]
        if len(seg) < 3:
            continue
        base = float(np.min(x[a : o + 1]))
        peak = float(np.max(x[o : min(len(x), o + w + 1)]))
        if peak - base < 0.15:
            continue  # the mouth does not open for it: no rise to time
        half = base + 0.5 * (peak - base)
        k = next(i for i in range(a, min(len(x), o + w + 1)) if x[i] >= half)
        R.append((k - o) * 1000.0 / fps)
    return float(np.median(R)) if R else None, len(R)


def closures(words, t0, n, fps=25.0):
    """frame index of the middle of every p / b / m phone inside the clip."""
    out = []
    for w in words:
        for p in w.get("phones", []):
            if p["p"].upper().rstrip("012") in CLOSERS:
                f = int(round(((p["start"] + p["end"]) / 2 - t0) * fps))
                if 0 <= f < n:
                    out.append(f)
    return out


def closure_hit(op, frames, frac=CLOSE_FRAC):
    x = norm(op)
    hits = [bool(np.min(x[max(0, f - 1) : f + 2]) <= frac) for f in frames]
    return (float(np.mean(hits)) if hits else None), len(hits)


def closure_real(gap_iod, frames, thr=CLOSED_IOD):
    """share of p / b / m phones where the lips are shut in absolute terms: within +-1 frame of the phone's middle the gap, in eye
    distances, falls under thr. closure_hit reads the shot's own range, so a slit passes it on an open-mouthed shot."""
    g = np.asarray(gap_iod, float)
    hits = [bool(np.nanmin(g[max(0, f - 1) : f + 2]) <= thr) for f in frames if np.isfinite(g[max(0, f - 1) : f + 2]).any()]
    return (float(np.mean(hits)) if hits else None), len(hits)


def class_contrast(gap, classes, opens):
    """how far the delivered height follows the card, class by class. classes: the plan's viseme per frame ("" = out); opens: {class:
    the card's opening}. -> (slope of gap / the aa median against the card's opening / aa's, r, {class: median gap}); (None, None, {})
    without an aa or a second class."""
    g, c = np.asarray(gap, float), np.asarray(classes, object)
    m = np.isfinite(g) & np.array([x in opens for x in c], bool)
    aa = m & (c == "aa")
    if aa.sum() < 3 or len(set(c[m])) < 2 or np.median(g[aa]) <= 0:
        return None, None, {}
    x = np.array([opens[v] for v in c[m]], float) / opens["aa"]
    y = g[m] / np.median(g[aa])
    med = {v: round(float(np.median(g[m & (c == v)])), 4) for v in sorted(set(c[m]))}
    return round(float(np.polyfit(x, y, 1)[0]), 3), round(float(np.corrcoef(x, y)[0, 1]), 3), med


def open_in_silence(op, env, thr=SIL_RMS):
    sil = np.convolve((env < thr).astype(float), np.ones(3), "same") >= 3
    if sil.sum() < 3:
        return None
    lo, hi = np.percentile(op, 10), np.percentile(op, 90)
    return float(np.mean((op[sil] - lo) / (hi - lo + 1e-9)))


def snap_metrics(op, vox, words, t0, fps=25.0, dt=None):
    """dt: per frame, s the plan was moved earlier (coart.logged_offsets), words already moved with it (coart.logged_words). Silence
    is then read where the mouth was planned to meet it; rise_ms stays the raw delay against the sound."""
    op = np.asarray(op, float)
    env = envelope(vox, len(op), fps)
    r, nr = rise_ms(op, env, fps)
    h, nh = closure_hit(op, closures(words, t0, len(op), fps))
    k = np.arange(len(op))
    s = open_in_silence(op, env if dt is None else np.interp(k + np.asarray(dt) * fps, k, env))
    return dict(
        rise_ms=None if r is None else round(r),
        onsets=nr,
        closure_hit=None if h is None else round(h, 3),
        closures=nh,
        open_in_silence=None if s is None else round(s, 3),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lsdub snap")
    ap.add_argument("--open", required=True, help=".npy per-frame opening")
    ap.add_argument("--vox", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args(argv)
    print(json.dumps(snap_metrics(np.load(a.open), a.vox, json.load(open(a.align))["words"], a.t0, a.fps)))


if __name__ == "__main__":
    main()
