#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""event_judge.py: the rendered mouth read against the EVENTS of the vocal, one by one. CPU.

sync_truth reads a mouth against the vocal's envelope at 6 Hz; a mouth that rises once per phrase can pass it. This reads the same
per-frame mouth opening against the sung syllable onsets, the lyric's p/b/m closures and the wide-vowel nuclei, all taken off the
vocal stem by the judge's own front end (judge_phones), never off the dub's plan:

  onset_hit     fraction of the judge's onsets with a mouth RISE within TOL (one rise serves one onset): the first difference of the
                opening, centred, at least RISE of the shot's p5..p95 range and a local maximum
  closure_hit   fraction of the lyric's P/B/M onsets with a SHUT frame within TOL: the inner-lip gap at or under SHUT_IOD eye
                distances (lip contact), an absolute line, the same on the render and the take
  spurious      shut runs that start during sung time, clear of the clip's first and last EDGE_S, with no closure within SPUR of their
                start; spurious_take the same on the take
  open_ratio    like for like: the median of the render's peak opening in the PEAK window after each wide-vowel onset (a wide vowel
                that carries a confirmed onset) / the median of the take's peak opening in the same window after each of the take's
                own rises (what a singer scores against themself is ~1); both series in eye distances
  pass          onset_hit >= ONSET_MIN, closure_hit >= CLOSURE_MIN when closure_n >= CLOSURE_N, spurious <= spurious_take when the take
                was read, open_ratio >= OPEN_MIN when read

Without the recogniser (no weights, no torch) the onsets are the flux detector's and there are no closures and no open_ratio: the
reading is still a reading, with fewer parts. With the lyric, the onset set is lyric-confirmed (judge_phones.confirmed).

  lsdub event-judge --video render.mp4 --vocal vox.wav [--lyrics lyrics.txt] [--stt stt.json] [--take take.mp4] [--work work]
                    [--json out.json] [--strip out.png]
  (the vocal cut to start with the shot's first frame, as a sing work dir's vox.wav is)
"""

import argparse
import json
import os

import numpy as np

from . import judge_phones as JP
from . import judge_vocal as JV

TOL = 0.06  # s each way: an onset's rise, a closure's shut
SPUR = 0.12  # s: a shut run with no closure this near its start is spurious
RISE = 0.10  # of the shot's p5..p95 range per two frames
SHUT_IOD = 0.03  # inner-lip gap in eye distances at or under which the lips are shut (lip contact); the one shut line, absolute
PEAK = (-1.0, 2.0)  # a peak is read from TOL before an event to 2 x TOL after it (the apex follows the rise)
ONSET_MIN, CLOSURE_MIN, OPEN_MIN, CLOSURE_N = 0.90, 0.85, 0.85, 2
PASS = dict(onset_hit=ONSET_MIN, closure_hit=CLOSURE_MIN, open_ratio=OPEN_MIN, closure_n=CLOSURE_N)
MIN_FRAMES = 12
KEYS = ("onset_hit", "onset_n", "closure_hit", "closure_n", "spurious", "spurious_take", "open_ratio", "open_n", "pass")


def centres(n, fps):
    return (np.arange(n) + 0.5) / fps


def rises(op, fps):
    """times (s) of the mouth's opening rises."""
    op = np.asarray(op, float)
    if len(op) < 3:
        return []
    lo, hi = np.percentile(op, 5), np.percentile(op, 95)
    d = np.zeros(len(op))
    d[1:-1] = op[2:] - op[:-2]
    peak = (d >= RISE * max(hi - lo, 1e-9)) & (d >= np.roll(d, 1)) & (d > np.roll(d, -1))  # a tie breaks to the earlier frame
    peak[0] = peak[-1] = False
    return [round(float(t), 3) for t in centres(len(op), fps)[peak]]


def shut_frames(op, line=SHUT_IOD):
    return np.asarray(op, float) <= line


def shut_runs(op, fps, line=SHUT_IOD):
    """start times (s) of the runs of shut frames."""
    m = shut_frames(op, line)
    starts = np.where(m & ~np.concatenate([[False], m[:-1]]))[0]
    return [round(float(t), 3) for t in centres(len(op), fps)[starts]]


def match(events, marks, tol):
    """one-to-one nearest matching of events to marks within tol -> (hits, signed errors s)."""
    used, hits, errs = set(), 0, []
    for e in events:
        best = None
        for i, m in enumerate(marks):
            if i in used or abs(m - e) > tol:
                continue
            if best is None or abs(m - e) < abs(marks[best] - e):
                best = i
        if best is not None:
            used.add(best)
            hits += 1
            errs.append(marks[best] - e)
    return hits, errs


def is_sung(sung, t):
    """sung: a bool per judge_phones.HOP row, or None (everything is sung)."""
    if sung is None:
        return True
    i = int(t / JP.HOP)
    return bool(sung[min(max(i, 0), len(sung) - 1)]) if len(sung) else True


def peaks_after(op, t, events):
    """the series' peak in the PEAK window after each event (s), one value per event that has frames in its window."""
    out = []
    for e in events:
        m = (t >= e + PEAK[0] * TOL) & (t <= e + PEAK[1] * TOL)
        if m.any():
            out.append(float(op[m].max()))
    return out


def score(op, fps, onsets, closures=(), sung=None, wide=None, take_op=None, line=SHUT_IOD):
    """the reading of an opening series op (one per frame, frame 0 at t = 0, in eye distances) against event times (s). -> dict of KEYS
    plus `events`. line: the shut gap (SHUT_IOD; a series in another unit passes its own)."""
    none = dict.fromkeys(KEYS)
    if op is None:
        return none
    op = np.asarray(op, float)
    if len(op) < MIN_FRAMES or not np.isfinite(op).all():
        return none
    onsets, closures = [float(t) for t in onsets], [float(t) for t in closures]
    R = rises(op, fps)
    hits, errs = match(onsets, R, TOL)
    r = dict(onset_hit=round(hits / len(onsets), 3) if onsets else None, onset_n=len(onsets))
    r["rise_n"] = len(R)
    r["onset_err_ms"] = round(1000 * float(np.median(errs)), 1) if errs else None
    t = centres(len(op), fps)
    shut = shut_frames(op, line)
    ch = [any(shut & (np.abs(t - c) <= TOL)) for c in closures]
    r["closure_hit"] = round(sum(ch) / len(closures), 3) if closures else None
    r["closure_n"] = len(closures)
    r["spurious"] = spurious(op, fps, closures, sung, line)
    r["spurious_take"] = None
    r["open_ratio"], r["open_n"], r["take_peak"] = None, 0, None
    tp = None
    if take_op is not None:
        take_op = np.asarray(take_op, float)
        if len(take_op) >= MIN_FRAMES and np.isfinite(take_op).all():
            r["spurious_take"] = spurious(take_op, fps, closures, sung, line)
            tk = peaks_after(take_op, centres(len(take_op), fps), rises(take_op, fps))
            tp = float(np.median(tk)) if tk else None
            r["take_peak"] = round(tp, 4) if tp else None
    if wide and tp:
        peaks = peaks_after(op, t, wide)
        if peaks:
            r["open_ratio"], r["open_n"] = round(float(np.median(peaks)) / tp, 3), len(peaks)
    r["events"] = dict(
        onsets=onsets,
        closures=closures,
        rises=R,
        shuts=shut_runs(op, fps, line),
        spurious=spurious_at(op, fps, closures, sung, line),
        wide=list(wide or []),
        hits=[bool(c) for c in ch],
    )
    r["pass"] = passes(r)
    return r


def spurious_at(op, fps, closures, sung, line=SHUT_IOD):
    """start times of the shut runs in sung time, inside the clip's edges (judge_phones.EDGE_S: the cut is not a closure), with no
    closure near."""
    runs = JP.edge(shut_runs(op, fps, line), len(op) / fps)
    return [s for s in runs if is_sung(sung, s) and all(abs(s - c) > SPUR for c in closures)]


def spurious(op, fps, closures, sung, line=SHUT_IOD):
    return len(spurious_at(op, fps, closures, sung, line))


def passes(r):
    """the rule on a reading. None when there is no onset reading."""
    if not r or r.get("onset_hit") is None:
        return None
    ok = r["onset_hit"] >= ONSET_MIN
    if r.get("closure_n", 0) >= CLOSURE_N and r.get("closure_hit") is not None:
        ok = ok and r["closure_hit"] >= CLOSURE_MIN
    if r.get("spurious_take") is not None:
        ok = ok and r["spurious"] <= r["spurious_take"]
    if r.get("open_ratio") is not None:
        ok = ok and r["open_ratio"] >= OPEN_MIN
    return bool(ok)


def why(r):
    """the reasons a reading does not pass, in words; '' when it passes or is unread."""
    if not r or r.get("onset_hit") is None or passes(r):
        return ""
    out = []
    if r["onset_hit"] < ONSET_MIN:
        out.append(f"mouth misses sung onsets ({r['onset_hit']:.2f} of {r['onset_n']} hit, need {ONSET_MIN:.2f})")
    if r.get("closure_n", 0) >= CLOSURE_N and r.get("closure_hit") is not None and r["closure_hit"] < CLOSURE_MIN:
        out.append(f"lips miss p/b/m closures ({r['closure_hit']:.2f} of {r['closure_n']} shut, need {CLOSURE_MIN:.2f})")
    if r.get("spurious_take") is not None and r["spurious"] > r["spurious_take"]:
        out.append(f"mouth shuts where the singer does not ({r['spurious']} vs {r['spurious_take']} on the take)")
    if r.get("open_ratio") is not None and r["open_ratio"] < OPEN_MIN:
        out.append(f"vowels under-open ({r['open_ratio']:.2f} of the take's, need {OPEN_MIN:.2f})")
    return "; ".join(out)


# -- events off the vocal ---------------------------------------------------------------------------------------------------------------
def events(x, phones=None, lyric_words=None):
    """every event the judge reads off vocal samples x (judge_phones.SR, t = 0 at sample 0): onsets, closures, wide, sung, words."""
    x = np.asarray(x, float)
    logp = JP.logposteriors(x, phones) if phones is not None and len(x) >= JP.SR // 10 else None
    words = JP.align_lyric(logp, lyric_words, phones) if logp is not None and lyric_words else []
    dur = len(x) / JP.SR
    flux = JP.edge(JP.flux_onsets(x), dur)
    dets = JP.edge(JP.onsets(x, logp, phones), dur)
    onsets = JP.confirmed(dets, words) if words else dets
    wide = JP.wide(words) if words else [t for t, lab in nucleus_labels(logp, x, phones) if lab[:1] in JP.IPA_WIDE]
    return dict(
        onsets=onsets,
        detected=dets,
        flux=flux,
        closures=JP.edge(JP.closures(words), dur),
        wide=[o for o in onsets if any(abs(o - w) <= TOL for w in wide)],  # the wide vowels that carry a confirmed onset, at the onset
        sung=JP.sung_mask(x),
        words=words,
    )


def nucleus_labels(logp, x, phones):
    """(nucleus time, its vowel label) for the recogniser's nuclei; [] without the recogniser."""
    if logp is None or phones is None:
        return []
    nuc = set(JP.nuclei(logp, x, phones))
    return [(a, s) for s, a, b in JP.segments(logp, phones) if round(float(a), 3) in nuc]


def cut(ev, t0, t1):
    """the events of a window [t0, t1) of a longer stem, re-timed to t0 = 0; the window's own edges (judge_phones.EDGE_S) excluded, as
    a clip's are: no mouth can rise at a cut."""
    out = {}
    for k in ("onsets", "detected", "flux", "closures", "wide"):
        out[k] = [round(t - t0, 3) for t in ev.get(k, []) if t0 + JP.EDGE_S <= t <= t1 - JP.EDGE_S]
    s = ev.get("sung")
    out["sung"] = None if s is None else np.asarray(s)[int(t0 / JP.HOP) : int(t1 / JP.HOP)]
    out["words"] = [w for w in ev.get("words", []) if t0 <= w["start"] < t1]
    return out


def read(op, x, fps, lyric_words=None, take_op=None, phones=None):
    """score() on events taken off x. -> dict of KEYS, events, and the aligned words."""
    ev = events(x, phones, lyric_words)
    r = score(op, fps, ev["onsets"], ev["closures"], ev["sung"], ev["wide"], take_op)
    r["words"] = ev["words"]
    r["detected"] = ev["detected"]
    return r


def read_wav(op, wav, fps, lyric_words=None, take_op=None, phones=None):
    """read() on a vocal file; every value None when there is no file."""
    if not wav or not os.path.exists(wav):
        return dict.fromkeys(KEYS)
    return read(op, JV.read_mono(wav), fps, lyric_words, take_op, phones)


# -- the opening off a video (the judge's own measurement, judge_mouth) ---------------------------------------------------------------
def opening_of(video, work=None, frames=0, fps=25.0):
    """(gap in eye distances per frame, gap over face height per frame, fps) of the delivered shot. work: a sing work dir whose
    medium.json names the gate's box (read as data: the paste's median centre, 1.6 face heights a side)."""
    from . import judge_mouth as JM

    F, vf = JM.frames(video, frames or 0)
    if not F:
        return None, None, vf or fps
    fps = vf or fps
    boxes = None
    if work and os.path.exists(os.path.join(work, "medium.json")):
        with open(os.path.join(work, "medium.json"), encoding="utf-8") as f:
            y0, y1, x0, x1 = JV.gate_box(json.load(f), F[0].shape)
        boxes = {0: [x0, y0, x1, y1]}
        roi = [y0, y1, x0, x1]
    else:
        roi = JM.roi_of(F, boxes)
    giod, gface, _ = JM.opening(F, roi, fps)
    return giod, gface, fps


def lyric_from(lyrics, stt=None):
    """the lyric words of a --lyrics file, windowed by a --stt transcript when given."""
    if not lyrics:
        return None
    with open(lyrics, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    heard = []
    if stt and os.path.exists(stt):
        with open(stt, encoding="utf-8") as f:
            heard = [w.get("w", "") for w in json.load(f)]
    return JP.window(lines, heard)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lsdub event-judge")
    ap.add_argument("--video", required=True, help="the delivered shot")
    ap.add_argument("--vocal", required=True, help="the vocal stem, cut to start with the shot's first frame")
    ap.add_argument("--lyrics", help="the lyric, one line per line; without it only the flux onsets are read")
    ap.add_argument("--stt", help="a transcript (words with w/start/end) that picks the lyric window of a cut that starts mid-line")
    ap.add_argument("--take", help="the shot's take: adds spurious_take and open_ratio")
    ap.add_argument("--work", help="a sing work dir: the mouth is read on the gate's box (its medium.json)")
    ap.add_argument("--models", help="the recogniser's folder (default: LSDUB_PHONE_MODELS, LSDUB_HF_MODELS/wav2vec2-espeak, else the hub)")
    ap.add_argument("--no-model", action="store_true", help="flux onsets only, no recogniser")
    ap.add_argument("--frames", type=int, help="read this many frames (default: all)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--json")
    ap.add_argument("--strip", help="write the review strip PNG here")
    a = ap.parse_args(argv)
    giod, op, fps = opening_of(a.video, a.work, a.frames, a.fps)
    take = None
    if a.take:
        take, _, _ = opening_of(a.take, a.work, a.frames, a.fps)
    phones = None if a.no_model else JP.Phones(a.models)
    x = JV.read_mono(a.vocal)
    r = read(giod if giod is not None else op, x, fps, lyric_from(a.lyrics, a.stt), take, phones)
    r["why"] = why(r)
    print(json.dumps({k: r.get(k) for k in KEYS + ("why",)}))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if a.strip:
        from . import event_strip as ES

        ES.strip(r, x, giod if giod is not None else op, fps, a.strip, take_op=take, title=os.path.basename(a.video))
    return 0 if r.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
