# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""phone_plan.py: the aligner's phones as the mouth makes them (--coart-phones, --coart-function-gate). Opt-in.

The viseme map is fifteen classes: AH, AE, AY and AW are all `aa`. A plan on those classes cannot tell a passing vowel from a full one and
loses the second half of every diphthong. Here the phones keep their identity:

  merge     a vowel the aligner cut into repeats (AY AY AY) is one vowel
  reduced   AH, IH under no stress (digit 0; without digits, shorter than REDUCED_S) get a low, unrounded class of their own
  glides    AY, AW, OY, EY and OW become a nucleus and an offglide: the nucleus holds the note, the glide takes GLIDE_SHARE of it, GLIDE_S at most
  dental    F, V, TH, DH last at least EVENT_S where anything beside them can give it (stretch_dental), an aligner's 20 ms is less than a frame
  gate      a closed-class word gets less opening only when it is quick and not loud (function_gain); a held word is left whole

phones() -> a phone list as viseme_track.phone_list gives it, each with `k`, the opening gain of its word; curves() reads it as sung_dyn reads the
fifteen classes. CLOSED_CLASS is written by hand."""

import numpy as np

from . import viseme_track as vt

CLOSED_CLASS = frozenset(
    """a an the and or but so if as at by for from in into of off on onto out over to up with about than then
    i i'm i'd i'll i've me my mine we we're us our you you're your he his him she her it it's its they they're them their
    this that these those there here what who when where which
    am is are was were be been do does did don't can could will would shall should have has had not no""".split()
)
EXTRA_OPEN = {
    "ae": 0.85,
    "ah": 0.7,
    "ax": 0.36,
    "ix": 0.26,
    "gi": 0.6,
    "gu": 0.5,
}  # gi, gu: the offglides. A sung glide shapes the lips on a note still sounding, so it stays open
EXTRA_WIDE = {"ae": 0.3, "ah": 0.05, "ax": 0.05, "ix": 0.25, "gi": 0.5, "gu": -0.7}
BASE = {
    "ae": "aa",
    "ah": "aa",
    "ax": "aa",
    "ix": "ih",
    "gi": "ih",
    "gu": "ou",
}  # the fifteen-class name of each: what the readers of the plan's log know
GLIDES = {"AY": ("aa", "gi"), "AW": ("aa", "gu"), "OY": ("oh", "gi"), "EY": ("E", "gi"), "OW": ("oh", "gu")}
ROUNDED = ("ou", "gu", "oh")
VOWELS = frozenset(("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"))
DENTAL = ("FF", "TH")
REDUCED_S = 0.12  # s: an AH or IH shorter than this, with no stress digit to read, is reduced
GLIDE_SHARE, GLIDE_MIN_S, GLIDE_S = 0.3, 0.06, 0.12  # the offglide's share of its vowel, the least and the most of it, s
EVENT_S = 0.07  # s: the least a dental gesture lasts
MIN_LEFT = 0.03  # s: what a phone keeps when a dental takes time from it
QUICK_S, HELD_S = 0.1, 0.24  # s: a word's longest vowel at or under QUICK_S is quick, at or over HELD_S it is held
SOFT_LEVEL, LOUD_LEVEL = 0.7, 0.95  # the vocal's drive, 0..1: from soft to loud a quick word gets back up to LOUD_KEEP of what it lost
LOUD_KEEP = 0.3
FLOOR = 0.5  # the most a word loses: what the animator rule took from every closed-class word


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def strip(p):
    return "".join(c for c in p if not c.isdigit())


def word_key(w):
    return str(w).lower().strip('.,!?;:"()[]{}')


def function_gain(word, held_s, level=1.0):
    """the opening kept on a word: 1 for an open-class or held word, down to FLOOR for a quick, soft closed-class one. held_s: the word's longest vowel,
    s; level: the vocal's peak drive on the word, 0..1."""
    if word_key(word) not in CLOSED_CLASS:
        return 1.0
    quick = 1.0 - float(smoothstep((held_s - QUICK_S) / (HELD_S - QUICK_S)))
    loud = float(smoothstep((level - SOFT_LEVEL) / (LOUD_LEVEL - SOFT_LEVEL)))
    return 1.0 - (1.0 - FLOOR) * quick * (1.0 - LOUD_KEEP * loud)


def merged(phones):
    """one word's phones with a repeated phone joined: the aligner's spread of a held vowel into equal pieces is one vowel."""
    out = []
    for p in phones:
        if out and strip(out[-1]["p"]) == strip(p["p"]) and p["start"] - out[-1]["end"] < 0.03:
            out[-1] = dict(out[-1], end=p["end"])
        else:
            out.append(dict(p))
    return out


def is_reduced(p):
    """AH / IH with no stress on it: digit 0, or no digit and short. Any other vowel keeps its class."""
    a = strip(p["p"])
    if a not in ("AH", "IH"):
        return False
    d = "".join(c for c in p["p"] if c.isdigit())
    return d == "0" if d else p["end"] - p["start"] < REDUCED_S


def vowel_class(p):
    """the class of a vowel that is not a diphthong."""
    a = strip(p["p"])
    if is_reduced(p):
        return "ax" if a == "AH" else "ix"
    return {"AE": "ae", "AH": "ah"}.get(a) or vt.ARPA[a]


def split_glide(p, nucleus, glide):
    """the diphthong as two phones: the nucleus, then the offglide on its tail."""
    dur = p["end"] - p["start"]
    g = float(np.clip(GLIDE_SHARE * dur, min(GLIDE_MIN_S, 0.5 * dur), GLIDE_S))
    m = p["end"] - g
    return [dict(p, end=m, vis=nucleus, part="nucleus"), dict(p, start=m, vis=glide, part="glide")]


def stretch_dental(ph):
    """a dental shorter than EVENT_S grows to it: first from a vowel of its word beside it, whichever side has more, then into the silence beside it, then from any
    other phone of its word but a closure. Each phone keeps MIN_LEFT; what is still short stays short (a lone F between two words with nothing to give)."""
    for i, p in enumerate(ph):
        lack = EVENT_S - (p["end"] - p["start"])
        if p["vis"] not in DENTAL or lack <= 0:
            continue
        nxt, prv = ph[i + 1] if i + 1 < len(ph) else None, ph[i - 1] if i else None
        ours = lambda q, vowel: q is not None and q["wi"] == p["wi"] and q["vis"] != "PP" and (strip(q["p"]) in VOWELS) == vowel
        room = lambda q, vowel: max(q["end"] - q["start"] - MIN_LEFT, 0.0) if ours(q, vowel) else 0.0
        take = lambda q, d: (
            (p.update(end=p["end"] + d), q.update(start=q["start"] + d))
            if q is nxt
            else (p.update(start=p["start"] - d), q.update(end=q["end"] - d))
        )
        for q in sorted((nxt, prv), key=lambda q: -room(q, True)):
            d = min(lack, room(q, True))
            if d > 0:
                take(q, d)
                lack -= d
        d = min(lack, np.inf if nxt is None else max(nxt["start"] - p["end"], 0.0))
        p["end"] += d
        lack -= d
        d = min(lack, np.inf if prv is None else max(p["start"] - prv["end"], 0.0))
        p["start"] -= d
        lack -= d
        for q in (nxt, prv):
            d = min(lack, room(q, False))
            if d > 0:
                take(q, d)
                lack -= d
    return ph


def phones(align, drive=None, reduce=True, gate=False):
    """the plan's phones (start, end, p, word, wi, vis, k). drive: (t, level) of the vocal, or None; reduce: --coart-phones; gate: --coart-function-gate."""
    level = (lambda a, b: 1.0) if drive is None else (lambda a, b: float(np.interp(np.linspace(a, b, 9), *drive).max()))
    out = []
    source = vt.phone_list(align)
    for wi, w in enumerate(align["words"]):
        ph = [p for p in source if p["wi"] == wi]
        if reduce:
            ph = merged(ph)
        vowels = [p for p in ph if strip(p["p"]) in VOWELS]
        k = 1.0
        if gate and vowels:
            k = function_gain(w["w"], max(p["end"] - p["start"] for p in vowels), level(vowels[0]["start"], vowels[-1]["end"]))
        done = []
        for p in ph:
            a = strip(p["p"])
            if not reduce or a not in VOWELS:
                done.append(dict(p, k=k if a in VOWELS else 1.0))
            elif a in GLIDES:
                done += [dict(q, k=k) for q in split_glide(p, *GLIDES[a])]
            else:
                done.append(dict(p, vis=vowel_class(p), k=k))
        out += done
    return stretch_dental(out) if reduce else out


def open_of(v):
    return EXTRA_OPEN.get(v, vt.OPEN.get(v, 0.0))


def curves(ph, ts, n, rate, lead=0.06, release=0.11):
    """what sung_dyn reads off the fifteen classes, off these phones: (shape opening, width, closure) per row, each class weighted as viseme_track.build
    weighs it, the phone's `k` on the opening. wide: sung_dyn.WID + EXTRA_WIDE."""
    from .sung_dyn import WID

    wide = {**WID, **EXTRA_WIDE}
    tt = ts + (np.arange(n) + 0.5) / rate
    cls = sorted({p["vis"] for p in ph} | {"sil", "PP"})  # two classes at least: the runner-up is read below
    ix = {v: i for i, v in enumerate(cls)}
    W, G = np.zeros((n, len(cls))), np.zeros((n, len(cls)))
    W[:, ix["sil"]] = 0.15
    for p in ph:
        a0, a1, b0, b1 = p["start"] - lead, p["start"], p["end"], p["end"] + release
        act = np.zeros(n)
        act[(tt >= a1) & (tt <= b0)] = 1
        m = (tt > a0) & (tt < a1)
        act[m] = 0.5 - 0.5 * np.cos(np.pi * (tt[m] - a0) / lead)
        m = (tt > b0) & (tt < b1)
        act[m] = 0.5 + 0.5 * np.cos(np.pi * (tt[m] - b0) / release)
        pr = vt.PRIO.get(BASE.get(p["vis"], p["vis"]), 1.0)
        W[:, ix[p["vis"]]] += act * pr
        G[:, ix[p["vis"]]] += act * pr * p.get("k", 1.0)
    d = np.argsort(-W, 1)[:, :2]
    r = np.arange(n)
    wd, ws = W[r, d[:, 0]], np.maximum(W[r, d[:, 1]], 0.0)
    mix = np.minimum(ws / np.maximum(wd + ws, 1e-9), 0.5)
    gain = lambda c: np.where(W[r, c] > 0, np.where(G[r, c] > 0, G[r, c] / np.maximum(W[r, c], 1e-9), 1.0), 1.0)
    op = np.array([open_of(v) for v in cls])
    wi = np.array([wide.get(v, 0.0) for v in cls])
    so = (1 - mix) * op[d[:, 0]] * gain(d[:, 0]) + mix * op[d[:, 1]] * gain(d[:, 1])
    sw = (1 - mix) * wi[d[:, 0]] + mix * wi[d[:, 1]]
    return so, sw, np.array([cls[c] == "PP" for c in d[:, 0]])


def event_frames(ph, tc, classes, pad=0.0):
    """per frame (centres tc): the share 0..1 of the frame inside a phone of these classes, one per class. An event hits a whole frame or none it touches."""
    out = {v: np.zeros(len(tc)) for v in classes}
    if len(tc) < 2:
        return out
    half = 0.5 * float(np.median(np.diff(tc)))
    for p in ph:
        if p["vis"] in out:
            hit = (tc + half > p["start"] - pad) & (tc - half < p["end"] + pad)
            out[p["vis"]][hit] = 1.0
    return out


def rounding(ph, tc, lead=0.1, release=0.08):
    """the lips' rounding 0..1 per frame: up over `lead` s before a rounded phone (lips round ahead of the sound), down over `release` after it. Full on ou and
    its offglide, 0.7 on oh."""
    r = np.zeros(len(tc))
    for p in ph:
        if p["vis"] not in ROUNDED:
            continue
        s = 0.7 if p["vis"] == "oh" else 1.0
        rise = smoothstep((tc - (p["start"] - lead)) / lead)
        fall = 1.0 - smoothstep((tc - p["end"]) / release)
        r = np.maximum(r, s * np.minimum(rise, fall))
    return r


def base_class(v):
    return BASE.get(v, v)
