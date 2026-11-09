# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""nucleus_events.py: the vocal's syllable events written onto the drive track. One stage of the drive chain, behind one paint flag,
--warp-events (off | vocal), fed by --warp-audio (the lead stem). Pure: frames and arrays in, frames out; no pixel changes.

  drive (the track's frames) -> [events] -> the hold stage -> the closure stage -> the aperture stage -> the warp's targets

Every track frame gains `ev`, the event it sits in: {k, kind, t_on, vowel, db, bin, cons}
  kind     nucleus   a sung syllable's start, from the phone network's nuclei and the spectral-flux onsets merged MIN_GAP apart
           closure   a lip closure, from the alignment's P/B/M onsets (the closure stage decides what to do with it)
           rest      the voice off for at least REST_S (an event of its own: the hold stage may decay into it)
  t_on     the event's start, in the clip's clock (frame 0 at 0 s)
  vowel    the nucleus's vowel class: the aligned lyric phone over the nucleus when the alignment was read off the vocal (aligner
           ctc), else the recogniser's own label mapped to the nearest ARPAbet vowel; None on a closure or rest
  db       the nucleus's peak level under the clip's nucleus mean (0 = an average syllable)
  bin      0 | 1 | 2: the peak level against the clip's nuclei, mean - sd / mean + sd (a coarse per-syllable loudness, never a
           frame-by-frame envelope)
  cons     a consonant starts between this nucleus and the one before it (the aligned phones, else the recogniser's runs): the
           syllable has an articulated onset; a re-articulation inside a held vowel has none
  src      which detector found the nucleus: nuc (the recogniser), flux (the spectral flux), nucflux (both within MIN_GAP)
`open`, `vis`, `vis2`, `mix` are untouched here: this stage annotates, the hold stage reads the annotation.

  events(x, words, dur, logp=None, net=None) -> [{kind, t_on, vowel, db, bin}]     the event list off the vocal
  annotate(frames, ev, fps) -> (frames, info)                                       the list onto the frames
"""

import numpy as np

from . import phone_net as PN

MODES = ("off", "vocal")
REST_S = 0.15  # s: the voice off this long is a rest
PEAK_S = 0.15  # s after a nucleus over which its level peak is read
BINS = (0.15, 0.45, 0.8)  # the loudness bin values the hold stage reads: three bins about the clip's nucleus mean +- sd
CONS_SLACK = 0.02  # s: a consonant aligned up to this late still belongs to the onset it starts
CLOSERS = ("P", "B", "M")
WIDE_FIRST = {"ɑ": "AA", "a": "AE", "æ": "AE", "ʌ": "AH", "ə": "AH", "ɐ": "AH", "ɔ": "AO", "ɛ": "EH", "e": "EY", "ɪ": "IH", "i": "IY",
              "o": "OW", "ʊ": "UH", "u": "UW", "ɚ": "ER", "ɜ": "ER", "ɝ": "ER", "y": "IY", "ø": "OW", "œ": "EH"}  # fmt: skip


def _arpa_of_label(lab):
    """the nearest ARPAbet vowel of a recogniser label (its first letter; a diphthong by its first element)."""
    for c in str(lab):
        if c in WIDE_FIRST:
            return WIDE_FIRST[c]
    return "AH"


def vowel_at(words, t, reach=PN.MIN_GAP):
    """the aligned vowel phone over time t (or starting within reach of it), from words with phones; None when there is none."""
    best, dist = None, None
    for w in words or []:
        for p in w.get("phones") or []:
            ph = str(p["p"]).rstrip("012")
            if ph not in PN.ARPA_VOWELS:
                continue
            if p["start"] <= t < p["end"]:
                return ph
            d = abs(p["start"] - t)
            if d <= reach and (dist is None or d < dist):
                best, dist = ph, d
    return best


def consonant_starts(words, logp=None, net=None):
    """the start times of the consonants: the aligned non-vowel phones when there are words, else the recogniser's consonant runs."""
    out = [p["start"] for w in words or [] for p in w.get("phones") or [] if str(p["p"]).rstrip("012") not in PN.ARPA_VOWELS]
    if not out and logp is not None and net is not None:
        out = [a for lab, a, b in PN.runs(logp, net) if lab[0] not in PN.VOWEL_LETTERS]
    return sorted(out)


def rests(sung_mask, dur, hop=PN.HOP, min_s=REST_S):
    """[(start, end)] of the voice being off for min_s or longer, inside [0, dur]."""
    m = np.asarray(sung_mask, bool)
    out, i, n = [], 0, len(m)
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if (j - i) * hop >= min_s:
                out.append((round(i * hop, 3), round(min(dur, j * hop), 3)))
            i = j
        else:
            i += 1
    return out


def leads_into_voice(sung_mask, t, hop=PN.HOP, span=PEAK_S, share=0.5):
    """an onset is a syllable's start only when the voice is on over most of the span after it (a note's release is not an onset)."""
    m = np.asarray(sung_mask, bool)
    a = int(t / hop)
    seg = m[a : max(a + 1, a + int(span / hop))]
    return bool(len(seg)) and float(seg.mean()) >= share


def loudness(db, times, hop=PN.HOP, peak_s=PEAK_S):
    """(peak level per time, bin per time): each nucleus's peak over the next peak_s, against the clip's nuclei (mean +- sd)."""
    db = np.asarray(db, float)
    pk = []
    for t in times:
        a = int(t / hop)
        seg = db[a : max(a + 1, a + int(peak_s / hop))]
        pk.append(float(seg.max()) if len(seg) else float(db[min(a, len(db) - 1)]) if len(db) else 0.0)
    pk = np.asarray(pk)
    if not len(pk):
        return [], []
    mu, sd = float(pk.mean()), float(pk.std())
    bins = [0 if v < mu - sd else 2 if v > mu + sd else 1 for v in pk]
    return [round(float(v - mu), 1) for v in pk], bins


def events(x, words, dur, logp=None, net=None, vocal_read=False):
    """the event list off vocal samples x (PN.SR, t = 0 at sample 0), the aligned words (clip clock) and the clip length dur."""
    x = np.asarray(x, float)
    on = PN.sung(x)
    nuc_t = PN.nuclei(logp, x, net) if logp is not None and net is not None else []
    flux_t = PN.flux_onsets(x)
    ev = [t for t in PN.vocal_events(x, logp, net) if t < dur and leads_into_voice(on, t)]
    labels = PN.nucleus_labels(logp, x, net) if logp is not None and net is not None else []
    db = PN.level_db(x)
    pk, bins = loudness(db, ev)
    cons_times = consonant_starts(words, logp, net)
    out, prev = [], None
    for t, d, b in zip(ev, pk, bins):
        v = vowel_at(words, t) if vocal_read else None
        if v is None:
            near = [(abs(a - t), lab) for a, lab in labels if abs(a - t) <= PN.MIN_GAP]
            v = _arpa_of_label(min(near)[1]) if near else (vowel_at(words, t) or "AH")
        lo = (prev if prev is not None else -1.0) - 1e-9
        cons = any(lo < c <= t + CONS_SLACK for c in cons_times)  # a consonant between this nucleus and the one before it
        src = ("nuc" if any(abs(a - t) <= PN.MIN_GAP for a in nuc_t) else "") + (
            "flux" if any(abs(a - t) <= PN.MIN_GAP for a in flux_t) else ""
        )
        out.append(dict(kind="nucleus", t_on=round(float(t), 3), vowel=v, db=d, bin=int(b), cons=bool(cons), src=src or "nuc"))
        prev = t
    for w in words or []:
        for p in w.get("phones") or []:
            if str(p["p"]).rstrip("012") in CLOSERS and 0 <= p["start"] < dur:
                out.append(
                    dict(kind="closure", t_on=round(float(p["start"]), 3), t_off=round(float(p["end"]), 3), vowel=None, db=None, bin=None)
                )
    for a, b in rests(PN.sung(x), dur):
        out.append(dict(kind="rest", t_on=a, t_off=b, vowel=None, db=None, bin=None))
    out.sort(key=lambda e: (e["t_on"], ("closure", "nucleus", "rest").index(e["kind"])))
    return out


def annotate(frames, ev, fps, t0=0.0):
    """-> (new frames with `ev`, info). A frame inside a closure's [t_on, t_off) carries the closure; else inside a rest, the rest;
    else the nucleus whose t_on is the latest at or before the frame's time; frames before the first nucleus carry {k: -1, kind:
    'rest'}. The input frames are not changed."""
    out = []
    nuc = [(e["t_on"], k) for k, e in enumerate(ev) if e["kind"] == "nucleus"]
    spans = [(e["t_on"], e.get("t_off", e["t_on"]), k) for k, e in enumerate(ev) if e["kind"] != "nucleus"]
    starts = [t for t, _ in nuc]
    for i, f in enumerate(frames):
        t = (f.get("t", t0 + i / fps) if isinstance(f, dict) else t0 + i / fps) - t0
        cur = k = None
        for a, b, kk in spans:
            if a <= t < b and (cur is None or ev[kk]["kind"] == "closure"):
                cur, k = ev[kk], kk
                if cur["kind"] == "closure":
                    break
        if cur is None:
            j = int(np.searchsorted(starts, t + 1e-9, side="right")) - 1
            if j >= 0:
                k = nuc[j][1]
                cur = ev[k]
        if cur is None:
            tag = dict(k=-1, kind="rest", t_on=None, vowel=None, db=None, bin=None, cons=False)
        else:
            tag = dict(
                k=k,
                kind=cur["kind"],
                t_on=cur["t_on"],
                vowel=cur["vowel"],
                db=cur["db"],
                bin=cur["bin"],
                cons=cur.get("cons", False),
                src=cur.get("src"),
            )
            if "t_off" in cur:
                tag["t_off"] = cur["t_off"]
        out.append(dict(f, ev=tag))
    info = dict(
        mode="vocal",
        nuclei=sum(1 for e in ev if e["kind"] == "nucleus"),
        closures=sum(1 for e in ev if e["kind"] == "closure"),
        rests=sum(1 for e in ev if e["kind"] == "rest"),
        events=[dict(e) for e in ev],
    )
    return out, info
