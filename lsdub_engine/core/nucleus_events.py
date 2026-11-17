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
  src      which detector found the nucleus: nuc (the recogniser), flux (the spectral flux), nucflux (both within MIN_GAP); held
           (--warp-nuclei lyric) a detection inside a lyric vowel already begun: the note going on, no new syllable
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
NUCLEI = ("all", "lyric")  # --warp-nuclei: which detections are syllable events
LYRIC_S = 0.10  # s: nuclei "lyric" takes a detection as a vowel's onset when the aligned vowel starts within this of it
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


def lyric_vowels(words):
    """the start times of the aligned lyric's vowel phones, in the words' clock."""
    return sorted(float(p["start"]) for w in words or [] for p in w.get("phones") or [] if str(p["p"]).rstrip("012") in PN.ARPA_VOWELS)


def _nearest(nuc, starts, reach, taken=()):
    """per start time, the index of the nearest free detection within reach of it (None when there is none), claimed in time order."""
    keep, out = set(taken), []
    for v in starts:
        near = [(abs(e["t_on"] - v), i) for i, e in enumerate(nuc) if i not in keep and abs(e["t_on"] - v) <= reach]
        out.append(min(near)[1] if near else None)
        if near:
            keep.add(out[-1])
    return out


def confirmed(ev, words, reach=LYRIC_S):
    """the onset half of --warp-nuclei lyric (`supported` is the whole of it): the event list with only the nuclei at a lyric vowel's
    START. Each aligned vowel keeps the detection nearest its start within `reach`, one detection per vowel; every other detection (a
    flux peak or a recogniser vowel inside a held note, a breath, a consonant burst) is dropped and its frames stay in the hold before
    it. A kept nucleus carries `cons` when it or a dropped detection since the kept one before it had one.
    A lyric with no aligned vowel confirms nothing and the list comes back whole. -> (events, dropped count)."""
    vs = lyric_vowels(words)
    nuc = [e for e in ev if e["kind"] == "nucleus"]
    if not vs or not nuc:
        return list(ev), 0
    keep = {i for i in _nearest(nuc, vs, reach) if i is not None}
    out, cons, k = [], False, 0
    for e in ev:
        if e["kind"] != "nucleus":
            out.append(e)
            continue
        cons = cons or bool(e.get("cons"))
        if k in keep:
            out.append(dict(e, cons=cons))
            cons = False
        k += 1
    return out, len(nuc) - len(keep)


def lyric_spans(words):
    """[(start, end, vowel class)] of the aligned lyric's vowel phones, in the words' clock, by start."""
    return sorted(
        (float(p["start"]), float(p["end"]), str(p["p"]).rstrip("012"))
        for w in words or []
        for p in w.get("phones") or []
        if str(p["p"]).rstrip("012") in PN.ARPA_VOWELS
    )


def supported(ev, words, reach=LYRIC_S):
    """--warp-nuclei lyric: the event list with only the detections a sung lyric vowel supports. Three ways a vowel supports one:

      onset    the detection nearest the vowel's start within `reach`, one per vowel (`confirmed`'s rule)
      late     a vowel no detection starts: the first free detection inside its span [start, end) is its onset (a note that swells in,
               or an aligned start ahead of the sound: without it the mouth sat at rest through the whole vowel)
      held     every other detection inside a vowel's span: the vowel still being sung (melisma, a re-swell on a held note). It stays
               an event so the hold has a target for as long as the vowel lasts, but it is no new syllable: `src` "held", `cons`
               False, so the hold stage makes no release before it and the opening moves only by its own loudness bin

    Each kept detection takes its vowel's class (the detection confirmed by "every"'s EH is an EH, whatever phone the frame under it
    carried). A detection inside no aligned vowel (a breath, a consonant burst, a flux peak in a gap) is dropped, and an onset carries
    `cons` when it or a dropped detection since the kept one before it had one.
    A lyric with no aligned vowel supports nothing and the list comes back whole. -> (events, dropped count)."""
    vs = lyric_spans(words)
    nuc = [e for e in ev if e["kind"] == "nucleus"]
    if not vs or not nuc:
        return list(ev), 0
    role = {}  # detection index -> (onset | held, vowel class)
    on = _nearest(nuc, [a for a, _, _ in vs], reach)
    for i, (_, _, ph) in zip(on, vs):
        if i is not None:
            role[i] = ("onset", ph)
    for i, (a, b, ph) in zip(on, vs):  # late: after every vowel had its own onset's pick
        if i is None:
            inside = [j for j, e in enumerate(nuc) if j not in role and a <= e["t_on"] < b]
            if inside:
                role[inside[0]] = ("onset", ph)
    for a, b, ph in vs:
        for j, e in enumerate(nuc):
            if j not in role and a <= e["t_on"] < b:
                role[j] = ("held", ph)
    out, cons, k = [], False, 0
    for e in ev:
        if e["kind"] != "nucleus":
            out.append(e)
            continue
        cons = cons or bool(e.get("cons"))
        if k in role:
            kind, ph = role[k]
            if kind == "onset":
                out.append(dict(e, vowel=ph, cons=cons))
                cons = False
            else:
                out.append(dict(e, vowel=ph, cons=False, src="held"))
        k += 1
    return out, len(nuc) - len(role)


def events(x, words, dur, logp=None, net=None, vocal_read=False, nuclei="all"):
    """the event list off vocal samples x (PN.SR, t = 0 at sample 0), the aligned words (clip clock) and the clip length dur.
    nuclei (NUCLEI): "lyric" keeps only the detections a sung lyric vowel supports (`supported`)."""
    if nuclei not in NUCLEI:
        raise ValueError(f"--warp-nuclei {nuclei!r}")
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
    if nuclei == "lyric":
        out, _ = supported(out, words)
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
