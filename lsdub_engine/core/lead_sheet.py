# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lead_sheet.py: which sung words the singer on screen sings. A speech-to-text pass reads a backing stutter as words, the lead vocal split
keeps it wherever it shares the lead's pitch, and the alignment then spreads a lyric word over it. This reads the lead vocal for what a lead
line does not do and writes the song out as the lyric sheet with the rest in braces, for a person to read and correct before paint:

  it's about that time {time time time ti ti ti ti ti ti time}  10.04-11.93

Evidence, all of it read off the audio and shown per call:
  stutter    a train of narrow level notches, voiced throughout, closer than a sung syllable runs: "time time ti ti ti". Consonants in a lead line
             break the voicing; a vibrato does not reach the depth.
  register   a stretch sung outside the clip's own pitch range, as a harmony or an answer line is.
  heard      Whisper's word against the sheet's: a word in a run that the sheet writes differently is what the run was misheard as.
  relisten   optional: the run re-heard on its own (Whisper, unprompted): a repeated token is a run of fragments, not a lyric.
The stem share and stereo width are read and reported: on a mix where the backing sits apart they count, on one where they sit with the lead
(this is usual for a stutter edit) they do not, so neither decides alone.

Overrides, one per line, `pass 10.04 11.93` or `lead 10.04 10.49`, '#' for comments: a person's word beats the reading. `lead` removes that time
from what is passed; `pass` adds it.

  python -m lsdub_engine.core.lead_sheet --align align.json --vocal lead.wav [--lyrics sheet.txt] [--overrides o.txt] --out lead_sheet [--apply align_lead.json]

read_sheet(...) -> {spans, words, lines}; honour(align, sheet) -> the alignment with the passed time taken out of it. Nothing here runs unless asked."""

import argparse, difflib, json, os, re
import numpy as np

from . import vocal_retime as VR

NOTCH_DB = 6.0  # a notch: this far under the loudest within LOCAL
NOTCH_S = 0.07  # s: no wider. A real closure is wider, or breaks the voicing
RUN_GAP = 0.3, 0.08  # s, longest and shortest step from notch to notch
RUN_MIN = 7  # notches. Five or six in a row do turn up in a fast lead line
VOICED = 0.5  # the sound stays at least this voiced across a notch
OFF_SEMI = 5.0  # semitones past the middle half of the clip's pitch: another voice
OFF_S = 0.2  # s, at least
FULL_S = 0.18  # s: a fragment this long is the whole word, shorter its first letters
LOW = 0.6  # confidence under this is marked, not decided silently
PASS = "pass-over"


def pitch(x):
    """Hz per frame at vocal_retime's hop, 0 where unvoiced: the autocorrelation peak inside the singing range."""
    N, H = int(VR.WIN * VR.SR), int(VR.HOP * VR.SR)
    lo, hi = int(VR.SR / VR.F0_HI), int(VR.SR / VR.F0_LO)
    n = max(0, (len(x) - N) // H + 1)
    if not n:
        return np.zeros(0)
    fr = np.stack([x[k * H : k * H + N] for k in range(n)]) * np.hanning(N)
    a = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * N, axis=1)) ** 2, axis=1)[:, : hi + 1]
    k = lo + np.argmax(a[:, lo:], axis=1)
    return np.where(a[np.arange(n), k] > 0.5 * np.maximum(a[:, 0], 1e-12), VR.SR / k, 0.0)


def runs_of(m):
    d = np.diff(np.r_[0, np.asarray(m, int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def notches(fx):
    """(first frame, one past the last) of each narrow voiced notch in the level."""
    db, v = fx["db"], fx["voiced"]
    k = max(1, int(round(VR.LOCAL / VR.HOP)))
    deep = VR.nearby(db, k, 100) - db >= NOTCH_DB
    keep = []
    for a, b in runs_of(deep):
        if (b - a) * VR.HOP <= NOTCH_S and v[max(0, a - 2) : b + 2].mean() >= VOICED:
            keep.append((int(a), int(b)))
    return keep


def stutters(fx):
    """the trains of notches: [(first notch's frame, last notch's frame, [each notch's frame])]."""
    N = [(a + b) // 2 for a, b in notches(fx)]
    v = fx["voiced"]
    out, cur = [], N[:1]
    for p, q in zip(N, N[1:]):
        step = (q - p) * VR.HOP
        if RUN_GAP[1] <= step <= RUN_GAP[0] and v[p:q].mean() >= VOICED:
            cur.append(q)
            continue
        if len(cur) >= RUN_MIN:
            out.append((cur[0], cur[-1], cur))
        cur = [q]
    if len(cur) >= RUN_MIN:
        out.append((cur[0], cur[-1], cur))
    return out


def off_register(f0):
    """the frames sung outside the clip's own range: [(first, one past the last)]."""
    v = f0[f0 > 0]
    if len(v) < 20:
        return []
    lo, hi = np.percentile(v, [25, 75]) * 2.0 ** (np.array([-OFF_SEMI, OFF_SEMI]) / 12)
    off = (f0 > 0) & ((f0 < lo) | (f0 > hi))
    # a short dropout inside a stretch does not end it
    w = int(round(0.04 / VR.HOP))
    off = np.convolve(off.astype(float), np.ones(2 * w + 1), "same") > 0
    keep = [(int(a), int(b)) for a, b in runs_of(off)]
    return [(a + w, b - w) for a, b in keep if (b - a - 2 * w) * VR.HOP >= OFF_S]


def frame_t(k):
    return k * VR.HOP + VR.WIN / 2


def spans_of(x, words):
    """the passed time on a vocal: stutter trains and off-register stretches, a train opening at the end of the word it echoes. -> [dict(t0, t1, kinds, cuts)]"""
    fx = VR.features(x)
    s = []
    for a, b, N in stutters(fx):
        s.append(dict(t0=frame_t(a), t1=frame_t(b), kinds=["stutter"], cuts=[frame_t(k) for k in N]))
    for a, b in off_register(pitch(x)):
        s.append(dict(t0=frame_t(a), t1=frame_t(b), kinds=["register"], cuts=[]))
    s.sort(key=lambda q: q["t0"])
    out = []
    for q in s:
        if out and q["t0"] - out[-1]["t1"] < 0.15:
            p = out[-1]
            p["t1"] = max(p["t1"], q["t1"])
            p["kinds"] = sorted(set(p["kinds"]) | set(q["kinds"]))
            p["cuts"] = sorted(p["cuts"] + q["cuts"] + ([q["t0"]] if not q["cuts"] else []))
        else:
            out.append(q)
    for q in out:  # the word it starts inside keeps its time when most of it stands before: the train then opens on its end
        for w in words:
            a, b = float(w["start"]), float(w["end"])
            if a < q["t0"] < b and q["t0"] - a > 0.5 * (b - a) or a < q["t0"] and b > q["t0"] and b - q["t0"] < 0.25:
                q["t0"] = b
    return [dict(q, t0=round(q["t0"], 3), t1=round(q["t1"], 3)) for q in out if q["t1"] > q["t0"]]


def word_is(w):
    return re.sub(r"[^a-z0-9']", "", str(w).lower()).strip("'")


def sheet_lines(words, lyrics):
    """{word index: sheet line} on the sheet's token order (difflib), or {} with no sheet."""
    if not lyrics:
        return {}
    toks = [
        (word_is(t), i)
        for i, ln in enumerate(lyrics.splitlines())
        if ln.strip() and not ln.strip().startswith("[")
        for t in re.findall(r"[^\s\-]+", ln)
    ]
    toks = [t for t in toks if t[0]]
    a = [word_is(w.get("lyric") or w["w"]) for w in words]
    out = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [t for t, _ in toks], a, autojunk=False).get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(j1, j2):
                out[k] = toks[min(i1 + k - j1, i2 - 1)][1]
    last = 0
    for k in range(len(a)):
        last = out.setdefault(k, last)
    return out


def repeated(tokens, n=3):
    """a token heard n times in a row or more: the relisten's mark of a run."""
    t = [word_is(x) for x in tokens]
    best = run = 0
    for a, b in zip(t, t[1:]):
        run = run + 1 if a and a == b else 0
        best = max(best, run)
    return best + 1 >= n


def parse_overrides(text):
    """[(`pass` | `lead`, t0, t1)] from the override text; a line it cannot read is an error, never skipped in silence."""
    out = []
    for ln in str(text or "").splitlines():
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        m = re.fullmatch(r"(pass|lead)\s+([\d.]+)\s*[- ]\s*([\d.]+)", ln, re.I)
        if not m or float(m[3]) <= float(m[2]):
            raise ValueError(f"lead_sheet: override '{ln}': use `pass 10.04 11.93` or `lead 10.04 10.49`")
        out.append((m[1].lower(), float(m[2]), float(m[3])))
    return out


def override_spans(spans, ov):
    """the spans with a person's edits on them, in order; what they touch says so."""
    s = [dict(q) for q in spans]
    for kind, a, b in ov:
        keep = []
        for q in s:
            if kind == "lead" and q["t0"] < b and q["t1"] > a:
                if q["t0"] < a:
                    keep.append(dict(q, t1=a, kinds=sorted(set(q["kinds"]) | {"override"})))
                if q["t1"] > b:
                    keep.append(dict(q, t0=b, kinds=sorted(set(q["kinds"]) | {"override"})))
            else:
                keep.append(q)
        s = keep
        if kind == "pass":
            s.append(dict(t0=a, t1=b, kinds=["override"], cuts=[]))
    s.sort(key=lambda q: q["t0"])
    out = []
    for q in s:
        if out and q["t0"] <= out[-1]["t1"]:
            p = out[-1]
            p.update(t1=max(p["t1"], q["t1"]), kinds=sorted(set(p["kinds"]) | set(q["kinds"])), cuts=sorted(p["cuts"] + q["cuts"]))
        else:
            out.append(q)
    return [q for q in out if q["t1"] - q["t0"] >= 0.05]


def fragments(q, echo):
    """the run in pieces: the word it echoes, its first letters on the short ones."""
    cuts = [q["t0"], *[c for c in q["cuts"] if q["t0"] + 0.05 < c < q["t1"] - 0.05], q["t1"]]
    short = (echo[:2] if len(echo) > 3 else echo) or "?"
    return [dict(w=echo if b - a >= FULL_S else short, t0=round(a, 3), t1=round(b, 3)) for a, b in zip(cuts, cuts[1:])]


def confidence(q, relisten=None):
    """(0..1, reason) for a span: register alone is near certain, a train alone is a reading of rhythm, a person's word is final."""
    if "override" in q["kinds"]:
        return 1.0, "set by hand"
    c, why = 0.0, []
    if "register" in q["kinds"]:
        c, _ = 0.85, why.append("sung outside the clip's pitch range")
    if "stutter" in q["kinds"]:
        c = max(c, 0.7)
        why.append(f"{len(q['cuts'])} voiced notches, a stutter train")
    if len(q["kinds"]) > 1:
        c = 0.9
    if relisten is not None:
        if repeated(relisten):
            c = min(0.95, c + 0.1)
            why.append("re-heard alone as a repeat: " + " ".join(relisten[:6]))
        else:
            c -= 0.2
            why.append("re-heard alone with no repeat")
    return round(c, 2), "; ".join(why)


def read_sheet(align, x, lyrics=None, overrides="", relisten=None):
    """the song's calls. align: the alignment; x: the lead vocal at SR; relisten(t0, t1) -> tokens, optional: re-hear a span alone."""
    timed = VR.retime(align, "", x=x)
    W = [dict(w) for w in timed["words"]]
    spans = spans_of(x, W)
    for q in spans:
        q["relisten"] = None if relisten is None else list(relisten(q["t0"], q["t1"]))
    spans = override_spans(spans, parse_overrides(overrides))
    lines = sheet_lines(align["words"], lyrics)
    calls = []
    for i, (w, a) in enumerate(zip(W, align["words"])):
        s, e = float(w["start"]), float(w["end"])
        over = sum(max(0.0, min(e, q["t1"]) - max(s, q["t0"])) for q in spans)
        share = over / max(e - s, 1e-3)
        c = dict(i=i, w=w["w"], t0=round(s, 3), t1=round(e, 3), call="LEAD", conf=1.0, reason="", line=lines.get(i, 0))
        sung = a.get("src") in (None, "lyric", "lyric~")
        if not e > s:
            c.update(conf=0.5, reason="no time of its own in the vocal")
        elif share >= 0.5:
            q = max(spans, key=lambda q: min(e, q["t1"]) - max(s, q["t0"]))
            room = (s < q["t0"] - 0.08) or (i + 1 < len(W) and float(W[i + 1]["start"]) - q["t1"] >= 0.15)
            if (
                sung and room
            ):  # on the sheet and heard as written: the lyric is sung, the run covers its old times. What is left of the line holds it
                c.update(
                    conf=0.55,
                    reason=f"on the sheet and heard so, moved out of the run at {q['t0']:.2f}-{q['t1']:.2f}",
                    move=[q["t0"], q["t1"]],
                )
            else:
                cf, why = confidence(q, q.get("relisten"))
                c.update(
                    call="PASS",
                    conf=round(cf * (0.9 if sung else 1.0), 2),
                    reason=why + ("" if sung else "; the sheet writes another word here"),
                )
        calls.append(c)
    prev = None
    for q in spans:
        leads = [c for c in calls if c["call"] == "LEAD" and c["t1"] <= q["t0"] + 0.02]
        prev = leads[-1]["w"] if leads else ""
        q["conf"], q["reason"] = confidence(q, q.get("relisten"))
        q["fragments"] = fragments(q, word_is(prev))
    return dict(spans=spans, words=calls, low=LOW, overrides=parse_overrides(overrides))


def as_text(S):
    """the lyric sheet: lead lines plain, passed runs in braces with their times, `?` on a call under LOW."""
    items = [(c["t0"], "w", c) for c in S["words"]] + [(q["t0"], "s", q) for q in S["spans"]]
    items.sort(key=lambda it: (it[0], it[1] == "w"))
    L, cur, line = [], [], None
    notes = []
    for _, kind, it in items:
        if kind == "w":
            if it["call"] == "PASS":
                continue
            if line is not None and it["line"] != line:
                L.append(" ".join(cur))
                cur = []
            line = it["line"]
            cur.append(it["w"] + ("?" if it["conf"] < LOW else ""))
        else:
            pw = [c["w"] for c in S["words"] if c["call"] == "PASS" and it["t0"] - 0.05 <= c["t0"] and c["t1"] <= it["t1"] + 0.25]
            cur.append("{" + ("? " if it["conf"] < LOW else "") + " ".join(f["w"] for f in it["fragments"]) + "}")
            notes.append(
                f"{it['t0']:.2f}-{it['t1']:.2f} s  passed over, confidence {it['conf']:.2f}: {it['reason']}"
                + (f"; the alignment's {' '.join(pw)} go with it" if pw else "")
            )
    if cur:
        L.append(" ".join(cur))
    low = [f"{c['t0']:.2f}-{c['t1']:.2f} s  {c['w']}: {c['call']} at {c['conf']:.2f}, {c['reason']}" for c in S["words"] if c["conf"] < LOW]
    out = (
        L
        + ["", "# what is passed over: its mouth stays the take's, it drives nothing and Verify does not read it"]
        + [f"# {n}" for n in notes]
    )
    if low:
        out += ["# low confidence, read these: `word?` above"] + [f"# {n}" for n in low]
    out += ["# correct with `pass 10.04 11.93` / `lead 10.04 10.49`, one per line, in the overrides"]
    return "\n".join(out) + "\n"


def honour(align, S):
    """the alignment with the passed time out of it: a passed word keeps its place and no phones of its own, a lyric moved out of a run is held
    by the time left to it, and `pass_over` names the spans for the retime and for Verify. The lyric is untouched."""
    spans = [[q["t0"], q["t1"]] for q in S["spans"]]
    if not spans:
        return dict(align)
    calls = {c["i"]: c for c in S["words"]}
    W = []
    for i, w in enumerate(align["words"]):
        c, w = calls[i], dict(w)
        if c["call"] == "PASS":
            q = next((q for q in S["spans"] if q["t0"] < c["t1"] and q["t1"] > c["t0"]), None)
            a, b = (max(c["t0"], q["t0"]), min(c["t1"], q["t1"])) if q else (c["t0"], c["t1"])
            w.update(start=a, end=b, vocal=PASS, phones=[dict(p="SIL", start=a, end=b)])
        elif c.get("move"):
            a, b = c["move"]
            s, e = float(w["start"]), float(w["end"])
            to = max(b, s) if e > b else None
            if to is None:  # the line's end falls inside the run: the word closes before it
                to, e, b = s, min(e, a), s
            scale = max(e - to, 0.05) / max(float(w["end"]) - s, 1e-3)
            mv = lambda t: round(to + (float(t) - s) * scale, 3)
            w.update(start=mv(w["start"]), end=mv(w["end"]), phones=[dict(p, start=mv(p["start"]), end=mv(p["end"])) for p in w["phones"]])
        W.append(w)
    return dict(align, words=W, pass_over=spans)


def write(S, stem):
    with open(stem + ".json", "w") as f:
        json.dump(S, f, indent=1)
    with open(stem + ".txt", "w", encoding="utf-8") as f:
        f.write(as_text(S))
    return stem + ".txt", stem + ".json"


def is_passed(spans, t):
    """per time, True inside a passed span."""
    t = np.asarray(t, float)
    m = np.zeros(t.shape, bool)
    for a, b in spans or ():
        m |= (t >= a) & (t < b)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", required=True)
    ap.add_argument("--vocal", required=True)
    ap.add_argument("--lyrics")
    ap.add_argument("--overrides", help="a file of `pass t0 t1` / `lead t0 t1` lines")
    ap.add_argument("--out", required=True, help="stem: writes <out>.txt and <out>.json")
    ap.add_argument("--apply", help="also write the alignment with the passed time taken out")
    a = ap.parse_args(argv)
    A = json.load(open(a.align))
    S = read_sheet(
        A,
        VR.read_mono(a.vocal),
        open(a.lyrics, encoding="utf-8").read() if a.lyrics else None,
        open(a.overrides).read() if a.overrides and os.path.exists(a.overrides) else "",
    )
    t, _ = write(S, a.out)
    print(open(t, encoding="utf-8").read())
    if a.apply:
        with open(a.apply, "w") as f:
            json.dump(honour(A, S), f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
