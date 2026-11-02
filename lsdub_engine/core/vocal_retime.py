# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""vocal_retime.py: word and phone times read off the vocal itself. A sung held vowel stretches a word far past what a speech aligner
expects, and the aligner then spreads the rest of the line: the consonants land where the singer is still on the vowel before.

The vocal fixes the consonants on its own: a voiceless phone is a stretch with no pitch and high-band noise, a closure is a dip in the
level, a sung nasal is the low band standing over the mid band with no dip at all. Each frame reads as voiced, noisy, quiet, under the
phrase, nasal or silent; a Viterbi pass walks the lyric's phone sequence over those frames, the vowels free to hold as long as the singer
holds them and sitting on the loud part, the glides holding the quiet run-in before them, the consonants held short. The walk is free at
both ends: leading and trailing silence belong to neither the first nor the last word. The lyric and its order are kept: only times move.
What the vocal never reaches (a clip cut before the last word ends) stays past its end.

  python -m lsdub_engine.core.vocal_retime --align align.json --audio lead.wav --out align_retimed.json

retime(align, audio) -> the alignment with new times, each word's old ones in start0 / end0. Nothing else in the file changes."""

import argparse, json
import numpy as np
from scipy.signal import stft

from ..io import ff

SR = 16000
HOP = 0.01  # s
WIN = 0.04  # s, the pitch window
F0_LO, F0_HI = 80.0, 700.0
HF_HZ = 3500.0
LOCAL = 0.15  # s each way: a frame's level reads against the loudest near it
PHRASE = 0.5  # s each way: the same against the phrase, where a slow glide's run-in reads as under the vowel it leads to
RATIO = 0.5  # s each way: a frame's low-to-mid ratio reads against the usual open sound near it
LO_HZ, MID_HZ, TOP_HZ = 500.0, 700.0, 2400.0
BAND = 1.5  # s: a phone is charged for going further than this from where the aligner put it
OUTSIDE = 0.6  # score per frame out there. A charge and not a wall: no alignment, however wrong, leaves the walk with no path
ZERO_DB = -45.0  # under the clip's p95: silence
FLOOR_DB, HAS_GAPS_DB = (
    6.0,
    -30.0,
)  # a noisy clip's silence: this far over its quietest twentieth, when that sits this far under its p95 (a clip with no gaps has no floor to read)
PASS = "pass-over"  # a word the lead sheet passed: the walk leaves it where it is
LOST_S = 0.6  # a shot whose words sit this far off, median: a free walk that has run away on a long song, or a plan that lost it. Not a number to report
OFF_S = 0.1  # a word this far from the vocal counts as off
HOME = (
    1.5  # score of a phone starting where the aligner started it, once, however long it then stays: what the vocal says nothing about stays
)
HOME_S = 0.4  # s: that score falls to nothing this far off

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
HISS = {"S", "SH", "F", "TH", "CH", "JH", "HH"}
STOP = {"P", "T", "K"}
SOFT = {"M", "N", "NG", "B", "D", "G"}
GLIDE = {"W", "Y", "R", "L"}
MIN_S = dict(V=0.08, H=0.05, S=0.03, N=0.03, W=0.02, G=0.04, Q=0.0)  # shortest stay by class
MOVE = dict(V=0.0, H=-0.1, S=-0.3, N=-0.3, W=-0.3, G=-0.03, Q=0.0, O=0.0)  # score per frame of staying past it
SHARE = 0.35  # a vowel keeps at least this share of the length the aligner gave it: a long one is not squeezed out to feed its neighbours
CLOSED = 50.0  # score per frame of a phone in time the lead sheet passed over
EARN = 3.0  # a hiss has to beat this to be placed: sung lines drop them and an alignment can have the wrong word, and a blip earns no phone
CUT = 1.0  # score per phone left unsung when the walk stops early: the vocal ended on the lyric, with silence or the clip's end after it
UNSUNG = 12.0  # ... and per phone the alignment has inside the audio: a word the aligner heard is not dropped past the end to feed the vowels before it


def phone_class(p):
    """V vowel, H hiss, S voiceless stop, N a closure or nasal that stays voiced, G a glide, W a soft fricative (reads as its neighbours)."""
    p = "".join(c for c in str(p).upper() if c.isalpha())
    return "V" if p in VOWELS else "H" if p in HISS else "S" if p in STOP else "N" if p in SOFT else "G" if p in GLIDE else "W"


def read_mono(path):
    raw = ff.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def voicing(x):
    """per frame: the autocorrelation peak in the pitch range, 0..1."""
    N, H = int(WIN * SR), int(HOP * SR)
    lo, hi = int(SR / F0_HI), int(SR / F0_LO)
    n = max(0, (len(x) - N) // H + 1)
    if not n:
        return np.zeros(0)
    fr = np.stack([x[k * H : k * H + N] for k in range(n)]) * np.hanning(N)
    a = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * N, axis=1)) ** 2, axis=1)[:, : hi + 1]
    z = np.maximum(a[:, 0], 1e-12)
    return np.clip(a[:, lo:].max(1) / z, 0.0, 1.0) * (z > 1e-10)


def nearby(a, k, q):
    """per frame, the q-th percentile of a within k frames each way."""
    return np.array([np.percentile(a[max(0, i - k) : i + k + 1], q) for i in range(len(a))])


def zero_db(db):
    """what reads as silence, in db under the clip's p95: ZERO_DB, or just over a noise floor that sits above it."""
    q = float(np.percentile(db, 5))
    return max(ZERO_DB, q + FLOOR_DB) if q < HAS_GAPS_DB else ZERO_DB


def features(x):
    """dict of per-frame readings at HOP: voiced, hiss, quiet, under, nasal, silent (0..1), and db under the clip's p95. A clip shorter than a
    window reads as one window."""
    x = np.pad(np.asarray(x, float), (0, max(0, int(WIN * SR) + 2 * int(HOP * SR) - len(x))))
    v = voicing(x)
    n = len(v)
    N, H = int(WIN * SR), int(HOP * SR)
    f, _, Z = stft(x, SR, nperseg=N, noverlap=N - H, nfft=1024, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    n = min(n, P.shape[1])
    v, P = v[:n], P[:, :n]
    total = P.sum(0) + 1e-12
    db = 10 * np.log10(total)
    db -= np.percentile(db, 95)
    hf = P[f >= HF_HZ].sum(0) / total
    sm = lambda a, lo, hi: np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    silent = sm(zero_db(db) - db, -3.0, 6.0)
    voiced = sm(v, 0.35, 0.7) * (1 - silent)
    hiss = np.maximum(sm(hf, 0.02, 0.15), 1 - sm(v, 0.3, 0.6)) * (1 - silent)
    quiet = sm(nearby(db, max(1, int(round(LOCAL / HOP))), 100) - db, 4.0, 14.0)
    under = sm(nearby(db, max(1, int(round(PHRASE / HOP))), 100) - db, 6.0, 14.0)
    ratio = 10 * np.log10((P[f < LO_HZ].sum(0) + 1e-12) / (P[(f >= MID_HZ) & (f < TOP_HZ)].sum(0) + 1e-12))
    nasal = sm(ratio - nearby(ratio, max(1, int(round(RATIO / HOP))), 25), 8.0, 16.0) * voiced
    return dict(voiced=voiced, hiss=hiss, quiet=quiet, under=under, nasal=nasal, silent=silent, db=db)


def emissions(fx):
    """per frame, the score of each class. Q is the gap between words, O what lies outside the lyric: sound there costs nothing."""
    v, h, q, u, m, s = fx["voiced"], fx["hiss"], fx["quiet"], fx["under"], fx["nasal"], fx["silent"]
    c = np.maximum(q, m)  # a closure: the level dips, or the sound has gone to the low band
    return dict(
        V=v * (1 - c) * (1 - 0.6 * u) - 1.5 * h * (1 - v) - 2.0 * s - 0.5 * c,
        H=1.6 * h * (1 - v) - v - 0.5 * s,
        S=1.2 * np.maximum(h * (1 - v), q) - v * (1 - q) - 0.3 * s,
        N=0.8 * v * c - 0.6 * v * (1 - c) - h * (1 - v) - s,
        W=0.0 * v,
        G=0.5 * v * u - 0.2 * v * (1 - u) - h * (1 - v) - s,
        Q=1.5 * s + 0.3 * h * (1 - v) - 0.8 * v,
        O=1.5 * s + 0.3 * h * (1 - v),
    )


def sequence(words, free=False):
    """the state chain: (word index, phone index or -1 for the gap, class, shortest stay in frames), gap states between the words. free: no
    length of the alignment's: a vowel's shortest stay is its class's."""
    out = []
    for wi, w in enumerate(words):
        for pi, p in enumerate(w.get("phones") or ()):
            c = phone_class(p["p"])
            least = max(MIN_S[c], SHARE * (float(p["end"]) - float(p["start"]))) if c == "V" and not free else MIN_S[c]
            out.append((wi, pi, c, max(1, int(round(least / HOP))), float(p["start"]), float(p["end"])))
        if wi + 1 < len(words):
            out.append((wi, -1, "Q", 1, float(w["end"]), float(words[wi + 1]["start"])))
    return out


def walk(seq, em, t0=0.0, band=BAND, closed=None, home=None):
    """Viterbi over the frames, each phone's chain of shortest-stay states then a held state; a gap may be skipped, and an open gap before
    the first phone and after the last takes what the lyric does not, so the walk starts and stops anywhere. A consonant the vocal never
    sounds may be skipped at a price. It may also stop mid-lyric at the last
    frame. -> for each entry of seq, (first frame, one past the last), or None past the end."""
    n = len(next(iter(em.values())))
    seq = [(-1, -1, "O", 1, -np.inf, np.inf), *seq, (-1, -1, "O", 1, -np.inf, np.inf)]  # the two open ends
    last = np.cumsum([s[3] for s in seq]) - 1
    base = last - np.array([s[3] for s in seq]) + 1
    S = int(last[-1]) + 1
    owner = np.repeat(np.arange(len(seq)), [s[3] for s in seq])
    E = np.stack([em[s[2]] for s in seq], 1)  # n x phones
    tt = t0 + np.arange(n) * HOP + WIN / 2
    a = np.array([s[4] for s in seq])
    b = np.array([s[5] for s in seq])
    move = np.array([MOVE[s[2]] for s in seq])
    E = E - OUTSIDE * ((tt[:, None] < a - band) | (tt[:, None] > b + band))
    if closed is not None:
        hold = np.array([s[2] in "QO" for s in seq])
        E[closed] = np.where(hold, 0.0, -CLOSED)
    near = (HOME if home is None else home) * np.maximum(0.0, 1 - np.abs(tt[:, None] - np.nan_to_num(a, neginf=-1e9, posinf=1e9)) / HOME_S)[
        :, 1:
    ]  # n x entries from the second
    gap = np.array([s[2] in "QO" for s in seq])
    over_cost = np.where(
        gap[:-1], 0.0, np.where(np.array([s[2] == "H" for s in seq[:-1]]), EARN, -np.inf)
    )  # entry i - 1 stepped over to reach i
    sung = np.array([s[2] not in "QO" for s in seq])  # phones
    inside = np.isfinite(a) & (a > -1e5) & (a < tt[-1])  # (a free walk's phones have no time of their own to read)
    left = np.cumsum((sung & inside)[::-1])[::-1]  # from each entry on, the phones the alignment has inside the audio
    quit_ = -CUT * (len(seq) - 2 - np.arange(len(seq) - 1)) - UNSUNG * left[1:]
    D = np.full(S, -np.inf)
    D[0] = E[0, 0]
    D[base[1]] = E[0, 1] + near[0, 0]
    back = np.full((n, S), -1, np.int32)
    inner = np.ones(S, bool)
    inner[base] = False
    for t in range(1, n):
        N_ = np.full(S, -np.inf)
        bk = np.full(S, -1, np.int32)
        # within a chain
        N_[inner] = D[np.flatnonzero(inner) - 1]
        bk[inner] = np.flatnonzero(inner) - 1
        # the held state stays
        st = D[last] + move
        up = st > N_[last]
        N_[last[up]] = st[up]
        bk[last[up]] = last[up]
        # into each phone from the last one, or over a gap
        c, src = D[last[:-1]].copy(), last[:-1].copy()
        alt = np.r_[-np.inf, D[last[:-2]] + over_cost[1:]]
        over = alt > c
        c[over], src[over] = alt[over], last[np.flatnonzero(over) - 1]
        c = c + near[t]
        up = c > N_[base[1:]]
        N_[base[1:][up]] = c[up]
        bk[base[1:][up]] = src[up]
        stop = D[last[:-1]] + quit_
        j = int(np.argmax(stop))
        if stop[j] > N_[base[-1]]:
            N_[base[-1]], bk[base[-1]] = stop[j], last[j]
        D, back[t] = N_ + E[t, owner], bk
    k = int(
        np.argmax(D - UNSUNG * np.r_[left[1:], 0][owner])
    )  # the last frame is a stop too: what the alignment heard is not left past it for free
    path = np.zeros(n, int)
    for t in range(n - 1, -1, -1):
        path[t] = owner[k]
        k = back[t, k] if t else k
    span = [None] * len(seq)
    for i in range(len(seq)):
        w = np.flatnonzero(path == i)
        if len(w):
            span[i] = (int(w[0]), int(w[-1]) + 1)
    return span[1:-1]


def retime(align, audio, x=None, free=False):
    """the alignment with its phones' and words' times read off the vocal. Words without phones keep theirs. free: the lyric alone is walked, no
    time of the alignment's used, for a reading of a plan that must not lean on the plan."""
    x = read_mono(audio) if x is None else np.asarray(x, float)
    fx = features(x)
    n = len(fx["db"])
    W = [w for w in align["words"] if w.get("phones") and w.get("vocal") != PASS]
    if not W or n < 10:
        return dict(align)
    seq = sequence(W, free)
    closed = None
    if align.get("pass_over"):
        tt = np.arange(n) * HOP + WIN / 2
        closed = np.zeros(n, bool)
        for a, b in align["pass_over"]:
            closed |= (tt >= a) & (tt < b)
    if free:
        seq = [(*q[:4], -1e6, 1e6) for q in seq]
    span = walk(seq, emissions(fx), closed=closed, home=0.0 if free else HOME)
    end = len(x) / SR
    edge = lambda k: 0.0 if k == 0 else end if k == n else k * HOP + (WIN - HOP) / 2  # frame k's window centres at k * HOP + WIN / 2
    times = {}
    for s, sp in zip(seq, span):
        if s[1] >= 0:
            times[s[0], s[1]] = None if sp is None else (edge(sp[0]), edge(sp[1]))
    out = dict(align, retimed="vocal")
    new = []
    cur = 0.0
    k = 0
    reached = max((key for key, t in times.items() if t is not None), default=(-1, -1))
    for w in align["words"]:
        if not w.get("phones") or w.get("vocal") == PASS:
            new.append(dict(w))
            continue
        ph = []
        for pi, p in enumerate(w["phones"]):
            t = times[k, pi]
            if t is None and (k, pi) > reached:  # past the vocal's end: each keeps its own length out there
                d = max(0.01, float(p["end"]) - float(p["start"]))
                t = (cur, cur + d)
            elif t is None:  # a consonant the vocal never sounds: no time of its own
                t = (cur, cur)
            cur = max(cur, t[1])
            ph.append(dict(p, start=round(t[0], 3), end=round(t[1], 3)))
        new.append(dict(w, start=ph[0]["start"], end=ph[-1]["end"], start0=w["start"], end0=w["end"], phones=ph))
        k += 1
    out["words"] = new
    return out


def plan_hits(words, x, t0, t1, fx=None):
    """the plan read against the vocal's own frames, no walk and no alignment: of its phones starting in t0..t1 the share of hiss time on hiss, of closures with
    a dip or a nasal in them, of vowel time on open voiced sound. {} with none. A reading, it moves nothing. fx: features(x), when the caller has them."""
    fx = features(x) if fx is None else fx
    n = len(fx["db"])
    tt = np.arange(n) * HOP + WIN / 2
    hit = dict(H=fx["hiss"] * (1 - fx["voiced"]), N=np.maximum(fx["quiet"], fx["nasal"]), V=fx["voiced"] * (1 - fx["quiet"]))
    got = dict(H=[], N=[], V=[])
    for w in words:
        if w.get("vocal") == PASS:
            continue
        for p in w.get("phones") or ():
            a, b = float(p["start"]), float(p["end"])
            c = phone_class(p["p"])
            c = "N" if c == "S" else c
            if t0 <= a < t1 and c in got and b > a:
                m = (tt >= a) & (tt < b)
                if m.any():
                    got[c].append(float(hit[c][m].max() if c == "N" else hit[c][m].mean()))
    return {f"vocal_{k}": round(float(np.mean(v)), 2) for k, v in zip(("hiss", "shut", "open"), (got["H"], got["N"], got["V"])) if v}


def word_offsets(words, vocal_words, t0, t1, far=OFF_S):
    """the plan's words against where the vocal has them, on words starting in t0..t1 (matched by order): the median and worst start
    offset, ms (+ = the plan is late), and how many sit past `far`. {} with no words. A reading, it moves nothing."""
    pairs = [(a, b) for a, b in zip(words, vocal_words) if t0 <= b["start"] < t1 and a.get("phones") and a.get("vocal") != PASS]
    if not pairs:
        return {}
    d = np.array([float(a["start"]) - float(b["start"]) for a, b in pairs])
    if np.median(np.abs(d)) > LOST_S:
        return dict(vocal_lost=True)
    k = int(np.argmax(np.abs(d)))
    return dict(
        vocal_off_ms=int(round(1000 * float(np.median(np.abs(d))))),
        vocal_off_n=int((np.abs(d) > far).sum()),
        vocal_words=len(d),
        vocal_worst=f"{pairs[k][0]['w']} {int(round(1000 * float(d[k]))):+d}",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = retime(json.load(open(a.align)), a.audio)
    json.dump(r, open(a.out, "w"), indent=1)
    print(f"vocal_retime: {sum('start0' in w for w in r['words'])} words retimed -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
