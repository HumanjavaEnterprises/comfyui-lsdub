# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""vocal_retime.py: word and phone times read off the vocal itself. A sung held vowel stretches a word far past what a speech aligner
expects, and the aligner then spreads the rest of the line: the consonants land where the singer is still on the vowel before.

The vocal fixes the consonants on its own: a voiceless phone is a stretch with no pitch and high-band noise, a closure is a dip in the
level. Each frame reads as voiced, noisy, quiet or silent; a Viterbi pass walks the lyric's phone sequence over those frames, the
vowels free to hold as long as the singer holds them, the consonants held short. The lyric and its order are kept: only times move.
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
BAND = 0.9  # s: a phone stays this near where the aligner put it
ZERO_DB = -45.0  # under the clip's p95: silence
OFF_S = 0.1  # a word this far from the vocal counts as off
HOME = 0.3  # score per frame for a phone inside the aligner's own span: what the vocal does not contradict stays where it was
HOME_PAD = 0.06  # s

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
HISS = {"S", "SH", "F", "TH", "CH", "JH", "HH"}
STOP = {"P", "T", "K"}
SOFT = {"M", "N", "NG", "B", "D", "G"}
MIN_S = dict(V=0.08, H=0.03, S=0.03, N=0.03, W=0.02, Q=0.0)  # shortest stay by class
MOVE = dict(V=0.0, H=-0.1, S=-0.3, N=-0.2, W=-0.3, Q=0.0)  # score per frame of staying past it


def phone_class(p):
    """V vowel, H hiss, S voiceless stop, N a closure that stays voiced, W a voiced glide or a soft fricative (reads as its neighbours)."""
    p = "".join(c for c in str(p).upper() if c.isalpha())
    return "V" if p in VOWELS else "H" if p in HISS else "S" if p in STOP else "N" if p in SOFT else "W"


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


def features(x):
    """dict of per-frame readings at HOP: voiced, hiss, quiet, silent (0..1), and db under the clip's p95."""
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
    k = max(1, int(round(LOCAL / HOP)))
    peak = np.array([db[max(0, i - k) : i + k + 1].max() for i in range(n)])
    sm = lambda a, lo, hi: np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    silent = sm(ZERO_DB - db, -3.0, 6.0)
    voiced = sm(v, 0.35, 0.7) * (1 - silent)
    hiss = np.maximum(sm(hf, 0.02, 0.15), 1 - sm(v, 0.3, 0.6)) * (1 - silent)
    quiet = sm(peak - db, 4.0, 14.0)
    return dict(voiced=voiced, hiss=hiss, quiet=quiet, silent=silent, db=db)


def emissions(fx):
    """per frame, the score of each class. Q is the gap between words."""
    v, h, q, s = fx["voiced"], fx["hiss"], fx["quiet"], fx["silent"]
    return dict(
        V=v * (1 - q) - 1.5 * h * (1 - v) - 2.0 * s - 0.5 * q,
        H=1.6 * h * (1 - v) - v - 0.5 * s,
        S=1.2 * np.maximum(h * (1 - v), q) - v * (1 - q) - 0.3 * s,
        N=0.9 * v * q - 0.6 * v * (1 - q) - h * (1 - v) - s,
        W=0.0 * v,
        Q=1.5 * s + 0.3 * h * (1 - v) - 0.8 * v,
    )


def sequence(words):
    """the state chain: (word index, phone index or -1 for the gap, class, shortest stay in frames), gap states between the words."""
    out = []
    for wi, w in enumerate(words):
        for pi, p in enumerate(w.get("phones") or ()):
            c = phone_class(p["p"])
            out.append((wi, pi, c, max(1, int(round(MIN_S[c] / HOP))), float(p["start"]), float(p["end"])))
        if wi + 1 < len(words):
            out.append((wi, -1, "Q", 1, float(w["end"]), float(words[wi + 1]["start"])))
    return out


def walk(seq, em, t0=0.0, band=BAND):
    """Viterbi over the frames, each phone's chain of shortest-stay states then a held state; a gap may be skipped. It may stop anywhere at
    the last frame. -> for each entry of seq, (first frame, one past the last), or None past the end."""
    n = len(next(iter(em.values())))
    base, pos = [], 0
    for s in seq:
        base.append(pos)
        pos += s[3]
    S = pos
    last = np.array([base[i] + seq[i][3] - 1 for i in range(len(seq))])
    owner = np.zeros(S, int)
    for i, s in enumerate(seq):
        owner[base[i] : base[i] + s[3]] = i
    E = np.stack([em[s[2]] for s in seq], 1)  # n x phones
    tt = t0 + (np.arange(n) + 0.5) * HOP
    a = np.array([s[4] for s in seq]) - band
    b = np.array([s[5] for s in seq]) + band
    move = np.array([MOVE[s[2]] for s in seq])
    E = E + HOME * ((tt[:, None] >= a + band - HOME_PAD) & (tt[:, None] <= b - band + HOME_PAD))
    gap = np.array([s[2] == "Q" for s in seq])
    D = np.full(S, -np.inf)
    D[0] = E[0, 0]
    if gap[0] and len(seq) > 1:
        D[base[1]] = E[0, 1]
    back = np.full((n, S), -1, np.int32)
    idx = np.arange(S)
    for t in range(1, n):
        N_ = np.full(S, -np.inf)
        bk = np.full(S, -1, np.int32)
        inb = (tt[t] >= a) & (tt[t] <= b)
        # within a chain
        inner = np.ones(S, bool)
        inner[np.array(base)] = False
        N_[inner] = D[idx[inner] - 1]
        bk[inner] = idx[inner] - 1
        # the held state stays
        st = D[last] + move
        up = st > N_[last]
        N_[last[up]] = st[up]
        bk[last[up]] = last[up]
        # into each phone from the last one, or over a gap
        for i in range(1, len(seq)):
            c = D[last[i - 1]]
            src = last[i - 1]
            if i >= 2 and gap[i - 1] and D[last[i - 2]] > c:
                c, src = D[last[i - 2]], last[i - 2]
            if c > N_[base[i]]:
                N_[base[i]] = c
                bk[base[i]] = src
        N_ = N_ + E[t, owner]
        N_[~inb[owner]] = -np.inf
        D, back[t] = N_, bk
    k = int(np.argmax(D))
    path = np.zeros(n, int)
    for t in range(n - 1, -1, -1):
        path[t] = owner[k]
        k = back[t, k] if t else k
    span = [None] * len(seq)
    for i in range(len(seq)):
        w = np.flatnonzero(path == i)
        if len(w):
            span[i] = (int(w[0]), int(w[-1]) + 1)
    return span


def retime(align, audio, x=None):
    """the alignment with its phones' and words' times read off the vocal. Words without phones keep theirs."""
    x = read_mono(audio) if x is None else np.asarray(x, float)
    fx = features(x)
    n = len(fx["db"])
    W = [w for w in align["words"] if w.get("phones")]
    if not W or n < 10:
        return dict(align)
    seq = sequence(W)
    span = walk(seq, emissions(fx))
    end = n * HOP
    times = {}
    for s, sp in zip(seq, span):
        if s[1] >= 0:
            times[s[0], s[1]] = None if sp is None else (sp[0] * HOP, sp[1] * HOP)
    out = dict(align, retimed="vocal")
    new = []
    cur = 0.0
    k = 0
    for w in align["words"]:
        if not w.get("phones"):
            new.append(dict(w))
            continue
        ph = []
        for pi, p in enumerate(w["phones"]):
            t = times[k, pi]
            if t is None:  # past the vocal's end, or a phone the walk skipped: no time of its own
                d = max(0.01, float(p["end"]) - float(p["start"]))
                t = (max(cur, end), max(cur, end) + d) if cur >= end - 2 * HOP or pi == 0 else (cur, cur)
            cur = max(cur, t[1])
            ph.append(dict(p, start=round(t[0], 3), end=round(t[1], 3)))
        new.append(dict(w, start=ph[0]["start"], end=ph[-1]["end"], start0=w["start"], end0=w["end"], phones=ph))
        k += 1
    out["words"] = new
    return out


def word_offsets(words, vocal_words, t0, t1, far=OFF_S):
    """the plan's words against where the vocal has them, on words starting in t0..t1 (matched by order): the median and worst start
    offset, ms (+ = the plan is late), and how many sit past `far`. {} with no words. A reading, it moves nothing."""
    pairs = [(a, b) for a, b in zip(words, vocal_words) if t0 <= b["start"] < t1 and a.get("phones")]
    if not pairs:
        return {}
    d = np.array([float(a["start"]) - float(b["start"]) for a, b in pairs])
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
