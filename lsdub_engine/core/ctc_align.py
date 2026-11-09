# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""ctc_align.py: the lyric's phones forced onto a CTC recogniser's frames (`lsdub lyric-align --aligner ctc`). Pure: arrays in, words out.

The speech-to-text words start sung words early by a few hundred ms, and a speech aligner has no model of a held vowel; this
aligns the lyric's phones (CMUdict, through the driver's phone network vocabulary) directly on the recogniser's per-frame
log-posteriors with a monotone, blank-aware Viterbi, then:
  - moves every boundary CORR_S earlier (a CTC spike lands that late on the acoustic onset),
  - lets the phones of a word abut (a CTC spike is followed by blank frames; a vowel's are its sustain, and a gap inside a word would
    read as silence downstream),
  - anchors each word's first vowel to the nearest detected nucleus within ANCHOR_S when there is one, the word's leading
    consonants moving with it (a melisma slide the DP parked late is pulled back onto the sung onset); a word with no nucleus near
    keeps its CTC time.

  viterbi(logp, seq, blank)                      -> [(start row, end row)] per token, end exclusive
  align(logp, words, blank, t0, nuclei=None)     -> [{w, start, end, align: "ctc", phones: [{p, start, end}]}]
      words = [(word, [(ARPAbet phone, vocabulary id), ...]), ...] in lyric order; nuclei in the same clock as the output (t0 added)
"""

import numpy as np

FRAME = 0.02  # s per recogniser row
CORR_S = 0.02  # s every boundary moves earlier
ANCHOR_S = 0.08  # s: a word's first vowel snaps to a nucleus this near
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
NEG = -1e18


def viterbi(logp, seq, blank):
    """monotone CTC alignment of token ids seq onto the rows of logp. States: blank, tok0, blank, tok1, ..., blank; a blank between
    two different tokens may be skipped, the one between two equal tokens may not. -> [(start row, end row)] per token."""
    logp = np.asarray(logp, float)
    T, n = logp.shape[0], len(seq)
    if n == 0 or T == 0:
        return []
    S = 2 * n + 1
    tok = np.array([blank if j % 2 == 0 else int(seq[j // 2]) for j in range(S)])
    can_skip = np.zeros(S, bool)
    for j in range(3, S, 2):
        can_skip[j] = seq[j // 2] != seq[j // 2 - 1]
    emit = logp[:, tok]
    D = np.full((T, S), NEG)
    back = np.zeros((T, S), np.int8)
    D[0, 0], D[0, 1] = emit[0, 0], emit[0, 1]
    for t in range(1, T):
        p = D[t - 1]
        stay = p
        step = np.concatenate([[NEG], p[:-1]])
        skip = np.where(can_skip, np.concatenate([[NEG, NEG], p[:-2]]), NEG)
        cands = np.stack([stay, step, skip])
        k = cands.argmax(0)
        D[t] = cands[k, np.arange(S)] + emit[t]
        back[t] = k
    j = S - 1 if D[T - 1, S - 1] >= D[T - 1, S - 2] else S - 2
    spans = {}
    for t in range(T - 1, -1, -1):
        if j % 2:
            i = j // 2
            a, b = spans.get(i, (t, t + 1))
            spans[i] = (t, max(b, t + 1))
        j -= int(back[t, j])
    return [spans.get(i, (0, 0)) for i in range(n)]


def _first_vowel(phones):
    return next((k for k, p in enumerate(phones) if p["p"].rstrip("012") in VOWELS), None)


def anchored(words, nuclei, anchor=ANCHOR_S, frame=FRAME):
    """each word's first vowel start moved onto the nearest nucleus within anchor, its leading consonants with it; nothing moves
    further than anchor, no word starts before the one before it ends by more than it already did. In place; -> the words."""
    nuclei = sorted(float(t) for t in (nuclei or []))
    if not nuclei:
        return words
    prev_end = None
    for w in words:
        ph = w["phones"]
        k = _first_vowel(ph)
        if k is None:
            prev_end = w["end"]
            continue
        v0 = ph[k]["start"]
        near = min(nuclei, key=lambda t: abs(t - v0))
        d = near - v0
        if abs(d) <= anchor and d != 0.0:
            if prev_end is not None and d < 0:  # moving earlier: not past the previous word's end
                d = max(d, min(0.0, prev_end - w["start"]))
            lo = ph[k]["end"] - frame
            for p in ph[:k]:
                p["start"], p["end"] = round(min(lo, p["start"] + d), 3), round(min(lo, p["end"] + d), 3)
            ph[k]["start"] = round(min(lo, v0 + d), 3)
            w["start"] = ph[0]["start"]
            w["anchor_s"] = round(d, 3)
        prev_end = w["end"]
    return words


def align(logp, words, blank, t0=0.0, nuclei=None, frame=FRAME, corr=CORR_S, anchor=ANCHOR_S):
    """the lyric words on the frames. words = [(word, [(phone, token id), ...])]; nuclei (s, same clock as the output) anchor the
    first vowels. [] with no words, no phones or no frames."""
    words = [(w, list(ps)) for w, ps in words if ps]
    logp = np.asarray(logp, float)
    if not words or logp.ndim != 2 or logp.shape[0] < 2:
        return []
    seq = [tok for _, ps in words for _, tok in ps]
    spans = viterbi(logp, seq, blank)
    out, k = [], 0
    for w, ps in words:
        phs = []
        for p, _ in ps:
            a, b = spans[k]
            k += 1
            phs.append({"p": p, "start": round(max(0.0, a * frame - corr) + t0, 3), "end": round(max(0.0, b * frame - corr) + t0, 3)})
        for p, q in zip(phs, phs[1:]):  # the phones of a word abut: a CTC spike's blank frames are the phone's own sustain
            p["end"] = max(p["end"], q["start"])
        out.append({"w": w, "start": phs[0]["start"], "end": phs[-1]["end"], "align": "ctc", "phones": phs})
    return anchored(out, nuclei, anchor, frame) if nuclei else out
