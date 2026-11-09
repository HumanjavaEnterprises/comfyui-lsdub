# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""judge_phones.py: the event judge's own phone front end. CPU; the one model it reads is a public wav2vec2 CTC phone recogniser.

event_judge reads a rendered mouth against EVENTS taken from the vocal stem: the sung syllable onsets, the p/b/m lip closures of the
lyric and the wide-vowel nuclei. Every event comes from here, from the judge's own code, and nothing here is shared with the dub's
driver (the driver has its own phone network and its own aligner; the two must never import each other). tests/test_judge_frozen.py
holds the import rules and pins what this produces on fixtures.

  Phones(cache_dir)      the recogniser, loaded by local path when its files are there (no hub call), by repo id otherwise
  logposteriors(x, P)    CTC log-posteriors at FRAME s per row for samples x at SR
  segments(logp, P)      argmax runs: [(label, start, end)]
  nuclei(logp, x, P)     syllable nuclei: a vowel-token start gated to one per nucleus (after a consonant or silence, or after a
                         vowel when the level dipped DIP_DB between the two), sung frames only, MIN_GAP apart
  flux_onsets(x)         lyric-free onsets: half-wave log-spectral flux in BAND, adaptive median threshold; the fallback with no model
  onsets(x, logp, P)     the two detectors AGREEING (a nucleus, CORR_S earlier, and a flux onset within MIN_GAP; the earlier time
                         kept); flux alone when there is no model
  align_lyric(logp, words, P)   the lyric's phones on the frames: a blank-aware Viterbi, every boundary CORR_S earlier
  confirmed(onsets, words)      the onsets with an aligned vowel start within CONFIRM_S, one per vowel (the nearest), nothing added
  edge(ts, dur)          events inside the first or last EDGE_S dropped: no mouth can rise before its second frame
  closures(words)        the aligned P/B/M onsets; wide(words) the aligned wide-vowel starts
  sung_mask(x)           10 ms frames where the voice is on (level over its floor)

Change nothing here to improve a dub. A change here is a change of the instrument, and every reading taken before it stops being
comparable with every reading taken after it.
"""

import json
import os
import re

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import stft

from .._models import hf_snapshot

MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
NEED = ("config.json", "vocab.json", "preprocessor_config.json", "pytorch_model.bin")
SR = 16000
FRAME = 0.02  # s, the recogniser's row
HOP = 0.01  # s, the level and sung-mask rows
WIN = 0.04  # s, the level window
FLOOR_DB = -35.0  # under the clip's p95: the voice floor
SUNG_DB = -10.0  # a frame this far under the floor still counts as sung (the gate's slack)
NUC_DB = -6.0  # a nucleus must sit over the floor less this
DIP_DB = 3.0  # a vowel after a vowel is a new nucleus when the level dipped this much between them
MIN_GAP = 0.08  # s between two onsets
CORR_S = 0.02  # s: a CTC spike lands this late on the acoustic onset; every aligned boundary moves this much earlier
CONFIRM_S = 0.10  # s: a detected onset counts when an aligned vowel starts within this
EDGE_S = 0.10  # s at either end of a clip where no event is read
BAND = (200.0, 4000.0)  # Hz, the flux band
NFFT, FHOP = 768, 160  # the flux STFT: 48 ms window, 10 ms hop at SR
FLUX_WIN, FLUX_K, FLUX_FLOOR, FLUX_PEAK = 0.3, 1.5, 0.08, 0.1  # s median window, its gain, its floor, s peak window

IPA_VOWELS = set("aeiouyɐɑɒæɔəɘɚɛɜɝɞɤɨɪɯɵøœɶʉʊʌʏ")
IPA_WIDE = set("aɑæʌɐɔ")  # a wide-open vowel by its first letter
ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
ARPA_WIDE = {"AA", "AE", "AH", "AO", "AY", "AW"}
CLOSERS = ("P", "B", "M")
ARPA2IPA = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ",
    "ER": "ɚ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}  # fmt: skip
IPA_ALT = {"ɑː": "ɑ", "ʌ": "ə", "ɔː": "ɔ", "ɚ": "ɜ", "iː": "i", "uː": "u", "ɡ": "g", "ɹ": "r", "oʊ": "o", "eɪ": "e"}
# a letter's phone when a word is in no dictionary
LETTERS = {
    "a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F", "g": "G", "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "o": "OW", "p": "P", "q": "K", "r": "R", "s": "S", "t": "T", "u": "AH", "v": "V", "w": "W", "x": "S", "y": "IY", "z": "Z",
}  # fmt: skip


def models_dir(cache_dir=None):
    """where the recogniser's files are looked for: cache_dir, else LSDUB_PHONE_MODELS, else LSDUB_HF_MODELS/wav2vec2-espeak."""
    if cache_dir:
        return cache_dir
    if os.environ.get("LSDUB_PHONE_MODELS"):
        return os.environ["LSDUB_PHONE_MODELS"]
    if os.environ.get("LSDUB_HF_MODELS"):
        return os.path.join(os.environ["LSDUB_HF_MODELS"], "wav2vec2-espeak")
    return None


def model_source(cache_dir=None):
    """the local snapshot path when the files are there, else the repo id (loads over the network)."""
    return hf_snapshot(models_dir(cache_dir), MODEL, NEED) or MODEL


class Phones:
    """the recogniser and its vocabulary. torch and transformers load here, not at import."""

    def __init__(self, cache_dir=None, device="cpu"):
        from transformers import AutoModelForCTC

        self.source = model_source(cache_dir)
        self.model = AutoModelForCTC.from_pretrained(self.source).eval().to(device)
        self.device = device
        vocab = self.model.config.to_dict().get("vocab") or _vocab(self.source)
        self.labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        self.blank = int(vocab.get("<pad>", 0))
        self.specials = {i for i, t in enumerate(self.labels) if t.startswith("<")}
        self.index = {t: i for i, t in enumerate(self.labels)}

    def tok(self, arpa):
        """the vocabulary id of an ARPAbet phone, by its IPA spelling, an alternative spelling, its first letter, else <unk>."""
        ipa = ARPA2IPA.get(arpa, "")
        for c in (ipa, IPA_ALT.get(ipa, ""), ipa[:1]):
            if c in self.index:
                return self.index[c]
        return self.index.get("<unk>", 3)


def _vocab(source):
    path = os.path.join(source, "vocab.json") if os.path.isdir(source) else None
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source).get_vocab()


def logposteriors(x, P):
    """(T, V) log-softmax rows at FRAME for samples x at SR (any level: normalised here as the recogniser was trained)."""
    import torch

    x = np.asarray(x, np.float32)
    x = (x - x.mean()) / (x.std() + 1e-7)
    with torch.no_grad():
        lg = P.model(torch.from_numpy(x)[None].to(P.device)).logits[0]
        return torch.log_softmax(lg.float(), -1).cpu().numpy().astype(np.float64)


def segments(logp, P):
    """argmax runs of one label: [(label, start s, end s)]; blank rows end nothing, a repeat after a blank is a new run."""
    ids = np.asarray(logp).argmax(-1)
    out = []
    for i, k in enumerate(ids):
        if k in P.specials:
            continue
        lab, t = P.labels[k], round(i * FRAME, 3)
        if out and out[-1][0] == lab and t - out[-1][2] <= FRAME * 1.5:
            out[-1][2] = round(t + FRAME, 3)
        else:
            out.append([lab, t, round(t + FRAME, 3)])
    return [tuple(s) for s in out]


def level_db(x):
    """RMS level per HOP (WIN window) in dB under the clip's p95."""
    x = np.asarray(x, float)
    N, H = int(WIN * SR), int(HOP * SR)
    n = max(1, (len(x) - N) // H + 1)
    x = np.pad(x, (0, max(0, (n - 1) * H + N - len(x))))
    fr = np.lib.stride_tricks.as_strided(x, (n, N), (x.strides[0] * H, x.strides[0]))
    db = 10 * np.log10(np.mean(fr * fr, 1) + 1e-12)
    return db - np.percentile(db, 95)


def sung_mask(x):
    """True per HOP where the voice is on: level over FLOOR_DB + SUNG_DB."""
    return level_db(x) > FLOOR_DB + SUNG_DB


def _dedupe(ts, gap):
    out = []
    for t in sorted(ts):
        if not out or t - out[-1] >= gap:
            out.append(round(float(t), 3))
    return out


def nuclei(logp, x, P):
    """syllable nuclei (s): vowel-run starts, one per nucleus, on sung frames."""
    db = level_db(x)
    out, prev = [], None
    for s, a, b in segments(logp, P):
        if s[0] in IPA_VOWELS:
            ok = prev is None or prev[0][0] not in IPA_VOWELS
            if not ok:
                seg = db[int(prev[1] / HOP) : int(a / HOP) + 1]
                ok = len(seg) > 2 and (max(seg[0], seg[-1]) - seg.min()) >= DIP_DB and seg.argmin() not in (0, len(seg) - 1)
            if ok and db[min(int(a / HOP) + 2, len(db) - 1)] > FLOOR_DB + NUC_DB and (not out or a - out[-1] >= MIN_GAP):
                out.append(round(float(a), 3))
        prev = (s, a, b)
    return out


def flux_curve(x):
    """(flux 0..1 per FHOP row, row times s): half-wave rectified log-spectral flux in BAND."""
    x = np.asarray(x, float)
    if len(x) < NFFT:
        x = np.pad(x, (0, NFFT - len(x)))
    f, t, Z = stft(x, SR, nperseg=NFFT, noverlap=NFFT - FHOP, boundary=None, padded=False)
    L = np.log1p(100 * np.abs(Z[(f >= BAND[0]) & (f <= BAND[1])]))
    d = np.diff(L, axis=1, prepend=L[:, :1])
    flux = np.maximum(d, 0).sum(0)
    return flux / (flux.max() + 1e-9), t


def flux_onsets(x):
    """onsets (s) off the flux curve: peaks over FLUX_K x the local median + FLUX_FLOOR, one per FLUX_PEAK window, MIN_GAP apart."""
    flux, t = flux_curve(x)
    w = int(FLUX_WIN * SR / FHOP) | 1
    thr = median_filter(flux, size=w) * FLUX_K + FLUX_FLOOR
    cand = (flux > thr) & (flux == maximum_filter1d(flux, size=int(FLUX_PEAK * SR / FHOP) | 1))
    return _dedupe(t[cand], MIN_GAP)


def agreed(nuc, flux):
    """the onsets both detectors found: a nucleus (CORR_S earlier) with a flux onset within MIN_GAP, the earlier time of the two."""
    nuc = [round(t - CORR_S, 3) for t in nuc]
    both = [t for t in nuc if any(abs(t - f) <= MIN_GAP for f in flux)] + [f for f in flux if any(abs(f - t) <= MIN_GAP for t in nuc)]
    return _dedupe(both, MIN_GAP)


def onsets(x, logp=None, P=None):
    """the judge's onset set: the two detectors agreeing; flux alone with no model."""
    fl = flux_onsets(x)
    return fl if logp is None or P is None else agreed(nuclei(logp, x, P), fl)


def edge(ts, dur):
    """the times inside [EDGE_S, dur - EDGE_S]."""
    return [t for t in ts if EDGE_S <= t <= dur - EDGE_S]


# -- the lyric ------------------------------------------------------------------------------------------------------------------------
_DICT = {}


def cmudict():
    """word -> ARPAbet phones from the CMU dictionary soundswallower ships; {} when that package is absent."""
    if _DICT:
        return _DICT
    try:
        import soundswallower

        path = os.path.join(os.path.dirname(soundswallower.__file__), "model", "en-us", "dict.txt")
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts or parts[0].startswith(";;"):
                    continue
                w = re.sub(r"\(\d+\)$", "", parts[0].lower())
                _DICT.setdefault(w, [re.sub(r"\d", "", p).upper() for p in parts[1:]])
    except (ImportError, OSError):
        _DICT.setdefault("", [])
    return _DICT


def norm(w):
    return re.sub(r"[^a-z']", "", str(w).lower())


def phones_of(word):
    """ARPAbet phones of a lyric word: the dictionary, the word without its apostrophe, else one phone per letter."""
    d, w = cmudict(), norm(word)
    ps = d.get(w) or d.get(w.replace("'", ""))
    return list(ps) if ps else [LETTERS[c] for c in w if c in LETTERS] or ["AH"]


def window(lines, heard):
    """the lyric words between the first and last word a transcript heard (a cut that starts mid-line): lines of text, heard words."""
    toks = [norm(w) for line in lines for w in str(line).split() if norm(w)]
    heard = [norm(w) for w in heard if norm(w)]
    if not heard:
        return toks
    first = next((i for i, w in enumerate(toks) if w == heard[0]), 0)
    last = max((i for i, w in enumerate(toks) if w == heard[-1]), default=len(toks) - 1)
    return toks[first : last + 1] if last >= first else toks


def viterbi(logp, seq, blank):
    """monotone CTC alignment of token ids seq to the rows of logp: states blank, tok0, blank, tok1, ... (a blank may be skipped, a
    repeated token may not skip its blank). -> [(start row, end row)] per token, end exclusive."""
    T, n = logp.shape[0], len(seq)
    S = 2 * n + 1
    tok = np.array([blank if j % 2 == 0 else seq[j // 2] for j in range(S)])
    skip = np.zeros(S, bool)  # state j may come from j - 2
    for j in range(3, S, 2):
        skip[j] = seq[j // 2] != seq[j // 2 - 1]
    NEG = -1e18
    D = np.full((T, S), NEG)
    B = np.zeros((T, S), np.int32)
    D[0, 0], D[0, 1] = logp[0, tok[0]], logp[0, tok[1]]
    emit = logp[:, tok]
    for t in range(1, T):
        prev = D[t - 1]
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.where(skip, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)
        stack = np.stack([prev, c1, c2])
        arg = stack.argmax(0)
        D[t] = stack[arg, np.arange(S)] + emit[t]
        B[t] = np.arange(S) - arg
    j = S - 1 if D[T - 1, S - 1] >= D[T - 1, S - 2] else S - 2
    spans = {}
    for t in range(T - 1, -1, -1):
        if j % 2:
            a, b = spans.get(j // 2, (t, t + 1))
            spans[j // 2] = (t, max(b, t + 1))
        j = B[t, j]
    return [spans.get(i, (0, 0)) for i in range(n)]


def align_lyric(logp, words, P):
    """the lyric words on the frames -> [{w, start, end, phones: [{p, start, end}]}], every boundary CORR_S earlier (floored at 0).
    Words with no phones, or no frames to put them on, give []."""
    words = [w for w in (norm(w) for w in words) if w]
    if not words or logp is None or len(logp) < 2:
        return []
    phones = [phones_of(w) for w in words]
    seq = [P.tok(p) for ps in phones for p in ps]
    spans = viterbi(np.asarray(logp, float), seq, P.blank)
    out, k = [], 0
    for w, ps in zip(words, phones):
        phs = []
        for p in ps:
            a, b = spans[k]
            k += 1
            phs.append({"p": p, "start": round(max(0.0, a * FRAME - CORR_S), 3), "end": round(max(0.0, b * FRAME - CORR_S), 3)})
        out.append({"w": w, "start": phs[0]["start"], "end": phs[-1]["end"], "phones": phs})
    return out


def vowel_starts(words):
    return _dedupe([p["start"] for w in words for p in w["phones"] if p["p"] in ARPA_VOWELS], 0.0)


def closures(words):
    """the lyric's lip closures (s): every aligned P/B/M onset."""
    return _dedupe([p["start"] for w in words for p in w["phones"] if p["p"] in CLOSERS], 0.0)


def wide(words):
    """the aligned wide-open vowel starts (s)."""
    return _dedupe([p["start"] for w in words for p in w["phones"] if p["p"] in ARPA_WIDE], 0.0)


def confirmed(dets, words):
    """the onset set confirmed by the lyric: a detection kept when an aligned vowel starts within CONFIRM_S, one detection per vowel
    (the nearest); a vowel start with no detection adds nothing (the aligner is the weaker instrument); MIN_GAP apart."""
    vs = vowel_starts(words)
    if not vs:
        return list(dets)
    out, used = [], set()
    for v in vs:
        near = [(abs(d - v), i) for i, d in enumerate(dets) if i not in used and abs(d - v) <= CONFIRM_S]
        if near:
            _, i = min(near)
            used.add(i)
            out.append(dets[i])
    return _dedupe(out, MIN_GAP)
