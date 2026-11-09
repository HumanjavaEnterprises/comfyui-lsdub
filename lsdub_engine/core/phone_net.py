# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""phone_net.py: the dub driver's phone network, a CTC phone recogniser read off the lead vocal. CPU; one public model.

The drive needs three things the lyric aligner's word times cannot give it: where each sung syllable starts (its nucleus, the
vowel's first frame), the recogniser's frame-level evidence to align the lyric's phones on (ctc_align), and a lyric-free onset
when the model is not at hand (flux). Everything here is the driver's own reading of the vocal; the audit side has its own and the
two never import each other.

  PhoneNet(cache_dir)      the recogniser, loaded by local path; missing files are fetched (pinned, sha-checked) first
  posteriors(x, net)       CTC log-posteriors, one row per FRAME, for samples x at SR
  runs(logp, net)          argmax runs of one label: [(label, start s, end s)]
  level_db(x) / sung(x)    RMS level per HOP under the clip's p95, and the frames where the voice is on
  nuclei(logp, x, net)     syllable nuclei (s): a vowel run's start, one per nucleus (after a consonant or silence, or after a
                           vowel when the level dipped DIP_DB between the two), on sung frames, MIN_GAP apart, CORR_S earlier
                           (a CTC spike lands after the acoustic onset)
  flux_onsets(x)           lyric-free onsets (s): half-wave log-spectral flux in BAND over an adaptive median threshold
  vocal_events(x, logp, net)   nuclei and flux onsets merged MIN_GAP apart: the drive's syllable events
  nucleus_labels(logp, x, net) the nuclei with the recogniser's vowel label on each
  arpa_token(net, phone)   the recogniser's vocabulary id of an ARPAbet phone (its IPA spelling, an alternative, its first letter)

Pure parts (runs, nuclei, flux_onsets, merge, level_db) take arrays and return lists; torch and transformers load inside PhoneNet only.
"""

import json
import os

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import stft

from .._models import fetch_pinned, hf_snapshot

MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
NEED = ("config.json", "vocab.json", "preprocessor_config.json", "pytorch_model.bin")
SR = 16000
FRAME = 0.02  # s per recogniser row
HOP = 0.01  # s per level row
WIN = 0.04  # s, the level window
FLOOR_DB = -35.0  # under the clip's p95: the voice floor
SUNG_DB = -10.0  # a frame this far under the floor still counts as sung
NUC_DB = -6.0  # a nucleus sits over the floor less this
DIP_DB = 3.0  # a vowel after a vowel is a new nucleus when the level dipped this much between them
MIN_GAP = 0.08  # s between two events
CORR_S = 0.02  # s: a CTC spike's lateness on the acoustic onset
BAND = (200.0, 4000.0)  # Hz, the flux band
NFFT, FHOP = 768, 160  # flux STFT: 48 ms window, 10 ms hop at SR
FLUX_WIN, FLUX_K, FLUX_FLOOR, FLUX_PEAK = 0.3, 1.5, 0.08, 0.1

VOWEL_LETTERS = set("aeiouyɐɑɒæɔəɘɚɛɜɝɞɤɨɪɯɵøœɶʉʊʌʏ")
ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
IPA = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ",
    "ER": "ɚ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}  # fmt: skip
IPA_ALT = {"ɑː": "ɑ", "ʌ": "ə", "ɔː": "ɔ", "ɚ": "ɜ", "iː": "i", "uː": "u", "ɡ": "g", "ɹ": "r", "oʊ": "o", "eɪ": "e"}


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
    """the local snapshot path when the files are there, else the repo id (PhoneNet then fetches the pinned files)."""
    return hf_snapshot(models_dir(cache_dir), MODEL, NEED) or MODEL


def _read_vocab(source):
    """the vocabulary from the snapshot's vocab.json: never through a tokenizer class (the phoneme tokenizer needs phonemizer and
    espeak, which LSDub does not install)."""
    with open(os.path.join(source, "vocab.json"), encoding="utf-8") as f:
        return json.load(f)


class PhoneNet:
    """the recogniser and its vocabulary; torch and transformers load here, not at import."""

    def __init__(self, cache_dir=None, device="cpu"):
        from transformers import AutoModelForCTC

        self.source = model_source(cache_dir)
        if self.source == MODEL:  # not there yet: the four pinned files, sha-checked, into the models folder; then by path
            self.source = fetch_pinned(MODEL, models_dir(cache_dir))
        self.model = AutoModelForCTC.from_pretrained(self.source, use_safetensors=False).eval().to(device)
        self.device = device
        vocab = self.model.config.to_dict().get("vocab") or _read_vocab(self.source)
        self.labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        self.blank = int(vocab.get("<pad>", 0))
        self.special = {i for i, t in enumerate(self.labels) if t.startswith("<")}
        self.index = {t: i for i, t in enumerate(self.labels)}

    def release(self):
        self.model = None


def arpa_token(net, phone):
    """the vocabulary id of an ARPAbet phone: its IPA spelling, an alternative spelling, its first letter, else <unk>."""
    ipa = IPA.get(str(phone).rstrip("012").upper(), "")
    for c in (ipa, IPA_ALT.get(ipa, ""), ipa[:1]):
        if c in net.index:
            return net.index[c]
    return net.index.get("<unk>", 3)


def posteriors(x, net):
    """(T, V) log-softmax rows at FRAME for samples x at SR, level-normalised as the recogniser was trained."""
    import torch

    x = np.asarray(x, np.float32)
    x = (x - x.mean()) / (x.std() + 1e-7)
    with torch.no_grad():
        lg = net.model(torch.from_numpy(x)[None].to(net.device)).logits[0]
        return torch.log_softmax(lg.float(), -1).cpu().numpy().astype(np.float64)


def runs(logp, net):
    """argmax runs of one label: [(label, start s, end s)]; special rows end nothing, a repeat after a blank is a new run."""
    ids = np.asarray(logp).argmax(-1)
    out = []
    for i, k in enumerate(ids):
        if k in net.special:
            continue
        lab, t = net.labels[k], round(i * FRAME, 3)
        if out and out[-1][0] == lab and t - out[-1][2] <= FRAME * 1.5:
            out[-1][2] = round(t + FRAME, 3)
        else:
            out.append([lab, t, round(t + FRAME, 3)])
    return [tuple(r) for r in out]


def level_db(x):
    """RMS level per HOP (WIN window) in dB under the clip's p95."""
    x = np.asarray(x, float)
    N, H = int(WIN * SR), int(HOP * SR)
    n = max(1, (len(x) - N) // H + 1)
    x = np.pad(x, (0, max(0, (n - 1) * H + N - len(x))))
    fr = np.lib.stride_tricks.as_strided(x, (n, N), (x.strides[0] * H, x.strides[0]))
    db = 10 * np.log10(np.mean(fr * fr, 1) + 1e-12)
    return db - np.percentile(db, 95)


def sung(x):
    """True per HOP where the voice is on."""
    return level_db(x) > FLOOR_DB + SUNG_DB


def merge(ts, gap=MIN_GAP):
    """sorted times, any two closer than gap collapsed onto the earlier."""
    out = []
    for t in sorted(float(t) for t in ts):
        if not out or t - out[-1] >= gap:
            out.append(round(t, 3))
    return out


def nuclei(logp, x, net, corr=CORR_S):
    """syllable nuclei (s): vowel-run starts, one per nucleus, on sung frames, corr earlier."""
    db = level_db(x)
    out, prev = [], None
    for lab, a, b in runs(logp, net):
        if lab[0] in VOWEL_LETTERS:
            new = prev is None or prev[0][0] not in VOWEL_LETTERS
            if not new:  # vowel after vowel: a new nucleus only over a dip in the level between the two
                seg = db[int(prev[1] / HOP) : int(a / HOP) + 1]
                new = len(seg) > 2 and (max(seg[0], seg[-1]) - seg.min()) >= DIP_DB and seg.argmin() not in (0, len(seg) - 1)
            loud = db[min(int(a / HOP) + 2, len(db) - 1)] > FLOOR_DB + NUC_DB
            if new and loud and (not out or a - out[-1] >= MIN_GAP):
                out.append(round(float(a), 3))
        prev = (lab, a, b)
    return [round(max(0.0, t - corr), 3) for t in out]


def nucleus_labels(logp, x, net, corr=CORR_S):
    """[(nucleus time, its vowel label)] for the nuclei of (logp, x), times as nuclei() gives them."""
    nuc = set(nuclei(logp, x, net, corr=0.0))
    return [(round(max(0.0, a - corr), 3), lab) for lab, a, b in runs(logp, net) if round(float(a), 3) in nuc]


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
    return merge(t[cand])


def vocal_events(x, logp=None, net=None):
    """the drive's syllable events (s): nuclei and flux onsets merged MIN_GAP apart; flux alone without the model."""
    fl = flux_onsets(x)
    if logp is None or net is None:
        return fl
    return merge(list(nuclei(logp, x, net)) + list(fl))
