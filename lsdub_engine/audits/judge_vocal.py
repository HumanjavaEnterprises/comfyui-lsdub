# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""judge_vocal.py: the sync judge's own reading of the vocal stem. CPU, no models.

sync_truth builds its two reference curves (loud, open) from these per-frame readings. They are a frozen copy of the readings
vocal_retime.features made when the judge was written (LSDub 0.1.8), kept here so the judge never moves when the dub's driver changes
its own features. The driver (sing, vocal_retime, viseme_track, mouth_*, recipes) must not import this module, and this module imports
nothing from the driver: tests/test_judge_frozen.py holds both rules and pins the curves this produces on a fixture.

Change nothing here to improve a dub. A change here is a change of the instrument, and every reading taken before it stops being
comparable with every reading taken after it.
"""

import numpy as np
from scipy.signal import stft

from ..io import ff

SR = 16000
HOP = 0.01  # s
WIN = 0.04  # s, the pitch window
F0_LO, F0_HI = 80.0, 700.0
HF_HZ = 3500.0
LOCAL = 0.15  # s each way: a frame's level reads against the loudest near it
PHRASE = 0.5  # s each way: the same against the phrase
RATIO = 0.5  # s each way: a frame's low-to-mid ratio reads against the usual open sound near it
LO_HZ, MID_HZ, TOP_HZ = 500.0, 700.0, 2400.0
ZERO_DB = -45.0  # under the clip's p95: silence
FLOOR_DB, HAS_GAPS_DB = 6.0, -30.0  # a noisy clip's silence: this far over its quietest twentieth, when that sits this far under its p95


def read_mono(path):
    """the file as mono float64 samples at SR."""
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
    """dict of per-frame readings at HOP: voiced, quiet, nasal (0..1) and db under the clip's p95 (the four the judge reads). A clip
    shorter than a window reads as one window."""
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
    sm = lambda a, lo, hi: np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    silent = sm(zero_db(db) - db, -3.0, 6.0)
    voiced = sm(v, 0.35, 0.7) * (1 - silent)
    quiet = sm(nearby(db, max(1, int(round(LOCAL / HOP))), 100) - db, 4.0, 14.0)
    ratio = 10 * np.log10((P[f < LO_HZ].sum(0) + 1e-12) / (P[(f >= MID_HZ) & (f < TOP_HZ)].sum(0) + 1e-12))
    nasal = sm(ratio - nearby(ratio, max(1, int(round(RATIO / HOP))), 25), 8.0, 16.0) * voiced
    return dict(voiced=voiced, quiet=quiet, nasal=nasal, db=db)


def gate_box(R, shape, roi_pad=1.0):
    """the sing gate's box on a composite (its medium.json R): the paste's median centre, 1.6 face heights a side."""
    cx = np.median([f["centre"][0] for f in R["per_frame"]])
    cy = np.median([f["centre"][1] for f in R["per_frame"]])
    s = R["face_h_px"] * 1.6 * roi_pad
    H, W = shape[:2]
    return [int(max(0, cy - s / 2)), int(min(H, cy + s / 2)), int(max(0, cx - s / 2)), int(min(W, cx + s / 2))]
