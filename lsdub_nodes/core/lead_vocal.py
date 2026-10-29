#!/usr/bin/env python3
"""lead_vocal.py — keep the LEAD singer, drop the backing lines (melody-guided harmonic mask).

The vocal stem from vocal_split still holds every voice: lead, harmonies, ad-libs, doubles. The mouth must move
only for the lead. The lead is the dominant, continuous melody line, so:

  1. f0 track    harmonic-sum salience over 70-1100 Hz on a 10-cent grid (centre channel weighted: leads sit in
                 the middle, doubles and harmonies are usually spread), Viterbi path with a pitch-jump cost so the
                 track follows ONE voice instead of hopping to whichever harmony is loudest in a frame
  2. voicing     lead is singing where the path's salience is a large share of the frame's vocal energy and the
                 vocal is above a floor; short gaps are bridged, blips dropped
  3. mask        soft comb on the lead harmonics (width grows with the partial), plus the consonant band
                 (>2.5 kHz noise) in the 90 ms before / 40 ms after a lead voiced run, so word onsets survive
  4. outputs     lead = mask x vocals, backing = vocals - lead, and the lead activity curve (0..1 at 100 Hz)
                 that the mouth drive multiplies in (sung_dyn --lead-activity)

Pure DSP (numpy + torch STFT), CPU-fast: ~1-2 s per minute of audio.

  python -m lsdub.core.lead_vocal --in vocals.wav --out-lead lead.wav [--out-backing backing.wav] [--out-activity act.csv]
"""
import argparse, csv
import numpy as np
import torch

SR = 44100; N_FFT = 4096; HOP = 441                                       # 100 frames / s
FMIN, FMAX, CENTS = 70.0, 1100.0, 10.0
NH = 12                                                                 # harmonics in the salience sum


def _stft(x):
    w = torch.hann_window(N_FFT)
    return torch.stft(torch.as_tensor(x, dtype=torch.float32), N_FFT, HOP, window=w, return_complex=True)


def _istft(X, n):
    return torch.istft(X, N_FFT, HOP, window=torch.hann_window(N_FFT), length=n)


def salience(mag):
    """mag (F, T) -> (cands Hz (C,), S (C, T)) harmonic-sum salience with decaying weights."""
    cands = FMIN * 2 ** (np.arange(0, 1200 * np.log2(FMAX / FMIN), CENTS) / 1200)
    binhz = SR / N_FFT; m = np.sqrt(mag)                                  # compressed magnitude
    S = np.zeros((len(cands), mag.shape[1]), np.float32)
    for h in range(1, NH + 1):
        b = cands * h / binhz
        ok = b < mag.shape[0] - 2
        b0 = np.floor(b[ok]).astype(int); fr = (b[ok] - b0)[:, None]
        # peak-pick +/- 1 bin around the expected partial (tolerates slight inharmonicity / vibrato)
        v = np.maximum.reduce([m[b0 - 1], m[b0] * (1 - fr) + m[b0 + 1] * fr, m[b0 + 1], m[np.minimum(b0 + 2, m.shape[0] - 1)] * 0.7])
        S[ok] += (0.84 ** (h - 1)) * v
    return cands, S


def viterbi(S, jump_cost=0.004, max_jump_cents=500):
    """one continuous pitch path through the salience; cost per cent of jump, hard cap on huge leaps."""
    C, T = S.shape
    L = np.log(S / (S.max(0, keepdims=True) + 1e-9) + 1e-3)               # per-frame normalised log salience
    d = np.abs(np.arange(C)[:, None] - np.arange(C)[None, :]) * CENTS
    trans = -jump_cost * d; trans[d > max_jump_cents] = -1e3
    score = L[:, 0].copy(); back = np.zeros((C, T), np.int32)
    for t in range(1, T):
        tot = score[None, :] + trans                                      # (to, from)
        back[:, t] = tot.argmax(1); score = tot.max(1) + L[:, t]
    path = np.zeros(T, int); path[-1] = score.argmax()
    for t in range(T - 1, 0, -1): path[t - 1] = back[path[t], t]
    return path


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def extract(vocals, share=0.35, floor_db=-40.0, bridge_s=0.08, min_run_s=0.06):
    """vocals (2, n) float -> dict(lead (2,n), backing (2,n), f0 (T,), active (T,) 0..1, t (T,))."""
    vocals = np.asarray(vocals, np.float32); n = vocals.shape[1]
    mid = vocals.mean(0); side = 0.5 * (vocals[0] - vocals[1])
    M = _stft(mid).numpy(); Sd = _stft(side).numpy()
    mag = np.abs(M) - 0.5 * np.abs(Sd)                                     # favour the centred voice
    mag = np.clip(mag, 0, None)
    cands, S = salience(mag)
    path = viterbi(S); f0 = cands[path]; T = len(f0)
    sal = S[path, np.arange(T)]
    # voicing: the path's salience vs the frame's total compressed energy, and an absolute vocal floor
    energy = np.sqrt(mag).sum(0) + 1e-9
    ratio = sal / energy; ratio /= np.percentile(ratio[energy > np.percentile(energy, 50)], 90) + 1e-9
    lvl = 20 * np.log10(np.sqrt((np.abs(M) ** 2).sum(0)) + 1e-9); lvl -= np.percentile(lvl, 98)
    voiced = (ratio > share) & (lvl > floor_db)
    br = int(bridge_s * 100); mr = int(min_run_s * 100)
    for a, b in _runs(~voiced):                                           # bridge short gaps
        if 0 < a and b < T and b - a <= br: voiced[a:b] = True
    for a, b in _runs(voiced):                                            # drop blips
        if b - a < mr: voiced[a:b] = False
    # harmonic comb
    F = M.shape[0]; fb = np.arange(F) * SR / N_FFT
    mask = np.zeros((F, T), np.float32)
    vt = np.flatnonzero(voiced); f = f0[vt][None, :]
    k = np.maximum(1, np.round(fb[:, None] / f))                         # nearest harmonic per bin
    wid = np.maximum(1.6 * SR / N_FFT, 0.03 * k * f)
    mask[:, vt] = np.exp(-2.0 * ((fb[:, None] - k * f) / wid) ** 2)
    # consonants: high band around voiced-run onsets / offsets
    hi = fb > 2500
    for a, b in _runs(voiced):
        lo_t = max(0, a - 9); hi_t = min(T, b + 4)
        mask[np.ix_(hi, np.r_[lo_t:a, b:hi_t])] = np.maximum(mask[np.ix_(hi, np.r_[lo_t:a, b:hi_t])], 0.8)
    mask = np.clip(mask, 0, 1)
    lead = np.stack([_istft(torch.as_tensor(_stft(vocals[c]).numpy() * mask), n).numpy() for c in range(2)])
    act = np.zeros(T, np.float32)
    lead_lvl = 20 * np.log10(np.sqrt(((np.abs(M) * mask) ** 2).sum(0)) + 1e-9); lead_lvl -= np.percentile(lead_lvl[voiced], 95) if voiced.any() else 0
    act[voiced] = np.clip((lead_lvl[voiced] - floor_db) / -floor_db, 0, 1)
    return dict(lead=lead, backing=vocals - lead, f0=np.where(voiced, f0, 0.0), active=act,
                t=np.arange(T) * HOP / SR)


def _read(path):
    import subprocess
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def _write(path, x):
    import subprocess
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-', '-c:a', 'pcm_s16le', path],
                   input=np.ascontiguousarray(x.T, np.float32).tobytes(), check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out-lead', required=True)
    ap.add_argument('--out-backing'); ap.add_argument('--out-activity')
    ap.add_argument('--share', type=float, default=0.35); ap.add_argument('--floor-db', type=float, default=-40.0)
    a = ap.parse_args(argv)
    d = extract(_read(a.inp), a.share, a.floor_db)
    _write(a.out_lead, d['lead'])
    if a.out_backing: _write(a.out_backing, d['backing'])
    if a.out_activity:
        with open(a.out_activity, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['time_s', 'lead_active', 'f0_hz'])
            for i in range(len(d['t'])): w.writerow([f"{d['t'][i]:.3f}", f"{d['active'][i]:.4f}", f"{d['f0'][i]:.1f}"])
    v = (d['active'] > 0).mean()
    print(f"lead: voiced {v:.0%} of frames -> {a.out_lead}")


if __name__ == '__main__':
    main()
