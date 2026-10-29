#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sung_dyn.py — v14: a SUNG mouth-dynamics curve from the vocal itself, for the warp's coarticulation planner.

The coart planner (lsdub/core/coart.py) sets each phone segment's opening from the segment's MEAN audio openness, so
inside a held sung vowel the mouth is flat and its motion comes from the take, not the singing. In singing the vowel
decides the mouth's SHAPE and the voice's loudness decides how far it OPENS (lip/jaw opening tracks loudness more than
pitch). This writes that as an external dynamics source for `lsdub paint --coart-dyn <csv> --coart-dyn-mix <w>`:

  shape   the lyrics' visemes (viseme_track.build at --rate, min-hold 1): nominal opening OPEN[v] and a width WID[v]
          (spread E / ih wide, rounded oh / ou / RR / CH narrow), blended with the runner-up by the track's mix
  drive   the dry vocal's loudness envelope (RMS -> dB, top at p98, --floor-db), attack --attack-ms / release
          --release-ms, blended with Audio2Face's jawOpen (--a2f-csv, --a2f-mix) when given
  gap     shape opening x (base + (1 - base) x drive); 0 on P / B / M (the planner keeps closures hard anyway)
  width   shape width + a little loudness spread; rounding from A2F funnel / pucker when given
  both    zero-phase low-passed at --lowpass-hz (the mouth does not move faster than ~8-10 Hz)

Output CSV (song timeline, one row per 1/rate interval start): time_s, geo_gap_cm, geo_width_cm (arbitrary units: the
planner percentile-maps them onto the take's own range), plus drive / shape_open for inspection.

  python -m lsdub_engine.core.sung_dyn --align align.json --audio vocal.wav [--a2f-csv a2f.csv] --t0 6.17 --dur 5 --out dyn.csv
"""
import argparse, csv
from ..io import ff
import numpy as np
from . import viseme_track as vt

WID = {'sil': 0.0, 'PP': 0.0, 'FF': 0.1, 'TH': 0.1, 'DD': 0.15, 'kk': 0.1, 'CH': -0.3, 'SS': 0.25, 'nn': 0.1,
       'RR': -0.35, 'aa': 0.15, 'E': 0.45, 'ih': 0.5, 'oh': -0.45, 'ou': -0.7}


def _zero_phase_lp(x, rate, fc):
    """forward + backward one-pole low-pass (zero phase) at cut-off fc Hz."""
    if fc <= 0: return x
    al = 1.0 - np.exp(-2 * np.pi * fc / rate)
    y = np.array(x, float)
    for i in range(1, len(y)): y[i] = y[i - 1] + al * (y[i] - y[i - 1])               # forward
    for i in range(len(y) - 2, -1, -1): y[i] = y[i + 1] + al * (y[i] - y[i + 1])      # backward (cancels the lag)
    return y


def envelope(audio, t_start, n, rate, floor_db=-35.0, attack_ms=30.0, release_ms=120.0):
    """the vocal's loudness 0..1 at `rate` Hz from t_start (song s), attack / release smoothed."""
    sr = 16000
    raw = ff.run(['ffmpeg', '-v', 'error', '-ss', f'{max(0.0, t_start):.3f}', '-t', f'{n / rate + 0.1:.3f}', '-i', audio,
                          '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    hop = sr / rate
    w = int(0.02 * sr)
    rms = np.array([np.sqrt(np.mean(x[max(0, int(i * hop + hop / 2) - w // 2): int(i * hop + hop / 2) + w // 2] ** 2) + 1e-12)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-9)
    db -= np.percentile(db, 98)
    e = np.clip((db - floor_db) / -floor_db, 0, 1)
    ka = 1 - np.exp(-1000.0 / (attack_ms * rate))
    kr = 1 - np.exp(-1000.0 / (release_ms * rate))
    y = np.zeros(n)
    for i in range(n):
        p = y[i - 1] if i else e[0]
        y[i] = p + (ka if e[i] > p else kr) * (e[i] - p)
    return y


def a2f_curves(path, tt):
    with open(path) as f:
        r = csv.reader(f)
        h = next(r)
        R = np.array([[float(v) for v in row] for row in r if row])
    t = R[:, h.index('time_s')]
    get = lambda k: np.interp(tt, t, R[:, h.index(k)]) if k in h else np.zeros(len(tt))
    jaw = get('jawOpen')
    jaw = np.clip(jaw / max(1e-6, np.percentile(R[:, h.index('jawOpen')], 95)), 0, 1.2)
    rnd = np.clip(np.maximum(get('mouthFunnel'), get('mouthPucker')), 0, 1)
    return jaw, rnd


def pump(drive, rate, gate=0.35, ratio=4.0, hold_ms=60.0, open_floor=0.55):
    """hard gate + compressor on the 0..1 drive: below `gate` the mouth is SHUT (0), above it the phrase is squashed
    toward full (open_floor..1, `ratio`:1), so phrases punch and the silences between them read as silence.
    `hold_ms` keeps the gate open briefly so a dip inside a word does not chatter. Returns (driven, gate_open)."""
    on = drive >= gate
    h = int(round(hold_ms * rate / 1000))
    g = on.copy()
    last = -10 ** 9
    for i in range(len(on)):
        if on[i]: last = i
        elif i - last <= h: g[i] = True
    x = np.clip((drive - gate) / max(1e-6, 1 - gate), 0, 1) ** (1.0 / ratio)
    return np.where(g, open_floor + (1 - open_floor) * x, 0.0), g


def phrase_mask(align, tt, lead=0.10, release=0.12, join_s=0.25, lyric_only=True):
    """True inside the aligned LEAD phrases: lyric words (the karaoke timing), merged across gaps < join_s, padded by
    the anticipation lead / release. Words the aligner heard but the lyrics do not have (src sung-only, STT
    hallucinations) are left out unless sandwiched inside a lyric phrase."""
    keep = lead_words(align) if lyric_only else sorted(align.get('words', []), key=lambda w: w['start'])
    spans = []
    for w in keep:
        if spans and w['start'] - spans[-1][1] < join_s: spans[-1][1] = max(spans[-1][1], w['end'])
        else: spans.append([w['start'], w['end']])
    m = np.zeros(len(tt), bool)
    for a, b in spans: m |= (tt >= a - lead) & (tt <= b + release)
    return m


def lead_words(align):
    """the LEAD's words: lyric-sheet words, plus a heard word sandwiched inside a lyric phrase ('em'->'them').
    Backup lines (src sung-only, e.g. the 'da da da' scat) and STT hallucinations drop out."""
    W = sorted(align.get('words', []), key=lambda w: w['start'])
    keep = []
    for i, w in enumerate(W):
        src = str(w.get('src', 'lyric'))
        if src.startswith('lyric'):
            keep.append(w)
            continue
        nb = [W[j] for j in (i - 1, i + 1) if 0 <= j < len(W) and str(W[j].get('src', '')).startswith('lyric')]
        if len(nb) == 2 and w['start'] - nb[0]['end'] < 0.5 and nb[1]['start'] - w['end'] < 0.5: keep.append(w)
    return keep


def build(align, audio, t0, dur, rate=100.0, a2f=None, a2f_mix=0.35, base=0.2, floor_db=-35.0, attack_ms=30.0,
          release_ms=120.0, lowpass_hz=9.0, lead=0.10, pad=0.5, pump_gate=0.0, pump_ratio=4.0, pump_hold_ms=60.0):
    ts = t0 - pad
    n = int(round((dur + 2 * pad) * rate))
    tt = ts + (np.arange(n) + 0.5) / rate
    fr = vt.build(align, ts, n, rate, lead=lead, release=0.11, min_hold=1)
    so = np.array([(1 - f['mix']) * vt.OPEN[f['vis']] + f['mix'] * vt.OPEN[f['vis2']] for f in fr])
    sw = np.array([(1 - f['mix']) * WID[f['vis']] + f['mix'] * WID[f['vis2']] for f in fr])
    closure = np.array([f['vis'] == 'PP' for f in fr])
    drive = envelope(audio, ts, n, rate, floor_db, attack_ms, release_ms)
    if a2f:
        jaw, rnd = a2f_curves(a2f, tt)
        drive = (1 - a2f_mix) * drive + a2f_mix * np.clip(jaw, 0, 1)
        sw = sw - 0.4 * rnd                                          # A2F's rounding narrows the mouth
    gate = np.ones(n, bool)
    if pump_gate > 0:
        drive, gate = pump(drive, rate, pump_gate, pump_ratio, pump_hold_ms)
        gate &= phrase_mask(align, tt, lead, 0.12)                    # the STT/karaoke word timing keys the gate
    gap = so * (base + (1 - base) * drive) * gate
    gap[closure] = 0.0
    width = sw + 0.1 * drive
    gap = np.clip(_zero_phase_lp(gap, rate, lowpass_hz), 0, None)
    width = _zero_phase_lp(width, rate, lowpass_hz)
    return dict(t=ts + np.arange(n) / rate, gap=gap, width=width, drive=drive, shape_open=so)


def main(argv=None):
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', required=True)
    ap.add_argument('--audio', required=True, help='the dry vocal stem')
    ap.add_argument('--a2f-csv')
    ap.add_argument('--a2f-mix', type=float, default=0.35)
    ap.add_argument('--t0', type=float, required=True)
    ap.add_argument('--dur', type=float, required=True)
    ap.add_argument('--rate', type=float, default=100.0)
    ap.add_argument('--base', type=float, default=0.2)
    ap.add_argument('--floor-db', type=float, default=-35.0)
    ap.add_argument('--attack-ms', type=float, default=30.0)
    ap.add_argument('--release-ms', type=float, default=120.0)
    ap.add_argument('--lowpass-hz', type=float, default=9.0)
    ap.add_argument('--lead', type=float, default=0.10)
    ap.add_argument('--pump-gate', type=float, default=0.0, help='hard gate + compressor on the drive: below this '
                    '(0..1 of the floor-db range) the mouth shuts, above it phrases squash toward full; 0 = off')
    ap.add_argument('--pump-ratio', type=float, default=4.0)
    ap.add_argument('--pump-hold-ms', type=float, default=60.0)
    ap.add_argument('--lead-align-out', help='also write the alignment with only the lead words (for the viseme track)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    d = build(json.load(open(a.align)), a.audio, a.t0, a.dur, a.rate, a.a2f_csv, a.a2f_mix, a.base, a.floor_db,
              a.attack_ms, a.release_ms, a.lowpass_hz, a.lead,
              pump_gate=a.pump_gate, pump_ratio=a.pump_ratio, pump_hold_ms=a.pump_hold_ms)
    if a.lead_align_out:
        al = json.load(open(a.align))
        al['words'] = lead_words(al)
        json.dump(al, open(a.lead_align_out, 'w'), indent=0)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'geo_gap_cm', 'geo_width_cm', 'drive', 'shape_open'])
        for i in range(len(d['t'])):
            w.writerow([f"{d['t'][i]:.4f}", f"{d['gap'][i]:.5f}", f"{d['width'][i]:.5f}", f"{d['drive'][i]:.4f}", f"{d['shape_open'][i]:.4f}"])
    print(f"sung dyn: {len(d['t'])} rows @ {a.rate:g} Hz, song {d['t'][0]:.2f}-{d['t'][-1]:.2f} s -> {a.out}")


if __name__ == '__main__':
    main()
