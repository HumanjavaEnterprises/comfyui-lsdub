#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""animator_pass.py — rebuild a viseme track the way an animator would (docs/ANIMATOR_RULES.md).

Input: the word/phone alignment (tools/lyric_align.py JSON) + the dry vocal. Output: a track in the same per-frame
format `lsdub paint` and `lsdub hybrid` read ({t, vis, vis2, mix, open, word, phone}), plus the
animator channels: 'jaw' (envelope, slow), 'alpha' (articulation strength), 'cls' (closer|round|vowel|other|rest),
'apex' (True on a target's held apex).

Rules applied (numbers from the doc):
  1  two channels: JAW from the dry-vocal RMS (70 ms smoothing, <=0.12/frame velocity); LIP from phonemes, no jaw in it
  2  picture leads sound: 120 ms attack (150 ms for rounded/protruded), apex at phone onset, hold to 75 %, 120/150 ms
     decay; closers after a vowel apex 1 frame early; --apex-lead shifts every apex (default 1 frame early)
  3  tongue-only consonants (DD kk nn, L, TH) take the nearest vowel's lip shape in the word; non-closer consonants
     < 60 ms dropped; same-group neighbours merged; <= 1 new target per 2 frames; function words alpha x0.5
  4  PP/FF alpha 1, hard, apex held >= 2 frames (3 if open either side), JAW <= 0.25 over the apex; rounding 0.8-1.0
  5  smoothstep ease in/out, lip weight change <= 0.25 per frame; pop count reported
  6  vowels > 200 ms hold their lip shape; JAW stays live (envelope)
  7  consonant strength from 8-20 kHz energy z-score: weak x0.6, strong x1.25
  8  anticipation: a jaw rise > 0.5 within 4 frames gets a -0.1 dip 3-4 frames before
 10  global gain G (close-up 0.6) on everything but closers; open cap (close-up 0.75)
Rule 9 (chin/cheek coupling) is applied by the renderers (mouth_paint clean plate, hybrid_nudge chin follow).

  lsdub animate --align align.json --audio lead.wav --t0 8.04 --frames 200 \
      --out vis_anim.json [--shot close|medium|wide] [--json-report r.json]
"""
import argparse, json
from ..io import ff
import numpy as np

from .viseme_track import VIS, envelope_open, phone_list

VOWELS = {'AA', 'AE', 'AH', 'AY', 'AW', 'EH', 'EY', 'IH', 'IY', 'AO', 'OW', 'OY', 'UW', 'UH', 'ER'}
LIPHEAVY = {'UW', 'OW', 'OY', 'W', 'S', 'Z', 'SH', 'ZH', 'CH', 'JH', 'UH'}
TONGUE = {'L', 'N', 'T', 'D', 'G', 'K', 'NG', 'TH', 'DH'}
FUNCTION = {'a', 'an', 'the', 'and', 'of', 'to', 'in', 'on', 'at', 'is', 'it', 'for', 'or', 'but', 'so', 'as', 'by',
            'with', 'from', 'that', 'this', 'be', 'are', 'was', 'i', 'you', 'we', 'they', 'he', 'she', 'my', 'your',
            "don't", 'do', 'not', 'no', 'just', 'than', 'then', 'them'}
GROUP = {'PP': 'PP', 'FF': 'FF', 'SS': 'SC', 'CH': 'SC', 'aa': 'AE', 'E': 'AE', 'ih': 'IE', 'oh': 'OU', 'ou': 'OU'}
GAIN = {'close': 0.6, 'medium': 0.8, 'wide': 1.0}
OPENMAX = {'close': 0.75, 'medium': 0.85, 'wide': 1.0}


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def hf_energy(audio, t0, dur, sr=48000):
    raw = ff.run(['ffmpeg', '-v', 'error', '-ss', f'{max(0, t0):.3f}', '-t', f'{dur:.3f}', '-i', audio, '-ac', '1',
                          '-ar', str(sr), '-af', 'highpass=f=8000,lowpass=f=20000', '-f', 'f32le', '-'], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def song_hf_stats(audio):
    raw = ff.run(['ffmpeg', '-v', 'error', '-i', audio, '-ac', '1', '-ar', '48000', '-af', 'highpass=f=8000,lowpass=f=20000',
                          '-f', 'f32le', '-'], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    w = 2400
    e = np.sqrt(np.mean(x[:len(x) // w * w].reshape(-1, w) ** 2, 1))
    e = e[e > np.percentile(e, 30)]
    return float(np.log(e + 1e-9).mean()), float(np.log(e + 1e-9).std() + 1e-6)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub animate')
    ap.add_argument('--align', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--t0', type=float, required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--fps', type=float, default=25)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shot', choices=list(GAIN), default='close')
    ap.add_argument('--gain', type=float)
    ap.add_argument('--open-max', type=float)
    ap.add_argument('--apex-lead', type=float, default=0.04,
                    help='s; every apex this much before phone onset (picture leads sound)')
    ap.add_argument('--json-report')
    a = ap.parse_args(argv)
    fps, n, t0 = a.fps, a.frames, a.t0
    G = a.gain if a.gain is not None else GAIN[a.shot]
    OMAX = a.open_max if a.open_max is not None else OPENMAX[a.shot]
    align = json.load(open(a.align))
    STRESS = {}
    for w in align['words']:                       # strip CMUdict stress digits (AE1 -> AE) before the viseme map,
        for q in w['phones']:                      # keeping the digit for rule 3's stress weighting
            d = ''.join(c for c in q['p'] if c.isdigit())
            if d:
                STRESS[round(q['start'], 3)] = d
                q['p'] = ''.join(c for c in q['p'] if not c.isdigit())
    ph = phone_list(align)
    for q in ph: q['stress'] = STRESS.get(round(q['start'], 3))
    tt = t0 + (np.arange(n) + 0.5) / fps
    ph = [p for p in ph if p['end'] > t0 - 0.5 and p['start'] < t0 + n / fps + 0.5]

    # ---- 1. classify + alpha
    mu, sd = song_hf_stats(a.audio)
    words = {}
    for p in ph: words.setdefault((p['word'], round(p.get('wstart', 0), 3)), []).append(p)
    for i, p in enumerate(ph):
        arpa = ''.join(c for c in p['p'] if not c.isdigit())
        dur = p['end'] - p['start']
        stress = p.get('stress')                   # CMUdict stress digit when the aligner kept it
        if p['vis'] in ('PP', 'FF'): p['cls'], p['alpha'] = 'closer', 1.0
        elif arpa in VOWELS: p['cls'], p['alpha'] = 'vowel', ((1.0 if stress == '1' else 0.6) if stress else (1.0 if dur >= 0.12 else 0.6))
        elif arpa in TONGUE: p['cls'], p['alpha'] = 'tongue', 0.6
        else: p['cls'], p['alpha'] = 'other', 0.6
        if p['vis'] in ('oh', 'ou') or arpa == 'W':
            p['cls'] = 'round' if p['cls'] != 'closer' else 'closer'
            p['alpha'] = max(p['alpha'], 0.9)
        if p['word'].lower().strip(".,!?'") in FUNCTION: p['alpha'] = max(0.8, p['alpha'] * 0.5) if p['cls'] == 'closer' else p['alpha'] * 0.5
        p['sustain'] = p['cls'] in ('vowel', 'round') and dur > 0.20
        if p['cls'] not in ('vowel', 'round'):      # rule 7: consonant strength from 8-20 kHz energy
            z = (np.log(hf_energy(a.audio, p['start'], max(dur, 0.04)) + 1e-9) - mu) / sd
            p['z'] = round(float(z), 2)
            if z <= -1: p['alpha'] *= 0.6
            elif z >= 1: p['alpha'] = min(1.0, p['alpha'] * 1.25)
            if p['cls'] == 'closer': p['alpha'] = max(p['alpha'], 1.0)   # closers stay fully closed (hold shortens)
    # ---- 3. coarticulation
    for i, p in enumerate(ph):
        if p['cls'] == 'tongue':
            same = [q for q in ph[i + 1:] if q['word'] == p['word'] and q['cls'] in ('vowel', 'round')]
            back = [q for q in ph[:i][::-1] if q['word'] == p['word'] and q['cls'] in ('vowel', 'round')]
            src = (same or back or [None])[0]
            p['lip'] = src['vis'] if src else None
        else: p['lip'] = p['vis']
    T = [p for p in ph if p['lip'] and not (p['cls'] in ('other', 'tongue') and p['end'] - p['start'] < 0.06)]
    merged = []
    for p in T:
        if merged and GROUP.get(merged[-1]['lip'], merged[-1]['lip']) == GROUP.get(p['lip'], p['lip']) and p['start'] - merged[-1]['end'] < 0.05:
            q = merged[-1]
            q['end'] = p['end']
            q['alpha'] = max(q['alpha'], p['alpha'])
            q['sustain'] = q['sustain'] or p['sustain']
        else: merged.append(dict(p))
    T = []
    for p in merged:                                 # budget: <= 1 new target per 2 frames
        if T and p['start'] - T[-1]['start'] < 2 / fps and p['cls'] != 'closer':
            if p['alpha'] > T[-1]['alpha'] and T[-1]['cls'] != 'closer': T[-1] = p
            continue
        T.append(p)

    # ---- 2/4/5. keyed lip curves
    L = np.zeros((n, len(VIS)))
    apex = np.zeros(n, bool)
    cls = np.array(['rest'] * n, dtype=object)
    for i, p in enumerate(T):
        heavy = p['p'].rstrip('012') in LIPHEAVY or p['cls'] == 'round'
        att = 0.150 if heavy else 0.120
        dec = att
        on = p['start'] - a.apex_lead
        if p['cls'] == 'closer' and i > 0 and T[i - 1]['cls'] in ('vowel', 'round'): on -= 1 / fps
        if p['cls'] == 'closer' and (i == 0 or p['start'] - T[i - 1]['end'] > 0.15): att = max(att, 0.200)
        dur = max(p['end'] - p['start'], 1 / fps)
        hold_end = on + max(0.75 * dur, (3 if p['cls'] == 'closer' else 0) / fps if p['cls'] == 'closer' else 0)
        if p['cls'] == 'closer': hold_end = max(hold_end, on + 2 / fps)
        if p['cls'] == 'closer' and p.get('z', 0) <= -1: hold_end = on + 2 / fps
        if p['cls'] not in ('closer',): hold_end = min(hold_end, on + 0.75 * dur) if p['cls'] != 'vowel' or not p['sustain'] else on + 0.75 * dur
        if p['cls'] in ('other', 'tongue'): hold_end = min(hold_end, on + 2 / fps)
        w = np.zeros(n)
        w = np.maximum(w, smoothstep((tt - (on - att)) / att) * (tt < on))
        w[(tt >= on) & (tt <= hold_end)] = 1
        m = tt > hold_end
        w[m] = np.maximum(w[m], 1 - smoothstep((tt[m] - hold_end) / dec))
        amp = p['alpha'] * (1.0 if p['cls'] == 'closer' else G)
        k = VIS.index(p['lip'])
        L[:, k] = np.maximum(L[:, k], w * amp)
        m = (tt >= on) & (tt <= hold_end)
        apex |= m
        cls[m & ((cls == 'rest') | (p['cls'] == 'closer'))] = p['cls'] if p['cls'] in ('closer', 'round', 'vowel') else 'other'
    # per-frame change cap 0.25 (closers exempt so they can snap shut on time)
    for f in range(1, n):
        d = L[f] - L[f - 1]
        lim = np.full(len(VIS), 0.25)
        lim[[VIS.index('PP'), VIS.index('FF')]] = 1.0
        L[f] = L[f - 1] + np.clip(d, -lim, lim)
    rest = np.clip(1 - L.max(1), 0, 1)
    L[:, 0] = np.maximum(L[:, 0], rest)

    # ---- 1/4/6/8. jaw
    env = envelope_open(a.audio, t0, n, fps)
    k = np.ones(3) / 3
    jaw = np.convolve(np.pad(env, 1, mode='edge'), k, 'valid')     # ~70-80 ms at 25 fps
    for f in range(1, n): jaw[f] = jaw[f - 1] + np.clip(jaw[f] - jaw[f - 1], -0.12, 0.12)
    jaw_pre = jaw.copy()
    for f in range(n - 4):
        if jaw_pre[f + 4] - jaw_pre[f] > 0.5:
            for g in (f, f + 1): jaw[g] = max(0, jaw[g] - 0.1)
    closer_apex = (cls == 'closer')
    jaw[closer_apex] = np.minimum(jaw[closer_apex], 0.25)
    jaw = np.clip(jaw * OMAX, 0, 1)

    frames, pops = [], 0
    for f in range(n):
        o = np.argsort(-L[f])
        d, s = o[0], o[1]
        tot = L[f, d] + L[f, s]
        mix = float(L[f, s] / tot) if tot > 0 else 0.0
        frames.append({'t': round(float(t0 + f / fps), 3), 'vis': VIS[d], 'vis2': VIS[s], 'mix': round(min(mix, 0.5), 3),
                       'open': round(float(jaw[f]), 3), 'jaw': round(float(jaw[f]), 3), 'alpha': round(float(L[f, d]), 3),
                       'cls': str(cls[f]), 'apex': bool(apex[f])})
    dl = np.abs(np.diff(L, axis=0)).sum(1)
    med = np.median(dl[dl > 1e-4]) if (dl > 1e-4).any() else 0
    pops = int((dl > 2.5 * med).sum()) if med > 0 else 0
    out = {'fps': fps, 't0': t0, 'align': a.align, 'animator': {'G': G, 'open_max': OMAX, 'apex_lead': a.apex_lead, 'shot': a.shot},
           'visemes': VIS, 'library_map': {'kk': 'DD', 'nn': 'DD'}, 'frames': frames,
           'targets': [{'p': p['p'], 'lip': p['lip'], 'cls': p['cls'], 'alpha': round(p['alpha'], 2), 'start': p['start'],
                        'end': p['end'], 'sustain': p['sustain'], 'word': p['word']} for p in T]}
    json.dump(out, open(a.out, 'w'), indent=1)
    rep = {'targets': len(T), 'phones_in_window': len(ph), 'closer_frames': int(closer_apex.sum()),
           'round_frames': int((cls == 'round').sum()), 'lip_pops_gt_2.5x_median': pops, 'G': G, 'open_max': OMAX}
    print(json.dumps(rep))
    if a.json_report: json.dump(rep, open(a.json_report, 'w'), indent=1)


if __name__ == '__main__':
    main()
