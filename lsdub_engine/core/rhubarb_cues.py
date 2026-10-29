#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""rhubarb_cues.py — Rhubarb Lip Sync (MIT; bundled PocketSphinx + CMU en-us model, BSD-style) as a cross-check /
fallback viseme source.

  run:    lsdub rhubarb run --audio vocal_dry.wav --t0 8.04 --t1 16.04 \
              --dialog "the sung words ..." --out rhubarb_018.json
  compare: lsdub rhubarb compare --rhubarb rhubarb_018.json --track vis_018.json

Output JSON (song time): {t0, t1, cues: [{start, end, shape, vis}]}. Rhubarb's 9 shapes are mapped to our 15-viseme
set (lsdub visemes) as a best representative; the reverse map (VIS2RH) is what the agreement uses, since
Rhubarb's set is the coarser one. viseme_track.py --rhubarb uses this file as a closure cross-check or a full fallback.
"""
import argparse, json, os, subprocess, tempfile
from ..io import ff
import numpy as np

import shutil
RHUBARB = os.environ.get('LIPS_RHUBARB') or shutil.which('rhubarb') or 'rhubarb'
# Rhubarb shapes (README): A closed M/B/P; B slightly open, clenched teeth (most consonants, EE); C open (EH, AE);
# D wide open (AA); E slightly rounded (AO, ER); F puckered (UW, OW, W); G F/V teeth on lip; H L (tongue up); X rest.
RH2VIS = {'A': 'PP', 'B': 'DD', 'C': 'E', 'D': 'aa', 'E': 'oh', 'F': 'ou', 'G': 'FF', 'H': 'nn', 'X': 'sil'}
VIS2RH = {'sil': 'X', 'PP': 'A', 'FF': 'G', 'TH': 'B', 'DD': 'B', 'kk': 'B', 'CH': 'B', 'SS': 'B', 'nn': 'H', 'RR': 'E',
          'aa': 'D', 'E': 'C', 'ih': 'B', 'oh': 'E', 'ou': 'F'}
# coarse classes for a fairer agreement (Rhubarb and our track disagree a lot on B/C/H which look alike on screen)
CLS = {'A': 'closed', 'X': 'rest', 'B': 'near', 'G': 'near', 'H': 'near', 'C': 'open', 'D': 'open', 'E': 'round', 'F': 'round'}


def run(a):
    with tempfile.TemporaryDirectory() as d:
        ff.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{a.t0:.3f}', '-t', f'{a.t1 - a.t0:.3f}', '-i', a.audio, '-ac', '1',
                        '-ar', '16000', f'{d}/a.wav'], check=True)
        txt = open(a.dialog).read() if os.path.exists(a.dialog) else a.dialog
        open(f'{d}/dialog.txt', 'w').write(' '.join(txt.split()))       # one line: Rhubarb glues words across newlines
        subprocess.run([a.rhubarb, '-q', '-f', 'json', '--extendedShapes', 'GHX', '-d', f'{d}/dialog.txt', '-o', f'{d}/r.json',
                        f'{d}/a.wav'], check=True)
        r = json.load(open(f'{d}/r.json'))
    cues = [{'start': round(c['start'] + a.t0, 3), 'end': round(c['end'] + a.t0, 3), 'shape': c['value'], 'vis': RH2VIS[c['value']]}
            for c in r['mouthCues']]
    json.dump({'audio': a.audio, 't0': a.t0, 't1': a.t1, 'rhubarb': '1.14.0', 'cues': cues}, open(a.out, 'w'), indent=1)
    print(' '.join(f"{c['shape']}@{c['start']:.2f}" for c in cues))


def shape_at(cues, t):
    for c in cues:
        if c['start'] <= t < c['end']: return c['shape']
    return 'X'


def compare(rh, track):
    fr = track['frames']
    t = np.array([f['t'] + 0.5 / track['fps'] for f in fr])
    ours = [VIS2RH[f['vis']] for f in fr]
    theirs = [shape_at(rh['cues'], x) for x in t]
    exact = float(np.mean([o == r for o, r in zip(ours, theirs)]))
    coarse = float(np.mean([CLS[o] == CLS[r] for o, r in zip(ours, theirs)]))
    pp = [k for k, f in enumerate(fr) if f['vis'] == 'PP']
    pp_hit = sum(1 for k in pp if 'A' in theirs[max(0, k - 2):k + 3])
    rhA = [k for k, r in enumerate(theirs) if r == 'A']
    rhA_hit = sum(1 for k in rhA if any(f['vis'] == 'PP' for f in fr[max(0, k - 2):k + 3]))
    return {'frames': len(fr), 'exact_shape_agree': round(exact, 3), 'coarse_class_agree': round(coarse, 3),
            'our_PP_frames_with_rhubarb_A_pm2': f'{pp_hit}/{len(pp)}', 'rhubarb_A_frames_with_our_PP_pm2': f'{rhA_hit}/{len(rhA)}',
            'confusion_coarse': {f'{a}->{b}': int(sum(1 for o, r in zip(ours, theirs) if CLS[o] == a and CLS[r] == b))
                                 for a in ('closed', 'near', 'open', 'round', 'rest') for b in ('closed', 'near', 'open', 'round', 'rest')
                                 if any(CLS[o] == a and CLS[r] == b for o, r in zip(ours, theirs))}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--audio', required=True)
    r.add_argument('--t0', type=float, required=True)
    r.add_argument('--t1', type=float, required=True)
    r.add_argument('--dialog', required=True, help='text or a file')
    r.add_argument('--out', required=True)
    r.add_argument('--rhubarb', default=RHUBARB)
    c = sp.add_parser('compare')
    c.add_argument('--rhubarb', required=True)
    c.add_argument('--track', required=True, nargs='+')
    a = ap.parse_args(argv)
    if a.cmd == 'run': run(a)
    else:
        rh = json.load(open(a.rhubarb))
        for tr in a.track: print(tr, json.dumps(compare(rh, json.load(open(tr)))))


if __name__ == '__main__':
    main()
