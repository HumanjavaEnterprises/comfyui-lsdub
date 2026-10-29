#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""viseme_track.py — phone timings (lsdub align JSON) -> per-frame viseme track + ARKit-pose curves.

Viseme set: the 15 Oculus/Meta OVR visemes (sil PP FF TH DD kk CH SS nn RR aa E ih oh ou). The mouth library renders 13
images (kk and nn share the DD "etc" mouth). ARPAbet -> viseme table below; HH is transparent (takes the next vowel).

Co-articulation per phone: a raised-cosine activation that starts --lead s BEFORE the phone (anticipation: lips round for
OO / close for M before the sound), plateaus over the phone, releases over --release s. Lip closures (PP) and lip-teeth
(FF) win over neighbours (priority weights), weak tongue consonants (DD kk nn RR) only partly displace the vowel around
them. Per frame the dominant viseme is chosen with a minimum hold (--min-hold frames); the second-strongest and its share
are kept for cross-fades. Openness: the vowel's nominal jaw opening x the dry-vocal envelope (or an A2F jawOpen curve).

  lsdub visemes --align align_018.json --t0 8.04 --frames 200 --fps 25 \
      --audio vocal_dry.wav --out vis_018.json [--arkit-csv vis_018_arkit.csv] [--strip-png s.png]
JSON per frame: {t, vis, vis2, mix (0..1 share of vis2), open (0..1), word, phone, wi}.
v12: the track also carries what the aligner knew about each word (lyric_check.word_table: heard, src, the lyric word,
Whisper's probability, mfa / sphinx / whisper+spread, the phrase's fallback, phones, held-note stretch) as `words` (the
words overlapping the track), `words_meta` (absolute alignment + lyric-sheet paths) and per frame `wi` (the index of the
frame's word in the alignment's words, -1 none): the mouth card's weak marks and heard: diffs (lsdub annotate, lsdub
paint --annotate) read them from the track. --no-words writes the v11 format (byte for byte).
"""
import argparse, csv, json, os
from ..io import ff
import numpy as np

VIS = ['sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS', 'nn', 'RR', 'aa', 'E', 'ih', 'oh', 'ou']
LIB = {'kk': 'DD', 'nn': 'DD'}          # visemes that share a library mouth
ARPA = {'P': 'PP', 'B': 'PP', 'M': 'PP', 'F': 'FF', 'V': 'FF', 'TH': 'TH', 'DH': 'TH', 'T': 'DD', 'D': 'DD',
        'K': 'kk', 'G': 'kk', 'NG': 'kk', 'CH': 'CH', 'JH': 'CH', 'SH': 'CH', 'ZH': 'CH', 'S': 'SS', 'Z': 'SS',
        'N': 'nn', 'L': 'nn', 'R': 'RR', 'ER': 'RR', 'Y': 'ih', 'W': 'ou',
        'AA': 'aa', 'AE': 'aa', 'AH': 'aa', 'AY': 'aa', 'AW': 'aa', 'EH': 'E', 'EY': 'E', 'IH': 'ih', 'IY': 'ih',
        'AO': 'oh', 'OW': 'oh', 'OY': 'oh', 'UW': 'ou', 'UH': 'ou', 'HH': None}
PRIO = {'PP': 3.0, 'FF': 2.2, 'ou': 1.3, 'oh': 1.2, 'CH': 1.2, 'TH': 1.0, 'SS': 0.9, 'aa': 1.0, 'E': 1.0, 'ih': 1.0,
        'DD': 0.55, 'kk': 0.45, 'nn': 0.5, 'RR': 0.7, 'sil': 0.6}
# nominal jaw opening per viseme (0 closed .. 1 wide)
OPEN = {'sil': 0.0, 'PP': 0.0, 'FF': 0.12, 'TH': 0.2, 'DD': 0.22, 'kk': 0.25, 'CH': 0.2, 'SS': 0.1, 'nn': 0.2,
        'RR': 0.25, 'aa': 1.0, 'E': 0.6, 'ih': 0.4, 'oh': 0.75, 'ou': 0.3}
# ARKit pose per viseme (Audio2Face / ARKit 52 names) — for the guide head (approach b)
ARKIT = {
    'sil': {'mouthClose': 0.05},
    'PP': {'mouthClose': 0.55, 'mouthPressLeft': 0.6, 'mouthPressRight': 0.6, 'mouthRollLower': 0.25, 'mouthRollUpper': 0.2, 'jawOpen': 0.06},
    'FF': {'jawOpen': 0.12, 'mouthRollLower': 0.7, 'mouthUpperUpLeft': 0.3, 'mouthUpperUpRight': 0.3, 'mouthClose': 0.1},
    'TH': {'jawOpen': 0.2, 'tongueOut': 0.5, 'mouthUpperUpLeft': 0.15, 'mouthUpperUpRight': 0.15},
    'DD': {'jawOpen': 0.18, 'mouthStretchLeft': 0.15, 'mouthStretchRight': 0.15},
    'kk': {'jawOpen': 0.22, 'mouthStretchLeft': 0.1, 'mouthStretchRight': 0.1},
    'CH': {'jawOpen': 0.15, 'mouthFunnel': 0.55, 'mouthPucker': 0.25, 'mouthUpperUpLeft': 0.2, 'mouthUpperUpRight': 0.2},
    'SS': {'jawOpen': 0.08, 'mouthStretchLeft': 0.35, 'mouthStretchRight': 0.35, 'mouthSmileLeft': 0.2, 'mouthSmileRight': 0.2},
    'nn': {'jawOpen': 0.18, 'mouthStretchLeft': 0.1, 'mouthStretchRight': 0.1},
    'RR': {'jawOpen': 0.2, 'mouthFunnel': 0.3, 'mouthPucker': 0.3},
    'aa': {'jawOpen': 0.65, 'mouthLowerDownLeft': 0.35, 'mouthLowerDownRight': 0.35, 'mouthUpperUpLeft': 0.15, 'mouthUpperUpRight': 0.15},
    'E': {'jawOpen': 0.38, 'mouthStretchLeft': 0.4, 'mouthStretchRight': 0.4, 'mouthLowerDownLeft': 0.3, 'mouthLowerDownRight': 0.3},
    'ih': {'jawOpen': 0.22, 'mouthSmileLeft': 0.35, 'mouthSmileRight': 0.35, 'mouthStretchLeft': 0.3, 'mouthStretchRight': 0.3},
    'oh': {'jawOpen': 0.5, 'mouthFunnel': 0.6, 'mouthPucker': 0.2},
    'ou': {'jawOpen': 0.15, 'mouthPucker': 0.85, 'mouthFunnel': 0.35},
}


def envelope_open(audio, t0, n, fps, floor_db=-30.0):
    raw = ff.run(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-t', f'{n / fps + 0.2:.3f}', '-i', audio, '-ac', '1',
                          '-ar', '16000', '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    hop = 16000 / fps
    out = []
    for i in range(n):
        c = int(i * hop + hop / 2)
        s = x[max(0, c - 480): c + 480]
        out.append(np.sqrt(np.mean(s ** 2)) if len(s) else 0)
    db = 20 * np.log10(np.array(out) + 1e-9)
    db -= np.percentile(db, 98)
    return np.clip((db - floor_db) / -floor_db, 0, 1) ** 0.7


def phone_list(align):
    ph = []
    for wi, w in enumerate(align['words']):
        for p in w['phones']:
            ph.append({'p': p['p'], 'start': p['start'], 'end': p['end'], 'word': w['w'], 'wi': wi})
    for i, p in enumerate(ph):      # HH / unknown -> next vowel's viseme
        v = ARPA.get(p['p'])
        if v is None:
            nxt = next((ARPA.get(q['p']) for q in ph[i + 1:] if ARPA.get(q['p'])), 'aa')
            v = nxt
        p['vis'] = v
    return ph


RH2VIS = {'A': 'PP', 'B': 'DD', 'C': 'E', 'D': 'aa', 'E': 'oh', 'F': 'ou', 'G': 'FF', 'H': 'nn'}


def rhubarb_phones(rh, mode):
    """Rhubarb cues (lsdub rhubarb JSON) as pseudo-phones. mode 'closures': only A (M/B/P) and G (F/V) cues, added
    on top of the alignment's phones (a second opinion on lip closures); 'fallback': every non-rest cue (no alignment)."""
    keep = ('A', 'G') if mode == 'closures' else tuple(RH2VIS)
    return [{'p': 'rh' + c['shape'], 'start': c['start'], 'end': c['end'], 'word': '', 'vis': RH2VIS[c['shape']]}
            for c in rh['cues'] if c['shape'] in keep]


def build(align, t0, n, fps, lead=0.06, release=0.05, min_hold=3, open_curve=None, extra_ph=(), word_index=False,
          word_close=0.0):
    ph = (phone_list(align) if align else []) + list(extra_ph)
    tt = t0 + (np.arange(n) + 0.5) / fps
    W = np.zeros((n, len(VIS)))
    W[:, 0] = 0.15         # small silence floor so gaps fall to rest
    word, phone, wix = [''] * n, [''] * n, [-1] * n
    for p in ph:
        a0, a1 = p['start'] - lead, p['start']
        b0, b1 = p['end'], p['end'] + release
        act = np.zeros(n)
        act[(tt >= a1) & (tt <= b0)] = 1
        m = (tt > a0) & (tt < a1)
        act[m] = 0.5 - 0.5 * np.cos(np.pi * (tt[m] - a0) / lead)
        m = (tt > b0) & (tt < b1)
        act[m] = 0.5 + 0.5 * np.cos(np.pi * (tt[m] - b0) / release)
        if p['end'] - p['start'] < 1 / fps:            # a phone shorter than a frame still gets one frame's peak
            k = int(np.clip(np.round((p['start'] + p['end']) / 2 * fps - t0 * fps - 0.5), 0, n - 1))
            act[k] = max(act[k], 1)
        W[:, VIS.index(p['vis'])] += act * PRIO[p['vis']]
        if p['word'] or not align:
            for k in np.where((tt >= p['start']) & (tt < p['end']))[0]: word[k], phone[k], wix[k] = p['word'], p['p'], p.get('wi', -1)
    # v13: a near-closed "rest" lip position BETWEEN words -- the mouth falls toward base in the gap between
    # one word and the next (the rise/fall of speech). A raised-cosine boost to the silence class, centred on
    # each inter-WORD boundary (NOT phone edges), over +-word_close s, moderate so a real phone still wins.
    if word_close and ph:
        wp = [p for p in ph if p.get('word')]
        spans = {}                                              # word-occurrence -> [start, end]
        for p in wp:
            key = p.get('wi', p['word'])
            s, e = spans.get(key, (p['start'], p['end']))
            spans[key] = (min(s, p['start']), max(e, p['end']))
        iv = sorted(spans.values())
        bnds = [(a[1] + b[0]) / 2 for a, b in zip(iv, iv[1:])]   # the MIDPOINT between consecutive words
        if bnds:
            bnds = np.array(bnds)
            peak = float(np.median(W.sum(1))) * 0.9 or 0.9       # moderate: deepens gaps, won't swamp a phone
            db = np.min(np.abs(tt[:, None] - bnds[None, :]), axis=1)
            near = db < word_close
            W[near, 0] += peak * (0.5 + 0.5 * np.cos(np.pi * db[near] / word_close))
    # dominant with minimum hold
    order = np.argsort(-W, 1)
    dom = order[:, 0].copy()
    i = 0
    while i < n:
        j = i
        while j < n and dom[j] == dom[i]: j += 1
        if j - i < min_hold and i > 0:
            # too short: keep only if it is a closure (PP/FF must be seen); else merge into the previous run
            if VIS[dom[i]] in ('PP', 'FF') and j < n:
                dom[i:min(n, i + min_hold)] = dom[i]
                j = min(n, i + min_hold)
            else: dom[i:j] = dom[i - 1]
        i = j
    frames = []
    for k in range(n):
        w = W[k].copy()
        d = dom[k]
        w2 = w.copy()
        w2[d] = -1
        s = int(np.argmax(w2))
        tot = w[d] + max(0, w[s])
        mix = float(max(0, w[s]) / tot) if tot > 0 else 0.0
        mix = min(mix, 0.5)
        vo = 0.5 * OPEN[VIS[d]] * (1 - mix) + 0.5 * OPEN[VIS[s]] * mix + 0.5 * OPEN[VIS[d]]
        o = float(np.clip(vo * (open_curve[k] if open_curve is not None else 1.0), 0, 1))
        frames.append({'t': round(float(t0 + k / fps), 3), 'vis': VIS[d], 'vis2': VIS[s], 'mix': round(mix, 3), 'open': round(o, 3),
                       'word': word[k], 'phone': phone[k]})
        if word_index: frames[-1]['wi'] = int(wix[k])     # v12: the frame's word in the alignment (the card's word facts)
    return frames


def arkit_csv(frames, fps, path, out_fps=60):
    names = sorted({k for p in ARKIT.values() for k in p})
    t = np.array([f['t'] for f in frames])
    n = int((t[-1] - t[0]) * out_fps) + 1
    P = np.array([[ARKIT[f['vis']].get(k, 0) * (1 - f['mix']) + ARKIT[f['vis2']].get(k, 0) * f['mix'] for k in names] for f in frames])
    # openness scales the jaw/lower-lip channels
    o = np.array([f['open'] for f in frames])
    base = np.array([0.5 * OPEN[f['vis']] + 0.5 * OPEN[f['vis2']] for f in frames])
    sc = np.where(base > 0.05, np.clip(o / np.maximum(base, 0.05), 0.4, 1.4), 1.0)
    for j, k in enumerate(names):
        if k in ('jawOpen', 'mouthLowerDownLeft', 'mouthLowerDownRight'): P[:, j] *= sc
    # light smoothing (3-tap) so the head does not snap
    P = np.stack([np.convolve(np.pad(P[:, j], 1, mode='edge'), [0.25, 0.5, 0.25], 'valid') for j in range(P.shape[1])], 1)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'time_s'] + names)
        for i in range(n):
            ti = t[0] + i / out_fps
            w.writerow([i, f'{ti:.4f}'] + [f'{np.interp(ti, t, P[:, j]):.4f}' for j in range(len(names))])


COL = {'sil': (90, 90, 90), 'PP': (230, 60, 60), 'FF': (240, 140, 40), 'TH': (220, 200, 60), 'DD': (150, 150, 120), 'kk': (130, 130, 110),
       'CH': (200, 90, 200), 'SS': (90, 200, 220), 'nn': (140, 160, 130), 'RR': (170, 120, 90), 'aa': (60, 200, 90),
       'E': (110, 220, 160), 'ih': (160, 230, 120), 'oh': (70, 130, 240), 'ou': (40, 80, 220)}


def strip_frame(frames, k, W=1920, H=520, span_s=4.0, fps=25, library=None):
    """Timeline strip image (BGR) for video frame k: words + phones + viseme bars, playhead centred; optional current
    library mouth at the left."""
    import cv2
    img = np.full((H, W, 3), 12, np.uint8)
    n = len(frames)
    x0 = 0
    if library is not None:
        m = library.get(LIB.get(frames[k]['vis'], frames[k]['vis']))
        if m is not None:
            s = H - 60
            mm = cv2.resize(m, (s, s))
            img[40:40 + s, 20:20 + s] = mm
            cv2.putText(img, f"{frames[k]['vis']}", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        x0 = H
    tw = W - x0 - 20
    px = tw / (span_s * fps)
    c = x0 + tw / 2
    def X(i): return int(c + (i - k) * px)
    y_w, y_p, y_v = 70, 150, 230
    cv2.putText(img, 'words', (x0 + 6, y_w - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(img, 'visemes (Oculus 15) + openness', (x0 + 6, y_v - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    lo, hi = max(0, int(k - span_s * fps / 2) - 2), min(n, int(k + span_s * fps / 2) + 2)
    for key, y, sz in (('word', y_w, 1.0), ('phone', y_p, 0.75)):
        i = lo
        while i < hi:
            j = i
            while j < hi and frames[j][key] == frames[i][key]: j += 1
            if frames[i][key]:
                cv2.rectangle(img, (X(i), y - 28), (X(j) - 2, y + 10), (60, 60, 70), -1)
                cv2.putText(img, frames[i][key], (X(i) + 4, y), cv2.FONT_HERSHEY_SIMPLEX, sz, (255, 255, 255), 2, cv2.LINE_AA)
            i = j
    for i in range(lo, hi):
        f = frames[i]
        h = int(20 + 180 * f['open'])
        cv2.rectangle(img, (X(i), y_v + 200 - h), (X(i + 1) - 1, y_v + 200), COL[f['vis']], -1)
        if i == lo or frames[i - 1]['vis'] != f['vis']:
            cv2.putText(img, f['vis'], (X(i) + 2, y_v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[f['vis']], 2, cv2.LINE_AA)
    cv2.line(img, (int(c), 30), (int(c), H - 10), (255, 255, 255), 2)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', help='lyric_align.py JSON (omit with --rhubarb-mode fallback)')
    ap.add_argument('--rhubarb', help='lsdub rhubarb JSON (song time)')
    ap.add_argument('--rhubarb-mode', choices=['closures', 'fallback'], default='closures')
    ap.add_argument('--t0', type=float, required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--fps', type=float, default=25)
    ap.add_argument('--audio', help='vocal stem for openness (envelope)')
    ap.add_argument('--a2f-csv', help='A2F curves; jawOpen used for openness')
    ap.add_argument('--lead', type=float, default=0.06)
    ap.add_argument('--release', type=float, default=0.05)
    ap.add_argument('--min-hold', type=int, default=3)
    ap.add_argument('--word-close', type=float, default=0.0, help='v13: a near-closed rest lip position between words '
                    '(seconds each side of every word boundary; 0 = off). The mouth falls toward base at word edges.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--arkit-csv')
    ap.add_argument('--strip-png')
    ap.add_argument('--no-words', dest='words', action='store_false', help='v12: no words / words_meta block and no per-frame wi '
                    '(the v11 track format, byte for byte)')
    a = ap.parse_args(argv)
    align = json.load(open(a.align)) if a.align else None
    oc = None
    if not align and not (a.rhubarb and a.rhubarb_mode == 'fallback'): raise SystemExit('--align, or --rhubarb with --rhubarb-mode fallback')
    xp = rhubarb_phones(json.load(open(a.rhubarb)), a.rhubarb_mode) if a.rhubarb else []
    if a.a2f_csv:
        with open(a.a2f_csv) as f:
            r = csv.reader(f)
            h = next(r)
            R = np.array([[float(x) for x in row] for row in r])
        j = np.interp(a.t0 + np.arange(a.frames) / a.fps, R[:, h.index('time_s')], R[:, h.index('jawOpen')])
        oc = np.clip(j / max(1e-6, np.percentile(j, 95)), 0, 1.2)
    elif a.audio:
        oc = envelope_open(a.audio, a.t0, a.frames, a.fps)
    words = bool(a.words and align)
    fr = build(align, a.t0, a.frames, a.fps, a.lead, a.release, a.min_hold, oc, xp, word_index=words,
               word_close=a.word_close)
    T = {'fps': a.fps, 't0': a.t0, 'align': a.align, 'rhubarb': a.rhubarb and f'{a.rhubarb} ({a.rhubarb_mode})', 'visemes': VIS, 'library_map': LIB}
    if words:                                    # v12: the aligner's per-word facts, for the mouth card / lyric check
        from . import lyric_check
        lyr = lyric_check.find_file(align.get('lyrics'), a.align)
        T['words'] = lyric_check.word_table(align, a.t0, a.t0 + a.frames / a.fps, open(lyr, encoding='utf-8').read() if lyr else None)
        T['words_meta'] = dict(align=os.path.abspath(a.align), lyrics=lyr and os.path.abspath(lyr), aligner=align.get('aligner'),
                               lyric_match_ratio=align.get('lyric_match_ratio'), lyric_from_notes=lyric_check.lyric_words(align)[1])
    T['frames'] = fr
    json.dump(T, open(a.out, 'w'), indent=0)
    if a.arkit_csv: arkit_csv(fr, a.fps, a.arkit_csv)
    if a.strip_png:
        import cv2
        cv2.imwrite(a.strip_png, strip_frame(fr, len(fr) // 2, span_s=len(fr) / a.fps))
    runs = []
    for f in fr:
        if not runs or runs[-1][0] != f['vis']: runs.append([f['vis'], 1])
        else: runs[-1][1] += 1
    print(' '.join(f'{v}x{c}' for v, c in runs))


if __name__ == '__main__':
    main()
