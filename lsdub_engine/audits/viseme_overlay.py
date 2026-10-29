#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""viseme_overlay.py — `lsdub annotate`: the lyrics annotated to simple mouth shapes, one compact card per frame.

  lsdub annotate --video in.mp4 --track track.json --out out.mp4 [--corner auto|lower-right|lower-left|band] [--scale 1.0]
                 [--lyrics lyrics.txt] [--skip-words words.txt] [--vocalisations auto|skip|rest|mouth]
                 [--faces heads.json] [--face-cache faces.json] [--edl edl.json] [--overlay-out card.mov] [--log place.json]
                 [--align align.json] [--lyric-check check.json] [--lyrics-suggested lyrics_suggested.txt] [--lyric-sheet f]

THE CARD (lower-right corner of the picture by default): the word being sung (big) between the previous and the next
word (dimmed); the current viseme class (the 15 Oculus visemes of the track) as a SIMPLE MOUTH GLYPH -- procedurally
drawn line icons, no image assets: rest = a closed line, PP = lips pressed, FF = upper teeth on the lower lip, TH =
tongue tip between the teeth, DD / nn / kk / SS / CH = small openings (tongue up / tongue up, narrower / tongue back /
teeth together / pursed and flared), RR / ou = pucker, oh = round, aa = wide open, E / ih = wide and stretched -- with
its class name (and the runner-up class when the track mixes it in); the track's openness as a small bar; the next ~1 s
of viseme classes as coloured ticks (height = openness). A skipped vocalisation (`lsdub track skip`, or the same
lexicon applied here to an unmarked track) shows its word dimmed and struck through with 'skip' (the default, as lsdub
paint's since v11; 'rest' with --vocalisations rest; with mouth / an unmarked track under auto -- v9, mouthed -- the
word is kept and tagged 'ad-lib'). --scale 1 is designed for 1280x720 (default: picture height / 720); 1.0 stays legible on a 2x mouth zoom.
v12 (the card as a toggle, lsdub paint --annotate): what the aligner knew about the word (lsdub/core/lyric_check.py; the
track's `words` block from `lsdub visemes`, else its alignment, --align; the card grows 16 px for a second word line when a
track has such facts, else it is v10a's card): a WEAK word (confidence < 0.6: whisper-timed,
phrase fell back, phones squeezed / stretched, low Whisper probability) gets a dotted underline and a small '?' badge, its
reason on the second line; a word whose HEARD token differs from its lyric word gets 'heard: x' on the second line under
the written word (+ 'lyric: y' when the card's word is the sung one); the 1 s timeline marks such frames above their
ticks. Skipped / rested vocalisations show neither. --lyric-check writes the check (JSON + .md: every mismatch, weak word
and vocalisation span, video frames + song time), --lyrics-suggested the lyric sheet with '[heard: x]' suggestions.
FACE RULE (captions never cover a face): per frame the card's box is tested against every face box -- MediaPipe face
landmarks on the frame (cached in --face-cache), plus the boxes of --faces (a keep-out JSON: {"frames": [[{"b": [x0, y0,
x1, y1]}, ...]]}, e.g. host-studio's lyric_faces.py) -- each padded like the studio face gate (10 %, 16 % above). A clip
(one --edl segment, or the whole video) uses the lower-right corner when it is clear on every frame of the clip, else
the lower-left, else a band appended BELOW the picture for that clip (whole clip, so the card never jumps; the video
then carries the band throughout, empty for the corner clips). --corner band forces the band. The placement per clip
is printed and written to --log.
EDL (--edl): {"segments": [{"label": "S1", "track": "vis_S1.json" | null, "first": 0, "nf": 94 | "frames": [k, ...],
"vocalisations": "auto", "note": "raw take"}]} -- consecutive pieces of the video, each annotated from its own track
(track frame per video frame: first + i, or the explicit list, e.g. slow-motion repeats); a null track = no card.
Multi-face: a track with a face_id names it on its card; several tracks for one segment ("tracks": [...]) stack their
cards. Audio is copied from --video.
"""
import argparse, json, os, sys
from ..io import ff
import numpy as np, cv2

from ..core import vocalisations as vz

VIS = ['sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS', 'nn', 'RR', 'aa', 'E', 'ih', 'oh', 'ou']
# RGB, grouped: closures warm, tongue / teeth consonants neutral, round blue, open green
COL = {'sil': (118, 118, 124), 'PP': (236, 72, 72), 'FF': (246, 142, 52), 'TH': (240, 202, 64), 'DD': (204, 196, 126),
       'kk': (172, 166, 112), 'nn': (188, 180, 140), 'SS': (88, 212, 222), 'CH': (202, 122, 224), 'RR': (120, 158, 246),
       'ou': (78, 112, 246), 'oh': (66, 166, 255), 'aa': (72, 212, 102), 'E': (150, 230, 120), 'ih': (204, 240, 144)}
NAME = {'sil': 'rest', 'PP': 'P B M', 'FF': 'F V', 'TH': 'TH', 'DD': 'T D', 'kk': 'K G', 'CH': 'CH SH J', 'SS': 'S Z', 'nn': 'N L',
        'RR': 'R', 'aa': 'AA', 'E': 'EH', 'ih': 'EE', 'oh': 'OH', 'ou': 'OO W'}
BG, INK, DIM, CAV, TEETH, TONGUE = (22, 22, 30), (246, 246, 246), (128, 128, 138), (52, 20, 34), (236, 234, 224), (214, 104, 118)
TAG = {'skip': (244, 170, 40), 'rest': (130, 180, 255), 'ad-lib': (150, 150, 160)}
WEAK, HEARD = (255, 214, 80), (255, 122, 170)   # v12: a weakly aligned word / a heard token that is not the lyric word
FONTS = {False: ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf', '/Library/Fonts/Arial.ttf'),
         True: ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
                '/Library/Fonts/Arial Bold.ttf')}
_FC = {}
SS_ = 3                                  # supersampling of the card (drawn at 3x, area-downscaled: anti-aliased)
CARD = (340, 128)                        # card size at --scale 1 (1280x720): clears a close-up's padded face box
LINE2 = 16                               # v12: +16 px for the second word line (heard: / a weak word's reason) when the
                                         # annotated tracks carry word facts; without facts the card is v10a's, pixel for pixel
MARGIN = 14


def _font(size, bold=False):
    """DejaVu Sans (regular / bold) at size px; the regular face when there is no bold; PIL's own font as the last resort."""
    from PIL import ImageFont
    key = (int(size), bool(bold))
    if key not in _FC:
        f = None
        for p in FONTS[bool(bold)] + FONTS[False]:
            if os.path.exists(p):
                f = ImageFont.truetype(p, int(size))
                break
        if f is None:
            try: f = ImageFont.load_default(size=int(size))
            except TypeError: f = ImageFont.load_default()
        _FC[key] = f
    return _FC[key]


# ------------------------------------------------------------------------------------------------ the glyphs --
# per class: outer half-width w, upper / lower lip thickness hu / hl, opening half-width wi / half-height oi, opening
# shape, extras. Units: the glyph box is x in [-1, 1], y in [-0.72, 0.72].
GLYPH = {
    'sil': dict(w=0.84, hu=0.25, hl=0.29, wi=0.0, oi=0.0, shape='line'),
    'PP': dict(w=0.76, hu=0.16, hl=0.18, wi=0.0, oi=0.0, shape='press'),
    'FF': dict(w=0.84, hu=0.24, hl=0.17, wi=0.56, oi=0.11, shape='slit', tu=0.85, tuck=True),
    'TH': dict(w=0.86, hu=0.23, hl=0.25, wi=0.5, oi=0.14, shape='oval', tu=0.38, tongue='tip'),
    'DD': dict(w=0.88, hu=0.23, hl=0.25, wi=0.52, oi=0.15, shape='oval', tu=0.3, tongue='up'),
    'nn': dict(w=0.86, hu=0.23, hl=0.25, wi=0.44, oi=0.11, shape='oval', tongue='up'),
    'kk': dict(w=0.88, hu=0.23, hl=0.25, wi=0.5, oi=0.17, shape='oval', tu=0.28, tongue='back'),
    'SS': dict(w=0.94, hu=0.22, hl=0.23, wi=0.66, oi=0.11, shape='slit', tu=0.46, tl=0.46),
    'CH': dict(w=0.66, hu=0.25, hl=0.27, wi=0.34, oi=0.19, shape='square', tu=0.3, tl=0.3, flare=True),
    'RR': dict(w=0.6, hu=0.26, hl=0.28, wi=0.27, oi=0.16, shape='round', pucker=4),
    'ou': dict(w=0.48, hu=0.27, hl=0.29, wi=0.15, oi=0.15, shape='round', pucker=6),
    'oh': dict(w=0.64, hu=0.22, hl=0.24, wi=0.34, oi=0.33, shape='round', tongue='low'),
    'aa': dict(w=0.84, hu=0.19, hl=0.21, wi=0.58, oi=0.42, shape='oval', tu=0.2, tongue='low'),
    'E': dict(w=0.98, hu=0.19, hl=0.22, wi=0.74, oi=0.24, shape='wide', tu=0.3),
    'ih': dict(w=1.0, hu=0.18, hl=0.2, wi=0.76, oi=0.13, shape='smile', tu=0.42, tl=0.3),
}


def _opening(g, n=48):
    """the inner-lip contour (closed, x right / y down), or the closed line for line / press."""
    wi, oi, sh = g['wi'], g['oi'], g['shape']
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    if sh == 'round': return np.stack([wi * c, oi * s], 1)
    if sh == 'square':
        e = 0.45
        return np.stack([wi * np.sign(c) * np.abs(c) ** e, oi * np.sign(s) * np.abs(s) ** e], 1)
    if sh in ('wide', 'smile', 'slit', 'oval'):
        p = 0.75 if sh in ('wide', 'oval') else 0.55
        y = oi * np.sign(s) * np.abs(s) ** p
        if sh == 'smile': y = y + 0.06 * (1 - (wi * c / max(wi, 1e-6)) ** 2) * 0.5   # a slight upward curve at the corners
        return np.stack([wi * c, y], 1)
    return None


def _outer(g, n=40):
    """the outer lip contour: upper lip (cupid's bow) from the left corner to the right, lower lip back."""
    w, hu, hl, oi = g['w'], g['hu'], g['hl'], g['oi']
    x = np.linspace(-w, w, n)
    u = x / w
    bow = 1 - 0.28 * np.exp(-(u / 0.18) ** 2)                                      # the dip at the philtrum
    top = -(oi * 0.9 + hu * 1.05) * np.sqrt(np.clip(1 - u ** 2, 0, 1)) ** 0.7 * bow
    bot = (oi * 0.9 + hl * 1.1) * np.sqrt(np.clip(1 - u ** 2, 0, 1)) ** 0.85
    if g.get('tuck'): bot = bot * 0.75                                             # FF: the lower lip rolls in under the teeth
    return np.vstack([np.stack([x, top], 1), np.stack([x[::-1], bot[::-1]], 1)])


def glyph(vis, size, dim=False, rest=False):
    """the mouth glyph of viseme class `vis` as an RGB image (size = (w, h) px) on the card background + its alpha."""
    W, H = size
    S = SS_ * 2
    Wb, Hb = W * S, H * S
    img = np.zeros((Hb, Wb, 3), np.uint8)
    img[:] = BG
    al = np.zeros((Hb, Wb), np.uint8)
    g = GLYPH.get('sil' if rest else vis, GLYPH['sil'])
    sc = min(Wb / 2.25, Hb / 1.6)
    cx, cy = Wb / 2, Hb / 2
    P = lambda q: np.round(np.asarray(q, float) * sc + [cx, cy]).astype(np.int32)
    Pt = lambda q: tuple(int(v) for v in P(q))
    lw = max(1, int(round(0.055 * sc)))
    ink = INK
    cav, teeth, tongue = CAV, TEETH, TONGUE
    outer = _outer(g)
    inner = _opening(g)
    cv2.fillPoly(img, [P(outer)], (178, 92, 104))
    cv2.fillPoly(al, [P(outer)], 255)          # lips, a soft lip colour
    if inner is not None:
        cv2.fillPoly(img, [P(inner)], cav)
        mask = np.zeros((Hb, Wb), np.uint8)
        cv2.fillPoly(mask, [P(inner)], 255)
        tm = np.zeros_like(mask)
        ytop, ybot = inner[:, 1].min(), inner[:, 1].max()
        hgt = ybot - ytop
        if g.get('tu'): cv2.rectangle(tm, Pt([-g['wi'] - 0.1, ytop - 0.2]), Pt([g['wi'] + 0.1, ytop + g['tu'] * hgt]), 255, -1)
        if g.get('tl'): cv2.rectangle(tm, Pt([-g['wi'] - 0.1, ybot - g['tl'] * hgt]), Pt([g['wi'] + 0.1, ybot + 0.2]), 255, -1)
        tm &= mask
        img[tm > 0] = teeth
        if tm.any() and hgt * sc > 6 * S:                                          # a few tooth gaps
            for k in np.linspace(-g['wi'] * 0.62, g['wi'] * 0.62, 4):
                xx = int(P([k, 0])[0])
                col = tm[:, xx] > 0
                img[col, max(0, xx - max(1, lw // 4)):xx + max(1, lw // 4)] = (170, 168, 160)
        tg = g.get('tongue')
        if tg:
            tmask = np.zeros_like(mask)
            if tg == 'low': cv2.ellipse(tmask, Pt([0, g['oi'] * 0.95]), (int(g['wi'] * 0.62 * sc), int(g['oi'] * 0.55 * sc)), 0, 180, 360, 255, -1)
            elif tg == 'tip': cv2.ellipse(tmask, Pt([0, ytop + g['tu'] * hgt + 0.1]), (int(g['wi'] * 0.4 * sc), int(0.16 * sc)), 0, 0, 360, 255, -1)
            elif tg == 'up': cv2.ellipse(tmask, Pt([0, -g['oi'] * 0.35]), (int(g['wi'] * 0.4 * sc), int(g['oi'] * 0.6 * sc)), 0, 0, 360, 255, -1)
            elif tg == 'back': cv2.ellipse(tmask, Pt([0, g['oi'] * 0.75]), (int(g['wi'] * 0.85 * sc), int(g['oi'] * 0.7 * sc)), 0, 180, 360, 255, -1)
            if tg != 'tip': tmask &= mask
            img[tmask > 0] = tongue
        cv2.polylines(img, [P(inner)], True, ink, max(1, int(0.6 * lw)), cv2.LINE_AA)
    else:   # closed: the lips' contact line (pressed: a heavy dark line, the corners marked)
        u = np.linspace(-g['w'] * 0.96, g['w'] * 0.96, 30)
        line = np.stack([u, 0.02 * (1 - (u / g['w']) ** 2) - 0.01], 1)
        if g['shape'] == 'press': cv2.polylines(img, [P(line)], False, (96, 28, 44), int(lw * 1.8), cv2.LINE_AA)
        else: cv2.polylines(img, [P(line)], False, ink, max(1, int(0.8 * lw)), cv2.LINE_AA)
        if g['shape'] == 'press':
            for sx in (-1, 1):
                cv2.line(img, Pt([sx * g['w'] * 0.96, -0.1]), Pt([sx * g['w'] * 0.96, 0.1]), ink, lw, cv2.LINE_AA)
    cv2.polylines(img, [P(outer)], True, ink, lw, cv2.LINE_AA)
    if g.get('pucker'):
        k = g['pucker']
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + (np.pi / k if k == 4 else 0)
        for a in ang:
            r0, r1 = 1.0, 1.32
            ex, ey = g['w'] * np.cos(a), (g['oi'] + g['hu']) * 1.05 * np.sin(a)
            cv2.line(img, Pt([ex * r0 * 1.06, ey * r0 * 1.06]), Pt([ex * r1, ey * r1]), ink, lw, cv2.LINE_AA)
    if g.get('flare'):
        for sx in (-1, 1):
            for dy in (-0.12, 0.12):
                cv2.line(img, Pt([sx * g['w'] * 1.06, dy * 0.6]), Pt([sx * g['w'] * 1.3, dy * 1.6]), ink, lw, cv2.LINE_AA)
    al = np.maximum(al, (np.abs(img.astype(int) - BG).max(2) > 8).astype(np.uint8) * 255)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    al = cv2.resize(al, (W, H), interpolation=cv2.INTER_AREA)
    if dim:
        img = (0.35 * img.astype(np.float32) + 0.65 * np.array(BG, np.float32)).astype(np.uint8)
    return img, al


# ------------------------------------------------------------------------------------------------ the track --
class TrackView:
    """per-frame card content of one viseme track: word, previous / next word, class, runner-up, openness, skip state."""

    def __init__(self, path, mode='skip', skip_words=None, lyrics=None, align=None, label='', note='', merge_gap=vz.MERGE_GAP):
        T = json.load(open(path))
        self.T = T
        fr = T.get('frames') or []
        self.n = len(fr)
        self.fps = float(T.get('fps', 25))
        self.label, self.note, self.face_id = label, note, T.get('face_id')
        self.fr = fr
        marked = 'skips' in T
        self.mode = ('skip' if marked else 'mouth') if mode == 'auto' else mode
        # the spans exactly as the render finds them (vocalisations.for_warp: the same alignment lookup and merge gap)
        found = vz.from_track(T) if marked else vz.find_spans(T, align, skip_words or (), lyrics, merge_gap, track_path=path)
        wm = (T.get('words_meta') or {}).get('align')                        # v12: the word facts may also use the
        align = align or (wm if wm and os.path.exists(wm) else None)         # alignment the track names absolutely
        self.vocal = vz.skip_mask(found, self.n)
        self.found = found
        words = [str(f.get('word', '') or '') for f in fr]
        self.words = words
        runs = []
        k = 0
        while k < self.n:
            j = k
            while j + 1 < self.n and words[j + 1] == words[k]: j += 1
            runs.append((k, j, words[k]))
            k = j + 1
        self.run_of = np.zeros(self.n, int)
        for r, (a, b, _) in enumerate(runs): self.run_of[a:b + 1] = r
        self.runs = runs
        self.info = self.word_facts(T, align, path)

    def word_facts(self, T, align=None, path=None):
        """v12: per frame the aligner's facts about its word (lyric_check.word_state: conf, weak, reasons, diff, heard,
        lyric) or None: from the track's words block + per-frame wi (lsdub visemes since v12), else from its alignment
        (the track's 'align' path / --align) mapped onto the frames as viseme_track.build assigns words."""
        from ..core import lyric_check as lc
        self.recs, self.wi, self.words_src, self.meta = {}, [-1] * self.n, None, dict(T.get('words_meta') or {})
        recs = {int(r['i']): r for r in T.get('words') or [] if isinstance(r, dict) and 'i' in r}
        wi = [f.get('wi') for f in self.fr]
        if recs and all(x is not None for x in wi):
            self.words_src = 'track'
        else:
            A = vz.load_align(T, align, path)
            ai = vz._frame_word_index(T, A) if A is not None and self.n else None
            if ai is None: return [None] * self.n
            lyr = lc.find_file(A.get('lyrics'), A.get('_path'))
            sheet = open(lyr, encoding='utf-8').read() if lyr else None
            t0 = float(T.get('t0', self.fr[0].get('t', 0.0)))
            recs = {r['i']: r for r in lc.word_table(A, t0, t0 + self.n / self.fps, sheet)}
            miss = sorted({int(k) for k in ai if k >= 0} - set(recs))      # (a frame's word that starts / ends outside the window)
            if miss:
                allr = lc.word_table(A, sheet=sheet)
                recs.update({k: allr[k] for k in miss})
            wi = [int(x) for x in ai]
            self.words_src = A.get('_path')
            self.meta.setdefault('align', os.path.abspath(A['_path']) if A.get('_path') else None)
            self.meta.setdefault('lyrics', lyr and os.path.abspath(lyr))
        self.recs = recs
        self.wi = [int(x) for x in wi]
        st = {i: lc.word_state(r) for i, r in recs.items()}
        return [st.get(w) if w >= 0 else None for w in self.wi]

    def facts(self, k):
        """the word facts of frame k when the card shows them (not on a skipped / rested vocalisation), else None."""
        if not (0 <= k < self.n) or self.state(k) in ('skip', 'rest'): return None
        return self.info[k]

    def prev_next(self, k):
        r = self.run_of[k]
        cur = self.runs[r][2]
        prv = next((self.runs[q][2] for q in range(r - 1, -1, -1) if self.runs[q][2]), '')
        nxt = next((self.runs[q][2] for q in range(r + 1, len(self.runs)) if self.runs[q][2]), '')
        return prv, cur, nxt

    def state(self, k):
        """'skip' | 'rest' | 'ad-lib' | '' for frame k."""
        if not (0 <= k < self.n) or not self.vocal[k]: return ''
        return {'skip': 'skip', 'rest': 'rest'}.get(self.mode, 'ad-lib')


# ------------------------------------------------------------------------------------------------- the card --
class Card:
    def __init__(self, scale=1.0, line2=False, cards=None):
        self.l2 = LINE2 if line2 else 0                                      # v12: the second word line (word facts)
        # v13: the FACE's own detailed mouth shapes (lsdub_engine.core.mouth_cards) instead of the generic glyph
        self.cards_dir = cards if (cards and os.path.isdir(cards)) else None
        self.card_tiles = {}
        self.s = float(scale)
        self.W, self.H = int(round(CARD[0] * self.s)), int(round((CARD[1] + self.l2) * self.s))
        self.cache = {}
        b = SS_ * self.s
        self.f_word, self.f_side, self.f_small, self.f_tag = _font(31 * b, True), _font(15 * b), _font(12 * b), _font(12 * b, True)
        self.f_cls = _font(15 * b, True)
        self.f_line2 = _font(13 * b)
        self.f_line2b = _font(13 * b, True)

    def _face_tile(self, vis):
        """the face's own detailed mouth-shape tile (RGB) for viseme `vis`, or None (fall back to the glyph)."""
        if not self.cards_dir: return None
        if vis not in self.card_tiles:
            p = os.path.join(self.cards_dir, f'{vis}.png')
            im = cv2.imread(p) if os.path.exists(p) else None
            self.card_tiles[vis] = None if im is None else im[:, :, ::-1].copy()   # BGR -> RGB
        return self.card_tiles[vis]

    def render(self, tv, k):
        """the card for frame k of TrackView tv -> (RGB image, alpha), both (H, W)."""
        key = (id(tv), k)
        if key in self.cache: return self.cache[key]
        from PIL import Image, ImageDraw
        s = SS_ * self.s
        W, H = self.W * SS_, self.H * SS_
        k = int(np.clip(k, 0, tv.n - 1))
        f = tv.fr[k]
        st = tv.state(k)
        img = Image.new('RGB', (W, H), BG)
        al = Image.new('L', (W, H), 0)
        d = ImageDraw.Draw(img)
        da = ImageDraw.Draw(al)
        r = int(14 * s)
        d.rounded_rectangle([0, 0, W - 1, H - 1], r, fill=BG)
        da.rounded_rectangle([0, 0, W - 1, H - 1], r, fill=205)
        vis = f.get('vis', 'sil')
        v2 = f.get('vis2', vis)
        mix = float(f.get('mix', 0.0) or 0.0)
        op = float(f.get('open', 0.0) or 0.0)
        # glyph (left; v12: centred on the taller card)
        gx, gy, gw, gh = int(9 * s), int((9 + self.l2 / 2) * s), int(94 * s), int(66 * s)
        tile = self._face_tile('sil' if st == 'rest' else vis)
        if tile is not None:                                                 # the FACE's own shape (square), fit the slot
            d2 = gh
            tt = cv2.resize(tile, (d2, d2), interpolation=cv2.INTER_AREA)
            if st == 'skip': tt = (tt.astype(np.float32) * 0.45).astype(np.uint8)
            ox = gx + (gw - d2) // 2
            img.paste(Image.fromarray(tt), (ox, gy))
            da.rectangle([ox, gy, ox + d2, gy + gh], fill=255)
        else:
            gi, ga = glyph(vis, (gw, gh), dim=(st == 'skip'), rest=(st == 'rest'))
            img.paste(Image.fromarray(gi), (gx, gy))
            al.paste(Image.fromarray(np.maximum(ga, 205)), (gx, gy))
        cls = 'sil' if st == 'rest' else vis
        col = COL.get(cls, INK) if st != 'skip' else DIM
        lab = cls + (f' +{v2}' if (mix >= 0.3 and v2 != vis and st == '') else '')
        d.text((gx + gw // 2, gy + gh + int(4 * s)), lab, font=self.f_cls, fill=col, anchor='mt')
        d.text((gx + gw // 2, gy + gh + int(22 * s)), "take's own" if st == 'skip' else NAME.get(cls, ''), font=self.f_small, fill=DIM, anchor='mt')
        # words (right): previous (dim) / CURRENT (big) / next (dim)
        x0 = int(114 * s)
        xr = W - int(10 * s)
        prv, cur, nxt = tv.prev_next(k)
        head = ' · '.join(x for x in (tv.label, f'f{k}', tv.note) if x) + (f'  face {tv.face_id}' if tv.face_id is not None else '')
        d.text((x0, int(8 * s)), head, font=self.f_small, fill=DIM)
        yb = int(60 * s)                                                     # word baseline
        cw = cur if cur else '·'
        fw = self.f_word
        wcur = d.textlength(cw, font=fw)
        while wcur > (xr - x0) * 0.62 and fw.size > 14 * s:
            fw = _font(fw.size * 0.9, True)
            wcur = d.textlength(cw, font=fw)
        def fit(t, room):
            if not t or room <= 0: return ''
            while t and d.textlength(t, font=self.f_side) > room: t = t[:-2] + '…' if len(t) > 2 else ''
            return t
        room = (xr - x0 - wcur) / 2 - 10 * s
        pv, nx = fit(prv, room), fit(nxt, room)
        wp = d.textlength(pv, font=self.f_side) if pv else 0
        xc = x0 + (wp + 10 * s if pv else 0)
        if pv: d.text((x0, yb), pv, font=self.f_side, fill=DIM, anchor='ls')
        wc_col = DIM if st in ('skip', 'rest') else (INK if cur else DIM)
        d.text((xc, yb), cw, font=fw, fill=wc_col, anchor='ls')
        if st in ('skip', 'rest') and cur:                                   # struck through
            ym = yb - int(fw.size * 0.33)
            d.line([xc - 2 * s, ym, xc + wcur + 2 * s, ym], fill=TAG[st], width=max(2, int(3 * s)))
        if nx: d.text((xc + wcur + 10 * s, yb), nx, font=self.f_side, fill=DIM, anchor='ls')
        fx = tv.facts(k) if cur and self.l2 else None                       # v12: weak mark + heard: line (never on a skip)
        if fx and fx['weak']:
            yu = yb + int(4 * s)
            dot = max(2, int(2.2 * s))
            x = xc
            while x < xc + wcur:                                             # dotted underline
                d.ellipse([x, yu, x + dot, yu + dot], fill=WEAK)
                x += 2.2 * dot
            rq = int(7 * s)
            qx, qy = int(xc + wcur + rq + 2 * s), yb - int(fw.size * 0.72)
            d.ellipse([qx - rq, qy - rq, qx + rq, qy + rq], fill=WEAK)       # the '?' badge at the word's top right
            d.text((qx, qy), '?', font=self.f_tag, fill=(18, 18, 22), anchor='mm')
        if fx and (fx['diff'] or fx['weak']):
            y2 = yb + int((self.l2 + 2) * s)
            x2 = xc
            room = xr - xc
            if fx['diff']:
                parts = [('heard: ', self.f_line2, HEARD), (str(fx['heard'] or '').strip(), self.f_line2b, HEARD)]
                if fx['kind'] == 'not-in-lyric': parts.append(('  not in the lyric', self.f_line2, DIM))
                elif fx['lyric'] and fx['word'] != fx['lyric']: parts.append((f"  lyric: {fx['lyric_text']}", self.f_line2, DIM))
            else:
                from ..core import lyric_check as lc
                parts = [('? ' + lc.short_reason(fx['reasons']), self.f_line2, WEAK)]
            for t, f, c in parts:
                while t and d.textlength(t, font=f) > room: t = t[:-2] + '…' if len(t) > 2 else ''
                if not t: break
                d.text((x2, y2), t, font=f, fill=c, anchor='ls')
                w_ = d.textlength(t, font=f)
                x2 += w_
                room -= w_
        if st:                                                               # the tag: skip / rest / ad-lib
            tw = d.textlength(st, font=self.f_tag) + 12 * s
            tx1 = xr
            tx0 = tx1 - tw
            ty0 = int(6 * s)
            ty1 = ty0 + int(18 * s)
            d.rounded_rectangle([tx0, ty0, tx1, ty1], int(5 * s), fill=TAG[st])
            d.text(((tx0 + tx1) / 2, (ty0 + ty1) / 2), st, font=self.f_tag, fill=(18, 18, 22), anchor='mm')
        # openness bar
        by = int((74 + self.l2) * s)
        bh = int(9 * s)
        bx0 = x0 + int(40 * s)
        bx1 = xr - int(36 * s)
        d.text((x0, by + bh // 2), 'open', font=self.f_small, fill=DIM, anchor='lm')
        d.rounded_rectangle([bx0, by, bx1, by + bh], int(3 * s), fill=(48, 48, 58))
        if op > 0.005: d.rounded_rectangle([bx0, by, bx0 + max(int(bh), int((bx1 - bx0) * min(1.0, op))), by + bh], int(3 * s), fill=col if st != 'skip' else DIM)
        d.text((xr, by + bh // 2), f'{op:.2f}', font=self.f_small, fill=DIM, anchor='rm')
        # timeline: the next ~1 s of viseme classes (this frame first, white outline)
        ty = int((92 + self.l2) * s)
        th = int(26 * s)
        nt = int(round(tv.fps))
        tx0 = x0
        tx1 = xr - int(30 * s)
        step = (tx1 - tx0) / nt
        mg = min(1 * s, step * 0.2)                      # inter-bar gap scales with step (high fps => many thin bars)
        for j in range(nt):
            q = k + j
            if q >= tv.n: break
            fq = tv.fr[q]
            c = fq.get('vis', 'sil')
            sq = tv.state(q)
            hgt = int(th * (0.22 + 0.78 * min(1.0, float(fq.get('open', 0) or 0))))
            xa = int(tx0 + j * step + mg)
            xb = max(xa + 1, int(tx0 + (j + 1) * step - mg))
            fill = DIM if sq == 'skip' else COL.get('sil' if sq == 'rest' else c, INK)
            d.rectangle([xa, ty + th - hgt, xb, ty + th], fill=fill)
            if sq == 'skip': d.line([xa, ty + th - hgt, xb, ty + th], fill=BG, width=max(1, int(s)))
            if j == 0: d.rectangle([xa - int(1.5 * s), ty - int(1.5 * s), xb + int(1.5 * s), ty + th + int(1.5 * s)], outline=INK, width=max(1, int(1.2 * s)))
            fq_ = tv.facts(q) if self.l2 else None                           # v12: weak / heard frames marked above their ticks
            if fq_ and (fq_['diff'] or fq_['weak']):
                d.rectangle([xa, ty - int(5 * s), xb, ty - int(3 * s)], fill=HEARD if fq_['diff'] else WEAK)
        d.text((xr, ty + th // 2), '1 s', font=self.f_small, fill=DIM, anchor='rm')
        out = (np.asarray(img.resize((self.W, self.H), Image.LANCZOS)), np.asarray(al.resize((self.W, self.H), Image.LANCZOS)))
        if len(self.cache) > 4000: self.cache.clear()
        self.cache[key] = out
        return out


# ------------------------------------------------------------------------------------------------- the faces --
def face_boxes(frames_iter, n, cache=None, video=None):
    """MediaPipe face landmarks (IMAGE mode, up to 4 faces) -> per frame [x0, y0, x1, y1] boxes (the mesh's extent:
    brow to chin, the forehead added); cached as JSON {video, n, frames}."""
    if cache and os.path.exists(cache):
        z = json.load(open(cache))
        if z.get('n') == n and (video is None or z.get('video') == os.path.abspath(video)): return z['frames']
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    from .._models import face_landmarker_path
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_landmarker_path()), running_mode=vision.RunningMode.IMAGE, num_faces=4,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
    out = []
    for fr in frames_iter:
        H, W = fr.shape[:2]
        bx = []
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr[:, :, ::-1])))
        for f in r.face_landmarks:
            xs = np.array([p.x for p in f]) * W
            ys = np.array([p.y for p in f]) * H
            h = ys.max() - ys.min()
            bx.append([int(xs.min()), int(ys.min() - 0.18 * h), int(xs.max()), int(ys.max())])
        out.append(bx)
    lm.close()
    if cache: json.dump(dict(video=None if video is None else os.path.abspath(video), n=n, frames=out), open(cache, 'w'))
    return out


def gating(h, sure=0.5, yunet_floor=0.9):
    """the studio face gate's GATING boxes (lyric_face_check.py): MediaPipe landmarker faces, BlazeFace >= sure, YuNet >= 0.9;
    a plain box (no source) always gates. Weak detections and person-matte heads are advisory there."""
    if not isinstance(h, dict) or 'src' not in h: return True
    src, sc = h['src'], h.get('score', 1.0)
    return src == 'mplm' or (src in ('mpfd', 'scrfd') and sc >= sure) or (src == 'yunet' and sc >= max(sure, yunet_floor))


def keepout(path, strict=False):
    """a keep-out JSON (host-studio lyric_faces.py: {"frames": [[{"b": [...], "src": ..., "score": ...}]]}, or a list of
    lists of boxes) -> per frame [[x0, y0, x1, y1], ...]: the gating boxes, or every box with strict."""
    z = json.load(open(path))
    fr = z['frames'] if isinstance(z, dict) else z
    return [[(h['b'] if isinstance(h, dict) else h) for h in f if strict or gating(h)] for f in fr]


def overlaps(rect, boxes, pad=0.1):
    x0, y0, x1, y1 = rect
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if x0 < b[2] + bw * pad and x1 > b[0] - bw * pad and y0 < b[3] + bh * pad and y1 > b[1] - bh * pad * 1.6: return True
    return False


# ------------------------------------------------------------------------------------------------------ main --
def iter_frames(path):
    c = cv2.VideoCapture(path)
    while True:
        ok, f = c.read()
        if not ok: break
        yield f
    c.release()


def probe(path):
    """(frame count, fps, (H, W)) by decoding once (a container's frame count can be wrong)."""
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 25.0
    n = 0
    shape = None
    while c.grab():
        if shape is None:
            ok, f = c.retrieve()
            shape = f.shape[:2] if ok else None
        n += 1
    c.release()
    return n, fps, shape


def segments_of(a, n):
    """the EDL -> [(label, [TrackView], [track frame per video frame])] covering the video's n frames."""
    if a.edl:
        E = json.load(open(a.edl))
        out = []
        base = os.path.dirname(os.path.abspath(a.edl))
        for s in E['segments']:
            paths = s.get('tracks') or ([s['track']] if s.get('track') else [])
            paths = [p if os.path.isabs(p) else os.path.join(base, p) for p in paths]
            fr = s.get('frames') or list(range(int(s.get('first', 0)), int(s.get('first', 0)) + int(s['nf'])))
            tvs = [TrackView(p, s.get('vocalisations', a.vocalisations), a.skip_words, a.lyrics, getattr(a, 'align', None), s.get('label', ''),
                             s.get('note', ''), getattr(a, 'vocal_merge_gap', vz.MERGE_GAP)) for p in paths]
            out.append((s.get('label', ''), tvs, list(fr)))
        tot = sum(len(f) for _, _, f in out)
        if tot != n: print(f'annotate: the EDL covers {tot} frames, the video has {n}', file=sys.stderr)
        return out
    tv = TrackView(a.track, a.vocalisations, a.skip_words, a.lyrics, getattr(a, 'align', None), os.path.splitext(os.path.basename(a.track))[0][:24],
                   '', getattr(a, 'vocal_merge_gap', vz.MERGE_GAP))
    return [('', [tv], list(range(n)))]


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub annotate', description=__doc__.split('\n')[0])
    ap.add_argument('--video', required=True)
    ap.add_argument('--track', help='viseme track (one clip; or --edl)')
    ap.add_argument('--edl', help='segments -> tracks (see the module doc)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--corner', choices=['auto', 'lower-right', 'lower-left', 'band'], default='auto', help='auto (= lower-right) = '
                    'lower-right, else lower-left, else a band below the picture; lower-left = that corner first; band = forced')
    ap.add_argument('--scale', type=float, help='card scale (1 = designed for 1280x720; default picture height / 720)')
    ap.add_argument('--lyrics', help='lyric sheet with ( ... ) / [skip] markup (an unmarked track)')
    ap.add_argument('--skip-words')
    ap.add_argument('--vocalisations', choices=['auto', 'skip', 'rest', 'mouth'], default='skip',
                                                     help='what the render did with the vocalisations (as lsdub paint, whose default is skip since v11): '
                                                          'mouth = v9 (mouthed, tagged ad-lib); auto = skip on a marked track, mouth on an unmarked one')
    ap.add_argument('--faces', help='extra keep-out boxes JSON (e.g. host-studio lyric_faces.py heads.json; its gating boxes)')
    ap.add_argument('--faces-strict', action='store_true', help='with --faces: every box, also the weak / person-matte ones')
    ap.add_argument('--face-cache', help='cache of the MediaPipe face boxes (default <out>_faces.json)')
    ap.add_argument('--no-mediapipe', action='store_true', help='no MediaPipe face detection: only --faces boxes')
    ap.add_argument('--face-pad', type=float, default=0.1, help='face box padding (x its size; 1.6x above), as the studio face gate')
    ap.add_argument('--overlay-out', help='also write the card layer alone (RGBA .mov, the output size) for an independent face check')
    ap.add_argument('--cards', help="the face's detailed mouth-shape cards dir (lsdub_engine.core.mouth_cards; <kit>/cards) -- draw the face's own shape instead of the generic glyph")
    ap.add_argument('--log', help='placement per clip (JSON)')
    ap.add_argument('--crf', type=int, default=16)
    ap.add_argument('--align', help='v12: the word/phone alignment JSON of tracks without a words block whose "align" path is '
                    'relative to another directory (their weak marks / heard: diffs come from it)')
    ap.add_argument('--lyric-check', help='v12: write the lyric check here (JSON; the .md next to it): every heard / lyric '
                    'mismatch, weak word and vocalisation span of the annotated frames')
    ap.add_argument('--lyrics-suggested', help='v12: write the lyric sheet with each mismatched word replaced by [heard: x] here '
                    '(suggestions only)')
    ap.add_argument('--lyric-sheet', help='v12: the lyric sheet for --lyrics-suggested (default: --lyrics, else the track\'s '
                    'words_meta / its alignment\'s lyrics path)')
    ap.add_argument('--vocal-merge-gap', type=float, default=vz.MERGE_GAP, help='v12: as lsdub paint --vocal-merge-gap (s): '
                    'vocalisation spans merge across a silence this short with no lexical word')
    a = ap.parse_args(argv)
    if a.corner == 'auto': a.corner = 'lower-right'                         # the existing rule (lower-right, lower-left, band)
    if not a.track and not a.edl: ap.error('--track or --edl')
    n, fps, shp = probe(a.video)
    if not n or shp is None: ap.error(f'no frames in {a.video}')
    segs = segments_of(a, n)
    facts = any(x is not None for _, tvs, _ in segs for tv in tvs for x in tv.info)   # v12: the second line only with word facts
    H, W = shp
    card = Card(a.scale if a.scale else H / 720.0, facts, cards=getattr(a, 'cards', None))
    cw, ch = card.W, card.H
    m = int(round(MARGIN * card.s))
    boxes = [[] for _ in range(n)]
    if a.corner != 'band':
        if not a.no_mediapipe:
            mpb = face_boxes(iter_frames(a.video), n, a.face_cache or os.path.splitext(a.out)[0] + '_faces.json', a.video)
            for i in range(n): boxes[i] += mpb[i]
        if a.faces:
            ko = keepout(a.faces, a.faces_strict)
            for i in range(min(n, len(ko))): boxes[i] += ko[i]
    rects = {'lower-right': (W - m - cw, H - m - ch, W - m, H - m), 'lower-left': (m, H - m - ch, m + cw, H - m)}
    order = {'lower-right': ['lower-right', 'lower-left', 'band'], 'lower-left': ['lower-left', 'lower-right', 'band'], 'band': ['band']}[a.corner]
    place, f0, log = [], 0, []
    for label, tvs, fr in segs:
        idx = range(f0, f0 + len(fr))
        stack = max(1, len(tvs))
        blocked = {}
        for c in ('lower-right', 'lower-left'):
            x0, y0, x1, y1 = rects[c]
            r = (x0, y1 - stack * ch - (stack - 1) * m // 2, x1, y1)
            blocked[c] = [i for i in idx if overlaps(r, boxes[i], a.face_pad)]
        pl = next(c for c in order if c == 'band' or not blocked[c]) if tvs else 'none'
        place.append(pl)
        log.append(dict(label=label, frames=[f0, f0 + len(fr) - 1], placement=pl, blocked_lower_right=len(blocked['lower-right']),
                                          blocked_lower_left=len(blocked['lower-left']), faces=int(sum(len(boxes[i]) > 0 for i in idx)),
                                          first_blocked_lower_right=blocked['lower-right'][:3]))
        f0 += len(fr)
    band = any(p == 'band' for p in place)
    bh = (stack_max := max([len(t) for _, t, _ in segs] + [1])) * ch + (stack_max + 1) * m if band else 0
    OH = H + bh
    if OH % 2: OH += 1                                                      # libx264 yuv420p needs an even height (a band can make OH odd)
    for L in log: print(f"annotate {L['label'] or 'clip'} frames {L['frames'][0]}-{L['frames'][1]}: {L['placement']} "
                        f"(face-overlap frames: lower-right {L['blocked_lower_right']}, lower-left {L['blocked_lower_left']}; frames with a face {L['faces']})")
    if a.log: json.dump(dict(video=a.video, size=[W, H], out_size=[W, OH], card=[cw, ch], scale=card.s, band=band, clips=log), open(a.log, 'w'), indent=1)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{OH}', '-r', str(fps), '-i', '-', '-i', a.video,
           '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-crf', str(a.crf), '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
           '-movflags', '+faststart', a.out]
    pr = ff.popen(cmd, stdin=ff.PIPE)
    po = None
    if a.overlay_out:
        po = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{OH}', '-r', str(fps), '-i', '-',
                               '-c:v', 'qtrle', a.overlay_out], stdin=ff.PIPE)
    i = 0
    src = iter_frames(a.video)
    for (label, tvs, fr), pl in zip(segs, place):
        for k in fr:
            if i >= n: break
            out = np.zeros((OH, W, 3), np.uint8)
            out[:H] = next(src)
            lay = np.zeros((OH, W, 4), np.uint8) if po else None
            for j, tv in enumerate(tvs):
                cimg, cal = card.render(tv, k)
                cimg = cimg[:, :, ::-1]
                if pl == 'band': x0, y0 = W - m - cw, H + m + j * (ch + m)
                else:
                    x0 = rects[pl][0]
                    y0 = H - m - ch - j * (ch + m // 2)
                if pl == 'none': continue
                reg = out[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
                A = cal.astype(np.float32)[..., None] / 255
                out[y0:y0 + ch, x0:x0 + cw] = np.clip(cimg.astype(np.float32) * A + reg * (1 - A) + 0.5, 0, 255).astype(np.uint8)
                if lay is not None:
                    lay[y0:y0 + ch, x0:x0 + cw, :3] = cimg[:, :, ::-1]
                    lay[y0:y0 + ch, x0:x0 + cw, 3] = cal
            pr.stdin.write(out.tobytes())
            if po: po.stdin.write(lay.tobytes())
            i += 1
    while i < n:                                                            # (frames the EDL does not cover: no card)
        out = np.zeros((OH, W, 3), np.uint8)
        out[:H] = next(src)
        pr.stdin.write(out.tobytes())
        if po: po.stdin.write(np.zeros((OH, W, 4), np.uint8).tobytes())
        i += 1
    pr.stdin.close()
    pr.wait()
    if po:
        po.stdin.close()
        po.wait()
    print(f'annotate: {n} frames, card {cw}x{ch} (scale {card.s:.2f}), ' + (f'band {bh} px below the picture' if band else 'no band') + f' -> {a.out}')
    if a.lyric_check or a.lyrics_suggested: lyric_outputs(a, segs, n)
    return 0


def lyric_outputs(a, segs, n):
    """v12: the lyric check of the annotated frames (--lyric-check JSON + .md) and the suggested lyric sheet."""
    from ..core import lyric_check as lc
    S, f0, meta = [], 0, {}
    for label, tvs, fr in segs:
        for tv in tvs:
            S.append(dict(label=label or tv.label, offset=f0, frames=fr, wi=tv.wi, recs=tv.recs, vocal=list(tv.vocal), fps=tv.fps,
                          spans=(tv.found or {}).get('spans') or [], mode=tv.mode))
            for k, v in tv.meta.items(): meta.setdefault(k, v)
        f0 += len(fr)
    C = lc.build_check(S, dict(what='lsdub lyric check (v12): what the aligner heard vs the lyric sheet, per word on screen',
                               video=os.path.abspath(a.video), annotated=os.path.abspath(a.out), align=meta.get('align'), lyrics=None))
    sheet = a.lyric_sheet or a.lyrics or meta.get('lyrics')
    C['lyrics'] = sheet and os.path.abspath(sheet)
    if a.lyric_check:
        j, m = lc.write_check(C, a.lyric_check)
        print(f"lyric check: {C['counts']['mismatches']} mismatch(es), {C['counts']['weak']} weak word(s), "
              f"{C['counts']['vocalisation_spans']} vocalisation span(s) -> {j}, {m}")
    if a.lyrics_suggested:
        A = None
        if meta.get('align') and os.path.exists(meta['align']): A = json.load(open(meta['align']))
        if not sheet or not os.path.exists(sheet) or A is None:
            print(f'lyrics suggested: no lyric sheet / alignment found ({sheet}, {meta.get("align")}): not written', file=sys.stderr)
            return C
        txt, placed = lc.suggest_lyrics(open(sheet, encoding='utf-8').read(), A, C['mismatches'])
        open(a.lyrics_suggested, 'w', encoding='utf-8').write(txt)
        print(f'lyrics suggested: {sum(p for _, p in placed)} of {len(placed)} placed -> {a.lyrics_suggested}')
    return C


if __name__ == '__main__':
    sys.exit(main())
