#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""vocalisations.py — the non-lexical vocalisations / ad-libs of a song ("uh uh uh", "ah ah ah", "da da da", "mm",
"hey", "whoa") found in a viseme track, marked, and handled by the warp: skip them (the take's own mouth), drive them to
rest, or mouth them (v9).

  lsdub track skip --track vis.json --out vis_skip.json [--skip-words words.txt] [--lyrics lyrics.txt] [--align a.json]
  lsdub track scan --align align.json [--lyrics lyrics.txt] [--skip-words words.txt] [--shots shots.json]
  lsdub paint --mode warp ... [--vocalisations auto|skip|rest|mouth] [--skip-words f] [--skip-lyrics f] [--vocal-ease 3]

WHAT COUNTS. Matching is on the aligned WORDS (whole tokens, after lower-casing and stripping punctuation), never on
parts of words: 'da' is not 'dance' or 'day', 'oh' is not 'ohio'.
  - the LEXICON below; elongated spellings count (aaah, ooooh, mmmm, heyyy, daaa), and so does one token that is a
    lexicon syllable repeated (lalala, hahaha, dadada, uhuh; not the real words in REAL_REPEATS: nana, dada) or a
    hyphenated run whose every part is a vocalisation (uh-huh, uh-oh, la-la-la, ooh-ooh). 'yeah' (CONDITIONAL) only as
    a bare ad-lib: no lexical word sung within 0.25 s on either side of it.
  - --skip-words FILE: one token per line ('#' comments), same normalisation, added to the lexicon.
  - lyrics MARKUP (--lyrics / --skip-lyrics FILE, the lyric sheet): words inside ( ... ) on a line, or after a [skip]
    tag to the end of its line, are skipped. Section headers ([Verse 1], [Chorus], any other [...]) are not lyrics.
    The sheet is mapped onto the aligned words by sequence alignment (difflib), so a marked word is found where it is
    sung even when the alignment added or renamed words around it.
SPANS. Consecutive vocalisations form a span; two spans are merged across a silence of at most --merge-gap s (0.75)
that holds no lexical word (a breath inside "da da (breath) da" does not hand the mouth back for a few frames).
TRACK FIELDS (lsdub track skip): frames[i].skip = true and frames[i].skip_reason ('lexicon:da', 'skip-words:x',
'markup:paren', 'markup:tag', 'yeah:bare', 'gap') on the skipped frames; top level skips = [{f0, f1, t0, t1, words,
reasons}] (an empty list = processed, nothing found) and skip_rules (what was applied).
THE CONSUMER (lsdub paint --mode warp, mouth_warp.run -> for_warp): the track's own fields when it carries them, else
the same detection at paint time (so tracks written before v10a work). The default is --vocalisations skip (since v11:
the platform skips ad-libs by default, on marked and unmarked tracks alike); --vocalisations mouth is v9 (a v9 command
with --vocalisations mouth added renders v9 bit for bit); auto is v10a's default (skip on a track marked by `lsdub
track skip`, mouth on an unmarked one, the log saying how many spans it mouthed). A track without vocalisations
renders exactly the v9 frames in every mode:
  skip  the span is the TAKE's own frame (LTX already rendered something natural there): no warp, no teeth /
        layer stack, no erase. The warp strength s (1 = v9, 0 = the take) is eased zero-phase at each span edge: a
        centred box filter of --vocal-ease frames (3) over the keep mask, then a C2 smootherstep, so 1 -> 0.79 -> 0.21
        -> 0 across the edge, symmetric about it (no lag, no step). The residual (target - take) is scaled by s before
        the frame is rendered and the finished frame is dissolved toward the take by s (s = 0: the take exactly). The
        v9 gate does not score the fit to the audio on a skipped frame (there is no target) and does not re-make a frame
        that is the take; its other rulers still run.
  rest  the span's visemes become the library's REST mouth: closed lips at ease, the rest band of the 'sil' mouth
        narrowed to the library's own resting opening (lsdub's mouthlib: 0 .. 0.15 of the take's range), written into
        the track as 'sil' mixed toward the closure 'PP' (mouth_warp's band arithmetic gives exactly that band). The
        warp's own zero-phase target smoothing makes the transitions.
  mouth v9: the vocalisations are mouthed like words.
"""
import argparse, copy, difflib, json, os, re, sys, tempfile
import numpy as np

LEXICON = frozenset((
    # the founder's list (2026-10-24)
    'uh', 'uh-huh', 'ah', 'aah', 'oh', 'ooh', 'oo', 'mm', 'mmm', 'hmm', 'hey', 'ha', 'hah', 'la', 'na', 'da', 'doo', 'whoa',
    'woo', 'ey', 'eh', 'huh',
    # the same sounds in other spellings / fillers
    'aa', 'ahh', 'ohh', 'um', 'umm', 'hm', 'mhm', 'dah', 'lah', 'woah', 'whoah'))
CONDITIONAL = frozenset(('yeah', 'yeh'))                     # vocalisations only as a bare ad-lib
REAL_REPEATS = frozenset(('nana', 'dada', 'mama', 'papa'))   # real words that look like a repeated syllable
BARE_GAP = 0.25                                              # s: 'yeah' with no lexical word this close is an ad-lib
MERGE_GAP = 0.75                                             # s: silence inside a run of vocalisations


# ------------------------------------------------------------------------------------------------- the tokens --
def norm(tok):
    """lower case, curly quotes straightened, everything but letters / apostrophes / hyphens / spaces dropped."""
    t = str(tok or '').lower().replace('’', "'").replace('‘', "'")
    return re.sub(r"[^a-z'\- ]", '', t).strip(" '-")


def _forms(p):
    """the token, its elongations cut to 2 letters (ooooh -> ooh), and to 1 (daaa -> da)."""
    return (p, re.sub(r'(.)\1{2,}', r'\1\1', p), re.sub(r'(.)\1+', r'\1', p))


def _in(p, lex):
    return any(f in lex for f in _forms(p))


def part_class(p, lex=LEXICON, cond=CONDITIONAL):
    """one hyphen-free part: 'v' (a vocalisation), 'c' (conditional: 'yeah'), None (a word)."""
    if not p: return None
    if _in(p, lex): return 'v'
    if _in(p, cond): return 'c'
    for f in _forms(p):
        if f in REAL_REPEATS: return None
        for k in range(1, len(f) // 2 + 1):          # lalala, hahaha, dadada: a lexicon syllable repeated
            if len(f) % k == 0 and len(f) // k >= 2 and f[:k] * (len(f) // k) == f and _in(f[:k], lex): return 'v'
    return None


def token_class(tok, lex=LEXICON, cond=CONDITIONAL):
    """a whole (aligned) word: 'v' when every part (split on hyphens / spaces) is a vocalisation and at least one is
    unconditional, 'c' when every part is conditional ('yeah', 'yeah-yeah'), else None (a word)."""
    parts = [x for x in re.split(r'[-\s]+', norm(tok)) if x]
    if not parts: return None
    cl = [part_class(p, lex, cond) for p in parts]
    if any(c is None for c in cl): return None
    return 'v' if 'v' in cl else 'c'


def read_words(path):
    """--skip-words: one token per line, '#' comments; normalised."""
    out = set()
    for line in open(path, encoding='utf-8'):
        t = norm(line.split('#', 1)[0])
        if t: out.add(t)
    return frozenset(out)


# ----------------------------------------------------------------------------------------------- lyrics markup --
_REPEAT = re.compile(r'^\s*x?\s*\d+\s*x?\s*$', re.I)


def lyric_marks(text):
    """the lyric sheet -> [(token, flag)]: flag 'paren' inside ( ... ), 'tag' after [skip] to the end of the line, None
    otherwise. Other [...] (section headers) are dropped; '(x2)'-style repeat marks are dropped."""
    out = []
    for line in str(text).splitlines():
        line = re.sub(r'\[(?!skip\])[^\]]*\]', ' ', line, flags=re.I)      # section headers
        tag = False
        depth = 0
        buf = ''
        def flush(buf, flag):
            for w in buf.split():
                t = norm(w)
                if t: out.append((t, flag))
        i = 0
        low = line.lower()
        while i < len(line):
            if low.startswith('[skip]', i):
                flush(buf, 'tag' if tag else ('paren' if depth else None))
                buf = ''
                tag = True
                i += 6
                continue
            ch = line[i]
            if ch == '(':
                flush(buf, 'tag' if tag else ('paren' if depth else None))
                buf = ''
                depth += 1
                j = line.find(')', i)
                if j > i and _REPEAT.match(line[i + 1:j]):
                    depth -= 1
                    i = j + 1
                    continue
            elif ch == ')' and depth:
                flush(buf, 'tag' if tag else 'paren')
                buf = ''
                depth -= 1
            else: buf += ch
            i += 1
        flush(buf, 'tag' if tag else ('paren' if depth else None))
    return out


def markup_flags(words, marks):
    """per word of `words` (normalised aligned words, in order) the markup flag of the lyric token it maps to (difflib
    on the two sequences): equal blocks map 1:1; a replaced stretch takes the flag when all its lyric tokens share it;
    a word the sheet does not have (inserted) takes it when the sheet tokens on both sides carry it."""
    a = [t for t, _ in marks]
    fl = [f for _, f in marks]
    out = [None] * len(words)
    sm = difflib.SequenceMatcher(None, a, list(words), autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1): out[j1 + k] = fl[i1 + k]
        elif op == 'replace':
            f = set(fl[i1:i2])
            if len(f) == 1 and None not in f:
                for j in range(j1, j2): out[j] = fl[i1]
        elif op == 'insert':
            if 0 < i1 < len(a) and fl[i1 - 1] and fl[i1 - 1] == fl[i1]:
                for j in range(j1, j2): out[j] = fl[i1]
    return out


# --------------------------------------------------------------------------------------------- classification --
def classify(occ, extra=frozenset(), marks=None, all_words=None):
    """occ: word occurrences in time order (dicts with w, t0, t1, optional ai = index into all_words). Sets
    occ['vocal'] (bool) and occ['reason']. marks: lyric_marks() of the sheet; all_words: the full aligned word list the
    sheet is mapped onto (default: the occurrences themselves)."""
    extra = frozenset(extra or ())
    mf = None
    if marks:
        if all_words is not None:
            fw = markup_flags([norm(w) for w in all_words], marks)
            mf = [fw[o['ai']] if o.get('ai') is not None else None for o in occ]
        else:
            mf = markup_flags([norm(o['w']) for o in occ], marks)
    for k, o in enumerate(occ):
        t = norm(o['w'])
        o['vocal'] = False
        o['reason'] = None
        if mf is not None and mf[k]:
            o['vocal'], o['reason'] = True, f'markup:{mf[k]}'
            continue
        c = token_class(t, LEXICON) if t else None
        if c == 'v':
            o['vocal'], o['reason'] = True, f'lexicon:{t}'
            continue
        if extra and t and token_class(t, LEXICON | extra) == 'v':
            o['vocal'], o['reason'] = True, f'skip-words:{t}'
            continue
        o['cls'] = c
    for k, o in enumerate(occ):                               # 'yeah': only when bare (no lexical word close by)
        if o.get('cls') != 'c': continue
        def ok(j, gap):
            if j < 0 or j >= len(occ): return True
            q = occ[j]
            return q['vocal'] or q.get('cls') == 'c' or gap >= BARE_GAP
        if ok(k - 1, o['t0'] - occ[k - 1]['t1'] if k > 0 else 9) and ok(k + 1, occ[k + 1]['t0'] - o['t1'] if k + 1 < len(occ) else 9):
            o['vocal'], o['reason'] = True, 'yeah:bare'
    return occ


def group(occ, merge_gap=MERGE_GAP, key=('t0', 't1')):
    """consecutive vocal occurrences -> spans (lists of occurrence indices), merged across a silence <= merge_gap (s)
    with no lexical word in it."""
    spans = []
    prev_vocal = False
    for k, o in enumerate(occ):
        if not o['vocal']:
            prev_vocal = False
            continue
        if prev_vocal and spans and o[key[0]] - occ[spans[-1][-1]][key[1]] <= merge_gap + 1e-9: spans[-1].append(k)
        else: spans.append([k])
        prev_vocal = True
    return spans


# ------------------------------------------------------------------------------------------------- alignment --
def load_align(T, align=None, track_path=None):
    """the word/phone alignment a track was built from: `align` (a path or a dict), else the track's 'align' path as
    given (cwd), else relative to the track file's directory and its parents. None when it cannot be found."""
    if isinstance(align, dict): return align
    cands = [align] if align else []
    p = T.get('align') if isinstance(T, dict) else None
    if p:
        cands.append(p)
        if track_path and not os.path.isabs(p):
            d = os.path.dirname(os.path.abspath(track_path))
            while True:
                cands.append(os.path.join(d, p))
                nd = os.path.dirname(d)
                if nd == d: break
                d = nd
    for c in cands:
        if c and os.path.exists(c):
            try:
                A = json.load(open(c))
                if isinstance(A, dict) and isinstance(A.get('words'), list):
                    A['_path'] = c
                    return A
            except Exception:
                continue
    return None


def _frame_word_index(T, A):
    """per track frame the index of the aligned word sounding at its centre (as viseme_track.build assigns words), or
    -1; None when the alignment does not agree with the track's own word field."""
    fr = T['frames']
    n = len(fr)
    fps = float(T.get('fps', 25))
    t0 = float(T.get('t0', fr[0].get('t', 0.0) if fr else 0.0))
    tt = t0 + (np.arange(n) + 0.5) / fps
    ai = np.full(n, -1)
    for j, w in enumerate(A['words']):
        for p in w.get('phones') or [dict(start=w['start'], end=w['end'])]:
            ai[(tt >= p['start']) & (tt < p['end'])] = j
    words = [norm(f.get('word', '')) for f in fr]
    has = [k for k in range(n) if words[k]]
    if not has: return ai
    agree = sum(ai[k] >= 0 and norm(A['words'][ai[k]].get('w', '')) == words[k] for k in has)
    return ai if agree >= 0.9 * len(has) else None


def track_occurrences(T, A=None):
    """the track's word occurrences in frame order: {w, f0, f1, t0, t1, ai (aligned word index or None), src}."""
    fr = T['frames']
    n = len(fr)
    fps = float(T.get('fps', 25))
    t0 = float(T.get('t0', fr[0].get('t', 0.0) if fr else 0.0))
    ai = _frame_word_index(T, A) if A is not None else None
    key = [(norm(f.get('word', '')), int(ai[k]) if ai is not None and norm(f.get('word', '')) else -1) for k, f in enumerate(fr)]
    occ = []
    k = 0
    while k < n:
        j = k
        while j + 1 < n and key[j + 1] == key[k]: j += 1
        w, a = key[k]
        if w:
            o = dict(w=fr[k].get('word', ''), f0=k, f1=j, t0=t0 + k / fps, t1=t0 + (j + 1) / fps, ai=a if a >= 0 else None)
            if o['ai'] is not None:
                aw = A['words'][o['ai']]
                o.update(t0=float(aw['start']), t1=float(aw['end']), src=aw.get('src'))
            occ.append(o)
        k = j + 1
    return occ


# ------------------------------------------------------------------------------------------------ the track --
def find_spans(T, align=None, skip_words=(), lyrics=None, merge_gap=MERGE_GAP, track_path=None):
    """-> dict(spans=[{f0, f1, t0, t1, words, reasons}], frames={frame: reason}, align=path or None, occ=occurrences)."""
    fr = T.get('frames') or []
    fps = float(T.get('fps', 25))
    n = len(fr)
    A = load_align(T, align, track_path)
    occ = track_occurrences(T, A)
    marks = lyric_marks(open(lyrics, encoding='utf-8').read()) if lyrics else None
    extra = read_words(skip_words) if isinstance(skip_words, str) else frozenset(norm(w) for w in skip_words or ())
    use_all = A is not None and marks is not None and all(o.get('ai') is not None for o in occ)
    classify(occ, extra, marks, [w.get('w', '') for w in A['words']] if use_all else None)
    t0 = float(T.get('t0', fr[0].get('t', 0.0) if fr else 0.0))
    # frame-domain gaps (the silence between two occurrences, in frames / fps)
    for o in occ: o['_a'], o['_b'] = o['f0'] / fps, (o['f1'] + 1) / fps
    spans, frames = [], {}
    for g in group(occ, merge_gap, key=('_a', '_b')):
        f0, f1 = occ[g[0]]['f0'], occ[g[-1]]['f1']
        for k in range(f0, f1 + 1): frames[k] = 'gap'
        for i in g:
            for k in range(occ[i]['f0'], occ[i]['f1'] + 1): frames[k] = occ[i]['reason']
        spans.append(dict(f0=int(f0), f1=int(f1), t0=round(t0 + f0 / fps, 3), t1=round(t0 + (f1 + 1) / fps, 3),
                          words=[occ[i]['w'] for i in g], reasons=sorted({occ[i]['reason'] for i in g})))
    for o in occ:
        o.pop('_a', None)
        o.pop('_b', None)
    return dict(spans=spans, frames=frames, align=None if A is None else A.get('_path', '(given)'), occ=occ, n=n)


def mark_track(T, found, rules=None):
    """write the skip fields into a copy of the track."""
    T = copy.deepcopy(T)
    for k, f in enumerate(T.get('frames') or []):
        f.pop('skip', None)
        f.pop('skip_reason', None)
        if k in found['frames']:
            f['skip'] = True
            f['skip_reason'] = found['frames'][k]
    T['skips'] = found['spans']
    T['skip_rules'] = dict(rules or {}, lexicon=sorted(LEXICON), conditional=sorted(CONDITIONAL),
                                                       align=found.get('align'))
    return T


def from_track(T):
    """the skip fields of a processed track (lsdub track skip) -> the same dict as find_spans."""
    frames = {k: f.get('skip_reason', 'track') for k, f in enumerate(T.get('frames') or []) if f.get('skip')}
    return dict(spans=list(T.get('skips') or []), frames=frames, align=(T.get('skip_rules') or {}).get('align'), occ=None,
                n=len(T.get('frames') or []))


def spans_of(T, **kw):
    """a track's spans: its own fields when it was processed (top-level 'skips'), else detected now."""
    return from_track(T) if 'skips' in T else find_spans(T, **kw)


def skip_mask(found, n):
    m = np.zeros(n, bool)
    for k in found['frames']:
        if 0 <= k < n: m[k] = True
    return m


def strength(mask, ease=3):
    """the warp strength per frame: 1 outside the spans, 0 inside, eased ZERO-PHASE at each edge: a centred box filter
    of `ease` frames over the keep mask (edges held), then a C2 smootherstep (ease 3: 1, 0.79, 0.21, 0 across an edge)."""
    keep = 1.0 - np.asarray(mask, float)
    e = int(ease or 0)
    if e > 1 and len(keep):
        # centred (zero-phase) box of e frames; an even e gets half-weight end taps, so the kernel stays symmetric
        k = np.ones(e if e % 2 else e + 1)
        k[[0, -1]] = 1.0 if e % 2 else 0.5
        k /= k.sum()
        h = len(k) // 2
        keep = np.convolve(np.concatenate([np.repeat(keep[:1], h), keep, np.repeat(keep[-1:], h)]), k, 'valid')
    keep = np.clip(keep, 0.0, 1.0)
    return keep * keep * keep * (keep * (6 * keep - 15) + 10)


def rest_label(lib=None):
    """the (vis, vis2, mix) that gives the library's rest mouth in mouth_warp's band arithmetic: 'sil' mixed toward
    'PP' so the opening band is (0, the library's own rest opening + 0.15). -> (vis, vis2, mix, band_hi)."""
    from . import mouth_warp as mw
    bo = mw.library_bands(lib)[0] if lib and os.path.isdir(lib) else dict(mw.V_OPEN)
    u_rest = 0.0
    if lib and os.path.isdir(lib):
        try:
            L = {v: np.load(os.path.join(lib, f'{v}_lm.npy')) for v in ('sil', 'PP', 'aa')}
            g = {v: float(np.linalg.norm(L[v][13, :2] - L[v][14, :2])) for v in L}
            u_rest = (g['sil'] - g['PP']) / max(1e-6, g['aa'] - g['PP'])
        except Exception:
            u_rest = 0.0
    hi_sil, hi_pp = bo.get('sil', mw.V_OPEN['sil'])[1], bo.get('PP', mw.V_OPEN['PP'])[1]
    hi = float(np.clip(max(0.0, u_rest) + 0.15, hi_pp, hi_sil))
    mix = (hi_sil - hi) / max(1e-6, hi_sil - hi_pp)
    return 'sil', 'PP', round(float(mix), 4), hi


# ---------------------------------------------------------------------------------- the consumer (paint warp) --
class Vocal:
    """--vocalisations skip | rest for one --mode warp render (hooks called by mouth_warp.run)."""

    def __init__(self, a, T, found, mode):
        self.a, self.T, self.found, self.mode = a, T, found, mode
        n = len(T['frames'])
        self.mask = skip_mask(found, n)
        self.args = a
        self._tmp = None
        self.s = strength(self.mask, getattr(a, 'vocal_ease', 3)) if mode == 'skip' else np.ones(n)
        if mode == 'rest':
            v1, v2, mix, hi = rest_label(getattr(a, 'lib', None))
            T2 = copy.deepcopy(T)
            self.rest = dict(vis=v1, vis2=v2, mix=mix, band=[0.0, round(hi, 3)])
            for k in np.nonzero(self.mask)[0]:
                f = T2['frames'][k]
                f.update(vis=v1, vis2=v2, mix=mix, open=0.0)
            fd, self._tmp = tempfile.mkstemp(suffix='.json', prefix='lsdub_rest_')
            os.close(fd)
            json.dump(T2, open(self._tmp, 'w'))
            self.args = copy.copy(a)
            self.args.track = self._tmp

    def describe(self):
        sp = ', '.join(f"f{s['f0']}-{s['f1']} ({s['t0']:.2f}-{s['t1']:.2f} s) {' '.join(s['words'][:6])}{' ...' if len(s['words']) > 6 else ''}"
                       for s in self.found['spans'])
        return f"vocalisations {self.mode}: {len(self.found['spans'])} span(s) [{sp}]"

    def fit(self, n):
        """mask / strength to the clip's frame count (a track shorter than the clip: its last frame holds, as in WarpClip)."""
        def to(x):
            x = np.asarray(x)[:n]
            return np.concatenate([x, np.repeat(x[-1:], n - len(x))]) if 0 < len(x) < n else x
        self.mask, self.s = to(self.mask).astype(bool), to(self.s).astype(float)

    def before_batch(self, clip):
        """skip: the residual (target - take) scaled by the strength, before any frame is rendered."""
        self.clip = clip
        self.fit(clip.n)
        if self.mode != 'skip': return
        s = self.s[:clip.n]
        if (s < 1).any():
            clip.dFe = clip.dFe * s[:, None]
            clip.Fe = clip.F + clip.dFe

    def blend(self, i, out, take):
        s = float(self.s[i])
        if s >= 1.0: return out
        if s <= 0.0: return take.copy()
        return np.clip(take.astype(np.float32) + s * (out.astype(np.float32) - take.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8)

    def _ent(self, i):
        e = dict(vocal_skip=bool(self.mask[i]))
        if self.mode == 'skip': e['vocal_s'] = round(float(self.s[i]), 4)
        if self.mask[i]: e['vocal_reason'] = self.found['frames'].get(int(i))
        return e

    def after_batch(self, clip):
        """skip: every finished frame dissolved toward the take by its strength (0: the take exactly), the masks scaled;
        the per-frame render (the v9 gate's maker) does the same. The gate reads clip.vocal_skip / vocal_s."""
        n = clip.n
        clip.vocal_skip = self.mask[:n].copy()
        clip.vocal_s = self.s[:n].copy()
        clip.vocal_mode = self.mode
        for i in range(n):
            if self.mask[i] or self.s[i] < 1: clip.log[i].update(self._ent(i))
        if self.mode != 'skip': return
        for i in np.nonzero(self.s[:n] < 1)[0]:
            clip.outs[i] = self.blend(i, clip.outs[i], clip.frames[i])
            clip.masks[i] = clip.masks[i] * float(self.s[i])
        orig = clip.render_frame

        def render_frame(i, ov=None, fold=False):
            out, mask, ent, P = orig(i, ov, fold)
            if self.s[i] < 1:
                out = self.blend(i, out, clip.frames[i])
                mask = mask * float(self.s[i])
                ent.update(self._ent(i))
            return out, mask, ent, P
        clip.render_frame = render_frame

    def finish(self, clip, extra):
        """the warp.json block + the per-frame fields again (the gate may have replaced a frame's log entry)."""
        n = clip.n
        for i in range(n):
            if self.mask[i] or self.s[i] < 1: clip.log[i].update(self._ent(i))
        blk = dict(mode=self.mode, spans=self.found['spans'], source='track' if 'skips' in self.T else 'detected at paint',
                   align=self.found.get('align'), frames_take=int((self.s[:n] <= 0).sum()) if self.mode == 'skip' else 0,
                   frames_eased=int(((self.s[:n] > 0) & (self.s[:n] < 1)).sum()) if self.mode == 'skip' else 0,
                   frames_marked=int(self.mask[:n].sum()))
        if self.mode == 'skip': blk['ease'] = int(getattr(self.a, 'vocal_ease', 3))
        if self.mode == 'rest': blk['rest'] = self.rest
        if self._tmp and os.path.exists(self._tmp): os.remove(self._tmp)
        print(f"{self.describe()}; {blk['frames_marked']} frames marked" +
              (f", {blk['frames_take']} are the take's own, {blk['frames_eased']} eased" if self.mode == 'skip' else f", driven to rest (band 0..{self.rest['band'][1]})"))
        out = dict(extra or {})
        out['vocal'] = blk
        return out


def for_warp(a):
    """mouth_warp.run's hook: a Vocal for --vocalisations skip | rest when the track has vocalisations, else None (then
    the render is exactly the v9 one)."""
    mode = getattr(a, 'vocalisations', 'mouth') or 'mouth'
    if mode == 'mouth': return None
    T = json.load(open(a.track))
    if not T.get('frames'): return None
    if mode == 'auto':
        if 'skips' in T: mode = 'skip'
        else:                                    # an unmarked track: v9 behaviour, and say so when it has vocalisations
            try:
                f = find_spans(T, getattr(a, 'vocal_align', None), getattr(a, 'skip_words', None) or (), getattr(a, 'skip_lyrics', None),
                               getattr(a, 'vocal_merge_gap', MERGE_GAP), a.track)
                if f['spans']:
                    where = ', '.join('f%d-%d' % (x['f0'], x['f1']) for x in f['spans'])
                    print(f"vocalisations: {len(f['spans'])} span(s) in this unmarked track ({where}) are MOUTHED (v9). "
                          "--vocalisations skip | rest, or mark the track with `lsdub track skip`, to skip them")
            except Exception:
                pass
            return None
    if 'skips' in T:
        found = from_track(T)
        if getattr(a, 'skip_words', None) or getattr(a, 'skip_lyrics', None):
            print('vocalisations: the track carries its own skip fields (lsdub track skip); --skip-words / --skip-lyrics are not re-applied')
    else:
        found = find_spans(T, getattr(a, 'vocal_align', None), getattr(a, 'skip_words', None) or (), getattr(a, 'skip_lyrics', None),
                           getattr(a, 'vocal_merge_gap', MERGE_GAP), a.track)
    if not found['spans']:
        print(f'vocalisations {mode}: none in the track' + ('' if found.get('align') or 'skips' in T else ' (no alignment found: matched on the track\'s words)'))
        return None
    vx = Vocal(a, T, found, mode)
    print(vx.describe())
    return vx


# ------------------------------------------------------------------------------------------------------- CLI --
def scan_alignment(A, skip_words=(), lyrics=None, merge_gap=MERGE_GAP):
    """the whole song: every vocalisation span in an alignment -> [{t0, t1, words, reasons, idx}]."""
    occ = [dict(w=w.get('w', ''), t0=float(w['start']), t1=float(w['end']), ai=j, src=w.get('src')) for j, w in enumerate(A['words'])]
    occ.sort(key=lambda o: (o['t0'], o['t1']))
    marks = lyric_marks(open(lyrics, encoding='utf-8').read()) if lyrics else None
    extra = read_words(skip_words) if isinstance(skip_words, str) else frozenset(norm(w) for w in skip_words or ())
    classify(occ, extra, marks, [w.get('w', '') for w in A['words']] if marks else None)
    out = []
    for g in group(occ, merge_gap):
        out.append(dict(t0=round(occ[g[0]]['t0'], 3), t1=round(occ[g[-1]]['t1'], 3), words=[occ[i]['w'] for i in g],
                        reasons=sorted({occ[i]['reason'] for i in g}), idx=[occ[i]['ai'] for i in g],
                        src=sorted({str(occ[i].get('src')) for i in g})))
    return out, occ


def _print_spans(spans, fps=None):
    for s in spans:
        fr = f"f{s['f0']}-{s['f1']}  " if 'f0' in s else ''
        w = s['words']
        ws = ' '.join(w) if len(w) <= 12 else ' '.join(w[:12]) + f' ... ({len(w)} words)'
        print(f"  {fr}{s['t0']:7.2f}-{s['t1']:7.2f} s  {ws}   [{', '.join(s['reasons'])}]" + (f"  in {s['shots']}" if s.get('shots') else ''))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub track', description='viseme-track post-processes: skip = mark the non-lexical '
                                 'vocalisations (uh uh uh, ah, da da, mm, hey ...); scan = list them in a whole-song alignment')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s1 = sub.add_parser('skip', help='write skip / skip_reason on the frames of a viseme track + the skips[] span list')
    s1.add_argument('--track', required=True)
    s1.add_argument('--out', required=True)
    s1.add_argument('--skip-words', help='extra tokens to skip, one per line')
    s1.add_argument('--lyrics', help='lyric sheet with ( ... ) / [skip] markup')
    s1.add_argument('--align', help='the alignment JSON the track was built from (default: the track\'s "align" path)')
    s1.add_argument('--merge-gap', type=float, default=MERGE_GAP, help='s; merge spans across a silence this short (no lexical word in it)')
    s2 = sub.add_parser('scan', help='every vocalisation span of an alignment (the whole song), optionally per shot')
    s2.add_argument('--align', required=True)
    s2.add_argument('--skip-words')
    s2.add_argument('--lyrics')
    s2.add_argument('--shots', help='JSON list of {id, t0, t1} (or nf + fps 25): report which shot each span falls in')
    s2.add_argument('--merge-gap', type=float, default=MERGE_GAP)
    s2.add_argument('--json', help='write the spans here')
    a = ap.parse_args(argv)
    if a.cmd == 'skip':
        T = json.load(open(a.track))
        found = find_spans(T, a.align, a.skip_words or (), a.lyrics, a.merge_gap, a.track)
        T2 = mark_track(T, found, dict(skip_words=a.skip_words, lyrics=a.lyrics, merge_gap=a.merge_gap))
        json.dump(T2, open(a.out, 'w'), indent=0)
        print(f"{a.track}: {len(found['spans'])} vocalisation span(s), {len(found['frames'])} of {found['n']} frames"
              + ('' if found['align'] else ' (no alignment found: matched on the track\'s own words)'))
        _print_spans(found['spans'])
        return 0
    A = json.load(open(a.align))
    spans, occ = scan_alignment(A, a.skip_words or (), a.lyrics, a.merge_gap)
    if a.shots:
        sh = json.load(open(a.shots))
        for s in spans:
            hit = []
            for x in sh:
                t1 = x.get('t1', x['t0'] + x.get('nf', 0) / 25.0)
                if s['t0'] < t1 and s['t1'] > x['t0']: hit.append(f"{x['id']} ({max(s['t0'], x['t0']):.2f}-{min(s['t1'], t1):.2f})")
            s['shots'] = ', '.join(hit) if hit else '-'
    nv = sum(o['vocal'] for o in occ)
    print(f"{a.align}: {len(spans)} vocalisation span(s), {nv} of {len(occ)} words")
    _print_spans(spans)
    if a.json: json.dump(spans, open(a.json, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
