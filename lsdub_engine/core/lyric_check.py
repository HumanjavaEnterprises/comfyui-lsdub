#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lyric_check.py — v12: what the aligner already knows about every sung word, for the mouth card and the lyric check.

`lsdub align` (lyric_align.py, read here, never changed) records per aligned word what Whisper HEARD (`heard`, the raw
token), how the word was reconciled with the lyric sheet (`src`: lyric | lyric~ | sung | sung-only), how it was timed
(`align`: mfa | sphinx | whisper+spread), Whisper's word probability (`p_whisper`), its phones, the held-note stretch
(`held_s`, `tail_snap_s`), and per phrase the method and why a phrase fell back (`phrases`). The lyric-sheet word of a
`sung` word lives only in the reconcile notes ("'them' vs lyric 'em' -> 'them'"); `lyric_words` reads it back from them.
`lsdub visemes` persists the per-word record (`word_table`) into the viseme track as its `words` block (+ `wi` per frame),
so the card and the check need nothing but the track.

CONFIDENCE (per word, 0..1, from that evidence only; it never changes the track or the render):
  1.0 minus the penalties below; WEAK = confidence < WEAK_BELOW (0.6).
    0.30  whisper-timed: the word's phrase failed the forced aligner's sanity check and fell back to Whisper's word times
          with the phones spread by rule (align = whisper+spread): the word is timed by Whisper only.
    0.10  phrase fell back: the phrase's first-choice aligner (MFA with --aligner best / mfa) FAILED its sanity check (the
          report gives why) and the backup chain timed the word (not counted again on a whisper-timed word; a phrase
          where MFA passed but the backup chain scored better on the envelope is not a fallback).
    0.50  squeezed: less than SQUEEZE_S (40 ms) per phone on average -- close to the forced aligners' own floor of 30 ms
          per phone (3 HMM states x 10 ms), i.e. the aligner compressed the word to fit; also under one 25 fps frame.
    0.30  stretched: more than STRETCH_S (0.6 s) per phone after the held-note stretch the aligner logged.
    0.5 x (0.6 - p) / 0.6  Whisper's word probability p below 0.6 (no penalty when Whisper gave none).
HEARD vs LYRIC (`same_word`): equal after normalisation -- case, curly quotes, punctuation, leading / trailing apostrophes
and hyphens ("-da" = "da"), numbers spelled out ("400" = "four hundred"; a lyric word inside a heard number counts),
simple stemming (-'s, -s, -es, -ies, -ed, -ing, a g dropped with an apostrophe: "dancin'" = "dancing" = "dance"; a final e
only matters against a suffixed form, so time / tim and hope / hop stay different).
A word whose heard token differs from its lyric word is a MISMATCH ("heard:" on the card); a `sung-only` word (Whisper
heard it, the sheet does not have it) is one too, unless it is a skipped vocalisation (those are listed as spans).

  lsdub annotate ... --lyric-check out_lyric_check.json [--lyrics-suggested lyrics_suggested.txt]
  lsdub paint ... --annotate      (writes <out>_lyric_check.json / .md and <out>_lyrics_suggested.txt)
"""
import ast, difflib, json, os, re

WEAK_BELOW = 0.6
P_SURE = 0.6                  # Whisper probability at / above which the word counts as heard for sure
SQUEEZE_S = 0.040             # s per phone: near the aligners' 30 ms floor (3 HMM states x 10 ms) = squeezed to fit
STRETCH_S = 0.6               # s per phone (held-note stretch removed) = stretched
PEN = dict(whisper_timed=0.30, phrase_fallback=0.10, squeezed=0.50, stretched=0.30, p_scale=0.5)


# ------------------------------------------------------------------------------------------------ the normaliser --
_ONES = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '
         'seventeen eighteen nineteen').split()
_TENS = ('_ _ twenty thirty forty fifty sixty seventy eighty ninety').split()
_ORD = {'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth', 'eight': 'eighth', 'nine': 'ninth', 'twelve': 'twelfth'}


def spell_number(n):
    """a non-negative int in words: 400 -> 'four hundred', 21 -> 'twenty one'."""
    n = int(n)
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ('' if n % 10 == 0 else ' ' + _ONES[n % 10])
    if n < 1000: return _ONES[n // 100] + ' hundred' + ('' if n % 100 == 0 else ' ' + spell_number(n % 100))
    for unit, name in ((10 ** 9, 'billion'), (10 ** 6, 'million'), (1000, 'thousand')):
        if n >= unit: return spell_number(n // unit) + ' ' + name + ('' if n % unit == 0 else ' ' + spell_number(n % unit))


def _ordinal(words):
    last = words[-1]
    last = _ORD.get(last) or (last[:-1] + 'ieth' if last.endswith('y') else last + 'th')
    return words[:-1] + [last]


def tokens(s):
    """a word / token as heard or written -> its normalised parts: lower case, curly quotes straightened, split on
    spaces / hyphens / slashes, punctuation dropped, leading / trailing apostrophes stripped, numbers spelled out;
    "dancin'" keeps a mark of its dropped g (-> 'dancing')."""
    s = str(s or '').lower().replace('’', "'").replace('‘', "'")
    out = []
    for part in re.split(r"[\s\-/]+", s):
        g = bool(re.search(r"[a-z]in'$", re.sub(r"[^a-z0-9']", '', part)))
        p = re.sub(r"[^a-z0-9']", '', part).strip("'")
        if not p: continue
        m = re.fullmatch(r'(\d+)(st|nd|rd|th|s)?', p)
        if m and len(m.group(1)) <= 12:
            w = spell_number(int(m.group(1))).split()
            out += _ordinal(w) if m.group(2) in ('st', 'nd', 'rd', 'th') else w
            continue
        out.append(p + 'g' if g else p)
    return out


def _base(w):
    """(the word with apostrophes dropped and one of -ies / -es / -s, -ing, -ed stripped (a doubled final consonant
    undone after -ing / -ed), whether a suffix was stripped)."""
    w = w.replace("'", '')
    w0 = w
    if len(w) > 4 and w.endswith('ies'): w = w[:-3] + 'y'
    elif len(w) > 4 and re.search(r'(ch|sh|x|ss|z)es$', w): w = w[:-2]
    elif len(w) > 3 and w.endswith('s') and not w.endswith('ss'): w = w[:-1]
    if len(w) > 5 and w.endswith('ing'): w = w[:-3]
    elif len(w) > 4 and w.endswith('ed'): w = w[:-2]
    if w != w0 and len(w) > 3 and w[-1] == w[-2] and w[-1] not in 'aeiouls': w = w[:-1]
    return w, w != w0


def _drop_e(w):
    return w[:-1] if len(w) > 3 and w.endswith('e') else w


def stem(w):
    """simple stemming (the comparison form): _base, then a final e dropped: dance / dances / danced / dancing / dancin'
    -> danc. same_word drops the e only when one side carried a suffix (time / tim, hope / hop stay different)."""
    return _drop_e(_base(w)[0])


def _same_tok(a, b):
    (x, sx), (y, sy) = _base(a), _base(b)
    return x == y or ((sx or sy) and _drop_e(x) == _drop_e(y))


def norm_word(s):
    """the comparison form: the stems of the tokens, space-joined ('' = nothing left)."""
    return ' '.join(stem(t) for t in tokens(s))


def same_word(lyric, heard):
    """does the heard token say the lyric word? (after normalisation; a lyric word inside a heard number counts: the
    aligner splits '400' into 'four' + 'hundred', both heard as '400')."""
    L, H = tokens(lyric), tokens(heard)
    if not L or not H: return L == H
    if len(L) == len(H) and all(_same_tok(x, y) for x, y in zip(L, H)): return True
    return len(L) == 1 and len(H) > 1 and any(_same_tok(L[0], y) for y in H)


# --------------------------------------------------------------------------------- reading lyric_align's output --
def _lit(s):
    s = s.strip()
    return None if s == 'None' else ast.literal_eval(s)


def lyric_words(A):
    """{aligned word index: the lyric-sheet word it was reconciled with (None = not in the sheet)}. src lyric / lyric~:
    the word itself (lyric spelling); sung: read back from lyric_align.reconcile's notes ("'them' vs lyric 'em' ->
    'them'", one per replaced word, in Whisper order = the words' idx order); sung-only: None. When the notes do not line
    up with the words (an older alignment), sung words get None and 'lyric_from_notes' is False."""
    W = A.get('words') or []
    out = {}
    notes = []
    for s in A.get('reconcile_notes') or []:
        if ' vs lyric ' in s and ' -> ' in s:
            a, rest = s.split(' vs lyric ', 1)
            b, c = rest.rsplit(' -> ', 1)
            try: notes.append((_lit(a), _lit(b), _lit(c)))
            except (ValueError, SyntaxError): notes.append(None)
    order = sorted(range(len(W)), key=lambda i: (W[i].get('idx', i), i))
    repl = [i for i in order if W[i].get('src') in ('lyric~', 'sung')]
    ok = len(repl) == len(notes) and all(n is not None and n[0] == W[i].get('heard') and n[2] == W[i].get('w')
                                         for n, i in zip(notes, repl))
    for i, w in enumerate(W):
        src = w.get('src')
        out[i] = w.get('w') if src in ('lyric', 'lyric~', None) else None
    if ok:
        for n, i in zip(notes, repl): out[i] = n[1] if W[i].get('src') == 'sung' else W[i].get('w')
    return out, ok


def phrase_info(A):
    """{aligned word index: {phrase, why, chain, chain_why}} from the alignment's phrase reports: `phrase` / `why` = the
    first-choice aligner's phrase method and its failure reason (--aligner best / mfa: the report whose seg holds the
    word; sphinx: the chain's own report); `chain` / `chain_why` = the sphinx chain's (sub)phrase that timed the word
    (matched by text, in order: the chain reports its phrases in word order, split-retries skipped)."""
    W = A.get('words') or []
    out = {i: {} for i in range(len(W))}
    reps = [r for r in A.get('phrases') or [] if isinstance(r, dict)]
    segs = [r for r in reps if 'seg' in r]
    chain = next((r['sphinx_chain_report'] for r in reps if 'sphinx_chain_report' in r), None)
    if chain is None: chain = [r for r in reps if 'seg' not in r and 'phrase' in r]
    for i, w in enumerate(W):
        mid = (float(w['start']) + float(w['end'])) / 2
        r = next((r for r in segs if r['seg'][0] - 1e-6 <= mid <= r['seg'][1] + 1e-6), None)
        if r: out[i].update(phrase=r.get('method'), why=r.get('mfa_why') or r.get('why'))
    order = sorted(range(len(W)), key=lambda i: (W[i].get('idx', i), i))
    k = 0
    for r in chain:
        if r.get('method') not in ('sphinx', 'fallback') or not r.get('phrase'): continue
        ws = r['phrase'].split()
        s = next((s for s in range(k, len(order) - len(ws) + 1) if [W[j].get('w') for j in order[s:s + len(ws)]] == ws), None)
        if s is None: continue
        for j in order[s:s + len(ws)]:
            out[j].update(chain=r['method'], chain_why=r.get('why'))
            if not segs: out[j].update(phrase=r['method'], why=r.get('why'))
        k = s + len(ws)
    return out


def word_table(A, t0=None, t1=None, sheet=None):
    """the per-word records of alignment A (lyric_align's JSON as a dict), the words overlapping [t0, t1] (song time;
    None = all): {i (index in A['words']), w, start, end, src, heard, lyric, p, align, ph, held, tail, phrase, why} +
    lyric_text (the lyric word as the sheet writes it, "'em") when the sheet's text is given."""
    W = A.get('words') or []
    lyr, _ = lyric_words(A)
    phi = phrase_info(A)
    out = []
    sm = sheet_map(sheet, A) if sheet else {}
    for i, w in enumerate(W):
        if t0 is not None and float(w['end']) <= t0: continue
        if t1 is not None and float(w['start']) >= t1: continue
        pi = phi.get(i) or {}
        why = (pi.get('chain_why') or pi.get('why')) if w.get('align') == 'whisper+spread' else pi.get('why')
        r = dict(i=i, w=w.get('w'), start=w.get('start'), end=w.get('end'), src=w.get('src'), heard=w.get('heard'), lyric=lyr.get(i),
                 p=w.get('p_whisper'), align=w.get('align'), ph=' '.join(p['p'] for p in w.get('phones') or []),
                 held=w.get('held_s'), tail=w.get('tail_snap_s'), phrase=pi.get('phrase'), why=why)
        if i in sm: r['lyric_text'] = sm[i][1]
        out.append({k: v for k, v in r.items() if v is not None or k in ('i', 'w', 'start', 'end', 'lyric', 'heard')})
    return out


# -------------------------------------------------------------------------------------------------- the rules --
def confidence(r, raw=False):
    """(confidence 0..1, [reasons, the heaviest penalty first]) of one word record (word_table), by the rule in the
    module doc; rounded to 2 decimals unless raw (weak is decided on the raw value)."""
    pen = []
    wt = r.get('align') == 'whisper+spread'
    if wt: pen.append((PEN['whisper_timed'], 'whisper-timed' + (f" ({r['why']})" if r.get('why') else '')))
    ph = str(r.get('phrase') or '')
    if not wt and ph and ph not in ('mfa', 'sphinx') and r.get('why'):    # a FAILED first choice (not a better-scoring alt)
        pen.append((PEN['phrase_fallback'], f'phrase fell back to {ph}' + (f" ({r['why']})" if r.get('why') else '')))
    n = len(str(r.get('ph') or '').split())
    if n:
        dur = float(r['end']) - float(r['start'])
        core = dur - float(r.get('held') or 0) - float(r.get('tail') or 0)
        if dur / n < SQUEEZE_S: pen.append((PEN['squeezed'], f'squeezed: {n} phone{"s" * (n > 1)} in {dur:.2f} s'))
        elif core / n > STRETCH_S: pen.append((PEN['stretched'], f'stretched: {n} phone{"s" * (n > 1)} in {core:.2f} s'))
    p = r.get('p')
    if p is not None and float(p) < P_SURE: pen.append((PEN['p_scale'] * (P_SURE - float(p)) / P_SURE, f'whisper p {float(p):.2f}'))
    c = max(0.0, min(1.0, 1.0 - sum(x for x, _ in pen)))
    return (c if raw else round(c, 2)), [t for _, t in sorted(pen, key=lambda q: -q[0])]


def short_reason(reasons):
    """the card's one-line form of the first reason."""
    if not reasons: return ''
    r = reasons[0]
    if r.startswith('whisper-timed'): return 'whisper-timed'
    if r.startswith('phrase fell back'): return 'phrase fell back'
    if r.startswith(('squeezed', 'stretched')): return r.split(': ', 1)[1].replace(' phones', ' ph').replace(' phone', ' ph')
    return r


def word_state(r):
    """everything the card and the check need about one word record: conf, weak, reasons, diff (heard differs from the
    lyric word), heard, lyric, kind (lyric~ / sung / not-in-lyric)."""
    c, why = confidence(r, raw=True)
    heard, lyric, src = r.get('heard'), r.get('lyric'), r.get('src')
    diff = False
    kind = None
    if heard not in (None, ''):
        if lyric not in (None, ''):
            diff = not same_word(lyric, heard)
            kind = src or 'lyric'
        elif src in ('sung', 'sung-only'):
            diff = True
            kind = 'not-in-lyric'
    return dict(conf=round(c, 2), weak=c < WEAK_BELOW, reasons=why, diff=diff, kind=kind if diff else None, heard=heard, lyric=lyric,
                lyric_text=r.get('lyric_text') or lyric, word=r.get('w'))


# ------------------------------------------------------------------------------------------------- the check --
def build_check(segs, meta=None):
    """the lyric check of a render. segs: [dict(label, offset (first video frame of the segment), frames ([track frame
    per video frame of the segment]), wi ([aligned word index per track frame, -1 none]), recs ({i: word record}),
    vocal ([bool per track frame]: a skipped / rested vocalisation), spans (vocalisations.find_spans spans), mode
    ('skip' | 'rest' | 'mouth' | 'ad-lib'), fps)]. Only what is ON SCREEN is listed: a word with at least one video frame
    (once, by aligned index; a word cut between two segments gets both labels, 'S2+S3', and the range over both; it is a
    vocalisation when every one of its frames is a skipped / rested one), a span with at least one video frame."""
    acc, recs, spans, fps0 = {}, {}, [], 25.0
    for s in segs:
        fps = float(s.get('fps', 25))
        fps0 = fps
        off = int(s.get('offset', 0))
        fr = list(s['frames'])
        shown_vocal = s.get('mode') in ('skip', 'rest')
        for v, k in enumerate(fr):
            if not (0 <= k < len(s['wi'])) or s['wi'][k] is None or s['wi'][k] < 0: continue
            i = int(s['wi'][k])
            if i not in s['recs']: continue
            x = acc.setdefault(i, dict(segs=[], frames=[], vocal=[]))
            if s.get('label', '') not in x['segs']: x['segs'].append(s.get('label', ''))
            x['frames'].append(off + v)
            x['vocal'].append(bool(shown_vocal and 0 <= k < len(s['vocal']) and s['vocal'][k]))
            recs.setdefault(i, s['recs'][i])
        for sp in s.get('spans') or []:
            vf = [v for v, k in enumerate(fr) if sp['f0'] <= k <= sp['f1']]
            if not vf: continue
            spans.append(dict(seg=s.get('label', ''), frames=[off + vf[0], off + vf[-1]], t=sp.get('t0'), t_end=sp.get('t1'),
                              cut_t=round((off + vf[0]) / fps, 3), words=list(sp.get('words') or []),
                              reasons=list(sp.get('reasons') or []), mode=s.get('mode')))
    words = []
    for i in sorted(acc, key=lambda i: (float(recs[i]['start']), i)):
        r, x = recs[i], acc[i]
        st = word_state(r)
        f0, f1 = min(x['frames']), max(x['frames'])
        words.append(dict(seg='+'.join(x['segs']), i=i, word=r.get('w'), t=r.get('start'), t_end=r.get('end'), frames=[f0, f1],
                          cut_t=round(f0 / fps0, 3), lyric=st['lyric_text'], heard=st['heard'], src=r.get('src'), align=r.get('align'),
                          p_whisper=r.get('p'), confidence=st['conf'], weak=st['weak'], reasons=st['reasons'], mismatch=st['diff'],
                          kind=st['kind'], vocalisation=all(x['vocal'])))
    mism = [w for w in words if w['mismatch'] and not w['vocalisation']]
    weak = [w for w in words if w['weak'] and not w['vocalisation']]
    rule = dict(weak_below=WEAK_BELOW, penalties=dict(PEN), p_sure=P_SURE, squeeze_s=SQUEEZE_S, stretch_s=STRETCH_S)
    return dict(dict(meta or {}), rule=rule,
                counts=dict(words=len(words), mismatches=len(mism), weak=len(weak), vocalisation_spans=len(spans),
                            whisper_timed=sum(w['align'] == 'whisper+spread' for w in words)),
                mismatches=mism, weak=weak, vocalisation_spans=spans, words=words)


def _fr(x):
    return '-' if not x else (f'f{x[0]}' if x[0] == x[1] else f'f{x[0]}-{x[1]}')


def check_md(C):
    """the lyric check as Markdown."""
    L = [f"# Lyric check{(' -- ' + os.path.basename(C['video'])) if C.get('video') else ''}", '']
    if C.get('align'): L.append(f"Alignment: `{C['align']}`" + (f"; lyric sheet `{C['lyrics']}`" if C.get('lyrics') else ''))
    n = C['counts']
    L += [f"{n['words']} words in the render: **{n['mismatches']} heard / lyric mismatches**, **{n['weak']} weak words** "
          f"(confidence < {C['rule']['weak_below']}), {n['vocalisation_spans']} vocalisation span(s), {n['whisper_timed']} "
          f"whisper-timed words. Frames are the render's (video) frames; t = song time.", '',
          '## Heard / lyric mismatches', '']
    if C['mismatches']:
        L += ['| shot | frames | t (s) | lyric word | heard | word on the card | kind | confidence | weak |', '|---|---|---|---|---|---|---|---|---|']
        for w in C['mismatches']:
            L.append(f"| {w['seg']} | {_fr(w['frames'])} | {w['t']:.2f}-{w['t_end']:.2f} | {w['lyric'] if w['lyric'] else '(not in the sheet)'} | "
                     f"{w['heard']} | {w['word']} | {w['kind']} | {w['confidence']:.2f} | {'yes' if w['weak'] else 'no'} |")
    else: L.append('None.')
    L += ['', '## Weak words', '']
    if C['weak']:
        L += ['| shot | frames | t (s) | word | confidence | why | timing |', '|---|---|---|---|---|---|---|']
        for w in C['weak']:
            L.append(f"| {w['seg']} | {_fr(w['frames'])} | {w['t']:.2f}-{w['t_end']:.2f} | {w['word']} | {w['confidence']:.2f} | "
                     f"{'; '.join(w['reasons'])} | {w['align']} |")
    else: L.append('None.')
    L += ['', '## Vocalisation spans (skipped / rested / mouthed as the render did)', '']
    if C['vocalisation_spans']:
        L += ['| shot | frames | t (s) | words | why | mode |', '|---|---|---|---|---|---|']
        for s in C['vocalisation_spans']:
            ws = ' '.join(s['words']) if len(s['words']) <= 12 else ' '.join(s['words'][:12]) + ' ...'
            L.append(f"| {s['seg']} | {_fr(s['frames'])} | {s['t']:.2f}-{s['t_end']:.2f} | {ws} | {', '.join(s['reasons'])} | {s['mode']} |")
    else: L.append('None.')
    r = C['rule']
    p = r['penalties']
    L += ['', '## The rule', '',
          f"confidence = 1 - {p['whisper_timed']} whisper-timed - {p['phrase_fallback']} phrase fell back (not on a whisper-timed "
          f"word) - {p['squeezed']} squeezed (< {r['squeeze_s'] * 1000:.0f} ms per phone) - {p['stretched']} stretched (> "
          f"{r['stretch_s']} s per phone after the held-note stretch) - {p['p_scale']} x ({r['p_sure']} - p) / {r['p_sure']} "
          f"(Whisper probability p < {r['p_sure']}); weak below {r['weak_below']}. Mismatch = the heard token is not the lyric "
          "word after normalisation (case, punctuation, numbers spelled out, simple stemming). Nothing here changes the track.", '']
    return '\n'.join(L)


def write_check(C, json_path, md_path=None):
    json.dump(C, open(json_path, 'w'), indent=1)
    md_path = md_path or os.path.splitext(json_path)[0] + '.md'
    open(md_path, 'w').write(check_md(C))
    return json_path, md_path


# ------------------------------------------------------------------------------------------ suggested lyrics --
def sheet_tokens(text):
    """the lyric sheet's sung tokens as lyric_align.lyric_tokens reads them (lines starting with '[' skipped, split on
    spaces / hyphens), with their character spans: [(norm token, start, end)]."""
    out = []
    pos = 0
    for line in str(text).splitlines(keepends=True):
        if line.strip() and not line.strip().startswith('['):
            for m in re.finditer(r"[^\s\-]+", line):
                t = re.sub(r"[^a-z0-9']", '', m.group(0).lower().replace('’', "'"))
                t = t.strip("'") if t != "'em" else 'em'
                if t: out.append((t, pos + m.start(), pos + m.end()))
        pos += len(line)
    return out


def sheet_map(text, A):
    """{aligned word index: (sheet token index, the token as written)}: the sheet's tokens mapped onto the aligned words'
    lyric words in song order (difflib: equal blocks and same-length replaced blocks 1:1)."""
    toks = sheet_tokens(text)
    lyr, _ = lyric_words(A)
    W = A.get('words') or []
    seq = [(i, re.sub(r"[^a-z0-9']", '', str(lyr[i]).lower()).strip("'")) for i in range(len(W)) if lyr.get(i)]
    sm = difflib.SequenceMatcher(None, [t for t, _, _ in toks], [t for _, t in seq], autojunk=False)
    out = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal' or (op == 'replace' and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                _, a, b = toks[i1 + k]
                out[seq[j1 + k][0]] = (i1 + k, re.sub(r"^[^\w']+|[^\w']+$", '', text[a:b]))
    return out


def suggest_lyrics(text, A, mismatches):
    """the lyric sheet with each mismatched word replaced by '[heard: x]' (suggestions only). mismatches: [{i, heard}]
    (aligned word index). The sheet is mapped onto the aligned words (sheet_map); a heard word with no lyric word ('not
    in the sheet') is inserted after the sheet word sung before it. Returns (text, [(i, placed: bool)])."""
    toks = sheet_tokens(text)
    at = {i: t for i, (t, _) in sheet_map(text, A).items()}
    edits, placed = [], []
    for m in sorted(mismatches, key=lambda m: m['i']):
        i = m['i']
        h = str(m.get('heard') or '').strip()
        if i in at:
            _, a, b = toks[at[i]]
            edits.append((a, b, len(edits), f'[heard: {h}]'))
            placed.append((i, True))
            continue
        prev = max((j for j in at if j < i), default=None)            # not in the sheet: after the word sung before it
        if prev is not None:
            _, a, b = toks[at[prev]]
            edits.append((b, b, len(edits), f' [heard: {h}]'))
            placed.append((i, True))
        else: placed.append((i, False))
    for a, b, _, s in sorted(edits, reverse=True): text = text[:a] + s + text[b:]   # back to front; two inserts at one
    return text, placed                                                             # place keep their sung order


def find_file(path, *bases):
    """a path as given (cwd), else relative to each base directory and its parents; None when not found."""
    if not path: return None
    if os.path.isabs(path) or os.path.exists(path): return path if os.path.exists(path) else None
    for b in bases:
        if not b: continue
        d = os.path.abspath(b if os.path.isdir(b) else os.path.dirname(b))
        while True:
            c = os.path.join(d, path)
            if os.path.exists(c): return c
            nd = os.path.dirname(d)
            if nd == d: break
            d = nd
    return None
