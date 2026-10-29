#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lyric_align.py — sung audio + lyrics -> word and phone timings (JSON + Praat TextGrid).

Three stages:
  1. STT: faster-whisper (MIT code; Whisper weights MIT) with word timestamps over the window -> what was SUNG.
  2. Reconcile against the lyrics file: the best-matching lyric span is found (difflib). Lyric spelling wins
     ("400" -> "four hundred"); a different sung word wins over the lyric ("'em" sung as "them"); both are logged.
  3. Forced alignment per phrase (whisper gaps > --gap s) with SoundSwallower (BSD, CMU Sphinx; en-us acoustic
     model, Alpha Cephei BSD-style) using CMUdict pronunciations, OOV words via NRC g2p (MIT) or --extra-dict.
     Same libs as ReadAlongs Studio 1.2.2 (soundswallower 0.6.7, g2p 2.3.2) — called directly, because ReadAlongs'
     TextGrid has no phone tier. Each phrase's Sphinx result is sanity-checked against whisper word times and the
     vocal envelope; a phrase that fails falls back to whisper word times + phones spread by vowel-weighted duration
     (sung vowels hold most of the note).

  # stage 1 needs faster_whisper (run automatically with --whisper-python unless --stt-json is given)
  lsdub align --audio vocal_dry.wav \
      --lyrics lyrics.txt --t0 6 --t1 17.5 --out align_018.json \
      [--textgrid out.TextGrid] [--extra-dict "zyxx=Z IH K S"]
All times in the output are SOURCE-AUDIO (song/stem) time.
"""
import argparse, difflib, json, os, re, sys, tempfile
from ..io import ff
import numpy as np

VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
EXTRA = {}   # add out-of-dictionary words with --extra-dict "word=PH ON ES"
NUM = {'400': 'four hundred', '4': 'four'}


def norm(w):
    w = w.lower().replace('’', "'").strip()
    w = re.sub(r"[^a-z0-9']", '', w)
    return w.strip("'") if w not in ("'em",) else 'em'


def load_pcm(path, t0, t1):
    raw = ff.run(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-t', f'{t1 - t0:.3f}', '-i', path, '-ac', '1',
                          '-ar', '16000', '-f', 's16le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16)


def envelope(x, hop=160):
    n = len(x) // hop
    f = x[:n * hop].astype(np.float32).reshape(n, hop) / 32768
    return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)    # 10 ms frames, dBFS


def spread_lyrics(tokens, env, t0, hop_s=0.01, floor_db=-35.0):
    """fallback timing: the lyric tokens evenly over the voiced span (env within floor_db of its peak)."""
    if not tokens or len(env) == 0: return []
    v = np.flatnonzero(env >= env.max() + floor_db)
    a, b = (v[0], v[-1] + 1) if len(v) else (0, len(env))
    d = (b - a) * hop_s / len(tokens)
    return [{'w': w, 'start': round(t0 + a * hop_s + i * d, 3), 'end': round(t0 + a * hop_s + (i + 1) * d, 3), 'p': 0.0,
             'heard': '', 'src': 'lyric', 'align': 'spread'} for i, w in enumerate(tokens)]


GAP_CLOSE = 0.12                     # s: an unvoiced gap this long inside a word closes the mouth
SIL_DB = 20 * np.log10(0.02)          # dBFS: the sync gate's silence level (audits.snap.SIL_RMS)
HOLD_MAX = 0.8                        # s: the longest held-note stretch given to the word before a voiced gap
FILL_MIN = 0.3                        # s: an uncovered voiced stretch this long gets words (an unheard line, or 'da')
FILL_SYL = 0.3                        # s per filled vocalisation syllable
LOW_MATCH = 0.5                       # lyric match ratio below which speech-to-text timing is not trusted


def voiced_thr(env, floor_db=35.0):
    """voiced = within floor_db of the window peak, and never below the sync gate's silence level."""
    return max(float(env.max()) - floor_db, SIL_DB) if len(env) else SIL_DB


def voiced_runs(env, thr, t0, lo, hi, gap=GAP_CLOSE, hop_s=0.01):
    """the voiced stretches in [lo, hi) (source time), unvoiced gaps shorter than `gap` bridged -> [(start, end)]."""
    i0 = max(0, int(round((lo - t0) / hop_s)))
    i1 = min(len(env), int(round((hi - t0) / hop_s)))
    idx = np.flatnonzero(env[i0:i1] > thr) if i1 > i0 else np.array([], int)
    if not len(idx): return []
    runs, s, prev = [], idx[0], idx[0]
    for j in idx[1:]:
        if (j - prev - 1) * hop_s >= gap - 1e-9:
            runs.append((s, prev + 1))
            s = j
        prev = j
    runs.append((s, prev + 1))
    return [(t0 + (i0 + a) * hop_s, t0 + (i0 + b) * hop_s) for a, b in runs]


def warp_to_runs(items, runs):
    """items [(label, weight)] laid end to end over the voiced runs joined (the gaps hold nothing) -> [(label, start, end)];
    an item that crosses a gap is cut in two, so the mouth closes over the gap."""
    total = sum(e - s for s, e in runs)
    W = float(sum(w for _, w in items))
    if total <= 0 or W <= 0: return []
    out, pos = [], 0.0
    for lab, w in items:
        a, b = pos, pos + total * w / W
        pos = b
        acc = 0.0
        for s, e in runs:
            lo, hi = max(a, acc), min(b, acc + e - s)
            if hi - lo > 1e-6: out.append((lab, s + lo - acc, s + hi - acc))
            acc += e - s
    return out


def phone_weight(p):
    return 4.0 if p.rstrip('012') in VOWELS else 1.0


def spread_voiced(w, phones, env, thr, t0, gap=GAP_CLOSE):
    """the fallback spread, clipped to the voice: phones (vowels 4x) over the word's voiced frames only; an unvoiced gap
    of `gap` s or more inside the word holds no phone (the mouth closes there). A word with no voiced frame: plain spread."""
    runs = voiced_runs(env, thr, t0, w['start'], w['end'], gap)
    if not runs: return spread(w, phones)
    pcs = warp_to_runs([(i, phone_weight(p)) for i, p in enumerate(phones)], runs)
    return [{'p': phones[i], 'start': float(a), 'end': float(b)} for i, a, b in pcs]


def syllables(word):
    return max(1, len(re.findall(r'[aeiouy]+', word)))


def place_lyrics(lines, env, t0, thr, lo=None, hi=None):
    """no usable speech-to-text timing: the lyric lines laid over the voiced stretches of [lo, hi) in order, each word
    weighted by its syllables; nothing lands in an unvoiced gap. -> whisper-like word dicts (align 'voiced-place')."""
    lo = t0 if lo is None else lo
    hi = t0 + len(env) / 100 if hi is None else hi
    toks = [w for ln in lines for w in ln]
    runs = voiced_runs(env, thr, t0, lo, hi)
    if not toks or not runs: return []
    pcs = warp_to_runs([(i, syllables(w)) for i, w in enumerate(toks)], runs)
    out = {}
    for i, a, b in pcs:
        o = out.setdefault(i, {'w': toks[i], 'start': a, 'end': b, 'p': 0.0, 'heard': '', 'src': 'lyric', 'align': 'voiced-place'})
        o['end'] = b
    return [out[i] for i in sorted(out)]


def lyric_lines(path):
    """the lyric sheet as lines of normalised tokens (section headers and empty lines dropped)."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('['): continue
        t = [n for n in (norm(w) for w in re.split(r"[\s\-]+", line)) if n]
        if t: out.append(t)
    return out


def unheard_lines(lines, words):
    """lyric lines none of whose words were aligned -> [(line index, tokens, lo, hi)]: lo / hi = the end of the last
    heard word of an earlier line and the start of the first heard word of a later line (None = open)."""
    flat = [(li, w) for li, ln in enumerate(lines) for w in ln]
    sm = difflib.SequenceMatcher(None, [w for _, w in flat], [w['w'] for w in words], autojunk=False)
    got = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n): got[a + k] = words[b + k]
    heard = {}
    for k, (li, _) in enumerate(flat):
        if k in got: heard.setdefault(li, []).append(got[k])
    out = []
    for li, ln in enumerate(lines):
        if li in heard: continue
        before = [w['end'] for j in heard if j < li for w in heard[j]]
        after = [w['start'] for j in heard if j > li for w in heard[j]]
        out.append((li, ln, max(before) if before else None, min(after) if after else None))
    return out


def fill_voiced(final, lines, env, thr, t0, t1, extra=None, min_len=FILL_MIN, syl=FILL_SYL):
    """singing that no word covers: an unheard lyric line goes in the longest uncovered voiced stretch between its
    neighbours; any other uncovered stretch of min_len s or more is a vocalisation ('da' per syl s, src 'vocalisation').
    Returns the added words (align 'voiced-fill'), phones clipped to the voice."""
    extra = extra or {}

    def free():
        cov = sorted((w['start'], w['end']) for w in final)
        pcs = []
        for s, e in voiced_runs(env, thr, t0, t0, t1):
            cur = s
            for a, b in cov:
                if b <= cur or a >= e: continue
                if a - cur >= min_len: pcs.append((cur, a))
                cur = max(cur, b)
            if e - cur >= min_len: pcs.append((cur, e))
        return pcs

    added = []
    for li, toks, lo, hi in unheard_lines(lines, final) if lines else []:
        c = [(s, e) for s, e in free() if (lo is None or s >= lo - 1e-6) and (hi is None or e <= hi + 1e-6)]
        if not c: continue
        s, e = max(c, key=lambda r: r[1] - r[0])
        for w in place_lyrics([toks], env, t0, thr, s, e):
            w.update(align='voiced-fill', p_whisper=0.0, phones=spread_voiced(w, dict_phones(None, w['w'], extra), env, thr, t0))
            final.append(w)
            added.append(w)
    for s, e in free():
        k = max(1, int(round((e - s) / syl)))
        for i in range(k):
            w = {'w': 'da', 'start': s + i * (e - s) / k, 'end': s + (i + 1) * (e - s) / k, 'p': 0.0, 'heard': '',
                 'src': 'vocalisation', 'align': 'voiced-fill', 'p_whisper': 0.0}
            w['phones'] = spread_voiced(w, ['D', 'AH'], env, thr, t0)
            final.append(w)
            added.append(w)
    final.sort(key=lambda w: w['start'])
    return added


def stt(audio, t0, t1, model='large-v3', prompt='', device='cpu', device_index=0, download_root=None):
    from faster_whisper import WhisperModel
    with tempfile.TemporaryDirectory() as d:
        p = f'{d}/w.wav'
        ff.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t0), '-t', str(t1 - t0), '-i', audio, '-ac', '1', '-ar', '16000', p], check=True)
        try:
            m = WhisperModel(model, device=device, device_index=device_index, download_root=download_root,
                             compute_type='int8' if device == 'cpu' else 'float16')
        except (RuntimeError, ValueError) as e:                       # no usable GPU runtime: the CPU path
            print(f'stt: {device} unavailable ({e}); using the CPU', file=sys.stderr)
            m = WhisperModel(model, device='cpu', compute_type='int8', download_root=download_root)
        import numpy as np
        raw = ff.run(['ffmpeg', '-v', 'error', '-i', p, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], capture_output=True, check=True).stdout
        pcm16 = np.frombuffer(raw, np.float32)          # decoded here, not by faster-whisper (its PyAV call varies by version)
        segs, _ = m.transcribe(pcm16, word_timestamps=True, language='en', vad_filter=False, initial_prompt=prompt or None)
        return [{'w': w.word.strip(), 'start': round(w.start + t0, 3), 'end': round(w.end + t0, 3), 'p': round(w.probability, 3)}
                for s in segs for w in s.words]


def lyric_tokens(path):
    toks = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('['): continue
        for w in re.split(r"[\s\-]+", line):
            n = norm(w)
            if n: toks.append(n)
    return toks


def reconcile(whisper, lyr):
    """Map whisper words onto the best lyric span. Returns words [{w, start, end, src, heard}] (whisper timing)."""
    ws = []
    for x in whisper:          # expand numbers so they can match the lyric spelling
        n = norm(x['w'])
        if n in NUM:
            parts = NUM[n].split()
            d = (x['end'] - x['start']) / len(parts)
            ws += [dict(x, w=p, start=x['start'] + i * d, end=x['start'] + (i + 1) * d, heard=x['w']) for i, p in enumerate(parts)]
        elif n: ws.append(dict(x, w=n, heard=x['w']))
    wt = [x['w'] for x in ws]
    best = (0, 0)
    for s in range(len(lyr)):
        r = difflib.SequenceMatcher(None, wt, lyr[s:s + len(wt) + 3], autojunk=False).ratio()
        if r > best[0]: best = (r, s)
    span = lyr[best[1]:best[1] + len(wt) + 3]
    out, notes = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wt, span, autojunk=False).get_opcodes():
        if op == 'equal':
            out += [dict(ws[i], src='lyric') for i in range(i1, i2)]
        elif op == 'replace':
            for k, i in enumerate(range(i1, i2)):
                lw = span[j1 + k] if j1 + k < j2 else None
                close = lw and difflib.SequenceMatcher(None, ws[i]['w'], lw).ratio() >= 0.6 and lw != 'em'
                out.append(dict(ws[i], w=lw if close else ws[i]['w'], src='lyric~' if close else 'sung'))
                notes.append(f"{ws[i]['heard']!r} vs lyric {lw!r} -> {out[-1]['w']!r}")
        elif op == 'delete':
            out += [dict(ws[i], src='sung-only') for i in range(i1, i2)]
            notes.append('sung, not in lyric: ' + ' '.join(wt[i1:i2]))
        elif op == 'insert' and j1 > 0 and i1 < len(wt):
            notes.append('lyric, not heard: ' + ' '.join(span[j1:j2]))
    return out, round(best[0], 3), notes


def decoder():
    import soundswallower as ss
    d = ss.Decoder(hmm=ss.get_model_path('en-us'), dict=ss.get_model_path('en-us/dict.txt'), samprate=16000,
                   loglevel='FATAL', beam=1e-100, wbeam=1e-80, pbeam=1e-100)
    return d


def pron(d, w, extra):
    if w in extra: return extra[w]
    try:
        if d.lookup_word(w) is not None: return None   # already in CMUdict
    except Exception: pass
    try:
        from g2p import make_g2p
        s = make_g2p('eng', 'eng-arpabet')(w).output_string.strip()
        if s: return s
    except Exception: pass
    print(f'align: no pronunciation for {w!r}; reading it off the spelling (add a pronunciation for a better one)', file=sys.stderr)
    return ' '.join(letter_phones(w))


_DIGRAPH = {'ch': ['CH'], 'sh': ['SH'], 'th': ['TH'], 'ph': ['F'], 'ng': ['NG'], 'ck': ['K'], 'wh': ['W'], 'qu': ['K', 'W']}
_CONS = {'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G', 'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N',
         'p': 'P', 'q': 'K', 'r': 'R', 's': 'S', 't': 'T', 'v': 'V', 'w': 'W', 'x': 'K S', 'z': 'Z'}
_VGROUP = {'ee': 'IY', 'ea': 'IY', 'ie': 'IY', 'oo': 'UW', 'ou': 'AW', 'ow': 'OW', 'ai': 'EY', 'ay': 'EY', 'ei': 'EY',
           'ey': 'EY', 'oa': 'OW', 'oi': 'OY', 'oy': 'OY', 'au': 'AO', 'aw': 'AO'}
_V1 = {'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AA', 'u': 'AH', 'y': 'IY'}


def letter_phones(w):
    """last resort for a word no dictionary or g2p knows: phones read off the spelling (a rough but mouth-shaped guess:
    the closures and vowels land where the letters put them)."""
    w = re.sub(r"[^a-z]", '', w.lower())
    out, i = [], 0
    while i < len(w):
        two = w[i:i + 2]
        if two in _DIGRAPH:
            out += _DIGRAPH[two]
            i += 2
        elif w[i] in 'aeiouy' and not (w[i] == 'y' and i == 0):
            j = i
            while j < len(w) and w[j] in 'aeiouy': j += 1
            g = w[i:j]
            if g == 'e' and j == len(w) and out: pass                    # a silent final e
            elif g in _VGROUP: out.append(_VGROUP[g])
            elif len(g) >= 2 and g[0] in 'ei' and g[1] in 'aou': out += [_V1['i'].replace('IH', 'IY'), _V1.get(g[1], 'AH')]
            else: out.append(_V1[g[0]])
            i = j
        else:
            if w[i] in _CONS and not (out and w[i] == w[i - 1]): out += _CONS[w[i]].split()
            i += 1
    return out or ['AH']


def dict_phones(d, w, extra):
    if w in extra: return extra[w].split()
    try:
        from g2p import make_g2p
        s = make_g2p('eng', 'eng-arpabet')(w).output_string.split()
        if s: return s
    except Exception: pass
    return letter_phones(w)


def sphinx_phrase(pcm, words, extra, off):
    d = decoder()
    for w in words:
        p = pron(d, w['w'], extra)
        if p: d.add_word(w['w'], p, True)
    d.set_align_text(' '.join(w['w'] for w in words))
    d.start_utt()
    d.process_raw(pcm.tobytes(), full_utt=True)
    d.end_utt()
    out = []
    for we in d.alignment:
        if we.name in ('<sil>', '(NULL)', '<s>', '</s>') or we.name.startswith('++'): continue
        out.append({'w': re.sub(r'\(\d+\)$', '', we.name), 'start': off + we.start / 100, 'end': off + (we.start + we.duration) / 100,
                    'phones': [{'p': pe.name, 'start': off + pe.start / 100, 'end': off + (pe.start + pe.duration) / 100}
                               for pe in we if pe.name != 'SIL']})
    return out


MFA_DIR = os.environ.get('LIPS_MFA_DIR', os.path.expanduser('~/.cache/lsdub/mfa'))


def mfa_dict_has(words, path):
    have = set()
    for line in open(path, encoding='utf-8'):
        k = line.split('\t', 1)[0]
        if k in words: have.add(k)
    return have


def stressed(phones):
    """MFA's english_us_arpa phones carry stress digits; give a bare vowel 1 (first) or 0."""
    out, first = [], True
    for p in phones:
        if p in VOWELS:
            out.append(p + ('1' if first else '0'))
            first = False
        else: out.append(p)
    return out


def mfa_batch(pcm, t0, segs, extra, mfa_dir=MFA_DIR, beam=100, retry_beam=400, jobs=4):
    """Montreal Forced Aligner (MIT; english_us_arpa acoustic model + dictionary, CC BY 4.0) on several phrase segments in
    one run. segs = [(s0, s1, [word dicts])] in source time. Returns {k: [ {w,start,end,phones:[{p,start,end}]} ]} with
    absolute times; a segment MFA could not align is missing (or has an 'spn' word) and the caller falls back."""
    import shutil, wave as _w
    env = dict(os.environ, MFA_ROOT_DIR=f'{mfa_dir}/mfa_root', PATH=f'{mfa_dir}/env/bin:' + os.environ['PATH'])
    base = f'{mfa_dir}/mfa_root/pretrained_models/dictionary/english_us_arpa.dict'
    vocab = {w['w'] for _, _, ws in segs for w in ws}
    have = mfa_dict_has(vocab, base)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f'{d}/c/s')
        shutil.copy(base, f'{d}/dict.txt')
        with open(f'{d}/dict.txt', 'a', encoding='utf-8') as f:
            for w in sorted(vocab - have | (vocab & set(extra))):
                ph = stressed([p.rstrip('012') for p in (extra[w].split() if w in extra else dict_phones(None, w, extra))])
                f.write(f"{w}\t0.99\t0.05\t1.0\t1.0\t{' '.join(ph)}\n")
        for k, (s0, s1, ws) in enumerate(segs):
            x = pcm[int((s0 - t0) * 16000): int((s1 - t0) * 16000)]
            with _w.open(f'{d}/c/s/seg{k:02d}.wav', 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(x.tobytes())
            open(f'{d}/c/s/seg{k:02d}.lab', 'w').write(' '.join(w['w'] for w in ws))
        mfa = f'{mfa_dir}/env/bin/mfa' if os.path.exists(f'{mfa_dir}/env/bin/mfa') else 'mfa'
        r = ff.tool([mfa, 'align', f'{d}/c', f'{d}/dict.txt', 'english_us_arpa', f'{d}/o', '--clean', '--single_speaker',
                            '--beam', str(beam), '--retry_beam', str(retry_beam), '-j', str(jobs), '--output_format', 'json'],
                           env=env, capture_output=True, text=True)
        if r.returncode:
            print('MFA failed:', r.stdout[-800:], r.stderr[-800:], file=sys.stderr)
            return {}
        out = {}
        for k, (s0, s1, ws) in enumerate(segs):
            fp = f'{d}/o/s/seg{k:02d}.json'
            if not os.path.exists(fp): continue
            T = json.load(open(fp))['tiers']
            phs = [(a + s0, b + s0, p) for a, b, p in T['phones']['entries'] if p not in ('', 'sil', 'sp', 'spn')]
            res = []
            for a, b, w in T['words']['entries']:
                if w in ('', '<eps>'): continue
                res.append({'w': w, 'start': a + s0, 'end': b + s0,
                            'phones': [{'p': p, 'start': pa, 'end': pb} for pa, pb, p in phs if pa >= a + s0 - 1e-4 and pb <= b + s0 + 1e-4]})
            out[k] = res
        return out


def env_cut(env, t0, lo, hi):
    """Quietest 10 ms frame in [lo, hi] (source time) - where to cut between two phrases."""
    i0, i1 = max(0, int((lo - t0) * 100)), max(int((lo - t0) * 100) + 1, int((hi - t0) * 100))
    return t0 + (i0 + int(np.argmin(env[i0:i1]))) / 100 + 0.005


def env_bounds(env, thr, t0):
    """Listen-proxy boundaries: voiced-run onsets/offsets and deep dips (>= 4 dB) inside voiced runs, source time."""
    v = env > thr
    b = []
    for i in range(1, len(v)):
        if v[i] != v[i - 1]: b.append(t0 + i / 100)
    s = np.convolve(env, np.ones(3) / 3, 'same')
    b += [t0 + k / 100 for k in range(2, len(s) - 2) if s[k] < s[k - 2] - 4 and s[k] < s[k + 2] - 4 and s[k] > thr - 10]
    return np.array(sorted(b))


def env_score(ws, bounds, env, thr, t0, seg=None):
    """Lower = better. Mean per word: distance of the onset to the nearest envelope boundary (capped 0.3 s) + the unvoiced
    fraction of the word + 0.5 x the time a non-final word runs past 0.75 s (the 'one word swallowed the line' failure);
    plus the voiced seconds in the segment before the first / after the last word (singing that belongs to no word -
    gaps BETWEEN words are fine, the held-note stretch gives them to the earlier word)."""
    sc = []
    for i, w in enumerate(ws):
        d = min(0.3, float(np.min(np.abs(bounds - w['start'])))) if len(bounds) else 0.3
        i0 = int((w['start'] - t0) * 100)
        i1 = max(i0 + 1, int((w['end'] - t0) * 100))
        un = 1 - float((env[i0:i1] > thr).mean())
        long = max(0.0, w['end'] - w['start'] - 0.75) if i < len(ws) - 1 else 0.0
        sc.append(d + un + 0.5 * long)
    orphan = 0.0
    if seg:
        v = env > thr
        for lo, hi in ((seg[0], ws[0]['start']), (ws[-1]['end'], seg[1])):
            if hi > lo: orphan += float(v[int((lo - t0) * 100):int((hi - t0) * 100)].sum()) / 100
    return float(np.mean(sc)) + orphan


def extend_tail(ws, seg, env, thr, t0):
    """Envelope snap for a phrase's END: if singing continues after the last word inside its segment (a held note the
    aligner cut short), stretch the last word's last vowel to the last voiced frame. Returns the seconds added."""
    v = env > thr
    w = ws[-1]
    i0, i1 = int((w['end'] - t0) * 100), int((seg[1] - t0) * 100)
    j, last, gap = i0, None, 0                 # follow the note while it continues (dropouts < 100 ms allowed)
    while j < min(i1, len(v)) and gap < 10:
        if v[j]: last, gap = j, 0
        else: gap += 1
        j += 1
    if last is None: return 0.0
    new = t0 + (last + 1) / 100
    d = new - w['end']
    if d < 0.03: return 0.0
    ph = w['phones']
    k = max([i for i, p in enumerate(ph) if p['p'].rstrip('012') in VOWELS] or [len(ph) - 1])
    ph[k]['end'] += d
    for p in ph[k + 1:]:
        p['start'] += d
        p['end'] += d
    w['end'] = new
    w['tail_snap_s'] = round(d, 3)
    return d


def snap(words, env, thr, t0):
    """Whisper word edges often sit in the breath/silence before a sung note: move a start that lies in silence forward to
    the first voiced 10 ms frame inside the word, and an end back to the last voiced frame (keeps >= 60 ms)."""
    v = env > thr
    for w in words:
        i0, i1 = int((w['start'] - t0) * 100), int((w['end'] - t0) * 100)
        idx = np.where(v[max(0, i0):max(i0 + 1, i1)])[0]
        if len(idx) == 0: continue
        s, e = t0 + (max(0, i0) + idx[0]) / 100, t0 + (max(0, i0) + idx[-1] + 1) / 100
        if e - s >= 0.06:
            w['whisper_start'], w['whisper_end'] = w['start'], w['end']
            w['start'], w['end'] = s, e


def spread(w, phones):
    """Fallback: distribute phones over the whisper word; vowels weigh 4x a consonant (held sung notes)."""
    wt = np.array([4.0 if p.rstrip('012') in VOWELS else 1.0 for p in phones])
    wt /= wt.sum()
    e = w['start'] + np.concatenate([[0], np.cumsum(wt)]) * (w['end'] - w['start'])
    return [{'p': p, 'start': float(e[i]), 'end': float(e[i + 1])} for i, p in enumerate(phones)]


def write_textgrid(path, words, t0, t1):
    def tier(name, items):
        iv, t = [], t0
        for s, e, lab in items:
            if s > t + 1e-4: iv.append((t, s, ''))
            iv.append((max(s, t), e, lab))
            t = e
        if t < t1: iv.append((t, t1, ''))
        body = '\n'.join(f'        intervals [{i + 1}]:\n            xmin = {a:.3f}\n            xmax = {b:.3f}\n            text = "{l}"'
                         for i, (a, b, l) in enumerate(iv))
        return (f'    item [{{}}]:\n        class = "IntervalTier"\n        name = "{name}"\n        xmin = {t0:.3f}\n'
                f'        xmax = {t1:.3f}\n        intervals: size = {len(iv)}\n{body}')
    tiers = [tier('words', [(w['start'], w['end'], w['w']) for w in words]),
             tier('phones', [(p['start'], p['end'], p['p']) for w in words for p in w['phones']])]
    s = (f'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = {t0:.3f}\nxmax = {t1:.3f}\ntiers? <exists>\n'
         f'size = {len(tiers)}\nitem []:\n' + '\n'.join(t.replace('item [{}]', f'item [{i + 1}]') for i, t in enumerate(tiers)) + '\n')
    open(path, 'w').write(s)


def rhubarb_crosscheck(words, rh):
    """Per word: offset of the nearest Rhubarb mouth-cue change (into a non-rest shape) to the word onset, within
    +-0.35 s, and whether a word with P/B/M has a Rhubarb A (closed) cue on it. Rhubarb has no word tier, so this is a
    proxy; it is only as good as Rhubarb's own PocketSphinx pass on singing."""
    cues = rh['cues']
    ch = [c['start'] for i, c in enumerate(cues) if c['shape'] not in ('X',) and (i == 0 or cues[i - 1]['shape'] != c['shape'])]
    out = []
    for w in words:
        if w['end'] < rh['t0'] or w['start'] > rh['t1']: continue
        near = min(ch, key=lambda t: abs(t - w['start'])) if ch else None
        off = round(near - w['start'], 3) if near is not None and abs(near - w['start']) <= 0.35 else None
        r = {'w': w['w'], 'start': w['start'], 'rhubarb_onset_off_s': off}
        for p in w['phones']:
            if p['p'].rstrip('012') in ('P', 'B', 'M'):
                hit = any(c['shape'] == 'A' and c['start'] < p['end'] + 0.06 and c['end'] > p['start'] - 0.06 for c in cues)
                r.setdefault('closures', []).append({'p': p['p'], 't': round(p['start'], 3), 'rhubarb_A': hit})
        out.append(r)
    offs = [abs(r['rhubarb_onset_off_s']) for r in out if r['rhubarb_onset_off_s'] is not None]
    cl = [c['rhubarb_A'] for r in out for c in r.get('closures', [])]
    return {'words': out, 'onset_matched': f'{len(offs)}/{len(out)}', 'onset_abs_median_s': round(float(np.median(offs)), 3) if offs else None,
            'closures_agree': f'{sum(cl)}/{len(cl)}'}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--lyrics', required=True)
    ap.add_argument('--t0', type=float, required=True)
    ap.add_argument('--t1', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--textgrid')
    ap.add_argument('--stt-json', help='cached stage-1 words; written here if missing')
    ap.add_argument('--whisper-python', default=os.environ.get('LIPS_WHISPER_PYTHON'), help='python with faster_whisper (default: this one, in-process)')
    ap.add_argument('--whisper-model', default='large-v3')
    ap.add_argument('--device', default='cpu', help='speech-to-text device: cpu (default) or cuda')
    ap.add_argument('--device-index', type=int, default=0)
    ap.add_argument('--whisper-dir', help='where speech-to-text weights are downloaded (default: the Hugging Face cache)')
    ap.add_argument('--prompt', default='')
    ap.add_argument('--gap', type=float, default=0.30, help='whisper gap that splits phrases (s)')
    ap.add_argument('--pad', type=float, default=0.25, help='audio padding around each phrase (s)')
    ap.add_argument('--max-dev', type=float, default=0.35, help='max |sphinx-whisper| word-edge deviation before fallback (s)')
    ap.add_argument('--extra-dict', action='append', default=[], help='"word=PH ON ES"')
    ap.add_argument('--snap-db', type=float, default=25.0, help='voiced = within this many dB of the window peak (edge snap)')
    ap.add_argument('--stt-only', action='store_true')
    ap.add_argument('--aligner', choices=['sphinx', 'mfa', 'best'], default='sphinx',
                    help='mfa = Montreal Forced Aligner per phrase, sphinx/whisper chain for phrases MFA fails; best = per phrase, '
                         'MFA vs the sphinx/whisper chain, whichever sits better on the vocal envelope (env_score)')
    ap.add_argument('--mfa-max-phrase', type=float, default=3.0, help='mfa: split whisper phrases longer than this (s) at their largest gap')
    ap.add_argument('--mfa-max-dev', type=float, default=0.6, help='mfa: max |mfa-whisper| word-edge deviation before fallback (s)')
    ap.add_argument('--rhubarb-json', help='cross-check: Rhubarb cues (lsdub rhubarb JSON, song time); per-word report')
    a = ap.parse_args(argv)
    extra = dict(EXTRA)
    extra.update({k.strip(): v.strip() for k, v in (e.split('=', 1) for e in a.extra_dict)})

    if a.stt_only:
        json.dump(stt(a.audio, a.t0, a.t1, a.whisper_model, a.prompt, a.device, a.device_index, a.whisper_dir), open(a.stt_json, 'w'), indent=1)
        return
    sj = a.stt_json or a.out.replace('.json', '') + '_stt.json'
    if not os.path.exists(sj) and not a.whisper_python:
        json.dump(stt(a.audio, a.t0, a.t1, a.whisper_model, a.prompt, a.device, a.device_index, a.whisper_dir), open(sj, 'w'), indent=1)
    if not os.path.exists(sj):
        ff.python(a.whisper_python, 'lsdub_engine.core.lyric_align', ['--stt-only', '--audio', a.audio, '--lyrics', a.lyrics,
                        '--t0', str(a.t0), '--t1', str(a.t1), '--out', a.out, '--stt-json', sj, '--whisper-model', a.whisper_model,
                        '--prompt', a.prompt], check=True)
    whisper = json.load(open(sj))
    words, ratio, notes = reconcile(whisper, lyric_tokens(a.lyrics))

    pcm = load_pcm(a.audio, a.t0, a.t1)
    env = envelope(pcm)
    thr = env.max() - 35
    vthr = voiced_thr(env)
    lines = lyric_lines(a.lyrics)
    if not words or ratio < LOW_MATCH:                                  # speech-to-text heard nothing, or not the lyrics
        words = place_lyrics(lines, env, a.t0, vthr)                    # (a scat chorus, a heavy mix): the lyric sheet over
        notes.append(f'lyric match {ratio} < {LOW_MATCH}: speech-to-text timing not used; the lyrics were placed over '   # the
                     'the voiced stretches (low confidence)')                                                            # voice
        if not words: words = spread_lyrics(lyric_tokens(a.lyrics), env, a.t0)
        if not words: raise RuntimeError('align: no lyrics and no sung words heard')
    snap(words, env, env.max() - a.snap_db, a.t0)
    phrases, cur = [], [words[0]]
    for w in words[1:]:
        if w['start'] - cur[-1]['end'] > a.gap:
            phrases.append(cur)
            cur = [w]
        else: cur.append(w)
    phrases.append(cur)
    for i, w in enumerate(words): w['idx'] = i
    bounds = env_bounds(env, env.max() - a.snap_db, a.t0)

    def sphinx_chain(phrases):
        final, report = [], []
        todo = list(phrases)
        while todo:
            ph = todo.pop(0)
            s0, s1 = max(a.t0, ph[0]['start'] - a.pad), min(a.t1, ph[-1]['end'] + a.pad)
            seg = pcm[int((s0 - a.t0) * 16000): int((s1 - a.t0) * 16000)]
            res, why = None, ''
            try:
                res = sphinx_phrase(seg, ph, extra, s0)
                if len(res) != len(ph):
                    why = f'word count {len(res)}!={len(ph)}'
                    res = None
            except Exception as e:
                why = f'sphinx error {e}'
            if res:
                dev = max(max(abs(r['start'] - w['start']), abs(r['end'] - w['end'])) for r, w in zip(res, ph))
                # envelope check: the fraction of each word's frames that are voiced
                vo = [float((env[int((r['start'] - a.t0) * 100):int((r['end'] - a.t0) * 100)] > thr).mean() or 0) for r in res]
                if dev > a.max_dev:
                    why = f'max word-edge dev {dev:.2f}s vs whisper'
                    res = None
                elif min(vo) < 0.3:
                    why = f'word on silence (voiced {min(vo):.2f})'
                    res = None
            if not res and len(ph) >= 4:     # Sphinx often copes with a shorter phrase: split at the largest gap, retry
                k = 1 + int(np.argmax([ph[i + 1]['start'] - ph[i]['end'] for i in range(len(ph) - 1)]))
                todo[:0] = [ph[:k], ph[k:]]
                report.append({'phrase': ' '.join(w['w'] for w in ph), 'method': 'split-retry', 'why': why})
                continue
            if res:
                for r, w in zip(res, ph): r.update(src=w['src'], heard=w['heard'], p_whisper=w['p'], align='sphinx', idx=w['idx'])
                final += res
                report.append({'phrase': ' '.join(w['w'] for w in ph), 'method': 'sphinx', 'max_dev_s': round(float(dev), 3)})
            else:
                for w in ph:
                    ps = dict_phones(None, w['w'], extra)
                    ph_ = spread_voiced(w, ps, env, vthr, a.t0)
                    final.append({'w': w['w'], 'start': ph_[0]['start'], 'end': ph_[-1]['end'], 'src': w['src'], 'heard': w['heard'],
                                  'p_whisper': w['p'], 'align': 'whisper+spread', 'phones': ph_, 'idx': w['idx']})
                report.append({'phrase': ' '.join(w['w'] for w in ph), 'method': 'fallback', 'why': why})
        return final, report

    if a.aligner == 'sphinx':
        final, report = sphinx_chain(phrases)
    else:
        mph = []
        for ph in phrases:                        # long whisper phrases -> split at the largest gap (MFA drifts on long sung runs)
            st = [ph]
            while st:
                q = st.pop(0)
                if q[-1]['end'] - q[0]['start'] > a.mfa_max_phrase and len(q) > 1:
                    k = 1 + int(np.argmax([q[i + 1]['start'] - q[i]['end'] for i in range(len(q) - 1)]))
                    st[:0] = [q[:k], q[k:]]
                else: mph.append(q)
        segs = []
        for i, ph in enumerate(mph):              # cut between phrases at the quietest point (whisper edges are loose)
            s0 = ph[0]['start'] - 0.6 if i == 0 else env_cut(env, a.t0, mph[i - 1][-1]['end'] - 0.1, ph[0]['start'] + 0.35)
            s1 = ph[-1]['end'] + 0.9 if i == len(mph) - 1 else env_cut(env, a.t0, ph[-1]['end'] - 0.1, mph[i + 1][0]['start'] + 0.35)
            segs.append((max(a.t0, s0), min(a.t1, s1), ph))
        got = mfa_batch(pcm, a.t0, segs, extra)
        base, base_rep = sphinx_chain(phrases) if a.aligner == 'best' else ([], [])
        by_idx = {w['idx']: w for w in base}
        final, report, failed = [], [], []
        for k, (s0, s1, ph) in enumerate(segs):
            res = raw = got.get(k)
            why = ''
            if not res or len(res) != len(ph) or any(r['w'] != w['w'] or not r['phones'] for r, w in zip(res, ph)):
                why = 'mfa: no/partial alignment'
                res = None
            if res:
                dev = max(max(abs(r['start'] - w['start']), abs(r['end'] - w['end'])) for r, w in zip(res, ph))
                vo = [float((env[int((r['start'] - a.t0) * 100):max(int((r['start'] - a.t0) * 100) + 1, int((r['end'] - a.t0) * 100))] > thr).mean())
                      for r in res]
                if dev > a.mfa_max_dev:
                    why = f'mfa: max word-edge dev {dev:.2f}s vs whisper'
                    res = None
                elif min(vo) < 0.3:
                    why = f'mfa: word on silence (voiced {min(vo):.2f})'
                    res = None
            if res:
                for r, w in zip(res, ph): r.update(src=w['src'], heard=w['heard'], p_whisper=w['p'], align='mfa', idx=w['idx'])
            rep = {'phrase': ' '.join(w['w'] for w in ph), 'seg': [round(s0, 3), round(s1, 3)],
                   'mfa_words': [[r['w'], round(r['start'], 3), round(r['end'], 3)] for r in raw or []]}
            if a.aligner == 'best':
                alt = [by_idx[w['idx']] for w in ph]
                for c in (res, alt):
                    if c: extend_tail(c, (s0, s1), env, env.max() - a.snap_db, a.t0)
                sm = env_score(res, bounds, env, thr, a.t0, (s0, s1)) if res else None
                sa = env_score(alt, bounds, env, thr, a.t0, (s0, s1))
                rep.update(mfa_score=sm and round(sm, 3), alt_score=round(sa, 3), alt_method=alt[0]['align'], mfa_why=why or None)
                if res and sm <= sa:
                    final += res
                    rep['method'] = 'mfa'
                else:
                    final += alt
                    rep['method'] = alt[0]['align'] + ' (best)'
                report.append(rep)
            elif res:
                extend_tail(res, (s0, s1), env, env.max() - a.snap_db, a.t0)
                final += res
                rep.update(method='mfa', max_dev_s=round(float(dev), 3))
                report.append(rep)
            else:
                failed.append(ph)
                rep.update(method='mfa-failed', why=why)
                report.append(rep)
        if failed:
            f2, r2 = sphinx_chain(failed)
            final += f2
            report += r2
        if a.aligner == 'best':
            report.append({'sphinx_chain_report': base_rep})
    final.sort(key=lambda w: w['start'])
    for prev, w in zip(final, final[1:]):   # mixed sphinx/whisper phrases can overlap: split the overlap at its midpoint
        if w['start'] < prev['end']:
            m = (w['start'] + prev['end']) / 2
            m = min(max(m, prev['start'] + 0.04), w['end'] - 0.04)
            for x, lo, hi in ((prev, prev['start'], m), (w, m, w['end'])):
                s0, s1 = x['start'], x['end']
                f = (hi - lo) / max(1e-6, s1 - s0)
                for p in x['phones']: p['start'], p['end'] = lo + (p['start'] - s0) * f, lo + (p['end'] - s0) * f
                x['start'], x['end'] = lo, hi
    v = env > env.max() - a.snap_db
    for prev, w in zip(final, final[1:]):   # a voiced gap between words = a held note: stretch the earlier word's last vowel
        g0, g1 = int((prev['end'] - a.t0) * 100), int((w['start'] - a.t0) * 100)
        if g1 - g0 >= 5 and v[g0:g1].mean() > 0.7:
            run = voiced_runs(env, vthr, a.t0, prev['end'], w['start'])   # up to the first unvoiced gap, at most HOLD_MAX
            if not run or run[0][0] > prev['end'] + 0.03: continue
            d = min(run[0][1], prev['end'] + HOLD_MAX) - prev['end']
            if d < 0.03: continue
            ph = prev['phones']
            k = max([i for i, p in enumerate(ph) if p['p'].rstrip('012') in VOWELS] or [len(ph) - 1])
            ph[k]['end'] += d
            for p in ph[k + 1:]:
                p['start'] += d
                p['end'] += d
            prev['end'] += d
            prev['held_s'] = round(d, 3)
    if final and final[-1]['end'] < a.t1:   # the last word's held note, to the same limits
        w = final[-1]
        run = voiced_runs(env, vthr, a.t0, w['end'], a.t1)
        if run and run[0][0] <= w['end'] + 0.03:
            d = min(run[0][1], w['end'] + HOLD_MAX) - w['end']
            if d >= 0.03:
                ph = w['phones']
                k = max([i for i, p in enumerate(ph) if p['p'].rstrip('012') in VOWELS] or [len(ph) - 1])
                ph[k]['end'] += d
                for p in ph[k + 1:]:
                    p['start'] += d
                    p['end'] += d
                w['end'] += d
                w['held_s'] = round(d, 3)
    fills = fill_voiced(final, lines, env, vthr, a.t0, a.t1, extra)   # singing no word covers: unheard lines, then 'da'
    if fills:
        notes.append(f"{len(fills)} word(s) placed in voiced stretches no aligned word covered: "
                     + ' '.join(w['w'] for w in fills[:24]) + (' ...' if len(fills) > 24 else ''))
    for w in final:
        w['start'], w['end'] = round(w['start'], 3), round(w['end'], 3)
        for p in w['phones']:
            p['p'] = p['p'].rstrip('012')
            p['start'], p['end'] = round(p['start'], 3), round(p['end'], 3)
    xc = rhubarb_crosscheck(final, json.load(open(a.rhubarb_json))) if a.rhubarb_json else None
    out = {'audio': a.audio, 'aligner': a.aligner, 'rhubarb_crosscheck': xc, 'lyrics': a.lyrics, 't0': a.t0, 't1': a.t1, 'lyric_match_ratio': ratio, 'reconcile_notes': notes,
           'phrases': report, 'words': final}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    if a.textgrid: write_textgrid(a.textgrid, final, a.t0, a.t1)
    print(f'lyric match {ratio}; notes: {notes}')
    for r in report: print(' ', r)
    print(' '.join(f"{w['w']}[{w['start']:.2f}-{w['end']:.2f}]" for w in final))


if __name__ == '__main__':
    main()
