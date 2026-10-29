#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""grammar.py: `lsdub grammar`, the shot-grammar gate. Every sung word must land on a shot where its singer's face is
big enough for the kind of line it is (research: "Sync the singers by shot size", recommendation 1).

  tiers   singer face height: close-up >= 300, medium 120-300, wide 25-120, none (< 25 or no singer face).
          Basis (--px-basis): native (DEFAULT) = delivered-frame px, as the research report's tier table states it;
          1080 = cast face_px, 1080-normalised. The warp's own 300 px dub-safe gate (cast.SAFE_PX) is on the 1080
          basis: on a 768-px delivery a 300 px@1080 face is only 213 delivered px.
  needs   hook / key chorus line, exposed solo lyric (a **X sings** beat)  -> a close-up of the OWNING singer
          verse line                                                       -> a medium or better (medium = tier-2 crop)
          held note (the tail of a word held >= --held s), breath, instrumental -> anything (wides belong here)
  plus    a sung word where a NON-singer face is the largest face (> half of its frames that have a face) is flagged,
          and one whose singer's mouth is not visible (cast: no landmarks / mouth_occluded) on most of its frames.
          Delegates and greys are named by family (pleiadian, grey). A word is judged against ITS singer.

Ownership reuses `lsdub cast plan`'s script parser (script_singers); a '(cont.)' cut with no singer of its own inherits
the last singers of the same song section. The cast scan may be one long video (the delivery cut) or one clip per cut.

  lsdub grammar --cast cast_v3_2.json --cut storyboard_v3/cut.json --align align_full_best.json \\
                --script storyboard_v3/SCRIPT.md --song-offset 12 --out grammar.json --md grammar.md
"""
import argparse, json, os, re, sys
import cv2, numpy as np

from .cast import script_singers, words_in, family
from .. import presets as P

CU_PX, MED_PX, WIDE_PX = 240.0, 120.0, 25.0   # the CU line is a parameter (--cu-px); 240 delivered px by default:
WARP_GATE_1080 = 300.0                        # QA: 06C / 11C / SNG_sky_final at 252-280 native are clean CUs. The
                                              # warp's own dub-safe gate stays 300 px on the 1080 basis (cast.SAFE_PX)
ORDER = {'none': 0, 'wide': 1, 'medium': 2, 'cu': 3}
NEED = {'hook': 'cu', 'solo': 'cu', 'verse': 'medium', 'held': 'none'}
SOLO_RE = re.compile(r'\*\*\s*([A-Za-z]+)\s+sings', re.I)   # a bold '**<Name> sings**' beat


def tier(px, cu=CU_PX):
    return 'cu' if px >= cu else 'medium' if px >= MED_PX else 'wide' if px >= WIDE_PX else 'none'


def _norm(w):
    return re.sub(r"[^a-z']", '', w.lower().replace('’', "'"))


def hook_words(words, hooks=()):
    """-> set of word indices (into words) that are part of a hook phrase (contiguous, case/punctuation-blind)."""
    seq = [_norm(w['w']) for w in words]
    out = set()
    for h in hooks:
        hp = [_norm(x) for x in h.split()]
        for i in range(len(seq) - len(hp) + 1):
            if seq[i:i + len(hp)] == hp:
                out |= set(range(i, i + len(hp)))
    return out


def _first(lyric):
    """first sung word of a script Lyric cell ('"Pleiadians don't dance..."' -> 'pleiadians'; '(cont.)' -> '')."""
    m = re.search(r'["\u201c]\s*\**([A-Za-z\'\u2019]+)', lyric or '')
    return _norm(m.group(1)) if m else ''


def script_beats(text, preset=None):
    """-> {cut: {'singers': set, 'solo': bool, 'section': str}} for every beat row. Singers come from cast.script_singers;
    solo = a bold '**Red sings**' / '**Sky sings**' beat (the exposed, SING-tagged lines)."""
    names = P.get(preset).names
    solo_rx = re.compile(r'\*\*\s*(' + '|'.join(re.escape(k) for k in names) + r')\s+sings', re.I) if names else SOLO_RE
    sing = script_singers(text, preset)
    out, section, lyric_col = {}, '', None
    for line in text.splitlines():
        if line.startswith('#'):
            section = re.sub(r'^#+\s*', '', line).strip()
            lyric_col = None
            continue
        hdr = [x.strip() for x in line.strip().strip('|').split('|')]
        if hdr and hdr[0] == '#' and 'Story beat' in hdr:
            lyric_col = hdr.index('Lyric') if 'Lyric' in hdr else None
            continue
        m = re.match(r'\|\s*(\d+)\s*\|', line)
        if not m: continue
        c = int(m.group(1))
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        lyr = cells[lyric_col] if lyric_col is not None and len(cells) > lyric_col else ''
        out[c] = {'singers': set(sing.get(c, set())), 'solo': bool(solo_rx.search(line)), 'section': section, 'lyric': lyr}
    return out


def owners(beats, n_cuts, singers=()):
    """Per cut: (owner set, source). '(cont.)' cuts inherit the last singers of the same section. singers: the
    'any_singer' set (empty = any named character, resolved by grammar())."""
    SINGERS = tuple(singers or ())
    out, last, sec = {}, set(), None
    for ci in range(n_cuts):
        b = beats.get(ci)
        if b is None:                                                  # not in the script: keep the section and its singers
            out[ci] = (set(last), 'inherited') if last else (set(SINGERS), 'any_singer')
            continue
        if b['section'] != sec: last, sec = set(), b['section']
        if b['singers']:
            out[ci] = (set(b['singers']), 'script')
            last = set(b['singers'])
        elif last:
            out[ci] = (set(last), 'inherited')
        else:
            out[ci] = (set(SINGERS), 'any_singer')
    return out


MIC_SHARE = 0.10   # chrome share of the lip zone above which the mouth counts as behind a mic (v3_2: clear CUs <= 0.06,
                   # cut 19 Red with the mic at her lips 0.12-0.22)


def mic_share(img, box):
    """CPU stand-in for `lsdub occluders` (GPU): the share of the lip zone (face box rows 0.62-0.95, middle half) that
    reads chrome / grey (HSV S < 50, V > 70). Skin, lips and the dark cavity are saturated or dark; a mic grille or
    a chrome body is not. Teeth can be grey-white, so the zone stops above the chin and the threshold is a share."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    z = img[max(0, int(y0 + 0.62 * h)):max(0, int(y0 + 0.95 * h)), max(0, int(x0 + 0.25 * w)):max(0, int(x1 - 0.25 * w))]
    if z.size == 0: return 0.0
    hsv = cv2.cvtColor(z, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] < 50) & (hsv[..., 2] > 70)).mean())


def add_mic_share(cast, video, cut=None, fps=25.0, chars=None, preset=None):
    """Annotate every singer frame with mic_share, reading the delivered video once. A per-clip scan (NNN.mp4 results,
    clip-relative frames) is offset by its cut's start, exactly as face_index does; that needs `cut`."""
    chars = set(chars) if chars else P.singer_set(preset, cast)
    want = {}
    for off, r in _sources(cast, cut, fps):
        for t in r['tracks']:
            if t['character'] in chars:
                for x in t['frames']: want.setdefault(off + x['f'], []).append(x)
    cap = cv2.VideoCapture(video)
    i = 0
    while want:
        ok, img = cap.read()
        if not ok: break
        for x in want.pop(i, []): x['mic_share'] = round(mic_share(img, x['box']), 3)
        i += 1
    return cast


def _sources(cast, cut, fps):
    """-> [(video frame offset, result)]: 0 for a whole-video scan; the cut's start frame for each per-clip NNN.mp4."""
    res = cast['results']
    by_clip = {int(os.path.splitext(os.path.basename(r['video']))[0]): r for r in res
               if os.path.splitext(os.path.basename(r['video']))[0].isdigit()}
    if not by_clip:
        return [(0, r) for r in res]
    if cut is None:
        raise ValueError('a per-clip cast scan needs the cut list to place its frames in the video')
    return [(int(round(cut['cut'][ci]['start'] * fps)), r) for ci, r in by_clip.items() if ci < len(cut['cut'])]


def face_index(cast, cut, fps):
    """-> {video frame: [(character, face_px, mouth_visible), ...]} from a whole-video scan or a per-clip (NNN.mp4)
    scan. mouth_visible = the landmarker read the face AND cast did not mark the mouth occluded (a mic, a hand)."""
    idx = {}
    for off, r in _sources(cast, cut, fps):
        for t in r['tracks']:
            for x in t['frames']:
                vis = bool(x.get('landmarks', True)) and not x.get('mouth_occluded') and x.get('mic_share', 0.0) <= MIC_SHARE
                idx.setdefault(off + x['f'], []).append((t['character'], float(x['face_px']), vis))
    return idx


def word_class(w, gi, hooks, solo, held_s):
    if gi in hooks: return 'hook'
    if solo: return 'solo'
    return 'verse'


def name(ch, preset=None):
    """Report label: a family (the three near-identical Pleiadian delegates, the greys) is named by family, so one
    character never carries two names across cuts."""
    f = family(ch, preset)
    return f if f != ch else ch


def grammar(cast, cut, words, beats, song_offset=12.0, fps=25.0, held_s=0.6, held_lead=0.25, min_frames=2,
            hooks=None, px_scale=1.0, min_visible=0.5, cu_px=CU_PX, preset=None):
    """One row per cut: the words it carries, what each needs, what the singer face gives, PASS / FLAG + a fix.
    px_scale multiplies cast face_px (1080-normalised): H / 1080 = delivered-frame px (the default in main), 1.0 = the
    1080 basis. A word is checked against ITS singer (the owner of the cut holding its attack), so a held note running over a
    cut is judged by who sings it."""
    FI = face_index(cast, cut, fps)
    pre = P.get(preset)
    ANY = P.singer_set(pre, cast)
    HW = hook_words(words, pre.hooks if hooks is None else hooks)
    gidx = {id(w): i for i, w in enumerate(words)}
    OWN = owners(beats, len(cut['cut']), pre.singers or ())
    starts = np.array([c['start'] for c in cut['cut']])
    rows = []
    for ci, c in enumerate(cut['cut']):
        b = beats.get(ci, {'singers': set(), 'solo': False, 'section': ''})
        own, osrc = OWN[ci]
        g0 = int(round(c['start'] * fps))
        g1 = int(round((c['start'] + c['dur']) * fps)) - 1
        ws = words_in(words, c['start'] - song_offset, c['start'] + c['dur'] - song_offset)
        W, flags, pres_all = [], [], []
        for w in ws:
            a = max(g0, int(np.floor((w['start'] + song_offset) * fps)))
            e = min(g1, int(np.ceil((w['end'] + song_offset) * fps)) - 1)
            if e - a + 1 < min_frames: continue
            t_att = w['start'] + 0.5 * min(w['end'] - w['start'], held_lead) + song_offset   # the attack's middle
            wc = int(np.clip(np.searchsorted(starts, t_att, 'right') - 1, 0, len(starts) - 1))
            span = range(int(np.clip(np.searchsorted(starts, w['start'] + song_offset, 'right') - 1, 0, len(starts) - 1)),
                         int(np.clip(np.searchsorted(starts, w['end'] + song_offset, 'left') - 1, 0, len(starts) - 1)) + 1)
            # a word sung ahead of its line's cut (the line's first word, held across the cut) belongs to that line's cut
            wc = next((k for k in span if k != wc and _first(beats.get(k, {}).get('lyric', '')) == _norm(w['w'])), wc)
            wown = OWN[wc][0] or ANY                                   # the word's singer
            bw = beats.get(wc, b)
            cls = word_class(w, gidx[id(w)], HW, bw.get('solo'), held_s)
            # a held note: only its attack needs the line's tier; the tail is 'held' (anything goes). A tail that runs
            # into this cut (held point at or before its first frame) has no attack here: the whole word is held.
            held_from = int(np.floor((w['start'] + held_lead + song_offset) * fps)) if w['end'] - w['start'] >= held_s else None
            fr = list(range(a, e + 1))
            if held_from is not None and held_from <= a:
                cls, att = 'held', []
            else:
                att = [f for f in fr if held_from is None or f < held_from]
            need = NEED[cls]
            def own_face(f):
                fs = [(p * px_scale, v) for ch, p, v in FI.get(f, []) if ch in wown]
                return max(fs) if fs else (0.0, False)
            OF = [own_face(f) for f in (att or fr)]
            px = float(np.median([p for p, _ in OF]))
            got = tier(px, cu_px)
            pres = [p for p, _ in OF if p > 0]
            pres_all.extend(pres)
            seen = [v for p, v in OF if p > 0]
            vis = float(np.mean(seen)) if seen else 0.0
            hidden = need != 'none' and bool(seen) and vis < min_visible
            big = []
            for f in fr:
                fs = FI.get(f, [])
                if fs:
                    ch, p, _ = max(fs, key=lambda x: x[1])
                    if ch not in wown and p * px_scale >= WIDE_PX: big.append(name(ch, pre))
            nfaces = sum(bool(FI.get(f)) for f in fr)
            nonsinger = bool(big) and len(big) > 0.5 * nfaces            # of the frames that HAVE a face
            ok = ORDER[got] >= ORDER[need]
            rec = {'w': w['w'], 'song_t': [round(w['start'], 3), round(w['end'], 3)], 'frames': [a - g0, e - g0], 'class': cls,
                   'singer': sorted(wown), 'held_tail': held_from is not None, 'need': need, 'owner_px': round(px, 1),
                   'owner_px_present': round(float(np.median(pres)), 1) if pres else 0.0,
                   'tier': got, 'tier_ok': ok, 'mouth_visible': round(vis, 2), 'mouth_hidden': hidden,
                   'largest_nonsinger': max(set(big), key=big.count) if nonsinger else None, 'faceless': nfaces == 0}
            W.append(rec)
            if not ok: flags.append('tier')
            if nonsinger: flags.append('nonsinger')
            if hidden: flags.append('mouth_hidden')
        need_c = max((x['need'] for x in W), key=ORDER.get, default=None)
        cut_px = float(np.median(pres_all)) if pres_all else 0.0        # median singer px over every sung frame she is in
        verdict = 'FLAG' if flags else 'PASS'
        ns = sorted({x['largest_nonsinger'] for x in W if x['largest_nonsinger']})
        rows.append({'cut': ci, 'shot': c['shot'], 'video_t': [round(c['start'], 3), round(c['start'] + c['dur'], 3)],
                     'section': b.get('section'), 'owners': sorted(own), 'owner_src': osrc, 'solo': b.get('solo', False),
                     'sung': bool(W), 'need': need_c, 'tier': tier(cut_px, cu_px) if W else None,
                     'owner_px': round(cut_px, 1) if W else None,           # median singer px over frames she is in
                     'n_owner_absent': sum(x['owner_px_present'] == 0 for x in W),
                     'owner_px_med': round(float(np.median([x['owner_px'] for x in W])), 1) if W else None,
                     'largest_nonsinger': ns, 'verdict': verdict, 'n_tier_fail': sum(not x['tier_ok'] for x in W),
                     'n_nonsinger': sum(bool(x['largest_nonsinger']) for x in W),
                     'n_mouth_hidden': sum(x['mouth_hidden'] for x in W),
                     'words': ' '.join(x['w'] for x in W), 'word_checks': W, 'fix': fix(W, own, cu_px),
                     'method': method(W) if verdict == 'PASS' else None})
    return rows


def _who(own):
    return ' + '.join(sorted(own))


def fix(W, own, cu_px=CU_PX):
    """A suggested re-stage for the failing words (None when the cut passes). Notes add up: a cutaway note never hides
    a largest-face or hidden-mouth note."""
    out = []
    bad = [x for x in W if not x['tier_ok']]
    cu = any(x['need'] == 'cu' for x in bad)
    if bad and all(x['faceless'] for x in bad):
        out.append(f"insert / cutaway (no face) over sung {'hook' if cu else 'verse'} words: slide it onto an instrumental bar "
                   f"or held note, or replace with a {'CU' if cu else 'medium'} of {_who(own)}")
    elif cu:
        kinds = sorted({x['class'] for x in bad if x['need'] == 'cu'})
        out.append(f"re-stage as a CU of {_who(own)} ({'/'.join(kinds)} line; face >= {cu_px:.0f} px)")
    elif bad:
        worst = min((x['tier'] for x in bad), key=ORDER.get)
        out.append(f"re-stage as a medium (or CU) of {_who(own)} (verse line on a {worst}; face >= {MED_PX:.0f} px, "
                   "then tier-2 `lsdub medium`) or move the cut so the words land on a held note / instrumental bar")
    if any(x['mouth_hidden'] for x in W):
        out.append(f"mouth hidden (mic / hand / prop) on most sung frames: re-stage with {_who(own)}'s mouth clear, or treat as a cutaway")
    ns = sorted({x['largest_nonsinger'] for x in W if x['largest_nonsinger']})
    if ns:
        out.append(f"re-frame: {', '.join(ns)} is the largest face on sung words; put {_who(own)} in front or cut to them")
    return '; '.join(out) or None


def method(W):
    """For a passing sung cut: which dub path its words need."""
    if not W: return 'none (instrumental)'
    t = min((x['tier'] for x in W if x['need'] != 'none'), key=ORDER.get, default=None)
    return {'cu': 'tier-1 warp (lsdub paint)', 'medium': 'tier-2 crop (lsdub medium)'}.get(t, 'none (held / breath only)')


def summary(rows):
    s = {'cuts': len(rows), 'sung_cuts': sum(r['sung'] for r in rows), 'pass': sum(r['verdict'] == 'PASS' for r in rows),
         'flag': sum(r['verdict'] == 'FLAG' for r in rows),
         'flag_tier': sum(r['n_tier_fail'] > 0 for r in rows), 'flag_nonsinger': sum(r['n_nonsinger'] > 0 for r in rows),
         'sung_words': sum(len(r['word_checks']) for r in rows),
         'words_failing_tier': sum(r['n_tier_fail'] for r in rows), 'words_nonsinger_largest': sum(r['n_nonsinger'] for r in rows),
         'flag_mouth_hidden': sum(r['n_mouth_hidden'] > 0 for r in rows), 'words_mouth_hidden': sum(r['n_mouth_hidden'] for r in rows)}
    s['by_tier'] = {k: sum(r['tier'] == k for r in rows if r['sung']) for k in ORDER}
    return s


BASIS = {'native': 'delivered-frame px (the report\'s tier table)',
         '1080': '1080-normalised px (cast face_px; the basis of the warp\'s 300 px dub-safe gate, SAFE_PX in cast.py)'}


def to_md(rows, S, basis='native', cu_px=CU_PX):
    L = ['# Shot-grammar gate', '',
         f'Face px basis: **{basis}** = {BASIS[basis]}. Tiers: CU >= {cu_px:.0f} (--cu-px), medium {MED_PX:.0f}-{cu_px:.0f}, '
         f'wide {WIDE_PX:.0f}-{MED_PX:.0f}, none below. Hook / solo lines need a CU of the word\'s singer, verses a medium, '
         f'held-note tails anything. (The warp\'s own dub-safe gate is {WARP_GATE_1080:.0f} px on the 1080 basis.) '
         'The mouth must be visible; the mouth must be visible (cast: landmarks, not occluded) on most sung frames.', '',
         '| cut | shot | v (s) | owner | need | got | px | mouth vis | largest non-singer | verdict | words | fix / method |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if not r['sung']:
            L.append(f"| {r['cut']} | {r['shot']} | {r['video_t'][0]:.2f} | - | - | - | - | - | - | PASS | (instrumental) | |")
            continue
        ws = ' '.join(f"**{x['w']}**" if (not x['tier_ok'] or x['largest_nonsinger'] or x['mouth_hidden']) else x['w'] for x in r['word_checks'])
        vis = [x['mouth_visible'] for x in r['word_checks'] if x['owner_px'] > 0]
        own = ' + '.join(o.split('_')[0] for o in r['owners']) + ('*' if r['owner_src'] != 'script' else '')
        L.append(f"| {r['cut']} | {r['shot']} | {r['video_t'][0]:.2f} | {own} | {r['need']} | {r['tier']} | {r['owner_px']:.0f} | "
                 f"{f'{np.mean(vis):.0%}' if vis else '-'} | {', '.join(r['largest_nonsinger']) or '-'} | {'**FLAG**' if r['verdict'] == 'FLAG' else 'PASS'} | {ws} | {r['fix'] or r['method'] or ''} |")
    L += ['', '`*` owner inherited from the section (a (cont.) cut) or any singer. **Bold** words fail.', '',
          '## Summary', '', '```', json.dumps(S, indent=1), '```']
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub grammar')
    ap.add_argument('--cast', required=True, help='lsdub cast scan json')
    ap.add_argument('--cut', required=True)
    ap.add_argument('--align', required=True)
    ap.add_argument('--script', required=True)
    ap.add_argument('--song-offset', type=float, default=12.0)
    ap.add_argument('--fps', type=float, default=25.0)
    ap.add_argument('--held', type=float, default=0.6, help='a word this long (s) is a held note after its first --held-lead s')
    ap.add_argument('--held-lead', type=float, default=0.25)
    ap.add_argument('--hook', action='append', help="hook phrase (repeatable; default: the preset's hooks)")
    ap.add_argument('--preset', help='cast preset (presets.py), e.g. landing')
    ap.add_argument('--px-basis', choices=['native', '1080'], default='native',
                    help="native (default) = delivered-frame px, the research report's tier table; 1080 = cast face_px, "
                         "1080-normalised, the basis of the warp's 300 px dub-safe gate (cast.SAFE_PX)")
    ap.add_argument('--cu-px', type=float, default=CU_PX, help='close-up line in the chosen basis (default 240 delivered px)')
    ap.add_argument('--video', help='the delivered video: adds a CPU mic-at-the-lips check to mouth visibility')
    ap.add_argument('--min-visible', type=float, default=0.5, help='share of sung frames the singer\'s mouth must be visible on')
    ap.add_argument('--out', required=True)
    ap.add_argument('--md')
    a = ap.parse_args(argv)
    cast, cut = json.load(open(a.cast)), json.load(open(a.cut))
    words = json.load(open(a.align))['words']
    beats = script_beats(open(a.script).read(), a.preset)
    if a.video: add_mic_share(cast, a.video, cut, a.fps, preset=a.preset)
    H = (cast['results'][0].get('size') or [0, 1080])[1]
    k = H / 1080.0 if a.px_basis == 'native' else 1.0
    rows = grammar(cast, cut, words, beats, a.song_offset, a.fps, a.held, a.held_lead, hooks=tuple(a.hook) if a.hook else None, px_scale=k, min_visible=a.min_visible, cu_px=a.cu_px, preset=a.preset)
    S = summary(rows)
    json.dump({'cast': a.cast, 'cut': a.cut, 'align': a.align, 'script': a.script, 'song_offset': a.song_offset, 'px_basis': a.px_basis,
               'thresholds': {'cu': a.cu_px, 'warp_gate_1080': WARP_GATE_1080, 'medium': MED_PX, 'wide': WIDE_PX}, 'summary': S, 'cuts': rows}, open(a.out, 'w'), indent=1)
    if a.md: open(a.md, 'w').write(to_md(rows, S, a.px_basis, a.cu_px))
    for r in rows:
        if r['verdict'] == 'FLAG':
            print(f"FLAG cut {r['cut']:2d} {r['shot']:<18} need {r['need']:<6} got {r['tier']:<6} {r['owner_px']:5.0f}px  {r['fix']}")
    print(json.dumps(S))
    return 0


if __name__ == '__main__':
    sys.exit(main())
