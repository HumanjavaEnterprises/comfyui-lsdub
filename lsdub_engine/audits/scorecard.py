#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""scorecard.py: `lsdub scorecard`, score a delivered film SHOT BY SHOT, so the weak shots show up front (founder: "score
the videos shot to shot, so we see up front which shots score lower vs higher"). CPU.

Per cut (every cut of cut.json):
  sung cuts   sync      measured singer mouth opening vs the aligned vocal opening (sync-qa film mode): corr, best lag
              silence   open-in-silence (the mouth rests when the vocal does)
              snap      onset rise time, p / b / m closure hit-rate (lsdub snap)
              grammar   the singer's tier + px (delivered px, CU line 240) and mouth visible / mic (lsdub grammar)
  all cuts    motion    median frame-to-frame change (a static shot = a still image playing as video)
              flicker   temporal pops: frame-diff spikes far above the shot's own change
              sharpness Laplacian variance on the main face (or the frame centre) vs the film's median
              canon     cast canon similarity of the main named character (an off-model look scores low)
              faces     face count, the largest non-singer on a sung cut
One 0-100 SHOT SCORE = 100 x sum(weight x component in 0..1) / sum(weights of the components present). The reason
is the component that loses the most points (weight x (1 - value)).

WEIGHTS (sung cut / unsung cut):
  sync 28 / -, silence 8 / -, snap 6 / -, travel 12 / -, grammar tier 10 / -, mouth visible 8 / -, canon 10 / 30,
  motion 6 / 25, flicker 8 / 25, sharpness 4 / 20
  and a sung shot's score is then multiplied by (0.5 + 0.5 x sync): a shot that does not sync cannot score well.
Component values:
  sync      clip((corr_pm1 - 0.2) / 0.6)            corr 0.2 -> 0, 0.8 -> 1 (corr_pm1 = best within +-1 frame)
  silence   1 - clip((open_in_silence - 0.15) / 0.4)
  snap      mean of: closure_hit, 1 - clip((rise_ms - 40) / 160)   (whichever are measurable)
  travel    clip((p95 - p5 of the opening - 0.04) / 0.10): how far the mouth moves (catches a held gape)
  tier      the singer's tier vs the line's need: met 1, one tier short 0.6, two 0.3, none 0.1
  mouth     the singer's mouth-visible share (cast landmarks, no occlusion, no mic at the lips)
  canon     clip((similarity - 0.45) / 0.25) of the singer's face, or else the largest face >= 80 px; an 'unknown'
            face >= 100 px on a SUNG cut -> 0.3 (an extra in an insert is not scored)
CLASSES: PERF (sung, singer medium or closer, mouth visible >= 50%): scored, a mouth travel < 0.10 caps at 55;
  COVERAGE (sung, no readable singer): n/a, reason no face -> too small -> mouth hidden; INSERT (unsung): capped at 90.
  motion    clip(median frame diff / 0.8)            (0..255 grey levels at 1/4 size)
  flicker   1 - clip(pops / 3)
  sharpness clip(face Laplacian variance / the film's median)

  lsdub scorecard --video film.mp4 --cuts cut.json --vocal lead.wav --offset 12 --align align.json
                  [--script SCRIPT.md] [--cast cast.json] [--compare old.mp4 --compare-cast old_cast.json]
                  --out dir/name
"""
import argparse, csv, json, os, sys
import cv2, numpy as np

W_SUNG = dict(sync=28, silence=8, snap=6, travel=12, tier=10, mouth=8, canon=10, motion=6, flicker=8, sharpness=4)
W_PLAIN = dict(canon=30, motion=25, flicker=25, sharpness=20)
MIN_OVERLAP, MIN_SUNG_S = 0.08, 0.25   # a word counts in a cut when >= 2 frames of it land there; a cut is sung with >= 0.25 s
TIER_ORDER = {'none': 0, 'wide': 1, 'medium': 2, 'cu': 3}
GREEN, AMBER = 75, 55
SMALL = 4


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


# ---------------------------------------------------------------------------------------------------- pixels
def scan_pixels(video, cuts, fps, cast=None):
    """one pass over the film: per cut the frame-diff series (1/SMALL size grey), the middle frame (thumbnail) and the
    sharpness of the main face (cast box) or the centre on 3 sampled frames."""
    cap = cv2.VideoCapture(video)
    spans = [(int(round(c['start'] * fps)), int(round((c['start'] + c['dur']) * fps))) for c in cuts['cut']]
    big = biggest_boxes(cast, spans) if cast else {}
    samples = {}
    for ci, (a, b) in enumerate(spans):
        for f in np.linspace(a, b - 1, 3).astype(int): samples.setdefault(int(f), []).append(ci)
    diffs = [[] for _ in spans]
    thumbs = {}
    sharp = [[] for _ in spans]
    prev = None
    ci = 0
    g = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        while ci < len(spans) and g >= spans[ci][1]:
            ci += 1
            prev = None
        if ci >= len(spans): break
        sm = cv2.cvtColor(cv2.resize(fr, (fr.shape[1] // SMALL, fr.shape[0] // SMALL), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None and g > spans[ci][0]: diffs[ci].append(float(np.abs(sm - prev).mean()))
        prev = sm
        if g == (spans[ci][0] + spans[ci][1]) // 2: thumbs[ci] = cv2.resize(fr, (256, int(256 * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA)
        if g in samples:
            for c2 in samples[g]: sharp[c2].append(lap_var(fr, big.get((c2, g))))
        g += 1
    return spans, diffs, thumbs, sharp


def biggest_boxes(cast, spans):
    """{(cut, frame): box of the largest face} for the sampled frames, from a whole-video cast scan."""
    by_f = {}
    for r in cast['results']:
        for t in r['tracks']:
            for x in t['frames']:
                if x['f'] not in by_f or x['face_px'] > by_f[x['f']][1]: by_f[x['f']] = (x['box'], x['face_px'])
    out = {}
    for ci, (a, b) in enumerate(spans):
        for f in np.linspace(a, b - 1, 3).astype(int):
            if int(f) in by_f: out[(ci, int(f))] = by_f[int(f)][0]
    return out


def lap_var(img, box=None):
    H, W = img.shape[:2]
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 >= 16 and y1 - y0 >= 16: img = img[y0:y1, x0:x1]
    else:
        img = img[H // 4:3 * H // 4, W // 4:3 * W // 4]
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def motion_flicker(d):
    """-> (median diff, pops): a pop = a frame-to-frame change far above the shot's own (robust z > 6 and > 4 levels)."""
    d = np.asarray(d, float)
    if len(d) < 3: return (float(np.median(d)) if len(d) else 0.0), 0
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) * 1.4826 + 0.3
    return med, int(((d - med) / mad > 6).__and__(d > 4).sum())


# ---------------------------------------------------------------------------------------------------- identity
CANON_MIN_PX = 80.0


def canon(cast, a, b, singers, min_px=CANON_MIN_PX):
    """the main face of the cut for the canon check: the singer's face if one is on screen, else the largest face of
    >= min_px (an incidental extra never sets the canon score) -> (character, similarity, px); the face count (median
    per frame) and the largest non-singer."""
    per_frame = {}
    best = None
    ns = None
    for r in cast['results']:
        for t in r['tracks']:
            fr = [x for x in t['frames'] if a <= x['f'] < b]
            if not fr: continue
            for x in fr: per_frame[x['f']] = per_frame.get(x['f'], 0) + 1
            px = float(np.median([x['face_px'] for x in fr]))
            sc = t.get('scores') or {}
            sim = float(sc.get(t['character'], max(sc.values()) if sc else np.nan)) if t['character'] != 'unknown' else (float(max(sc.values())) if sc else np.nan)
            is_singer = t['character'] in singers
            rank = (is_singer, px)
            if (is_singer or px >= min_px) and (best is None or rank > best[3]):
                best = (t['character'], sim, px, rank)
            if singers and t['character'] not in singers and t['character'] != 'unknown' and (ns is None or px > ns[1]): ns = (t['character'], px)
    count = float(np.median([per_frame.get(f, 0) for f in range(a, b)])) if b > a else 0.0
    return (best[:3] if best else None), count, ns


# ---------------------------------------------------------------------------------------------------- combine
def _fin(x):
    return x is not None and np.isfinite(x)


def components(row, film_sharp_med):
    for k in ('corr', 'corr_pm1', 'open_in_silence', 'rise_ms', 'closure_hit', 'sharpness', 'canon_sim', 'mouth_range'):
        if k in row and not _fin(row[k]): row[k] = None                     # NaN (a flat series) = not measurable
    c = {}
    if row['sung'] and row.get('corr') is not None:
        c['sync'] = clip01((max(row['corr'], row.get('corr_pm1') or row['corr']) - 0.2) / 0.6)
        if row.get('open_in_silence') is not None: c['silence'] = 1 - clip01((row['open_in_silence'] - 0.15) / 0.4)
        sn = [v for v in (row.get('closure_hit'), None if row.get('rise_ms') is None else 1 - clip01((row['rise_ms'] - 40) / 160)) if v is not None]
        if sn: c['snap'] = float(np.mean(sn))
        if row.get('mouth_range') is not None: c['travel'] = clip01((row['mouth_range'] - 0.04) / 0.10)
    elif row['sung']:
        c['sync'] = 0.0                                                      # a sung cut with no measurable mouth
    if row['sung'] and row.get('tier') is not None and row.get('need') is not None:
        short = TIER_ORDER[row['need']] - TIER_ORDER[row['tier']]
        c['tier'] = 1.0 if short <= 0 else {1: 0.6, 2: 0.3}.get(short, 0.1)
    if row['sung'] and row.get('mouth_visible') is not None: c['mouth'] = clip01(row['mouth_visible'])
    if row.get('canon_char') and (row['canon_char'] == 'unknown' or row.get('canon_sim') is not None):
        c['canon'] = 0.3 if row['canon_char'] == 'unknown' and row['canon_px'] >= 100 and row['sung'] else (
            None if row['canon_char'] == 'unknown' else clip01((row['canon_sim'] - 0.45) / 0.25))
        if c['canon'] is None: del c['canon']
    c['motion'] = clip01(row['motion'] / 0.8)
    c['flicker'] = 1 - clip01(row['pops'] / 3)
    if row.get('sharpness') is not None and film_sharp_med: c['sharpness'] = clip01(row['sharpness'] / film_sharp_med)
    return c


TRAVEL_CAP, TRAVEL_CAP_SCORE, INSERT_CAP = 0.10, 55.0, 90.0
CLASSES = ('PERF', 'COVERAGE', 'INSERT')


def shot_class(row):
    """PERF = sung, the singer at medium or closer AND her mouth visible >= 50%; COVERAGE = sung but the singer absent /
    wide / hidden (an editorial choice, not a lip-sync defect: shown grey, 'n/a'); INSERT = unsung."""
    if not row['sung']: return 'INSERT'
    if row.get('tier') in ('medium', 'cu') and (row.get('mouth_visible') or 0) >= 0.5: return 'PERF'
    if row.get('tier') is None and row.get('corr') is not None: return 'PERF'           # no grammar data: judge on sync
    return 'COVERAGE'


def coverage_reason(row):
    """reason priority: no face -> singer too small -> mouth hidden -> sync."""
    if not row.get('owner_px') and row.get('tier') in (None, 'none'): return 'n/a: no singer on screen'
    if row.get('tier') in ('none', 'wide'): return f"n/a: singer too small ({row.get('tier')} {row.get('owner_px') or 0:.0f} px)"
    if (row.get('mouth_visible') or 0) < 0.5: return f"n/a: mouth hidden ({row.get('mouth_visible') or 0:.0%} visible)"
    return 'n/a'


def shot_score(c, sung):
    W = W_SUNG if sung else W_PLAIN
    keys = [k for k in c if k in W]
    if not keys: return None, None
    tot = sum(W[k] for k in keys)
    s = 100.0 * sum(W[k] * c[k] for k in keys) / tot
    if sung and 'sync' in c: s *= 0.5 + 0.5 * c['sync']               # a sung shot that does not sync cannot score well
    worst = max(keys, key=lambda k: W[k] * (1 - c[k]))
    if W[worst] * (1 - c[worst]) <= 1e-9: worst = None
    return round(s, 1), worst


REASON = {'sync': 'mouth off the vocal', 'silence': 'mouth open in the silences', 'snap': 'soft attacks / missed closures', 'travel': 'mouth barely moves (held gape / mumble)',
          'tier': 'singer too small for the line', 'mouth': 'mouth hidden (mic / hand / prop)', 'canon': 'off-model look',
          'motion': 'static / near-still', 'flicker': 'temporal pops', 'sharpness': 'soft image'}


def reason(row, worst):
    if worst is None: return ''
    v = row['components'][worst]
    extra = {'sync': f" (corr {row.get('corr')})", 'silence': f" ({row.get('open_in_silence')})", 'tier': f" ({row.get('tier')} {row.get('owner_px')} px, needs {row.get('need')})",
             'canon': f" ({row.get('canon_char')} {row.get('canon_sim')})", 'motion': f" (diff {row['motion']:.2f})", 'flicker': f" ({row['pops']} pops)",
             'sharpness': f" ({row.get('sharpness')})", 'mouth': f" ({row.get('mouth_visible')})", 'snap': f" (rise {row.get('rise_ms')}, closures {row.get('closure_hit')})", 'travel': f" (p5-p95 {row.get('mouth_range')})"}
    return f"{REASON[worst]}{extra.get(worst, '')} [{v:.2f}]"


def finalize(r, smed):
    """components -> class -> score / reason (COVERAGE n/a, INSERT capped, PERF travel-capped)."""
    r['components'] = {k: round(v, 3) for k, v in components(r, smed).items()}
    r['class'] = shot_class(r)
    s, worst = shot_score(r['components'], r['sung'])
    r['raw_score'] = s
    if r['class'] == 'COVERAGE':
        r['score'] = None
        r['reason'] = coverage_reason(r)
    elif r['class'] == 'INSERT':
        r['score'] = None if s is None else min(INSERT_CAP, s)
        r['reason'] = reason(r, worst) or 'pass'
    else:
        if r.get('mouth_range') is not None and r['mouth_range'] < TRAVEL_CAP and s is not None and s > TRAVEL_CAP_SCORE:
            s = TRAVEL_CAP_SCORE
            worst = 'travel'                                        # a held gape cannot rank as a performance
        r['score'] = s
        r['reason'] = reason(r, worst)
    return r


# ---------------------------------------------------------------------------------------------------- run
def score(video, cuts, vocal, offset=12.0, align=None, script=None, cast=None, work=None, preset=None):
    from . import sync_qa as Q
    from ..core import grammar as G
    from ..core.cast import words_in
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    work = work or os.path.splitext(video)[0] + '_scorecard_work'
    os.makedirs(work, exist_ok=True)
    spans, diffs, thumbs, sharp = scan_pixels(video, cuts, fps, cast)
    words = json.load(open(align))['words'] if align else []
    gram = {}
    if cast and script and align:
        cast = json.loads(json.dumps(cast))
        G.add_mic_share(cast, video, cuts, fps, preset=preset)
        beats = G.script_beats(open(script).read(), preset)
        for r in G.grammar(cast, cuts, words, beats, offset, fps, px_scale=H / 1080.0, preset=preset): gram[r['cut']] = r
        from .. import presets as P
        owners = G.owners(beats, len(cuts['cut']), P.get(preset).singers or ())
    else:
        owners = {}
    film = {r['cut']: r for r in Q.score_film(video, cuts, vocal, offset, align, cast, os.path.join(work, 'syncqa'), preset=preset)}
    rows = []
    for ci, c in enumerate(cuts['cut']):
        a, b = spans[ci]
        t0 = c['start'] - offset
        ws = [w for w in words_in(words, t0, t0 + c['dur']) if min(w['end'], t0 + c['dur']) - max(w['start'], t0) >= MIN_OVERLAP] if words else []
        med, pops = motion_flicker(diffs[ci])
        gr = gram.get(ci, {})
        row = {'cut': ci, 'shot': c.get('shot'), 'video_t': [round(c['start'], 2), round(c['start'] + c['dur'], 2)],
               'sung': sum(min(w['end'], t0 + c['dur']) - max(w['start'], t0) for w in ws) >= MIN_SUNG_S, 'words': ' '.join(w['w'] for w in ws),
               'motion': round(med, 3), 'static': med < 0.8 * 0.5, 'pops': pops,
               'sharpness': round(float(np.median(sharp[ci])), 1) if sharp[ci] else None}
        f = film.get(ci, {})
        for k in ('corr', 'best', 'lag_ms', 'open_in_silence', 'rise_ms', 'closure_hit', 'closures', 'mouth_range'): row[k] = f.get(k)
        if f.get('corr') is not None: row['corr_pm1'] = f.get('corr_pm1') or f['corr']
        if gr:
            vis = [x['mouth_visible'] for x in gr['word_checks'] if x['owner_px'] > 0]
            row.update(need=gr['need'], tier=gr['tier'], owner_px=gr['owner_px'], grammar=gr['verdict'],
                       mouth_visible=round(float(np.mean(vis)), 2) if vis else (0.0 if gr['sung'] else None),
                       largest_nonsinger=gr['largest_nonsinger'])
        if cast:
            sing = set(owners.get(ci, (set(), ''))[0]) if owners else set()
            best, count, ns = canon(cast, a, b, sing)
            row['faces'] = count
            if best: row.update(canon_char=best[0], canon_sim=round(best[1], 3), canon_px=round(best[2], 1))
            if ns and 'largest_nonsinger' not in row: row['largest_nonsinger'] = [ns[0]]
        rows.append(row)
    sh = [r['sharpness'] for r in rows if r['sharpness']]
    smed = float(np.median(sh)) if sh else None
    for r in rows: finalize(r, smed)
    return rows, thumbs


def compare(new, old):
    """match each new cut to the old cut it overlaps most in TIME (a re-cut film keeps working); warns when the edits differ."""
    def ov(a, b): return max(0.0, min(a['video_t'][1], b['video_t'][1]) - max(a['video_t'][0], b['video_t'][0]))
    if [r['video_t'] for r in new] != [r['video_t'] for r in old]:
        print('scorecard: the two films have different cut lists; matching cuts by time overlap', file=sys.stderr)
    for r in new:
        p = max(old, key=lambda q: ov(r, q), default=None)
        if p is not None and ov(r, p) < 0.5 * (r['video_t'][1] - r['video_t'][0]): p = None
        r['old_cut'] = None if p is None else p['cut']
        r['old_class'] = None if p is None else p.get('class')
        r['old_score'] = p['score'] if p else None
        r['delta'] = None if not p or p['score'] is None or r['score'] is None else round(r['score'] - p['score'], 1)
    return new


def arrow(d):
    return '' if d is None else ('↑' if d > 2 else '↓' if d < -2 else '=')


def ordered(rows):
    """class by class (PERF, COVERAGE, INSERT), worst first inside each."""
    return sorted(rows, key=lambda r: (CLASSES.index(r.get('class', 'INSERT')), r['score'] is None, r['score'] if r['score'] is not None else 0))


def write(rows, thumbs, out, video, compared=False):
    R = ordered(rows)
    json.dump({'video': video, 'weights': {'sung': W_SUNG, 'unsung': W_PLAIN}, 'classes': CLASSES, 'cuts': R}, open(out + '.json', 'w'), indent=1)
    cols = ['class', 'rank', 'cut', 'shot', 'score'] + (['old_score', 'delta'] if compared else []) + ['sung', 'corr', 'open_in_silence', 'rise_ms',
            'closure_hit', 'mouth_range', 'tier', 'owner_px', 'mouth_visible', 'canon_char', 'canon_sim', 'motion', 'static', 'pops', 'sharpness', 'faces', 'reason']
    rank = {}
    for r in R: rank[r['cut']] = sum(1 for q in R if q.get('class') == r.get('class') and R.index(q) <= R.index(r))
    with open(out + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in R: w.writerow([r.get('class'), rank[r['cut']]] + [r.get(k) for k in cols[2:]])
    L = [f'# Shot scorecard: {os.path.basename(video)}', '',
         f'Three classes, each ranked worst first. PERF = sung, singer medium or closer, mouth visible >= 50%: scored 0-100 (green >= {GREEN}, '
         f'amber >= {AMBER}, red below; a mouth travel under {TRAVEL_CAP} caps at {TRAVEL_CAP_SCORE:.0f}). COVERAGE = sung but no readable singer on '
         f'screen: n/a, grey (an editorial choice, not a sync score). INSERT = unsung: pass, capped at {INSERT_CAP:.0f}. '
         'Weights: `lsdub/audits/scorecard.py` docstring.', '']
    head = '| # | cut | shot | score | ' + ('was | Δ | ' if compared else '') + 'sync corr | silence | travel | tier px | mouth | canon | motion | pops | reason |'
    sep = '|---|---|---|---|' + ('---|---|' if compared else '') + '---|---|---|---|---|---|---|---|---|'
    for cl in CLASSES:
        part = [r for r in R if r.get('class') == cl]
        if not part: continue
        L += [f'## {cl} ({len(part)})', '', head, sep]
        for r in part:
            sc = ('n/a' if cl == 'COVERAGE' else '-') if r['score'] is None else f"{r['score']:.0f}"
            cmp_ = (f"{(r.get('old_class') or '-').lower() if r.get('old_score') is None else round(r['old_score'])} | {arrow(r.get('delta'))} {'' if r.get('delta') is None else (format(r['delta'], '+.0f'))} | " if compared else '')
            L.append(f"| {rank[r['cut']]} | {r['cut']} | {r['shot']} | **{sc}** | {cmp_}{'' if r.get('corr') is None else (format(r['corr'], '+.2f'))} | "
                     f"{'' if r.get('open_in_silence') is None else r['open_in_silence']} | {'' if r.get('mouth_range') is None else r['mouth_range']} | "
                     f"{'' if not r.get('tier') else (str(r['tier']) + ' ' + format(r.get('owner_px') or 0, '.0f'))} | "
                     f"{'' if r.get('mouth_visible') is None else (format(r['mouth_visible'], '.0%'))} | {'' if not r.get('canon_char') or r.get('canon_sim') is None else (str(r['canon_char']) + ' ' + format(r['canon_sim'], '.2f'))} | "
                     f"{r['motion']:.2f}{' STATIC' if r.get('static') else ''} | {r['pops']} | {r['reason']} |")
        L.append('')
    open(out + '.md', 'w').write('\n'.join(L) + '\n')
    sheet(rows, thumbs, out + '_sheet.jpg', compared)


def sheet(rows, thumbs, path, compared=False, cols=8):
    tiles = []
    for r in sorted(rows, key=lambda r: r['cut']):
        t = thumbs.get(r['cut'])
        if t is None: continue
        t = t.copy()
        s = r['score']
        cl = r.get('class', 'PERF')
        if cl == 'COVERAGE': col, sc = (150, 150, 150), 'n/a cover'
        elif cl == 'INSERT': col, sc = (200, 170, 120), 'insert ' + ('-' if s is None else f'{s:.0f}')
        else:
            col = (90, 90, 90) if s is None else (60, 200, 60) if s >= GREEN else (0, 170, 255) if s >= AMBER else (40, 40, 230)
            sc = '-' if s is None else f"{s:.0f}"
        t = cv2.copyMakeBorder(t, 6, 30, 6, 6, cv2.BORDER_CONSTANT, value=col)
        lab = f"{r['cut']} {r['shot']}"[:22]
        if compared and r.get('delta') is not None: sc += f" ({r['delta']:+.0f})"
        cv2.putText(t, lab, (8, t.shape[0] - 18), 0, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(t, sc, (8, t.shape[0] - 4), 0, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(t)
    if not tiles: return
    h, _w = max(x.shape[0] for x in tiles), tiles[0].shape[1]
    tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub scorecard')
    ap.add_argument('--video', required=True)
    ap.add_argument('--cuts', required=True)
    ap.add_argument('--vocal', required=True)
    ap.add_argument('--offset', type=float, default=12.0)
    ap.add_argument('--align', help='lyric alignment (sung cuts, sync target, closures)')
    ap.add_argument('--script', help='shooting script (singer ownership -> grammar tier / mouth / non-singer)')
    ap.add_argument('--preset', help='cast preset (presets.py), e.g. landing')
    ap.add_argument('--cast', help='whole-video cast scan of --video (identity, canon, faces, singer box)')
    ap.add_argument('--compare', help='an older cut of the same film')
    ap.add_argument('--compare-cast', help='cast scan of --compare')
    ap.add_argument('--out', required=True, help='output prefix: <out>.md/.json/.csv/_sheet.jpg')
    a = ap.parse_args(argv)
    cuts = json.load(open(a.cuts))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows, thumbs = score(a.video, cuts, a.vocal, a.offset, a.align, a.script, json.load(open(a.cast)) if a.cast else None, a.out + '_work', preset=a.preset)
    if a.compare:
        old, othumbs = score(a.compare, cuts, a.vocal, a.offset, a.align, a.script, json.load(open(a.compare_cast)) if a.compare_cast else None, a.out + '_old_work', preset=a.preset)
        write(old, othumbs, a.out + '_old', a.compare)
        rows = compare(rows, old)
    write(rows, thumbs, a.out, a.video, bool(a.compare))
    R = sorted([r for r in rows if r['score'] is not None and r.get('class') == 'PERF'], key=lambda r: r['score'])
    print({cl: sum(r.get('class') == cl for r in rows) for cl in CLASSES})
    print('PERF worst:', ', '.join(f"{r['cut']} {r['shot']} {r['score']:.0f}" for r in R[:8]))
    print('PERF best: ', ', '.join(f"{r['cut']} {r['shot']} {r['score']:.0f}" for r in R[-8:]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
