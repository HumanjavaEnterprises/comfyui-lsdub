# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v10a vocalisations (lsdub/core/vocalisations.py): the lexicon (real words never match), the lyrics markup, the span
list of a track (merged across a short silence, never across a word), `lsdub track skip`, the zero-phase ease, and the
warp consumer on a synthetic clip, CPU only: --vocalisations skip (the default since v11, on any track) leaves the span
as the take's own frames (no warp, no layers, no erase) with an eased edge; mouth -- and auto on an unmarked track -- is
the v9 render; auto on a marked track skips; rest drives the span to the rest band; the gate does not score the fit to the audio on a skipped
frame and never re-makes a frame that is the take."""
import json, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from lsdub_engine.core import vocalisations as vz, mouth_paint, mouth_warp as mw, frame_gate as fgm
from lsdub_engine.audits import frame_gate as fg


# ------------------------------------------------------------------------------------------------- the lexicon --
@pytest.mark.parametrize('tok', ['uh', 'Uh,', 'uh-huh', 'ah', 'aah', 'Aaaah!', 'oh', 'ooh', 'ooooh', 'oo', 'mm', 'mmm', 'mmmmm',
                                 'hmm', 'hey', 'heyyy', 'ha', 'hah', 'hahaha', 'la', 'la-la-la', 'lalala', 'na', 'nanana', 'da',
                                 'da-da', 'dadada', 'doo', 'whoa', 'woo', 'ey', 'eh', 'huh', 'uh-oh', 'mm-hmm', 'ooh-ooh'])
def test_vocalisations_match(tok):
    assert vz.token_class(tok) == 'v', tok


@pytest.mark.parametrize('tok', ['dance', 'day', 'dad', "don't", 'a', 'away', 'ohio', 'hat', 'they', 'la-di-da-day', 'oh-so-serious',
                                 'nana', 'dada', 'hum', 'humanity', 'heal', 'yea', 'say', 'na-me', 'uhm-what', "'em", 'to', 'do'])
def test_real_words_never_match(tok):
    assert vz.token_class(tok) is None, tok


def test_yeah_only_as_a_bare_adlib():
    occ = [dict(w=w, t0=t0, t1=t1) for w, t0, t1 in [('oh', 0.0, 0.3), ('yeah', 0.32, 0.6), ('I', 1.5, 1.6), ('know', 1.6, 1.9),
                                                     ('yeah', 1.92, 2.2), ('you', 2.21, 2.4), ('yeah', 4.0, 4.3)]]
    vz.classify(occ)
    assert [o['vocal'] for o in occ] == [True, True, False, False, False, False, True]
    assert occ[1]['reason'] == 'yeah:bare' and occ[6]['reason'] == 'yeah:bare' and occ[0]['reason'] == 'lexicon:oh'


def test_skip_words_extend_the_lexicon(tmp_path):
    p = tmp_path / 'w.txt'
    p.write_text('# ad-libs of this song\nskrrt\nwop\n')
    occ = [dict(w=w, t0=k, t1=k + 0.5) for k, w in enumerate(['skrrt', 'doo-wop', 'wop', 'wopper'])]
    vz.classify(occ, vz.read_words(str(p)))
    assert [o['vocal'] for o in occ] == [True, True, True, False]
    assert occ[0]['reason'] == 'skip-words:skrrt' and occ[1]['reason'] == 'skip-words:doo-wop'


# ---------------------------------------------------------------------------------------------- lyrics markup --
def test_lyric_markup_parens_and_skip_tag():
    text = '[Verse 1]\nFuture focus (ooh ooh), lights\n[skip] Get down get down\n(yeah yeah)\nHey you (x2)\n'
    m = vz.lyric_marks(text)
    assert m == [('future', None), ('focus', None), ('ooh', 'paren'), ('ooh', 'paren'), ('lights', None), ('get', 'tag'), ('down', 'tag'),
                 ('get', 'tag'), ('down', 'tag'), ('yeah', 'paren'), ('yeah', 'paren'), ('hey', None), ('you', None)]


def test_markup_maps_onto_the_sung_words():
    marks = vz.lyric_marks('we sent a gold record\n(get down get down)\nmusically refined')
    sung = ['we', 'sent', 'a', 'gold', 'record', 'get', 'down', 'get', 'get', 'down', 'musically', 'refined', 'tonight']
    f = vz.markup_flags(sung, marks)
    assert f[:5] == [None] * 5 and f[5:10] == ['paren'] * 5 and f[10:] == [None] * 3     # the extra 'get' is inside the span


# ----------------------------------------------------------------------------------------------- the track --
def track(words, fps=25.0, t0=10.0):
    """a viseme track whose frames carry `words` (one per frame, '' = silence)."""
    return dict(fps=fps, t0=t0, frames=[dict(t=round(t0 + k / fps, 3), vis='aa' if w else 'sil', vis2='sil', mix=0.0, open=0.8 if w else 0.0,
                                             word=w, phone='AA' if w else '') for k, w in enumerate(words)])


def test_spans_merge_across_a_short_silence_never_across_a_word():
    w = ['dance'] * 5 + ['da'] * 4 + [''] * 6 + ['da'] * 3 + ['dance'] * 4 + ['da'] * 3 + [''] * 30 + ['uh'] * 2 + ['never'] * 3
    f = vz.find_spans(track(w))
    assert [(s['f0'], s['f1']) for s in f['spans']] == [(5, 17), (22, 24), (55, 56)]      # 6 silent frames (0.24 s) merged
    assert f['frames'][10] == 'gap' and f['frames'][5] == 'lexicon:da' and f['frames'][55] == 'lexicon:uh'
    assert 4 not in f['frames'] and 18 not in f['frames'] and 25 not in f['frames']
    f2 = vz.find_spans(track(w), merge_gap=0.2)                                             # 0.24 s > 0.2: not merged
    assert [(s['f0'], s['f1']) for s in f2['spans']][:2] == [(5, 8), (15, 17)]


def test_spans_use_the_alignment_words():
    # the alignment knows each 'da' separately and the sung-only source; the track only has merged frame words
    A = dict(words=[dict(w='dance', start=10.0, end=10.2, src='lyric', phones=[dict(p='D', start=10.0, end=10.2)]),
                    dict(w='da', start=10.2, end=10.36, src='sung-only', phones=[dict(p='D', start=10.2, end=10.36)]),
                    dict(w='da', start=10.36, end=10.52, src='sung-only', phones=[dict(p='AA', start=10.36, end=10.52)]),
                    dict(w='yeah', start=10.8, end=11.0, src='sung', phones=[dict(p='Y', start=10.8, end=11.0)])])
    T = track(['dance'] * 5 + ['da'] * 8 + [''] * 7 + ['yeah'] * 5 + [''] * 5)
    f = vz.find_spans(T, align=A)
    assert all(o['ai'] is not None for o in f['occ']) and len(f['occ']) == 4
    assert [(s['f0'], s['f1'], s['words']) for s in f['spans']] == [(5, 24, ['da', 'da', 'yeah'])]
    assert f['frames'][20] == 'yeah:bare' and f['frames'][15] == 'gap'


def test_track_skip_cli_writes_the_fields(tmp_path, capsys):
    T = track(['so'] * 4 + ['ooh'] * 6 + ['serious'] * 4)
    p = tmp_path / 't.json'
    json.dump(T, open(p, 'w'))
    out = tmp_path / 'o.json'
    assert vz.main(['skip', '--track', str(p), '--out', str(out)]) == 0
    T2 = json.load(open(out))
    txt = capsys.readouterr().out
    assert T2['skips'] == [dict(f0=4, f1=9, t0=10.16, t1=10.4, words=['ooh'], reasons=['lexicon:ooh'])]
    assert [k for k, fr in enumerate(T2['frames']) if fr.get('skip')] == list(range(4, 10)) and T2['frames'][4]['skip_reason'] == 'lexicon:ooh'
    assert 'f4-9' in txt and '10.16' in txt and 'ooh' in txt
    assert vz.from_track(T2)['frames'] == vz.find_spans(T)['frames']           # the consumer reads back what was written
    # a processed track with nothing found: an empty list (processed), no frame flagged
    p2 = tmp_path / 'n.json'
    json.dump(track(['so', 'serious']), open(p2, 'w'))
    vz.main(['skip', '--track', str(p2), '--out', str(p2)])
    assert json.load(open(p2))['skips'] == []


def test_ease_is_zero_phase():
    m = np.zeros(20, bool)
    m[6:13] = True
    s = vz.strength(m, 3)
    assert np.all(s[m][1:-1] == 0) and np.all(s[:4] == 1) and np.all(s[15:] == 1)
    assert np.allclose(s[[5, 6]], [0.790, 0.210], atol=1e-3) and np.allclose(s[[12, 13]], [0.210, 0.790], atol=1e-3)
    m2 = np.zeros(21, bool)
    m2[7:14] = True
    s2 = vz.strength(m2, 3)
    assert np.allclose(s2, s2[::-1])   # symmetric: no lag
    for e in (2, 4, 5):
        s3 = vz.strength(m2, e)
        assert np.allclose(s3, s3[::-1]) and np.all(np.diff(s3[:11]) <= 1e-12)
    assert np.array_equal(vz.strength(m2, 1), (~m2).astype(float))


# ------------------------------------------------------------------------------------- the warp consumer --
from test_frame_gate_maker import detect
from test_frame_gate import face, draw


def make_take(tmp, n=30):
    """a take that sings its own words: open at the ends, SHUT over frames 6..23 (so a target 'aa' there needs the warp)."""
    import subprocess
    gaps = np.where((np.arange(n) >= 6) & (np.arange(n) < 24), 1.2, 6.0)
    gaps = np.convolve(np.pad(gaps, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
    S = np.array([face(g) for g in gaps])
    frames = [draw(P) for P in S]
    vid = str(tmp / 'take.mp4')
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '320x300', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', vid], stdin=subprocess.PIPE)
    for f in frames: pr.stdin.write(f.tobytes())
    pr.stdin.close()
    pr.wait()
    np.savez(str(tmp / 'trk.npz'), S3=S, fps=25.0, found=1.0, n=n, video=vid)
    json.dump(dict(frames=[dict(vis='aa', vis2='aa', mix=0.0, open=0.9) for _ in range(n)]), open(str(tmp / 'vis.json'), 'w'))
    return vid, frames, S


def paint_args(tmp, vid, n, track_path, extra=()):
    cap = []
    run = mw.run
    mw.run = lambda a: cap.append(a)
    try:
        mouth_paint.main(['--mode', 'warp', '--video', vid, '--track', str(track_path), '--frames', str(n), '--out', str(tmp / 'o.mp4'),
                          '--warp-track-cache', str(tmp / 'trk.npz'), '--warp-json', str(tmp / 'w.json'), '--mask-out', str(tmp / 'm.npz')] + list(extra))
    finally:
        mw.run = run
    return cap[0]


@pytest.fixture(scope='module')
def vclip(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('voc')
    vid, frames, S = make_take(tmp, n=30)
    T = json.load(open(tmp / 'vis.json'))                         # every frame asks for an open 'aa' over a mostly shut take
    words = ['dance'] * 10 + ['da'] * 10 + ['dance'] * 10
    for f, w in zip(T['frames'], words): f['word'] = w
    json.dump(T, open(tmp / 'vis_w.json', 'w'))
    T0 = json.load(open(tmp / 'vis.json'))
    json.dump(T0, open(tmp / 'vis_nw.json', 'w'))   # the same visemes, no words
    return tmp, vid, frames


def run_clip(a):
    """mouth_warp.run up to the write, returning the clip (the hooks of run() included)."""
    vx = vz.for_warp(a)
    clip = mw.WarpClip(vx.args if vx else a)
    if vx: vx.before_batch(clip)
    clip.run_batch()
    if vx: vx.after_batch(clip)
    return clip, vx


def test_skip_leaves_the_take_and_eases_the_edges(vclip):
    tmp, vid, frames = vclip
    n = len(frames)
    clip, vx = run_clip(paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'skip']))
    assert vx is not None and vx.mode == 'skip' and [(s['f0'], s['f1']) for s in vx.found['spans']] == [(10, 19)]
    take = clip.frames
    s = clip.vocal_s
    for i in range(11, 19):                                       # strength 0: the take's own frame, exactly
        assert s[i] == 0 and np.array_equal(clip.outs[i], take[i]), i
        assert np.allclose(clip.dFe[i], 0) and not np.any(clip.masks[i])
    assert 0 < s[10] < 0.5 < s[9] < 1 and 0 < s[19] < 0.5 < s[20] < 1   # eased: 0.21 inside, 0.79 outside, both edges
    d = lambda i: float(np.abs(clip.outs[i].astype(int) - take[i].astype(int)).mean())
    assert d(7) > d(9) > d(10) > 0 and d(22) > d(20) > d(19) > 0 # the change toward the take is monotone across an edge
    # the per-frame render (the gate's maker) gives the same eased frames
    for i in (9, 10, 14, 19, 20):
        out, mask, ent, P = clip.render_frame(i, {})
        assert np.array_equal(out, clip.outs[i]) and ent['vocal_s'] == round(float(s[i]), 4), i


def test_mouth_and_auto_on_an_unmarked_track_are_v9(vclip, capsys):
    tmp, vid, frames = vclip
    n = len(frames)
    base, _ = run_clip(paint_args(tmp, vid, n, tmp / 'vis_nw.json', ['--vocalisations', 'mouth']))
    a_m = paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'mouth'])
    assert vz.for_warp(a_m) is None
    mo, _ = run_clip(a_m)
    a_auto = paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'auto'])
    au, vx = run_clip(a_auto)
    assert vx is None                          # auto (v10a's default) on an UNMARKED track: v9 ...
    assert 'f10-19' in capsys.readouterr().out                             # ... with a hint naming the mouthed span
    sk, vx = run_clip(paint_args(tmp, vid, n, tmp / 'vis_nw.json'))
    assert vx is None   # the default, nothing to skip: v9
    for i in range(n):
        assert np.array_equal(mo.outs[i], base.outs[i]) and np.array_equal(au.outs[i], base.outs[i]) and np.array_equal(sk.outs[i], base.outs[i]), i
    assert not np.array_equal(mo.outs[14], frames[14])                     # (mouth: the 'da' frames are warped open)


def test_the_default_skips_the_vocalisations_on_any_track(vclip):
    # v11: the platform skips ad-libs by default -- on an unmarked track (detected at paint time) and on a marked one alike;
    # --vocalisations mouth is the way back to v9
    tmp, vid, frames = vclip
    n = len(frames)
    a = paint_args(tmp, vid, n, tmp / 'vis_w.json')
    assert a.vocalisations == 'skip'
    d, vx = run_clip(a)
    assert vx is not None and vx.mode == 'skip' and [(s['f0'], s['f1']) for s in vx.found['spans']] == [(10, 19)]
    sk, _ = run_clip(paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'skip']))
    mo, _ = run_clip(paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'mouth']))
    for i in range(n):
        assert np.array_equal(d.outs[i], sk.outs[i]), i                    # the default IS the explicit skip
    for i in range(11, 19):
        assert np.array_equal(d.outs[i], d.frames[i]) and not np.array_equal(mo.outs[i], mo.frames[i]), i   # the take's own mouth vs mouthed
    assert vz.main(['skip', '--track', str(tmp / 'vis_w.json'), '--out', str(tmp / 'vis_k2.json')]) == 0
    k, vx = run_clip(paint_args(tmp, vid, n, tmp / 'vis_k2.json'))
    assert vx is not None and 'skips' in vx.T
    for i in range(n):
        assert np.array_equal(k.outs[i], d.outs[i]), i                     # marked or detected at paint time: the same render


def test_auto_skips_a_marked_track_and_mouth_ignores_the_marks(vclip):
    tmp, vid, frames = vclip
    n = len(frames)
    assert vz.main(['skip', '--track', str(tmp / 'vis_w.json'), '--out', str(tmp / 'vis_k.json')]) == 0
    k, vx = run_clip(paint_args(tmp, vid, n, tmp / 'vis_k.json', ['--vocalisations', 'auto']))   # marked track, auto: skip, from the track's fields
    assert vx is not None and vx.mode == 'skip' and 'skips' in vx.T
    s_, _ = run_clip(paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'skip']))   # detected at paint: the same
    m_, _ = run_clip(paint_args(tmp, vid, n, tmp / 'vis_k.json', ['--vocalisations', 'mouth']))  # the marks change nothing in v9
    base, _ = run_clip(paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'mouth']))
    for i in range(n):
        assert np.array_equal(k.outs[i], s_.outs[i]) and np.array_equal(m_.outs[i], base.outs[i]), i


def test_rest_drives_the_span_to_the_rest_band(vclip):
    tmp, vid, frames = vclip
    n = len(frames)
    a = paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'rest'])
    clip, vx = run_clip(a)
    assert vx.mode == 'rest' and vx.args.track != a.track and os.path.exists(vx.args.track)
    r = clip.model.rng
    u = (clip.Fe[:, 0] - r['g_lo']) / (r['g_hi'] - r['g_lo'])
    assert np.all(u[12:18] <= vx.rest['band'][1] + 0.02), u[12:18]         # inside the rest band (0 .. 0.15)
    assert np.all(u[2:7] > 0.5)                                            # the words still open the mouth
    vx.finish(clip, None)
    assert not os.path.exists(vx.args.track)        # the re-labelled track is a temp file


def test_paint_writes_the_vocal_block(vclip):
    tmp, vid, frames = vclip
    n = len(frames)
    a = paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--vocalisations', 'skip'])
    mw.run(a)
    w = json.load(open(a.warp_json))
    assert w['vocal']['mode'] == 'skip' and w['vocal']['frames_take'] == 8 and w['vocal']['frames_eased'] == 4
    assert w['frames'][14]['vocal_skip'] and w['frames'][14]['vocal_s'] == 0 and 'vocal_s' not in w['frames'][3]
    rp = fg.replay(w, np.load(str(tmp / 'trk.npz'))['S3'], frames[0].shape[:2])
    assert rp['skip'][14] and not rp['skip'][3]                            # the offline checker sees the skipped frames too


def test_gate_does_not_score_the_audio_fit_on_skipped_frames(vclip):
    tmp, vid, frames = vclip
    n = len(frames)
    a = paint_args(tmp, vid, n, tmp / 'vis_w.json', ['--warp-gate', 'repair', '--vocalisations', 'skip'])
    clip, vx = run_clip(a)
    plan = fgm.plan_of(clip, fold=False)
    assert plan['skip'] is not None and plan['skip'][14] and not plan['skip'][3]
    ck = fgm.make_checker(clip, a, lmk=detect)
    M = ck.metrics()
    v = np.asarray(M['viseme'], float)
    assert np.all(np.isnan(v[10:20])) and np.isfinite(v[:10]).sum() >= 8
    assert fgm.is_take(clip, 14) and not fgm.is_take(clip, 10) and not fgm.is_take(clip, 3)
    # a frame that IS the take is observed as the take is (its own lips, no plan outline): no ghost / leak excess by construction
    assert ck._plan_P(14) is None and ck._plan_P(10) is not None and ck._plan_P(3) is not None
    for i in range(11, 19):
        assert ck.obs[i]['ghost'] == ck.obs_k[i]['ghost'] and ck.obs[i]['leak'] == ck.obs_k[i]['leak'], i
        assert np.nan_to_num(M['leak'][i]) == 0 and np.nan_to_num(M['ghost'][i]) == 0, i
    res = fgm.gate(clip, a, lmk=detect)['gate']
    remade = {t for r in res['repairs'] for t in r.get('remade', [])}
    assert not (remade & set(range(11, 19)))                               # a frame that is the take is never re-made
    for i in range(11, 19): assert np.array_equal(clip.outs[i], clip.frames[i]), i


def test_cli_knows_track():
    from lsdub_engine import cli
    assert cli.CMDS['track'] == 'core.vocalisations' and 'lsdub track' in cli.__doc__
