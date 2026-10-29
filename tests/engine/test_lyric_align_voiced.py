# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lyric_align on a voice with gaps (CPU, no models): the fallback spread closes the mouth over an unvoiced gap, a lyric
match of 0 still places the lyrics on the voice, uncovered singing gets the unheard line or vocalisations, and the
Align node warns / raises on a low lyric match; LSDub Dub passes its vocalisations mode to the warp."""
import numpy as np
import pytest

from lsdub_engine.core import lyric_align as LA


def env_of(*parts):
    """10 ms dB frames: ('v', s) voiced, ('u', s) unvoiced."""
    return np.concatenate([np.full(int(round(s * 100)), -10.0 if k == 'v' else -90.0) for k, s in parts])


def covered(phones, t):
    return any(p['start'] <= t < p['end'] for p in phones)


def test_spread_closes_the_mouth_over_a_gap():
    env = env_of(('v', 0.5), ('u', 0.2), ('v', 0.5))                    # a 200 ms gap at 0.5-0.7 s
    w = {'w': 'away', 'start': 0.0, 'end': 1.2}
    ph = LA.spread_voiced(w, ['AH', 'W', 'EY'], env, LA.voiced_thr(env), 0.0)
    assert not any(covered(ph, t) for t in (0.52, 0.6, 0.68))           # no phone in the gap: the mouth closes
    assert covered(ph, 0.1) and covered(ph, 1.1)
    assert all(p['end'] <= 0.5 + 1e-6 or p['start'] >= 0.7 - 1e-6 for p in ph)
    old = LA.spread(w, ['AH', 'W', 'EY'])                                # the old spread held a phone across it
    assert covered(old, 0.6)


def test_a_short_dip_does_not_close_the_mouth():
    env = env_of(('v', 0.5), ('u', 0.05), ('v', 0.5))                   # 50 ms < GAP_CLOSE: bridged
    ph = LA.spread_voiced({'w': 'oh', 'start': 0.0, 'end': 1.05}, ['OW'], env, LA.voiced_thr(env), 0.0)
    assert len(ph) == 1 and ph[0]['start'] == 0.0 and ph[0]['end'] == pytest.approx(1.05)


def test_ratio_zero_still_places_the_lyrics_on_the_voice():
    env = env_of(('u', 0.5), ('v', 1.0), ('u', 0.5), ('v', 1.0), ('u', 0.3))
    lines = [['fly', 'me'], ['to', 'the', 'moon']]
    W = LA.place_lyrics(lines, env, 10.0, LA.voiced_thr(env))
    assert [w['w'] for w in W] == ['fly', 'me', 'to', 'the', 'moon']
    assert W[0]['start'] == pytest.approx(10.5) and W[-1]['end'] == pytest.approx(13.0)
    assert all(w['align'] == 'voiced-place' for w in W)
    assert all(b['start'] >= a['end'] - 1e-6 for a, b in zip(W, W[1:]))  # in order, no overlap


def test_unheard_line_and_scat_fill_the_uncovered_singing():
    env = env_of(('v', 1.0), ('v', 1.0), ('u', 0.3), ('v', 1.0))
    lines = [['hello', 'you'], ['why', 'try']]
    final = [{'w': 'hello', 'start': 0.0, 'end': 0.5, 'phones': []}, {'w': 'you', 'start': 0.5, 'end': 1.0, 'phones': []}]
    added = LA.fill_voiced(final, lines, env, LA.voiced_thr(env), 0.0, 3.3)
    words = [(w['w'], w['src']) for w in added]
    assert ('why', 'lyric') in words and ('try', 'lyric') in words
    assert any(w == 'da' and s == 'vocalisation' for w, s in words)        # the other stretch: a vocalisation
    assert all(w['align'] == 'voiced-fill' and w['phones'] for w in added)
    assert all(b['start'] >= a['start'] for a, b in zip(final, final[1:]))


def test_letter_phones_keep_the_closures():
    assert LA.letter_phones('pleiadians')[0] == 'P' and 'N' in LA.letter_phones('pleiadians')
    assert LA.letter_phones('zzz') == ['Z']


def test_align_node_warns_and_raises_on_a_low_match(pack):
    A = pack.nodes_audio
    assert A.check_match(0.9, True) == '' and A.check_match(0.0, False) == ''
    assert 'placed over the voiced' in A.check_match(0.35, True)
    with pytest.raises(RuntimeError, match='match what is sung'):
        A.check_match(0.1, True)


def test_lyric_prompt(pack, tmp_path):
    p = tmp_path / 'l.txt'
    p.write_text('[Chorus]\nFly me\n\nTo the moon!\n')
    assert pack.nodes_audio.lyric_prompt(str(p)) == 'Fly me. To the moon!'


def test_dub_passes_the_vocalisations_mode(pack, tmp_path, monkeypatch):
    M, T, P = pack.nodes_mouth, pack.lsdub_types, pack.project
    seen = {}
    import importlib
    sing = importlib.import_module(pack.__name__ + '.lsdub_engine.core.sing')
    continuity = importlib.import_module(pack.__name__ + '.lsdub_engine.core.continuity')

    def fake_run(take, align, vocal, t0, out, work, pa, **kw):
        seen['pa'] = list(pa)
        open(out, 'w').write('x')
        return {'pass': True, 'corr': 0.9}
    monkeypatch.setattr(sing, 'run', fake_run)
    monkeypatch.setattr(continuity, 'run', lambda take, sung, render, report=None: open(render, 'w').write('x'))
    monkeypatch.setattr(M, 'extract_shot', lambda src, f0, f1, out: out)
    p = P.Project(root=str(tmp_path), name='cut').ensure()
    s = {'shot': 1, 'f0': 0, 'f1': 9, 'start': 0.0, 'verdict': 'DUB', 'target': 'a', 'box': [0, 0, 10, 10]}
    st = p.shot(1, 'k')
    M.LSDubDub.run_shot(p, T.Source(path='v.mp4'), s, T.Song(mix='m.wav'), T.Align(align='a.json'),
                        T.Kit(dir=str(tmp_path), identity='a'), M.DEFAULT_TWEAKS, None, None, st, 'mouth')
    assert seen['pa'][-2:] == ['--vocalisations', 'mouth']
    sch = M.LSDubDub.define_schema()
    c = next(i for i in sch.inputs if i.id == 'vocalisations')
    assert c.kw['options'] == ['skip', 'rest', 'mouth'] and c.kw['default'] == 'skip'
    k = lambda v: M.LSDubDub._shot_key(p, T.Source(path='v.mp4', hash='h'), s, T.Song(mix='m'), T.Align(align='a', hash='a'),
                                       T.Kit(dir='d', identity='a', hash='k'), M.DEFAULT_TWEAKS, None, None, v)
    assert k('mouth') != k('skip') and k('skip') == k('skip')
