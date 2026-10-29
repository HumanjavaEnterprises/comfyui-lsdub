"""Tests for the classical-animation stylized mouth rules (docs/MOUTH-ANIMATION-INSIGHTS.md)."""
from lsdub_engine import stylized as st


def _runs(an):
    out, cur, c = [], an[0], 1
    for a in an[1:]:
        if a == cur:
            c += 1
        else:
            out.append((cur, c))
            cur, c = a, 1
    out.append((cur, c))
    return out


def _track(seq, opens=None):
    opens = opens or [st.OPENNESS[st.VIS2ANCHOR.get(v, 'rest')] for v in seq]
    return [{'vis': v, 'vis2': v, 'mix': 0, 'open': o} for v, o in zip(seq, opens)]


def test_phoneme_map():
    assert st.phoneme_to_anchor('M') == 'closed'
    assert st.phoneme_to_anchor('AA1') == 'ah'      # stress digit stripped
    assert st.phoneme_to_anchor('F') == 'fv'
    assert st.phoneme_to_anchor('L') == 'L'
    assert st.phoneme_to_anchor('UW0') == 'oo'
    assert st.phoneme_to_anchor('ZZZ') == 'cons'    # unknown -> nearest open consonant


def test_preset_collapse():
    # southpark set has no fv/L/cons/ai; they must remap into the 4+rest set
    for a in ('fv', 'L', 'cons', 'ai'):
        assert st._nearest_in(a, st.PRESETS['southpark']) in st.PRESETS['southpark']
    # crude = Rhubarb 6
    assert st.PRESETS['crude'] == {'closed', 'cons', 'ee', 'ah', 'oh', 'oo'}


def test_min_hold_and_closure_hold():
    frames = _track(['sil', 'aa', 'aa', 'aa', 'PP', 'PP', 'oh', 'oh', 'sil', 'sil'])
    s = st.build_stylized_track(frames, fps=24, preset='classic')
    runs = _runs([x['anchor'] for x in s])
    # every run except possibly the trailing silence holds >= 2 frames (on twos)
    assert all(ln >= 2 for a, ln in runs[:-1])
    # a closure is present and held >= 2 frames (held-then-popped, never a 1-frame flap)
    assert any(a == 'closed' and ln >= 2 for a, ln in runs)


def test_snap_no_mix():
    # stylized frames carry a single anchor/rep (snap) — there is no vis2/mix cross-fade field
    s = st.build_stylized_track(_track(['aa', 'aa', 'oh', 'oh']), fps=24)
    for fr in s:
        assert set(fr) == {'anchor', 'rep', 'open', 'sq_w', 'sq_h', 'accent'}
        assert fr['rep'] in st.ANCHOR_REP.values()


def test_lead_shape_appears_before_sound():
    # a vowel whose audio starts at frame 4 should have its shape visible a frame or two earlier
    frames = _track(['sil', 'sil', 'sil', 'sil', 'aa', 'aa', 'aa', 'aa'])
    s = st.build_stylized_track(frames, fps=24, preset='classic', min_hold=1)
    an = [x['anchor'] for x in s]
    assert an[4] == 'ah'
    assert 'ah' in an[2:4]          # appears 1-2 frames before the sound, never after


def test_accent_on_vowel_onset():
    frames = _track(['PP', 'PP', 'aa', 'aa', 'aa'])
    s = st.build_stylized_track(frames, fps=24, preset='classic')
    # the first vowel frame after the closure is accented and opens wider than its nominal
    onset = next(i for i, x in enumerate(s) if x['anchor'] == 'ah')
    assert s[onset]['accent'] is True


def test_squash_volume_conserving():
    # open vowel narrows horizontally (sq_w < 1); closed widens slightly (sq_w >= 1)
    s = st.build_stylized_track(_track(['aa'] * 4 + ['PP'] * 4), fps=24)
    wide_open = [x for x in s if x['anchor'] == 'ah'][0]
    closed = [x for x in s if x['anchor'] == 'closed'][0]
    assert wide_open['sq_w'] < 1.0 < wide_open['sq_h']
    assert closed['sq_w'] >= 1.0


def test_singing_holds_vowel_across_gap():
    # a short rest gap between two identical held vowels fills with the vowel when singing
    frames = _track(['aa'] * 3 + ['sil'] * 2 + ['aa'] * 3)
    plain = [x['anchor'] for x in st.build_stylized_track(frames, fps=24, singing=False)]
    sung = [x['anchor'] for x in st.build_stylized_track(frames, fps=24, singing=True)]
    assert sung.count('ah') > plain.count('ah')


def test_puppet_gape_fractions():
    # rest/closed = 0 gape; open vowel reaches toward 1.0; the hinge is openness-only
    frames = _track(['sil', 'PP', 'aa', 'aa', 'aa'])
    g = st.puppet_gape_track(frames, fps=24, accent=False)
    assert g[0] == 0.0 and g[1] == 0.0
    assert max(g) > 0.6
    assert st.GAPE['closed'] == 0.0 and st.GAPE['ah'] == 1.0 and st.GAPE['ee'] < st.GAPE['oh']


def test_puppet_one_open_per_syllable():
    # an intervocalic consonant must not re-open the jaw above the surrounding vowel floor
    frames = _track(['aa', 'aa', 'DD', 'aa', 'aa'])
    g = st.puppet_gape_track(frames, fps=24, accent=False)
    assert g[2] <= max(g[1], g[3]) + 1e-6
