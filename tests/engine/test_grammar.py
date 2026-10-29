"""lsdub grammar: tiers, hook matching, script ownership/inheritance, the per-cut verdicts and fixes (synthetic, no models)."""
from lsdub_engine.core import grammar as G
from lsdub_engine import presets as P

SCRIPT = """### 5.1 Verse
| # | v | s | Lyric | Story beat | Shot | Cam / framing | Motion | Status |
|---|---|---|---|---|---|---|---|---|
| 0 | 0-1 | 0 | "x" | **Red sings** to the lens | A | CU | i2v | SING |
| 1 | 1-2 | 1 | (cont.) | The crowd | B | wide | i2v | keep |
| 2 | 2-3 | 2 | "y" | Sky sings at the decks | C | MS | i2v | keep |
### 5.2 Chorus
| # | v | s | Lyric | Story beat | Shot | Cam / framing | Motion | Status |
|---|---|---|---|---|---|---|---|---|
| 3 | 3-4 | 3 | "hook" | Lasers | D | wide | i2v | keep |
| 4 | 4-5 | 4 | (none) | Lasers | E | wide | i2v | keep |
"""


def _cast(tracks, occluded=()):
    """tracks: [(character, f0, f1, px)] on one whole video (25 fps); occluded: frames whose mouth is hidden."""
    return {'results': [{'video': 'whole.mp4', 'size': [1920, 1080],
                         'tracks': [{'character': c, 'frames': [{'f': f, 'face_px': px, 'landmarks': True, 'mouth_occluded': f in occluded}
                                                                for f in range(a, b + 1)]} for c, a, b, px in tracks]}]}


def _cut(n=5):
    return {'cut': [{'shot': s, 'start': float(i), 'dur': 1.0} for i, s in enumerate('ABCDE'[:n])]}


def _w(w, a, b):
    return {'w': w, 'start': a, 'end': b}


def test_tiers_and_hook_words():
    assert [G.tier(p, 300) for p in (300, 299, 120, 119, 25, 24, 0)] == ['cu', 'medium', 'medium', 'wide', 'wide', 'none', 'none']
    assert G.CU_PX == 240 and G.tier(252) == 'cu' and G.tier(239) == 'medium' and G.WARP_GATE_1080 == 300
    W = [_w(x, 0, 1) for x in "oh Pleiadians DON'T dance, why don’t they try".split()]
    assert G.hook_words(W, P.LANDING.hooks) == {1, 2, 3, 4, 5, 6, 7}


def test_script_beats_solo_section_and_inheritance():
    B = G.script_beats(SCRIPT, 'landing')
    assert B[0]['solo'] and B[0]['singers'] == {'red_nova'} and not B[2]['solo'] and B[2]['singers'] == {'sky_lux'}
    O = G.owners(B, 5, P.LANDING.singers)
    assert O[1] == ({'red_nova'}, 'inherited')                      # (cont.) in the same section
    assert O[3] == ({'red_nova', 'sky_lux'}, 'any_singer')          # new section: no carry-over


def test_grammar_verdicts():
    B = G.script_beats(SCRIPT, 'landing')
    words = [_w('love', 0.1, 0.5),                     # cut 0: solo, Red CU 350 -> PASS
             _w('more', 1.1, 1.5),                     # cut 1: verse, Red wide 80 -> FLAG (needs medium)
             _w('beat', 2.1, 2.5),                     # cut 2: verse, Sky medium 150, but Merope 400 is the largest -> FLAG nonsinger
             _w('pleiadians', 3.0, 3.3), _w("don't", 3.3, 3.5), _w('dance', 3.5, 3.9),   # cut 3: hook on a medium -> FLAG (needs CU)
             _w('ooh', 4.0, 4.04), _w('aaah', 4.04, 4.96)]  # cut 4: 1 frame (ignored) + a held note: attack frames need medium
    cast = _cast([('red_nova', 0, 24, 350), ('red_nova', 25, 49, 80), ('sky_lux', 50, 74, 150), ('merope', 50, 74, 400),
                  ('sky_lux', 75, 99, 200), ('red_nova', 100, 124, 130)])
    R = G.grammar(cast, _cut(), words, B, song_offset=0.0, preset='landing')
    v = {r['cut']: r for r in R}
    assert v[0]['verdict'] == 'PASS' and v[0]['need'] == 'cu' and v[0]['method'].startswith('tier-1')
    assert v[1]['verdict'] == 'FLAG' and v[1]['tier'] == 'wide' and 'medium' in v[1]['fix']
    assert v[2]['n_nonsinger'] == 1 and 'pleiadian' in v[2]['fix'] and v[2]['n_tier_fail'] == 0
    assert v[3]['need'] == 'cu' and v[3]['tier'] == 'medium' and 'CU' in v[3]['fix']
    assert [x['w'] for x in v[4]['word_checks']] == ['aaah'] and v[4]['word_checks'][0]['held_tail']
    assert v[4]['verdict'] == 'PASS' and v[4]['method'].startswith('tier-2')
    S = G.summary(R)
    assert S['flag'] == 3 and S['pass'] == 2 and S['words_nonsinger_largest'] == 1


def test_faceless_insert_and_px_basis():
    B = G.script_beats(SCRIPT, 'landing')
    R = G.grammar(_cast([]), _cut(), [_w('more', 1.1, 1.5)], B, song_offset=0.0, preset='landing')
    assert R[1]['verdict'] == 'FLAG' and R[1]['fix'].startswith('insert / cutaway')
    # native basis on a 768-tall frame: 400 @1080 -> 284 native -> medium
    R = G.grammar(_cast([('red_nova', 0, 24, 400)]), _cut(), [_w('love', 0.1, 0.5)], B, song_offset=0.0, px_scale=768 / 1080, cu_px=300, preset='landing')
    assert R[0]['tier'] == 'medium' and R[0]['verdict'] == 'FLAG'
    md = G.to_md(R, G.summary(R))
    assert '| 0 | A |' in md and '**FLAG**' in md


def test_held_tail_crossing_a_cut_is_held_and_judged_by_its_singer():
    """checker #1: 'aaah' 1.5-2.9 s starts on cut 1 (Red 150 px) and is held into cut 2 (Sky's cut, Sky 60 px).
    The tail in cut 2 has no attack: it is a held note (need none) and the largest-face check uses Red, the word's singer."""
    B = G.script_beats(SCRIPT, 'landing')
    R = G.grammar(_cast([('red_nova', 25, 49, 150), ('sky_lux', 50, 74, 60)]), _cut(), [_w('aaah', 1.5, 2.9)], B, song_offset=0.0, preset='landing')
    w2 = R[2]['word_checks'][0]
    assert w2['class'] == 'held' and w2['need'] == 'none' and w2['singer'] == ['red_nova']
    assert R[1]['verdict'] == 'PASS' and R[2]['verdict'] == 'FLAG' and w2['largest_nonsinger'] == 'sky_lux'
    assert R[2]['n_tier_fail'] == 0


def test_missing_cut_keeps_section_inheritance():
    """checker #4: a cut absent from the script does not reset the section's singers."""
    B = {0: {'singers': {'red_nova'}, 'solo': False, 'section': 'V'}, 2: {'singers': set(), 'solo': False, 'section': 'V'}}
    O = G.owners(B, 3, P.LANDING.singers)
    assert O[1] == ({'red_nova'}, 'inherited') and O[2] == ({'red_nova'}, 'inherited')


def test_hidden_mouth_flags_a_big_face():
    """QA #6 (D11): a 490 px singer face whose mouth is behind the mic on most sung frames is not a pass."""
    B = G.script_beats(SCRIPT, 'landing')
    words = [_w('love', 0.1, 0.9)]
    ok = G.grammar(_cast([('red_nova', 0, 24, 490)]), _cut(), words, B, song_offset=0.0, preset='landing')
    bad = G.grammar(_cast([('red_nova', 0, 24, 490)], occluded=set(range(0, 18))), _cut(), words, B, song_offset=0.0, preset='landing')
    assert ok[0]['verdict'] == 'PASS'
    assert bad[0]['verdict'] == 'FLAG' and bad[0]['n_mouth_hidden'] == 1 and 'mouth hidden' in bad[0]['fix']


def test_notes_add_up_and_cut_px_matches_its_tier():
    """QA #7: a cutaway note never hides the largest-face note. QA #8: the cut's px is the px of the word that sets its tier.
    QA #9: delegates are named by family."""
    B = G.script_beats(SCRIPT, 'landing')
    # cut 1 (Red owns, inherited): no Red face; Alcyone's 400 px face on the first half of the word, nothing after
    R = G.grammar(_cast([('alcyone', 25, 32, 400)]), _cut(), [_w('more', 1.0, 1.6)], B, song_offset=0.0, preset='landing')
    assert 'pleiadian is the largest face' in R[1]['fix'] and R[1]['largest_nonsinger'] == ['pleiadian']
    # cut 1: two words, one on a 94 px Red face, one with none -> tier none, px 0 (not the 94 median of a mix)
    R = G.grammar(_cast([('red_nova', 25, 35, 94)]), _cut(), [_w('a', 1.0, 1.4), _w('b', 1.5, 1.9)], B, song_offset=0.0, preset='landing')
    assert R[1]['tier'] == G.tier(R[1]['owner_px'])
    assert {G.name('maia', 'landing'), G.name('merope', 'landing'), G.name('grey_3', 'landing'), G.name('red_nova', 'landing')} == {'pleiadian', 'grey', 'red_nova'}


def test_main_defaults_to_native_basis(tmp_path):
    """checker #2: the default is delivered-frame px; 400 px@1080 on a 768 delivery is 284 native -> a medium."""
    import json
    cast = _cast([('red_nova', 0, 24, 400)])
    cast['results'][0]['size'] = [1344, 768]
    for n, o in (('cast', cast), ('cut', _cut()), ('align', {'words': [_w('love', 0.1, 0.5)]})):
        json.dump(o, open(tmp_path / f'{n}.json', 'w'))
    (tmp_path / 'S.md').write_text(SCRIPT)
    args = ['--cast', str(tmp_path / 'cast.json'), '--cut', str(tmp_path / 'cut.json'), '--align', str(tmp_path / 'align.json'),
            '--script', str(tmp_path / 'S.md'), '--song-offset', '0', '--out', str(tmp_path / 'g.json'), '--md', str(tmp_path / 'g.md')]
    G.main(args)
    r = json.load(open(tmp_path / 'g.json'))
    assert r['px_basis'] == 'native' and r['cuts'][0]['tier'] == 'cu' and r['thresholds']['cu'] == 240   # 284 >= 240
    G.main(args + ['--cu-px', '300'])
    assert json.load(open(tmp_path / 'g.json'))['cuts'][0]['tier'] == 'medium'
    assert 'delivered-frame px' in (tmp_path / 'g.md').read_text()
    G.main(args + ['--px-basis', '1080'])
    assert json.load(open(tmp_path / 'g.json'))['cuts'][0]['tier'] == 'cu'


def test_word_singer_is_the_attack_cut_not_an_early_onset():
    """A word whose onset leads the cut by a frame belongs to the cut its attack lands in (cut 2 = Sky), not cut 1."""
    B = G.script_beats(SCRIPT, 'landing')
    R = G.grammar(_cast([('sky_lux', 50, 74, 150)]), _cut(), [_w('beat', 1.98, 2.4)], B, song_offset=0.0, preset='landing')
    w = R[2]['word_checks'][0]
    assert w['singer'] == ['sky_lux'] and w['largest_nonsinger'] is None and R[2]['verdict'] == 'PASS'


def test_a_line_sung_ahead_of_its_cut_belongs_to_that_cut():
    """v3_2 cut 21: 'pleiadians' starts 0.23 s before cut 21, whose script Lyric cell begins with it (Sky's line): the
    word is Sky's, so Sky's face is not a 'largest non-singer'. A plain held note (checker #1) keeps its attack cut."""
    B = {0: {'singers': {'red_nova'}, 'solo': False, 'section': 'V'}, 1: {'singers': set(), 'solo': False, 'section': 'V'},
         2: {'singers': {'sky_lux'}, 'solo': False, 'section': 'P', 'lyric': '"Pleiadians don\'t dance"'}}
    R = G.grammar(_cast([('sky_lux', 50, 74, 150)]), _cut(3), [_w('pleiadians', 1.9, 2.6)], B, song_offset=0.0, preset='landing')
    w = R[2]['word_checks'][0]
    assert w['singer'] == ['sky_lux'] and w['largest_nonsinger'] is None
    assert G._first('"Pleiadians don\'t dance, don\'t ask \'em why"') == 'pleiadians' and G._first('(cont.)') == ''
    assert G.script_beats(SCRIPT, 'landing')[0]['lyric'] == '"x"'


def test_cut_px_is_the_singer_while_present():
    """QA2 cut 16: Red is in frame at ~55 px for part of the words, absent for the rest -> the cut reads wide ~55, not
    none / 0; the absent words still fail on their own."""
    B = G.script_beats(SCRIPT, 'landing')
    R = G.grammar(_cast([('red_nova', 25, 37, 55)]), _cut(), [_w('a', 1.0, 1.4), _w('b', 1.6, 1.9)], B, song_offset=0.0, preset='landing')
    assert R[1]['tier'] == 'wide' and R[1]['owner_px'] == 55 and R[1]['n_owner_absent'] == 1 and R[1]['verdict'] == 'FLAG'


def test_mic_share_flags_chrome_at_the_lips():
    """QA2 cut 19: a grey / chrome mic over the lip zone hides the mouth; skin and a dark open mouth do not."""
    import numpy as np
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (90, 120, 200)                # skin
    img[140:170, 80:120] = (20, 10, 40)                                               # an open mouth (dark)
    box = [50, 50, 150, 150]
    assert G.mic_share(img, box) < G.MIC_SHARE
    img[115:150, 85:115] = (190, 190, 192)                                            # chrome mic head at the lips
    assert G.mic_share(img, box) > G.MIC_SHARE
    cast = _cast([('red_nova', 0, 24, 400)])
    for x in cast['results'][0]['tracks'][0]['frames']: x['mic_share'] = 0.2
    R = G.grammar(cast, _cut(), [_w('love', 0.1, 0.9)], _B := G.script_beats(SCRIPT, 'landing'), song_offset=0.0, preset='landing')
    assert R[0]['n_mouth_hidden'] == 1


def test_mic_share_on_a_per_clip_scan_reads_the_cut_frames(tmp_path):
    """CHECK3 F2: a per-clip scan (NNN.mp4, clip-relative frames) reads the delivered video at cut start + f."""
    import numpy as np, cv2, pytest
    v = str(tmp_path / 'v.mp4')
    w = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*'mp4v'), 25, (200, 200))
    for g in range(60):
        img = np.zeros((200, 200, 3), np.uint8)
        img[:] = (90, 120, 200)
        if g >= 25: img[112:145, 85:115] = (190, 190, 192)                        # chrome at the lips only from 1.0 s
        w.write(img)
    w.release()
    cast = {'results': [{'video': 'clips/001.mp4', 'tracks': [{'character': 'red_nova',
             'frames': [{'f': f, 'face_px': 200, 'box': [50, 50, 150, 150]} for f in range(3)]}]}]}
    G.add_mic_share(cast, v, _cut(), 25.0, preset='landing')                                          # cut 1 starts at 1.0 s = frame 25
    assert all(x['mic_share'] > G.MIC_SHARE for x in cast['results'][0]['tracks'][0]['frames'])
    with pytest.raises(ValueError):
        G.add_mic_share(cast, v, preset='landing')                                                    # per-clip without the cut list
