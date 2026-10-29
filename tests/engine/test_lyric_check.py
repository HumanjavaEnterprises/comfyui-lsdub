# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v12 the mouth card's word facts (lsdub/core/lyric_check.py): the normaliser (heard vs lyric), the per-word confidence
rule (weak words), reading the lyric word back from lyric_align's reconcile notes, the lyric check writer, the suggested
lyric sheet, and `lsdub visemes` persisting the facts into the track (words block + per-frame wi; --no-words = v11)."""
import json

import pytest

from lsdub_engine.core import lyric_check as lc
from lsdub_engine.core import viseme_track as vt


# ------------------------------------------------------------------------------------------------ the normaliser --
@pytest.mark.parametrize('lyric, heard', [
    ('four', '400'), ('hundred', '400'), ('dance', 'Dance,'), ('em', "'em"), ("don't", 'Dont'), ("don't", "Don’t"),
    ('dancing', "dancin'"), ('dance', 'dances'), ('dance', 'danced'), ("humanity's", 'humanities'), ('pleiadians', 'Pleiadian'),
    ('da', '-da'), ('first', '1st'), ('twenty one', '21'), ('serious', 'serious,'), ('light years', 'light-years'), ('Always', 'always.'),
    ('running', 'run'), ('hoped', 'hope'), ('dancing', 'dance')])
def test_same_word_after_normalisation(lyric, heard):
    assert lc.same_word(lyric, heard)


@pytest.mark.parametrize('lyric, heard', [
    ('em', 'them'), ('abducted', 'Adopted'), ('why', 'da'), ('sing', 'sin'), ('thing', 'thin'), ('needs', 'thanks'), ('hum', 'lots'),
    ('four', '40'), ('a', ''), ('time', 'Tim'), ('hope', 'hop'), ('note', 'not'), ('fine', 'fin')])
def test_different_words_stay_different(lyric, heard):
    assert not lc.same_word(lyric, heard)


def test_numbers_and_tokens():
    assert lc.spell_number(400) == 'four hundred' and lc.spell_number(21) == 'twenty one' and lc.spell_number(1999) == \
        'one thousand nine hundred ninety nine'
    assert lc.tokens('400') == ['four', 'hundred'] and lc.tokens('2nd') == ['second'] and lc.tokens('-da') == ['da']
    assert lc.stem('dances') == lc.stem('dance') == lc.stem('dancing') == lc.stem('danced') == 'danc'
    assert lc.norm_word("Humanity's") == lc.norm_word('humanities') == 'humanity'


# --------------------------------------------------------------------------------------------- the confidence --
def rec(**k):
    r = dict(i=0, w='word', start=1.0, end=1.4, src='lyric', heard='word', lyric='word', p=0.95, align='mfa', ph='W ER D', phrase='mfa')
    r.update(k)
    return r


def test_confidence_rule():
    c, why = lc.confidence(rec())
    assert c == 1.0 and why == [] and not lc.word_state(rec())['weak']
    c, why = lc.confidence(rec(align='whisper+spread', why='max word-edge dev 0.54s vs whisper'))      # whisper-timed alone
    assert c == 0.7 and why[0].startswith('whisper-timed (max word-edge') and not lc.word_state(rec(align='whisper+spread'))['weak']
    c, why = lc.confidence(rec(align='whisper+spread', p=0.3))                                          # + low probability
    assert c == pytest.approx(1 - 0.3 - 0.5 * 0.3 / 0.6, abs=0.01) and lc.word_state(rec(align='whisper+spread', p=0.3))['weak']
    fb = 'mfa: max word-edge dev 0.64s vs whisper'
    c, why = lc.confidence(rec(start=9.17, end=9.26, ph='AH W EY', phrase='sphinx (best)', why=fb))    # squeezed: 30 ms per phone
    assert c == 0.4 and why[0] == 'squeezed: 3 phones in 0.09 s' and why[1].startswith('phrase fell back to sphinx (best)')
    assert lc.short_reason(why) == '3 ph in 0.09 s'                                                     # the heaviest reason first
    assert lc.confidence(rec(start=0, end=2.0, ph='AA'))[0] == 0.7                                      # stretched: 2 s on one phone
    assert lc.confidence(rec(start=0, end=2.0, ph='AA', held=1.5))[0] == 1.0                            # ... unless a logged held note
    assert lc.confidence(rec(phrase='sphinx (best)', why=fb))[0] == 0.9                                 # phrase fell back, word forced-aligned
    assert lc.confidence(rec(phrase='sphinx (best)', why=fb, align='whisper+spread'))[0] == 0.7         # ... not counted twice
    assert lc.confidence(rec(phrase='sphinx (best)'))[0] == 1.0                                         # MFA passed, the alt scored better
    assert lc.confidence(rec(p=None))[0] == 1.0 and lc.confidence(rec(p=0.6))[0] == 1.0                  # no / sure probability
    assert lc.confidence(rec(p=0.0, start=0, end=0.04, ph='D AA', align='whisper+spread'))[0] == 0.0    # clipped at 0
    assert lc.WEAK_BELOW == 0.6 and not lc.word_state(rec(phrase='whisper+spread (best)'))['weak']
    st = lc.word_state(rec(align='whisper+spread', p=0.47988))                                         # raw 0.5999: weak, shown 0.60
    assert st['conf'] == 0.6 and st['weak']
    assert lc.word_state(rec(src='lyric', lyric='four', heard='five'))['kind'] == 'lyric'                # kind = the word's src


# ----------------------------------------------------------------------------- reading lyric_align's output --
def align():
    W = [dict(w='ask', start=1.0, end=1.3, src='lyric', heard='ask', p_whisper=0.99, align='mfa', idx=0, phones=[dict(p='AE', start=1.0, end=1.2), dict(p='S', start=1.2, end=1.25), dict(p='K', start=1.25, end=1.3)]),
         dict(w='them', start=1.3, end=1.5, src='sung', heard='them', p_whisper=0.29, align='whisper+spread', idx=1, phones=[dict(p='DH', start=1.3, end=1.35), dict(p='EH', start=1.35, end=1.45), dict(p='M', start=1.45, end=1.5)]),
         dict(w='why', start=1.5, end=2.0, src='lyric', heard='why', p_whisper=0.98, align='whisper+spread', idx=2, phones=[dict(p='W', start=1.5, end=1.6), dict(p='AY', start=1.6, end=2.0)]),
         dict(w='abducted', start=3.0, end=3.6, src='lyric~', heard='Adopted', p_whisper=0.9, align='mfa', idx=3, phones=[dict(p='AE', start=3.0, end=3.6)]),
         dict(w='da', start=4.0, end=4.2, src='sung-only', heard='-da', p_whisper=0.9, align='sphinx', idx=4, phones=[dict(p='D', start=4.0, end=4.05), dict(p='AA', start=4.05, end=4.2)])]
    return dict(lyrics='lyrics.txt', aligner='best', words=W,
                reconcile_notes=["'them' vs lyric 'em' -> 'them'", "'Adopted' vs lyric 'abducted' -> 'abducted'", 'sung, not in lyric: da'],
                phrases=[dict(phrase='ask them why', seg=[0.8, 2.2], method='sphinx (best)', mfa_why='mfa: max word-edge dev 0.92s vs whisper'),
                         dict(sphinx_chain_report=[dict(phrase='ask them why', method='fallback', why='max word-edge dev 0.54s vs whisper')])])


def test_lyric_words_from_the_reconcile_notes():
    lyr, ok = lc.lyric_words(align())
    assert ok and lyr == {0: 'ask', 1: 'em', 2: 'why', 3: 'abducted', 4: None}
    A = align()
    A['reconcile_notes'] = []                                          # notes that do not line up: no guess
    assert lc.lyric_words(A) == ({0: 'ask', 1: None, 2: 'why', 3: 'abducted', 4: None}, False)


def test_word_table_and_states():
    T = lc.word_table(align(), 0.9, 2.1, sheet="Don't ask 'em why")
    assert [r['i'] for r in T] == [0, 1, 2]                                         # the words overlapping the window
    them = T[1]
    assert them['lyric'] == 'em' and them['lyric_text'] == "'em" and them['heard'] == 'them' and them['phrase'] == 'sphinx (best)'
    assert them['why'] == 'max word-edge dev 0.54s vs whisper'                      # a whisper-timed word: the chain's reason
    st = lc.word_state(them)
    assert st['diff'] and st['kind'] == 'sung' and st['weak'] and st['lyric_text'] == "'em"
    assert not lc.word_state(T[0])['diff'] and not lc.word_state(T[2])['diff']
    full = lc.word_table(align())
    assert lc.word_state(full[3])['kind'] == 'lyric~' and lc.word_state(full[4])['kind'] == 'not-in-lyric'


# ------------------------------------------------------------------------------------------- the check writer --
def test_build_and_write_the_check(tmp_path):
    full = {r['i']: r for r in lc.word_table(align())}
    # 25 fps, the segment's track starts at 0.9 s: ask f3-9, them f10-14, why f15-27, abducted f53-67, da f78-82 (a skipped span)
    wi = [-1] * 100
    for i, r in full.items():
        for k in range(100):
            if r['start'] <= 0.9 + (k + 0.5) / 25 < r['end']: wi[k] = i
    vocal = [wi[k] == 4 for k in range(100)]
    seg = dict(label='S3', offset=188, frames=list(range(100)), wi=wi, recs=full, vocal=vocal, fps=25.0, mode='skip',
               spans=[dict(f0=78, f1=82, t0=4.0, t1=4.2, words=['da'], reasons=['lexicon:da'])])
    C = lc.build_check([seg], dict(video='cut.mp4'))
    assert C['counts'] == dict(words=5, mismatches=2, weak=1, vocalisation_spans=1, whisper_timed=2)
    m = {w['word']: w for w in C['mismatches']}
    assert set(m) == {'them', 'abducted'}                                            # the skipped 'da' is listed as a span only
    assert m['them']['lyric'] == "'em" or m['them']['lyric'] == 'em'
    assert m['them']['heard'] == 'them' and m['them']['frames'] == [188 + 10, 188 + 14] and m['them']['t'] == 1.3 and m['them']['weak']
    assert m['abducted']['kind'] == 'lyric~' and C['weak'][0]['word'] == 'them'
    assert C['vocalisation_spans'][0]['frames'] == [188 + 78, 188 + 82] and C['vocalisation_spans'][0]['mode'] == 'skip'
    j, md = lc.write_check(C, str(tmp_path / 'x_lyric_check.json'))
    assert json.load(open(j))['counts']['mismatches'] == 2 and md.endswith('x_lyric_check.md')
    txt = open(md).read()
    assert '| S3 | f198-202 | 1.30-1.50 |' in txt and 'Adopted' in txt and '## Weak words' in txt and 'lexicon:da' in txt
    # a word on screen in two segments (a cut through it) is listed once, with both labels and the range over both
    C2 = lc.build_check([seg, dict(seg, label='S4', offset=288)])
    assert C2['counts']['words'] == 5 and C2['counts']['vocalisation_spans'] == 2
    t2 = next(w for w in C2['words'] if w['word'] == 'them')
    assert t2['seg'] == 'S3+S4' and t2['frames'] == [188 + 10, 288 + 14] and t2['cut_t'] == round(198 / 25, 3)
    # only what is on screen: a segment that shows track frames 0-9 has no 'them' (f10-14), no 'da', no span
    C3 = lc.build_check([dict(seg, frames=list(range(10)))])
    assert [w['word'] for w in C3['words']] == ['ask'] and C3['counts']['mismatches'] == 0 and C3['vocalisation_spans'] == []
    # a skipped 'da' cut between a segment that does not show it and one that does is still a vocalisation, not a mismatch
    C4 = lc.build_check([dict(seg, frames=list(range(60))), dict(seg, label='S4', offset=288, frames=list(range(60, 100)))])
    da = next(w for w in C4['words'] if w['word'] == 'da')
    assert da['seg'] == 'S4' and da['vocalisation'] and 'da' not in [w['word'] for w in C4['mismatches']]


def test_suggested_lyrics_mark_but_do_not_apply():
    sheet = "[Pre-Chorus]\nDon't ask 'em why\nLots of new age contact, abducted a few\n"
    A = align()
    A['words'] = [dict(w='dont', src='lyric', heard="Don't", start=0.5, end=0.9, phones=[], idx=-1)] + A['words']
    for k, w in enumerate(A['words']): w['idx'] = k
    mism = [dict(i=2, heard='them'), dict(i=4, heard='Adopted')]
    txt, placed = lc.suggest_lyrics(sheet, A, mism)
    assert txt == "[Pre-Chorus]\nDon't ask [heard: them] why\nLots of new age contact, [heard: Adopted] a few\n"
    assert placed == [(2, True), (4, True)]
    assert lc.suggest_lyrics(sheet, A, [])[0] == sheet                              # nothing heard differently: the sheet as is
    # two heard words the sheet does not have, after the same sheet word, keep their sung order
    A2 = align()
    A2['words'][4:4] = [dict(w='for', src='sung-only', heard='for', start=3.7, end=3.8, phones=[], idx=4),
                                      dict(w='watching', src='sung-only', heard='watching!', start=3.8, end=3.95, phones=[], idx=5)]
    for k, w in enumerate(A2['words']): w['idx'] = k
    txt2, pl2 = lc.suggest_lyrics("Don't ask 'em why\nLots of new age contact, abducted a few\n", A2,
                                  [dict(i=5, heard='watching!'), dict(i=4, heard='for')])
    assert 'abducted [heard: for] [heard: watching!] a few' in txt2 and pl2 == [(4, True), (5, True)]


# ----------------------------------------------------------------------------- lsdub visemes persists the facts --
def test_visemes_writes_the_words_block(tmp_path):
    A = align()
    (tmp_path / 'lyrics.txt').write_text("Don't ask 'em why\nWe abducted\n")
    json.dump(A, open(tmp_path / 'a.json', 'w'))
    out, old = tmp_path / 'v.json', tmp_path / 'v_old.json'
    vt.main(['--align', str(tmp_path / 'a.json'), '--t0', '0.9', '--frames', '40', '--fps', '25', '--out', str(out)])
    vt.main(['--align', str(tmp_path / 'a.json'), '--t0', '0.9', '--frames', '40', '--fps', '25', '--out', str(old), '--no-words'])
    T, O = json.load(open(out)), json.load(open(old))
    assert 'words' not in O and 'words_meta' not in O and all('wi' not in f for f in O['frames'])         # v11 format
    assert [{k: v for k, v in f.items() if k != 'wi'} for f in T['frames']] == O['frames']                # same frames
    assert {k: v for k, v in T.items() if k not in ('words', 'words_meta')} .keys() == O.keys()
    assert [r['i'] for r in T['words']] == [0, 1, 2] and T['words'][1]['lyric_text'] == "'em"
    assert T['words_meta']['lyrics'] == str(tmp_path / 'lyrics.txt') and T['words_meta']['lyric_from_notes']
    for f in T['frames']:                                                          # wi = the frame's word in the alignment
        assert (f['wi'] >= 0) == bool(f['word']) and (f['wi'] < 0 or A['words'][f['wi']]['w'] == f['word'])
