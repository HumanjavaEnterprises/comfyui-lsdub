"""lsdub scorecard: motion / pops, the components, the weighted score + its reason, compare deltas, outputs (synthetic)."""
import json
import numpy as np

from lsdub_engine.audits import scorecard as SC


def test_motion_and_pops():
    med, pops = SC.motion_flicker([1.0, 1.1, 0.9, 1.0, 12.0, 1.0, 1.05])
    assert 0.9 <= med <= 1.1 and pops == 1
    assert SC.motion_flicker([0.1] * 10) == (0.1, 0)


def test_score_weights_and_reason():
    row = {'sung': True, 'corr': 0.9, 'corr_pm1': 0.9, 'open_in_silence': 0.1, 'rise_ms': 40, 'closure_hit': 1.0,
           'need': 'cu', 'tier': 'cu', 'mouth_visible': 1.0, 'canon_char': 'red_nova', 'canon_sim': 0.8, 'canon_px': 300,
           'motion': 2.0, 'pops': 0, 'sharpness': 100.0}
    c = SC.components(row, 100.0)
    s, worst = SC.shot_score(c, True)
    assert s == 100.0 and worst is None
    bad = dict(row, corr=0.1, corr_pm1=0.1)                          # the mouth is off the vocal
    c2 = SC.components(bad, 100.0)
    s2, w2 = SC.shot_score(c2, True)
    assert w2 == 'sync' and abs(s2 - 0.5 * 100 * (88 - 28) / 88) < 0.06   # sync 28 of the 88 present (no travel), then x 0.5
    bad['components'] = c2
    assert SC.reason(bad, w2).startswith('mouth off the vocal')
    plain = {'sung': False, 'canon_char': 'unknown', 'canon_sim': 0.4, 'canon_px': 150, 'motion': 0.1, 'pops': 0, 'sharpness': 50.0}
    c3 = SC.components(plain, 100.0)
    s3, w3 = SC.shot_score(c3, False)
    assert set(c3) == {'motion', 'flicker', 'sharpness'} and w3 == 'motion'   # an extra in an insert is not canon-scored
    assert SC.components(dict(row, tier='medium'), 100.0)['tier'] == 0.6


def test_compare_and_write(tmp_path):
    new = [{'cut': 0, 'shot': 'A', 'score': 80.0, 'sung': True, 'class': 'PERF', 'motion': 1.0, 'pops': 0, 'reason': '', 'video_t': [0, 2]},
           {'cut': 1, 'shot': 'B', 'score': 40.0, 'sung': True, 'class': 'PERF', 'motion': 1.0, 'pops': 2, 'reason': 'x', 'video_t': [2, 4]}]
    old = [{'cut': 0, 'score': 70.0, 'video_t': [0, 2]}, {'cut': 1, 'score': 45.0, 'video_t': [2, 4]}]
    R = SC.compare(new, old)
    assert R[0]['delta'] == 10.0 and SC.arrow(R[0]['delta']) == '↑' and SC.arrow(R[1]['delta']) == '↓'
    thumbs = {0: np.zeros((144, 256, 3), np.uint8), 1: np.zeros((144, 256, 3), np.uint8)}
    out = str(tmp_path / 'sc')
    SC.write(R, thumbs, out, 'v.mp4', compared=True)
    J = json.load(open(out + '.json'))
    assert [r['cut'] for r in J['cuts']] == [1, 0]                   # worst first
    assert open(out + '.csv').read().splitlines()[1].startswith('PERF,1,1,B,40.0')
    assert (tmp_path / 'sc_sheet.jpg').exists() and '↓' in open(out + '.md').read()


def test_nan_measurements_are_not_scored_as_numbers():
    row = {'sung': True, 'corr': float('nan'), 'corr_pm1': float('nan'), 'open_in_silence': None, 'rise_ms': None, 'closure_hit': None,
           'motion': 1.0, 'pops': 0, 'sharpness': None}
    c = SC.components(row, 100.0)
    s, w = SC.shot_score(c, True)
    assert c['sync'] == 0.0 and np.isfinite(s)


def test_classes_reasons_and_caps():
    base = {'sung': True, 'corr': 0.9, 'corr_pm1': 0.9, 'open_in_silence': 0.1, 'rise_ms': 40, 'closure_hit': 1.0, 'need': 'cu',
            'tier': 'cu', 'owner_px': 300, 'mouth_visible': 1.0, 'motion': 2.0, 'pops': 0, 'sharpness': 100.0}
    assert SC.shot_class(base) == 'PERF' and SC.shot_class(dict(base, sung=False)) == 'INSERT'
    assert SC.shot_class(dict(base, tier='wide', owner_px=60)) == 'COVERAGE'
    assert SC.shot_class(dict(base, mouth_visible=0.2)) == 'COVERAGE'
    # reason priority: no face -> too small -> mouth hidden
    assert SC.coverage_reason(dict(base, tier='none', owner_px=0)) == 'n/a: no singer on screen'
    assert SC.coverage_reason(dict(base, tier='wide', owner_px=60)).startswith('n/a: singer too small')
    assert SC.coverage_reason(dict(base, mouth_visible=0.1)).startswith('n/a: mouth hidden')


def test_compare_matches_by_time_overlap(capsys):
    new = [{'cut': 0, 'score': 50.0, 'video_t': [0, 2]}, {'cut': 1, 'score': 60.0, 'video_t': [2, 5]}]
    old = [{'cut': 0, 'score': 40.0, 'video_t': [0, 1]}, {'cut': 1, 'score': 30.0, 'video_t': [1, 2.1]}, {'cut': 2, 'score': 20.0, 'video_t': [2.1, 5]}]
    R = SC.compare(new, old)
    assert R[1]['old_cut'] == 2 and R[1]['delta'] == 40.0 and 'different cut lists' in capsys.readouterr().err


def test_canon_uses_the_singer_or_a_big_face():
    cast = {'results': [{'tracks': [
        {'character': 'unknown', 'scores': {}, 'frames': [{'f': 1, 'face_px': 60, 'box': [0, 0, 1, 1]}]},          # a small extra, no scores
        {'character': 'sky_lux', 'scores': {'sky_lux': 0.8}, 'frames': [{'f': 1, 'face_px': 40, 'box': [0, 0, 1, 1]}]}]}]}
    best, count, ns = SC.canon(cast, 0, 3, {'sky_lux'})
    assert best[0] == 'sky_lux' and best[1] == 0.8 and count >= 0
    best, _, _ = SC.canon(cast, 0, 3, set())
    assert best is None                                                  # nobody >= 80 px and no singer: no canon check
    row = {'sung': False, 'canon_char': 'red_nova', 'canon_sim': float('nan'), 'canon_px': 200, 'motion': 1.0, 'pops': 0}
    assert 'canon' not in SC.components(row, None)                     # a NaN similarity is skipped, not a crash


def test_finalize_caps():
    perf = {'sung': True, 'corr': 0.9, 'corr_pm1': 0.9, 'open_in_silence': 0.1, 'rise_ms': 40, 'closure_hit': 1.0, 'need': 'cu',
            'tier': 'cu', 'owner_px': 300, 'mouth_visible': 1.0, 'motion': 2.0, 'pops': 0, 'sharpness': 100.0, 'mouth_range': 0.05}
    r = SC.finalize(dict(perf), 100.0)
    assert r['class'] == 'PERF' and r['score'] == SC.TRAVEL_CAP_SCORE and r['reason'].startswith('mouth barely moves')
    r = SC.finalize({'sung': False, 'motion': 2.0, 'pops': 0, 'sharpness': 100.0}, 100.0)
    assert r['class'] == 'INSERT' and r['score'] == SC.INSERT_CAP and r['reason'] == 'pass'
    r = SC.finalize(dict(perf, tier='wide', owner_px=50), 100.0)
    assert r['class'] == 'COVERAGE' and r['score'] is None and r['reason'].startswith('n/a: singer too small')
