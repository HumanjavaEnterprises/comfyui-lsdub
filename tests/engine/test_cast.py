# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub cast: tracking, identity decision, colour vote, script parsing, the dub plan and the dry-run hook (synthetic,
no models)."""
import numpy as np

from lsdub_engine.core import cast as C


def _face(x, y, s=100, yaw=0.0):
    b = [x, y, x + s, y + s]
    return {'box': b, 'det_box': b, 'face_h': s, 'yaw': yaw, 'lm': None}


def test_iou_and_link_keeps_two_faces_apart_through_a_gap():
    a, b = _face(100, 100), _face(600, 100)
    per = [[a, b], [_face(105, 102), _face(603, 100)], [_face(610, 101)], [_face(112, 104), _face(612, 100)]]
    T = C.link(per, max_gap=2)
    assert len(T) == 2
    assert [f for f, _ in T[0]] == [0, 1, 3]          # the left face bridges the missed frame 2
    assert [f for f, _ in T[1]] == [0, 1, 2, 3]
    assert C.iou(a['box'], a['box']) == 1.0 and C.iou(a['box'], b['box']) == 0.0


def test_runs_bridge_single_gaps():
    assert C.runs([0, 1, 2, 4, 5, 9]) == [[0, 5], [9, 9]]
    assert C.runs([]) == []


def test_decide_margin_unknown_and_family():
    assert C.decide({'singer_a': 0.9, 'singer_b': 0.6})[0] == 'singer_a'
    assert C.decide({'singer_a': 0.40, 'singer_b': 0.30})[0] == 'unknown'           # under the threshold
    assert C.decide({'singer_a': 0.66, 'singer_b': 0.65})[0] == 'unknown'           # no margin
    # near-identical Pleiadians: the margin is taken against the best OTHER family
    who, conf, runner = C.decide({'alcyone': 0.77, 'maia': 0.76, 'merope': 0.75, 'grey_1': 0.58}, preset='landing')
    assert who == 'alcyone' and runner == 'grey_1' and conf > 0.5
    assert C.family('maia', 'landing') == 'pleiadian' and C.family('grey_5', 'landing') == 'grey' and C.family('singer_a') == 'singer_a'


def test_colour_signature_separates_palettes():
    red = np.zeros((64, 64, 3), np.uint8)
    red[:] = (30, 60, 200)          # BGR: red-brown hair
    pink = np.zeros((64, 64, 3), np.uint8)
    pink[:] = (180, 60, 230)       # hot pink
    s_r, s_p = C.colour_sig(red), C.colour_sig(pink)
    assert abs(s_r.sum() - 1) < 1e-5
    assert C.colour_sim(s_r, s_r) > 0.99 and C.colour_sim(s_r, s_p) < 0.2


def test_score_track_prefers_the_matching_identity():
    rng = np.random.RandomState(0)
    e_r, e_s = rng.randn(1536), rng.randn(1536)
    n = lambda v: (v / np.linalg.norm(v)).astype(np.float32)
    sig = C.colour_sig(np.full((8, 8, 3), (30, 60, 200), np.uint8))
    db = {'name': np.array(['singer_a', 'singer_b']), 'emb': np.stack([n(e_r), n(e_s)]), 'emb_f': np.stack([n(e_r), n(e_s)]),
          'col_h': np.stack([sig, sig]), 'col_b': np.stack([sig, sig])}
    E = np.stack([n(e_r + 0.3 * rng.randn(1536))])
    sc = C.score_track(db, E, [sig], [sig], EF=E)
    assert sc['singer_a'] > sc['singer_b'] + 0.2


def test_mouth_occluded_flags_a_grey_bar_over_the_lips():
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (120, 150, 200)    # skin
    lm = np.zeros((478, 2), np.float32)
    lm[:] = (100, 100)
    ang = np.linspace(0, 2 * np.pi, len(C.LIPS), endpoint=False)
    lm[C.LIPS] = np.stack([100 + 30 * np.cos(ang), 140 + 10 * np.sin(ang)], 1)
    for i, (x, y) in zip(C.CHEEKS, [(60, 110), (140, 110), (65, 125), (135, 125), (70, 95), (130, 95)]):
        lm[i] = (x, y)
    clear, share0 = C.mouth_occluded(img, lm)
    img[128:152, 60:140] = (200, 200, 200)                              # a chrome mic across the mouth
    occ, share1 = C.mouth_occluded(img, lm)
    assert not clear and occ and share1 > share0


SCRIPT = """## 1. decisions
| # | Topic | Decision |
|---|---|---|
| 7 | Singing | Red and Sky are seen singing wherever |

### 5.1 Verse 1
| # | v | s | Lyric | Story beat | Shot | Cam | Motion | Status |
|---|---|---|---|---|---|---|---|---|
| 18 | 20.04-21.92 | 8.04 | "Four hundred" | **Red sings** into the mics, pointing | 05C | CU | i2v | keep |
| 21 | 25.67 | 13.67 | "..." | Battle-rap stance: Sky leans across the podium singing and pointing | 06A | MS | i2v | keep |
| 22 | 26.61 | 14.61 | (cont.) | Sky's studded belt | D07 | insert | i2v | keep |
| 56 | 85.67 | 73.67 | "..." | Full rave. The duo sing and dance at the decks | S5 | wide | x | keep |
"""


def test_script_singers_reads_only_beat_tables():
    S = C.script_singers(SCRIPT, 'landing')
    assert S == {18: {'red_nova'}, 21: {'sky_lux'}, 56: {'red_nova', 'sky_lux'}}   # row 7 of the decisions table ignored


def _cast_result(f0, f1, who='red_nova', safe=True, px=320.0):
    frames = [{'f': f, 'box': [0, 0, 1, 1], 'face_px': px, 'yaw': 2.0, 'mouth_occluded': False, 'landmarks': True,
               'dub_safe': safe} for f in range(f0, f1 + 1)]
    return {'video': '/x/018.mp4', 'tracks': [{'character': who, 'confidence': 1.0, 'f0': f0, 'f1': f1,
            'face_px_med': px, 'yaw_med': 2.0, 'dub_safe_runs': [[f0, f1]] if safe else [], 'frames': frames}]}


def test_plan_maps_song_words_to_clip_frames():
    cut = {'cut': [{'shot': f'S{i}', 'start': 0.0, 'dur': 1} for i in range(18)] + [{'shot': '05C', 'start': 20.0, 'dur': 1.88}]}
    cast = {'results': [_cast_result(0, 46)]}
    words = [{'w': 'four', 'start': 8.10, 'end': 8.40}, {'w': 'hundred', 'start': 8.40, 'end': 8.90},
             {'w': 'later', 'start': 30.0, 'end': 30.5}]
    P = C.plan(cast, cut, words, {18: {'red_nova'}}, song_offset=12.0, preset='landing')
    assert len(P) == 1 and P[0]['words'] == 'four hundred' and P[0]['role'] == 'singer'
    assert P[0]['word_spans'][0]['frames'] == [2, 9]           # (8.10 + 12 - 20) * 25 = 2.5 -> 2 ; end 10 -> 9
    assert P[0]['dub_safe'] is True and P[0]['near_safe'] is True
    S = C.summarise(P)
    assert S['red_nova']['dub_safe'] == 1 and S['red_nova']['singer_spans'] == 1


def test_plan_roles_and_unsafe():
    cut = {'cut': [{'shot': f'S{i}', 'start': 0.0, 'dur': 1} for i in range(18)] + [{'shot': '05C', 'start': 20.0, 'dur': 1.88}]}
    cast = {'results': [_cast_result(0, 46, who='sky_lux', safe=False, px=120.0)]}
    words = [{'w': 'four', 'start': 8.10, 'end': 8.40}]
    P = C.plan(cast, cut, words, {18: {'red_nova'}}, preset='landing')
    assert P[0]['role'] == 'not_singing' and not P[0]['dub_safe'] and not P[0]['near_safe']
    assert C.plan(cast, cut, words, {}, preset='landing')[0]['role'] == 'unassigned'


def test_hook_is_a_dry_run_listing():
    cut = {'cut': [{'shot': f'S{i}', 'start': 0.0, 'dur': 1} for i in range(18)] + [{'shot': '05C', 'start': 20.0, 'dur': 1.88}]}
    P = C.plan({'results': [_cast_result(0, 46)]}, cut, [{'w': 'four', 'start': 8.10, 'end': 8.40}], {18: {'red_nova'}}, preset='landing')
    cmds = C.hook(P, None, 'lead.wav', '/w', 'align.json')
    assert len(cmds) == 1
    txt = '\n'.join(cmds[0])
    assert 'lsdub visemes' in txt and 'lsdub paint' in txt and 'lsdub continuity' in txt and '--t0 8.0' in txt
