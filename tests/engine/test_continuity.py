# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
import json
import numpy as np
from lsdub_engine.core import continuity as c


def _shot(n=12):
    rng = np.random.default_rng(0)
    base = (rng.random((64, 64, 3)) * 60 + 100).astype(np.uint8)
    T = [base.copy() for _ in range(n)]
    R = []
    for i in range(n):
        r = base.copy()
        r[24:40, 20:44] = (60, 60, 160)            # a steady painted mouth
        R.append(r)
    return T, R


def test_checker_flags_a_one_frame_glitch_and_maker_fixes_it():
    T, R = _shot()
    R[6] = R[6].copy()
    R[6][24:40, 20:44] = (255, 255, 255)
    flag, s, M = c.check(T, R, k=4, z=3.5, err_floor=4.0)
    assert flag[6] and flag.sum() <= 2
    out, fixed, skipped = c.fix(T, R, flag, M)
    assert 6 in fixed and np.abs(out[6].astype(int) - R[5].astype(int)).mean() < 2


def test_blue_ghost_ruler():
    t = np.full((20, 20, 3), (90, 120, 180), np.uint8)
    r = t.copy()
    r[5:15, 5:15] = (200, 180, 150)
    assert c.blue_px(t, r, np.ones((20, 20), bool)) == 100 and c.blue_px(t, t, np.ones((20, 20), bool)) == 0


def test_squeeze_frames(tmp_path):
    p = tmp_path / 'w.json'
    json.dump(dict(range=dict(g_hi=20.0), frames=[dict(gap=12, egap=2), dict(gap=12, egap=10), dict(gap=3, egap=0.5)]), open(p, 'w'))
    assert c.squeeze_frames(str(p), 3).tolist() == [True, False, False]


def test_coherence_finds_a_three_frame_run_and_remakes_it():
    T, R = _shot(20)
    for i in (9, 10, 11):
        R[i] = R[i].copy()
        R[i][24:40, 20:44] = (240, 240, 240)
    M = c.paint_region(T, R)
    runs, _ = c.odd_frames(T, R, M)
    assert any(i <= 9 and i + L >= 12 for i, L, _ in runs)
    out, log = c.remake(T, R, runs, M)
    assert np.abs(out[10].astype(int) - R[8].astype(int)).mean() < 3


def test_texture_pop_is_a_short_burst_not_a_ramp():
    T, R = _shot(20)
    A = [r.copy() for r in R]
    for i in (6, 7, 8):
        R[i] = R[i].copy()
        R[i][24:40:2, 20:44] = 0   # striped slab on 3 frames
    M = [np.ones((64, 64), bool)] * 20
    assert c.texture_pops(T, R, A, M) == [(6, 3)]


def test_blue_ruler_ignores_cool_lit_teeth_but_keeps_a_ghost():
    """reco14 regression: 11C's open-mouth warp frames have cool-white teeth (BGR ~150,152,142; B - R = 8) where the
    take showed skin. With the old B - R > 6 margin they counted as a ghost, every open frame was flagged and the
    maker put the take's mouth back (the shipped 11C corr -0.35). Teeth must not count; a blue-grey ghost / a warped
    mic (BGR ~124,118,100; B - R = 24) still must."""
    t = np.full((20, 20, 3), (90, 120, 180), np.uint8)
    m = np.ones((20, 20), bool)
    teeth = t.copy()
    teeth[5:15, 5:15] = (150, 152, 142)
    ghost = t.copy()
    ghost[5:15, 5:15] = (124, 118, 100)
    faint = t.copy()
    faint[5:15, 5:15] = (160, 146, 152)          # shot B f91: B - R 8, B - G 14 (a true blue-grey)
    assert c.blue_px(t, teeth, m) == 0 and c.blue_px(t, ghost, m) == 100 and c.blue_px(t, faint, m) == 100
    T = [t] * 8
    R = [teeth] * 8                                   # a steady open mouth is not flagged on colour
    flag, s, _ = c.check(T, R, M=[m] * 8)
    assert int(s['blue'].max()) == 0 and not flag.any()


def test_fix_alt_keeps_a_clean_grown_frame_and_drops_a_desynced_take():
    """reco14 11C f10-14: a frame flagged only by growth, whose own paint is clean, keeps the render when the alt is
    ghosted (not the take), and spreads no ease into its neighbours; a frame whose own paint is bad still gets the take."""
    t = np.full((20, 20, 3), (90, 120, 180), np.uint8)
    m = np.ones((20, 20), bool)
    R = [t.copy() for _ in range(7)]
    for i in range(7): R[i][8:12, 8:12] = (60, 60, 200)              # our mouth: red-dark, not blue
    A = [x.copy() for x in R]
    for i in range(7): A[i][2:8, 2:8] = (160, 146, 152)              # the alternate carries a faint ghost
    flag = np.array([0, 0, 1, 1, 0, 0, 0], bool)
    keep = np.array([1, 1, 1, 0, 1, 1, 1], bool)
    out, w, src = c.fix_alt([t] * 7, R, A, flag, [m] * 7, keep=keep)
    assert src[2] == 'render' and np.array_equal(out[2], R[2])
    assert src[3] == 'take' and np.array_equal(out[3][8:12, 8:12], np.full((4, 4, 3), (90, 120, 180), np.uint8))
    assert np.array_equal(out[0], R[0])                             # nothing eased in from the kept frame 2
    assert np.array_equal(out[1], R[1])                             # nor into a frame beside a kept one (no half blend)


def test_remake_still_offers_the_take_where_it_follows_the_plan():
    """CHECK3 F3: desync only removes the take where it is off the plan; with desync False on the run the take is
    still a candidate and wins when it is the most coherent (here the run is the take itself, ghosted)."""
    T, R = _shot(20)
    for i in (9, 10):
        R[i] = R[i].copy()
        R[i][24:40, 20:44] = (240, 240, 240)
    M = c.paint_region(T, R)
    out, log = c.remake(T, R, [(9, 2, 9.0)], M, desync=np.zeros(20, bool))
    assert log[0]['used'] in ('take', 'interp')
    out2, log2 = c.remake(T, R, [(9, 2, 9.0)], M, desync=np.ones(20, bool))
    assert log2[0]['used'] != 'take'
    assert 0 < c.DESYNC_FRAC < 1


def test_squeeze_reverts_only_runs_that_show_a_lip_ghost():
    """v3_4 19 / 29: a squeeze run (take open, plan shut) with no ghost in the lip zone is the SUNG closure: keep it.
    A run with a blue-grey ghost in the upper (lip) part of the paint region still goes back to the take (sky B)."""
    t = np.full((40, 40, 3), (90, 120, 180), np.uint8)
    m = np.zeros((40, 40), bool)
    m[5:35, 5:35] = True
    clean = t.copy()
    clean[12:16, 10:30] = (60, 60, 200)
    lip_ghost = clean.copy()
    lip_ghost[8:14, 10:30] = (160, 146, 152)      # top of the region: a lip ghost
    prop = clean.copy()
    prop[28:34, 10:30] = (160, 146, 152)               # bottom of the region: a revealed mic
    sq = np.array([1, 1, 1, 0, 1, 1, 0], bool)
    T = [t] * 7
    R = [clean, clean, lip_ghost, clean, prop, prop, clean]
    keep, lz = c.squeeze_with_ghost(sq, T, R, [m] * 7)
    assert keep.tolist() == [True, True, True, False, False, False, False]
    assert lz[2] > c.BLUE_FLOOR and lz[4] == 0 and c.lipzone_ghost(t, prop, m) == 0
