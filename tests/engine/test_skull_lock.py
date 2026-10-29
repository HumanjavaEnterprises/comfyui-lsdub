# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Skull lock + face constraints: the mouth stays on the skull (one midline, mouth line parallel to the pupils, lips in
order), measured against the character's own take, never against a canon target."""
import numpy as np
from lsdub_engine import face_constraints as fc
from lsdub_engine.core.skull_lock import corrections, CTRL

MOUTH_PTS = [0, 13, 14, 17, 61, 291, 78, 308, 37, 267, 82, 312, 84, 314, 146, 375]


def face():
    """a frontal synthetic face: pupils 60 px apart (=62 mm), nose midline at x=110"""
    P = np.zeros((478, 2))
    P[:] = (110, 120)
    P[468], P[473] = (80, 100), (140, 100)
    for k, xy in {168: (110, 100), 6: (110, 112), 4: (110, 140), 1: (110, 143), 2: (110, 150), 94: (110, 151),
                  164: (110, 155), 0: (110, 160), 13: (110, 166), 14: (110, 168), 17: (110, 178), 152: (110, 215),
                  61: (88, 167), 291: (132, 167), 78: (92, 167), 308: (128, 167), 37: (102, 161), 267: (118, 161),
                  82: (104, 166), 312: (116, 166), 84: (104, 176), 314: (116, 176), 146: (92, 171), 375: (128, 171),
                  172: (70, 190), 397: (150, 190), 234: (58, 140), 454: (162, 140), 33: (70, 100), 263: (150, 100),
                  133: (95, 100), 362: (125, 100)}.items():
        P[k] = xy
    return P


def seq(P, n=6):
    return np.repeat(P[None], n, 0)


def test_identical_output_has_no_skull_violations():
    take = seq(face())
    cal = fc.calibrate(take)
    s = fc.summarize(fc.check(take.copy(), cal))
    for r in ('C1_midline', 'C2_axis', 'C3_upper_rigid', 'C10_lip_order', 'C15_cylinder'):
        assert s[r] == 0


def test_check_flags_offset_tilt_and_crossed_lips():
    take = seq(face())
    cal = fc.calibrate(take)
    shifted = take.copy()
    shifted[:, MOUTH_PTS, 0] += 4                       # mouth 4 px (~4 mm) off the midline
    assert fc.summarize(fc.check(shifted, cal))['C1_midline'] == len(take)
    tilted = take.copy()
    c = take[0, [61, 291]].mean(0)
    a = np.radians(6)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    tilted[:, MOUTH_PTS] = (take[:, MOUTH_PTS] - c) @ R.T + c                    # mouth line 6 deg off the pupils
    assert fc.summarize(fc.check(tilted, cal))['C2_axis'] == len(take)
    crossed = take.copy()
    crossed[:, 14, 1] = take[0, 13, 1] - 3                # lower inner lip above the upper
    assert fc.summarize(fc.check(crossed, cal))['C10_lip_order'] == len(take)


def test_canon_is_a_sanity_bound_not_a_target():
    take = seq(face())
    take[:, [61, 78, 146], 0] -= 12
    take[:, [291, 308, 375], 0] += 12   # a very wide mouth
    cal = fc.calibrate(take)
    assert any('mouth_ipd' in w for w in cal['warnings'])                        # flagged...
    assert fc.summarize(fc.check(take.copy(), cal))['C13_rounding'] == 0         # ...but the character keeps it


def test_skull_lock_moves_the_mouth_back_onto_the_midline_and_leaves_the_nose():
    take = seq(face())
    out = take.copy()
    out[:, MOUTH_PTS, 0] += 4
    D, _ = corrections(out, take, fc.IPD_MM, cal=fc.calibrate(take))
    ix = {k: i for i, k in enumerate(CTRL)}
    assert np.all(D[:, ix[13], 0] < -2.0)                                        # pulled back toward the midline
    assert 2 not in CTRL and 1 not in CTRL and 468 not in CTRL                   # nose / eyes are never moved


def test_skull_lock_levels_a_tilted_mouth():
    take = seq(face())
    out = take.copy()
    c = take[0, [61, 291]].mean(0)
    a = np.radians(3.5)   # under the 4 deg cap
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out[:, MOUTH_PTS] = (take[:, MOUTH_PTS] - c) @ R.T + c
    D, _ = corrections(out, take, fc.IPD_MM, cal=fc.calibrate(take))
    ix = {k: i for i, k in enumerate(CTRL)}
    fixed = fc.to_skull(out[0], CTRL) + D[0]
    v = fixed[ix[291]] - fixed[ix[61]]
    assert abs(np.degrees(np.arctan2(v[1], v[0]))) < 1.5                         # within the 1 deg target (+ noise)


