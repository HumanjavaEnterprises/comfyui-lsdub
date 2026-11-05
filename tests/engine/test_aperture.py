# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the aperture stage (lsdub/core/aperture.py), its pure parts: the range map, the loop's correction and the fold limit. Synthetic, no models."""

import numpy as np

from lsdub_engine.core import aperture as AP


def frames(opens):
    return [{"vis": "aa", "vis2": "aa", "mix": 0.0, "open": o, "word": "x"} for o in opens]


def test_the_map_sets_the_drives_p95_to_a_full_opening_and_keeps_its_shape():
    o = np.r_[np.zeros(20), np.linspace(0.0, 0.6, 80)]
    out, k = AP.mapped(frames(o))
    got = np.array([f["open"] for f in out])
    assert abs(np.percentile(got, 95) - 1.0) < 0.02
    assert abs(k - 1.0 / np.percentile(o, 95)) < 1e-9
    assert np.all(np.diff(got[20:]) >= 0) and got[:20].max() == 0.0  # closed stays closed, the order of openings is kept
    assert got.max() <= 1.0


def test_the_map_never_narrows_and_never_blows_up_a_quiet_shot():
    assert AP.mapped(frames(np.r_[np.linspace(0, 1.0, 40), np.ones(20)]))[1] == 1.0  # its p95 is already a full opening: unchanged
    assert AP.mapped(frames(np.full(50, 0.02)))[1] == 1.0 / AP.FLOOR  # a near-silent drive: the gain is capped


def test_the_map_leaves_every_other_key_and_the_input_alone():
    fr = frames([0.1, 0.3, 0.5])
    fr[1]["vis"] = "PP"
    out, _ = AP.mapped(fr)
    assert [f["vis"] for f in out] == ["aa", "PP", "aa"] and [f["open"] for f in fr] == [0.1, 0.3, 0.5]


def test_the_loop_scales_a_short_frame_toward_its_request():
    req = np.full(9, 10.0)
    ach = np.full(9, 7.5)  # the render got 75 % of the move
    got = AP.loop_residual(req, ach, req.copy(), np.ones(9, bool), sigma=0)
    assert np.allclose(got, 10.0 / 0.75)


def test_the_loop_holds_the_applied_move_within_hi_times_the_request():
    req = np.full(5, 10.0)
    got = AP.loop_residual(req, np.full(5, 3.0), req.copy(), np.ones(5, bool), sigma=0)
    assert np.allclose(got, AP.HI * 10.0)


def test_the_loop_leaves_a_frame_on_target_an_invalid_frame_and_a_wrong_way_frame_as_they_are():
    req = np.array([10.0, 10.0, 10.0, -8.0, 10.0])
    ach = np.array([9.5, 5.0, -4.0, -8.1, np.nan])  # within TOL / invalid / wrong sign / within TOL / unread
    valid = np.array([True, False, True, True, True])
    applied = np.array([10.0, 10.0, 10.0, -8.0, 10.0])
    assert np.allclose(AP.loop_residual(req, ach, applied, valid, sigma=0), applied)


def test_the_loop_narrows_an_overshoot_but_not_under_lo():
    req = np.full(5, 10.0)
    got = AP.loop_residual(req, np.full(5, 20.0), req.copy(), np.ones(5, bool), sigma=0)
    assert np.allclose(got, AP.LO * 10.0)


def test_the_loop_correction_is_eased_and_stays_on_valid_frames():
    req = np.full(11, 10.0)
    ach = np.full(11, 10.0)
    ach[5] = 5.0
    valid = np.ones(11, bool)
    valid[0] = False
    got = AP.loop_residual(req, ach, req.copy(), valid, sigma=1.0)
    assert got[5] > got[4] > got[3] > 10.0 and got[0] == 10.0  # spread to the neighbours, not to the invalid frame


def test_the_fold_limit_takes_the_largest_clear_share_of_the_step():
    det = lambda v: 1.0 - float(v[0]) / 20.0  # folds (det < 0.15) past 17
    base, want = np.array([10.0, 0.0]), np.array([30.0, 0.0])
    s = AP.fold_limited(det, base, want, 0.15, iters=12)
    assert abs((base + s * (want - base))[0] - 17.0) < 0.05
    assert AP.fold_limited(det, base, np.array([12.0, 0.0]), 0.15) == 1.0


def test_calibration_reads_the_takes_open_frames():
    g = np.linspace(1, 50, 60)
    assert abs(AP.calibration(0.004 * g, g) - 0.004) < 1e-12
    assert AP.calibration(np.full(60, np.nan), g) is None


def test_steady_eases_single_frame_noise_and_keeps_unread_frames_unread():
    m = np.full(21, 0.2)
    m[10] = 0.4  # one frame's landmark noise
    m[3] = np.nan
    got = AP.steady(m)
    assert got[10] < 0.3 and np.isnan(got[3]) and abs(np.nanmean(got) - np.nanmean(m)) < 0.01
