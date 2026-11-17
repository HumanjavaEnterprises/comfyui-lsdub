# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""--warp-rise floor (sync 0.4 step 3): a sung syllable's onset carries an opening rise the picture cannot lose."""

import numpy as np
import pytest

from lsdub_engine.audits import event_judge as EJ
from lsdub_engine.core import mouth_paint, mouth_warp as mw

FPS = 25.0
RNG = dict(g_lo=2.0, g_hi=12.0)
FLOOR = mw.RISE_FLOOR * 10.0  # the floor on the test range (g_hi - g_lo = 10 px)


def frames(n, onsets):
    fr = [dict(t=i / FPS, vis="aa", vis2="aa", mix=0.0, open=0.5) for i in range(n)]
    for k, on in enumerate(onsets):
        for i in range(n):
            if abs(i / FPS - on) <= 0.1:
                fr[i]["ev"] = dict(kind="nucleus", t_on=on, k=k)
    return fr


def weak(n=50, at=20, size=0.12):
    """a held half-open mouth with one small rise (x the range) over frames at-1 .. at+1."""
    g = np.full(n, 6.0)
    g[at:] += size * 10.0 / 2
    g[at + 1 :] += size * 10.0 / 2
    return g


def test_onset_frames_one_time_per_event_on_frame_centres():
    t, ons = mw.onset_frames(frames(50, [0.82, 1.4]), 50, FPS)
    assert ons == [0.82, 1.4]
    assert t[0] == pytest.approx(0.02) and t[20] == pytest.approx(0.82)
    assert mw.onset_frames([], 0, FPS)[1] == []


def test_a_weak_rise_is_lifted_to_the_floor_on_the_onset_frame():
    g = weak()
    t, ons = mw.onset_frames(frames(50, [0.82]), 50, FPS)
    e, info = mw.rise_floor(g, t, ons, RNG)
    assert info["lifted"] == 1 and info["short"] == 0
    assert e[21] - e[19] == pytest.approx(mw.RISE_FLOOR * 10.0)
    d = e[2:] - e[:-2]
    assert int(np.argmax(d)) + 1 == 20  # the largest rise is the onset's own frame
    assert np.array_equal(e[:18], g[:18]) and np.array_equal(e[23:], g[23:])  # two frames each way, nothing further


def test_the_judge_reads_the_lifted_rise_at_nine_tenths_and_not_the_planned_one():
    g = weak(size=0.10)
    t, ons = mw.onset_frames(frames(50, [0.82]), 50, FPS)
    e, _ = mw.rise_floor(g, t, ons, RNG)
    wave = 3.0 * np.sin(np.arange(50) / 7.0)  # the take's own slow mouth, which sets the judged range
    assert EJ.match(ons, EJ.rises(0.9 * e + wave, FPS), EJ.TOL)[0] == 1
    assert EJ.match(ons, EJ.rises(0.9 * g + wave, FPS), EJ.TOL)[0] == 0


def test_a_rise_at_the_floor_is_left_as_planned():
    g = weak(size=0.5)
    t, ons = mw.onset_frames(frames(50, [0.82]), 50, FPS)
    e, info = mw.rise_floor(g, t, ons, RNG)
    assert info["lifted"] == 0 and np.array_equal(e, g)


def test_never_under_the_trough_line_never_over_the_range_never_a_shut_frame():
    g = np.full(50, 2.0 + mw.RISE_TROUGH * 10.0)  # already on the trough line: the whole rise goes up
    t, ons = mw.onset_frames(frames(50, [0.82]), 50, FPS)
    e, info = mw.rise_floor(g, t, ons, RNG)
    assert e.min() >= 2.0 + mw.RISE_TROUGH * 10.0 - 1e-9 and e[21] - e[19] == pytest.approx(FLOOR)
    hi = np.full(50, 12.0)  # already wide open: the whole rise comes from the frame before
    e, info = mw.rise_floor(hi, t, ons, RNG)
    assert e.max() <= 12.0 + 1e-9 and e[21] - e[19] == pytest.approx(FLOOR)
    shut = np.zeros(50, bool)
    shut[18:20] = True
    g = weak()
    e, info = mw.rise_floor(g, t, ons, RNG, shut=shut)
    assert np.array_equal(e[18:20], g[18:20]) and e[21] - e[19] == pytest.approx(FLOOR)


def test_lift_never_lowers_a_frame():
    g = weak()
    t, ons = mw.onset_frames(frames(50, [0.82]), 50, FPS)
    e, info = mw.rise_floor(g, t, ons, RNG, lower=False)
    assert info["mode"] == "lift" and info["lifted"] == 1
    assert (e >= g - 1e-9).all() and e[21] - e[19] == pytest.approx(FLOOR)


def test_no_room_either_way_is_counted_short():
    g = np.full(50, 12.0)
    shut = np.ones(50, bool)
    t, ons = mw.onset_frames(frames(50, [0.82]), 50, FPS)
    e, info = mw.rise_floor(g, t, ons, RNG, shut=shut)
    assert info["lifted"] == 0 and info["short"] == 1 and np.array_equal(e, g)


def test_flag_parses_and_classic_is_off():
    ap = mouth_paint._parser()
    base = ["--video", "v.mp4", "--track", "t.json", "--out", "o.mp4"]
    assert ap.parse_args(base).warp_rise == "off"
    assert ap.parse_args([*base, "--warp-rise", "classic"]).warp_rise == "off"
    assert ap.parse_args([*base, "--warp-rise", "floor"]).warp_rise == "floor"
    assert ap.parse_args([*base, "--warp-rise", "lift"]).warp_rise == "lift"
