# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sync 0.4 step 4: --warp-onset rise: a syllable's rise sits on its onset (the apex RISE_LAG_S after it), not its apex."""

import numpy as np
import pytest

from lsdub_engine.core import mouth_paint as MP
from lsdub_engine.core import nucleus_hold as NH

FPS = 50.0
EV = [
    dict(kind="nucleus", t_on=0.30, vowel="AA", bin=1, cons=True, src="nuc"),
    dict(kind="nucleus", t_on=0.70, vowel="AA", bin=1, cons=True, src="nuc"),
]


def _frames(n=60):
    out = []
    for i in range(n):
        t = i / FPS
        k = 1 if t >= 0.70 else 0 if t >= 0.30 else -1
        ev = dict(k=k, kind="rest", t_on=None) if k < 0 else dict(EV[k], k=k)
        out.append(dict(t=t, vis="aa", open=0.2, ev=ev))
    return out


def _steepest(o):
    return int(np.argmax(np.diff(o))) / FPS


def test_apex_is_the_default_and_the_030_curve():
    a, ia = NH.held(_frames(), FPS)
    b, ib = NH.held(_frames(), FPS, onset="apex")
    assert a == b and ia["onset"] == "apex" and ia["apex_lead_s"] == NH.APEX_LEAD_S == 0.0
    o = np.array([f["open"] for f in a])
    assert o[int(0.30 * FPS)] == pytest.approx(NH.target_of("AA", 1), abs=1e-3)  # fully open on the onset


def test_rise_moves_the_attack_later_by_the_lag_and_nothing_else():
    a = np.array([f["open"] for f in NH.held(_frames(), FPS, onset="apex")[0]])
    r, info = NH.held(_frames(), FPS, onset="rise")
    r = np.array([f["open"] for f in r])
    k = int(round(NH.RISE_LAG_S * FPS))
    assert info["onset"] == "rise" and info["apex_lead_s"] == -NH.RISE_LAG_S
    assert _steepest(r[:30]) - _steepest(a[:30]) == pytest.approx(NH.RISE_LAG_S)
    assert abs(_steepest(r[:30]) + 0.5 / FPS - 0.30) <= 0.5 / FPS + 0.011  # the steepest step is at the onset, within a frame
    assert np.allclose(r[k:30], a[: 30 - k], atol=1e-3)  # the first syllable's attack: the same curve, RISE_LAG_S later
    assert r.max() == pytest.approx(a.max(), abs=1e-3)


def test_rise_is_a_named_mode():
    assert NH.ONSETS == ("apex", "rise") and NH.RISE_LAG_S == 0.04
    with pytest.raises(ValueError):
        NH.held(_frames(), FPS, onset="late")


def test_flag_parses_and_classic_is_the_030_value():
    ap = MP._parser()
    base = ["--video", "v.mp4", "--track", "t.json", "--out", "o.mp4"]
    assert ap.parse_args(base).warp_onset == "apex"
    assert ap.parse_args([*base, "--warp-onset", "classic"]).warp_onset == "apex"
    assert ap.parse_args([*base, "--warp-onset", "rise"]).warp_onset == "rise"
