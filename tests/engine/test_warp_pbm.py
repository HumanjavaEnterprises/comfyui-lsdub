# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""0.4 step 2: --warp-pbm contact: a P/B/M shuts to the lips' contact, not to the take's own nearest to it."""

import numpy as np

from lsdub_engine.core import mouth_paint as MP
from lsdub_engine.core import mouth_warp as MW

RNG = dict(g_lo=6.0, g_hi=66.0, w_lo=40.0, w_med=50.0, w_hi=60.0)  # a take that never seals: its shut gap is 6 px


def _fr(vis, n):
    return [dict(vis=vis, vis2=vis, mix=0.0, open=0.5) for _ in range(n)]


def test_off_presses_under_the_takes_own_shut_gap_only():
    F = np.tile([30.0, 50.0], (4, 1))
    g = MW.target_features(_fr("PP", 4), F, RNG, sigma=0, close_depth=0.05)[:, 0]
    assert np.allclose(g, 6.0 - 0.02 * 60.0)  # 4.8 px: still parted


def test_contact_reaches_the_seal_gap():
    F = np.tile([30.0, 50.0], (4, 1))
    g = MW.target_features(_fr("PP", 4), F, RNG, sigma=0, close_depth=0.05, seal=MW.SEAL_GAP)[:, 0]
    assert np.allclose(g, MW.SEAL_GAP) and MW.SEAL_GAP <= 0


def test_contact_never_lifts_a_deeper_press():
    rng = dict(RNG, g_lo=0.5, g_hi=60.5)  # a sealing take: close_depth 0.1 already goes under the contact
    F = np.tile([30.0, 50.0], (4, 1))
    off = MW.target_features(_fr("PP", 4), F, rng, sigma=0, close_depth=0.1)[:, 0]
    on = MW.target_features(_fr("PP", 4), F, rng, sigma=0, close_depth=0.1, seal=MW.SEAL_GAP)[:, 0]
    assert np.all(off < MW.SEAL_GAP) and np.allclose(on, off)


def test_contact_moves_only_the_closures():
    fr = _fr("aa", 3) + _fr("PP", 2) + _fr("aa", 3)
    F = np.tile([30.0, 50.0], (8, 1))
    off = MW.target_features(fr, F, RNG, sigma=0, close_depth=0.05)
    on = MW.target_features(fr, F, RNG, sigma=0, close_depth=0.05, seal=MW.SEAL_GAP)
    pp = np.array([f["vis"] == "PP" for f in fr])
    assert np.allclose(on[~pp], off[~pp]) and np.all(on[pp, 0] < off[pp, 0]) and np.allclose(on[:, 1], off[:, 1])


def test_contact_survives_the_easing():
    fr = _fr("aa", 6) + _fr("PP", 1) + _fr("aa", 6)
    F = np.tile([30.0, 50.0], (13, 1))
    on = MW.target_features(fr, F, RNG, sigma=0.8, close_depth=0.05, seal=MW.SEAL_GAP)
    assert on[6, 0] <= MW.SEAL_GAP + 1e-9  # a one-frame closure keeps its depth


def test_the_flag_is_off_by_default_and_classic_is_off():
    ap = MP._parser()
    base = ["--video", "v", "--track", "t", "--out", "o"]
    assert ap.parse_args(base).warp_pbm == "off"
    assert ap.parse_args([*base, "--warp-pbm", "classic"]).warp_pbm == "off"
    assert ap.parse_args([*base, "--warp-pbm", "contact"]).warp_pbm == "contact"
