# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the lead gate stage (lsdub/core/lead_gate.py): synthetic, no models."""

import numpy as np

from lsdub_engine.core import lead_gate as LG


def track(spec):
    """spec: [(vis, phone, word, open)] per frame."""
    return [{"vis": v, "vis2": v, "mix": 0.0, "phone": p, "word": w, "open": o} for v, p, w, o in spec]


def test_the_gate_holds_the_drive_down_where_the_take_is_shut_and_still():
    fps = 50.0
    take = np.r_[np.full(30, 0.6), np.full(40, 0.05), np.full(30, 0.6)]  # shut and still for 0.8 s in the middle
    drive = np.full(100, 0.8)
    o, g, kept = LG.gate(drive, take, fps, ease=0)
    assert kept == [(30, 69)] and o[30:70].max() == 0.0 and np.allclose(o[:30], 0.8) and np.allclose(o[70:], 0.8)


def test_the_gate_leaves_a_short_shut_a_moving_take_and_a_quiet_drive_alone():
    fps = 50.0
    short = np.r_[np.full(30, 0.6), np.full(10, 0.05), np.full(30, 0.6)]  # 0.2 s: a breath
    assert LG.gate(np.full(70, 0.8), short, fps)[2] == []
    moving = np.r_[np.full(30, 0.6), 0.02 + 0.12 * (np.arange(40) % 2), np.full(30, 0.6)]  # under SHUT but chattering
    assert LG.gate(np.full(100, 0.8), moving, fps)[2] == []
    shut = np.r_[np.full(30, 0.6), np.full(40, 0.05), np.full(30, 0.6)]
    assert LG.gate(np.full(100, 0.1), shut, fps)[2] == []  # the drive asks for nothing there


def test_the_gate_is_eased_and_never_opens_the_mouth_more():
    take = np.r_[np.full(30, 0.6), np.full(40, 0.05), np.full(30, 0.6)]
    drive = np.random.default_rng(0).random(100)
    drive[30:70] = 0.9
    o, g, _ = LG.gate(drive, take, 50.0)
    assert np.all(o <= drive + 1e-12) and 0 < g[29] < 1


def test_gated_frames_keep_every_other_key():
    fr = track([("aa", "AA1", "a", 0.8)] * 60)
    out, info = LG.gated(fr, np.full(60, 0.05), 50.0)
    assert info["frames"] == 60 and all(f["vis"] == "aa" and f["open"] == 0.0 for f in out) and fr[0]["open"] == 0.8
