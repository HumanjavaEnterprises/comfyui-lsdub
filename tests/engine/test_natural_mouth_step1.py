# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Natural mouth step 1 and the step 3 source test: the opt-in flags and the ruler. Golden hashes prove the defaults."""

import numpy as np, cv2, pytest

from lsdub_engine.audits import mouth_ruler as MR
from lsdub_engine.core import interior_take as IT, lip_layers as ll, medium as M, mouth_layers as ML, mouth_paint as MP
from lsdub_engine.core.mouth_warp import MouthBank

BASE = ["--video", "v", "--track", "t", "--out", "o"]


def face(n=478, gap=20.0):
    """a flat mouth: 61 / 291 = corners, 13 / 14 = aperture, inner contour on an ellipse."""
    P = np.zeros((n, 3))
    P[:, :2] = [100, 100]
    th = np.linspace(0, 2 * np.pi, len(ll.LIP_IN), endpoint=False)
    P[ll.LIP_IN, 0] = 100 + 40 * np.cos(th)
    P[ll.LIP_IN, 1] = 100 + gap * np.sin(th)
    o = np.linspace(0, 2 * np.pi, len(ll.LIP_OUT), endpoint=False)
    P[ll.LIP_OUT, 0] = 100 + 50 * np.cos(o)
    P[ll.LIP_OUT, 1] = 100 + (gap + 10) * np.sin(o)
    P[61], P[291], P[13], P[14] = [50, 100, 0], [150, 100, 0], [100, 100 - gap, 0], [100, 100 + gap, 0]
    return P


def frame(bgr=(60, 60, 170)):
    f = np.full((200, 200, 3), 150, np.uint8)
    cv2.ellipse(f, (100, 100), (40, 20), 0, 0, 360, bgr, -1)
    return f


def clip(n=8, bgr=(60, 60, 170)):
    F = [frame(bgr) for _ in range(n)]
    S = np.array([face() for _ in range(n)])
    feat = np.tile([40.0, 100.0], (n, 1))
    return F, S, feat, MouthBank(0, feat, np.zeros((n, 3, 2)), np.zeros((n, 3)), S[:, :, :2])


def test_new_flags_default_off():
    a = MP._parser().parse_args(BASE)
    assert (a.warp_tongue_pose, a.warp_fill_l3, a.warp_interior_source, a.warp_lossless) == ("zero", "hide", "model", False)
    assert ML.LayerCompositor.take_interior is None and ML.LayerCompositor.fill_l3 == "hide"


def test_slow_tongue_varies_without_slosh():
    c = ML.LayerCompositor.__new__(ML.LayerCompositor)
    c.n = 100
    v = ["E", "aa"] * 25 + ["E"] * 25 + ["aa"] * 25  # a per-frame flip barely moves it, a held vowel does
    c.slow_tongue(v, 6.0)
    assert np.ptp(c.tg[15:35, 0]) < 0.1 * np.ptp(c.tg[:, 0])
    assert np.ptp(c.tg[50:, 0]) > 0 and np.abs(np.diff(c.tg, axis=0)).max() < 0.25 * np.ptp(c.tg[:, 0]) + 1e-9


def test_rule_says_the_pose_needs_single():
    w = MP.validate_paint_args(MP._parser().parse_args([*BASE, "--mode", "warp", "--warp-tongue-pose", "lowpass"]))
    assert any("--warp-tongue-pose" in x and "single" in x for x in w)


def test_lossless_write_round_trips(tmp_path):
    p = str(tmp_path / "x.mp4")
    F = [np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8) for i in range(3)]
    M.write(F, p, lossless=True)
    assert all((a == b).all() for a, b in zip(F, M.read(p, 3)[0]))


def test_take_interior_fills_with_the_takes_pixels():
    F, S, Fe, bank = clip()
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    Q = S[0][ll.LIP_IN][:, :2]
    rgb, a = T.paint(0, Q, (0, 0, 200, 200), 0.4, np.zeros((200, 200), np.float32))
    assert a[100, 100] > 0.9 and np.abs(rgb[100, 100] - F[0][100, 100]).max() < 16
    assert a[5, 5] == 0


def test_pink_never_above_the_takes_share():
    F, S, Fe, bank = clip()
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    T.curve[:] = 0.2
    rgb, a = T.paint(0, S[0][ll.LIP_IN][:, :2], (0, 0, 200, 200), 0.4, np.zeros((200, 200), np.float32))
    assert T.log[0]["sank"] > 0.5
    assert cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2LAB)[100, 100, 0] < cv2.cvtColor(F[0], cv2.COLOR_BGR2LAB)[100, 100, 0] + 40


def test_closed_donor_leaves_the_model_layers():
    F, S, Fe, bank = clip()
    S[:] = face(gap=1.0)
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    assert T.paint(0, S[0][ll.LIP_IN][:, :2], (0, 0, 200, 200), 0.0, np.zeros((200, 200), np.float32))[1].max() == 0


def test_residual_is_zero_on_a_frozen_interior():
    F, S, *_ = clip(12)
    r = MR.interior_residual(F, S[:, :, :2])
    assert r["pairs"] > 0 and r["res_interior"] < 0.5


def test_reveal_sees_a_wider_mouth():
    T = np.array([face(gap=12.0)[:, :2] for _ in range(4)])
    for k in MR.STABLE:
        T[:, k] = [20 + 25 * MR.STABLE.index(k), 30 + (k % 7) * 3]
    W = T.copy()
    W[:, MR.LIPI] = np.array([face(gap=30.0)[:, :2] for _ in range(4)])[:, MR.LIPI]
    W[:, 13], W[:, 14] = [100, 70], [100, 130]
    C = {i: MR.canon(None, T)[i] for i in range(4)}
    same = MR.reveal(T, C, T, C, (200, 200))
    wide = MR.reveal(T, C, W, C, (200, 200))
    assert same["unseen_max"] == pytest.approx(0, abs=0.01) and wide["unseen_mean"] > 0.3 and wide["frames_past_take"] == 4
