# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v4 layering (lip_layers.py + the warp hooks): size cap, mouth scale, lip erase, lip colour lock. Synthetic, CPU only."""
import numpy as np, cv2

from lsdub_engine.core import lip_layers as ll, mouth_warp as mw, mouth_paint

SKIN = (120, 155, 205)       # BGR, warm skin (skin_px true)
LIP = (45, 40, 185)          # BGR, lipstick red


def _face(c=(160, 130), hw=40, hh=14, gap=0.0, H=260, W=320):
    """478-point landmark array with the outer / inner lip loops on ellipses around c (other points parked on the
    forehead), and an image: skin, red lips, a grey (teeth) inner mouth when gap > 0."""
    P = np.tile(np.array([c[0], c[1] - 90.0]), (478, 1))
    P[33] = (c[0] - 60, c[1] - 70)
    P[263] = (c[0] + 60, c[1] - 70)
    a = np.linspace(np.pi, -np.pi, len(ll.LIP_OUT), endpoint=False)
    P[ll.LIP_OUT] = np.stack([c[0] + hw * np.cos(a), c[1] + hh * np.sin(a)], 1)
    P[ll.LIP_IN] = np.stack([c[0] + 0.75 * hw * np.cos(a), c[1] + max(0.5, gap / 2) * np.sin(a)], 1)
    P[13], P[14] = (c[0], c[1] - gap / 2), (c[0], c[1] + gap / 2)
    P[0], P[17] = (c[0], c[1] - hh), (c[0], c[1] + hh)
    P[61], P[291] = (c[0] - hw, c[1]), (c[0] + hw, c[1])
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = SKIN
    cv2.ellipse(img, (int(c[0]), int(c[1])), (hw, hh), 0, 0, 360, LIP, -1)
    if gap > 0: cv2.ellipse(img, (int(c[0]), int(c[1])), (int(0.7 * hw), int(gap / 2)), 0, 0, 360, (150, 150, 150), -1)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    return img, P


def test_size_cap_never_past_take_p95_unless_take_is():
    own = np.array([10, 10, 10, 30, 10, 10, 10.0])
    tgt = np.array([10, 25, 10, 32, 10, 10, 12.0])
    t = ll.size_cap(tgt, own, hi=20.0, sigma=0)
    assert t[1] == 20.0                       # capped at the take's p95
    assert t[3] == 30.0                       # the take's own frame (30 > p95) is the face's own mouth: not shrunk below it
    assert t[6] == 12.0                       # under the cap: untouched
    lo = ll.size_cap(np.array([5.0, 9.0]), np.array([10.0, 10.0]), hi=20.0, lo=8.0, sigma=0)
    assert lo.tolist() == [8.0, 9.0]
    te = ll.size_cap(tgt, own, hi=20.0, sigma=1.0)
    assert (te <= np.maximum(20.0, own) + 1e-9).all() and te[1] <= 20.0


def test_mouth_scale_shrinks_lips_keeps_far_points():
    _, P = _face(hw=40, hh=14)
    base = P.copy()
    far = P.copy()
    far[5] = (160 + 150, 130 + 100)
    base[5] = far[5]
    L = ll.mouth_scale(far[None], base, 0.8, 0.7)[0]
    w0, w1 = np.linalg.norm(far[61] - far[291]), np.linalg.norm(L[61] - L[291])
    assert abs(w1 / w0 - 0.8) < 0.02
    assert abs(np.linalg.norm(L[0] - L[17]) / np.linalg.norm(far[0] - far[17]) - 0.7) < 0.02
    assert np.allclose(L[5], far[5])          # far from the mouth: unchanged (smooth falloff to none)


def test_erase_lips_leaves_only_skin():
    img, P = _face(gap=10.0)
    bald, m = ll.erase_lips(img, P)
    lab = cv2.cvtColor(bald, cv2.COLOR_BGR2LAB).astype(float)
    sk = cv2.cvtColor(np.uint8([[SKIN]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    hull = mouth_paint.hull_mask(P[ll.LIP_OUT], img.shape, 0.0, 3) > 0.5
    assert m[hull].all()
    assert np.abs(lab[hull][:, 1] - sk[1]).max() < 6 and np.abs(lab[hull][:, 0] - sk[0]).mean() < 4   # no red, no teeth
    assert np.array_equal(bald[~m], img[~m])                                                           # outside: untouched


def test_harmonic_fill_meets_boundary():
    img = np.zeros((40, 40, 3))
    img[..., 0] = np.linspace(0, 100, 40)[None]   # a horizontal ramp
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    f = ll.harmonic_fill(np.where(m[..., None], 255.0, img), m)
    assert np.abs(f[..., 0] - img[..., 0]).max() < 1.0                         # a linear ramp is harmonic: recovered


def test_lip_lock_moves_ab_keeps_L_and_skin():
    img, P = _face(gap=8.0)
    st = ll.lip_stats(img, P)
    assert st is not None
    ref = np.array([100.0, st[1] - 12, st[2] - 6])
    lk = ll.LipLock(ref, sigma=0).fit([st])
    out, w = lk.apply(img, P, 0)
    st2 = ll.lip_stats(out, P)
    assert abs(st2[1] - ref[1]) < 1.5 and abs(st2[2] - ref[2]) < 1.5           # a*/b* locked to the reference
    tissue = (w > 0.9)
    L0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    L1 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    assert np.abs(L1[tissue] - L0[tissue]).mean() < 2.0                         # L follows the shot
    assert np.array_equal(out[w <= 0.002], img[w <= 0.002])                     # skin, teeth: bit-identical
    teeth = (np.abs(img.astype(int) - 150).max(2) < 12)
    assert np.abs(out[teeth].astype(int) - img[teeth]).max() <= 2


def test_lip_lock_eases_and_resists_a_mistracked_frame():
    img, P = _face(gap=8.0)
    st = ll.lip_stats(img, P)
    bad = st.copy()
    bad[1] -= 20                                               # one frame's mask landed on skin
    lk = ll.LipLock([100, st[1], st[2]], sigma=2.0, clip=3.0).fit([st] * 5 + [bad] + [st] * 5)
    assert abs(lk.shift(5)[1]) < 3.0 + 1e-6 and abs(lk.shift(0)[1]) < 1.0


def test_erase_alpha_only_on_closures_over_an_open_take():
    rng = dict(g_lo=0.0, g_hi=40.0, w_lo=50.0, w_med=60.0, w_hi=70.0)
    a = mw.erase_alpha(np.array([30.0, 1.0, 30.0]), np.array([0.5, 0.5, 30.0]), rng, sigma=0)
    assert a[0] > 0.95 and a[1] < 0.01 and a[2] < 0.01


def test_v4_flags_parse(capsys):
    try: mouth_paint.main(['--help'])
    except SystemExit: pass
    out = capsys.readouterr().out
    for f in ('--warp-size-cap', '--warp-erase', '--warp-lip-lock', '--lip-erase', '--mouth-scale', '--mouth-open-cap'):
        assert f in out


def test_mouth_dims_and_take_size_rest_vs_p95():
    frames = []
    for gap in [0.0] * 6 + [6.0, 12.0, 20.0, 30.0]:
        _, P = _face(gap=gap)
        frames.append(P)
    ts = ll.take_size(np.array(frames))
    assert abs(ts['rest'][1]) < 1e-6 and ts['p95'][1] > 0.2      # rest = closed, p95 = its own big opening (IOD units)
    assert abs(ts['rest'][0] - 80 / 120) < 1e-6                     # width 2*hw over the 120 px inter-ocular distance
