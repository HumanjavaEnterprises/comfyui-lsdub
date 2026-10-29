# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""--mode warp (warp, don't paste): synthetic, CPU only. The dense warp must carry the take's own lip pixels to the
target landmarks (within 1 px) and leave every pixel beyond the falloff ring bit-identical."""
import numpy as np, cv2

from lsdub_engine.core import mouth_warp as mw


def _face(H=240, W=320):
    rng = np.random.default_rng(1)
    img = (rng.uniform(60, 200, (H, W, 3))).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    return img


def _lips(c=(160, 130), gap=4.0):
    """upper / lower lip points + corners around centre c."""
    xs = np.linspace(-30, 30, 7)
    up = np.stack([c[0] + xs, c[1] - gap / 2 - 6 * (1 - (xs / 30) ** 2)], 1)
    lo = np.stack([c[0] + xs[1:-1], c[1] + gap / 2 + 8 * (1 - (xs[1:-1] / 30) ** 2)], 1)
    return np.vstack([up, lo])


def _spot_image(pts, H=240, W=320):
    """one distinct dot per landmark so we can find where each pixel went."""
    img = np.full((H, W, 3), 128, np.uint8)
    for k, p in enumerate(pts):
        cv2.circle(img, tuple(np.round(p).astype(int)), 2, (0, 0, 255 - 3 * k), -1)
    return img


def test_lip_landmarks_reach_target_within_1px():
    src = _lips(gap=3.0)
    dst = src.copy()
    lower = src[:, 1] > 130
    dst[lower, 1] += 9.0   # open the jaw 9 px
    dst[:, 0] = 160 + (dst[:, 0] - 160) * 0.92                                             # and round a bit
    anchors = mw.ring_anchors(np.vstack([src, dst]), src.mean(0), 1.6)
    # sample the backward map at the targets: x_src = x_dst + w d(x_dst) must land on src
    ctrl = np.vstack([dst, anchors])
    disp = np.vstack([src - dst, np.zeros_like(anchors)])
    D = mw.tps_field(ctrl, disp, dst, 1e-3)
    assert np.abs(dst + D - src).max() < 1.0
    # and on pixels: a dot at each src lands at its dst
    for k in (0, 3, 6, 9):
        img = np.full((240, 320, 3), 128, np.uint8)
        cv2.circle(img, tuple(np.round(src[k]).astype(int)), 2, (0, 0, 255), -1)
        out, w = mw.warp_frame(img, src, dst, anchors, lam=1e-3)
        red = (out[..., 2].astype(int) - out[..., 1]) > 60
        yy, xx = np.nonzero(red)
        c = np.array([xx.mean(), yy.mean()])
        assert np.linalg.norm(c - np.round(src[k]) - (dst[k] - src[k])) < 1.0, (k, c, dst[k])


def test_pixels_beyond_mask_edge_unchanged():
    img = _face()
    src = _lips(gap=3.0)
    dst = src.copy()
    dst[src[:, 1] > 130, 1] += 10.0
    anchors = mw.ring_anchors(np.vstack([src, dst]), src.mean(0), 1.6)
    out, w = mw.warp_frame(img, src, dst, anchors)
    assert (w[w > 0] <= 1).all() and w.max() == 1.0
    outside = w == 0
    assert outside.sum() > 0.8 * w.size
    assert np.array_equal(out[outside], img[outside])


def test_confidence_deadband_leaves_matching_take_alone():
    dF = np.array([[0.02, 0.0], [0.5, 0.0], [0.0, -0.4]])
    c = mw.confidence(dF, np.array([1.0, 1.0]))
    assert c[0] == 0.0 and c[1] == 1.0 and c[2] == 1.0


def test_mouth_model_recovers_linear_structure():
    rng = np.random.default_rng(0)
    T, K = 60, len(mw.CTRL)
    F = np.stack([rng.uniform(2, 20, T), rng.uniform(50, 60, T)], 1)
    J = rng.normal(0, 1, (2, K, 3))
    base = rng.normal(0, 5, (K, 3))
    X = base + np.tensordot(F, J, axes=(1, 0))
    m = mw.MouthModel(X, F)
    assert np.allclose(m.J, J, atol=1e-6) and np.allclose(m.delta(np.array([1.0, 0.0])), J[0])


def test_target_bands_only_correct_what_is_wrong():
    fr = [dict(vis='PP', vis2='PP', mix=0, open=0)] * 3 + [dict(vis='aa', vis2='aa', mix=0, open=1.0)] * 3 + \
         [dict(vis='ou', vis2='ou', mix=0, open=0.2)] * 3 + [dict(vis='DD', vis2='DD', mix=0, open=0.1)] * 3
    rng = dict(g_lo=1.0, g_hi=11.0, g_max=12.0, w_lo=50.0, w_med=55.0, w_hi=60.0)
    F = np.tile([6.0, 55.0], (12, 1))
    F[9:, 0] = 4.0   # half-open, neutral width; DD frames a bit more closed
    Ft = mw.target_features(fr, F, rng, sigma=0)
    assert Ft[1, 0] < 1.5                      # PP: shut
    assert Ft[4, 0] >= 8                       # aa: opened at least to the band
    assert Ft[7, 1] < 54                       # ou: narrowed
    assert np.allclose(Ft[10], F[10])          # DD at a mid opening: already inside its band, untouched


def test_library_bands_follow_library_geometry(tmp_path):
    base = np.zeros((478, 2))
    base[61] = (-30, 0)
    base[291] = (30, 0)
    for v, gap, w in [('PP', 0, 60), ('sil', 2, 60), ('aa', 30, 56), ('ou', 10, 40), ('ih', 8, 72)]:
        L = base.copy()
        L[13] = (0, -gap / 2)
        L[14] = (0, gap / 2)
        L[61, 0] = -w / 2
        L[291, 0] = w / 2
        np.save(tmp_path / f'{v}_lm.npy', L)
    bo, bw = mw.library_bands(str(tmp_path))
    assert bo['aa'][1] >= 1.0 and bo['PP'][1] <= 0.15 and bo['ou'][1] < bo['aa'][0]
    assert bw['ou'][1] < 0 and bw['ih'][0] > 0 and 'sil' not in bw


def test_mode_b_samples_other_frame_and_keeps_edge():
    base = np.full((240, 320, 3), 100, np.uint8)
    other = np.full((240, 320, 3), 200, np.uint8)
    src = _lips(gap=3.0)
    dst = src.copy()
    dst[src[:, 1] > 130, 1] += 6
    an = mw.ring_anchors(np.vstack([src, dst]), src.mean(0), 1.6)
    out, w = mw.warp_frame(base, src, dst, an, src_img=other, anchors_src=an)
    assert np.array_equal(out[w == 0], base[w == 0])
    assert abs(int(out[int(dst[3, 1]), int(dst[3, 0]), 0]) - 200) <= 2      # inside: the other frame's pixels


def test_bank_prefers_matching_shape():
    F = np.array([[1.0, 50], [2, 50], [9, 50], [10, 50], [3, 50]])
    b = mw.MouthBank(0, F, None, np.zeros((5, 3)), None)
    j, d = b.nearest(0, np.array([10.0, 50]), np.array([10.0, 10.0]), exclude=0)
    assert j == 3 and d == 0


# --- r2: temporal smoothing of the warp field, cross-fades, head-frame falloff, fold guard, sync targeting ---

class _Pose:
    """identity-ish rigid pose: canonical (x, y, z) -> frame (x, y) with scale s and a translation."""
    def __init__(self, s=1.0, t=(0.0, 0.0, 0.0)):
        self.s = np.array([s])
        self.R = np.eye(3)[None]
        self.t = np.array([t], float)

    def rigid(self, i, X): return (self.s[i] * X @ self.R[i].T + self.t[i])[:, :2]


def test_smooth_residual_is_zero_phase_and_keeps_closures():
    d = np.zeros((41, 2))
    d[20] = [1.0, 2.0]
    r = mw.smooth_residual(d, 1.5)
    assert int(np.argmax(r[:, 0])) == 20 and np.allclose(r[:, 0], r[::-1, 0])     # centred: no lag, symmetric
    assert r[20, 0] < 1.0 and abs(r[:, 0].sum() - 1.0) < 1e-6                      # eased, mass kept
    c = np.zeros((11, 2))
    c[5, 0] = -1.0
    keep = np.zeros(11, bool)
    keep[5] = True
    assert mw.smooth_residual(c, 1.5, keep)[5, 0] == -1.0                         # a PP closure keeps its full depth
    assert np.array_equal(mw.smooth_residual(d, 0), d)


def test_soft_sources_cross_fade_instead_of_swap():
    idx = np.array([-1] * 6 + [7] * 6)
    m = mw.soft_sources(idx, 1.0)
    assert all(abs(sum(x.values()) - 1) < 1e-9 for x in m)
    assert m[0] == {-1: 1.0} and m[-1] == {7: 1.0}
    assert 0.3 < m[5][-1] < 0.9 and 0.1 < m[5][7] < 0.7                           # mixed around the switch
    w7 = [x.get(7, 0.0) for x in m]
    assert all(b >= a for a, b in zip(w7, w7[1:]))  # monotone ramp
    assert mw.soft_sources(idx, 0)[5] == {-1: 1.0}                                  # sigma 0: hard switch (r1)


def test_head_ellipse_falloff_is_smooth_fixed_and_exactly_zero_outside():
    X = np.array([[-30, 0, 0], [30, 0, 0], [0, -10, 0], [0, 15, 0]], float)
    pl = _Pose(1.0, (160, 120, 0))
    e = mw.HeadEllipse(X, np.zeros(3), 1.0, 2.0, q=100)
    w = e.weight(pl, 0, (240, 320))
    assert w[120, 160] == 1.0 and w[120, 160 + 29] == 1.0                         # core (rho <= 1)
    assert w[120, 160 + 61] == 0.0 and w[120 - 21, 160] == 0.0 and w[120 + 31, 160] == 0.0   # beyond rho = 2
    row = w[120, 160:225]
    assert np.all(np.diff(row) <= 1e-6)                     # monotone falloff
    assert np.abs(np.diff(row, 2)).max() < 0.01                                    # no crease (C1/C2 ramp)
    # carried by the pose: the same canonical ellipse, moved with the head
    w2 = e.weight(_Pose(1.0, (170, 125, 0)), 0, (240, 320))
    assert np.allclose(w2[125:, 10:], w[120:-5, :-10])
    an = e.anchors(pl, 0, 16)
    q = an - [160, 120]
    assert np.allclose(e.rho(q), 2.0)


def test_warp_with_ellipse_weight_leaves_outside_bit_identical():
    img = _face()
    src = _lips(gap=3.0)
    dst = src.copy()
    dst[src[:, 1] > 130, 1] += 10.0
    X = np.hstack([np.vstack([src, dst]) - [160, 130], np.zeros((2 * len(src), 1))])
    pl = _Pose(1.0, (160, 130, 0))
    e = mw.HeadEllipse(X, np.zeros(3), 1.0, 2.6, q=100)
    wmap = e.weight(pl, 0, img.shape)
    an = e.anchors(pl, 0)
    out, w = mw.warp_frame(img, src, dst, an, lam=0.01, weight=wmap, interp=cv2.INTER_LANCZOS4)
    assert np.array_equal(out[w == 0], img[w == 0]) and (w == 0).sum() > 0.6 * w.size


def test_fold_guard_detects_a_fold_and_scales_it_away():
    src = _lips(gap=3.0)
    dst = src.copy()
    X = np.hstack([np.vstack([src, dst]) - [160, 130], np.zeros((2 * len(src), 1))])
    pl = _Pose(1.0, (160, 130, 0))
    narrow = mw.HeadEllipse(X, np.zeros(3), 1.0, 1.15, q=100)
    wide = mw.HeadEllipse(X, np.zeros(3), 1.0, 4.0, q=100)
    def det(k, e):
        d = dst.copy()
        d[:, 1] += 10.0 * k                                         # a rigid jaw drop
        return mw.fold_det(src, d, e.anchors(pl, 0), e.weight(pl, 0, (240, 320)), 0.01)
    assert det(1.0, narrow) < 0.15 < det(1.0, wide)                                # a narrow falloff folds, a wide one does not
    k = mw.fold_scale(lambda s: det(s, narrow), 0.15)
    assert k < 1.0 and det(k, narrow) >= 0.15


def test_centre_targeting_goes_past_the_band_edge():
    assert mw._to_band(0.5, 0.0, 0.2, 0.0) == 0.2                                  # r1: band edge
    assert abs(mw._to_band(0.5, 0.0, 0.2, 0.8) - 0.12) < 1e-9                      # r2: toward the centre (0.1)
    assert mw._to_band(0.15, 0.0, 0.2, 0.8) == 0.15                                # inside the band: untouched
    assert abs(mw._to_band(-1.0, 0.3, np.inf, 1.0) - 0.8) < 1e-9                   # one-sided band: edge + 0.5


def test_class_floor_makes_spread_and_round_unambiguous():
    bo, bw = mw.class_floor(({'E': (0.8, 1.1)}, {'E': (0.05, mw.INF), 'oh': (-mw.INF, 0.03)}), 0.5)
    assert bw['E'][0] == 0.5 and bw['oh'][1] == -0.5 and bw['SS'][0] == 0.5 and bw['ou'][1] == -0.5


def test_jaw_reference_keeps_upper_lip_on_the_skull():
    rng = np.random.default_rng(0)
    T, K = 50, len(mw.CTRL)
    F = np.stack([rng.uniform(2, 20, T), rng.uniform(50, 60, T)], 1)
    J = rng.normal(0, 1, (2, K, 3))
    m = mw.MouthModel(rng.normal(0, 5, (K, 3)) + np.tensordot(F, J, axes=(1, 0)), F)
    ix = {k: j for j, k in enumerate(mw.CTRL)}
    up = [ix[k] for k in mw.UPPER]
    lo = [ix[k] for k in (14, 17, 84, 314)]
    rel = (m.J[0][lo].mean(0) - m.J[0][up].mean(0)).copy()
    wcol = m.J[1].copy()
    m.jaw_reference(up)
    assert np.allclose(m.J[0][up].mean(0), 0, atol=1e-9)                           # upper lip: no net motion with the gap
    assert np.allclose(m.J[0][lo].mean(0) - m.J[0][up].mean(0), rel)              # the lips still part by the same amount
    assert np.array_equal(m.J[1], wcol)


def test_interior_alpha_is_own_or_donor_not_a_mix():
    assert mw.interior_alpha(60, 70, 0.2) == 0.0                                   # take already open: keep its own teeth
    assert mw.interior_alpha(5, 80, 0.9) == 1.0                                    # shut take opened wide: donor
    assert 0 < mw.interior_alpha(52, 80, 0.5) < 1                                  # narrow ramp between


# --- v3 ---------------------------------------------------------------------------------------------------------------
_RNG = dict(g_lo=0.0, g_hi=10.0, w_lo=40.0, w_med=58.0, w_hi=60.0)      # a take parked near its p90 width


def test_class_floor_wide_uses_the_wider_half_range():
    _, half = mw.class_floor(None, 0.5, _RNG, 'half')
    _, wide = mw.class_floor(None, 0.5, _RNG, 'wide')
    assert half['E'][0] == 0.5 and half['ou'][1] == -0.5                          # r2: 0.5 x (60-58) = +1 px only
    assert abs(wide['E'][0] * (60 - 58) - 0.5 * 18) < 1e-9                        # v3: 0.5 x the wider half (18) = +9 px
    assert abs(wide['ou'][1] * (58 - 40) + 0.5 * 18) < 1e-9                       # round: wide half is its own -> -0.5


def _fr(vis, vis2, mix, n):
    return [dict(vis=vis, vis2=vis2, mix=mix, open=0.0) for _ in range(n)]


def test_width_mix_primary_keeps_the_vowels_width_band():
    n = 5
    F = np.tile([5.0, 50.0], (n, 1))
    bands = mw.class_floor(None, 0.5, _RNG, 'wide')
    fr = _fr('E', 'sil', 0.5, n)                                                  # E mixed 50/50 with sil (no width band)
    sw = mw.target_features(fr, F, _RNG, sigma=0, bands=bands, centre=0.8, width_mix='switch')
    pr = mw.target_features(fr, F, _RNG, sigma=0, bands=bands, centre=0.8, width_mix='primary')
    assert np.allclose(sw[:, 1], 50.0)                                            # r2: sil's (absent) band -> width dropped
    assert np.all(pr[:, 1] > 58.0)                                                # v3: E's spread target kept
    fr2 = _fr('E', 'ou', 0.7, n)                                                  # vis2 has its own band and dominates
    assert np.all(mw.target_features(fr2, F, _RNG, sigma=0, bands=bands, centre=0.8, width_mix='primary')[:, 1] < 58.0)


def test_close_depth_targets_below_the_shut_gap():
    n = 4
    F = np.tile([6.0, 58.0], (n, 1))
    fr = _fr('PP', 'PP', 0.0, n)
    t0 = mw.target_features(fr, F, _RNG, sigma=0, centre=0.8)[:, 0]
    t1 = mw.target_features(fr, F, _RNG, sigma=0, centre=0.8, close_depth=0.1)[:, 0]
    assert np.all(t0 >= 0) and np.all(t1 < 0) and np.all(t1 < t0)


def test_seal_fill_is_a_noop_above_threshold_and_local_below():
    img = _face()
    inner = _lips(gap=6.0)
    out, al = mw.seal_fill(img, inner, 0.5, 0.2)
    assert al == 0.0 and out is img
    out, al = mw.seal_fill(img, inner, 0.0, 0.2)
    assert al == 1.0
    ch = np.nonzero(np.abs(out.astype(int) - img.astype(int)).max(2))
    assert ch[0].size > 0 and ch[1].min() >= inner[:, 0].min() - 12 and ch[1].max() <= inner[:, 0].max() + 12
