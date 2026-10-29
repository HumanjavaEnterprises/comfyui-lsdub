"""lsdub medium (tier-2): the crop is a similarity, the nose-to-chin mask is feathered and exactly zero outside, the
composite keeps every outside pixel bit-for-bit, and colour match / regrain behave (synthetic, no models)."""
import numpy as np

from lsdub_engine.core import medium as M


def _lm(cx=100.0, cy=100.0, s=60.0):
    """A fake 478-landmark face: the lower oval on an ellipse below the nose, the nose base above the mouth."""
    lm = np.zeros((478, 2), np.float32)
    lm[:] = (cx, cy)
    for k, j in enumerate(M.OVAL_LOW):                               # cheek (left) -> chin -> cheek (right)
        a = np.pi * (1 - k / (len(M.OVAL_LOW) - 1))
        lm[j] = (cx + 0.45 * s * np.cos(a), cy + 0.5 * s * np.sin(a))
    for k, j in enumerate(M.NOSE_BASE):
        lm[j] = (cx + 0.2 * s * (k - 2) / 2, cy - 0.05 * s)
    lm[10], lm[152] = (cx, cy - 0.6 * s), (cx, cy + 0.5 * s)
    return lm


def test_mask_feathered_zero_outside_full_inside():
    m = M.lower_face_mask(_lm(), (200, 200, 3), grow=2, feather=4)
    assert m[115, 100] > 0.99                 # inside: the mouth/chin
    assert m[20, 100] == 0 and m[100, 20] == 0 and m[60, 100] == 0   # forehead / cheek-side / above the nose: exactly 0
    edge = m[100:200, 100]
    ramp = edge[(edge > 0) & (edge < 1)]
    assert len(ramp) >= 4                                            # a soft edge, several px wide
    assert np.all(np.diff(edge[np.argmax(edge):]) <= 1e-6)            # monotone falloff below the chin


def test_composite_keeps_outside_bit_exact():
    rng = np.random.default_rng(1)
    orig = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    painted = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    m = M.lower_face_mask(_lm(), orig.shape, 2, 4)
    o = M.composite(orig, painted, m)
    assert np.array_equal(o[m <= 0], orig[m <= 0])
    assert np.array_equal(o[m >= 1], painted[m >= 1])
    mid = (m > 0.4) & (m < 0.6)
    assert np.abs(o[mid].astype(int) - ((orig[mid].astype(int) + painted[mid].astype(int)) / 2)).mean() < 20


def test_crop_roundtrip_similarity():
    S, z, O = M.crop_geometry(100.0, 2.2, 330.0)
    assert O % 2 == 0 and abs(S * z - O) < 1e-6 and abs(z - 3.3) < 0.01
    A = M.crop_matrix(320.5, 200.25, S, z)
    assert A[0, 1] == 0 and A[1, 0] == 0 and A[0, 0] == A[1, 1]      # translate + one scale, no shear / rotation
    p = np.float32([[300, 190], [350, 230]])
    q = (A[:, :2] @ p.T).T + A[:, 2]
    assert np.allclose(M.uncrop_points(q, 320.5, 200.25, S, z), p, atol=1e-3)


def test_lab_match_and_regrain():
    rng = np.random.default_rng(0)
    orig = np.clip(rng.normal(140, 6, (120, 120, 3)), 0, 255).astype(np.uint8)
    m = np.zeros((120, 120), np.float32)
    m[30:90, 30:90] = 1
    off = np.clip(orig.astype(int) + 3, 0, 255).astype(np.uint8)       # a small global drift
    fixed, d = M.lab_match(off, orig, m)
    assert np.abs(fixed[m > 0].astype(int) - orig[m > 0].astype(int)).mean() < np.abs(off[m > 0].astype(int) - orig[m > 0].astype(int)).mean()
    smooth = np.full_like(orig, 140)
    g, s = M.regrain(smooth, orig, m, rng)
    assert s > 0 and M.grain(g)[m > 0.3].std() > 0.5 * M.grain(orig)[m > 0.3].std()


def test_gauss_and_fill():
    x = M.fill({0: [0.0], 4: [4.0]}, 6)
    assert np.allclose(x[:, 0], [0, 1, 2, 3, 4, 4])
    y = M._gauss(np.ones(10), 2.0)
    assert np.allclose(y, 1.0)


def test_valid_frames_drops_the_lost_track():
    """checker #3: no paste before the first / after the last detection or across a long gap; short gaps are bridged."""
    ok = M.valid_frames([2, 3, 5, 6, 15, 16], 20, max_gap=3)
    assert ok.tolist() == [False, False, True, True, True, True, True] + [False] * 8 + [True, True] + [False] * 3


def test_paste_back_is_exact_for_a_fractional_crop():
    """checker #5: a non-integer crop side (S = 312.4) and a mild zoom both land the painted crop back on the source
    geometry (a bright dot's centroid within 0.2 px), through the pre-shrink path."""
    for fh, tgt in ((142.0, 330.0), (240.0, 300.0)):
        S, z, O = M.crop_geometry(fh, 2.2, tgt)
        assert z > 1.0 and abs(S - round(S)) > 0.05 or fh == 240.0
        img = np.zeros((600, 800, 3), np.uint8)
        cv2 = __import__('cv2')
        cv2.circle(img, (403, 297), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cx, cy = 400.3, 300.7
        back = M.paste_back(M.crop_frame(img, cx, cy, S, z, O), cx, cy, S, z, img.shape)
        def cen(a):
            g = a[..., 0].astype(np.float64)
            ys, xs = np.mgrid[:g.shape[0], :g.shape[1]]
            return (g * xs).sum() / g.sum(), (g * ys).sum() / g.sum()
        assert np.hypot(*(np.subtract(cen(back), cen(img)))) < 0.2
