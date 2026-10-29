"""--lip-ref: lock every painted mouth's lip colour (Lab a/b) to one reference swatch; lightness follows the take."""
import numpy as np
import cv2

from lsdub_engine.core import mouth_morph as mm


def _face(lip_bgr, skin=(120, 150, 200), size=200):
    """synthetic face: skin field, lips = ring between an outer and an inner ellipse, with matching landmarks."""
    img = np.full((size, size, 3), skin, np.uint8)
    P = np.zeros((478, 2))
    c = np.array([size / 2, size / 2])
    t = np.linspace(0, 2 * np.pi, len(mm.LIP_OUT), endpoint=False)
    P[mm.LIP_OUT] = c + np.stack([45 * np.cos(t), 22 * np.sin(t)], 1)
    t = np.linspace(0, 2 * np.pi, len(mm.LIP_IN), endpoint=False)
    P[mm.LIP_IN] = c + np.stack([28 * np.cos(t), 5 * np.sin(t)], 1)
    cv2.ellipse(img, tuple(int(v) for v in c), (45, 22), 0, 0, 360, lip_bgr, -1)
    cv2.ellipse(img, tuple(int(v) for v in c), (28, 5), 0, 0, 360, (20, 20, 30), -1)
    return img, P


def _lab_ab(img, w):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.array([lab[..., c][w].mean() for c in range(3)])


def test_lip_ref_stats_reads_lip_tissue_only():
    img, P = _face((60, 40, 190))   # red lipstick
    st = mm.lip_ref_stats(img, P)
    assert st.shape == (3, 2)
    lab = cv2.cvtColor(np.uint8([[[60, 40, 190]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    assert np.allclose(st[:, 0], lab, atol=3)          # mean = the lip colour, not skin or mouth interior
    assert st[1, 1] < 3                                # flat swatch: tiny spread


def test_lock_moves_hue_keeps_take_lightness():
    ref, Pr = _face((60, 40, 190))                     # reference: red lips
    src, Ps = _face((150, 90, 160))                    # painted mouth: purple-ish lips (another shot's colour)
    take, Pt = _face((90, 110, 170), skin=(70, 90, 120))   # take: darker shot
    REF = mm.lip_ref_stats(ref, Pr)
    ws, wt = mm.lip_tissue_mask(Ps, src.shape), mm.lip_tissue_mask(Pt, take.shape)
    st = mm.lab_stats_lipref(src.astype(np.float32), ws, REF, take.astype(np.float32), wt)
    out = mm.apply_lab(src.astype(np.float32), st).astype(np.uint8)
    o, r, t = _lab_ab(out, ws), REF[:, 0], _lab_ab(take, wt)
    assert np.abs(o[1:] - r[1:]).max() < 4             # a/b locked to the reference lips
    assert abs(o[0] - t[0]) < 4                        # L follows the take (shot lighting)
    st1 = mm.lab_stats_lipref(src.astype(np.float32), ws, REF, take.astype(np.float32), wt, l_share=1.0)
    o1 = _lab_ab(mm.apply_lab(src.astype(np.float32), st1).astype(np.uint8), ws)
    assert abs(o1[0] - r[0]) < 4                       # l_share=1: lightness from the reference too


def test_lip_ref_no_region_is_skipped():
    src, Ps = _face((150, 90, 160))
    st = mm.lab_stats_lipref(src.astype(np.float32), np.zeros(src.shape[:2], bool), np.zeros((3, 2)), src, None)
    assert st == [0.0] * 9


def test_track_smooth_modes_zero_lag():
    from lsdub_engine.core.mouth_paint import smooth_track
    T = 40
    x = np.linspace(0, 39, T)[:, None, None] * np.ones((1, 3, 2))    # constant-velocity head move
    for m in ('k5', 'k9', 'sg9', 'none'):
        y = smooth_track(x, m)
        assert y.shape == x.shape
        assert np.abs(y[6:-6] - x[6:-6]).max() < 1e-6, m                  # centred: no lag on a linear move


def test_relight_lowfreq_carries_light_direction_keeps_detail():
    H, W = 120, 160
    xx = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    take = np.stack([60 + 120 * xx] * 3, -1)                            # lit from frame right
    rng = np.random.default_rng(0)
    det = rng.normal(0, 6, (H, W)).astype(np.float32)
    paint = np.stack([140 - 60 * xx + det] * 3, -1)                    # lit from frame left, with fine detail
    m = np.zeros((H, W), np.float32)
    m[20:100, 30:130] = 1
    out = mm.relight_lowfreq(paint, take, m, 20)
    L = lambda im: cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    lo, _lt, lp = L(out), L(take), L(paint)
    assert lo[20:100, 100:130].mean() > lo[20:100, 30:60].mean() + 20  # now brighter on the right, like the take
    assert abs(lo[m > 0].mean() - lp[m > 0].mean()) < 3                # overall level kept (colour match owns it)
    hp = lambda a: a - cv2.GaussianBlur(a, (0, 0), 2)
    assert np.corrcoef(hp(lo)[40:80, 60:100].ravel(), hp(lp)[40:80, 60:100].ravel())[0, 1] > 0.9   # detail kept
