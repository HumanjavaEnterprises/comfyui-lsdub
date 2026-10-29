"""pin: the painted lower face must move with the real skin, never swim (CPU, synthetic only)."""
import numpy as np
import cv2

from lsdub_engine import slip_audit as sa
from lsdub_engine import mouth_pin as mpin

H, W, N = 240, 320, 14


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, (H, W), np.uint8)
    t = cv2.GaussianBlur(t, (0, 0), 1.2)
    return cv2.merge([t, t, t])


def _shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _landmarks():
    """478 points: nose/cheek indices up top, philtrum/lip/lower-face indices below the 'subnasale' (index 2)."""
    p = np.zeros((478, 2))
    p[:] = (W / 2, H / 2)
    rng = np.random.default_rng(1)
    for k in set(sa.REG['nose']): p[k] = (W / 2 + rng.uniform(-14, 14), 70 + rng.uniform(-8, 8))
    for k in set(sa.REG['cheek']): p[k] = (W / 2 + rng.choice([-1, 1]) * rng.uniform(40, 60), 80 + rng.uniform(-10, 10))
    for k in set(sa.REG['phil']) | set(sa.REG['ulip']): p[k] = (W / 2 + rng.uniform(-18, 18), 105 + rng.uniform(-6, 6))
    p[2] = (W / 2, 88)
    for k in mpin.LOWER: p[k] = (W / 2 + rng.uniform(-45, 45), 100 + rng.uniform(0, 70))
    p[164] = (W / 2, 90)                                       # the hull reaches right up to the subnasale, like a real face
    return p


def _scene():
    """take: the whole face translates (head move). output: the lower face lags behind the head by 0..5 px (swims)."""
    base = _texture()
    p0 = _landmarks()
    head = [(1.5 * t, 0.8 * t) for t in range(N)]
    swim = [(-5 * np.sin(np.pi * t / (N - 1)), 3 * np.sin(np.pi * t / (N - 1))) for t in range(N)]
    T = [_shift(base, *head[t]) for t in range(N)]
    m = np.zeros((H, W), np.float32)
    m[95:, :] = 1
    m = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
    O = [(T[t] * (1 - m) + _shift(base, head[t][0] + swim[t][0], head[t][1] + swim[t][1]) * m).astype(np.uint8)
         for t in range(N)]
    return T, O, p0


def _slip(O, T, p0):
    At = sa.audit(T, p0)
    Ao = sa.audit(O, p0)
    rel_t = At['phil'][0] - At['nose'][0]
    rel_o = Ao['phil'][0] - Ao['nose'][0]
    return np.linalg.norm(rel_o - rel_t, axis=1)


def test_slip_audit_sees_the_swim():
    T, O, p0 = _scene()
    s = _slip(O, T, p0)
    assert s.max() > 3.0                      # the injected swim (up to ~5.8 px) is measured
    assert _slip(T, T, p0).max() < 0.3        # the take against itself: no slip


def test_pin_removes_the_swim():
    T, O, p0 = _scene()
    At = sa.audit(T, p0)
    before = _slip(O, T, p0).max()
    for _ in range(3):
        O, S, C = mpin.pin_pass(O, T, p0, At['phil'][0], At, sigma=0.0, max_px=16)
    after = _slip(O, T, p0).max()
    assert after < 0.35 * before and after < 1.5


def test_pin_never_moves_the_nose():
    T, O, p0 = _scene()
    At = sa.audit(T, p0)
    O2, _, _ = mpin.pin_pass(O, T, p0, At['phil'][0], At, sigma=0.0, max_px=16)
    for t in range(N):
        sn = int(np.floor(p0[2, 1] + At['phil'][0][t][1]))
        assert np.array_equal(O2[t][:sn], O[t][:sn])   # everything above the subnasale is untouched


def test_lag_frames_sign():
    t = np.arange(60)
    ref = np.sin(t / 3.0)
    late = np.r_[ref[0], ref[:-1]]
    L, r = sa.lag_frames(ref, late)
    assert abs(L - 1) < 0.3 and r > 0.9
