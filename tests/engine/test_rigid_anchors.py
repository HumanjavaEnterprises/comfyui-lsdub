"""--anchors strict: the painted mouth is placed by the SKULL, never by the take's own mouth (CPU, synthetic only).

A rigid head moves (yaw/roll/translation/scale) while the take's mouth opens and closes; with ANCH/TPS the cheeks, alar
base and jaw/oval anchors follow the opening, so the placed philtrum moves with it (r ~0.5 on real takes). With the
strict rigid set the placed philtrum must be independent of the opening."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lsdub_engine.core import mouth_morph as mm

T = 60
PHIL = [0, 164, 37, 267]          # philtrum / upper-lip border: rigid with the upper teeth


def _face(seed=3):
    """canonical 3D face: 478 points, upper face y<0, lower face y>0 (px-like units, z = depth)."""
    rng = np.random.default_rng(seed)
    P = np.zeros((478, 3))
    P[:, 0] = rng.uniform(-80, 80, 478)
    P[:, 1] = rng.uniform(-90, 110, 478)
    P[:, 2] = rng.uniform(-30, 10, 478)
    for k, (x, y, z) in {168: (0, -40, 5), 6: (0, -25, 8), 197: (0, -12, 14), 195: (0, -2, 18), 5: (0, 8, 22), 4: (0, 14, 25),
                         1: (0, 18, 26), 33: (-45, -38, -6), 133: (-18, -36, -2), 362: (18, -36, -2), 263: (45, -38, -6),
                         0: (0, 42, 10), 164: (0, 32, 14), 37: (-8, 41, 9), 267: (8, 41, 9)}.items():
        P[k] = (x, y, z)
    return P


def _take(P, rng):
    """per frame: rigid pose + the take's own mouth opening (moves lower face, cheeks by the mouth, alar base, oval)."""
    t = np.arange(T)
    o = 0.5 + 0.5 * np.sin(2 * np.pi * t / 9.0)          # open/close ~2.8 Hz, unrelated to the head
    yaw = 12 * np.sin(2 * np.pi * t / T)
    roll = 4 * np.cos(2 * np.pi * t / T)
    tx, ty = 300 + 2.0 * t, 200 + 25 * np.sin(2 * np.pi * t / T)
    s = 1.4 + 0.1 * np.sin(np.pi * t / T)
    nonrigid = sorted(set(range(478)) - set(mm.ANCH_STRICT) - set(PHIL))
    S3, truth = [], []
    for i in range(T):
        Q = P.copy()
        low = [k for k in nonrigid if P[k, 1] > 20]
        Q[low, 1] += 18 * o[i] * (P[low, 1] - 20) / 90              # jaw drop
        for k in (205, 425, 50, 280, 187, 411, 98, 327, 132, 361, 93, 323, 234, 454):   # cheeks / alar / oval ride the mouth
            Q[k, 1] += 9 * o[i]
            Q[k, 0] *= 1 + 0.06 * o[i]
        R = Rot.from_euler('yz', [yaw[i], roll[i]], degrees=True).as_matrix()
        tr = np.array([tx[i], ty[i], 0.0])
        S3.append(s[i] * Q @ R.T + tr + rng.normal(0, 0.15, Q.shape))   # small detector noise
        truth.append((s[i] * P[PHIL] @ R.T + tr)[:, :2])
    return np.array(S3), np.array(truth), o


def _anch_place(P, S3):
    out = []
    for i in range(T):
        tm = mm.tps_fit(P[mm.ANCH, :2], S3[i][mm.ANCH, :2], lam=1e-3)
        out.append(mm.tps_apply(tm, P[PHIL, :2]))
    return np.array(out)


def test_strict_philtrum_independent_of_take_mouth_opening():
    P = _face()
    S3, truth, o = _take(P, np.random.default_rng(0))
    pl = mm.RigidPlacement(P, S3, mm.ANCH_STRICT, smooth='sg13', lam=10.0)
    strict = np.array([pl.apply(i, P[PHIL]) for i in range(T)])
    anch = _anch_place(P, S3)
    e_s = np.linalg.norm(strict - truth, axis=2).mean(1)
    e_a = np.linalg.norm(anch - truth, axis=2).mean(1)
    # error of the placed philtrum (vertical) must not follow the opening
    r_s = abs(np.corrcoef((strict - truth)[:, :, 1].mean(1), o)[0, 1])
    r_a = abs(np.corrcoef((anch - truth)[:, :, 1].mean(1), o)[0, 1])
    assert np.percentile(e_s, 95) < 1.0, e_s           # sub-pixel on a ~150 px face
    assert r_s < 0.3, r_s
    assert r_a > 0.5 and np.percentile(e_a, 95) > 3 * np.percentile(e_s, 95), (r_a, e_a.max())   # the old path does slosh


def test_strict_rigid_only_and_inverse():
    P = _face()
    S3, truth, o = _take(P, np.random.default_rng(1))
    pl = mm.RigidPlacement(P, S3, mm.ANCH_STRICT, smooth='none', lam=0)
    for i in (0, 17, 59):
        back = pl.to_lib(i, pl.rigid3(i, P))
        assert np.allclose(back, P, atol=1e-6)
        assert np.linalg.norm(pl.apply(i, P[PHIL]) - truth[i], axis=1).max() < 1.0


def test_pose_smoothing_zero_lag():
    """constant velocity / acceleration head moves pass the parameter filter unchanged (no lag)."""
    t = np.arange(40, dtype=float)[:, None]
    X = np.hstack([0.5 * t, 0.02 * t ** 2, -t])
    for m in ('sg11', 'sg13', 'sg15'):
        assert np.allclose(mm.zero_phase(X, m), X, atol=1e-6), m
    f = mm.zero_phase(X[:, :1], 'ff13')
    assert np.abs(f[5:-5] - X[5:-5, :1]).max() < 0.05      # filtfilt: zero phase on a ramp


def test_plane_lift():
    P3 = np.array([[0, 0, 1.0], [10, 0, 3.0], [0, 10, -1.0], [10, 10, 1.0]])
    z = mm.plane_lift(np.array([[5.0, 5.0], [20.0, 0.0]]), P3)[:, 2]
    assert np.allclose(z, [1.0, 5.0])
