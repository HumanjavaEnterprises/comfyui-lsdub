"""v5 rigid teeth (teeth_layer.py + --warp-teeth rigid): teeth are bone. Synthetic, CPU only."""
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot

from lsdub_engine.core import teeth_layer as tl, lip_layers as ll, mouth_paint

SKIN = (120, 155, 205)       # BGR
LIP = (45, 40, 185)
TOOTH = (235, 240, 245)
CAV = (20, 15, 40)


class StubPose:
    """a RigidPlacement stand-in: canonical -> frame i = s_i R_i X + t_i (orthographic)."""

    def __init__(self, s, R, t):
        self.s, self.R, self.t = np.asarray(s, float), np.asarray(R, float), np.asarray(t, float)
        self.par = np.hstack([np.log(self.s)[:, None], Rot.from_matrix(self.R).as_rotvec(), self.t])

    def rigid3(self, i, X3): return self.s[i] * np.asarray(X3, float) @ self.R[i].T + self.t[i]
    def rigid(self, i, X3): return self.rigid3(i, X3)[:, :2]


def _P(c=(160.0, 130.0), hw=40, gap=20.0):
    """478 landmarks (z = 0): lips around c, eyes / nose bridge / chin placed for the head axes."""
    P = np.zeros((478, 3))
    P[:, :2] = (c[0], c[1] - 90)
    P[33, :2], P[263, :2] = (c[0] - 60, c[1] - 70), (c[0] + 60, c[1] - 70)
    P[168, :2] = (c[0], c[1] - 70)
    P[152, :2] = (c[0], c[1] + 60)
    a = np.linspace(np.pi, -np.pi, len(ll.LIP_OUT), endpoint=False)
    P[ll.LIP_OUT, :2] = np.stack([c[0] + hw * np.cos(a), c[1] + (gap / 2 + 10) * np.sin(a)], 1)
    P[ll.LIP_IN, :2] = np.stack([c[0] + 0.8 * hw * np.cos(a), c[1] + max(0.5, gap / 2) * np.sin(a)], 1)
    P[13, :2], P[14, :2] = (c[0], c[1] - gap / 2), (c[0], c[1] + gap / 2)
    P[61, :2], P[291, :2] = (c[0] - hw, c[1]), (c[0] + hw, c[1])
    return P


def _frame(P, tooth_w=8, tooth_h=9, H=260, W=320):
    """skin, red lips, dark cavity inside the inner lip, a row of upper incisors (tooth_w px each, 2 px dark gaps)
    hanging from the upper inner lip."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = SKIN
    cv2.fillPoly(img, [np.round(P[ll.LIP_OUT, :2]).astype(np.int32)], LIP)
    inner = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inner, [np.round(P[ll.LIP_IN, :2]).astype(np.int32)], 1)
    img[inner > 0] = CAV
    c = P[[13, 14], :2].mean(0)
    top = int(round(P[13, 1])) - 6
    teeth = np.zeros((H, W), np.uint8)
    for k in range(-3, 3):
        x0 = int(round(c[0] + k * (tooth_w + 2)))
        teeth[top:top + 6 + tooth_h, x0:x0 + tooth_w] = 1
    img[(teeth > 0) & (inner > 0)] = TOOTH
    return img


def _layer(gaps=(4.0, 20.0, 28.0, 12.0), poses=None, tooth_w=8):
    n = len(gaps)
    S = np.array([_P(gap=g) for g in gaps])
    if poses is None: poses = StubPose(np.ones(n), np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)))
    frames = [_frame(_P(gap=g), tooth_w) for g in gaps]
    F = np.stack([S[:, 13, 1] * 0 + np.array(gaps), np.full(n, 64.0)], 1)
    return tl.TeethLayer(frames, S, poses, S, F, up_idx=2, lo_idx=2), frames, S


def _teeth_cols(img, y0, y1):
    """per column, whether a bright (tooth) pixel sits in rows y0..y1."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return (lab[y0:y1, :, 0] > 200).any(0)


def test_upper_plate_size_constant_under_opening_changes():
    layer, frames, S = _layer()
    widths, edges = [], []
    for gap_t in (6.0, 24.0, 34.0, 44.0):
        # a take whose own teeth are STRETCHED (1.6x wider) and a target contour opened to gap_t
        take = _frame(_P(gap=gap_t), tooth_w=13)
        P_in = _P(gap=gap_t)[ll.LIP_IN, :2]
        out, m = layer.render(take, 0, P_in, gap_t)
        cols = np.nonzero(_teeth_cols(out, 100, 130) & (m.max(0) > 0.5))[0]
        runs = np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1)
        starts = [r[0] for r in runs[1:] if len(r)]                     # left edges of the whole teeth
        if gap_t > 10 and len(starts) >= 2: widths.append((float(np.median(np.diff(starts))), max(len(r) for r in runs)))
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0]
        cx = 160
        ys = np.nonzero(lab[:, cx + 3] > 200)[0]
        if len(ys): edges.append(ys.max())
    # every tooth stays the harvest frame's size (pitch 8 + 2 px gap = 10, width 8; the take's stretched teeth are 13 + 2),
    # whatever the opening
    assert len(widths) >= 2 and all(abs(p - 10) <= 0.5 and abs(w - 8) <= 1 for p, w in widths), widths
    # and the incisal edge does not move (upper teeth ride the skull, the pose is constant here)
    assert np.ptp(edges[1:]) <= 1, edges


def test_upper_teeth_fixed_to_the_skull_pose():
    n = 3
    R = Rot.from_euler('xyz', [[0, 0, 0], [0.1, -0.2, 0.05], [-0.15, 0.25, -0.1]]).as_matrix()
    pose = StubPose([1.0, 1.1, 0.9], R, [[0, 0, 0], [12, -7, 3], [-9, 5, -2]])
    layer, frames, S = _layer(gaps=(20.0, 20.0, 20.0), poses=pose)
    X = np.array([[150.0, 128.0], [172.0, 131.0], [161.0, 125.0]])                 # canonical points on the upper plate plane
    Ah = tl.plane_affine(pose, layer.h_up, layer.z_up, ref=layer.ref)
    xh = X @ Ah[:, :2].T + Ah[:, 2]                                                # where they sit in the harvest frame
    for i in range(n):
        for gap_t in (5.0, 30.0):                                                  # the opening never moves the upper teeth
            A_up, _ = layer.affines(i, gap_t)
            got = xh @ A_up[:, :2].T + A_up[:, 2]
            want = pose.rigid(i, np.hstack([X, np.full((3, 1), layer.z_up)]))
            assert np.abs(got - want).max() < 1e-6
        # no scaling beyond the pose's own: the plate's linear map = s_i x (projected rotation) relative to the harvest
        A_up, _ = layer.affines(i, 20.0)
        sv = np.linalg.svd(A_up[:, :2], compute_uv=False)
        assert sv.max() <= pose.s[i] / pose.s[layer.h_up] * 1.0 / np.cos(0.35) + 1e-6


def test_lower_teeth_rotate_about_the_hinge_without_scaling():
    layer, frames, S = _layer()
    u, d = layer.u, layer.d
    for th in (-0.2, 0.1, 0.3):
        R = tl.rot_axis(u, th)
        assert abs(np.linalg.det(R) - 1) < 1e-9 and np.allclose(R @ R.T, np.eye(3))   # a rotation: no scale, no shear in 3D
        assert np.allclose(R @ u, u)                                                    # about the lateral axis
        T = layer.T_lo
        T2 = (T - layer.H) @ R.T + layer.H
        assert abs(np.linalg.norm(T2 - layer.H) - np.linalg.norm(T - layer.H)) < 1e-9   # the incisor stays on its circle
    # the jaw angle from a target drop reproduces that drop along the head's down axis, opening = positive angle
    for delta in (-8.0, 5.0, 15.0):
        th = tl.jaw_angle(layer.T_lo, layer.H, u, d, delta)
        T2 = (layer.T_lo - layer.H) @ tl.rot_axis(u, th).T + layer.H
        assert abs((T2 - layer.T_lo) @ d - delta) < 1e-6 and np.sign(th) == np.sign(delta)
    # the projected lower plate keeps its width (lateral axis untouched) at any opening, only foreshortens vertically
    for gap_t in (8.0, 20.0, 34.0):
        _, A_lo = layer.affines(2, gap_t)
        ex = A_lo[:, :2] @ np.array([1.0, 0.0])
        assert abs(np.linalg.norm(ex) - 1.0) < 1e-6
        ey = A_lo[:, :2] @ np.array([0.0, 1.0])
        assert np.linalg.norm(ey) <= 1.0 + 1e-6


def test_render_only_inside_the_mouth_and_flag_parses(capsys):
    layer, frames, S = _layer()
    take = _frame(_P(gap=20.0))
    out, m = layer.render(take, 2, _P(gap=20.0)[ll.LIP_IN, :2], 20.0)
    far = np.ones(take.shape[:2], bool)
    far[60:200, 80:240] = False
    assert (out[far] == take[far]).all() and (m[far] == 0).all()
    try: mouth_paint.main(['--help'])
    except SystemExit: pass
    o = capsys.readouterr().out
    for f in ('--warp-teeth', '--warp-teeth-hinge', '--warp-teeth-jaw-gain'): assert f in o


# ------------------------------------------------------------------------------------------- v6 (anchor + jaw plan) --
def _skin_seq(shifts, H=240, W=320):
    """a textured skin patch (the upper-lip root) moved by integer shifts per frame; landmarks for it."""
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur((rng.random((H + 40, W + 40, 3)) * 255).astype(np.uint8), (0, 0), 1.5)
    frames, S = [], []
    for dx, dy in shifts:
        frames.append(np.ascontiguousarray(base[20 - dy:20 - dy + H, 20 - dx:20 - dx + W]))
        P = np.zeros((478, 3))
        P[:, :2] = (160 + dx, 120 + dy)
        a = np.linspace(0, 2 * np.pi, len(tl.ROOT), endpoint=False)
        P[tl.ROOT, :2] = np.stack([160 + dx + 30 * np.cos(a), 120 + dy + 20 * np.sin(a)], 1)
        P[33, :2], P[263, :2] = (100 + dx, 60 + dy), (220 + dx, 60 + dy)
        S.append(P)
    return frames, np.array(S)


def test_teeth_share_the_upper_lip_root_motion():
    # the skin jitters +-2 px frame to frame while the pose (here: still) does not see it; the pin must hand the skin's
    # every step to the teeth, and with no high-pass the absolute offset too
    sh = [(0, 0), (2, 1), (0, -1), (3, 2), (1, 0), (4, 2), (2, 1), (5, 3), (3, 1), (6, 2)]
    frames, S = _skin_seq(sh)
    pose = np.tile([160.0, 120.0], (len(sh), 1))
    c = tl.skin_pin(frames, S, pose, 0, sigma=0, noise=0, iod=120.0)
    assert np.abs(c - (np.array(sh, float) - sh[0])).max() < 0.35, c
    c4 = tl.skin_pin(frames, S, pose, 0, sigma=4.0, noise=0, iod=120.0)        # default: slow part left to the pose
    assert np.abs(np.diff(c4, axis=0) - np.diff(np.array(sh, float), axis=0)).max() < 0.8
    # the plates carry the pin: upper AND lower maps are the v5 maps translated by exactly the pin of that frame
    layer, _, _ = _layer()
    A0 = layer.affines(1, 20.0)
    layer.pin = np.tile([[3.5, -2.0]], (4, 1))
    A1 = layer.affines(1, 20.0)
    for a0, a1 in zip(A0, A1):
        assert np.allclose(a0[:, :2], a1[:, :2]) and np.allclose(a1[:, 2] - a0[:, 2], [3.5, -2.0])


def test_jaw_plan_smooth_anticipatory_hidden_on_closure():
    n = 80
    t = np.arange(n)
    rng = np.random.default_rng(0)
    clean = np.radians(8) * np.sin(2 * np.pi * t / 20.0)
    noisy = clean + np.radians(1.5) * rng.standard_normal(n)
    th = tl.plan_jaw(noisy)
    jerk = lambda x: np.percentile(np.abs(np.diff(np.degrees(x), 3)), 95)
    assert jerk(th) < 0.3 * jerk(noisy)                                       # smooth
    assert np.abs(np.diff(th)).max() <= np.radians(3.0) + 1e-3                # speed limit (degrees / frame)
    assert np.median(np.abs(th - clean)) < np.radians(1.0)                    # follows the target
    # zero-phase (no lag behind the target), and an explicit lead moves it earlier
    def lag(x):
        c = [np.corrcoef(x[5 + k:n - 5 + k], clean[5:n - 5])[0, 1] for k in range(-4, 5)]
        return int(np.argmax(c)) - 4
    assert lag(th) == 0
    assert lag(tl.plan_jaw(noisy, lead=2.0)) == -2
    # near-closed frames: the planned jaw sits at or past the hide angle (lower teeth below the lower lip), even when the
    # target asks for less
    hide = np.full(n, np.radians(6.0))
    hw = ((t >= 30) & (t < 45)).astype(float)
    th2 = tl.plan_jaw(noisy, hide=hide, hide_w=hw)
    assert (th2[hw > 0] >= np.radians(6.0) - np.radians(0.3)).all()
    # sustained vowels hold still
    hold = (t >= 60) & (t < 70)
    th3 = tl.plan_jaw(noisy, hold=hold)
    assert np.abs(np.diff(th3[61:69])).max() < np.abs(np.diff(th[61:69])).max()


def test_v6_flags_parse(capsys):
    try: mouth_paint.main(['--help'])
    except SystemExit: pass
    o = capsys.readouterr().out
    for f in ('--warp-teeth-anchor', '--warp-teeth-jaw', '--warp-teeth-tongue-drop', '--warp-teeth-plates', '--warp-teeth-throat',
              '--warp-teeth-mirror', '--warp-teeth-hide'): assert f in o
