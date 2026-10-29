"""plane-fit: the 3D similarity recovers a head rotation, and the arc metrics see a flat (unrotated) mouth."""
import numpy as np

from lsdub_engine.audits.plane_fit import ARC, RIG, arc_shape, sim3


def _face(seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 40, (478, 3))
    P[33] = [-60, 0, 0]
    P[263] = [60, 0, 0]                    # eye corners: level eye line
    P[0] = [0, 80, -20]
    P[40] = [-25, 82, -10]
    P[270] = [25, 82, -10]   # upper-lip arc on a curve (centre forward)
    P[61] = [-40, 90, 0]
    P[291] = [40, 90, 0]
    return P


def _rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def test_sim3_recovers_pose():
    X = _face()
    R = _rot_y(12) * 1.3
    t = np.array([5.0, -3.0, 2.0])
    Y = X @ R.T + t
    sR, tt = sim3(X[RIG], Y[RIG])
    assert np.allclose(X @ sR.T + tt, Y, atol=1e-6)


def test_arc_symmetric_when_frontal():
    X = _face()
    w = np.linalg.norm(X[291, :2] - X[61, :2])
    sag, asym = arc_shape(X, w)
    assert abs(asym) < 1e-9 and sag != 0


def test_flat_mouth_is_detected_on_yaw():
    X = _face()
    R = _rot_y(20)
    Y = X @ R.T                 # head turns 20 deg
    flat = Y.copy()
    flat[ARC] = X[ARC]                       # the mouth did NOT turn with it (pasted flat)
    w = np.linalg.norm(Y[291, :2] - Y[61, :2])
    _, a_pred = arc_shape(Y, w)
    _, a_flat = arc_shape(flat, w)
    assert abs(a_pred) > 0.02                                 # a real turned mouth is foreshortened on one side
    assert abs(a_flat - a_pred) > 0.02                        # the flat one is not: the metric sees it
