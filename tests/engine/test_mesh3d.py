# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""--mouth-render mesh3d: the library mouth is a 3D mesh projected by the rigid head pose (CPU, synthetic only).

A frontal library mouth turned by a pure yaw must foreshorten like a real one: on a flat lip line the projected
half-width is cos(yaw) of the frontal one; on a curved (wrap-around) lip line the FAR corner shortens more than cos(yaw)
and the near one less. A triangle that wraps past the silhouette flips its winding and is culled (left to the take)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lsdub_engine.core import mouth_morph as mm


def _lib():
    """478 x 3 library points: skull anchors with depth + a mouth (61 = frame-left corner, 291 = right, 13 = centre)."""
    rng = np.random.default_rng(0)
    P = np.zeros((478, 3))
    P[:, 0] = rng.uniform(-80, 80, 478)
    P[:, 1] = rng.uniform(-90, 110, 478)
    P[:, 2] = rng.uniform(-30, 10, 478)
    P[61] = (-30, 50, 0)
    P[291] = (30, 50, 0)
    P[13] = (0, 50, 0)
    return P


def _place(P, yaws, flat=True):
    if not flat:
        P = P.copy()
        P[[61, 291], 2] = 18.0                       # corners wrap back (farther from camera)
    S3 = np.array([P @ Rot.from_euler('y', y, degrees=True).as_matrix().T * 1.5 + [400, 300, 0] for y in yaws])
    return P, mm.RigidPlacement(P, S3, mm.MESH_ANCH, smooth='none', lam=0.0)


def test_flat_lip_line_foreshortens_by_cos_yaw():
    yaws = np.array([0, 10, 20, 30, 40])
    P, pl = _place(_lib(), yaws)
    for i, y in enumerate(yaws):
        q = pl.apply(i, P[[61, 291, 13]])
        for c in (0, 1):
            w = abs(q[c, 0] - q[2, 0]) / (30 * 1.5)
            assert abs(w - np.cos(np.radians(y))) < 1e-3


def test_curved_lip_far_corner_shortens_more_than_cos():
    yaws = np.array([0, 25])
    P, pl = _place(_lib(), yaws, flat=False)
    q = pl.apply(1, P[[61, 291, 13]])
    c = np.cos(np.radians(25))
    left, right = abs(q[0, 0] - q[2, 0]) / 45, abs(q[1, 0] - q[2, 0]) / 45
    far, near = sorted([left, right])
    assert far < c - 0.05 and near > c + 0.05
    # the far corner is the one with the larger depth after the pose
    z = pl.rigid3(1, P[[61, 291]])[:, 2]
    assert (left < right) == (z[0] > z[1])


def test_pwa3d_culls_triangles_that_turn_away():
    # a front quad + a side quad wrapping back at the corner. Turned -50 deg the side quad's winding flips and it folds
    # over the front one: plain pwa paints its (dark) texture across the front lip; pwa3d culls it
    import cv2
    V = np.array([[0, 0, 0], [40, 0, 0], [40, 40, 0], [0, 40, 0],
                  [40, 0, 0], [50, 0, 30], [50, 40, 30], [40, 40, 0]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    src = np.full((60, 200, 3), 200, np.uint8)
    src[:, 100:] = 50
    sp = np.vstack([V[:4, :2] + 10, V[4:, :2] + [100, 10]])               # front texture bright, side texture dark
    ref = np.sign(mm.tri_area(np.vstack([V[:4, :2], V[4:, :2] + [0, 0]]), tris))
    R = Rot.from_euler('y', -50, degrees=True).as_matrix()
    Q = V @ R.T + [60, 20, 0]
    box = (0, 0, 160, 100)
    assert np.sign(mm.tri_area(Q, tris[2:])).tolist() != ref[2:].tolist()   # the side quad turned away
    front = np.zeros((100, 160), np.uint8)
    for t in tris[:2]: cv2.fillConvexPoly(front, np.round(Q[t, :2]).astype(np.int32), 1)
    inner = cv2.erode(front, np.ones((3, 3), np.uint8)) > 0
    flat, _ = mm.pwa(src, sp, Q[:, :2], tris, box)
    out, ok = mm.pwa3d(src, sp, Q, Q[:, 2], tris, box, ref)
    assert (flat[inner][:, 0] < 100).mean() > 0.2                          # 2D warp: the far side folds over the lip
    assert (out[inner][:, 0] < 100).mean() < 0.01 and ok[inner].mean() > 0.95
