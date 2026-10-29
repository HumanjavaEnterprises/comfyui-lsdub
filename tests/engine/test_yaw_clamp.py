# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v12 --warp-yaw-clamp (mouth_warp.YawClamp / face_region) on a synthetic turned face, CPU only: the take's face ends at a
hair line just past its far mouth corner (the face mesh's own outline ends inside the cheek, as on mv30 NC); a wider planned
mouth whose far corner crosses onto the hair is clamped back inside, and a plan that stays inside is returned untouched."""
import os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
from test_frame_gate import face, draw
from lsdub_engine.core import mouth_warp as mw

HAIR = (35, 30, 95)                    # dark auburn (BGR): not skin


class FakeClip:
    def __init__(self, S, frames):
        self.S, self.frames, self.n = S, frames, len(S)
        self.H, self.W = frames[0].shape[:2]


def scene(hair_x=112):
    P = face(16.0, c=(160.0, 150.0))
    P[1, :2] = (140.0, 112.0)                    # the nose tip turned toward the left corner
    a = np.linspace(-np.pi, np.pi, len(mw.FACE_OVAL), endpoint=False)
    P[mw.FACE_OVAL, :2] = np.stack([160 + 42 * np.cos(a), 140 + 120 * np.sin(a)], 1)   # the mesh outline ends inside the cheek (x 118)
    img = draw(P)
    img[:, :hair_x] = HAIR
    return P, img


def outer_min_dist(img_face, dst, io):
    Q = dst[:20]
    d = cv2.distanceTransform(img_face.astype(np.uint8), cv2.DIST_L2, 5)
    m = np.zeros(img_face.shape, np.uint8)
    cv2.fillPoly(m, [np.round(Q).astype(np.int32)], 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(0.02 * io) + 1,) * 2))
    return float(d[m > 0].min())


def test_face_region_is_skin_and_the_mesh_never_the_hair():
    P, img = scene()
    io = float(np.linalg.norm(P[33, :2] - P[263, :2]))
    f = mw.face_region(img, P, io, (0, 0, img.shape[1], img.shape[0]))
    assert f[150, 120] and f[150, 160] and f[150, 200]                 # skin beyond the mesh outline, the mouth, the near cheek
    assert not f[150, 100] and not f[60, 105]                          # the hair is never face


def test_yaw_clamp_pulls_the_far_corner_inside_and_leaves_an_inside_plan_alone():
    P, img = scene()
    n = 3
    S = np.array([P] * n)
    clip = FakeClip(S, [img] * n)
    io = float(np.linalg.norm(P[33, :2] - P[263, :2]))
    src = P[mw.CTRL, :2].copy()
    ix = {k: j for j, k in enumerate(mw.CTRL)}
    wide = src.copy()
    c = src[[ix[61], ix[291]]].mean(0)
    wide[:, 0] = c[0] + (src[:, 0] - c[0]) * 1.6    # far corner -> x 106: hair
    inside = src.copy()
    inside[:, 1] = c[1] + (src[:, 1] - c[1]) * 1.3                                 # taller, not wider
    yc = mw.YawClamp(clip, 0.01, [src] * n, [wide, inside, inside])
    assert yc.need[0] > 0 and yc.need[1] == 0 and yc.need[2] == 0 and yc.b[1] > 0 and yc.b[2] > 0      # eased into the neighbours
    out = yc.apply(0, src, wide, ent := {})
    assert ent['yaw_clamp'] > 0
    assert out[ix[61], 0] > wide[ix[61], 0] + 3 and np.allclose(out[ix[291]], wide[ix[291]])         # only the far half moves
    fm = mw.face_region(img, P, io, (0, 0, img.shape[1], img.shape[0]))
    # no lip pixel (the outer polygon + the lip layer's 0.02 IOD) on the hair: M = 0.01 IOD to spare (less 1 px of rounding);
    # the unclamped plan's lip pixels reach the hair
    assert outer_min_dist(fm, out, io) >= 0.01 * io - 1.0 and outer_min_dist(fm, wide, io) == 0
    # the corner cover's face test on the clamped frames (LayerCompositor.face_of): the hair is not face; None where b = 0
    fb = yc.face_box(0, (90, 120, 60, 60))
    assert fb is not None and not fb[30, 5] and fb[30, 40]
    # a frame whose plan stays inside the face (and no eased share): the very same array back
    yc2 = mw.YawClamp(clip, 0.01, [src] * n, [inside] * n)
    assert yc2.b.max() == 0
    assert yc2.apply(1, src, inside) is inside and yc2.face_box(1, (90, 120, 60, 60)) is None
