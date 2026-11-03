# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""spaces: a crop-space occluder mask is passed to the crop paint as it is, mapped back to the take for the native paint with its
grow in take px, and refused in any other shape. A synthetic mic: a disk moving across a 96 x 160 take, n=6, m=2."""

import cv2, numpy as np
import pytest

from lsdub_engine.core import occluders, spaces as SP
from lsdub_engine.core.medium import crop_matrix

H, W, N, MM = 96, 160, 6, 2
S, Z = 40.0, 2.5
O = int(S * Z)
CENTRES = [(70.0 + 2 * k, 50.0 + k) for k in range(N)]
R = dict(per_frame=[{"centre": list(c)} for c in CENTRES], crop_src_px=S, crop_px=O, zoom=Z)


def disk(k):
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (int(round(66 + 2 * k)), int(round(56 + k))), 8, 255, -1)
    return m


def crop_masks():
    return np.stack(
        [cv2.warpAffine(disk(j // MM), crop_matrix(*CENTRES[j // MM], S, Z), (O, O), flags=cv2.INTER_LINEAR) for j in range(N * MM)]
    )


def square(k):
    """the take px the crop of take frame k sees."""
    Ai = cv2.invertAffineTransform(crop_matrix(*CENTRES[k], S, Z))
    return cv2.warpAffine(np.full((O, O), 255, np.uint8), Ai, (W, H)) > 0


def iou(a, b):
    a, b = a > 127, b > 127
    return (a & b).sum() / max(1, (a | b).sum())


def save(path, masks, **kw):
    np.savez_compressed(path, masks=masks, **kw)
    return str(path)


def bound(tmp_path):
    p = save(tmp_path / "occ.npz", crop_masks(), phrases=np.array(["microphone"]), coverage=np.ones((1, N * MM), np.float32))
    return SP.Occluder.open(p, N * MM).bind(SP.CropSpace.from_report(R, MM))


def test_crop_mask_passes_through_unchanged(tmp_path):
    occ = bound(tmp_path)
    before = open(occ.path, "rb").read()
    path, g = occ.for_space(SP.CropSpace.from_report(R, MM), 5)
    assert path == occ.path and g == 5
    assert open(path, "rb").read() == before


def test_back_map_lands_on_the_take_mask(tmp_path):
    occ = bound(tmp_path)
    out = str(tmp_path / "occ_native.npz")
    path, _ = occ.for_space(SP.NativeSpace(N, (H, W)), 3, out)
    with np.load(path) as z:
        M, files = z["masks"], set(z.files)
        cov = z["coverage"]
    assert M.shape == (N, H, W) and M.dtype == np.uint8
    assert files == {"masks", "phrases", "coverage"} and cov.shape == (1, N)
    for k in range(N):
        sq = square(k)
        assert iou(M[k] * sq, disk(k) * sq) > 0.98
        assert M[k][~sq].max() == 0


def test_frame_relation():
    c = SP.CropSpace.from_report(R, MM)
    assert [c.take_frame(j) for j in range(N * MM)] == [j // MM for j in range(N * MM)]
    assert c.n == N * MM and c.shape == (O, O) and c.scale() == Z
    nat = SP.NativeSpace(N, (H, W))
    assert [nat.take_frame(j) for j in range(N)] == list(range(N)) and nat.scale() == 1.0


@pytest.mark.parametrize(
    "shape, frames",
    [((N, H, W), N * MM), ((N * MM, O, O + 2), N * MM), ((N * MM + 1, O, O), N * MM), ((N * MM - 1, O, O), N * MM), ((N * MM, O), N * MM)],
)
def test_other_shapes_are_refused(tmp_path, shape, frames):
    p = save(tmp_path / "occ.npz", np.zeros(shape, np.uint8))
    with pytest.raises(ValueError) as e:
        SP.Occluder.open(p, frames, how="make it on crop.mp4")
    assert str(shape) in str(e.value) and f"{frames} square frames" in str(e.value) and "make it on crop.mp4" in str(e.value)


def test_side_not_the_crop_is_refused(tmp_path):
    p = save(tmp_path / "occ.npz", np.zeros((N * MM, O + 4, O + 4), np.uint8))
    occ = SP.Occluder.open(p, N * MM)
    with pytest.raises(ValueError) as e:
        occ.bind(SP.CropSpace.from_report(R, MM))
    assert f"{O + 4} px" in str(e.value) and f"{O} px" in str(e.value)


@pytest.mark.parametrize("g", [0, 3, 18])
@pytest.mark.parametrize("z", [1.0, 2.46, 2.768])
def test_grow_in_take_px(g, z):
    c = SP.CropSpace([(0, 0)], 100.0, int(100 * z), 2, z)
    assert SP.scaled_grow(g, c, SP.NativeSpace(1, (1, 1))) == max(0, round(g / z))


def test_grow_last_wins():
    assert SP.occluder_grow([]) == 3
    assert SP.occluder_grow(["--occluder-grow", "5", "--x", "1", "--occluder-grow", "18"]) == 18
    assert SP.occluder_grow(["--occluder-grow", "5", "--occluder-grow=9"]) == 9
    assert SP.occluder_grow(["--occluder-grow=9", "--occluder-grow", "4"]) == 4


def test_native_flags(tmp_path):
    occ = bound(tmp_path)
    out = str(tmp_path / "occ_native.npz")
    flags = SP.native_occluder(occ, N, (H, W), ["--occluder-plate", "--occluder-grow", "18"], out)
    assert flags == ("--occluder-mask", out, "--occluder-grow", str(round(18 / Z)))


def zone(m, grow):
    """occluders.restore's blend weight > 0.5 for one mask (0..1)."""
    outs, takes = [np.zeros(m.shape + (3,), np.uint8)], [np.full(m.shape + (3,), 255, np.uint8)]
    occluders.restore(outs, takes, [m.astype(np.float32)], grow)
    return outs[0][..., 0] > 127


def test_restore_in_take_px_matches_the_crop(tmp_path):
    occ = bound(tmp_path)
    out = str(tmp_path / "occ_native.npz")
    path, g_n = occ.for_space(SP.NativeSpace(N, (H, W)), 3, out)
    nat = occluders.load(path, N, (H, W))
    crop = occluders.load(occ.path, N * MM, (O, O))
    outs = [np.full((H, W, 3), 200, np.uint8) for _ in range(N)]
    takes = [np.full((H, W, 3), 10, np.uint8) for _ in range(N)]
    occluders.restore(outs, takes, nat, g_n)
    ring = np.ones((3, 3), np.uint8)
    for k in range(N):
        inner = cv2.erode(disk(k), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0  # 3 feather sigmas in from the edge
        assert inner.sum() > 30 and (outs[k][inner] == 10).all()
        zn = zone(nat[k], g_n)
        zc = (
            cv2.warpAffine(
                (zone(crop[k * MM], 3) * 255).astype(np.uint8), cv2.invertAffineTransform(crop_matrix(*CENTRES[k], S, Z)), (W, H)
            )
            > 127
        )
        assert not (zn & ~(cv2.dilate(zc.astype(np.uint8), ring) > 0)).any()
        assert not (zc & ~(cv2.dilate(zn.astype(np.uint8), ring) > 0)).any()
