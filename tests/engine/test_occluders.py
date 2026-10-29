# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
import numpy as np
from lsdub_engine.core import occluders


def test_restore_keeps_take_pixels_under_the_mask_only():
    outs = [np.full((40, 40, 3), 200, np.uint8) for _ in range(2)]
    takes = [np.full((40, 40, 3), 10, np.uint8) for _ in range(2)]
    occ = np.zeros((2, 40, 40), np.float32)
    occ[0, 15:25, 15:25] = 1
    cov = occluders.restore(outs, takes, occ, grow=0, feather=0)
    assert cov[0] == 100 and cov[1] == 0
    assert outs[0][20, 20, 0] == 10 and outs[0][2, 2, 0] == 200
    assert (outs[1] == 200).all()


def test_load_resizes_and_pads(tmp_path):
    p = tmp_path / 'occ.npz'
    np.savez_compressed(p, masks=np.full((1, 10, 20), 255, np.uint8))
    m = occluders.load(str(p), 3, (20, 40))
    assert m.shape == (3, 20, 40) and m[0].min() == 1.0 and m[2].max() == 0.0
