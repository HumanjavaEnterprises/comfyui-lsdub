# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
import numpy as np
from lsdub_engine import mouth_morph, mouth_paint


def test_flag_parses(capsys):
    try:
        mouth_paint.main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--clean-plate" in out and "--plate-debug" in out


def test_skin_px_keeps_skin_drops_earring_and_mic():
    # BGR: warm skin, saturated red earring, grey mic, green earring
    px = np.array([[[90, 130, 190], [20, 20, 230], [140, 140, 140], [40, 220, 40]]], np.uint8)
    assert mouth_morph.skin_px(px).tolist() == [[True, False, False, False]]
