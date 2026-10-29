# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
from lsdub_engine import mouth_morph, mouth_paint
import numpy as np


def test_flags_parse(capsys):
    try:
        mouth_paint.main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--transition" in out and "lipclose" in out and "--jaw-lock" in out


def _face():
    base = np.zeros((478, 2))
    base[:, 0] = 100.0
    base[:, 1] = np.linspace(0, 200, 478)
    base[17] = (100, 120)
    base[152] = (100, 160)
    base[61] = (80, 110)
    base[291] = (120, 110)
    return base


def test_jaw_lock_keeps_lip_to_chin_length():
    base = _face()
    L = np.repeat(base[None], 3, 0)
    L[1, 17, 1] += 12          # lower lip drops, chin stayed: lip->chin collapsed
    L[2, 152, 1] -= 8          # chin rode up toward the lip
    out, w = mouth_morph.jaw_lock(L, base)
    d = out[:, 152, 1] - out[:, 17, 1]
    assert np.allclose(d, base[152, 1] - base[17, 1])
    assert w[17] == 0 and w[152] == 1


