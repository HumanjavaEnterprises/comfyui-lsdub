# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v17 --warp-cavity fill: tongue + lower row painted in the opening; nothing on a near-closed mouth."""
import types
import numpy as np
from lsdub_engine.core.mouth_layers import LayerCompositor


def _stub():
    t = types.SimpleNamespace(light=np.array([150.0]), skin_med=150.0)
    return types.SimpleNamespace(t=t, cavity_fill=1.0, tongue_c=np.array([96.0, 103.0, 195.0]))


def _ellipse(cx, cy, rx, ry, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def test_open_mouth_gets_tongue_and_lowers():
    io = 280.0
    h, w = 120, 200
    up = np.zeros((h, w, 3), np.float32)
    up[30:45, 60:140] = 220
    ua = np.zeros((h, w), np.float32)
    ua[30:45, 60:140] = 1
    st = _stub()
    st.tongue_fill = 1.0
    rgb, a = LayerCompositor.fill_cavity(st, 0, _ellipse(100, 60, 70, 30), h, w, io, up, ua)
    assert a.max() > 0.9
    assert a[20:45].mean() < a[60:90].mean()                 # the fill sits low (the back / top stays the cavity's)
    lo = rgb[80:86, 90:110][a[80:86, 90:110] > 0.5]
    assert lo.size and lo.mean() > 100                        # a bright lower-incisor row near the floor
    tg = rgb[70, 100]
    assert tg[2] > tg[0] + 30              # pink tongue (BGR: red > blue)


def test_near_closed_mouth_untouched():
    io = 280.0
    h, w = 60, 200
    rgb, a = LayerCompositor.fill_cavity(_stub(), 0, _ellipse(100, 30, 70, 4), h, w, io, np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32))
    assert a.max() < 1e-3


def test_tongue_fill_lowers_the_tongue():
    io = 280.0
    h, w = 120, 200
    z = np.zeros((h, w, 3), np.float32)
    za = np.zeros((h, w), np.float32)
    cov = []
    for tk in (1.0, 0.6):
        st = _stub()
        st.tongue_fill = tk
        rgb, a = LayerCompositor.fill_cavity(st, 0, _ellipse(100, 60, 70, 30), h, w, io, z, za)
        cov.append(float((a[:, 95:105] > 0.5).sum()))
    assert cov[1] < cov[0]                                    # less present = a lower dorsum
