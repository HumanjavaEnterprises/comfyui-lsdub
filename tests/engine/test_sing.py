# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub sing + snap: frame picking keeps closures and peaks, the mic check, and the snap metrics (synthetic, no models)."""
import numpy as np

from lsdub_engine.core import sing as S
from lsdub_engine.audits import snap as SN


def test_pick_extremes_keeps_the_closure_and_the_peak():
    # internal 2x: a closure that only the odd internal frame reaches, and a peak only the odd frame reaches
    e = np.array([5, 5, 5, 5, 4, 0, 4, 5, 5, 5, 6, 12, 6, 5, 5, 5], float)
    t = np.array([5, 5, 1, 5, 5, 9, 5, 5], float)       # audio: closure at k=2, peak at k=5
    idx = S.pick_extremes(e, t, 2)
    assert idx[2] == 5 and e[idx[2]] == 0                # the most-closed internal frame near k=2
    assert idx[5] == 11 and e[idx[5]] == 12              # the most-open near k=5
    assert idx[0] == 0 and idx[3] == 6 and len(idx) == 8 # elsewhere: the on-time frame k*m
    assert S.pick_extremes(e[::2], t, 1) == list(range(8))


def test_mic_near_lips():
    crop = np.zeros((600, 600, 3), np.uint8)
    crop[:] = (90, 120, 200)
    assert not S.mic_near_lips(crop, 300)[1]
    crop[360:480, 230:370] = (185, 185, 190)              # a chrome mic head under the mouth
    assert S.mic_near_lips(crop, 300)[1]


def test_snap_metrics_pieces():
    fps = 25.0
    env = np.r_[np.zeros(10), np.full(10, 0.2), np.zeros(10)]
    op_fast = np.r_[np.zeros(10), [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.zeros(10)]
    op_slow = np.r_[np.zeros(10), [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1, 1, 1, 1], np.zeros(10)]
    rf, n1 = SN.rise_ms(op_fast, env, fps)
    rs, _ = SN.rise_ms(op_slow, env, fps)
    assert n1 == 1 and rf == 0.0 and rs == 120.0          # half the rise reached on the onset frame vs 3 frames later
    assert SN.open_in_silence(op_fast, env) < 0.05 and SN.open_in_silence(1 - op_fast, env) > 0.9
    words = [{'phones': [{'p': 'M', 'start': 0.40, 'end': 0.44}, {'p': 'AA', 'start': 0.44, 'end': 0.6},
                         {'p': 'B', 'start': 0.80, 'end': 0.84}]}]
    fr = SN.closures(words, 0.0, 30, fps)
    assert fr == [11, 20]
    op = 0.5 + 0.5 * np.abs(np.sin(np.arange(30) / 3.0))
    op[11] = 0.0
    op[10] = op[12] = 0.6
    op[19:22] = 1.0                                          # f20 stays open
    h, nh = SN.closure_hit(op, fr)
    assert nh == 2 and h == 0.5                           # the M closes, the B does not


def test_gate_no_mouth_has_all_keys(monkeypatch, tmp_path):
    import numpy as np
    from lsdub_engine.core import sing as S
    from lsdub_engine.audits import sync_qa as Q
    monkeypatch.setattr(S.MED, 'read', lambda v, n=None: ([np.zeros((64, 64, 3), np.uint8)] * 4, 25.0))
    monkeypatch.setattr(Q, 'measured_opening', lambda B, roi, fps: (None, 0.0))
    R = {'per_frame': [{'centre': (32, 32)}], 'face_h_px': 20}
    g = S.gate('x.mp4', R, np.zeros(4), 'v.wav', 'a.json', 0.0, 25.0, work=str(tmp_path))
    for k in ('corr', 'best', 'lag_ms', 'open_in_silence', 'rise_ms', 'closure_hit', 'pass'): assert k in g
    assert g['pass'] is False
