# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sing.retimed_align reads the vocal over the shot's own span (t0 .. t0 + n / fps), not from t0 to the song's end (0.2.1): the rest of
the song does not set the judge's loudness floor for this shot's keep/drop decision."""

import json

import numpy as np

from lsdub_engine.audits import sync_truth as ST
from lsdub_engine.core import sing as S, viseme_track as VT, vocal_retime as VR


def test_the_plan_truth_reads_only_the_shots_samples(monkeypatch, tmp_path):
    a, v = tmp_path / "a.json", tmp_path / "v.wav"
    a.write_text(json.dumps(dict(words=[dict(w="la", start=1.0, end=1.5)])))
    v.write_bytes(b"")
    x = np.arange(VR.SR * 30, dtype=float)  # a 30 s song; the sample value is its index
    monkeypatch.setattr(VR, "read_mono", lambda p: x)
    monkeypatch.setattr(VR, "retime", lambda align, audio, x=None, cap=None: dict(align, retimed="vocal"))
    monkeypatch.setattr(VT, "envelope_open", lambda *a_, **k: np.ones(50))
    monkeypatch.setattr(VT, "build", lambda al, *a_, **k: [dict(open=0.5)] * 50)
    seen = []
    monkeypatch.setattr(ST, "read", lambda o, clip, fps, keep=None: seen.append(clip) or dict(truth=0.4))
    S.retimed_align(str(a), str(v), str(tmp_path), 10.0, 50, 25.0)  # 50 frames at 25 fps from 10 s: 10 .. 12 s
    assert len(seen) == 2
    for clip in seen:
        assert len(clip) == 2 * VR.SR and clip[0] == 10 * VR.SR and clip[-1] == 12 * VR.SR - 1
