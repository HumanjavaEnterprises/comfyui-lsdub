# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""A shot near the end of a short song snippet: the vocal envelope reads past the file's end as silence (it used to
turn the whole curve into NaN, and the medium-shot dub crashed with negative image sizes); an all-NaN dynamics curve
is ignored; the jaw planner tolerates frames with no target."""
import subprocess

import numpy as np


def test_envelope_past_the_end_is_silence(tmp_path):
    from lsdub_engine.core import sung_dyn as SD
    wav = str(tmp_path / 'v.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=f=220:d=1', wav], check=True)
    e = SD.envelope(wav, 0.5, 200, 100.0)                                  # 0.5 .. 2.5 s of a 1 s file
    assert np.isfinite(e).all() and e[:30].max() > 0.5 and e[-50:].max() < 0.05
    e2 = SD.envelope(wav, -0.5, 100, 100.0)                                # starts before the file: leading silence
    assert np.isfinite(e2).all() and e2[:40].max() < 0.05 and e2[60:].max() > 0.5


def test_all_nan_dyn_curve_is_ignored(tmp_path):
    from lsdub_engine.core import coart
    p = tmp_path / 'dyn.csv'
    p.write_text('time_s,geo_gap_cm,geo_width_cm\n' + ''.join(f'{t / 100:.2f},nan,nan\n' for t in range(300)))
    assert coart.load_dyn(str(p), 0.5, 25, 25.0) is None


def test_jaw_plan_with_missing_targets():
    from lsdub_engine.core import teeth_layer as TL
    th = np.radians(np.r_[np.full(5, np.nan), np.linspace(2, 10, 20), np.full(5, np.nan)])
    out = TL.plan_jaw(th)
    assert np.isfinite(out).all()
    assert np.isfinite(TL.plan_jaw(np.full(10, np.nan))).all()


def test_lyrics_spread_when_nothing_is_heard():
    from lsdub_engine.core import lyric_align as LA
    env = np.r_[np.full(100, -90.0), np.full(300, -10.0), np.full(100, -90.0)]
    W = LA.spread_lyrics(['fly', 'me', 'away'], env, 2.0)
    assert [w['w'] for w in W] == ['fly', 'me', 'away'] and W[0]['start'] == 3.0 and W[-1]['end'] == 6.0
    assert LA.spread_lyrics([], env, 0.0) == []
