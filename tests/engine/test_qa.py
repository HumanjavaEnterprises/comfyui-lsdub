# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub qa: the shot detector (cut / fade / flash), the FIT verdict order, VAD, and the verify band (synthetic)."""
import subprocess
import numpy as np

from lsdub_engine.audits import qa as QA, scorecard as SC


def test_detect_cuts_spike_not_fade_not_flash():
    n = 60
    dp = np.full(n, 1.0)
    dh = np.full(n, 0.01)
    dp[20], dh[20] = 40.0, 0.4                                  # a hard cut
    dp[30:42] = np.linspace(6, 14, 12)
    dh[30:42] = 0.08        # a fade: elevated but never a spike
    dp[50], dh[50] = 60.0, 0.5                                  # a flash ...
    assert QA.detect_cuts(dh, dp) == [20, 50]
    fl = np.full(n, np.inf)
    fl[50] = 5.0                       # ... whose next frames return to the frame before it
    assert QA.detect_cuts(dh, dp, flash=fl) == [20]
    S = QA.shots_from_cuts([20], n, 25.0)
    assert [(s['f0'], s['f1'], s['frames']) for s in S] == [(0, 19, 20), (20, 59, 40)] and S[1]['start'] == 0.8


def test_match_cuts():
    share, missed, fp = QA.match_cuts([10, 31, 70], [10, 30, 50])
    assert abs(share - 2 / 3) < 1e-9 and missed == [50] and fp == [70]


def test_verdict_order_and_reasons():
    base = {'frames': 40, 'faces': 1, 'px1080': 350.0, 'px': 250.0, 'present': 1.0, 'yaw': 3.0, 'mouth_visible': 1.0, 'mic': 0.0}
    v = lambda **k: QA.verdict(dict(base, **k), True)
    assert v()[0] == 'DUB'
    assert v(px1080=200.0) == ('DUB-MEDIUM', 'medium 200 px @1080: crop path')
    assert v(yaw=25.0)[0] == 'DUB-MEDIUM' and v(yaw=50.0)[1].startswith('profile (yaw +50')
    assert v(px1080=90.0)[1] == 'too small (90 px @1080 < 120)'
    assert v(mouth_visible=0.2, mic=0.3)[1].startswith('mouth hidden (mic at the lips')
    assert v(frames=5)[1].startswith('too short (5 frames')
    assert v(present=0.3)[1].startswith('target on screen only 30%')
    s = {'frames': 40, 'faces': 2, 'present': 0.0, 'largest': {'who': 'maia', 'px1080': 400.0}}
    assert QA.verdict(s, True) == ('NO', 'wrong person (largest face = maia 400.0 px)') and QA.verdict({'frames': 40, 'faces': 0}, False) == ('NO', 'no face')


def test_vad(tmp_path):
    p = str(tmp_path / 'v.wav')                                 # 1 s silence, 1 s tone, 1 s silence
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', '-f', 'lavfi', '-i', 'sine=f=300:r=16000:d=1',
                    '-filter_complex', '[0]atrim=0:1[a];[0]atrim=0:1[c];[a][1][c]concat=n=3:v=0:a=1', p], check=True)
    act, e = QA.vad(p, 75, 25.0)
    assert not act[:20].any() and act[30:45].all() and not act[56:].any()
    act2, _ = QA.vad(p, 75, 25.0, offset=1.0)                   # the audio starts 1 s into the video
    assert act2[55:70].all() and not act2[:45].any()


def test_score_row_bands():
    r = {'verdict': 'DUB', 'voiced': True, 'corr': 0.9, 'corr_pm1': 0.9, 'open_in_silence': 0.1, 'rise_ms': 40, 'closure_hit': 1.0,
         'mouth_range': 0.2, 'owner_px': 350.0, 'mouth_visible': 1.0, 'motion': 2.0, 'pops': 0, 'sharpness': 100.0}
    assert QA.score_row(dict(r), 100.0, SC)['band'] == 'PASS'
    assert QA.score_row(dict(r, corr=0.1, corr_pm1=0.1), 100.0, SC)['band'] == 'FAIL'
    x = QA.score_row(dict(r, identity_cos=0.5), 100.0, SC)
    assert x['band'] == 'REVIEW' and x['reason'].startswith('identity drift')
    assert QA.score_row(dict(r, voiced=False), 100.0, SC)['band'] == 'n/a'
    assert QA.score_row(dict(r, owner_px=60.0), 100.0, SC)['class'] == 'COVERAGE'
