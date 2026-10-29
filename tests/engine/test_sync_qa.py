# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub sync-qa: the lag sign of tracking() and the plan-vs-pixels verdict (synthetic, no models)."""
import numpy as np

from lsdub_engine.audits import sync_qa as Q


def test_tracking_lag_sign_mouth_trails_is_positive():
    rng = np.random.default_rng(0)
    target = np.convolve(rng.normal(size=80), np.ones(3) / 3, 'same')
    ours = np.r_[np.zeros(2), target[:-2]]                         # the mouth does the same thing 2 frames LATER
    c0, cb, lag = Q.tracking(ours, target, 25.0)
    assert lag == 80.0 and cb > 0.95 and c0 < cb
    c0, cb, lag = Q.tracking(np.r_[target[2:], np.zeros(2)], target, 25.0)   # 2 frames EARLY
    assert lag == -80.0


def test_plan_pixels_disagree():
    assert Q.disagree(0.91, -0.41)                                  # 11C: the plan synced, the delivered pixels did not
    assert Q.disagree(0.9, float('nan')) and Q.disagree(0.9, None)  # pixels unmeasurable -> never trust the plan
    assert not Q.disagree(0.88, 0.80)                               # 58: both agree


def test_face_roi_contains_the_mouth_box_and_is_clipped():
    r = Q.face_roi([355, 460, 635, 783], (768, 1344))
    assert r[0] < 355 and r[1] >= 460 - 1 and r[2] < 635 and r[3] > 783 and r[0] >= 0 and r[3] <= 1344
    assert Q.face_roi([0, 768, 0, 1344], (768, 1344)) == [0, 768, 0, 1344]


def test_film_mode_singer_roi_and_cast_frames():
    """delivered-film mode helpers: the per-frame largest singer box from a cast scan, and a square ROI around it."""
    cast = {'results': [{'tracks': [
        {'character': 'red_nova', 'frames': [{'f': 10, 'box': [100, 100, 200, 220], 'face_px': 120}, {'f': 30, 'box': [0, 0, 1, 1], 'face_px': 9}]},
        {'character': 'alcyone', 'frames': [{'f': 10, 'box': [400, 100, 700, 400], 'face_px': 300}]},
        {'character': 'sky_lux', 'frames': [{'f': 11, 'box': [110, 100, 210, 220], 'face_px': 130}]}]}]}
    B = Q.cast_singer_frames(cast, 10, 20, preset='landing')
    assert set(B) == {10, 11} and B[10] == [100, 100, 200, 220]      # the non-singer and the out-of-range frame are ignored
    F = [np.zeros((768, 1344, 3), np.uint8)]
    r = Q.singer_roi(F, B, k=2.4)
    assert r[0] < 100 and r[1] > 220 and r[2] < 100 and r[3] > 210 and r[0] >= 0
