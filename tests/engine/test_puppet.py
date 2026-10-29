# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Pytest: puppet-mode jaw-hinge timing. Skips cleanly if cv2 isn't installed in the test env."""
import os, sys
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsdub_engine.anchors.mouth_paint_cutout import puppet_jaw, jaw_openness_scale


def _frame():
    f = np.full((200, 200, 3), 120, np.uint8)
    f[:100] = 80            # upper head
    f[100:] = 160           # lower jaw (distinct), hinge at y=100
    return f


def test_closed_is_unchanged():
    f = _frame()
    out = puppet_jaw(f, (100, 100), 26.0, 0.0)
    assert np.array_equal(out, f), "open=0 must leave the frame untouched"


def test_open_drops_the_jaw():
    f = _frame()
    out = puppet_jaw(f, (100, 100), 26.0, 0.8)
    changed = int((out != f).any(axis=2).sum())
    assert changed > 0, "opening must change pixels"
    # a dark interior must appear just below the hinge line
    strip = out[101:115, 70:130]
    assert strip.mean() < f[101:115, 70:130].mean(), "mouth interior should darken under the hinge"


def test_more_open_drops_further():
    f = _frame()
    small = puppet_jaw(f, (100, 100), 26.0, 0.3)
    big = puppet_jaw(f, (100, 100), 26.0, 0.9)
    n_small = int((small != f).any(axis=2).sum())
    n_big = int((big != f).any(axis=2).sum())
    assert n_big > n_small, "wider opening should move more pixels"


def test_cutout_openness_scale_monotonic():
    crop = np.tile(np.linspace(0, 255, 60, dtype=np.uint8)[:, None, None], (1, 60, 3))  # vertical gradient
    mask = np.zeros((60, 60), np.float32)
    mask[20:40, 20:40] = 1.0
    base = jaw_openness_scale(crop, mask, 0.0)
    more = jaw_openness_scale(crop, mask, 0.5)
    assert np.array_equal(base, crop)
    assert not np.array_equal(more, crop), "nonzero openness must change the crop"
