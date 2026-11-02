# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the nose against the take."""

import numpy as np
import pytest

from lsdub_engine.audits import nose as N


def face(dx=0.0, dy=0.0, size=120):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    v = 100 + 60 * np.sin((xx - dx) / 5.0) * np.cos((yy - dy) / 7.0)
    return np.dstack([v] * 3).astype(np.uint8)


def track(n):
    p = np.zeros((n, 478, 2))
    p[:, 98], p[:, 327] = [30, 40], [90, 40]
    p[:, 1], p[:, 4] = [60, 35], [60, 38]
    p[:, 2] = [60, 80]
    return p


def test_a_nose_the_render_moved_reads_its_shift_and_the_take_reads_none():
    t = [face()] * 3
    r = [face(), face(0, 4), face()]
    dx, dy = N.shifts(t, r, track(3))
    assert dy[0] == pytest.approx(0, abs=0.4) and dy[1] == pytest.approx(4, abs=0.7) and dy[2] == pytest.approx(0, abs=0.4)


def test_a_steady_offset_is_no_jump_and_a_single_snap_fails():
    iod = np.full(10, 100.0)
    z = np.zeros(10)
    assert N.stillness(z, np.full(10, 2.0), iod)["nose_ok"] and N.stillness(z, np.full(10, 2.0), iod)["nose_jumps"] == []
    y = z.copy()
    y[5:] = 5.0
    r = N.stillness(z, y, iod)
    assert not r["nose_ok"] and r["nose_jumps"] == [5] and r["nose_step"] == pytest.approx(0.05)


def test_many_small_jumps_fail_too():
    y = np.tile([0.0, 2.0], 10)
    r = N.stillness(np.zeros(20), y, np.full(20, 100.0))
    assert len(r["nose_jumps"]) > N.NOSE_FAIL and not r["nose_ok"]


def test_a_wide_take_pulled_shut_is_named_and_a_narrow_one_is_not():
    assert N.overpull([0.28, 0.28, 0.1, 0.28], [0.03, 0.2, 0.0, 0.11]) == [0, 3]


def test_a_frame_with_no_track_is_passed_over():
    t = track(2)
    t[1] = np.nan
    dx, dy = N.shifts([face()] * 2, [face()] * 2, t)
    assert np.isfinite(dy[0]) and np.isnan(dy[1])


def test_the_reading_moves_no_band():
    import inspect

    from lsdub_engine.audits import qa

    assert "nose_ok" not in inspect.getsource(qa.verify)
