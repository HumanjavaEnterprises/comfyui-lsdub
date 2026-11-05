# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the drive's shape stage (lsdub/core/drive_shape.py): synthetic, no models."""

import numpy as np
import pytest

from lsdub_engine.core import drive_shape as DS


def track(spec):
    """spec: [(vis, phone, word, open)] per frame."""
    return [{"vis": v, "vis2": v, "mix": 0.0, "phone": p, "word": w, "open": o} for v, p, w, o in spec]


def test_a_bilabial_with_no_shut_frame_gets_one_in_its_middle():
    fr = track([("aa", "AA1", "a", 0.9)] * 3 + [("aa", "M", "a", 0.8)] * 3 + [("aa", "AA1", "a", 0.9)] * 3)
    out, added = DS.closures(fr)
    assert added == [4] and out[4]["vis"] == "PP" and out[4]["open"] == 0.0 and out[4]["vis2"] == "aa"
    assert fr[4]["vis"] == "aa"  # the input is not changed


def test_a_bilabial_already_shut_or_shut_next_to_it_is_left_alone():
    fr = track([("aa", "AA1", "a", 0.9), ("PP", "", "", 0.0), ("aa", "B", "b", 0.8), ("aa", "B", "b", 0.8), ("aa", "AA1", "b", 0.9)])
    assert DS.closures(fr)[1] == []


def test_other_phones_never_close():
    fr = track([("aa", "AA1", "a", 0.9), ("DD", "T", "a", 0.2), ("SS", "S", "a", 0.1)])
    assert DS.closures(fr)[1] == []


def test_the_closing_edge_comes_early_and_the_opening_edge_keeps_its_time():
    o = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], float)
    got = DS.advance_closing(o, 1)
    assert got.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert DS.advance_closing(o, 0).tolist() == o.tolist()


def test_shaped_off_is_the_track_as_it_is_and_asym_advances_by_the_tracks_rate():
    fr = track([("aa", "AA1", "a", v) for v in (0.0, 0.5, 1.0, 1.0, 0.2, 0.0)])
    assert DS.shaped(fr, 50.0, "off")[0] == fr
    out, info = DS.shaped(fr, 50.0, "asym")
    assert info["advance_frames"] == 1 and [f["open"] for f in out] == [0.0, 0.5, 1.0, 0.2, 0.0, 0.0]
    with pytest.raises(ValueError):
        DS.shaped(fr, 50.0, "on")
