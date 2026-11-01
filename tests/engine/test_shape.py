# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the shape fix: the gap anchored at the top, closures that shut, Verify's shape readings, the card on the moved plan."""

import json

import numpy as np
import pytest

from lsdub_engine.audits import snap as SN, viseme_overlay as VO
from lsdub_engine.core import coart, mouth_warp as MW, recipes


def test_gap_top_is_proportional_and_leaves_width_to_its_map():
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(0, 10, 200), rng.uniform(5, 6, 200)], 1)
    ref = np.stack([rng.uniform(5, 20, 200), rng.uniform(1, 9, 200)], 1)  # an open-mouthed take
    q, _ = coart.range_match(X, ref, 1.0, None, None, "neutral")
    t, done = coart.range_match(X, ref, 1.0, None, None, "neutral", True)
    assert all(done)
    k = np.percentile(ref[:, 0], 95) / np.percentile(X[:, 0], 95)
    assert np.allclose(t[:, 0], X[:, 0] * k)  # 0 stays 0, a narrow class stays narrow
    assert np.array_equal(t[:, 1], q[:, 1])  # width: the map it had
    assert not np.allclose(q[:, 0], t[:, 0])


def test_gap_top_with_nothing_to_read_maps_nothing():
    X = np.zeros((10, 2))
    assert coart.gap_top_fn(X[:, 0], np.ones(10)) is None


def test_the_shape_stack_is_the_interior_stack_plus_its_two_flags():
    a, b = recipes.get("layers_v16_1_interior").argv, recipes.get("layers_v16_1_shape").argv
    assert b == (*a, "--coart-gap-anchor", "top", "--warp-erase-close", "--warp-interior-open-floor", "0.5")
    assert recipes.MOUTH_STACKS["layers_shape"] == "layers_v16_1_shape"
    assert "--warp-erase-close" not in recipes.get("layers_v16_1").argv


def test_seam_lands_both_inner_lips_on_the_contact_line():
    P = np.random.default_rng(1).uniform(0, 100, (478, 2))
    s, d = MW.lip_seam(P), MW.seam_dst(P)
    assert s.shape == d.shape
    n, k = len(MW.ll.LIP_OUT), len(MW.IN_PAIRS)
    assert np.array_equal(d[n + 1 : n + 1 + k], d[n + 2 + k :])  # the upper and lower points coincide on the target
    assert np.allclose(d[n + 1 + k // 2], (P[14] + P[13]) / 2)


def test_closure_real_is_absolute_where_closure_hit_is_relative():
    g = np.r_[np.full(10, 0.5), np.full(3, 0.09), np.full(10, 0.5)]  # an open-mouthed shot with a slit
    assert SN.closure_hit(g, [11])[0] == 1.0  # a slit passes the relative reading
    assert SN.closure_real(g, [11]) == (0.0, 1)
    g[11] = 0.01
    assert SN.closure_real(g, [11]) == (1.0, 1)
    assert SN.closure_real(np.full(5, np.nan), [2]) == (None, 0)


def test_class_contrast_reads_one_where_heights_follow_the_card():
    opens = {"aa": 0.4, "PP": 0.0, "ih": 0.2}
    c = ["aa"] * 5 + ["ih"] * 5 + ["PP"] * 5 + [""] * 3
    good = [0.4] * 5 + [0.2] * 5 + [0.0] * 5 + [9] * 3
    assert SN.class_contrast(good, c, opens)[0] == pytest.approx(1.0)
    assert SN.class_contrast([0.4] * 18, c, opens)[0] == pytest.approx(0.0)
    assert SN.class_contrast([0.1] * 4, ["ih"] * 4, opens) == (None, None, {})


def test_logged_classes_maps_the_paints_frames_onto_the_delivered(tmp_path):
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"frames": [{}] * 20}))
    (tmp_path / "w_coart.json").write_text(json.dumps({"segments": [{"v": "aa", "frames": [4, 7]}, {"v": "PP", "frames": []}]}))
    c = coart.logged_classes(str(w), 10)
    assert c[2:4] == ["aa", "aa"] and c[0] == "" and c[4] == ""
    assert coart.logged_classes(str(tmp_path / "x.json"), 10) is None


def test_card_reads_the_frame_the_paint_moved(tmp_path):
    w = tmp_path / "w.json"
    w.write_text("{}")
    assert VO.moved_frames(4, 25.0) == [0, 1, 2, 3]
    assert VO.moved_frames(4, 25.0, str(w)) == [0, 1, 2, 3]
    (tmp_path / "w_coart.json").write_text(json.dumps({"advance_s": 0.08}))
    assert VO.moved_frames(4, 25.0, str(w)) == [2, 3, 4, 5]
