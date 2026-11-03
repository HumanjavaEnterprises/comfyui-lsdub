# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the shape fix: the gap anchored at the top, closures that shut, Verify's shape readings, the card on the moved plan."""

import argparse
import json
import types

import cv2
import numpy as np
import pytest
from PIL import ImageDraw
from scipy.spatial import Delaunay

from lsdub_engine.audits import qa, snap as SN, viseme_overlay as VO
from lsdub_engine.core import coart, mouth_warp as MW, paint_api, recipes


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


def test_the_shape_stack_is_the_interior_stack_plus_its_flags():
    a, b = recipes.get("layers_v16_1_interior").argv, recipes.get("layers_v16_1_shape").argv
    assert b == (
        *a,
        "--coart-gap-anchor",
        "top",
        "--warp-erase-close",
        "--warp-erase-tone",
        "--warp-erase-near",
        "--warp-interior-open-floor",
        "0.5",
        "--warp-interior-mend",
        "--coart-closure-ramp",
        "1.0",
        "--coart-phones",
        "--coart-function-gate",
        "--warp-round",
        "0.6",
        "--warp-contact",
        "--warp-contact-reach",
        "--warp-nose-lock",
        "--warp-temporal",
        "1.5",
        "--warp-temporal-spare",
        "--warp-temporal-width",
        "2",
        "--coart-advance-ms",
        "40",
    )
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


# -- the plan: the gap anchored at every map it goes through ---------------------------------------------------------------------
def open_plan(anchor):
    """an open-mouthed take (its p5 gap is .4), a plan that goes to 0 between vowels: the shape of the height the anchor keeps."""
    from lsdub_engine.core import viseme_track as vt

    n, fps = 120, 25.0
    t = 0.2
    ph = []
    for i in range(20):
        ph.append(dict(p="AA" if i % 2 else "IH", start=round(t, 4), end=round(t + 0.2, 4)))
        t += 0.2
    al = dict(words=[dict(w="w", start=ph[0]["start"], end=ph[-1]["end"], phones=ph)])
    fr = vt.build(al, 0.0, n, fps, lead=0.06)
    u = np.stack([0.5 + 0.1 * np.sin(np.arange(n) / 3.0), 0.2 * np.cos(np.arange(n) / 5.0)], 1)
    u[::24, 0] = 0.02  # the take does shut, for a frame at a time: its p5 stays open, its manifold reaches the lips
    a = argparse.Namespace(
        warp_centre=0.8, warp_open_gain=1.0, warp_width_gain=1.0, warp_close_depth=0.0, coart_dom_width=60.0, coart_undershoot=0.6,
        coart_prior=0.0, coart_anticipate=100.0, coart_jaw_share="auto", coart_range=1.0, coart_gap_anchor=anchor, coart_dyn_mix=1.0,
    )  # fmt: skip
    d = np.stack([0.5 * (1 + np.sin(np.arange(n) / 2.0)), 0.2 * np.sin(np.arange(n) / 3.0)], 1)
    x, _, _ = coart.plan(u, fr, fps, 0.0, None, None, a, al, 0.0, None, dyn=d)
    return x[10:-10, 0], u[:, 0]


def test_the_anchor_holds_at_every_map_of_the_plan():
    """range_match runs on the external curve, on the blend and again on the fit: any one of the three left on the percentile map lifts the
    low end toward the take's open floor, and the plan's height contrast goes with it."""
    x, u = open_plan("top")
    assert np.percentile(x, 5) < 0.1 and np.median(x) < 0.4  # heights under the take's own p5 (.4) survive to the plan
    assert np.percentile(x, 95) == pytest.approx(np.percentile(u, 95), abs=0.01)  # the top lands on the take's top
    q, _ = open_plan("take")
    assert np.percentile(q, 5) > 0.15 and np.median(q) > 0.45  # `take` keeps the map it had


# -- the erase layer: closures that shut -------------------------------------------------------------------------------------------
def face(open_px):
    """478 landmarks of a mouth at (200, 200): the outer loop an ellipse, the inner lips open_px apart."""
    P = np.zeros((478, 2))
    P[:] = (200.0, 120.0)
    P[2] = (200.0, 150.0)
    for k, j in enumerate(MW.ll.LIP_OUT):
        th = 2 * np.pi * k / len(MW.ll.LIP_OUT)
        P[j] = (200 + 70 * np.cos(th), 200 + 34 * np.sin(th))
    pairs = MW.IN_PAIRS
    for k, (a, b) in enumerate(pairs):
        x = 200 + 42 * (2 * (k + 1) / (len(pairs) + 1) - 1)
        P[a] = (x, 200 - open_px / 2)
        P[b] = (x, 200 + open_px / 2)
    P[78], P[308] = (150.0, 200.0), (250.0, 200.0)
    P[61], P[291] = (130.0, 200.0), (270.0, 200.0)
    return P


class _Rigid:
    rigid = staticmethod(lambda i, c: c)


def eraser(close=True):
    """an EraseLayer on a take frame whose mouth is open: skin, red lips, a black gap."""
    P = face(24)
    img = np.full((400, 400, 3), (120, 150, 200), np.uint8)
    cv2.fillPoly(img, [P[MW.ll.LIP_OUT].astype(np.int32)], (60, 60, 170))
    cv2.fillPoly(img, [MW.ll.LIP_IN and P[[a for a, _ in MW.IN_PAIRS] + [b for _, b in MW.IN_PAIRS][::-1]].astype(np.int32)], (5, 5, 5))
    E = MW.EraseLayer.__new__(MW.EraseLayer)
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    E.ring = np.stack([200 + 110 * np.cos(th), 200 + 70 * np.sin(th)], 1)
    E.ring_c = E.ring
    E.P, E.pl, E.grow, E.close, E.lab, E.items = P, _Rigid, 0.22, close, [], {}
    E.bald = np.full_like(img, (120, 150, 200))
    E.lips_img = img
    E.lips_shut = MW.shut_lips(img, P) if close else None
    E.tris = Delaunay(np.vstack([MW.lip_seam(P), E.ring])).simplices
    return E, img


@pytest.mark.parametrize("target_gap", [0, 5])
def test_a_shut_closure_puts_none_of_the_takes_gap_on_the_face(target_gap):
    E, img = eraser()
    Pd = face(target_gap)
    E.composite(img.copy(), 0, E.P, Pd, 1.0, "closed")
    (x0, y0, w, h), layer, _ = E.items[0]
    mid = layer[200 - y0 - 3 : 200 - y0 + 4, 180 - x0 : 221 - x0]
    assert mid.mean() > 40  # no black between the lips; both sides meet on the contact line
    assert np.isfinite(layer).all()


def test_own_lips_are_left_alone_and_a_ref_too_shut_to_seam_does_not_shut():
    E, img = eraser()
    out = img.copy()
    E.composite(out, 0, E.P, E.P, 1.0, "own")
    (x0, y0, w, h), layer, _ = E.items[0]
    assert layer[200 - y0, 200 - x0].mean() < 40  # own: the warp's own open mouth stays
    F, img = eraser(close=False)
    F.tris = Delaunay(np.vstack([MW.lip_frame(F.P), F.ring])).simplices
    F.composite(img.copy(), 0, F.P, face(24), 1.0, "closed")  # no seam: the mesh it had
    assert F.items[0][1].shape[2] == 3


def test_a_near_shut_target_still_lands_both_lips_on_one_line(monkeypatch):
    E, img = eraser()
    got = []
    real = MW.mm.pwa
    monkeypatch.setattr(MW.mm, "pwa", lambda s, sp, dp, t, b: (got.append(np.array(dp)), real(s, sp, dp, t, b))[1])
    E.composite(img.copy(), 0, E.P, face(5), 1.0, "closed")
    n, k = len(MW.ll.LIP_OUT), len(MW.IN_PAIRS)
    assert np.allclose(got[0][n + 1 : n + 1 + k], got[0][n + 2 + k : n + 2 + 2 * k])


def test_no_collapsed_triangle_reaches_the_warp(monkeypatch):
    E, img = eraser()
    seen = []
    real = MW.mm.pwa

    def pwa(src, sp, dp, tris, box):
        seen.append(float(np.abs(MW.mm.tri_area(dp, tris)).min()))
        return real(src, sp, dp, tris, box)

    monkeypatch.setattr(MW.mm, "pwa", pwa)
    E.composite(img.copy(), 0, E.P, face(0), 1.0, "closed")
    assert seen and min(seen) >= 0.5


class _Model:
    rng = dict(g_lo=0.0, g_hi=10.0, w_lo=0.0, w_hi=10.0)
    delta = staticmethod(lambda d: np.tile([0.0, float(d[0]), 0.0], (len(MW.CTRL), 1)))


def fold_limited(close, monkeypatch, stats=None):
    """a clip with one closure the fold guard stopped short: the plan reaches 0, the guard kept half the move, the take is at 10."""
    n = 3
    got = {}

    class Layer:
        def __init__(self, *a, **k):
            got.update(k)

    monkeypatch.setattr(MW, "EraseLayer", Layer)
    monkeypatch.setattr(MW.ll, "lip_stats", lambda *a: stats)
    monkeypatch.setattr(MW.ll, "lip_fit", lambda *a: 1.0)
    monkeypatch.setattr(MW, "most_closed", lambda *a: 0)
    F = np.tile([10.0, 5.0], (n, 1))
    c = types.SimpleNamespace(
        g=lambda k, d=None: {"warp_erase": "off", "warp_erase_close": close}.get(k, d), n=n, frames=[None] * n, S=np.zeros((n, 478, 3)),
        F=F, Fe=F + [[-5.0, 0.0]] * n, dFe=np.tile([-5.0, 0.0], (n, 1)), dFe_pre_fold=np.tile([-10.0, 0.0], (n, 1)), rot=np.tile(np.eye(3), (n, 1, 1)), model=_Model, pl=None, pose=None, S3raw=np.zeros((n, 478, 3)),
    )  # fmt: skip
    MW.WarpClip.setup_erase(c)
    return c, got


def test_a_fold_limited_closure_is_read_on_its_plan(monkeypatch):
    c, got = fold_limited(True, monkeypatch)
    assert c.er_gap[1] == 0.0 and c.er_al[1] > 0.5  # the plan's closure, not the guard's half
    assert got["close"] is True
    o, got = fold_limited(False, monkeypatch)
    assert o.er_gap[1] == 5.0 and got["close"] is False  # off: Fe, as it was


def erase_kind(c, ov):
    kinds = []
    c.erase = types.SimpleNamespace(
        lab=[], composite=lambda out, i, Pt, Pd, al, kind: (kinds.append(kind), c.__dict__.update(last_dst=Pd))[0] or (out, None)
    )
    c.shut_target = lambda *a: MW.WarpClip.shut_target(c, *a)
    c.er_fit = np.ones(c.n)
    c.P_out = np.zeros((c.n, 478, 2))
    MW.WarpClip.erase_one(c, 1, np.zeros((4, 4, 3)), ov)
    return kinds[0]


def test_the_closure_shuts_in_the_round_and_in_its_repair(monkeypatch):
    c, _ = fold_limited(True, monkeypatch)
    assert erase_kind(c, {}) == "closed"
    assert erase_kind(c, {"dFe": np.array([-5.0, 0.0])}) == "closed"  # a gate repair: its dFe through the guard again
    o, _ = fold_limited(False, monkeypatch)
    assert erase_kind(o, {"erase_al": 1.0}) == "own"  # off: as it was
    assert erase_kind(o, {"erase_al": 1.0, "dFe": np.array([-5.0, 0.0])}) == "own"


def test_a_closure_the_fold_guard_stopped_short_lands_where_the_plan_said(monkeypatch):
    c, _ = fold_limited(True, monkeypatch)
    erase_kind(c, {})
    assert c.last_dst[MW.CTRL, 1] == pytest.approx(-5.0)  # the 5 the guard held back
    erase_kind(c, {"dFe": np.array([-10.0, 0.0])})
    assert np.all(c.last_dst == 0)  # nothing held back
    o, _ = fold_limited(False, monkeypatch)
    erase_kind(o, {"erase_al": 1.0})
    assert np.all(o.last_dst == 0)


def test_the_shut_lips_carry_no_dark_of_the_reference_at_their_corners():
    E, img = eraser()
    img = img.copy()
    cv2.circle(img, (147, 188), 2, (5, 5, 5), -1)  # a dark smile corner on the lips, past the seam's own corner
    cv2.circle(img, (253, 188), 2, (5, 5, 5), -1)
    clean = MW.clean_lips(img, E.P)
    L = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[..., 0]
    assert L(img)[188, 147] < 30 and L(clean)[188, 147] > 60 and L(clean)[188, 253] > 60
    assert np.array_equal(clean[20:60], img[20:60])  # the skin is left alone


def test_the_shut_layer_carries_a_contact_line():
    E, img = eraser()
    Pd = face(0)
    E.composite(img.copy(), 0, E.P, Pd, 1.0, "closed")
    (x0, y0, w, h), layer, _ = E.items[0]
    col = layer[170 - y0 : 231 - y0, 200 - x0, 1]
    assert abs(int(np.argmin(col)) + 170 - 200) <= 2  # the seam is the column's darkest pixel


def test_shut_lips_take_the_shots_colour_and_softness():
    E, img = eraser()
    have = MW.ll.lip_stats(MW.clean_lips(img, E.P), E.P)
    want = have[:3] + [20.0, -8.0, 4.0]
    got = MW.ll.lip_stats(MW.shut_lips(img, E.P, want), E.P)
    assert np.abs(got[:3] - want).max() < 3
    assert cv2.Laplacian(MW.shut_lips(img, E.P), cv2.CV_32F).var() < cv2.Laplacian(MW.clean_lips(img, E.P), cv2.CV_32F).var()


def test_the_shots_lips_reach_the_layer(monkeypatch):
    _, got = fold_limited(True, monkeypatch)
    assert got["lip_stat"] is None  # no stats, nothing to match
    st = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _, got = fold_limited(True, monkeypatch, st)
    assert list(got["lip_stat"]) == list(st[:3])


def test_the_shut_contact_is_one_soft_line():
    x = np.full((60, 120, 3), 150, np.uint8)
    y = 30 + np.array([0, 1, -1, 1, -1, 1, 0])
    line = np.stack([np.linspace(10, 110, 7), y], 1)
    s = MW.seam_shade(x, line, 100.0)
    col = s[:, 60, 1]
    assert col.min() < 110 and abs(int(np.argmin(col)) - 30) <= 2  # a shadow on the line
    assert col[5] == pytest.approx(150, abs=2)  # and only there


# -- interior donors -------------------------------------------------------------------------------------------------------------
def donors(open_floor):
    from lsdub_engine.core import interior_take as IT

    rng = np.random.default_rng(3)
    n = 60
    T = IT.TakeInterior.__new__(IT.TakeInterior)
    T.frames, T.open_floor = [None] * n, open_floor
    gap = rng.uniform(2, 40, n)
    T.ap = gap
    bank = types.SimpleNamespace(F=np.stack([gap, np.full(n, 100.0)], 1), pose=np.zeros((n, 3)))
    plan = np.stack([np.r_[np.full(30, 1.0), rng.uniform(1, 20, 30)], np.full(n, 100.0)], 1)  # shut lips first: a tight fit wants a slit
    return gap, T.pick(bank, plan, np.array([10.0, 10.0]), 0.0)


def test_open_floor_keeps_every_donor_open():
    gap, js = donors(0.5)
    assert gap[js].min() >= np.median(gap) - 1e-6  # fit and keep together: no frame ever holds a donor under it
    g0, j0 = donors(0.0)
    assert g0[j0].min() < np.median(g0) - 5  # no floor: the closest shape wins


def test_the_flag_reaches_the_donor_picker(monkeypatch):
    from lsdub_engine.core import interior_take as IT

    got = {}
    monkeypatch.setattr(IT, "TakeInterior", lambda *a, **k: got.update(k))
    c = types.SimpleNamespace(bank=None, scale=None, g=lambda k, d=None: d, sync_frames=None)
    MW.WarpClip.take_interior(c, [], [], [], lambda k, d=None: {"warp_interior_open_floor": 0.5}.get(k, d))
    assert got["open_floor"] == 0.5


# -- Verify --------------------------------------------------------------------------------------------------------------------
def test_closure_real_reads_one_frame_each_side():
    g = np.full(20, 0.5)
    g[9] = 0.01
    assert SN.closure_real(g, [10]) == (1.0, 1)
    assert SN.closure_real(g, [12]) == (0.0, 1)  # three frames away is not this closure
    assert SN.closure_real(g, [6]) == (0.0, 1)


def test_closure_real_threshold_splits_shut_from_a_dark_open_mouth():
    assert SN.closure_real(np.full(3, 0.031), [1]) == (1.0, 1)
    assert SN.closure_real(np.full(3, 0.06), [1]) == (0.0, 1)


def lit_mouth(dark):
    from lsdub_engine.audits import sync_qa as Q

    c = np.full((200, 200, 3), (90, 90, 170), np.uint8)
    if dark:
        cv2.line(c, (55, 100), (75, 100), (2, 2, 2), 3)
    pts = {
        0: (100, 70),
        13: (100, 100),
        14: (100, 100),
        17: (100, 130),
        40: (70, 75),
        81: (75, 100),
        270: (130, 75),
        311: (125, 100),
        61: (50, 100),
        291: (150, 100),
    }
    return Q.corner_dark(c, lambda j: np.array(pts[j], float))


def test_the_corner_reading_sees_what_the_middle_does_not():
    assert lit_mouth(False) < 0.05 and lit_mouth(True) > 0.8


def test_closure_corner_reads_the_most_shut_frame_and_the_worst_closure():
    g = np.array([0.5, 0.01, 0.5, 0.5, 0.5, 0.5, 0.02, 0.5])
    c = np.array([0.9, 0.1, 0.9, 0.0, 0.0, 0.0, 0.4, 0.0])
    assert SN.closure_corner(g, c, [1, 6]) == 0.4
    assert SN.closure_corner(g, c, []) is None


def test_shape_readings_are_absolute():
    gap = np.r_[np.full(10, 0.5), np.full(3, 0.09), np.full(10, 0.5)]
    w = [{"phones": [{"p": "M", "start": 0.42, "end": 0.48}]}]
    r = qa.shape_readings((gap, np.linspace(1, 2, len(gap))), None, w, 0.0, 25.0, SN)
    assert (r["closure_real"], r["closure_real_n"]) == (0.0, 1)  # a slit; the shot's own range would have passed it
    assert r["width_range"] == pytest.approx(0.9, abs=1e-3)
    assert qa.shape_readings(None, None, w, 0.0, 25.0, SN) == {}


def test_verify_reads_closures_on_the_moved_words(monkeypatch, tmp_path):
    from lsdub_engine.audits import sync_qa as Q

    w = tmp_path / "w.json"
    w.write_text(json.dumps({"frames": [{}] * 30}))
    (tmp_path / "w_coart.json").write_text(json.dumps({"advance_s": 0.2}))
    gap = np.full(30, 0.5)
    gap[9:12] = 0.0  # the mouth shuts on f10; the word's M reads f15 unmoved, f10 moved 200 ms earlier
    F = [np.zeros((64, 64, 3), np.uint8)] * 30
    op = np.arange(30.0)
    monkeypatch.setattr(Q, "read_span", lambda v, a, b: F)
    monkeypatch.setattr(Q, "singer_roi", lambda *a: [0, 64, 0, 64])
    monkeypatch.setattr(Q, "measured_opening", lambda B, roi, fps: (op, 1.0))
    monkeypatch.setattr(Q, "shape_of", lambda f: (gap, np.ones(30)))
    monkeypatch.setattr(SN, "envelope", lambda v, n, fps: op)
    monkeypatch.setattr(SN, "snap_metrics", lambda *a, **k: dict(open_in_silence=0.1))
    monkeypatch.setattr(qa.ff, "run", lambda *a, **k: None)
    words = [dict(w="m", start=0.6, end=0.64, phones=[dict(p="M", start=0.6, end=0.64)])]
    shot = dict(shot=0, f0=0, f1=29, start=0.0, warp_json=str(w))
    r = qa.measure_shot("v", shot, 25.0, 0.0, "v", np.ones(30, bool), None, words, lambda f: [], str(tmp_path), Q, SN)
    assert r["closure_real"] == 1.0


def test_classes_read_true_time_on_a_file_a_few_frames_short(tmp_path):
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"frames": [{}] * 216}))
    (tmp_path / "w_coart.json").write_text(json.dumps({"segments": [{"v": "aa", "frames": [100, 119]}], "plan_fps": 50.0}))
    assert coart.logged_classes(str(w), 108, 25.0).index("aa") == 50
    assert coart.logged_classes(str(w), 104, 25.0).index("aa") == 50  # four frames short: f50 is still f50
    (tmp_path / "w_coart.json").write_text(json.dumps({"segments": [{"v": "aa", "frames": [100, 119]}]}))
    assert coart.logged_classes(str(w), 104).index("aa") == 50  # a log from before plan_fps: the whole ratio


def test_offsets_read_true_time_on_a_file_a_few_frames_short(tmp_path):
    w = tmp_path / "w.json"
    w.write_text("{}")
    (tmp_path / "w_coart.json").write_text(json.dumps({"sync_dt": [0.0] * 100 + [0.08] * 100, "plan_fps": 50.0}))
    d = coart.logged_offsets(str(w), 96, 25.0)
    assert d[49] == 0.0 and d[50] == 0.08
    assert len(coart.logged_offsets(str(w), 100, 25.0)) == 100


def test_verify_reports_the_corners():
    gap = np.r_[np.full(10, 0.5), np.full(3, 0.01), np.full(10, 0.5)]
    w = [{"phones": [{"p": "M", "start": 0.42, "end": 0.48}]}]
    r = qa.shape_readings((gap, np.ones(len(gap)), np.full(len(gap), 0.6)), None, w, 0.0, 25.0, SN)
    assert r["closure_corner"] == 0.6
    assert "closure_corner" not in qa.shape_readings((gap, np.ones(len(gap))), None, w, 0.0, 25.0, SN)


def test_measured_opening_reads_the_corners(monkeypatch):
    from lsdub_engine.audits import sync_qa as Q

    lm = [types.SimpleNamespace(x=0.3 + 0.001 * k, y=0.2 + 0.001 * k) for k in range(478)]

    class Det:
        detect_for_video = staticmethod(lambda *a: types.SimpleNamespace(face_landmarks=[lm]))
        close = staticmethod(lambda: None)

    monkeypatch.setattr(
        Q.landmarker, "mp", lambda: types.SimpleNamespace(Image=lambda **k: None, ImageFormat=types.SimpleNamespace(SRGB=0))
    )
    monkeypatch.setattr(Q.landmarker, "get", lambda *a, **k: Det)
    F = [np.full((32, 32, 3), 120, np.uint8)] * 2
    Q.measured_opening(F, [0, 32, 0, 32])
    assert np.isfinite(Q.shape_of(F)[2]).all()


def test_the_shape_readings_do_not_hold_the_frames(monkeypatch):
    import gc
    import weakref

    from lsdub_engine.audits import sync_qa as Q

    class Frames(list):
        pass

    class Det:
        detect_for_video = staticmethod(lambda *a: types.SimpleNamespace(face_landmarks=[]))
        close = staticmethod(lambda: None)

    monkeypatch.setattr(
        Q.landmarker, "mp", lambda: types.SimpleNamespace(Image=lambda **k: None, ImageFormat=types.SimpleNamespace(SRGB=0))
    )
    monkeypatch.setattr(Q.landmarker, "get", lambda *a, **k: Det)
    F = Frames(np.zeros((8, 8, 3), np.uint8) for _ in range(3))
    Q.measured_opening(F, [0, 8, 0, 8])
    r = Q.shape_of(F)
    assert r is not None and len(r) == 3
    assert Q.shape_of(F) is None  # read once; the readings go with the read
    Q.measured_opening(F, [0, 8, 0, 8])
    r = weakref.ref(F)
    del F
    gc.collect()
    assert r() is None  # never read: the frames still went


# -- the card ----------------------------------------------------------------------------------------------------------------
def track_file(p):
    json.dump(
        dict(fps=25, t0=0.0, frames=[dict(t=k / 25, vis="aa", vis2="sil", mix=0.0, open=0.7, word="w", phone="") for k in range(20)]),
        open(p, "w"),
    )
    return str(p)


def moved_log(tmp_path):
    w = tmp_path / "w.json"
    w.write_text("{}")
    (tmp_path / "w_coart.json").write_text(json.dumps({"advance_s": 0.08}))
    return str(w)


def card_args(**k):
    return argparse.Namespace(edl=None, vocalisations="auto", skip_words=None, lyrics=None, align=None, vocal_merge_gap=0.3, **k)


def test_the_card_reads_the_moved_plan_from_a_track(tmp_path):
    p = track_file(tmp_path / "t.json")
    a = card_args(track=p, warp_json=moved_log(tmp_path))
    assert VO.segments_of(a, 4)[0][2] == [2, 3, 4, 5]
    assert VO.segments_of(card_args(track=p, warp_json=None), 4)[0][2] == [0, 1, 2, 3]


def test_the_card_reads_the_moved_plan_from_an_edl(tmp_path):
    p = track_file(tmp_path / "t.json")
    e = tmp_path / "e.json"
    e.write_text(json.dumps({"segments": [{"track": p, "first": 3, "nf": 3}, {"track": None, "frames": [7, 8]}]}))
    a = card_args(track=None, warp_json=moved_log(tmp_path))
    a.edl = str(e)
    s = VO.segments_of(a, 5)
    assert s[0][2] == [5, 6, 7] and s[1][2] == [7, 8]
    a.warp_json = None
    assert VO.segments_of(a, 5)[0][2] == [3, 4, 5]


def test_qa_and_paint_hand_the_card_their_warp_json():
    from lsdub_engine.audits import sync_qa as Q
    from lsdub_engine.core import mouth_card as MC

    assert Q.card_argv("v", "t", "o", "w.json")[-2:] == ["--warp-json", "w.json"]
    assert "--warp-json" not in Q.card_argv("v", "t", "o")
    a = argparse.Namespace(out="o.mp4", track="t", warp_json="w.json", mode="warp")
    argv = MC.annotate_args(a)
    assert argv[argv.index("--warp-json") + 1] == "w.json"


def test_the_card_says_it_is_the_plan(monkeypatch, tmp_path):
    p = track_file(tmp_path / "t.json")
    drawn = []
    real = ImageDraw.Draw

    def draw(img, *a):
        d = real(img, *a)
        t = d.text
        d.text = lambda xy, s, **k: (drawn.append(s), t(xy, s, **k))[1]
        return d

    monkeypatch.setattr(ImageDraw, "Draw", draw)
    tv = VO.TrackView(p, "auto")
    VO.Card(1.0).render(tv, 3)
    assert "PLAN" in drawn


# -- the flags --------------------------------------------------------------------------------------------------------------
def test_temporal_reads_both_spellings():
    assert paint_api.temporal(["--warp-temporal=1.5"]) == 1.5
    assert paint_api.temporal(["--warp-temporal", "1.5"]) == 1.5
    assert paint_api.temporal(["--warp-temporal-width=1.5"]) is None


def test_sing_paints_at_the_callers_temporal(monkeypatch, tmp_path):
    from lsdub_engine.core import sing as S

    F = [np.zeros((64, 64, 3), np.uint8)] * 4
    got = {}
    R = dict(
        face_h_px=64.0,
        zoom=1.0,
        crop_px=64,
        outside_max_preencode=0,
        not_pasted=[],
        picked=[0],
        timings=dict(crop=1.0, paint=2.0, paste=3.0),
    )
    monkeypatch.setattr(S.MED, "read", lambda v, n=None: (F, 25.0))
    monkeypatch.setattr(S.MED, "run", lambda t, tr, o, pa, *a, **k: (got.update(pa=[str(x) for x in pa]), R)[1])
    monkeypatch.setattr(S.steps, "call", lambda *a, **k: None)
    monkeypatch.setattr(S, "dyn_target", lambda *a: np.zeros(4))
    gate = dict(corr=0.9, best=0.9, lag_ms=0, open_in_silence=0.1, rise_ms=10, closure_hit=1, **{"pass": True})
    monkeypatch.setattr(S, "gate", lambda *a, **k: gate)
    S.run(
        "t.mp4",
        "a.json",
        "v.wav",
        0.0,
        str(tmp_path / "o.mp4"),
        str(tmp_path / "w"),
        ["--warp-temporal", "1.5"],
        box=[0, 0, 64, 64],
        report=str(tmp_path / "r.json"),
    )
    p = got["pa"]
    assert p[len(p) - 1 - p[::-1].index("--warp-temporal") + 1] == "3.0"  # the last one is the one paint keeps


# -- temporal that spares what the plan holds --------------------------------------------------------------------------------
def test_spare_marks_the_plans_moves_and_closures_not_its_jitter():
    t = np.arange(120)
    gap = 40 + 30 * np.sin(t / 8.0) + 0.3 * np.sin(t * 2.5)
    pp = np.zeros(120, bool)
    pp[60] = True
    a = MW.spare_frames(gap, pp)
    tops = [int(x) for x in np.arange(120) if gap[x] == gap[max(0, x - 6) : x + 7].max() and 6 < x < 113]
    assert all(a[k] > 0.5 for k in tops) and a[60] > 0.5
    assert np.mean(a < 0.05) > 0.5  # most of the clip still smooths; a flat wobble is not a peak


def test_the_flag_turns_the_sparing_on():
    gap, pp = 10 * np.sin(np.arange(60) / 4.0), np.zeros(60, bool)
    assert MW.temporal_spare(lambda k, d=None: d, gap, pp) is None
    assert MW.temporal_spare(lambda k, d=None: {"warp_temporal_spare": True}.get(k, d), gap, pp).max() > 0.5


def test_a_spared_peak_keeps_its_height_and_width_smooths_whole():
    dF = np.zeros((60, 2))
    dF[28:32, 0] = 50.0  # a class that holds four frames
    dF[28:32, 1] = 50.0
    a = MW.spare_frames(dF[:, 0], np.zeros(60, bool))
    plain = MW.smooth_residual(dF, 3.0)
    kept = MW.smooth_residual(dF, 3.0, None, a)
    assert plain[:, 0].max() < 30 and kept[:, 0].max() > 45
    assert np.array_equal(kept[:, 1], plain[:, 1])
    assert np.array_equal(MW.smooth_residual(dF, 3.0, None, None), plain)  # off: as it was


def test_close_slivers_keeps_a_corner_in_the_mouth_and_a_stretch_out():
    from lsdub_engine.core.mouth_layers import close_slivers

    op = np.ones((60, 120), bool)
    enc = np.ones((60, 120), np.float32)
    enc[10:14, 5:12] = 0.0  # a small corner
    enc[30:55, 40:110] = 0.0  # a large stretch
    out = close_slivers(enc, op, 100.0)
    assert out[12, 8] == 1.0 and out[40, 80] == 0.0 and out.dtype == enc.dtype
    assert close_slivers(np.ones((6, 6), np.float32), np.ones((6, 6), bool), 100.0).min() == 1.0


def test_mend_fill_stays_in_colour_where_a_donor_keeps_nothing_near():
    from lsdub_engine.core import interior_take as IT

    class B:
        pass

    t = IT.TakeInterior.__new__(IT.TakeInterior)
    P = np.zeros((478, 2))
    a = np.linspace(0, 2 * np.pi, len(IT.LIP_IN), endpoint=False)
    P[IT.LIP_IN] = np.stack([150 + 100 * np.cos(a), 50 + 30 * np.sin(a)], 1)
    f = np.full((100, 300, 3), (200, 200, 230), np.uint8)  # teeth over most of it, a few dark pixels at one end
    f[45:55, 55:75] = (40, 30, 90)
    t.S, t.frames, t.ap = [P], [f], np.array([0.5])
    t.ring = np.zeros((1, 3))
    for mend, ok in ((False, False), (True, True)):
        t.mend = mend
        lab = t.donor_lab(0, P[IT.LIP_IN], 100, 300)[0]
        far = lab[50, 230]
        assert (float(far[1]) > 100) is ok  # Lab zero is the blue the fill used to show


def test_ease_one_is_the_sparing_as_it_was():
    r = np.random.default_rng(3)
    g = np.cumsum(r.normal(0, 1, 80))
    pp = np.zeros(80, bool)
    pp[10:13] = True
    p = 0.15 * max(1e-6, float(np.ptp(np.percentile(g, [5, 95]))))
    a = np.zeros(80)
    a[MW.find_peaks(g, prominence=p)[0]] = 1.0
    a[MW.find_peaks(-g, prominence=p)[0]] = 1.0
    a[pp] = 1.0
    assert np.array_equal(MW.spare_frames(g, pp), np.clip(2.5 * MW.gaussian_filter1d(a, 1.0, mode="nearest"), 0.0, 1.0))


def test_closed_frames_hold_every_closure_class_through_the_sparing():
    pp = np.zeros(6, bool)
    pp[1] = True
    assert MW.closed_frames(6, pp, [3, 9]).tolist() == [False, True, False, True, False, False]
    assert MW.closed_frames(6, pp).tolist() == pp.tolist()


def test_tone_lips_lands_the_two_lips_on_the_takes_tone_and_keeps_their_mean():
    img = np.zeros((120, 200, 3), np.uint8)
    img[:] = (90, 90, 200)
    P = np.zeros((478, 2))
    a = np.linspace(np.pi, -np.pi, len(MW.ll.LIP_OUT), endpoint=False)
    P[MW.ll.LIP_OUT] = np.stack([100 - 70 * np.cos(a), 60 + 32 * np.sin(a)], 1)
    P[61], P[291] = (30, 60), (170, 60)
    for k, (x, y) in {78: (40, 60), 308: (160, 60)}.items():
        P[k] = (x, y)
    for t, (u, v) in zip(np.linspace(55, 145, 9), MW.IN_PAIRS):
        P[u], P[v] = (t, 60), (t, 60)
    t0 = MW.lip_tone(img, P)
    assert t0 is not None and abs(t0[0]) < 1
    out = MW.tone_lips(img, P, np.array([-30.0, 0.0, 0.0]))
    assert MW.lip_tone(out, P)[0] == pytest.approx(-30.0, abs=3.0)


def test_most_closed_passes_over_a_smile():
    gap = np.array([30.0, 20.0, 18.0, 2.0, 25.0, 22.0])
    w = np.array([100.0, 101.0, 99.0, 170.0, 100.0, 102.0])
    S, pose = None, np.zeros((6, 3))
    assert MW.most_closed(S, gap, pose) == 3
    assert MW.most_closed(S, gap, pose, width=w) == 1
