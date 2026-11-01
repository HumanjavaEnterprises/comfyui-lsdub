# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""behaviour pins for what the independent check's mutants walked through: each test here fails if the named wiring is taken out."""

import json, os, types

import cv2
import numpy as np
import pytest

from lsdub_engine.audits import qa, review as R, snap as SN
from lsdub_engine.core import coart, interior_take as IT, mouth_layers as ML, mouth_warp as MW

from test_coart import args, align_of, take, track_of, FPS
from test_shape import eraser, face, fold_limited


# -- --coart-closure-ramp ---------------------------------------------------------------------------------------------------------
def ramp_plan(**kw):
    n = 60
    al, _ = align_of([("AA", 0.2), ("M", 0.06), ("AA", 0.2), ("B", 0.06), ("AA", 0.4)], 0.2)
    u = take(n, gap=0.5, wobble=0.25)
    u[:, 0] = 0.5 + 0.45 * np.sin(np.arange(n) / 1.3)  # a take that swings hard and fast
    fr = track_of(n, al)
    return coart.plan(u, fr, FPS, 0.0, None, None, args(coart_vmax_scale=3.0, **kw), al, 0.0, None), fr, u


def test_the_plans_frames_beside_a_closure_are_held_under_the_ramp():
    (x0, pp0, _), _, _ = ramp_plan()
    (x1, pp1, _), _, _ = ramp_plan(coart_closure_ramp=1.0)
    k = np.flatnonzero(pp1)[0]
    assert (
        x1[k - 1, 0] - x1[k, 0] < x0[k - 1, 0] - x0[k, 0]
    )  # the step into the closure is smaller: without the ramped limits only the closure is bounded


def test_the_ramp_eases_the_width_too():
    calls = []
    real = coart.ramp_width
    coart.ramp_width = lambda *a: calls.append(a) or real(*a)
    try:
        ramp_plan(coart_closure_ramp=1.0)
        ramp_plan()
    finally:
        coart.ramp_width = real
    assert len(calls) == 1  # on with the ramp, off without


def test_a_slow_take_does_not_stretch_the_ramp_past_its_ceiling(monkeypatch):
    seen = {}
    real = coart.closure_ramp
    monkeypatch.setattr(coart, "closure_ramp", lambda c, n, top, a, b: seen.update(a=a, b=b) or real(c, n, top, a, b))
    n = 60
    al, _ = align_of([("AA", 0.2), ("M", 0.06), ("AA", 0.4)], 0.2)
    u = take(n, gap=0.5)
    u[:, 0] = 0.5 + 0.4 * np.sin(np.arange(n) / 14.0)  # closes over half a second
    coart.plan(u, track_of(n, al), FPS, 0.0, None, None, args(coart_closure_ramp=1.0), al, 0.0, None)
    assert seen["a"] <= coart.RAMP_MAX_S * FPS + 1e-9 and seen["b"] <= coart.RAMP_MAX_S * FPS + 1e-9


def test_a_vowel_between_two_close_closures_still_opens():
    lim = coart.closure_ramp({20: 0.05, 26: 0.05}, 60, 1.0, 7.5, 7.5)  # 120 ms apart at 50 fps
    assert max(lim.get(k, 9) for k in range(21, 26)) >= coart.RAMP_PEAK - 1e-6
    far = coart.closure_ramp({20: 0.05, 45: 0.05}, 60, 1.0, 7.5, 7.5)
    assert far[21] == pytest.approx(0.05 + 0.95 / 7.5)  # room to spare: the take's speed


# -- mend -----------------------------------------------------------------------------------------------------------------------------
def test_the_mend_flag_reaches_the_donor_picker(monkeypatch):
    got = {}
    monkeypatch.setattr(IT, "TakeInterior", lambda *a, **k: got.update(k))
    c = types.SimpleNamespace(bank=None, scale=None, g=lambda k, d=None: d, sync_frames=None)
    MW.WarpClip.take_interior(c, [], [], [], lambda k, d=None: {"warp_interior_mend": True}.get(k, d))
    assert got["mend"] is True


def test_the_mend_sinks_pink_toward_the_openings_dark_not_its_median():
    lab = np.zeros((20, 20, 3), np.float32)
    lab[:, :, 1:] = 128
    lab[:, :10, 0] = 20  # the dark half
    lab[:, 10:, 0] = 120
    nonp = np.ones((20, 20), bool)
    assert IT.sink_colour(lab, nonp, True)[0] < 25 < IT.sink_colour(lab, nonp, False)[0]


def test_the_enclosure_is_mended_only_under_the_flag(monkeypatch):
    got = []
    monkeypatch.setattr(ML, "close_slivers", lambda enc, o, io: got.append(io) or "closed")
    monkeypatch.setattr(ML.tl, "poly_mask", lambda *a: None)
    on, off = types.SimpleNamespace(mend=True), types.SimpleNamespace(mend=False)
    assert ML.mended_enclosure("enc", on, None, (4, 4), 9.0) == "closed" and got == [9.0]
    assert ML.mended_enclosure("enc", off, None, (4, 4), 9.0) == "enc" and ML.mended_enclosure("enc", None, None, (4, 4), 9.0) == "enc"


# -- lip tone on the layer ---------------------------------------------------------------------------------------------------------
def test_tone_reaches_the_shut_layer():
    E, img = eraser()
    Pd = face(0)
    E.tone = None
    E.composite(img.copy(), 0, E.P, Pd, 1.0, "closed")
    plain = E.items[0][1].copy()
    E.items.clear()
    E.tone = np.array([60.0, 0.0, 0.0])  # the upper lip far lighter than the lower
    E.composite(img.copy(), 0, E.P, Pd, 1.0, "closed")
    toned = E.items[0][1]
    assert np.abs(toned - plain).max() > 5


def test_the_reference_reads_width_only_under_the_tone_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(MW, "most_closed", lambda S, g, p, fit=None, width=None: seen.append(width) or 0)
    n = 3

    class _M:
        rng = dict(g_lo=0.0, g_hi=20.0, w_lo=0.0, w_hi=10.0)

    monkeypatch.setattr(MW, "EraseLayer", lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(MW.ll, "lip_stats", lambda *a: None)
    F = np.tile([10.0, 5.0], (n, 1))
    for tone in (False, True):
        c = types.SimpleNamespace(
            g=lambda k, d=None, tone=tone: {"warp_erase": "off", "warp_erase_close": True, "warp_erase_tone": tone}.get(k, d), n=n, frames=[None] * n, S=np.zeros((n, 478, 3)),
            F=F, Fe=F + [[-5.0, 0.0]] * n, dFe=np.tile([-5.0, 0.0], (n, 1)), dFe_pre_fold=np.tile([-10.0, 0.0], (n, 1)), rot=np.tile(np.eye(3), (n, 1, 1)), model=_M, pl=None, pose=None, S3raw=np.zeros((n, 478, 3)),
            closed_tone=lambda: None,
        )  # fmt: skip
        MW.WarpClip.setup_erase(c)
    assert seen[0] is None and seen[1] is not None and list(seen[1]) == [5.0] * 3


# -- reference near the closure ------------------------------------------------------------------------------------------------------
def pushin(n=60):
    S = np.zeros((n, 478, 3))
    iod = np.linspace(100, 160, n)
    S[:, 33, 0] = 100
    S[:, 263, 0] = 100 + iod
    return S


def test_near_closed_takes_a_frame_of_the_closures_size_and_time_over_the_shots_shuttest():
    S = pushin()
    gap = np.full(60, 10.0)
    gap[57] = 0.0  # the shuttest frame in the shot: the end of the push-in, a far bigger face
    gap[12] = 3.0  # shut enough, near the closure
    pose = np.zeros((60, 3))
    r = MW.near_closed(S, gap, pose, 15)
    assert r == 12 and MW.most_closed(S, gap, pose) == 57


def test_near_closed_with_nothing_of_its_size_is_most_closed():
    S = pushin()
    S[:, 263, 0] = 100 + np.r_[np.full(40, 100.0), np.full(20, 300.0)]
    gap = np.arange(60.0)
    pose = np.r_[np.zeros((40, 3)), np.full((20, 3), 9.0)]  # all the frames of this size look away
    assert MW.near_closed(S, gap, pose, 45) == MW.most_closed(S, gap, pose) == 0


def test_each_closure_gets_its_own_layer_under_the_flag(monkeypatch):
    made = []
    monkeypatch.setattr(MW, "EraseLayer", lambda f, S, pl, r, *a, **k: made.append(r) or types.SimpleNamespace(r=r))
    monkeypatch.setattr(MW, "near_closed", lambda S, g, p, at, *a: at)
    monkeypatch.setattr(MW, "most_closed", lambda *a: 0)
    monkeypatch.setattr(MW.ll, "lip_stats", lambda *a: None)
    n = 30

    class _M:
        rng = dict(g_lo=0.0, g_hi=20.0, w_lo=0.0, w_hi=10.0)

    F = np.tile([10.0, 5.0], (n, 1))
    Fe = F.copy()
    Fe[[5, 6, 20], 0] = 0.0
    c = types.SimpleNamespace(
        g=lambda k, d=None: {"warp_erase": "all", "warp_erase_close": True, "warp_erase_near": True}.get(k, d), n=n, frames=[None] * n, S=np.zeros((n, 478, 3)),
        F=F, Fe=Fe, dFe=Fe - F, dFe_pre_fold=Fe - F, rot=np.tile(np.eye(3), (n, 1, 1)), model=_M, pl=None, pose=None, S3raw=np.zeros((n, 478, 3)), closed_tone=lambda: None,
    )  # fmt: skip
    MW.WarpClip.setup_erase(c)
    assert isinstance(c.erase, MW.EraseSet) and len(set(made)) >= 2
    a = c.erase.which[5]
    assert c.erase.at(5).r == a and c.erase.which[20] != a


def test_erase_set_routes_frames_to_their_layers():
    done = []

    class L:
        def __init__(s, n):
            s.n, s.items = n, {}

        forget = lambda s, i: done.append((s.n, "forget", i))
        composite = lambda s, out, i, *a: done.append((s.n, "composite", i))
        apply_item = lambda s, i, img, st: done.append((s.n, "apply", i))
        stat_for = lambda s, i: s.n
        finish = lambda s, outs: done.append((s.n, "finish"))

    E = MW.EraseSet({0: L("a"), 9: L("b")}, {1: 0, 2: 9})
    E.forget(2), E.composite(None, 1), E.apply_item(2, None, None), E.finish([])
    assert done == [("b", "forget", 2), ("a", "composite", 1), ("b", "apply", 2), ("a", "finish"), ("b", "finish")] and E.stat_for(1) == "a"


# -- the sparing ------------------------------------------------------------------------------------------------------------------------
def test_the_ease_setting_reaches_the_sparing():
    gap, pp = 10 * np.sin(np.arange(60) / 4.0), np.zeros(60, bool)
    on = lambda e: lambda k, d=None: {"warp_temporal_spare": True, "warp_temporal_spare_ease": e}.get(k, d)
    a, b = MW.temporal_spare(on(1.0), gap, pp), MW.temporal_spare(on(2.0), gap, pp)
    assert (b > 0.5).sum() > (a > 0.5).sum()


def test_under_the_sparing_an_f_closure_holds_shut_like_a_p():
    n = 30
    pp = np.zeros(n, bool)
    a = types.SimpleNamespace(warp_temporal_spare=True, coart_closures=[14, 15, 90])
    shut = MW.spared_shut(a, n, pp)
    assert shut[14] and shut[15] and shut.sum() == 2
    dF = np.zeros((n, 2))
    dF[14:16, 0] = -9.0
    r = MW.smooth_residual(dF, 1.5, shut, MW.spare_frames(np.ones(n), shut))
    assert r[14, 0] == -9.0 and r[15, 0] == -9.0  # the closure's frames keep their full depth, its F as much as a P
    assert MW.spared_shut(types.SimpleNamespace(coart_closures=[14]), n, pp) is pp  # the flag off: only the PP frames, as it was


# -- Verify's reading -----------------------------------------------------------------------------------------------------------------
def test_verify_writes_the_vocal_reading_on_the_row(monkeypatch, tmp_path):
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"frames": []}))
    (tmp_path / "w_coart.json").write_text(json.dumps({"advance_s": 0.0}))
    words = [
        dict(w="a", start=1.0, end=1.5, phones=[dict(p="AH", start=1.0, end=1.5)]),
        dict(w="b", start=2.0, end=2.5, phones=[dict(p="AH", start=2.0, end=2.5)]),
    ]
    voc = [dict(w="a", start=1.0, end=1.5), dict(w="b", start=2.5, end=3.0)]
    r = qa.vocal_reading(words, voc, dict(warp_json=str(w), start=0.0, end=4.0), 0.0)
    assert r["vocal_off_n"] == 1 and r["vocal_worst"] == "b -500"
    sj = tmp_path / "s.json"
    sj.write_text(
        json.dumps(
            dict(
                fps=25.0,
                offset=0.0,
                shots=[dict(shot=0, f0=0, f1=9, start=0.0, end=4.0, verdict="DUB", warp_json=str(w), target=1, px1080=1)],
            )
        )
    )
    al = tmp_path / "a.json"
    al.write_text(json.dumps(dict(words=words)))
    from lsdub_engine.core import vocal_retime as V

    monkeypatch.setattr(qa, "vad", lambda *a: (np.ones(10, bool), np.ones(10)))
    from lsdub_engine.core import cast

    monkeypatch.setattr(cast, "Faces", lambda: None)
    monkeypatch.setattr(qa, "measure_shot", lambda *a, **k: dict(sharpness=None))
    monkeypatch.setattr(qa, "score_row", lambda r, *a: r.update(score=50.0, band="REVIEW"))
    monkeypatch.setattr(V, "retime", lambda A, v, x=None, free=False: dict(words=voc))
    monkeypatch.setattr(V, "read_mono", lambda p: np.zeros(1000))
    monkeypatch.setattr(V, "plan_hits", lambda *a: dict(vocal_hiss=0.5))
    v = tmp_path / "v.mp4"
    v.write_bytes(b"")
    monkeypatch.setattr(qa.cv2, "VideoCapture", lambda p: types.SimpleNamespace(get=lambda k: 10, release=lambda: None))
    _, rows = qa.verify(str(v), "voc.wav", str(sj), align=str(al), work=str(tmp_path / "w"))
    assert rows[0]["vocal_worst"] == "b -500" and rows[0]["vocal_hiss"] == 0.5


def test_the_vocal_reading_does_not_move_with_the_plan():
    from lsdub_engine.core import vocal_retime as V

    D = os.path.join(os.path.dirname(__file__), "data")
    A = json.load(open(os.path.join(D, "demo_align.json")))
    x = V.read_mono(os.path.join(D, "demo_lead_16k.flac"))
    B = json.loads(json.dumps(A))
    for w in B["words"]:
        w["start"] += 0.7
        w["end"] += 0.7
        for p in w["phones"]:
            p["start"] += 0.7
            p["end"] += 0.7
    a, b = V.retime(A, "", x=x, free=True), V.retime(B, "", x=x, free=True)
    assert [w["start"] for w in a["words"]][:29] != [] and np.abs(
        np.array([w["start"] for w in a["words"]]) - [w["start"] for w in b["words"]]
    ).max() < 0.65
    r = V.retime(A, "", x=x)
    assert V.word_offsets(r["words"], a["words"], 0.0, 16.0)["vocal_words"] == 30


def test_the_review_walks_the_lyric_free_of_the_plan(monkeypatch, tmp_path):
    from lsdub_engine.core import vocal_retime as V

    got = {}
    monkeypatch.setattr(V, "retime", lambda A, v, x=None, free=False: got.update(free=free) or dict(words=[]))
    monkeypatch.setattr(R, "run", lambda *a, **k: got.update(k) or 0)
    al = tmp_path / "a.json"
    al.write_text(json.dumps(dict(words=[], pass_over=[[1.0, 2.0]])))
    (tmp_path / "v.mp4").write_bytes(b"")
    R.main(["--video", str(tmp_path / "v.mp4"), "--align", str(al), "--vocal", "x.wav", "--out", str(tmp_path / "o.mp4")])
    assert got["free"] is True and got["passed"] == [[1.0, 2.0]]


# -- the review panel ---------------------------------------------------------------------------------------------------------------
def test_the_panel_names_its_card_word_card():
    assert R.Panel(1080, 1920).card.tag == "WORD CARD"


def test_the_zoom_box_does_not_breathe_with_a_jittering_mouth():
    n = 80
    c = np.tile([300.0, 200.0], (n, 1))
    w = 80.0 + 20.0 * (-1.0) ** np.arange(n)  # a tracker that swings every frame
    b = R.mouth_boxes(c, w, (480, 640, 3))
    assert np.ptp((b[:, 2] - b[:, 0])[10:-10]) < 8
