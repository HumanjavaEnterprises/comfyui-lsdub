# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""regressions: the erase's nearest closure, the free walk, the lost reading, the lead sheet's proposals, short clips, noise."""

import json, os, types

import numpy as np
import pytest

from lsdub_engine.audits import qa, snap as SN
from lsdub_engine.core import lead_sheet as LS, mouth_warp as MW, vocal_retime as V, vocalisations as VO


class Layer:
    def __init__(self, name):
        self.name, self.items, self.gone = name, {}, []

    def forget(self, i):
        self.gone.append(i)


def test_a_repaired_frame_outside_every_closure_takes_the_nearest():
    a, b = Layer("a"), Layer("b")
    e = MW.EraseSet({1: a, 2: b}, {5: 1, 6: 1, 30: 2})
    assert e.at(10) is a and e.at(25) is b and e.at(6) is a
    e.forget(10)
    assert a.gone == [10] and not e.has(10)


def word(w, *ph, **k):
    return dict(w=w, start=ph[0][1], end=ph[-1][2], phones=[dict(p=p, start=a, end=b) for p, a, b in ph], **k)


def test_a_free_walk_takes_no_length_from_the_plan():
    w = [word("a", ("AA", 0.0, 2.0))]
    assert V.sequence(w)[0][3] > V.sequence(w, free=True)[0][3]
    assert V.sequence(w, free=True)[0][3] == round(V.MIN_S["V"] / V.HOP)


def test_verify_walks_free(monkeypatch):
    seen = {}
    import lsdub_engine.core.vocal_retime as VR

    monkeypatch.setattr(VR, "read_mono", lambda p: np.zeros(100))
    monkeypatch.setattr(VR, "features", lambda x: "fx")
    monkeypatch.setattr(VR, "retime", lambda a, p, x=None, free=False: seen.update(free=free) or dict(words=[]))
    assert qa.vocal_walk({}, "v")[1][1] == "fx" and seen["free"] is True


def test_a_walk_that_ran_away_is_not_reported_as_a_number():
    plan = [dict(w="a", start=float(k), phones=[1]) for k in range(5)]
    assert V.word_offsets(plan, [dict(w="a", start=k + 0.5) for k in range(5)], 0, 9).get("vocal_off_ms") == 500
    assert V.word_offsets(plan, [dict(w="a", start=k + 3.0) for k in range(5)], 0, 19) == dict(vocal_lost=True)
    assert qa._vocal_cell(dict(vocal_lost=True, vocal_hiss=0.5)) == "lost; hiss 0.50"


def span(*kinds, relisten=None):
    return dict(t0=1.0, t1=2.0, kinds=list(kinds), cuts=[], relisten=relisten)


def test_pitch_alone_only_proposes():
    assert LS.proposes(span("register")) and not LS.proposes(span("register", "stutter")) and not LS.proposes(span("stutter"))
    assert not LS.proposes(span("override")) and not LS.proposes(span("register", relisten=["la", "la", "la", "la"]))


def sheet_of(monkeypatch, sp):
    monkeypatch.setattr(LS, "spans_of", lambda x, W: [sp])
    A = dict(words=[word("one", ("AA", 0.2, 1.1)), word("two", ("AA", 1.1, 1.9), src="sung-only"), word("three", ("AA", 2.5, 3.0))])
    monkeypatch.setattr(LS.VR, "retime", lambda a, p, x=None: a)
    return A, LS.read_sheet(A, np.zeros(16000))


def test_a_proposal_passes_no_word_and_leaves_the_mouth(monkeypatch):
    A, S = sheet_of(monkeypatch, span("register"))
    assert all(c["call"] == "LEAD" for c in S["words"])
    assert "proposed" in S["words"][1]["reason"] and S["words"][1]["conf"] < LS.LOW
    assert not LS.honour(A, S).get("pass_over")


def test_a_real_span_still_passes(monkeypatch):
    A, S = sheet_of(monkeypatch, span("stutter", "register"))
    assert S["words"][1]["call"] == "PASS"
    h = LS.honour(A, S)
    assert h["pass_over"] == [[1.0, 2.0]]
    p = [w for w in h["words"] if w.get("vocal") == "pass-over"]
    assert p[0]["start"] == 1.0 and p[-1]["end"] == 2.0  # the passed words reach across the whole span


def test_a_lyric_with_no_room_beside_the_run_is_passed_one_with_room_moves(monkeypatch):
    sp = span("stutter")
    monkeypatch.setattr(LS, "spans_of", lambda x, W: [sp])
    monkeypatch.setattr(LS.VR, "retime", lambda a, p, x=None: a)
    tight = dict(words=[word("one", ("AA", 1.0, 1.9), src="lyric"), word("two", ("AA", 2.0, 3.0), src="lyric")])
    assert LS.read_sheet(tight, np.zeros(16000))["words"][0]["call"] == "PASS"
    roomy = dict(words=[word("one", ("AA", 0.5, 1.9), src="lyric"), word("two", ("AA", 2.5, 3.0), src="lyric")])
    assert LS.read_sheet(roomy, np.zeros(16000))["words"][0].get("move") == [1.0, 2.0]


@pytest.mark.parametrize("bad", ["pass 1.2.3 4", "lead . 2"])
def test_a_bad_number_in_an_override_is_the_usage_line(bad):
    with pytest.raises(ValueError, match="use `pass"):
        LS.parse_overrides(bad)


@pytest.mark.parametrize("n", [0, 100, 400])
def test_a_clip_under_a_window_reads(n):
    assert len(V.features(np.zeros(n))["db"]) >= 1
    r = V.plan_hits([], np.zeros(n), 0.0, 1.0)
    assert r == {}


def test_silence_follows_a_noise_floor_only_on_a_clip_with_gaps():
    gaps = np.r_[np.full(50, -40.0), np.zeros(50)]
    assert V.zero_db(gaps) == pytest.approx(-34.0)
    assert V.zero_db(np.full(100, -21.0)) == V.ZERO_DB  # no gaps: nothing to read
    assert V.zero_db(np.r_[np.full(50, -80.0), np.zeros(50)]) == V.ZERO_DB  # a clean clip


def test_passed_time_stays_out_of_the_silence_reading(monkeypatch):
    op = np.r_[np.ones(20), np.zeros(10), np.linspace(0, 1, 10)]
    keep = np.r_[np.zeros(20, bool), np.ones(20, bool)]
    monkeypatch.setattr(SN, "envelope", lambda vox, n, fps: np.r_[np.zeros(30), np.ones(10)])
    a = SN.snap_metrics(op, None, [], 0.0, 20.0)
    b = SN.snap_metrics(op, None, [], 0.0, 20.0, keep=keep)
    assert a["open_in_silence"] > 0.5 > b["open_in_silence"]


def test_mouth_mode_with_nothing_passed_is_the_unchanged_render(tmp_path, monkeypatch):
    monkeypatch.setattr(VO, "find_spans", lambda *a: pytest.fail("mouth mode with nothing passed reads no spans"))
    T = tmp_path / "t.json"
    A = tmp_path / "a.json"
    A.write_text(json.dumps(dict(words=[word("a", ("AA", 0.0, 1.0))])))
    T.write_text(json.dumps(dict(align=str(A), frames=[dict(vis="aa")] * 4, fps=25, t0=0.0)))
    assert VO.for_warp(types.SimpleNamespace(vocalisations="mouth", track=str(T), vocal_align=None)) is None


def test_near_closed_takes_the_nearest_in_time_of_two_equally_shut():
    n = 40
    S = np.zeros((n, 478, 3))
    S[:, 263, 0] = 100.0
    g = np.ones(n)
    g[[3, 30]] = 0.0
    assert MW.near_closed(S, g, np.zeros((n, 3)), 28) == 30 and MW.near_closed(S, g, np.zeros((n, 3)), 6) == 3


def test_the_cli_stops_on_a_missing_overrides_file(tmp_path, monkeypatch):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(dict(words=[])))
    monkeypatch.setattr(LS.VR, "read_mono", lambda f: np.zeros(100))
    with pytest.raises(FileNotFoundError):
        LS.main(["--align", str(p), "--vocal", "x", "--overrides", str(tmp_path / "none"), "--out", str(tmp_path / "o")])


# -- short clips, the walk's end, the lead sheet's proposals --------------------------------------------------------------------
from lsdub_engine.core import mouth_layers as ML, phone_plan as PP  # noqa: E402


@pytest.mark.parametrize("ph", [[("F", 0.0, 0.02)], [("TH", 0.0, 0.02)], [("DH", 0.0, 0.02)], [("IH", 0.0, 0.03), ("F", 0.03, 0.05)]])
def test_a_song_final_dental_beside_no_vowel_stands(ph):
    d = PP.phones(dict(words=[word("x", *ph)]), None, True, True)
    assert d[-1]["end"] - d[-1]["start"] >= PP.EVENT_S - 1e-9


@pytest.mark.parametrize(
    "words",
    [[], [dict(w="hm", start=0, end=1, phones=[])], [word("x", ("SIL", 0.0, 1.0), vocal="pass-over")]],
)
def test_the_curve_stands_with_nothing_to_read(words):
    so, sw, c = PP.curves(PP.phones(dict(words=words), None, True, True), 0.0, 100, 100.0)
    assert so.shape == (100,) and not c.any()


def test_the_curve_stands_on_no_phones():
    assert PP.curves([], 0.0, 10, 100.0)[0].shape == (10,)


def test_the_teeth_reading_stands_on_lips_past_the_frame():
    fr = np.full((100, 100, 3), 128, np.uint8)
    q = np.zeros((478, 2))
    q[ML.LIP_IN, 1] = np.linspace(90, 108, len(ML.LIP_IN))
    q[ML.LIP_IN, 0] = np.linspace(30, 70, len(ML.LIP_IN))
    assert ML.LayerCompositor.take_teeth(None, [fr], [q]) is None


def test_the_teeth_reading_is_the_takes_teeth():
    fr = np.zeros((100, 100, 3), np.uint8)
    fr[40:60, 30:70] = 200
    q = np.zeros((478, 2))
    k = len(ML.LIP_IN)
    q[ML.LIP_IN] = np.c_[
        30 + 40 * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, k, endpoint=False))),
        50 + 9 * np.sin(np.linspace(0, 2 * np.pi, k, endpoint=False)),
    ]
    assert ML.LayerCompositor.take_teeth(None, [fr], [q]) == pytest.approx(200, abs=10)


def demo():
    d = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(os.path.join(d, "demo_align.json")):
        pytest.skip("the demo vocal is not in this tree")
    return json.load(open(os.path.join(d, "demo_align.json"))), V.read_mono(os.path.join(d, "demo_lead_16k.flac"))


def test_a_clip_cut_after_a_long_vowel_keeps_it():
    A, x = demo()
    i = [w["w"] for w in A["words"]].index("own")
    r = V.retime(dict(A, words=A["words"][: i + 1]), "", x=x[: int(7.3 * V.SR)])
    assert abs(r["words"][i]["start"] - 6.46) < 0.05  # dropped past the end, "my" took its time, 1 s off


def test_words_the_aligner_only_squeezed_in_stay_past_the_cut():
    """the clip cuts the song on "that": the line goes on to "time", 15.3 and 15.57 on the uncut stem, and the aligner fitted it inside."""
    A, x = demo()
    p = V.retime(A, "", x=x)["words"][-1]["phones"]
    assert p[-1]["p"] == "M" and p[-1]["start"] >= len(x) / V.SR


def test_the_final_m_past_the_cut_is_reported_as_such():
    A, x = demo()
    r = V.retime(A, "", x=x)["words"]
    assert [(w, q) for w, q, _ in SN.lost_closures(r, len(x) / V.SR)] == [("time", "M")]


def test_a_free_walk_keeps_its_open_end():
    t = np.arange(V.SR) / V.SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    words = [word("a", ("AA", 0.1, 0.8)), word("b", ("AA", 0.8, 1.6)), word("m", ("AA", 1.6, 1.9), ("M", 1.9, 2.0))]
    r = V.retime(dict(words=words), "", x=x, free=True)
    assert r["words"][-1]["phones"][-1]["start"] >= 1.0  # it stops where the audio does: the lyric's times are not read, nothing is charged


def test_verify_names_a_closure_past_the_clip_or_left_no_time():
    w = [word("time", ("T", 14.8, 15.2), ("M", 15.5, 15.6)), word("up", ("AH", 1.0, 1.1), ("P", 1.1, 1.1))]
    assert SN.lost_closures(w, 15.3) == [("time", "M", 15.5), ("up", "P", 1.1)]
    assert SN.lost_closures([word("later", ("M", 20.0, 20.1))], 15.3) == []
    moved = dict(word("time", ("M", 15.5, 15.6)), start0=14.8)
    assert SN.lost_closures([moved], 15.3) == [("time", "M", 15.5)]


def proposal_case():
    A = dict(words=[word("hello", ("AA", 0.5, 1.0)), word("there", ("AA", 1.0, 1.4)), word("bye", ("AA", 4.0, 4.5))])
    sp = dict(t0=0.9, t1=1.5, kinds=["register"], cuts=[], relisten=None, conf=0.85, reason="sung outside", fragments=[dict(w="x")])
    return A, sp


def test_a_proposal_moves_no_word_even_with_room(monkeypatch):
    A, sp = proposal_case()
    monkeypatch.setattr(LS, "spans_of", lambda x, W: [sp])
    monkeypatch.setattr(LS.VR, "retime", lambda a, p, x=None: a)
    S = LS.read_sheet(A, np.zeros(16000))
    assert not any(c.get("move") for c in S["words"]) and all(c["call"] == "LEAD" for c in S["words"])


def test_honour_applies_no_move_from_a_dropped_span():
    A, sp = proposal_case()
    real = dict(t0=4.0, t1=4.5, kinds=["stutter"], cuts=[], relisten=None, conf=0.9, reason="r", fragments=[dict(w="y")])
    c = [
        dict(i=0, w="hello", t0=0.5, t1=1.0, call="LEAD", conf=1),
        dict(i=1, w="there", t0=1.0, t1=1.4, call="LEAD", conf=0.55, move=[0.9, 1.5]),
        dict(i=2, w="bye", t0=4.0, t1=4.5, call="PASS", conf=0.9),
    ]
    h = LS.honour(A, dict(spans=[sp, real], words=c))
    assert (h["words"][1]["start"], h["words"][1]["end"]) == (1.0, 1.4) and h["pass_over"] == [[4.0, 4.5]]


def test_the_sheet_shows_a_proposal_with_a_mark_and_not_as_passed():
    A, sp = proposal_case()
    c = [dict(i=k, w=w["w"], t0=w["start"], t1=w["end"], call="LEAD", conf=1.0, reason="", line=0) for k, w in enumerate(A["words"])]
    t = LS.as_text(dict(spans=[sp], words=c))
    assert "{" not in t.split("#")[0] and "? proposed, not passed" in t and "passed over, confidence" not in t


def test_verify_reads_the_curve_flags_the_dub_logged(tmp_path):
    from lsdub_engine.core import coart

    w = tmp_path / "w.json"
    (tmp_path / "w_coart.json").write_text(json.dumps(dict(dyn_flags=["--phone-plan"])))
    assert coart.logged_dyn_flags(str(w)) == ["--phone-plan"]
    assert coart.logged_dyn_flags(str(tmp_path / "none.json")) == []


def test_a_rate_that_passes_its_gate_beats_one_that_fails():
    from lsdub_engine.core import sing

    good, bad = dict(corr=0.79, open_in_silence=0.28, **{"pass": True}), dict(corr=0.81, open_in_silence=0.351, **{"pass": False})
    assert sing.pick_paint(bad, good) == "native" and sing.pick_paint(good, bad) == "sing"


def test_of_two_that_pass_the_better_sync_wins_and_within_a_margin_the_quieter():
    from lsdub_engine.core import sing

    g = lambda c, o: dict(corr=c, open_in_silence=o, **{"pass": True})
    assert sing.pick_paint(g(0.9, 0.3), g(0.7, 0.1)) == "sing" and sing.pick_paint(g(0.7, 0.1), g(0.9, 0.3)) == "native"
    assert sing.pick_paint(g(0.8, 0.3), g(0.81, 0.1)) == "native" and sing.pick_paint(g(0.8, None), g(0.8, 0.3)) == "native"


def test_verifys_meta_carries_what_it_cannot_show(tmp_path, monkeypatch):
    p, sh = tmp_path / "a.json", tmp_path / "s.json"
    p.write_text(json.dumps(dict(words=[dict(word("time", ("M", 15.5, 15.6)), start0=14.8)])))
    sh.write_text(json.dumps(dict(fps=25.0, shots=[])))
    monkeypatch.setattr(qa.cv2, "VideoCapture", lambda v: types.SimpleNamespace(get=lambda k: 382, release=lambda: None))
    monkeypatch.setattr(qa, "vad", lambda *a: (np.zeros(382), np.zeros(382)))
    monkeypatch.setattr(qa, "vocal_walk", lambda *a: ([], None))
    import lsdub_engine.core.cast as C

    monkeypatch.setattr(C, "Faces", lambda: None)
    meta, rows = qa.verify("v.mp4", "x.wav", str(sh), align=str(p), work=str(tmp_path))
    assert meta["closures_lost"] == [("time", "M", 15.5)] and rows == []


def test_verifys_curve_gets_the_flags_the_dub_logged(tmp_path, monkeypatch):
    class Stop(Exception):
        pass

    seen = []

    def call(name, argv, **k):
        seen.append(argv)
        raise Stop

    monkeypatch.setattr(qa.steps, "call", call)
    monkeypatch.setattr(qa.ff, "run", lambda *a, **k: None)
    (tmp_path / "w_coart.json").write_text(json.dumps(dict(dyn_flags=["--phone-plan", "--function-gate"])))
    Q = types.SimpleNamespace(
        read_span=lambda *a: [0] * 4, singer_roi=lambda *a: [0, 1, 0, 1], measured_opening=lambda F, r, f: (np.zeros(4), 1.0)
    )
    sn = types.SimpleNamespace(envelope=lambda *a: np.zeros(4))
    s = dict(shot=0, f0=0, f1=3, start=0.0, box=[0, 0, 1, 1], warp_json=str(tmp_path / "w.json"))
    with pytest.raises(Stop):
        qa.measure_shot("v.mp4", s, 25.0, 0.0, "x.wav", np.ones(4), "a.json", [], None, str(tmp_path), Q, sn)
    assert seen[0][-2:] == ["--phone-plan", "--function-gate"]


def test_the_plan_log_names_the_curves_flags(tmp_path, monkeypatch):
    from lsdub_engine.core import coart

    T = tmp_path / "t.json"
    T.write_text(json.dumps(dict(fps=25, t0=0.0, frames=[dict(vis="sil")] * 6)))
    info = dict(
        closures=[],
        n_segments=0,
        dom_ms=0,
        undershoot=0,
        prior=0,
        lambda_open=0,
        lambda_width=0,
        closure_cap_p5=0,
        pauses=[],
        rest=[0, 0],
        range_ratio_plan=[1, 1],
        hull_projected=0,
        dyn_mix=0,
    )
    monkeypatch.setattr(coart, "jaw_split", lambda *a, **k: (1.0, 1.0))
    monkeypatch.setattr(coart, "plan", lambda *a: (np.zeros((6, 2)), np.zeros(6, bool), dict(info)))
    monkeypatch.setattr(coart, "press_closures", lambda Ft, *a: Ft)
    model = types.SimpleNamespace(rng=dict(g_lo=0.0, g_hi=1.0, w_lo=-1.0, w_med=0.0, w_hi=1.0))
    a = types.SimpleNamespace(
        track=str(T), coart_phones=True, coart_function_gate=False, warp_json=str(tmp_path / "w.json"), warp_deadband=0.04
    )
    coart.plan_clip(a, json.load(open(T))["frames"], np.zeros((6, 2)), model, None)
    assert coart.logged_dyn_flags(a.warp_json) == ["--phone-plan"]
