# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""pins for what the independent check found: the erase's nearest closure, the free walk, the lost reading, the lead sheet's proposals, short clips, noise."""

import json, types

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


def test_mouth_mode_with_nothing_passed_is_the_unchanged_render(tmp_path):
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
