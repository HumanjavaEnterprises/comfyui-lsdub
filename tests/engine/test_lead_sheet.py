# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the lead sheet: what the singer on screen sings, the rest in braces. The real demo vocal, its truth read off the audio: a stutter of ten or so
fragments from the end of "time" to a high answer note, the lead back on "it's" at 11.96."""

import json, os

import numpy as np
import pytest

from lsdub_engine.core import lead_sheet as LS, vocal_retime as V, viseme_track as VT, vocalisations as VO

DATA = os.path.join(os.path.dirname(__file__), "data")
SHEET = "aged like rocks, still in the box\ntimeless on the shelf, winding my own clock\nit's, it's about that time\nhits, it's about that time\nthis is about that time\n"


@pytest.fixture(scope="module")
def demo():
    if not os.path.exists(os.path.join(DATA, "demo_align.json")):
        pytest.skip("the demo vocal is not in this tree")
    A = json.load(open(os.path.join(DATA, "demo_align.json")))
    x = V.read_mono(os.path.join(DATA, "demo_lead_16k.flac"))
    return A, x, LS.read_sheet(A, x, SHEET)


def tone(s, f0=330.0):
    t = np.arange(int(s * V.SR)) / V.SR
    return 0.3 * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))


def word(w, ph, **k):
    return dict(w=w, start=ph[0][1], end=ph[-1][2], phones=[dict(p=p, start=a, end=b) for p, a, b in ph], **k)


def stutter(n=9, step=0.12):
    y = tone(n * step)
    for k in range(1, n):
        c = int(k * step * V.SR)
        y[c - 320 : c + 320] *= 0.05
    return y


def test_the_demo_run_is_found_where_the_audio_has_it(demo):
    _, _, S = demo
    assert len(S["spans"]) == 1
    q = S["spans"][0]
    assert q["t0"] == pytest.approx(10.04, abs=0.05) and q["t1"] == pytest.approx(11.93, abs=0.05)
    assert q["kinds"] == ["register", "stutter"] and q["conf"] >= LS.LOW


def test_the_demo_lead_truth_read_from_the_audio(demo):
    _, _, S = demo
    calls = {c["i"]: c["call"] for c in S["words"]}
    assert calls[20] == "PASS"  # "if": the sheet's "hits", no /h/ and no /f/ in the vocal, the stutter under it
    assert all(c == "LEAD" for i, c in calls.items() if i != 20)
    moved = S["words"][21]
    assert moved["w"] == "it's" and moved["conf"] < LS.LOW and moved.get("move")  # marked, not decided in silence


def test_the_sheet_reads_as_written_by_hand(demo):
    t = LS.as_text(demo[2])
    assert "about that time {time time time ti " in t and " time}\nit's? about that time\n" in t
    assert "10.04-11.9" in t and "low confidence" in t


def test_fragments_echo_the_word_before_with_short_pieces_cut_down(demo):
    f = demo[2]["spans"][0]["fragments"]
    assert [x["w"] for x in f][:3] == ["time"] * 3 and f[-1]["w"] == "time" and "ti" in [x["w"] for x in f]
    assert f[0]["t0"] == demo[2]["spans"][0]["t0"] and all(a["t1"] == b["t0"] for a, b in zip(f, f[1:]))


def test_honoured_the_passed_time_drives_no_phone_and_the_retime_stays_out(demo):
    A, x, S = demo
    H = LS.honour(A, S)
    assert H["pass_over"] == [[S["spans"][0]["t0"], S["spans"][0]["t1"]]] and [w["w"] for w in H["words"]] == [w["w"] for w in A["words"]]
    r = V.retime(H, "", x=x)
    a, b = H["pass_over"][0]
    for w in r["words"]:
        if w.get("vocal") == LS.PASS:
            assert [p["p"] for p in w["phones"]] == ["SIL"]
            continue
        for p in w["phones"]:
            assert p["end"] <= a + 0.02 or p["start"] >= b - 0.02, (w["w"], p)
    ph = r["words"][21]["phones"]
    assert ph[0]["start"] == pytest.approx(11.96, abs=0.06) and ph[-2]["start"] == pytest.approx(
        12.07, abs=0.03
    )  # the lead's own "it's", its /s/ on the hiss


def test_nothing_changes_for_an_alignment_with_no_pass_over(demo):
    A, x, _ = demo
    assert LS.honour(A, dict(spans=[], words=[])) == A
    plain = V.retime(A, "", x=x)
    assert "pass_over" not in plain and all("vocal" not in w for w in plain["words"])


def test_a_passed_word_marks_the_track_skip_so_no_mouth_comes_of_it_and_verify_does_not_score_it(demo):
    A, x, S = demo
    H = V.retime(LS.honour(A, S), "", x=x)
    occ = [dict(w=w["w"], t0=w["start"], t1=w["end"], ai=i, mark=w.get("vocal")) for i, w in enumerate(H["words"])]
    VO.classify(occ, all_words=[w["w"] for w in H["words"]])
    assert [o["w"] for o in occ if o["vocal"]] == ["if"] and occ[20]["reason"] == "markup:pass-over"
    assert VT.ARPA["SIL"] == "sil"
    assert not VT.phone_list(H)[0]["vis"] == "sil" and [p["vis"] for p in VT.phone_list(H) if p["word"] == "if" and p["wi"] == 20] == [
        "sil"
    ]
    fr = V.word_offsets(H["words"], H["words"], 0.0, 16.0)
    assert fr["vocal_words"] == 29  # 30 words, the passed one unread


def test_plan_hits_leaves_out_passed_phones(demo):
    A, x, S = demo
    H = V.retime(LS.honour(A, S), "", x=x)
    q = V.plan_hits(H["words"], x, 0.0, 15.3)
    assert set(q) == {"vocal_hiss", "vocal_shut", "vocal_open"} and q["vocal_hiss"] > 0.5


def test_override_lead_gives_time_back_and_pass_takes_it(demo):
    A, x, _ = demo
    S = LS.read_sheet(A, x, SHEET, "lead 10.04 11.0\npass 2.5 2.8  # a double")
    sp = [(q["t0"], q["t1"]) for q in S["spans"]]
    assert sp[0] == (2.5, 2.8) and sp[1][0] == 11.0 and sp[1][1] > 11.8
    assert S["spans"][0]["conf"] == 1.0 and "by hand" in S["spans"][0]["reason"]
    assert next(c for c in S["words"] if c["i"] == 6)["t0"] < 2.8


def test_override_the_whole_run_away_leaves_every_word_lead(demo):
    A, x, _ = demo
    S = LS.read_sheet(A, x, SHEET, "lead 9 13")
    assert S["spans"] == [] and all(c["call"] == "LEAD" for c in S["words"])


@pytest.mark.parametrize("bad", ["pass", "pass 3 2", "keep 1 2", "pass one two"])
def test_an_override_line_it_cannot_read_stops_it(bad):
    with pytest.raises(ValueError):
        LS.parse_overrides(bad)


def test_a_held_vowel_and_a_clean_line_pass_nothing():
    x = np.r_[tone(1.0), np.zeros(1600), tone(1.5, 300.0), np.zeros(1600), tone(1.0, 340.0)]
    A = dict(words=[word("oh", [("OW", 0.0, 1.0)]), word("oh", [("OW", 1.1, 2.6)]), word("oh", [("OW", 2.7, 3.7)])])
    assert LS.read_sheet(A, x)["spans"] == []


def test_a_stutter_is_a_train_and_six_notches_are_not():
    assert len(LS.stutters(V.features(np.r_[tone(0.5), stutter(10), tone(0.5)]))) == 1
    assert LS.stutters(V.features(np.r_[tone(0.5), stutter(6), tone(0.5)])) == []


def test_a_high_answer_is_off_register():
    x = np.r_[tone(1.5, 300.0), tone(0.5, 530.0), tone(1.5, 310.0)]
    r = LS.off_register(LS.pitch(x))
    assert len(r) == 1 and r[0][0] * V.HOP == pytest.approx(1.5, abs=0.06) and (r[0][1] - r[0][0]) * V.HOP == pytest.approx(0.5, abs=0.08)
    assert LS.off_register(LS.pitch(np.r_[tone(1.0, 300.0), tone(0.1, 530.0), tone(1.0, 300.0)])) == []  # too short to be a line


def test_relisten_lifts_a_repeat_and_doubts_a_word():
    q = dict(kinds=["stutter"], cuts=[1, 2, 3, 4, 5, 6, 7, 8])
    assert LS.confidence(q)[0] == 0.7 and LS.confidence(q, ["la", "la", "la"])[0] == 0.8 and LS.confidence(q, ["about", "that"])[0] == 0.5


def test_low_confidence_is_marked_in_the_braces():
    S = dict(
        words=[dict(i=0, w="a", t0=0.0, t1=1.0, call="LEAD", conf=1.0, reason="", line=0)],
        spans=[dict(t0=1.0, t1=2.0, conf=0.4, reason="x", fragments=[dict(w="a", t0=1.0, t1=2.0)])],
    )
    assert "a {? a}\n" in LS.as_text(S)


def test_is_passed():
    assert LS.is_passed([[1.0, 2.0]], [0.5, 1.0, 1.9, 2.0]).tolist() == [False, True, True, False]


def test_a_paint_in_mouth_mode_still_keeps_passed_time_out_of_the_singers_mouth(demo, tmp_path):
    import argparse

    A, x, S = demo
    H = V.retime(LS.honour(A, S), "", x=x)
    al, tr = tmp_path / "a.json", tmp_path / "t.json"
    al.write_text(json.dumps(H))
    n = 137
    t0 = 8.0
    VT.main(["--align", str(al), "--t0", str(t0), "--frames", str(n), "--fps", "25", "--out", str(tr)])
    a = argparse.Namespace(
        vocalisations="mouth", track=str(tr), vocal_align=str(al), skip_words=None, skip_lyrics=None, vocal_merge_gap=0.75, vocal_ease=3
    )
    T = json.load(open(tr))
    f = VO.find_spans(T, str(al))
    assert [q["reasons"] for q in f["spans"] if "markup:pass-over" in q["reasons"]] and f["spans"][0]["t0"] == pytest.approx(
        10.04, abs=0.05
    )
    assert VO.for_warp(a).mode == "skip"
    H["words"] = [dict(w, vocal=None) for w in H["words"]]
    al.write_text(json.dumps(H))
    assert VO.for_warp(a) is None  # no passed word, mouth mode is what it was


def test_the_passed_words_cover_the_whole_span_so_no_frame_of_it_is_left_to_the_plan(demo):
    A, x, S = demo
    H = LS.honour(A, S)
    a, b = H["pass_over"][0]
    w = [w for w in H["words"] if w.get("vocal") == LS.PASS]
    assert w[0]["start"] == a and w[-1]["end"] == b and all(p["end"] > p["start"] for q in w for p in q["phones"])


def test_a_span_with_no_word_in_it_gets_a_passed_word_of_its_own(demo):
    A, x, _ = demo
    S = LS.read_sheet(A, x, SHEET, "lead 9 13\npass 1.45 1.58")
    H = LS.honour(A, S)
    assert len(H["words"]) == len(A["words"]) + 1
    w = [w for w in H["words"] if w.get("vocal") == LS.PASS]
    assert (w[0]["start"], w[0]["end"]) == (1.45, 1.58)
    assert [q["start"] for q in H["words"]] == sorted(q["start"] for q in H["words"])
