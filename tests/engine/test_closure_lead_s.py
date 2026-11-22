# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""--warp-closure-lead-s (closure_plan.placed lead_s, read only under --warp-closure-lead phone), and the closure stage's default pinned
against a FROZEN output: tests/data/closure_plan_6ed5bb7.json holds fixed inputs (hand-built cases of each rule, and two real
event-stage outputs of S7 shots cut near their closures) with what 6ed5bb7's placed(frames, fps) returned on them (before the lead
flag existed), and what fd4b4e5's placed(..., lead="phone") returned (the lead flag's first commit, fixed at 0.02 s). The default and
lead plan must reproduce the first exactly, whatever lead_s says; phone at the default lead the second. The lead is a sweep parameter:
the window starts lead_s before the onset (gap and minimum rules kept, the release never moves), 0 <= lead_s <= LEAD_S. A track PP
further than TOL_S ahead of the onset and outside the phone window is a stray on a consonant as on a vowel; within TOL_S a
consonant's PP is kept."""

import json
import math
import os
from types import SimpleNamespace

import pytest

from lsdub_engine.core import closure_plan as CP, mouth_paint as MP, recipes as RC

DATA = json.load(open(os.path.join(os.path.dirname(__file__), "data", "closure_plan_6ed5bb7.json")))
CASES = sorted(DATA["cases"])


def run(c, **kw):
    fr = json.loads(json.dumps(c["frames"]))
    o, info = CP.placed(fr, c["fps"], c["t0"], **kw) if c["t0"] is not None else CP.placed(fr, c["fps"], **kw)
    return json.loads(json.dumps(o)), json.loads(json.dumps(info))


def parse(*extra):
    return MP._parser().parse_args(["--video", "v.mp4", "--track", "t.json", "--out", "o.mp4", *extra])


@pytest.mark.parametrize("name", CASES)
def test_default_and_plan_reproduce_the_frozen_pre_flag_output(name):
    c = DATA["cases"][name]
    for kw in ({}, dict(lead="plan"), dict(lead="plan", lead_s=0.07), dict(lead="plan", lead_s=-5.0)):  # plan never reads lead_s
        o, info = run(c, **kw)
        assert o == c["out"], (name, kw)
        assert info == c["info"], (name, kw)


@pytest.mark.parametrize("name", CASES)
def test_phone_at_the_default_lead_reproduces_the_first_flag_commit(name):
    c = DATA["cases"][name]
    for kw in (dict(lead="phone"), dict(lead="phone", lead_s=0.02), dict(lead="phone", lead_s=CP.LEAD_PHONE_S)):
        o, info = run(c, **kw)
        assert o == c["phone_fd4b4e5"]["out"], (name, kw)
        assert info == c["phone_fd4b4e5"]["info"], (name, kw)


def test_the_frozen_inputs_exercise_the_rules():
    """the fixture is not empty: windows, lifts, the nearest-frame fallback and the no-event / empty paths are all in it."""
    infos = {k: DATA["cases"][k]["info"] for k in CASES}
    assert infos["window_under_a_frame"]["frames_shut"] == 1
    assert infos["no_events"]["closures"] == 0 and infos["empty"]["closures"] == 0
    assert sum(len(i.get("lifted", [])) for i in infos.values()) >= 10
    assert sum(len(i.get("windows", [])) for i in infos.values()) >= 10
    assert any(DATA["cases"][k]["out"] != DATA["cases"][k]["phone_fd4b4e5"]["out"] for k in CASES)


def test_lead_s_validation():
    assert CP.lead_seconds(0) == 0.0 and CP.lead_seconds("0.12") == CP.LEAD_S
    for bad in (-0.001, 0.1201, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError):
            CP.lead_seconds(bad)
        with pytest.raises(ValueError):
            CP.placed(json.loads(json.dumps(DATA["cases"]["two_closures_track_pp_ahead"]["frames"])), 50.0, lead="phone", lead_s=bad)
    assert parse().warp_closure_lead_s == 0.02 == CP.LEAD_PHONE_S
    assert parse("--warp-closure-lead", "phone", "--warp-closure-lead-s", "0.06").warp_closure_lead_s == 0.06
    assert parse("--warp-closure-lead-s", "0").warp_closure_lead_s == 0.0
    assert parse("--warp-closure-lead", "phone").warp_closure_lead == "phone"  # the shorter flag is not taken as an abbreviation
    for bad in ("-0.01", "0.13", "nan", "abc"):
        with pytest.raises(SystemExit):
            parse("--warp-closure-lead-s", bad)


def test_lead_s_rule_names_the_inert_cases():
    when = next(r for r in RC.RULES if r.flags == ("--warp-closure-lead-s",)).when
    assert when(SimpleNamespace(warp_closures="aligned", warp_closure_lead="plan"))
    assert when(SimpleNamespace(warp_closures="plan", warp_closure_lead="phone"))
    assert not when(SimpleNamespace(warp_closures="aligned", warp_closure_lead="phone"))
    assert not when(SimpleNamespace(warp_closures="aligned-press", warp_closure_lead="phone"))


def windows(lead_s, name="two_closures_track_pp_ahead"):
    return [tuple(w) for w in run(DATA["cases"][name], lead="phone", lead_s=lead_s)[1]["windows"]]


def test_the_window_starts_lead_s_before_the_onset_and_ends_at_plans_release():
    c = DATA["cases"]["two_closures_track_pp_ahead"]
    plan = [tuple(w) for w in c["info"]["windows"]]  # [(0.88, 1.06), (1.48, 1.7)]: onsets 1.0, 1.6
    for lead_s in (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12):
        w = windows(lead_s)
        assert [b for _, b in w] == [b for _, b in plan]  # the release side never moves
        assert w == [(round(1.0 - lead_s, 3), 1.06), (round(1.6 - lead_s, 3), 1.7)], lead_s
    assert windows(CP.LEAD_S) == plan  # at plan's lead, plan's windows
    o12, _ = run(c, lead="phone", lead_s=CP.LEAD_S)
    assert [f.get("pp_aligned", False) for f in o12] == [f.get("pp_aligned", False) for f in c["out"]]


def test_gap_and_minimum_rules_hold_at_every_lead():
    c = DATA["cases"]["gap_rule_and_minimum"]  # a nucleus at 0.5, a closure at 0.52: GAP_S binds at any lead
    for lead_s in (0.0, 0.02, 0.06, 0.12):
        assert windows(lead_s, "gap_rule_and_minimum") == [tuple(w) for w in c["info"]["windows"]] == [(0.54, 0.58)]
    for lead_s in (0.0, 0.02, 0.03):  # a 0.01 s closure: MIN_S from plan's end, so the start is 0.97 whatever the shorter lead
        assert windows(lead_s, "short_closure") == [(0.97, 1.01)]
    assert windows(0.06, "short_closure") == [(0.94, 1.01)]


def test_shut_frames_grow_with_the_lead():
    for name in CASES:
        prev = None
        for lead_s in (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12):
            o, _ = run(DATA["cases"][name], lead="phone", lead_s=lead_s)
            s = {k for k, f in enumerate(o) if f.get("pp_aligned")}
            assert prev is None or prev <= s, (name, lead_s)
            prev = s


def test_a_consonants_track_pp_past_tol_ahead_of_the_window_is_a_stray():
    """two_closures: the T before the second M carries the track's PP at 1.50 .. 1.58 (onset 1.60). Under plan, 1.48 .. 1.70 is the
    shut window, so they are shut. Under phone at 0.02 the window is 1.58 .. 1.70: 1.50 and 1.52 sit 0.10 / 0.08 s ahead, past TOL_S,
    and are lifted by the stray rule like any PP no event owns (a tongue consonant is not a lip closure, and the closure's timing is
    the aligned onset's); 1.56 sits within TOL_S and is kept as the track has it (the stop running into the M); 1.58 is the window."""
    c = DATA["cases"]["two_closures_track_pp_ahead"]
    o, info = run(c, lead="phone")
    k = {round(f["t"], 2): i for i, f in enumerate(o)}
    for t in (1.5, 1.52):
        i = k[t]
        assert c["out"][i]["vis"] == "PP" and c["out"][i]["pp_aligned"]  # plan: inside its window
        assert i in info["lifted"] and i not in info["lifted_lead"]  # phone: the stray rule, not the vowel rule
        assert o[i]["vis"] == "DD" and o[i]["vis2"] == "sil" and o[i]["open"] > 0 and not o[i].get("pp_aligned")
    i = k[1.56]
    assert o[i]["vis"] == "PP" and not o[i].get("pp_aligned") and i not in info["lifted"]
    assert o[k[1.58]]["pp_aligned"]
    for t in (1.5, 1.52):  # a longer lead reaches them: they are shut again as the window covers them
        o8, _ = run(c, lead="phone", lead_s=0.1)
        assert o8[k[t]]["pp_aligned"]


def test_a_vowels_track_pp_within_tol_is_lifted_and_its_opening_given_back():
    c = DATA["cases"]["two_closures_track_pp_ahead"]  # the vowel ahead of the M at 1.0 carries PP at 0.88 .. 0.98
    o, info = run(c, lead="phone")
    k = {round(f["t"], 2): i for i, f in enumerate(o)}
    assert k[0.96] in info["lifted_lead"]
    for t in (0.88, 0.9, 0.92, 0.96):
        assert o[k[t]]["vis"] == "aa" and math.isclose(o[k[t]]["open"], 0.8)
    assert o[k[0.98]]["pp_aligned"]
