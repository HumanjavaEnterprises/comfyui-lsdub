# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""closure_plan: a closure's shut window is exact (LEAD_S ahead of the aligned onset, to the release, at least MIN_S); a closure right
after a nucleus starts GAP_S after it; a PP frame with no closure event within TOL_S is lifted to the class under it with a neighbour's
opening; frames without events come back unchanged."""

import pytest

from lsdub_engine.core import closure_plan as CP

FPS = 50.0


def frames(n, fps=FPS):
    return [dict(t=round(k / fps, 3), vis="aa", vis2="sil", mix=0.0, open=0.8, word="w", phone="AA") for k in range(n)]


def tag(frames, events):
    out = []
    for f in frames:
        t = f["t"]
        cur, k = None, -1
        for kk, e in enumerate(events):
            if e["kind"] in ("closure", "rest") and e["t_on"] <= t < e["t_off"]:
                cur, k = e, kk
        if cur is None:
            for kk, e in enumerate(events):
                if e["kind"] == "nucleus" and e["t_on"] <= t:
                    cur, k = e, kk
        ev = dict(
            k=k,
            kind=cur["kind"] if cur else "rest",
            t_on=cur["t_on"] if cur else None,
            vowel=cur.get("vowel") if cur else None,
            db=0.0,
            bin=1,
        )
        if cur and "t_off" in cur:
            ev["t_off"] = cur["t_off"]
        out.append(dict(f, ev=ev))
    return out


def test_the_closure_window_is_lead_s_ahead_of_the_onset_to_the_release():
    ev = [
        dict(kind="nucleus", t_on=0.2, vowel="AA"),
        dict(kind="closure", t_on=1.0, t_off=1.06),
        dict(kind="nucleus", t_on=1.1, vowel="IY"),
    ]
    out, info = CP.placed(tag(frames(100), ev), FPS)
    pp = [f["t"] for f in out if f["vis"] == "PP"]
    assert pp[0] == pytest.approx(0.88) and pp[-1] == pytest.approx(1.04)  # 1.0 - 0.12 .. 1.06 (end exclusive)
    assert all(f["open"] == 0.0 for f in out if f["vis"] == "PP") and all(f["vis2"] == "aa" for f in out if f["vis"] == "PP")
    assert info["closures"] == 1 and info["windows"] == [(0.88, 1.06)] and info["frames_shut"] == 9 and info["lifted"] == []
    assert all(f.get("pp_aligned") for f in out if f["vis"] == "PP") and not any(f.get("pp_aligned") for f in out if f["vis"] != "PP")


def test_a_closure_lasts_at_least_min_s_and_starts_gap_s_after_the_nucleus_before_it():
    ev = [dict(kind="nucleus", t_on=0.5, vowel="AA"), dict(kind="closure", t_on=0.52, t_off=0.53)]
    out, info = CP.placed(tag(frames(60), ev), FPS)
    assert info["windows"] == [(0.54, 0.58)]  # not before 0.5 + GAP_S; at least MIN_S long
    pp = [f["t"] for f in out if f["vis"] == "PP"]
    assert pp == [0.54, 0.56]


def test_a_stray_pp_frame_is_lifted_to_its_class_with_a_neighbours_opening():
    fr = frames(60)
    for k in range(20, 23):  # the plan's PP with no aligned closure near
        fr[k].update(vis="PP", vis2="E", open=0.0)
    for k in range(44, 46):  # the plan's PP where the alignment has one
        fr[k].update(vis="PP", vis2="aa", open=0.0)
    ev = [dict(kind="nucleus", t_on=0.1, vowel="AA"), dict(kind="closure", t_on=0.9, t_off=0.94)]
    out, info = CP.placed(tag(fr, ev), FPS)
    assert info["lifted"] == [20, 21, 22] and all(out[k]["vis"] == "E" and out[k]["open"] == 0.8 for k in (20, 21, 22))
    assert all(out[k]["vis"] == "PP" for k in range(39, 47))  # 0.78 .. 0.94: the aligned closure's window, the plan's PP kept inside it


def test_frames_without_events_come_back_unchanged_and_every_other_key_is_kept():
    fr = frames(10)
    out, info = CP.placed(fr, FPS)
    assert out == fr and info["closures"] == 0
    tagged = tag(frames(30), [dict(kind="nucleus", t_on=0.1, vowel="AA"), dict(kind="closure", t_on=0.4, t_off=0.44)])
    out, _ = CP.placed(tagged, FPS)
    assert all(set(f) - {"pp_aligned"} == set(g) for f, g in zip(out, tagged)) and tagged[20]["vis"] == "aa"  # the input is not changed


def test_frames_with_no_events_come_back_unchanged_with_an_empty_lifted_list():
    """the auto path's native candidate paints with no --warp-audio, so its track carries no events: the frames are returned as
    they are and `lifted` is a list like every other path's (the paint's log line takes its len; 0.2.0's int there failed every shot)."""
    frames = [dict(t=i / 25, vis="PP" if i in (3, 4) else "E", open=0.5) for i in range(10)]
    out, info = CP.placed(frames, 25)
    assert out == frames and info["lifted"] == [] and len(info["lifted"]) == 0 and info["closures"] == 0
