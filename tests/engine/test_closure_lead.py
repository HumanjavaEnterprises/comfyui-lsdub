# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""--warp-closure-lead (closure_plan.placed, lead plan | phone). plan is the 0.3.x placement, call for call. phone: a closure is shut
from LEAD_PHONE_S before the aligned P / B / M onset to plan's release end; the gap and minimum rules stay (a closure too short for
MIN_S starts earlier, never ends later); every frame from the window's start on is exactly plan's; a vowel frame ahead of the window
keeps its opening; a PP frame the track carries on a vowel ahead of the window is lifted with the stray rule (a PP on a consonant
there, and every PP on or after the onset, is kept). Through the hold and the shape stage the fall into a closure is the same one-step
fall plan makes, from the same held level: the shorter lead adds no steeper closing."""

import numpy as np
import pytest

from lsdub_engine.core import closure_plan as CP, drive_shape as DS, mouth_paint as MP, nucleus_hold as NH

FPS = 50.0


def frames(n, phone="AA", vis="aa", fps=FPS):
    return [dict(t=round(k / fps, 3), vis=vis, vis2="sil", mix=0.0, open=0.8, word="w", phone=phone) for k in range(n)]


def tag(fr, events):
    """the event stage's `ev` on each frame (the event the frame sits in: a closure / rest between its t_on and t_off, else the last
    nucleus started)."""
    out = []
    for f in fr:
        t = f["t"]
        cur, k = None, -1
        for kk, e in enumerate(events):
            if e["kind"] in ("closure", "rest") and e["t_on"] <= t < e["t_off"]:
                cur, k = e, kk
        if cur is None:
            for kk, e in enumerate(events):
                if e["kind"] == "nucleus" and e["t_on"] <= t:
                    cur, k = e, kk
        ev = dict(k=k, kind=cur["kind"] if cur else "rest", t_on=cur["t_on"] if cur else None, vowel=cur.get("vowel") if cur else None)
        ev.update(db=0.0, bin=1)
        if cur and "t_off" in cur:
            ev["t_off"] = cur["t_off"]
        out.append(dict(f, ev=ev))
    return out


EV = [
    dict(kind="nucleus", t_on=0.2, vowel="AA"),
    dict(kind="closure", t_on=1.0, t_off=1.06),
    dict(kind="nucleus", t_on=1.1, vowel="IY"),
    dict(kind="closure", t_on=1.6, t_off=1.7),
    dict(kind="nucleus", t_on=1.72, vowel="AH"),
]


def song(n=100):
    """a sung line: vowels, a consonant before the second closure, the bilabials on the closures, the track's own PP spread over the
    vowel ahead of each closure (as the viseme track paints an anticipated P / B / M) and a stray PP far from any closure."""
    fr = frames(n)
    for k in range(n):
        t = k / FPS
        if 1.0 <= t < 1.06 or 1.6 <= t < 1.7:
            fr[k].update(phone="M", vis="PP", vis2="aa", open=0.0)
        elif 1.1 <= t < 1.56:
            fr[k].update(phone="IY", vis="ih", open=0.5)
        elif 1.56 <= t < 1.6:
            fr[k].update(phone="S", vis="SS", open=0.3)
    for k in (47, 48, 49):  # 0.94 .. 0.98: the track's PP on the vowel ahead of the first M
        fr[k].update(vis="PP", vis2="aa", open=0.0)
    for k in (78, 79):  # 1.56, 1.58: the track's PP on the consonant ahead of the second M
        fr[k].update(vis="PP", vis2="SS", open=0.0)
    fr[25].update(vis="PP", vis2="aa", open=0.0)  # 0.5: a stray, no closure near
    return tag(fr, EV)


def shut_times(out):
    return [f["t"] for f in out if f.get("pp_aligned")]


def test_default_is_plan_call_for_call():
    for fr in (song(), tag(frames(60), [dict(kind="nucleus", t_on=0.5, vowel="AA"), dict(kind="closure", t_on=0.52, t_off=0.53)]),
               frames(10)):  # fmt: skip
        assert CP.placed(fr, FPS) == CP.placed(fr, FPS, lead="plan")
    ns = MP._parser().parse_args(["--video", "v.mp4", "--track", "t.json", "--out", "o.mp4"])
    assert ns.warp_closure_lead == "plan"
    assert (
        MP._parser().parse_args(["--video", "v", "--track", "t", "--out", "o", "--warp-closure-lead", "phone"]).warp_closure_lead == "phone"
    )
    with pytest.raises(SystemExit):
        MP._parser().parse_args(["--video", "v", "--track", "t", "--out", "o", "--warp-closure-lead", "early"])
    with pytest.raises(ValueError):
        CP.placed(song(), FPS, lead="early")


def test_plan_info_has_no_lead_keys_phone_info_does():
    _, i0 = CP.placed(song(), FPS)
    _, i1 = CP.placed(song(), FPS, lead="phone")
    assert "lead" not in i0 and "lifted_lead" not in i0
    assert i1["lead"] == "phone" and i1["lead_s"] == CP.LEAD_PHONE_S == 0.02


def test_phone_shuts_from_the_stated_lead_to_plans_release():
    out0, i0 = CP.placed(song(), FPS)
    out1, i1 = CP.placed(song(), FPS, lead="phone")
    assert i0["windows"] == [(0.88, 1.06), (1.48, 1.7)]
    assert i1["windows"] == [(0.98, 1.06), (1.58, 1.7)]  # onset - 0.02 .. the same release end
    s1 = shut_times(out1)
    assert s1 == [round(0.98 + 0.02 * k, 2) for k in range(4)] + [round(1.58 + 0.02 * k, 2) for k in range(6)]
    assert all(out1[k]["open"] == 0.0 and out1[k]["vis"] == "PP" for k in range(100) if out1[k].get("pp_aligned"))
    assert i1["frames_shut"] == 10 and i0["frames_shut"] == 20


def test_every_frame_from_the_window_start_on_is_plans():
    """inside the window and the release side, the whole curve, every key: identical to plan."""
    out0, _ = CP.placed(song(), FPS)
    out1, i1 = CP.placed(song(), FPS, lead="phone")
    for a, b in i1["windows"]:
        for k in range(100):
            if a - 1e-9 <= k / FPS < b + 0.06:  # the window, and TOL_S past its release
                assert out1[k] == out0[k], k
    for k in range(int(round(1.7 * FPS)), 100):  # everything after the last closure
        assert out1[k] == out0[k]


def test_a_vowel_ahead_of_the_window_keeps_its_opening():
    fr = song()
    out1, i1 = CP.placed(fr, FPS, lead="phone")
    for k in range(100):
        t = k / FPS
        if 0.6 <= t < 0.94 or 1.1 <= t < 1.56:  # vowel frames ahead of each window, no track PP on them
            assert out1[k] == {**fr[k]}, k
            assert out1[k]["open"] == fr[k]["open"] > 0 and out1[k]["vis"] != "PP"
    opens = np.array([f["open"] for f in out1])
    assert (opens[30:49] > 0.7).all()  # the whole run up to the window is the vowel's opening, no early dip anywhere


def test_a_track_pp_on_the_vowel_ahead_is_lifted_one_on_a_consonant_is_kept():
    out1, i1 = CP.placed(song(), FPS, lead="phone")
    assert i1["lifted_lead"] == [48]  # 0.96 (0.98 is the window's first frame)
    assert 47 in i1["lifted"]  # 0.94: past TOL_S of the onset, the stage's own stray rule lifts it
    for k in (47, 48):
        assert out1[k]["vis"] == "aa" and out1[k]["vis2"] == "sil" and out1[k]["open"] == 0.8 and not out1[k].get("pp_aligned")
    assert out1[49]["vis"] == "PP" and out1[49]["pp_aligned"]
    assert out1[78]["vis"] == "PP" and not out1[78].get("pp_aligned")  # the consonant's own PP stays as the track has it
    assert out1[79]["vis"] == "PP" and out1[79]["pp_aligned"]
    assert 25 in i1["lifted"] and 25 not in i1["lifted_lead"]  # the far stray: the stage's own rule, in both modes
    out0, i0 = CP.placed(song(), FPS)
    assert 25 in i0["lifted"] and 47 not in i0["lifted"] and out0[47]["vis"] == "PP"


def test_gap_and_minimum_rules_stay_and_the_release_never_moves():
    ev = [dict(kind="nucleus", t_on=0.5, vowel="AA"), dict(kind="closure", t_on=0.52, t_off=0.53)]
    _, i0 = CP.placed(tag(frames(60), ev), FPS)
    _, i1 = CP.placed(tag(frames(60), ev), FPS, lead="phone")
    assert i0["windows"] == i1["windows"] == [(0.54, 0.58)]  # GAP_S after the nucleus, MIN_S long, in both
    ev = [dict(kind="nucleus", t_on=0.2, vowel="AA"), dict(kind="closure", t_on=1.0, t_off=1.01)]
    out0, i0 = CP.placed(tag(frames(80), ev), FPS)
    out1, i1 = CP.placed(tag(frames(80), ev), FPS, lead="phone")
    assert i0["windows"] == [(0.88, 1.01)]
    assert i1["windows"] == [(0.97, 1.01)]  # MIN_S from plan's end: it starts earlier, it never ends later
    assert shut_times(out1)[-1] == shut_times(out0)[-1] == 1.0


def chain(lead):
    """closure stage -> hold (nucleus) -> shape (asym, the one-frame advance): the plan's opening per internal frame."""
    out, _ = CP.placed(song(), FPS, lead=lead)
    out, _ = NH.held(out, FPS, mode="nucleus")
    out, _ = DS.shaped(out, FPS, "asym")
    return np.array([f["open"] for f in out])


def test_fall_into_a_closure_is_no_steeper_than_plans():
    o0, o1 = chain("plan"), chain("phone")
    for t_on, lead1 in ((1.0, 2), (1.6, 3)):  # the second: the consonant's own PP at 1.56 is kept, the advance shuts 1.54
        k_on = int(round(t_on * FPS))
        z0 = int(np.flatnonzero(o0[: k_on + 1] > 0)[-1]) + 1  # the first shut frame before the onset
        z1 = int(np.flatnonzero(o1[: k_on + 1] > 0)[-1]) + 1
        assert z0 == k_on - 7 and z1 == k_on - lead1  # shut 0.14 s / 0.04 s ahead (the window's lead + the shape advance)
        f0, f1 = o0[z0 - 1] - o0[z0], o1[z1 - 1] - o1[z1]
        assert f1 <= f0 + 1e-9  # the one-step fall comes from the same held level: no steeper
        d1 = -np.diff(o1[z1 - 12 : z1 + 1])
        assert d1.max() == pytest.approx(f1) and (d1[:-1] <= 1e-9).all()  # before the fall the vowel holds: one fall, no early dip
    d0, d1 = -np.diff(o0), -np.diff(o1)
    assert d1.max() <= d0.max() + 1e-9  # over the whole curve


def test_whole_curve_differs_only_on_the_vowel_given_back():
    o0, o1 = chain("plan"), chain("phone")
    diff = np.flatnonzero(np.abs(o1 - o0) > 1e-9)
    t = diff / FPS
    assert ((t >= 0.86 - 1e-9) & (t < 0.96 + 1e-9) | (t >= 1.46 - 1e-9) & (t < 1.56 + 1e-9)).all()  # only lead-side frames
    assert (o1[diff] > o0[diff]).all()  # each of them more open: the vowel, not a new closing
