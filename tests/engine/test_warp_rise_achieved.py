# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""--warp-rise achieved (0.3.2 step B2): the onset rise read on the rendered mouth; only the short ones lifted, with an attack shape."""

import types

import numpy as np
import pytest

from lsdub_engine.core import mouth_paint, mouth_warp as mw, rise_loop as RL

FPS = 50.0  # the paint's internal rate (2x the output's 25)


def track(n, onsets, hold=0.1):
    """track frames: onsets (s) with a nucleus tagged from the onset to `hold` after it."""
    fr = [dict(t=0.0, vis="aa") for _ in range(n)]
    for k, on in enumerate(onsets):
        for i in range(n):
            if on - 0.02 <= (i + 0.5) / FPS <= on + hold:
                fr[i]["ev"] = dict(kind="nucleus", t_on=on, k=k)
    return fr


def ramp_reading(n, at, size, base=0.2):
    """a reading with one rise of `size` over frames at-2 .. at+2."""
    m = np.full(n, base)
    m[at - 1] += size * 0.25
    m[at] += size * 0.5
    m[at + 1] += size * 0.75
    m[at + 2 :] += size
    return m


def test_events_one_per_nucleus_with_its_last_frame():
    ev = RL.events(track(100, [0.5, 1.2], hold=0.1), 100, FPS)
    assert [round(e[0], 3) for e in ev] == [0.5, 1.2]
    assert ev[0][1] == int(np.floor(0.6 * FPS - 0.5)) and ev[1][1] == int(np.floor(1.3 * FPS - 0.5))
    assert RL.events([], 0, FPS) == []


def test_best_rise_reads_across_the_onset_and_picks_the_largest_centre():
    t, _ = mw.onset_frames(track(100, [0.5]), 100, FPS)
    m = ramp_reading(100, 25, 0.3)
    c, r = RL.best_rise(m, t, 0.5, 2)
    assert c == 25 and r == pytest.approx(0.3)
    m[24] = np.nan  # an unread end moves the centre to one with both ends read
    c, _ = RL.best_rise(m, t, 0.5, 2)
    assert c != 26 and c != 22
    assert RL.best_rise(np.full(100, np.nan), t, 0.5, 2) is None


def test_best_peak_reads_the_largest_peak_and_its_neighbours():
    t, _ = mw.onset_frames(track(100, [0.5]), 100, FPS)
    m = ramp_reading(100, 25, 0.3)
    c, r, nb = RL.best_peak(m, t, 0.5, 2)
    assert c == 25 and r == pytest.approx(0.3) and nb < r
    assert RL.best_peak(np.full(100, np.nan), t, 0.5, 2) is None


def test_a_rise_that_is_not_a_peak_is_short_even_when_big():
    m = np.zeros(60)
    m[20:] += 0.2
    m[24:] += 0.6  # a bigger rise one output frame (2 internal frames) after the first
    r, nb = RL.rise_at(m, 20, 2)
    assert r == pytest.approx(0.2) and nb == pytest.approx(0.6)
    assert RL.need_of(r, nb, 0.15) > 0
    assert RL.need_of(0.3, 0.5, 0.15) == pytest.approx(2 * 0.2 + 0.05 * 0.15)
    assert RL.need_of(0.3, 0.1, 0.15) == 0.0
    assert RL.need_of(0.05, 0.0, 0.15) == pytest.approx(0.10)


def test_attack_is_a_ramp_a_hold_and_a_release_never_a_bump():
    p = RL.attack(60, 20, 2, hold_to=26, stop=60, release=8)
    assert p[18] == 0.0 and p[19] == pytest.approx(0.25) and p[20] == pytest.approx(0.5) and p[22] == 1.0
    assert (p[22:27] == 1.0).all()  # held through the nucleus
    rel = p[27:35]
    assert (np.diff(rel) < 0).all() and rel[0] < 1.0 and p[35] == 0.0
    d = np.diff(p)
    assert np.abs(d).max() <= 0.25 + 1e-9  # no per-frame step bigger than the ramp's own
    assert (np.sign(d[np.abs(d) > 1e-12]) >= 0).sum() == 4  # rises only on the ramp, then only falls: one opening, no zigzag


def test_attack_ends_before_stop_and_does_not_fit_when_the_ramp_does_not():
    p = RL.attack(60, 20, 2, hold_to=40, stop=27, release=8)
    assert (p[27:] == 0.0).all() and p[22] == 1.0
    assert RL.attack(60, 20, 2, hold_to=40, stop=22, release=8) is None
    assert RL.attack(60, 1, 2, hold_to=40, stop=60, release=8) is None


def test_an_onset_that_rises_as_rendered_is_left_alone_and_a_short_one_is_planned():
    n = 120
    ons = [0.5, 1.5]
    fr = track(n, ons)
    t, _ = mw.onset_frames(fr, n, FPS)
    evs = RL.events(fr, n, FPS)
    m = ramp_reading(n, 25, 0.4) + ramp_reading(n, 75, 0.02) - 0.2  # spread ~0.42: onset 1 rises 0.4, onset 2 only 0.02
    lifts, cnt = RL.plan_lifts(m, t, evs, FPS, RL.spread(m), np.zeros(n, bool))
    assert cnt == dict(onsets=2, adequate=1, short=1, unread=0, no_room=0)
    assert [L["k"] for L in lifts] == [1]
    L = lifts[0]
    assert L["short"] == pytest.approx(RL.TARGET * RL.spread(m) - L["rise"])
    assert (L["p"][: L["c"] - 2 + 1] == 0).all()  # nothing before the ramp, so onset 1 is untouched


def test_the_release_ends_before_the_next_onsets_ramp_and_before_a_kept_shut_frame():
    n = 120
    ons = [0.5, 0.72]
    fr = track(n, ons, hold=0.3)
    t, _ = mw.onset_frames(fr, n, FPS)
    m = np.full(n, 0.2)
    m[60:] = 0.6  # only the second onset rises
    m[:5] = 0.0
    lifts, _ = RL.plan_lifts(m, t, RL.events(fr, n, FPS), FPS, RL.spread(m), np.zeros(n, bool))
    first = [L for L in lifts if L["k"] == 0]
    c2 = RL.best_rise(m, t, 0.72, 2)[0]
    assert first and (first[0]["p"][c2 - 2 :] == 0).all()
    blocked = np.zeros(n, bool)
    blocked[30] = True
    lifts, _ = RL.plan_lifts(m, t, RL.events(fr, n, FPS), FPS, RL.spread(m), blocked)
    assert (lifts[0]["p"][30:] == 0).all()
    blocked[24] = True  # a shut frame inside the ramp: no room
    _, cnt = RL.plan_lifts(m, t, RL.events(fr, n, FPS), FPS, RL.spread(m), blocked)
    assert cnt["no_room"] >= 1


def test_lift_series_takes_the_largest_lift_and_stays_under_the_ceiling():
    base = np.zeros(10)
    p1, p2 = np.zeros(10), np.zeros(10)
    p1[2:6] = 1.0
    p2[4:8] = 0.5
    out = RL.lift_series(base, [dict(k=0, p=p1), dict(k=1, p=p2)], {0: 2.0, 1: 2.0}, np.full(10, 1.5))
    assert out[3] == 1.5 and out[6] == 1.0 and out[0] == 0.0 and out[9] == 0.0
    high = np.full(10, 3.0)  # a frame already past the ceiling keeps its own value
    assert RL.lift_series(high, [dict(k=0, p=p1)], {0: 1.0}, np.full(10, 1.5))[3] == 3.0


class FakeClip:
    """a WarpClip stand-in: a frame's rendered opening = gain x its applied residual (the measure is patched to read it)."""

    def __init__(self, n, ons, dfe, gain=0.8, shut=None):
        self.n, self.fps, self.track_fps = n, FPS, FPS
        self.fr_t = track(n, ons, hold=0.12)
        self.F = np.zeros((n, 2))
        self.F[:, 0] = 2.0
        self.model = types.SimpleNamespace(rng=dict(g_lo=2.0, g_hi=12.0))
        self.dFe = np.zeros((n, 2))
        self.dFe[:, 0] = dfe
        self.Fe = self.F + self.dFe
        self.gain = gain
        self.outs = [None] * n
        self.masks = [None] * n
        self.P_out = [None] * n
        self.log = [dict(i=i) for i in range(n)]
        self.a = types.SimpleNamespace(warp_fold_guard=0.0)
        self.rise_shut = np.zeros(n, bool) if shut is None else shut
        self.renders = []
        for i in range(n):
            self.outs[i] = self.reading(i)

    def reading(self, i):
        return float(self.Fe[i, 0]) * self.gain * 0.01

    def render_frame(self, i, ov):
        self.renders.append(i)
        self.Fe[i] = self.F[i] + ov["dFe"]
        return self.reading(i), None, dict(i=i), None

    def fold_det_of(self, i, v):
        return 1.0


@pytest.fixture
def patched_measure(monkeypatch):
    from lsdub_engine.core import aperture

    def measure(frames, which=None):
        out = np.full(len(frames), np.nan)
        for i in range(len(frames)) if which is None else which:
            out[i] = frames[i]
        return out

    monkeypatch.setattr(aperture, "measure", measure)


def test_close_rise_lifts_only_the_short_onset_and_reaches_the_target(patched_measure):
    n = 150
    d = np.zeros(n)
    d[23:] += 6.0  # onset at 0.5 s: a big rendered rise
    d[60:70] = 0.0
    d[74:] = 0.3  # onset at 1.5 s: a rise of 0.3 canonical px, far under the target
    d[100:] = 0.0
    d[124:] = 5.0  # onset at 2.5 s: big again
    clip = FakeClip(n, [0.5, 1.5, 2.5], d)
    before = clip.dFe[:, 0].copy()
    s = RL.close_rise(clip, log=lambda *_: None)
    assert s["onsets"] == 3 and s["adequate"] == 2 and s["lifted"] == 1 and s["reached"] == 1
    changed = np.flatnonzero(np.abs(clip.dFe[:, 0] - before) > 1e-9)
    assert changed.min() >= 70 and changed.max() < 100  # only the short onset's frames
    assert (clip.dFe[:, 0] >= before - 1e-9).all()  # a lift never lowers a frame
    m = np.array([clip.reading(i) for i in range(n)])
    c = RL.best_rise(m, mw.onset_frames(clip.fr_t, n, FPS)[0], 1.5, 2)
    assert c[1] >= RL.TARGET * RL.spread(m) - 1e-6
    assert all("rise" in clip.log[i] for i in changed)


def test_close_rise_with_every_onset_adequate_renders_nothing(patched_measure):
    n = 100
    d = np.zeros(n)
    d[23:] = 6.0
    d[60:] = 0.0
    d[74:] = 6.0
    clip = FakeClip(n, [0.5, 1.5], d)
    s = RL.close_rise(clip, log=lambda *_: None)
    assert s["adequate"] == 2 and s["lifted"] == 0 and clip.renders == []


def test_flag_parses_and_the_existing_values_are_unchanged():
    ap = mouth_paint._parser()
    base = ["--video", "v.mp4", "--track", "t.json", "--out", "o.mp4"]
    assert ap.parse_args([*base, "--warp-rise", "achieved"]).warp_rise == "achieved"
    assert ap.parse_args(base).warp_rise == "off"
    assert ap.parse_args([*base, "--warp-rise", "lift"]).warp_rise == "lift"


def test_achieved_leaves_the_plan_stage_untouched():
    # the plan-time floor runs only for floor / lift; achieved does its work on the rendered frames
    assert 'g("warp_rise", "off") in ("floor", "lift")' in open(mw.__file__).read()


def test_reads_no_pitch_and_no_judge():
    import ast

    tree = ast.parse(open(RL.__file__).read())
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any("audits" in m or "judge" in m for m in mods)
    names = {n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert not any(w in x for x in names for w in ("pitch", "f0", "melody", "vocal_level"))


def test_a_rise_on_the_line_to_rounding_is_on_it():
    assert RL.need_of(0.3 * 0.048, -1.0, 0.3 * 0.048 + 1e-18) == 0.0
    assert RL.need_of(0.0, -1.0, 0.0144) == pytest.approx(0.0144)


class DeafClip(FakeClip):
    """the pixels never show a lift: a frame reads as its first render whatever residual it is given."""

    def render_frame(self, i, ov):
        self.renders.append(i)
        self.Fe[i] = self.F[i] + ov["dFe"]
        return self.outs[i], None, dict(i=i), None


def test_a_lift_that_does_not_deliver_is_taken_back(patched_measure):
    n = 150
    d = np.zeros(n)
    d[23:] += 6.0
    d[60:70] = 0.0
    d[74:] = 0.3
    d[100:] = 0.0
    d[124:] = 5.0
    clip = DeafClip(n, [0.5, 1.5, 2.5], d)
    before = clip.dFe.copy()
    s = RL.close_rise(clip, log=lambda *_: None)
    assert s["lifted"] == 1 and s["taken_back"] == 1 and s["reached"] == 0
    assert np.array_equal(clip.dFe, before)
    assert not any("rise" in e for e in clip.log)


def test_step_room_keeps_the_ramp_inside_the_clips_own_steps():
    m = np.zeros(100)
    m[10:] = 1.0  # one steep step of 1.0 in the clip
    m[50:] = 1.2  # a gentle one at the onset
    room = RL.step_room(m, 50, 2, q=100.0)
    assert room == pytest.approx(2.0 * (1.0 - 0.2))
    assert RL.step_room(m, 10, 2, q=100.0) == 0.0  # already the steepest step: no room
