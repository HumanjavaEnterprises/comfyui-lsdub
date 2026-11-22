# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""--warp-rest closed: inside the event stage's rests the hold eases the mouth down to the take's own most-closed and back, and every
syllable's release, attack and hold stay the 0.3.x curve (default `apart`: unchanged, info unchanged)."""

import json
import os
import types

import numpy as np
import pytest

from lsdub_engine.core import mouth_paint as MP
from lsdub_engine.core import mouth_warp as M
from lsdub_engine.core import nucleus_hold as NH

FPS = 50.0
# a phrase, a long rest (1.0 s), a phrase, a short rest (0.20 s), a last syllable, then a rest to the clip's end
EV = [
    dict(kind="nucleus", t_on=0.30, vowel="AA", bin=1, cons=True, src="nuc"),
    dict(kind="nucleus", t_on=0.60, vowel="IY", bin=1, cons=True, src="nuc"),
    dict(kind="rest", t_on=0.90, t_off=1.90),
    dict(kind="nucleus", t_on=2.10, vowel="EH", bin=1, cons=True, src="nuc"),
    dict(kind="rest", t_on=2.40, t_off=2.60),
    dict(kind="nucleus", t_on=2.76, vowel="AH", bin=1, cons=True, src="nuc"),
    dict(kind="rest", t_on=3.10, t_off=3.60),
]
N = int(3.6 * FPS)


def _frames(ev=EV, n=N, lead_rest=False):
    starts = sorted((e["t_on"], k) for k, e in enumerate(ev))
    out = []
    for i in range(n):
        t = i / FPS
        k = max((kk for a, kk in starts if a <= t), default=-1)
        tag = dict(k=-1, kind="rest", t_on=None) if k < 0 else dict(ev[k], k=k)
        out.append(dict(t=t, vis="aa", vis2="aa", mix=0.0, open=0.3, ev=tag))
    return out


def _o(frames):
    return np.array([f["open"] for f in frames])


def _i(t):
    return int(round(t * FPS))


def test_apart_is_the_default_and_the_030_curve_and_info():
    a, ia = NH.held(_frames(), FPS)
    b, ib = NH.held(_frames(), FPS, rest="apart")
    assert a == b and ia == ib
    assert "rest" not in ia and "rest_close_s" not in ia  # the warp json of a default paint carries nothing new
    assert NH.RESTS == ("apart", "closed")
    with pytest.raises(ValueError):
        NH.held(_frames(), FPS, rest="shut")


def test_a_long_rest_reaches_closed_and_eases_both_ways():
    ap = _o(NH.held(_frames(), FPS)[0])
    cl, info = NH.held(_frames(), FPS, rest="closed")
    cl = _o(cl)
    assert info["rest"] == "closed" and info["rests"] == 3
    mid = slice(_i(0.90 + NH.DECAY_S + NH.REST_CLOSE_S + 0.02), _i(1.90 - NH.REST_REOPEN_S - 0.02))
    assert np.all(ap[mid] == pytest.approx(NH.REST_OPEN, abs=1e-3))  # apart: lips apart through the rest
    assert np.all(cl[mid] == 0.0)  # closed: the take's own most-closed
    # the way down from the vowel is one smooth fall (no step back up, no frame steeper than the 0.3.x decay's own steepest step)
    down = cl[_i(0.90) : _i(0.90 + NH.DECAY_S + NH.REST_CLOSE_S) + 1]
    assert np.all(np.diff(down) <= 1e-9)
    steep_ap = np.max(-np.diff(ap[_i(0.90) : _i(0.90 + NH.DECAY_S) + 1]))
    assert np.max(-np.diff(down)) <= steep_ap + 1e-3
    assert np.array_equal(cl[_i(0.90) : _i(0.90 + NH.DECAY_S) + 1], ap[_i(0.90) : _i(0.90 + NH.DECAY_S) + 1])  # the decay itself is 0.3.x
    # the way back up ends at REST_OPEN by the next syllable's release, rising smoothly
    e = 2.10 - NH.ATTACK_S - NH.RELEASE_S
    up = cl[_i(e - NH.REST_REOPEN_S) : _i(e) + 1]
    assert np.all(np.diff(up) >= -1e-9) and up[-1] == pytest.approx(NH.REST_OPEN, abs=1e-3)
    assert np.max(np.diff(up)) <= NH.REST_OPEN * np.pi / 2 / (NH.REST_REOPEN_S * FPS) + 1e-3  # the raised cosine's own top slope


def test_every_syllable_is_the_030_curve_outside_the_rests():
    ap = _o(NH.held(_frames(), FPS)[0])
    cl = _o(NH.held(_frames(), FPS, rest="closed")[0])
    t = np.arange(N) / FPS
    inside = np.zeros(N, bool)
    for a, b, nxt in ((0.90, 1.90, 2.10), (2.40, 2.60, 2.76)):
        e = min(b, nxt - NH.ATTACK_S - NH.RELEASE_S)
        inside |= (t >= a - 1e-9) & (t <= e + 1e-9)
    inside |= t >= 3.10 - 1e-9
    assert np.array_equal(cl[~inside], ap[~inside])  # releases, attacks, apexes and holds: bit for bit
    for on in (2.10, 2.76):  # the onset windows (release + attack + apex) are untouched
        w = slice(_i(on - NH.ATTACK_S - NH.RELEASE_S), _i(on) + 1)
        assert np.array_equal(cl[w], ap[w])
    assert np.all(cl <= ap + 1e-12)  # it only ever closes


def test_a_short_rest_closes_only_part_way_and_never_steps():
    ap = _o(NH.held(_frames(), FPS)[0])
    cl = _o(NH.held(_frames(), FPS, rest="closed")[0])
    w = slice(_i(2.40), _i(2.76 - NH.ATTACK_S - NH.RELEASE_S) + 1)
    assert 0.0 < cl[w].min() < NH.REST_OPEN  # part way: the two ramps multiply
    assert np.max(np.abs(np.diff(cl[w]))) <= np.max(np.abs(np.diff(ap[w]))) + 0.03


def test_a_rest_at_the_end_stays_closed_and_one_at_the_start_is_closed_from_it():
    cl = _o(NH.held(_frames(), FPS, rest="closed")[0])
    assert np.all(cl[_i(3.10 + NH.DECAY_S + NH.REST_CLOSE_S + 0.02) :] == 0.0)  # no reopening after the last syllable
    ev = [
        dict(kind="rest", t_on=0.0, t_off=0.60),
        *[dict(e, t_on=e["t_on"] + 0.6, **({"t_off": e["t_off"] + 0.6} if "t_off" in e else {})) for e in EV[:3]],
    ]
    fr = _frames(ev, n=int(2.0 * FPS))
    a = _o(NH.held(fr, FPS)[0])
    c = _o(NH.held(fr, FPS, rest="closed")[0])
    assert c[0] == 0.0 and a[0] == pytest.approx(NH.REST_OPEN, abs=1e-3)
    e = 0.90 - NH.ATTACK_S - NH.RELEASE_S
    assert c[_i(e)] == pytest.approx(NH.REST_OPEN, abs=1e-3)  # back at the rest opening when the first phrase's release starts


def test_no_rest_event_no_change():
    """a soft held vowel, a consonant or a quiet tail is voice to the event stage: no rest event, so the hold is untouched."""
    ev = [e for e in EV if e["kind"] != "rest"]
    a = NH.held(_frames(ev), FPS)[0]
    c = NH.held(_frames(ev), FPS, rest="closed")[0]
    assert a == c


def test_closures_keep_their_frames():
    fr = _frames()
    fr[_i(1.2)] = dict(fr[_i(1.2)], vis="PP", open=0.0)
    c = NH.held(fr, FPS, rest="closed")[0]
    assert c[_i(1.2)] == fr[_i(1.2)]


def test_follow_mode_then_rest():
    lv = np.full(int(3.6 / 0.01), -6.0)
    a = _o(NH.held(_frames(), FPS, mode="nucleus-follow", level=lv)[0])
    c = _o(NH.held(_frames(), FPS, mode="nucleus-follow", level=lv, rest="closed")[0])
    assert np.all(c <= a + 1e-12) and c[_i(1.4)] == 0.0 and a[_i(1.4)] == pytest.approx(NH.REST_OPEN, abs=1e-3)


def test_flag_parses_default_apart():
    ap = MP._parser()
    base = ["--video", "v.mp4", "--track", "t.json", "--out", "o.mp4"]
    assert ap.parse_args(base).warp_rest == "apart"
    assert ap.parse_args([*base, "--warp-rest", "closed"]).warp_rest == "closed"
    with pytest.raises(SystemExit):
        ap.parse_args([*base, "--warp-rest", "shut"])


@pytest.mark.parametrize("rest", ["apart", "closed"])
def test_the_warp_hands_the_flag_to_the_hold(rest):
    stub = types.SimpleNamespace(drive_info={}, vocal_level=None)
    a = types.SimpleNamespace(
        warp_events="off", warp_closures="plan", warp_hold="nucleus", warp_lead_gate="off", warp_shape="off", warp_rest=rest
    )
    out = M.WarpClip.drive_stages(stub, a, _frames(), None, None, FPS)
    want = NH.held(_frames(), FPS, rest=rest)[0]
    assert out == want
    assert ("rest" in stub.drive_info["hold"]) == (rest == "closed")
    a0 = types.SimpleNamespace(warp_events="off", warp_closures="plan", warp_hold="nucleus", warp_lead_gate="off", warp_shape="off")
    assert (
        M.WarpClip.drive_stages(types.SimpleNamespace(drive_info={}, vocal_level=None), a0, _frames(), None, None, FPS)
        == NH.held(_frames(), FPS)[0]
    )


# the 26edb7a (0.3.x, before --warp-rest) held() on _frames(), frozen: opens and info per mode/onset (the follow level is below)
OLD = os.path.join(os.path.dirname(__file__), "data", "warp_rest_26edb7a_held.json")


@pytest.mark.parametrize("mode", ["nucleus", "nucleus-follow"])
@pytest.mark.parametrize("onset", ["apex", "rise"])
def test_apart_equals_the_frozen_26edb7a_curve(mode, onset):
    with open(OLD) as fh:
        want = json.load(fh)[f"{mode}/{onset}"]
    lv = -6.0 - 9.0 * np.abs(np.sin(np.arange(360) * 0.07)) if mode == "nucleus-follow" else None
    for kw in ({}, {"rest": "apart"}):
        fr, info = NH.held(_frames(), FPS, mode=mode, level=lv, onset=onset, **kw)
        assert [f["open"] for f in fr] == want["open"]
        assert json.loads(json.dumps(info)) == want["info"]


def _tail_case(b, n, fps=FPS):
    """a syllable, then a rest from 0.6 s to b with no nucleus after it, in a clip of n frames."""
    ev = [dict(kind="nucleus", t_on=0.30, vowel="AA", bin=1, cons=True, src="nuc"), dict(kind="rest", t_on=0.60, t_off=b)]
    starts = [(e["t_on"], k) for k, e in enumerate(ev)]
    fr = []
    for i in range(n):
        t = i / fps
        k = max((kk for a, kk in starts if a <= t), default=-1)
        tag = dict(k=-1, kind="rest", t_on=None) if k < 0 else dict(ev[k], k=k)
        fr.append(dict(t=t, vis="aa", vis2="aa", mix=0.0, open=0.3, ev=tag))
    return fr


@pytest.mark.parametrize("fps", [25.0, 50.0])
@pytest.mark.parametrize(
    "b,clip_s", [(1.50, 2.0), (4.13, None)]
)  # mid-clip; and the clip's last frame just after it (jules-rooftop-pop_mid)
def test_a_rest_ending_inside_the_clip_with_nothing_after_eases_back_no_step(fps, b, clip_s):
    n = int(round(clip_s * fps)) if clip_s else int(np.floor(b * fps + 1e-9)) + 2  # None: one frame past the rest's end, the last
    fr = _tail_case(b, n, fps)
    ap = np.array([f["open"] for f in NH.held(fr, fps)[0]])
    cl = np.array([f["open"] for f in NH.held(fr, fps, rest="closed")[0]])
    t = np.arange(n) / fps
    assert cl[(t > 0.60 + NH.DECAY_S + NH.REST_CLOSE_S + 0.02) & (t < b - NH.REST_REOPEN_S - 0.02)].max() == 0.0  # it does close
    assert np.all(cl <= ap + 1e-12)
    up_bound = NH.REST_OPEN * np.pi / 2 / (NH.REST_REOPEN_S * fps) + 1e-3  # the ease-out's own steepest step
    tail = cl[t >= b - NH.REST_REOPEN_S - 1e-9]
    assert np.all(np.diff(tail) >= -1e-9) and np.max(np.diff(tail)) <= up_bound  # one smooth rise, never a 0 -> REST_OPEN step
    assert np.max(np.diff(cl[t >= 0.60 - 1e-9])) <= up_bound
    assert np.array_equal(cl[t > b + 1e-9], ap[t > b + 1e-9])  # back on the 0.3.x curve after the rest
    assert cl[t <= b + 1e-9][-1] >= NH.REST_OPEN - up_bound  # all but done on the rest's last frame (the frame grid need not hit b)


def test_a_rest_running_to_the_last_frame_still_stays_closed():
    fr = _tail_case(2.0, int(2.0 * FPS))
    cl = np.array([f["open"] for f in NH.held(fr, FPS, rest="closed")[0]])
    assert np.all(cl[_i(0.60 + NH.DECAY_S + NH.REST_CLOSE_S + 0.02) :] == 0.0)


def test_closed_with_onset_rise_leaves_every_onset_as_apart():
    ap = _o(NH.held(_frames(), FPS, onset="rise")[0])
    cl = _o(NH.held(_frames(), FPS, onset="rise", rest="closed")[0])
    assert np.all(cl <= ap + 1e-12) and cl[_i(1.4)] == 0.0 and ap[_i(1.4)] == pytest.approx(NH.REST_OPEN, abs=1e-3)
    t = np.arange(N) / FPS
    inside = np.zeros(N, bool)
    for a, b in ((0.90, 1.90), (2.40, 2.60), (3.10, 3.60)):
        inside |= (t >= a - 1e-9) & (t <= b + 1e-9)
    assert np.array_equal(cl[~inside], ap[~inside])  # outside the rests: bit for bit
    for on in (0.30, 0.60, 2.10, 2.76):  # release + attack + apex, the apex RISE_LAG_S after the onset
        w = (t >= on + NH.RISE_LAG_S - NH.ATTACK_S - NH.RELEASE_S - 1e-9) & (t <= on + NH.RISE_LAG_S + 1e-9)
        assert w.any() and np.array_equal(cl[w], ap[w])
