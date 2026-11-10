# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""0.3.0: the vocal-driven mouth is sing's default, `classic` gives the 0.2.x mouth back on each switch, and Verify's sync band reads
the event judge (qa.event_band) with the envelope reading kept only as the floor."""

import pytest

from lsdub_engine.audits import qa, scorecard as SC
from lsdub_engine.core import mouth_paint as MP, sing as S

NEW = ["--warp-events", "vocal", "--warp-hold", "nucleus-follow", "--warp-closures", "aligned-press", "--warp-aperture", "p95",
       "--warp-realise", "events"]  # fmt: skip
CLASSIC = [x for f in S.STAGES for x in (f, "classic")]


def test_snap_carries_the_chain():
    p = S.SNAP["paint"]
    for f, v in zip(NEW[::2], NEW[1::2]):
        assert p[p.index(f) + 1] == v


def test_a_callers_value_replaces_snaps_and_classic_reads_as_02x():
    snap = S.snap_paint(CLASSIC, 2)
    assert not any(f in snap for f in S.STAGES)
    for f in S.STAGES:
        assert S.stage_value(CLASSIC, snap, f) == S.CLASSIC[f]
    assert S.warp_audio(CLASSIC, snap, "v.wav") == ()
    one = S.snap_paint(["--warp-hold", "nucleus"], 2)
    assert "--warp-hold" not in one and S.stage_value(["--warp-hold", "nucleus"], one, "--warp-hold") == "nucleus"
    assert S.warp_audio([], S.snap_paint([], 2), "v.wav") == ("--warp-audio", "v.wav")


def test_coart_and_own_targets_get_no_stage_flags():
    for t in ("coart", "own"):
        snap = S.snap_paint(["--warp-target", t], 2)
        assert not any(f in snap for f in S.STAGES) and "--warp-shape" in snap


def test_paint_reads_classic_as_the_old_value():
    a = MP._parser().parse_args(["--video", "v", "--track", "t", "--out", "o", *CLASSIC])
    assert (a.warp_events, a.warp_hold, a.warp_closures, a.warp_aperture, a.warp_realise) == ("off", "off", "plan", "off", "off")


def _row(**ev):
    base = dict(
        shot=0, start=0.0, end=2.0, verdict="DUB", voiced=True, owner_px=500, corr=0.6, corr_pm1=0.6, open_in_silence=0.0,
        mouth_visible=1.0, mouth_range=0.5, motion=1.0, pops=0, sharpness=100.0, truth=0.25, truth_lag_ms=0.0, truth_pass=False,
    )  # fmt: skip
    return dict(base, **ev)


GOOD = dict(ev_read="phones", ev_onset_hit=0.8, ev_onset_n=10, ev_chance=0.3, ev_closure_hit=1.0, ev_closure_n=2, ev_spurious=0,
            ev_spurious_take=1)  # fmt: skip


def test_the_odds_are_the_binomial_tail():
    assert qa.onset_odds(1.0, 5, 0.32) == pytest.approx(0.32**5, abs=1e-5)
    assert qa.onset_odds(0.0, 5, 0.3) == pytest.approx(1.0) and qa.onset_odds(None, 5, 0.3) is None


@pytest.mark.parametrize(
    "change, ok",
    [
        ({}, True),
        (dict(ev_onset_n=3, ev_onset_hit=1.0), False),  # too few onsets to read
        (dict(ev_onset_hit=0.4), False),  # 4 of 10 at chance 0.3: odds 0.35
        (dict(ev_closure_hit=0.0, ev_closure_n=1), False),
        (dict(ev_closure_hit=0.5, ev_closure_n=2), True),
        (dict(ev_closure_hit=0.33, ev_closure_n=3), False),
        (dict(ev_spurious=2, ev_spurious_take=1), False),
        (dict(ev_read="flux only: no weights"), None),
    ],
)
def test_event_band_lines(change, ok):
    assert qa.event_band(_row(**{**GOOD, **change}))[0] is ok


def test_the_event_band_decides_and_the_envelope_is_the_floor():
    a = qa.score_row(_row(**GOOD), 100.0, SC)  # old truth 0.25 would cap it under PASS in 0.2.x
    assert a["band"] == "PASS" and a["ev_band"] is True
    b = qa.score_row(_row(**{**GOOD, "ev_spurious": 3}), 100.0, SC)
    assert b["band"] == "REVIEW" and "shuts where the singer does not" in b["reason"]
    c = qa.score_row(_row(**{**GOOD, "truth": 0.05}), 100.0, SC)
    assert c["band"] == "FAIL" and "does not follow the vocal" in c["reason"]
    d = qa.score_row(_row(), 100.0, SC)  # no phones reading: the 0.2.x envelope band
    assert d["band"] == "REVIEW" and "ev_band" not in d


def test_classic_on_every_switch_is_the_022_paint_exactly():
    """sing's paint flags with every stage on classic parse to the same paint namespace as 0.2.2's SNAP (the flags as 0.2.2 sent them)."""
    snap_022 = [
        "--warp-sigma",
        "0.8",
        "--warp-temporal",
        "0.6",
        "--warp-close-depth",
        "0.05",
        "--warp-follow",
        "0.15",
        "--warp-shape",
        "asym",
    ]
    base = ["--video", "v", "--track", "t", "--out", "o", "--lib", "K", "--warp-track-mode", "chain", "--vocalisations", "skip"]
    new = vars(MP._parser().parse_args([*base, *CLASSIC, *S.snap_paint(CLASSIC, 2)]))
    old = vars(MP._parser().parse_args([*base, *snap_022]))
    assert new == old
    assert S.warp_audio(CLASSIC, S.snap_paint(CLASSIC, 2), "v.wav") == ()


def test_the_auto_pick_reads_the_event_band_with_the_envelope_as_floor():
    g = dict(truth=0.5, truth_lag_ms=0.0, corr=0.5, best=0.5, open_in_silence=0.1, pass_=True)
    s = dict(g, **{"pass": False, "truth": 0.25, "ev_band": True})  # the new mouth: inside the event band, under the old gate
    n = dict(g, **{"pass": True, "ev_band": False})
    assert S.pick_paint(s, n) == "sing"
    assert S.pick_paint(dict(s, truth=0.1), n) == "native"  # under the envelope floor: the event band cannot carry it
    assert S.pick_paint(dict(s, ev_band=None), dict(n, ev_band=None)) == "native"  # no phones reading: the 0.2.x rule
    assert S.pick_paint(dict(s, ev_band=False), n) == "native"  # neither inside: the 0.2.x rule
