# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub sync-truth: the mouth read against the vocal itself (synthetic vocal and mouth, no models), and how the sing gate and verify use it."""

import numpy as np
import pytest

from lsdub_engine.audits import sync_truth as ST
from lsdub_engine.core.vocal_retime import SR

FPS = 25.0


def sung(notes, dur, seed=0, vibrato=0.0):
    """a sung line: harmonic notes (start s, end s, f0) over a faint noise floor; vibrato swells the level of each note (0..1)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    x = 0.002 * rng.standard_normal(len(t))
    for a, b, f0 in notes:
        m = (t >= a) & (t < b)
        tt = t[m] - a
        env = np.minimum(1.0, np.minimum(tt, b - a - tt) / 0.03)  # 30 ms attack and release
        env *= 1.0 - vibrato * 0.5 * (1 + np.sin(2 * np.pi * 5.5 * tt))
        x[m] += 0.3 * env * sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in (1, 2, 3, 4))
    return x


LINE = [(0.28, 0.60, 220), (0.72, 1.00, 247), (1.12, 2.32, 262), (2.48, 2.80, 247), (2.92, 3.40, 220), (3.68, 4.60, 196)]  # on frame edges
DUR = 5.0


def mouth(notes, n, delay_frames=0.0, open_=0.12, shut=0.01):
    """a mouth that opens on each note and shuts between them, eased over two frames, `delay_frames` late (fractional ok)."""
    t = (np.arange(n) + 0.5) / FPS - delay_frames / FPS
    o = np.full(n, shut)
    for a, b, _ in notes:
        o[(t >= a) & (t < b)] = open_
    k = np.array([0.25, 0.5, 0.25])
    return np.convolve(np.pad(o, 1, mode="edge"), k, mode="valid")


N = int(DUR * FPS)
X = sung(LINE, DUR)


def test_a_mouth_on_the_notes_follows_the_song_on_time():
    r = ST.read(mouth(LINE, N), X, FPS)
    assert r["truth"] > 0.6 and abs(r["truth_lag_ms"]) <= 20 and r["truth_pass"] is True


@pytest.mark.parametrize("late", [2, 3, 4])
def test_a_mouth_two_frames_late_or_more_cannot_pass_and_reads_its_lag(late):
    r = ST.read(mouth(LINE, N, late), X, FPS)
    assert r["truth_pass"] is False
    assert abs(r["truth_lag_ms"] - 1000 * late / FPS) <= 25  # + = the mouth trails
    assert "trails" in ST.why(r) or "does not follow" in ST.why(r)


def test_one_frame_early_is_anticipation_and_passes():
    r = ST.read(mouth(LINE, N, -1), X, FPS)
    assert r["truth_pass"] is True and -60 <= r["truth_lag_ms"] <= -20


def test_the_on_time_reading_beats_the_late_one():
    on, late = (ST.read(mouth(LINE, N, d), X, FPS)["truth"] for d in (0, 3))
    assert on > late + 0.2


def test_a_mouth_off_the_song_reads_like_an_unsynced_take():
    other = [(0.0, 0.4, 0), (0.9, 1.3, 0), (1.8, 2.2, 0), (3.0, 3.3, 0), (4.4, 4.9, 0)]
    r = ST.read(mouth(other, N), X, FPS)
    assert r["truth"] < ST.TRUTH_FAIL and r["truth_pass"] is False


def test_a_held_note_that_swells_does_not_count_against_a_mouth_that_holds_it_open():
    """the long note (1.1-2.3 s) swells with vibrato: a raw RMS correlation reads the held-open mouth as off the swells; truth does not."""
    x = sung(LINE, DUR, vibrato=0.8)
    m = mouth(LINE, N)
    r = ST.read(m, x, FPS)
    assert r["truth_pass"] is True and r["truth"] > 0.55
    hop = int(SR / FPS)
    rms = np.array([np.sqrt(np.mean(x[i * hop : (i + 1) * hop] ** 2)) for i in range(N)])
    raw = float(np.corrcoef(m, rms)[0, 1])
    assert r["truth_level"] > raw


def test_frames_left_out_do_not_count():
    m = mouth(LINE, N)
    m[25:60] = 0.12 - m[25:60]  # a vocalisation span the take sings its own way
    keep = np.ones(N, bool)
    keep[25:60] = False
    assert ST.read(m, X, FPS, keep)["truth"] > ST.read(m, X, FPS)["truth"] + 0.1


def test_nothing_to_read_is_none_not_a_fail():
    silent = 0.002 * np.random.default_rng(1).standard_normal(int(DUR * SR))
    assert ST.read(mouth(LINE, N), silent * 0, FPS)["truth"] is None
    assert ST.read(mouth(LINE, 8), X, FPS)["truth"] is None  # too short
    assert ST.read(None, X, FPS)["truth"] is None
    assert ST.passes({"truth": None}) is None and ST.why({}) == ""


def test_a_flat_mouth_reads_zero_and_fails():
    r = ST.read(np.full(N, 0.05), X, FPS)
    assert r["truth"] == 0.0 and r["truth_pass"] is False


def test_per_frame_takes_each_frames_own_interval():
    t = np.arange(0, 1.0, 0.01) + 0.005
    a = (t >= 0.40).astype(float)  # the step lands on frame 10's interval [0.40, 0.44)
    f = ST.per_frame(a, t, 25, FPS)
    assert f[9] == 0.0 and f[10] == 1.0


def test_the_peak_is_sub_frame():
    c = np.array([0.0, 0.2, 0.6, 1.0, 0.8, 0.1, 0.0])  # m = 3; the top leans toward +1
    lag, v = ST.peak(c)
    assert 0.0 < lag < 0.5 and v == 1.0


# -- the sing gate's choice and verify's band -------------------------------------------------------------------------------------
def test_pick_paint_reads_the_audio_sync_when_both_gates_have_it():
    from lsdub_engine.core import sing

    g = lambda c, t: dict(corr=c, truth=t, open_in_silence=0.2, **{"pass": True})
    assert sing.pick_paint(g(0.9, 0.31), g(0.7, 0.45)) == "native"  # the plan corr says sing; the vocal says native
    assert sing.pick_paint(g(0.9, 0.45), dict(g(0.7, None))) == "sing"  # one unread: both by the plan corr


def _row(**k):
    r = dict(
        voiced=True, verdict="DUB", owner_px=500.0, corr=0.9, corr_pm1=0.9, open_in_silence=0.1, rise_ms=40, closure_hit=1.0,
        mouth_range=0.15, mouth_visible=1.0, sharpness=100.0, motion=1.0, pops=0,
    )  # fmt: skip
    r.update(k)
    return r


def test_verify_a_shot_that_follows_the_plan_but_not_the_song_cannot_pass():
    from lsdub_engine.audits import qa as QA, scorecard as SC

    good = QA.score_row(_row(truth=0.5, truth_lag_ms=0.0), 100.0, SC)
    late = QA.score_row(_row(truth=0.5, truth_lag_ms=120.0), 100.0, SC)
    off = QA.score_row(_row(truth=0.05, truth_lag_ms=0.0), 100.0, SC)
    weak = QA.score_row(_row(truth=0.25, truth_lag_ms=0.0), 100.0, SC)
    assert good["band"] == "PASS"
    assert late["band"] == "REVIEW" and "trails the vocal by 120 ms" in late["reason"]
    assert weak["band"] in ("REVIEW", "FAIL") and weak["score"] < QA.PASS_BAND
    assert off["band"] == "FAIL"
    assert QA.score_row(_row(), 100.0, SC)["band"] == "PASS"  # no truth reading: as before


def test_a_dub_that_keeps_its_takes_own_small_travel_is_not_a_held_gape():
    from lsdub_engine.audits import scorecard as SC

    assert SC.held_gape(dict(mouth_range=0.08))  # no take to read: the absolute line
    assert not SC.held_gape(dict(mouth_range=0.08, before=dict(mouth_range=0.09)))  # the take itself moves 0.09: kept
    assert SC.held_gape(dict(mouth_range=0.05, before=dict(mouth_range=0.12)))  # the dub flattened the take: held
    assert not SC.held_gape(dict(mouth_range=0.12, before=dict(mouth_range=0.5)))  # over the line: never


def test_a_dub_that_delivers_the_travel_its_plan_asked_is_not_a_held_gape():
    from lsdub_engine.audits import scorecard as SC

    take = dict(mouth_range=0.12)
    assert SC.held_gape(dict(mouth_range=0.07, before=take))  # 58% of the take, no plan read: held
    assert not SC.held_gape(dict(mouth_range=0.07, before=take, asked_travel=0.6))  # the plan asked 60% of the take: delivered
    assert SC.held_gape(dict(mouth_range=0.07, before=take, asked_travel=0.9))  # asked 90%, delivered 58%: held
    assert not SC.held_gape(dict(mouth_range=0.095, before=take, asked_travel=1.4))  # keeps 79% of its take: an ask past it adds nothing
    assert SC.held_gape(dict(mouth_range=0.04, before=take, asked_travel=0.2))  # under the floor: held whatever was asked


def test_verify_reads_the_asked_travel_from_the_warp_log_on_the_delivered_frames(tmp_path):
    import json

    from lsdub_engine.audits import qa as QA

    n = 40
    gap = 10 + 10 * np.sin(np.arange(n) / 2.0)  # the take travels 20 px
    tgap = 10 + 5 * np.sin(np.arange(n) / 2.0)  # the plan asks half of it
    tgap[1::2] = tgap[0::2]  # the odd paint frames are never delivered
    wj = tmp_path / "crop_painted_warp.json"
    wj.write_text(json.dumps(dict(frames=[dict(gap=float(g), tgap=float(t)) for g, t in zip(gap, tgap)])))
    (tmp_path / "medium.json").write_text(json.dumps(dict(frames=n // 2, fps_mult=2, picked=list(range(0, n, 2)))))
    got = QA.asked_travel(dict(warp_json=str(wj)))["asked_travel"]
    g, t = gap[0::2], tgap[0::2]
    assert got == pytest.approx((np.percentile(t, 95) - np.percentile(t, 5)) / (np.percentile(g, 95) - np.percentile(g, 5)), abs=1e-3)
    assert QA.asked_travel(dict(warp_json=str(wj), used="source")) == {}  # a shot left as source asks nothing
    assert QA.asked_travel(dict(warp_json=str(tmp_path / "missing.json"))) == {}  # no log: the take stays the reference
    (tmp_path / "empty.json").write_text(json.dumps(dict(frames=[])))
    assert QA.asked_travel(dict(warp_json=str(tmp_path / "empty.json"))) == {}  # a log with no frames: the same


def test_asked_travel_reads_a_1x_log_on_every_frame_not_the_sing_paints_picks(tmp_path):
    import json

    from lsdub_engine.audits import qa as QA

    n = 30  # a native render's log: one entry per delivered frame
    gap = 10 + 10 * np.sin(np.arange(n) / 2.0)
    tgap = gap.copy()
    tgap[: n // 2] = 10.0  # the first half is asked to hold: the sing paint's 2x picks would read mostly that half
    d = tmp_path / "native"
    d.mkdir()
    wj = d / "native_warp.json"
    wj.write_text(json.dumps(dict(frames=[dict(gap=float(g), tgap=float(t)) for g, t in zip(gap, tgap)])))
    (d / "medium.json").write_text(json.dumps(dict(frames=n, fps_mult=2, picked=list(range(0, 2 * n, 2)))))
    rng = lambda a: np.percentile(a, 95) - np.percentile(a, 5)
    assert QA.asked_travel(dict(warp_json=str(wj)))["asked_travel"] == pytest.approx(rng(tgap) / rng(gap), abs=1e-3)
    (d / "medium.json").write_text(json.dumps(dict(frames=n // 2, fps_mult=2, picked=[0, 2, 4, 6, 99])))  # a pick past the log
    assert QA.asked_travel(dict(warp_json=str(wj)))["asked_travel"] == pytest.approx(rng(tgap) / rng(gap), abs=1e-3)


@pytest.mark.parametrize("medium", [dict(picked=[float("inf")]), [1, 2, 3]])
def test_asked_travel_leaves_a_malformed_picks_file_to_the_take(tmp_path, medium):
    import json

    from lsdub_engine.audits import qa as QA

    wj = tmp_path / "w.json"
    wj.write_text(json.dumps(dict(frames=[dict(gap=float(g), tgap=float(g)) for g in range(10)])))
    (tmp_path / "medium.json").write_text(json.dumps(medium))
    assert QA.asked_travel(dict(warp_json=str(wj))) == {}


def test_asked_travel_gives_nothing_for_a_reading_that_is_not_a_number(tmp_path):
    import json

    from lsdub_engine.audits import qa as QA
    from lsdub_engine.audits import scorecard as SC

    wj = tmp_path / "w.json"
    wj.write_text(json.dumps(dict(frames=[dict(gap=float(g), tgap=float("nan")) for g in range(10)])))
    assert QA.asked_travel(dict(warp_json=str(wj))) == {}
    take = dict(mouth_range=0.12)
    for ask in (float("nan"), 0.0, -0.5, None):  # not a finite ask above 0: the take is the reference
        assert SC.held_gape(dict(mouth_range=0.07, before=take, asked_travel=ask))
    assert not SC.held_gape(dict(mouth_range=0.07, before=take, asked_travel=0.001))  # a tiny ask: only the floor holds
    assert SC.held_gape(dict(mouth_range=0.045, before=take, asked_travel=0.001))


def test_a_tie_between_peaks_takes_the_one_nearest_lag_0():
    c = np.array([-0.26, 0.03, 0.33, 0.51, 0.37, 0.17, 0.18, 0.33, 0.51, 0.47, 0.21, -0.05, -0.11, -0.01, 0.15, 0.18, 0.08])  # m = 8
    lag, v = ST.peak(c)
    assert -0.5 <= lag <= 0.5 and v == 0.51  # a repeat five frames early is as high: the lag is the one at 0
    c2 = c.copy()
    c2[3] = 0.60  # clearly higher: it is the lag
    assert ST.peak(c2)[0] < -4
