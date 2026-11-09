# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Verify's event reading (0.2.1): the event judge's onset / closure / stray-shut / vowel-opening reading and the per-onset lag, on each
dubbed shot, reported and moving no band. The review panel's EVENTS line: test_review_events.py (review.py does not ship)."""

import json

import numpy as np
import pytest

from lsdub_engine.audits import event_judge as EJ, judge_mouth as JM, qa, scorecard as SC, sync_qa as Q


def test_onset_lags_take_the_nearest_rise_inside_the_window_signed_late_positive():
    assert qa.onset_lags([1.0, 2.0, 3.0], [1.04, 1.5, 1.98, 3.5]) == pytest.approx([40.0, -20.0])
    assert qa.onset_lags([1.0], [1.19]) == pytest.approx([190.0]) and qa.onset_lags([1.0], [1.21]) == []
    assert qa.onset_lags([], [1.0]) == []


def test_no_vocal_is_unread_and_never_loads_the_recogniser(monkeypatch):
    monkeypatch.setattr(qa, "event_reading", None, raising=False)
    import lsdub_engine.audits.judge_phones as JP

    monkeypatch.setattr(JP, "Phones", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")))
    assert qa.vocal_events("/nonexistent.wav") == (None, "unread: no vocal")


def test_flux_only_without_local_weights_says_so(tmp_path, monkeypatch):
    import wave

    import lsdub_engine.audits.judge_phones as JP

    sr = 16000
    t = np.arange(sr * 2) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * ((t > 0.4) & (t < 1.2))).astype(np.float32)
    with wave.open(str(tmp_path / "v.wav"), "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(sr), f.writeframes((x * 32767).astype(np.int16).tobytes())
    monkeypatch.setattr(JP, "model_source", lambda d=None: JP.MODEL)  # no local snapshot
    monkeypatch.setattr(JP, "Phones", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a download")))
    ev, read = qa.vocal_events(str(tmp_path / "v.wav"))
    assert read.startswith("flux only") and set(ev) >= {"onsets", "closures", "wide", "sung"} and ev["closures"] == []


def _shot():
    return dict(shot=0, f0=0, f1=49, start=0.0, end=2.0, verdict="DUB", box=[10, 10, 50, 50])


def test_event_reading_scores_the_shots_window_and_reports_the_lag(monkeypatch):
    fps, n = 25.0, 50
    t = (np.arange(n) + 0.5) / fps
    op = np.where(((t > 0.52) & (t < 0.8)) | ((t > 1.24) & (t < 1.6)), 0.12, 0.01)  # rises at 0.54 and 1.26: 40 ms after each onset
    monkeypatch.setattr(Q, "read_span", lambda v, a, b: [np.zeros((64, 64, 3), np.uint8)] * (b - a))
    monkeypatch.setattr(JM, "roi_of", lambda F, boxes=None: [0, 64, 0, 64])
    monkeypatch.setattr(JM, "opening", lambda F, roi, fps=25.0: (op[: len(F)], op[: len(F)], 1.0))
    ev = dict(onsets=[0.5, 1.22], detected=[0.5, 1.22], flux=[], closures=[], wide=[], sung=np.ones(200, bool), words=[])
    r = qa.event_reading("v.mp4", _shot(), fps, 0.0, ev, "flux only: test")
    assert r["ev_onset_n"] == 2 and r["ev_onset_hit"] == 1.0 and r["ev_read"] == "flux only: test"
    assert r["ev_onset_lag_ms"] == pytest.approx(40.0) and r["ev_lag_n"] == 2
    assert "mouth rises a median" in r["ev_why"] and set(r) >= {f"ev_{k}" for k in qa.EV_KEYS}
    assert qa.event_reading("v.mp4", _shot(), fps, 0.0, None, "unread: no vocal") == {"ev_read": "unread: no vocal"}


def test_the_event_reading_moves_no_band():
    base = dict(
        shot=0, start=0.0, end=2.0, verdict="DUB", voiced=True, owner_px=500, corr=0.6, corr_pm1=0.6, open_in_silence=0.0,
        mouth_visible=1.0, mouth_range=0.08, motion=1.0, pops=0, sharpness=100.0, truth=0.7, truth_lag_ms=0.0, truth_pass=True,
    )  # fmt: skip
    a, b = dict(base), dict(base, ev_onset_hit=0.1, ev_onset_n=9, ev_closure_hit=0.0, ev_closure_n=3, ev_spurious=9, ev_spurious_take=0,
                            ev_open_ratio=0.2, ev_onset_lag_ms=150.0, ev_onset_lag_p90_ms=300.0, ev_pass=False)  # fmt: skip
    qa.score_row(a, 100.0, SC), qa.score_row(b, 100.0, SC)
    assert (a["score"], a["band"], a["reason"]) == (b["score"], b["band"], b["reason"])


def test_write_verify_carries_the_columns_the_cell_and_how_it_was_read(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "sheet", lambda *a: None)
    row = dict(shot=0, start=0.0, end=2.0, band="PASS", score=90.0, **{"class": "PERF"}, mouth_visible=1.0, reason="",
               ev_onset_hit=0.6, ev_onset_n=5, ev_closure_hit=0.0, ev_closure_n=1, ev_spurious=0, ev_spurious_take=0, ev_open_ratio=0.72,
               ev_onset_lag_ms=-19.0, ev_onset_lag_p90_ms=101.0, ev_read="phones", ev_why="vowels under-open")  # fmt: skip
    out = str(tmp_path / "v")
    qa.write_verify({"video": "v", "events_read": "phones"}, [row], out)
    md = open(out + ".md").read()
    assert "| events |" in md and "onsets 0.60 of 5, closures 0.00 of 1, stray shuts 0 (take 0), vowels 0.72 of the take's" in md
    assert "lag -19 / p90 +101 ms: vowels under-open" in md and "this film: phones" in md
    head = open(out + ".csv").read().splitlines()[0].split(",")
    assert {
        "ev_onset_hit",
        "ev_closure_hit",
        "ev_spurious",
        "ev_spurious_take",
        "ev_open_ratio",
        "ev_onset_lag_ms",
        "ev_onset_lag_p90_ms",
    } <= set(head)
    assert json.load(open(out + ".json"))["shots"][0]["ev_open_ratio"] == 0.72


def test_the_judge_is_called_not_copied():
    src = open(qa.__file__).read()
    assert "EJ.score(" in src and "EJ.cut(" in src and "EJ.events(" in src and "EJ.why(" in src
    assert EJ.TOL == 0.06  # the reading's tolerance is the judge's own
