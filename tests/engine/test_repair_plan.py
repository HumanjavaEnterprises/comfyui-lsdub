# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""0.4 step 1: the continuity repair reads the paint's plan (sing.REPAIR_PLAN, default warp since the 0.4 ledger; classic always off)."""

import json

import numpy as np
import pytest

from lsdub_engine.core import continuity as CONT
from lsdub_engine.core import sing as S


def _warp(path, n, gap=0.1, egap=0.3):
    json.dump(dict(frames=[dict(gap=gap + 0.001 * i, egap=egap + 0.001 * i) for i in range(n)]), open(path, "w"))
    return str(path)


def test_plan_of_reads_the_picked_frames(tmp_path):
    wj = _warp(tmp_path / "w.json", 20)
    g, e = CONT.plan_of(wj, picked=[0, 2, 4, 99])
    assert np.allclose(g, [0.1, 0.102, 0.104, 0.119]) and np.allclose(e, [0.3, 0.302, 0.304, 0.319])  # (clamped at the end)
    g, e = CONT.plan_of(wj, n=5)
    assert len(g) == len(e) == 5
    assert CONT.plan_of(str(tmp_path / "none.json")) is None
    json.dump(dict(frames=[dict(gap=0.1)]), open(tmp_path / "bad.json", "w"))
    assert CONT.plan_of(str(tmp_path / "bad.json")) is None  # no planned gap: not a plan


def test_the_mode_is_off_for_the_classic_mouth(monkeypatch):
    assert S.REPAIR_PLAN in S.REPAIR_PLANS
    monkeypatch.setattr(S, "REPAIR_PLAN", "warp")
    assert S.repair_plan([], True) == "warp"
    assert S.repair_plan([], False) == "off"
    assert S.repair_plan(["--warp-events", "classic"], True) == "off"
    assert S.repair_plan(["--warp-events", "off"], True) == "off"
    monkeypatch.setattr(S, "REPAIR_PLAN", "off")
    assert S.repair_plan([], True) == "off"


def test_delivered_plan_follows_the_winner(tmp_path):
    wj = _warp(tmp_path / "warp.json", 20)
    _warp(tmp_path / "native_warp.json", 10, gap=0.5, egap=0.7)
    info = dict(warp_json=wj, frames=10, fps_mult=2, medium=dict(picked=list(range(0, 20, 2))))
    g, e = S.delivered_plan(info, str(tmp_path))
    assert len(g) == 10 and np.isclose(g[1], 0.102)
    g, e = S.delivered_plan(dict(info, auto=dict(winner="native")), str(tmp_path))
    assert len(g) == 10 and np.isclose(g[0], 0.5) and np.isclose(e[0], 0.7)
    assert S.delivered_plan(dict(info, medium={}), str(tmp_path)) is None  # 2x paint, nothing says which frames: no plan
    assert len(S.delivered_plan(dict(info, medium={}, fps_mult=1), str(tmp_path))[0]) == 10
    assert S.delivered_plan(dict(frames=10), str(tmp_path)) is None


def _repair(monkeypatch, tmp_path, info, **kw):
    sung, out = tmp_path / "sung.mp4", tmp_path / "out.mp4"
    sung.write_bytes(b"sung")
    seen = []

    def fake(take, a, b, report=None, **k):
        seen.append(k)
        open(b, "wb").write(b"repaired")

    monkeypatch.setattr(CONT, "run", fake)
    S.repair("take.mp4", str(sung), str(out), info, str(tmp_path), **kw)
    return seen


def test_the_repair_hands_the_plan_over_only_when_on(monkeypatch, tmp_path):
    wj = _warp(tmp_path / "warp.json", 20)
    info = dict(warp_json=wj, frames=10, fps_mult=2, medium=dict(picked=list(range(0, 20, 2))))
    assert _repair(monkeypatch, tmp_path, info) == [{"charge": "remade"}]  # nothing recorded: 0.3.0's call
    assert _repair(monkeypatch, tmp_path, dict(info, repair_plan="off")) == [{"charge": "remade"}]
    (k,) = _repair(monkeypatch, tmp_path, dict(info, repair_plan="warp"))
    assert set(k) == {"charge", "plan"} and len(k["plan"][0]) == 10
    (k,) = _repair(monkeypatch, tmp_path, info, plan="warp")
    assert "plan" in k
    assert _repair(monkeypatch, tmp_path, dict(info, repair_plan="warp"), plan="off") == [{"charge": "remade"}]  # the caller wins
    assert _repair(monkeypatch, tmp_path, dict(repair_plan="warp")) == [{"charge": "remade"}]  # no plan to read
    with pytest.raises(ValueError):
        _repair(monkeypatch, tmp_path, info, plan="nope")


def _coherence(monkeypatch, runs, **kw):
    """continuity.run on 90 blank frames with its rulers stubbed: odd_frames hands out `runs`, remake logs what it is given."""
    n = 90
    F = [np.zeros((8, 8, 3), np.uint8)] * n
    monkeypatch.setattr(CONT, "_read", lambda p: (F, 25.0))
    blank = dict(err=np.zeros(n), hf=np.zeros(n), blue=np.zeros(n, int), core=np.zeros(n, bool), motion=np.zeros(n, bool))
    monkeypatch.setattr(CONT, "check", lambda *a, **k: (np.zeros(n, bool), blank, [np.ones((8, 8), bool)] * n))
    it = iter([runs])
    monkeypatch.setattr(CONT, "odd_frames", lambda T, R, M, skip=(): (next(it, []), None))
    given = []

    def fake_remake(T, R, runs, M, alts=(), desync=None, **k):
        given.extend((i, L, None if desync is None else bool(desync[i : i + L].any())) for i, L, _ in runs)
        return R, [dict(start=i, length=L, dev=d, used="interp", dev_after=d) for i, L, d in runs]

    monkeypatch.setattr(CONT, "remake", fake_remake)
    monkeypatch.setattr(CONT, "write", lambda *a, **k: None)
    CONT.run("t.mp4", "r.mp4", "o.mp4", **kw)
    return given


RUNS = [(20, 3, 9.0), (50, 2, 8.0)]


def test_run_leaves_a_run_the_plan_asked_for(monkeypatch):
    """the plan opens the mouth over frames 20..22 (a sung onset) and is flat at 50..51: the first run is the plan's, the second a pop."""
    assert _coherence(monkeypatch, RUNS) == [(20, 3, None), (50, 2, None)]  # no plan (0.3.0): both re-made
    take, plan = np.full(90, 0.2), np.full(90, 0.2)
    plan[20:23] = 0.6
    plan[70:] = 0.5  # (the plan's range)
    assert _coherence(monkeypatch, RUNS, plan=(take, plan)) == [(50, 2, False)]


def test_run_marks_where_the_take_is_off_the_plan(monkeypatch):
    plan = np.linspace(0.2, 0.3, 90)
    take = plan.copy()
    take[49:53] = 0.9  # the take's mouth is open where the plan has it shut: its own mouth must not come back there
    assert _coherence(monkeypatch, RUNS, plan=(take, plan)) == [(20, 3, False), (50, 2, True)]


def test_the_flag_defaults():
    import inspect

    assert inspect.signature(CONT.run).parameters["plan"].default is None
    assert inspect.signature(S.repair).parameters["plan"].default is None
