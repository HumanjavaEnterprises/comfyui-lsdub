# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The travel guard read against the aligner's render (sing.travel_guard ref="aligner") and the continuity pass's coherence budget charged
only with runs it re-makes (continuity.run charge="remade"), and sing's drive shape (--warp-shape asym in SNAP). All three are the
defaults since 0.1.10; the 0.1.9 values (ref "take", charge "all", --warp-shape off) stay one flag away. Synthetic, no models."""

import inspect, os

import numpy as np
import pytest

from lsdub_engine.core import continuity as CONT, sing as S


def _guard(monkeypatch, tmp_path, travels, alt_pass=True, ref=None):
    """travel_guard with the readings stubbed: travels = (take, retimed render, aligner render)."""
    it = iter(travels)
    monkeypatch.setattr(S, "mouth_open", lambda v, R, n: np.array([0.0, next(it)]))
    monkeypatch.setattr(S, "travel", lambda op: None if op is None else float(op[1]))
    calls = []

    def fake_run(take, align, vocal, t0, out, work, **k):
        calls.append(out)
        os.makedirs(work, exist_ok=True)
        open(out, "wb").write(b"aligner")
        return dict(frames=4, truth=0.40, **{"pass": alt_pass})

    monkeypatch.setattr(S, "run", fake_run)
    out = tmp_path / "sung.mp4"
    out.write_bytes(b"retimed")
    info = dict(frames=4, truth=0.6, retime=dict(used=True), **{"pass": True})
    args = (info, "take.mp4", "a.json", "v.wav", 0.0, str(out), str(tmp_path), {}, dict(paint_args=()))
    got = S.travel_guard(*args) if ref is None else S.travel_guard(*args, ref=ref)
    return got, out.read_bytes(), calls


# a take that rests half open (p5 0.0185, p95 0.1088): the retimed and the aligner's render both travel about 0.75 of it
REST_OPEN = (0.0902, 0.0673, 0.0677)
# a retime that shortened the openings: the aligner's render travels 1.43x the retimed one
SHORTENED = (0.1333, 0.0717, 0.1025)


def test_the_take_reference_is_unchanged(monkeypatch, tmp_path):
    """ref take (0.1.9): the same readings as before (0.0673 < 0.75 x 0.0902 and 0.0677 >= it): the aligner is delivered."""
    got, body, calls = _guard(monkeypatch, tmp_path, REST_OPEN, ref="take")
    assert body == b"aligner" and got["retime"]["guard"]["delivered"] == "aligner" and len(calls) == 1
    assert "shortened" not in got["retime"]["guard"]


def test_the_aligner_reference_keeps_a_retime_that_did_not_shorten_the_mouth(monkeypatch, tmp_path):
    got, body, calls = _guard(monkeypatch, tmp_path, REST_OPEN, ref="aligner")
    g = got["retime"]["guard"]
    assert body == b"retimed" and got["truth"] == 0.6 and len(calls) == 1
    assert g["held"] is True and g["shortened"] is False and g["delivered"] == "retimed" and g["aligner_travel"] == 0.0677


def test_the_aligner_reference_still_swaps_a_shortened_retime(monkeypatch, tmp_path):
    got, body, _ = _guard(monkeypatch, tmp_path, SHORTENED, ref="aligner")
    g = got["retime"]["guard"]
    assert body == b"aligner" and g["shortened"] is True and g["delivered"] == "aligner" and got["truth"] == 0.40


def test_the_aligner_reference_never_delivers_a_failing_aligner(monkeypatch, tmp_path):
    got, body, _ = _guard(monkeypatch, tmp_path, SHORTENED, alt_pass=False, ref="aligner")
    assert body == b"retimed" and got["retime"]["guard"]["delivered"] == "retimed"


def test_the_aligner_reference_does_not_fire_without_a_held_reading(monkeypatch, tmp_path):
    got, body, calls = _guard(monkeypatch, tmp_path, (0.108, 0.095), ref="aligner")
    assert body == b"retimed" and calls == [] and got["retime"]["guard"]["held"] is False


def test_the_floor_still_holds_against_the_aligner(monkeypatch, tmp_path):
    """under TRAVEL_FLOOR a mouth is a held gape whatever the aligner does: the aligner (passing, moving) is delivered."""
    got, body, _ = _guard(monkeypatch, tmp_path, (0.06, 0.045, 0.055), ref="aligner")
    assert body == b"aligner" and got["retime"]["guard"]["shortened"] is True


def test_an_unknown_reference_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _guard(monkeypatch, tmp_path, REST_OPEN, ref="plan")


def test_the_cli_flag_reaches_the_guard(monkeypatch):
    seen = {}
    monkeypatch.setattr(S, "run", lambda *a, **k: seen.update(k) or {"pass": True})
    base = ["--take", "t.mp4", "--align", "a.json", "--vocal", "v.wav", "--t0", "0", "--out", "o.mp4", "--work", "w"]
    S.main(base)
    assert seen["travel_ref"] == "aligner"  # the default
    S.main(base + ["--travel-guard", "take"])
    assert seen["travel_ref"] == "take"  # 0.1.9


def _repair_calls(monkeypatch, tmp_path, **kw):
    (tmp_path / "medium.json").write_text("{}")
    sung, out = tmp_path / "sung.mp4", tmp_path / "render.mp4"
    sung.write_bytes(b"sung")
    seen = []

    def fake(take, a, b, report=None, **k):
        seen.append(k)
        open(b, "wb").write(b"repaired")

    monkeypatch.setattr(CONT, "run", fake)
    monkeypatch.setattr(S, "delivered_truth", lambda v, info, work: dict(truth=0.6, truth_pass=True))
    monkeypatch.setattr(S, "repaired_silence", lambda v, info, work: 0.1)
    r = S.repair("take.mp4", str(sung), str(out), dict(truth=0.55, truth_pass=True), str(tmp_path), **kw)
    return r, seen


def test_the_repair_passes_its_charge(monkeypatch, tmp_path):
    r, seen = _repair_calls(monkeypatch, tmp_path)
    assert seen == [{"charge": "remade"}] and r["repair"] == "kept"  # the default
    r, seen = _repair_calls(monkeypatch, tmp_path, charge="all")
    assert seen == [{"charge": "all"}] and r["repair_truth"] == 0.6  # 0.1.9


def _coherence(monkeypatch, charge, rounds):
    """continuity.run on 90 blank frames with its rulers stubbed: odd_frames hands out `rounds` (one list of runs per round), remake
    logs what it is given (kept for a run over 3 frames, as the real one does without an alternate render)."""
    n = 90
    F = [np.zeros((8, 8, 3), np.uint8)] * n
    monkeypatch.setattr(CONT, "_read", lambda p: (F, 25.0))
    blank = dict(err=np.zeros(n), hf=np.zeros(n), blue=np.zeros(n, int), core=np.zeros(n, bool), motion=np.zeros(n, bool))
    monkeypatch.setattr(CONT, "check", lambda *a, **k: (np.zeros(n, bool), blank, [np.ones((8, 8), bool)] * n))
    it = iter(rounds)
    monkeypatch.setattr(CONT, "odd_frames", lambda T, R, M, skip=(): (next(it, []), None))
    given = []

    def fake_remake(T, R, runs, M, alts=(), desync=None, **k):
        given.extend((i, L) for i, L, _ in runs)
        return R, [dict(start=i, length=L, dev=d, used="kept" if CONT.kept_as_is(L, alts) else "interp", dev_after=d) for i, L, d in runs]

    monkeypatch.setattr(CONT, "remake", fake_remake)
    monkeypatch.setattr(CONT, "write", lambda *a, **k: None)
    rep = CONT.run("t.mp4", "r.mp4", "o.mp4", charge=charge)
    return given, rep


# the shape read on a real shot: a 7-frame run kept as it is, a 2-frame blip, then a 3-frame dip, on a 90-frame shot (budget 9 frames)
ROUNDS = [[(33, 7, 88.35), (47, 2, 37.77)], [(15, 3, 43.37)]]


def test_all_charge_counts_kept_runs(monkeypatch):
    """charge all (0.1.9): a kept 7-frame run and a 2-frame blip spend the 9-frame share, the 3-frame dip is never tried."""
    given, rep = _coherence(monkeypatch, "all", ROUNDS)
    assert given == [(33, 7), (47, 2)] and rep["coherence_frames"] == 9


def test_remade_charge_leaves_a_kept_run_free(monkeypatch):
    given, rep = _coherence(monkeypatch, "remade", ROUNDS)
    assert given == [(33, 7), (47, 2), (15, 3)] and rep["coherence_frames"] == 5
    assert [l["used"] for l in rep["coherence"]] == ["kept", "interp", "interp"]


def test_remade_charge_still_caps_what_is_remade(monkeypatch):
    given, rep = _coherence(monkeypatch, "remade", [[(10, 3, 9.0), (20, 3, 8.0), (30, 3, 7.0), (40, 3, 6.0)]])
    assert given == [(10, 3), (20, 3), (30, 3)] and rep["coherence_frames"] == 9


def test_an_unknown_charge_is_refused():
    with pytest.raises(ValueError):
        CONT.run("t.mp4", "r.mp4", "o.mp4", charge="none")


def test_kept_as_is_is_the_remake_rule():
    assert CONT.kept_as_is(4, ()) and not CONT.kept_as_is(3, ()) and not CONT.kept_as_is(7, [np.zeros(1)])


# the default settings: sing travel_ref aligner, repair charge remade, paint --warp-shape asym
ACCEPTED = dict(travel_ref="aligner", charge="remade", warp_shape="asym")
V019 = dict(travel_ref="take", charge="all", warp_shape="off")


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


def test_the_defaults_are_the_accepted_settings():
    assert _default(S.run, "travel_ref") == ACCEPTED["travel_ref"]
    assert _default(S.repair, "charge") == _default(CONT.run, "charge") == ACCEPTED["charge"]
    p = S.SNAP["paint"]
    assert p[p.index("--warp-shape") + 1] == ACCEPTED["warp_shape"]
    assert S.snap_paint(["--lib", "kit"], 2) == p  # what sing hands paint when the caller names no --warp-shape


def test_the_cli_defaults_are_the_accepted_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(CONT, "run", lambda *a, **k: seen.update(k))
    CONT.main(["--take", "t.mp4", "--render", "r.mp4", "--out", "o.mp4"])
    assert seen["charge"] == ACCEPTED["charge"]
    CONT.main(["--take", "t.mp4", "--render", "r.mp4", "--out", "o.mp4", "--coherence-charge", V019["charge"]])
    assert seen["charge"] == V019["charge"]


def test_the_019_values_are_one_flag_away():
    """0.1.9: --travel-guard take, --coherence-charge all, and `-- --warp-shape off` after sing's flags, which leaves SNAP as it was."""
    assert V019["travel_ref"] in S.TRAVEL_REFS and V019["charge"] in CONT.CHARGES
    snap_019 = ["--warp-sigma", "0.8", "--warp-temporal", "0.6", "--warp-close-depth", "0.05", "--warp-follow", "0.15"]
    assert S.snap_paint(["--warp-shape", V019["warp_shape"]], 2) == snap_019
    assert S.snap_paint(["--warp-shape=closures"], 2) == snap_019  # the caller's own shape, in either spelling, wins
    assert S.SNAP["paint"] == [*snap_019, "--warp-shape", "asym"]  # SNAP itself is left alone


def test_a_callers_shape_reaches_paint(monkeypatch):
    """sing's paint argv: the caller's --warp-shape is the only one in it; with none named, SNAP's asym is."""
    from lsdub_engine.core import paint_api

    for flags, want in (([], ["asym"]), (["--warp-shape", "off"], ["off"])):
        pa = paint_api.flags(user_overrides=flags, operational=S.snap_paint(flags, 2))
        assert [pa[i + 1] for i, x in enumerate(pa) if x == "--warp-shape"] == want
