# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The sing path's sync fixes (synthetic, no models): the plain target held near the plan (--warp-follow), the plan re-read off the vocal
(vocal_retime as a step), and the continuity repair kept only when the mouth still follows the vocal."""

import json
import os

import numpy as np

from lsdub_engine.core import mouth_warp as MW, sing as S

RNG = dict(g_lo=0.0, g_hi=10.0, w_lo=4.0, w_med=5.0, w_hi=6.0)


def frames(opens, vis="aa"):
    return [dict(vis=vis, vis2=vis, mix=0.0, open=o) for o in opens]


def test_follow_holds_a_held_vowels_opening_to_the_plan():
    """the take's mouth sits wide open (9 of 10) through a vowel whose plan has already closed to 0.2: the band alone keeps the take's
    gape (it is inside aa's band), follow brings it down to the plan's neighbourhood."""
    F = np.column_stack([np.full(6, 9.0), np.full(6, 5.0)])
    fr = frames([0.9, 0.9, 0.2, 0.2, 0.9, 0.9])
    plain = MW.target_features(fr, F, RNG, sigma=0.0)
    held = MW.target_features(fr, F, RNG, sigma=0.0, follow=0.15)
    assert np.allclose(plain[:, 0], 9.0)  # without it: the take, untouched
    assert np.allclose(held[[0, 1, 4, 5], 0], 9.0)  # on the plan already: untouched
    assert np.all(held[[2, 3], 0] <= (0.2 + 0.15) * 10 + 1e-9)  # held within +-0.15 of the plan's 0.2


def test_follow_leaves_the_take_alone_near_the_plan_and_keeps_closures():
    F = np.column_stack([np.array([5.0, 5.2, 0.4]), np.full(3, 5.0)])
    fr = frames([0.5, 0.55, 0.0], vis="E")
    fr[2]["vis"] = fr[2]["vis2"] = "PP"
    a = MW.target_features(fr, F, RNG, sigma=0.0, follow=0.15)
    b = MW.target_features(fr, F, RNG, sigma=0.0)
    assert np.allclose(a[:2, 0], F[:2, 0])  # the take's own small motion near the plan passes
    assert np.allclose(a[2], b[2])  # a closure is the closure's own band either way


def test_follow_is_off_by_default_and_on_in_snap():
    F = np.column_stack([np.full(3, 9.0), np.full(3, 5.0)])
    fr = frames([0.2, 0.2, 0.2])
    assert np.allclose(MW.target_features(fr, F, RNG, sigma=0.0), MW.target_features(fr, F, RNG, sigma=0.0, follow=0.0))
    p = S.SNAP["paint"]
    assert p[p.index("--warp-follow") + 1] == "0.15"


def _retime_inputs(monkeypatch, tmp_path, truths):
    """retimed_align with the walk and the two plan readings stubbed: truths = (aligned plan, retimed plan)."""
    from lsdub_engine.audits import sync_truth as ST
    from lsdub_engine.core import viseme_track as VT, vocal_retime as VR

    a, v = tmp_path / "a.json", tmp_path / "v.wav"
    a.write_text(json.dumps(dict(words=[dict(w="la", start=1.0, end=1.5)])))
    v.write_bytes(b"")
    seen = {}
    monkeypatch.setattr(VR, "read_mono", lambda p: np.zeros(16000))
    monkeypatch.setattr(VR, "retime", lambda align, audio, x=None, cap=None: seen.update(cap=cap) or dict(align, retimed="vocal"))
    monkeypatch.setattr(VT, "envelope_open", lambda *x, **k: np.ones(10))
    monkeypatch.setattr(VT, "build", lambda al, *x, **k: [dict(open=1.0 if al.get("retimed") else 0.0)] * 10)
    it = iter(truths)
    monkeypatch.setattr(ST, "read", lambda o, x, fps, keep=None: dict(truth=next(it)))
    return str(a), str(v), seen


def test_sing_plans_from_the_vocal_retime_when_it_makes_the_plan_better(monkeypatch, tmp_path):
    a, v, seen = _retime_inputs(monkeypatch, tmp_path, (0.30, 0.42))
    out, note = S.retimed_align(a, v, str(tmp_path), 0.0, 10, 25.0)
    assert out == str(tmp_path / "align_vocal.json") and json.load(open(out))["retimed"] == "vocal"
    assert note == {"used": True, "cap_s": S.RETIME_CAP, "plan_truth": [0.30, 0.42]} and seen["cap"] == S.RETIME_CAP


def test_sing_keeps_the_aligner_when_the_retime_does_not_help_this_shot(monkeypatch, tmp_path):
    a, v, _ = _retime_inputs(monkeypatch, tmp_path, (0.42, 0.40))
    out, note = S.retimed_align(a, v, str(tmp_path), 0.0, 10, 25.0)
    assert out == a and note["used"] is False and not (tmp_path / "align_vocal.json").exists()
    assert S.retimed_align(str(tmp_path / "none.json"), v, str(tmp_path))[0] == str(tmp_path / "none.json")  # nothing to read


def _w(w, a, b, a0, b0):
    ph = [dict(p="AA", start=a, end=b)]
    return dict(w=w, start=a, end=b, start0=a0, end0=b0, phones=ph, phones_aligned=[dict(p="AA", start=a0, end=b0, start0=a0, end0=b0)])


def test_the_bounded_retime_holds_far_moves_and_ad_libs_at_the_aligners_times():
    from lsdub_engine.core.vocal_retime import bounded

    W = [_w("we", 0.0, 0.5, 0.0, 0.4), _w("loud", 0.45, 2.9, 1.8, 2.8), _w("da", 3.0, 3.6, 3.2, 3.5), _w("go", 3.6, 4.0, 3.5, 4.0)]
    out = bounded(W, 0.25)
    assert (out[0]["start"], out[0]["end"]) == (0.0, 0.5)  # moved 0.1: kept
    assert (out[1]["start"], out[1]["end"]) == (1.8, 2.8) and out[1]["phones"][0]["start"] == 1.8  # moved 1.35 s: the aligner's
    assert (out[2]["start"], out[2]["end"]) == (3.2, 3.5)  # an ad-lib: never moves, so its span does not grow
    assert out[3]["start"] >= 3.5 and out[3]["phones"][0]["start"] >= 3.5  # its neighbour trimmed off it
    assert all(a["end"] <= b["start"] + 1e-9 for a, b in zip(out, out[1:]))


def test_the_bounded_walk_cannot_move_a_phone_past_the_cap(tmp_path):
    from lsdub_engine.core import vocal_retime as VR

    sr = VR.SR
    t = np.arange(int(3.0 * sr)) / sr
    x = 0.002 * np.random.default_rng(0).standard_normal(len(t))
    m = (t >= 1.8) & (t < 2.4)  # the vowel is really sung 0.8 s after the aligner put it
    x[m] += 0.3 * np.sin(2 * np.pi * 220 * t[m])
    A = dict(words=[dict(w="la", start=1.0, end=1.6, phones=[dict(p="L", start=1.0, end=1.1), dict(p="AA", start=1.1, end=1.6)])])
    free = VR.retime(A, "", x=x)["words"][0]
    capped = VR.retime(A, "", x=x, cap=0.25)["words"][0]
    assert abs(free["start"] - 1.0) > 0.25  # the anchored walk follows the vocal that far
    assert abs(capped["start"] - 1.0) <= 0.25 + 1e-9 and abs(capped["end"] - 1.6) <= 0.25 + 1e-9


def test_truth_on_the_vocal_file_reads_every_frame_and_reports_the_kept_ones(tmp_path):
    import wave

    from lsdub_engine.audits import sync_truth as ST
    from lsdub_engine.core.vocal_retime import SR

    from test_sync_truth import FPS, LINE, N, X, mouth  # the same synthetic line

    p = tmp_path / "v.wav"
    with wave.open(str(p), "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(SR)
        f.writeframes((np.clip(X, -1, 1) * 32767).astype("<i2").tobytes())
    m = mouth(LINE, N)
    m[25:60] = 0.12 - m[25:60]
    keep = np.ones(N, bool)
    keep[25:60] = False
    r = ST.read_wav(m, str(p), FPS, keep)
    assert r["truth_kept"] > r["truth"] + 0.1  # the handed-back frames count in the headline, not in the kept reading
    assert "truth_kept" not in ST.read_wav(m, str(p), FPS)


def test_run_hands_the_retimed_alignment_on_and_can_be_told_not_to(monkeypatch, tmp_path):
    F = [np.zeros((64, 64, 3), np.uint8)] * 4
    R = dict(face_h_px=64.0, zoom=1.0, crop_px=64, outside_max_preencode=0, not_pasted=[], picked=[0], timings={})
    monkeypatch.setattr(S.MED, "read", lambda v, n=None: (F, 25.0))
    monkeypatch.setattr(S.MED, "run", lambda *a, **k: R)
    calls = []
    monkeypatch.setattr(S.steps, "call", lambda step, argv, **k: calls.append((step, list(argv))))
    monkeypatch.setattr(S, "dyn_target", lambda *a: np.zeros(4))
    monkeypatch.setattr(S, "retimed_align", lambda align, vocal, work, *a: (work + "/align_vocal.json", {"used": True}))
    monkeypatch.setattr(S, "travel_guard", lambda info, *a: info)  # its own tests below
    gated = []
    g = dict(corr=0.9, best=0.9, lag_ms=0, open_in_silence=0.1, rise_ms=10, closure_hit=1, **{"pass": True})
    monkeypatch.setattr(S, "gate", lambda v, R, t, vocal, align, *a, **k: gated.append(align) or g)
    for retime in (True, False):
        calls.clear()
        w = tmp_path / str(retime)
        info = S.run("t.mp4", "a.json", "v.wav", 0.0, str(w / "o.mp4"), str(w), box=[0, 0, 64, 64], vocal_retime=retime)
        want = str(w) + "/align_vocal.json" if retime else "a.json"
        assert info["align"] == want and gated[-1] == want
        assert [a[a.index("--align") + 1] for s, a in calls if s in ("sung_dyn", "visemes")] == [want, want]


def _repair(monkeypatch, tmp_path, before, after, silence=0.2):
    """S.repair with the continuity pass and the delivered readings stubbed: before = the gate's reading, after = the repaired shot's
    truth, silence = its open-in-silence."""
    (tmp_path / "medium.json").write_text("{}")
    sung, out = tmp_path / "sung.mp4", tmp_path / "render.mp4"
    sung.write_bytes(b"sung")
    from lsdub_engine.core import continuity as CONT

    monkeypatch.setattr(CONT, "run", lambda take, a, b, report=None: open(b, "wb").write(b"repaired"))
    monkeypatch.setattr(S, "delivered_truth", lambda v, info, work: after)
    monkeypatch.setattr(S, "repaired_silence", lambda v, info, work: silence)
    r = S.repair("take.mp4", str(sung), str(out), before, str(tmp_path))
    return r, out.read_bytes()


def test_a_repair_that_costs_the_sync_is_undone(monkeypatch, tmp_path):
    r, got = _repair(monkeypatch, tmp_path, dict(truth=0.45, truth_pass=True), dict(truth=0.30, truth_pass=True))
    assert r["repair"] == "undone" and got == b"sung"


def test_a_repair_that_turns_a_pass_into_a_fail_is_undone(monkeypatch, tmp_path):
    r, got = _repair(monkeypatch, tmp_path, dict(truth=0.32, truth_pass=True), dict(truth=0.31, truth_pass=False))
    assert r["repair"] == "undone" and got == b"sung"


def test_a_repair_that_keeps_the_sync_is_delivered(monkeypatch, tmp_path):
    r, got = _repair(monkeypatch, tmp_path, dict(truth=0.45, truth_pass=True), dict(truth=0.44, truth_pass=True))
    assert r == {"repair": "kept", "repair_truth": 0.44, "repair_silence": 0.2} and got == b"repaired"


def test_a_repair_that_opens_the_silences_past_the_line_is_undone(monkeypatch, tmp_path):
    before = dict(truth=0.45, truth_pass=True, open_in_silence=0.30)
    r, got = _repair(monkeypatch, tmp_path, before, dict(truth=0.45, truth_pass=True), silence=0.37)
    assert r["repair"] == "undone" and got == b"sung"
    r, got = _repair(monkeypatch, tmp_path, before, dict(truth=0.45, truth_pass=True), silence=0.34)
    assert r["repair"] == "kept"


def test_a_retime_gain_under_the_margin_keeps_the_aligner(monkeypatch, tmp_path):
    a, v, _ = _retime_inputs(monkeypatch, tmp_path, (0.497, 0.506))
    out, note = S.retimed_align(a, v, str(tmp_path), 0.0, 10, 25.0)
    assert out == a and note["used"] is False and S.RETIME_MARGIN == 0.02


def _guard(monkeypatch, tmp_path, travels, alt_pass=True, occ=None):
    """travel_guard with the readings stubbed: travels = (take, retimed render, aligner render)."""
    it = iter(travels)
    monkeypatch.setattr(S, "mouth_open", lambda v, R, n: np.array([0.0, next(it)]))
    monkeypatch.setattr(S, "travel", lambda op: None if op is None else float(op[1]))
    calls = []
    alt_info = dict(frames=4, truth=0.40, **{"pass": alt_pass})

    def fake_run(take, align, vocal, t0, out, work, **k):
        occ = os.path.join(work, "occ.npz")
        calls.append(dict(align=align, retime=k.get("vocal_retime"), out=out, occ=open(occ, "rb").read() if os.path.exists(occ) else None))
        os.makedirs(work, exist_ok=True)
        open(out, "wb").write(b"aligner")
        return dict(alt_info)

    monkeypatch.setattr(S, "run", fake_run)
    out = tmp_path / "sung.mp4"
    out.write_bytes(b"retimed")
    if occ is not None:
        (tmp_path / "occ.npz").write_bytes(occ)
    info = dict(frames=4, truth=0.6, retime=dict(used=True), **{"pass": True})
    got = S.travel_guard(info, "take.mp4", "a.json", "v.wav", 0.0, str(out), str(tmp_path), {}, dict(paint_args=()))
    return got, out.read_bytes(), calls


def test_a_retimed_held_gape_is_delivered_from_the_aligner_when_that_one_is_not(monkeypatch, tmp_path):
    got, body, calls = _guard(monkeypatch, tmp_path, (0.108, 0.072, 0.103))
    assert body == b"aligner" and calls[0]["align"] == "a.json" and calls[0]["retime"] is False
    assert got["retime"]["used"] is False and got["retime"]["guard"]["delivered"] == "aligner" and got["truth"] == 0.40


def test_the_aligner_render_paints_with_the_shots_occluder_mask(monkeypatch, tmp_path):
    _, _, calls = _guard(monkeypatch, tmp_path, (0.108, 0.072, 0.103), occ=b"mask")
    assert calls[0]["occ"] == b"mask"  # a mic at the lips stays in front on the second render too
    (tmp_path / "none").mkdir()
    _, _, calls = _guard(monkeypatch, tmp_path / "none", (0.108, 0.072, 0.103))
    assert calls[0]["occ"] is None


def test_a_retimed_shot_that_moves_like_its_take_is_kept(monkeypatch, tmp_path):
    got, body, calls = _guard(monkeypatch, tmp_path, (0.108, 0.095))
    assert body == b"retimed" and calls == [] and got["retime"]["guard"]["held"] is False


def test_the_retimed_render_stays_when_the_aligner_is_no_better(monkeypatch, tmp_path):
    got, body, _ = _guard(monkeypatch, tmp_path, (0.108, 0.072, 0.070))
    assert body == b"retimed" and got["retime"]["guard"]["delivered"] == "retimed"
    got, body, _ = _guard(monkeypatch, tmp_path, (0.108, 0.072, 0.103), alt_pass=False)
    assert body == b"retimed"


def test_an_unread_gate_keeps_the_repair(monkeypatch, tmp_path):
    r, got = _repair(monkeypatch, tmp_path, dict(truth=None), dict(truth=0.1, truth_pass=False))
    assert r == {"repair": "kept"} and got == b"repaired"


def test_the_gates_box_is_the_paste_centre(tmp_path):
    R = {"per_frame": [{"centre": (100, 50)}, {"centre": (110, 60)}], "face_h_px": 50}
    assert S.face_roi(R, (200, 300, 3)) == [15, 95, 65, 145]
    json.dumps(S.face_roi(R, (200, 300, 3)))  # plain ints: it goes into a report


def test_the_drive_pump_and_compressor_are_opt_in_and_reach_both_curves(monkeypatch, tmp_path):
    F = [np.zeros((64, 64, 3), np.uint8)] * 4
    R = dict(face_h_px=64.0, zoom=1.0, crop_px=64, outside_max_preencode=0, not_pasted=[], picked=[0], timings={})
    monkeypatch.setattr(S.MED, "read", lambda v, n=None: (F, 25.0))
    monkeypatch.setattr(S.MED, "run", lambda *a, **k: R)
    monkeypatch.setattr(S, "dyn_target", lambda *a: np.zeros(4))
    monkeypatch.setattr(S, "compressed_vocal", lambda vocal, work: work + "/vocal_drive.wav")
    g = dict(corr=0.9, best=0.9, lag_ms=0, open_in_silence=0.1, rise_ms=10, closure_hit=1, **{"pass": True})
    gated = []
    monkeypatch.setattr(S, "gate", lambda v, R, t, vocal, *a, **k: gated.append(vocal) or g)
    for kw in ({}, dict(pump=(0.35, 4.0), drive_comp=True)):
        calls = []
        monkeypatch.setattr(S.steps, "call", lambda step, argv, **k: calls.append((step, list(argv))))
        w = str(tmp_path / str(len(kw)))
        S.run("t.mp4", "a.json", "v.wav", 0.0, w + "/o.mp4", w, box=[0, 0, 64, 64], vocal_retime=False, **kw)
        plan = [a for s, a in calls if s in ("sung_dyn", "visemes")]
        assert all(("--pump-gate" in a) == bool(kw) for a in plan)
        assert all(a[a.index("--audio") + 1] == (w + "/vocal_drive.wav" if kw else "v.wav") for a in plan)
        assert gated[-1] == "v.wav"  # the gate always reads the lead itself


def test_the_viseme_envelope_goes_through_the_pump(tmp_path, monkeypatch):
    from lsdub_engine.core import viseme_track as VT

    a = tmp_path / "a.json"
    a.write_text(json.dumps(dict(words=[dict(w="la", start=0.0, end=1.0, phones=[dict(p="AA", start=0.0, end=1.0)])])))
    env = np.linspace(0.0, 1.0, 25)
    monkeypatch.setattr(VT, "envelope_open", lambda *x, **k: env.copy())
    out = {}
    for gate in (0, 0.35):
        o = tmp_path / f"t{gate}.json"
        VT.main(
            ["--align", str(a), "--t0", "0", "--frames", "25", "--fps", "25", "--audio", "v.wav", "--pump-gate", str(gate), "--out", str(o)]
        )
        out[gate] = np.array([f["open"] for f in json.load(open(o))["frames"]])
    assert out[0.35][2] == 0.0 < out[0][2]  # under the gate: shut
    assert out[0.35][12] > out[0][12]  # over it: squashed toward full


def test_verify_reads_closures_against_the_alignment_the_dub_planned_from(tmp_path):
    from lsdub_engine.audits import qa as QA

    work = tmp_path / "shot" / "work"
    work.mkdir(parents=True)
    rv = work / "align_vocal.json"
    rv.write_text(json.dumps(dict(words=[])))
    s = dict(warp_json=str(work / "crop_painted_warp.json"))
    gate = tmp_path / "shot" / "gate.json"
    gate.write_text(json.dumps(dict(align=str(rv), retime=dict(used=True))))
    assert QA.shot_align(s, "project.json") == str(rv)
    gate.write_text(json.dumps(dict(align="project.json", retime=dict(used=False))))  # the retime stood aside
    assert QA.shot_align(s, "project.json") == "project.json"
    assert QA.shot_align(dict(s, used="source"), "project.json") == "project.json"
    assert QA.shot_align({}, "project.json") == "project.json"
