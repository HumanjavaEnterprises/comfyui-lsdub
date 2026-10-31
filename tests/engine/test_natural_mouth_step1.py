# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Natural mouth step 1 and the step 3 source test: the opt-in flags and the ruler. Golden hashes prove the defaults."""

import numpy as np, cv2, pytest

from lsdub_engine.audits import mouth_ruler as MR
from lsdub_engine.core import interior_take as IT, lip_layers as ll, medium as M, mouth_layers as ML, mouth_paint as MP
from lsdub_engine.core.mouth_warp import MouthBank

BASE = ["--video", "v", "--track", "t", "--out", "o"]


def face(n=478, gap=20.0):
    """a flat mouth: 61 / 291 = corners, 13 / 14 = aperture, inner contour on an ellipse."""
    P = np.zeros((n, 3))
    P[:, :2] = [100, 100]
    th = np.linspace(0, 2 * np.pi, len(ll.LIP_IN), endpoint=False)
    P[ll.LIP_IN, 0] = 100 + 40 * np.cos(th)
    P[ll.LIP_IN, 1] = 100 + gap * np.sin(th)
    o = np.linspace(0, 2 * np.pi, len(ll.LIP_OUT), endpoint=False)
    P[ll.LIP_OUT, 0] = 100 + 50 * np.cos(o)
    P[ll.LIP_OUT, 1] = 100 + (gap + 10) * np.sin(o)
    P[61], P[291], P[13], P[14] = [50, 100, 0], [150, 100, 0], [100, 100 - gap, 0], [100, 100 + gap, 0]
    return P


def frame(bgr=(60, 60, 170)):
    f = np.full((200, 200, 3), 150, np.uint8)
    cv2.ellipse(f, (100, 100), (40, 20), 0, 0, 360, bgr, -1)
    return f


def clip(n=8, bgr=(60, 60, 170)):
    F = [frame(bgr) for _ in range(n)]
    S = np.array([face() for _ in range(n)])
    feat = np.tile([40.0, 100.0], (n, 1))
    return F, S, feat, MouthBank(0, feat, np.zeros((n, 3, 2)), np.zeros((n, 3)), S[:, :, :2])


def test_new_flags_default_off():
    a = MP._parser().parse_args(BASE)
    assert (a.warp_tongue_pose, a.warp_fill_l3, a.warp_interior_source, a.warp_lossless) == ("zero", "hide", "model", False)
    assert ML.LayerCompositor.take_interior is None and ML.LayerCompositor.fill_l3 == "hide"


def test_slow_tongue_varies_without_slosh():
    c = ML.LayerCompositor.__new__(ML.LayerCompositor)
    c.n = 100
    v = ["E", "aa"] * 25 + ["E"] * 25 + ["aa"] * 25  # a per-frame flip barely moves it, a held vowel does
    c.slow_tongue(v, 6.0)
    assert np.ptp(c.tg[15:35, 0]) < 0.1 * np.ptp(c.tg[:, 0])
    assert np.ptp(c.tg[50:, 0]) > 0 and np.abs(np.diff(c.tg, axis=0)).max() < 0.25 * np.ptp(c.tg[:, 0]) + 1e-9


def test_rule_says_the_pose_needs_single():
    w = MP.validate_paint_args(MP._parser().parse_args([*BASE, "--mode", "warp", "--warp-tongue-pose", "lowpass"]))
    assert any("--warp-tongue-pose" in x and "single" in x for x in w)


def test_lossless_write_round_trips(tmp_path):
    p = str(tmp_path / "x.mp4")
    F = [np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8) for i in range(3)]
    M.write(F, p, lossless=True)
    assert all((a == b).all() for a, b in zip(F, M.read(p, 3)[0]))


def test_take_interior_fills_with_the_takes_pixels(monkeypatch):
    monkeypatch.setattr(IT, "PINK_ROOM", 1.0)
    F, S, Fe, bank = clip()
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    Q = S[0][ll.LIP_IN][:, :2]
    rgb, a = T.paint(0, Q, (0, 0, 200, 200), 0.4, np.zeros((200, 200), np.float32))
    assert a[100, 100] > 0.9 and np.abs(rgb[100, 100] - F[0][100, 100]).max() < 16
    assert a[5, 5] == 0


def test_pink_never_above_the_takes_share():
    F, S, Fe, bank = clip()
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    T.curve[:] = 0.2
    rgb, a = T.paint(0, S[0][ll.LIP_IN][:, :2], (0, 0, 200, 200), 0.4, np.zeros((200, 200), np.float32))
    assert T.log[0]["sank"] > 0.5
    assert cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2LAB)[100, 100, 0] < cv2.cvtColor(F[0], cv2.COLOR_BGR2LAB)[100, 100, 0] + 40


def test_pink_cap_attacks_at_once_and_releases_slowly():
    F, S, Fe, bank = clip()
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    Q = S[0][ll.LIP_IN][:, :2]
    zero = np.zeros((200, 200), np.float32)
    paint = lambda i: T.paint(i, Q, (0, 0, 200, 200), 0.4, zero)
    T.curve[:] = 2.0  # the take's share is out of reach: nothing to cap
    paint(0)
    assert T.log[0]["sank"] == 0
    T.curve[:] = 0.2
    for i in (1, 2, 3):
        paint(i)
        assert T.log[i]["held"] <= T.log[i]["target"] + 1e-3  # the first breach frame is held too, not a fifth of it
    top = T.log[3]["sank"]
    T.curve[:] = 2.0
    tail = []
    for i in range(4, 8):
        paint(i)
        tail.append(T.log[i]["sank"])
    assert tail[0] == pytest.approx(IT.RELEASE * top, abs=2e-3)  # eased, not dropped to 0
    assert all(b < a for a, b in zip(tail, tail[1:])) and tail[-1] > 0


def test_pink_cap_lands_on_the_target_not_far_under_it():
    lab = np.full((64, 256, 3), 120.0, np.float32)
    lab[..., 1] = np.linspace(143, 190, 256, dtype=np.float32)[None, :]  # every pixel pink, each leaving the test at its own blend
    vis = np.ones(lab.shape[:2], bool)
    pm = np.ones((*lab.shape[:2], 1), np.float32)
    base = np.array([120.0, 128.0, 128.0], np.float32)
    for tgt in (0.2, 0.5, 0.8):
        k = IT.least_sink(lab, pm, base, vis, 0, vis.size, tgt)
        held = float(IT.pink(lab * (1 - pm * k) + base * pm * k).mean())
        assert 0.8 * tgt <= held <= tgt


def test_least_sink_says_when_nothing_holds_the_share():
    lab = np.full((8, 8, 3), 120.0, np.float32)
    vis = np.ones((8, 8), bool)
    assert IT.least_sink(lab, np.ones((8, 8, 1), np.float32), lab[0, 0], vis, 64, 64, 0.5) == 1.0  # the part it cannot sink is over


def test_pink_tail_decays_paints_and_stops():
    F, S, Fe, bank = clip()
    Q = S[0][ll.LIP_IN][:, :2]
    zero = np.zeros((200, 200), np.float32)
    run = lambda T, i: T.paint(i, Q, (0, 0, 200, 200), 0.4, zero)[0]
    make = lambda: IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    clean, tailed, faded = make(), make(), make()
    for T in (clean, tailed, faded):
        T.curve[:] = 2.0  # no frame needs the cap: only the stored tail can sink anything
    tailed.sank[0] = 0.5
    faded.sank[0] = IT.SANK_MIN * 1.01
    a, b, c = run(clean, 1), run(tailed, 1), run(faded, 1)
    assert tailed.log[1]["sank"] == pytest.approx(IT.RELEASE * 0.5, abs=1e-3)  # stored, decayed
    assert tailed.sank[1] == pytest.approx(IT.RELEASE * 0.5, abs=1e-3)
    assert np.abs(a - b).max() > 1  # the tail changes the pixels
    assert faded.log[1]["sank"] == 0 and faded.sank[1] == 0  # under the floor it is off ...
    assert np.array_equal(a, c)  # ... and paints nothing


def test_interior_state_round_trips_and_clears():
    F, S, Fe, bank = clip()
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    T.log[3], T.gain[3], T.sank[3] = dict(sank=0.4), 0.9, 0.4
    st = T.state(3)
    T.put(3, T.EMPTY)
    assert 3 not in T.log and 3 not in T.gain and 3 not in T.sank
    T.put(3, st)
    assert T.state(3) == st


def test_a_rejected_gate_candidate_leaves_no_interior_state():
    import types
    from lsdub_engine.core import frame_gate as FG

    F, S, Fe, bank = clip()
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    T.log[2], T.gain[2], T.sank[2] = dict(sank=0.1), 1.0, 0.1
    clip_ = types.SimpleNamespace(comp=types.SimpleNamespace(take_interior=T))
    held = FG.interior_hold(clip_, [2])
    T.log[2], T.gain[2], T.sank[2] = dict(sank=0.9), 0.5, 0.9  # a candidate paints frame 2
    got = FG.interior_release(clip_, held)
    assert T.sank[2] == 0.1 and T.log[2]["sank"] == 0.1  # rejected: the kept render's state is back
    FG.interior_keep(clip_, got)
    assert T.sank[2] == 0.9 and T.gain[2] == 0.5  # kept
    held = FG.interior_hold(clip_, [2])  # a candidate that paints no interior (layers off), kept
    got = FG.interior_release(clip_, held)
    FG.interior_keep(clip_, got)
    assert 2 not in T.sank and 2 not in T.log


def test_closed_donor_leaves_the_model_layers():
    F, S, Fe, bank = clip()
    S[:] = face(gap=1.0)
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    assert T.paint(0, S[0][ll.LIP_IN][:, :2], (0, 0, 200, 200), 0.0, np.zeros((200, 200), np.float32))[1].max() == 0


def test_residual_is_zero_on_a_frozen_interior():
    F, S, *_ = clip(12)
    r = MR.interior_residual(F, S[:, :, :2])
    assert r["pairs"] > 0 and r["res_interior"] < 0.5


def test_reveal_sees_a_wider_mouth():
    T = np.array([face(gap=12.0)[:, :2] for _ in range(4)])
    for k in MR.STABLE:
        T[:, k] = [20 + 25 * MR.STABLE.index(k), 30 + (k % 7) * 3]
    W = T.copy()
    W[:, MR.LIPI] = np.array([face(gap=30.0)[:, :2] for _ in range(4)])[:, MR.LIPI]
    W[:, 13], W[:, 14] = [100, 70], [100, 130]
    C = {i: MR.canon(None, T)[i] for i in range(4)}
    same = MR.reveal(T, C, T, C, (200, 200))
    wide = MR.reveal(T, C, W, C, (200, 200))
    assert same["unseen_max"] == pytest.approx(0, abs=0.01) and wide["unseen_mean"] > 0.3 and wide["frames_past_take"] == 4


def _pick(gap_take, gap_plan):
    import types

    t = IT.TakeInterior.__new__(IT.TakeInterior)
    n = len(gap_take)
    t.frames, t.ap = [None] * len(gap_plan), np.asarray(gap_take, float) / 100
    bank = types.SimpleNamespace(F=np.c_[gap_take, np.full(n, 100.0)], pose=np.zeros((n, 3)))
    return t.pick(bank, np.c_[gap_plan, np.full(len(gap_plan), 100.0)], np.array([10.0, 10.0]), 2.0)


def test_interior_donor_is_never_narrower_than_the_plan():
    take = [8, 12, 20, 30, 40, 50] * 2
    js = _pick(take, [40] * 12)
    assert all(take[j] >= 0.85 * 40 for j in js)


def test_interior_donor_past_the_takes_widest_uses_the_widest():
    take = [8, 20, 50, 48] * 2
    assert all(take[j] >= 45 for j in _pick(take, [80] * 8))


def test_interior_donor_holds_through_a_wobble():
    take = list(range(30, 50)) * 2
    plan = [40 + 2 * (-1) ** i for i in range(40)]
    js = _pick(take, plan)
    assert (js[1:] != js[:-1]).sum() <= 1


def test_teeth_fit_flags_default_off():
    a = MP._parser().parse_args(BASE)
    assert a.warp_teeth_fit == "off" and a.occluder_plate is False and ML.LayerCompositor.teeth_fit is None


def test_teeth_fit_lifts_to_the_take_and_never_above():
    F = [np.full((200, 200, 3), 235, np.uint8) for _ in range(8)]
    S = np.array([face() for _ in range(8)])
    fit = IT.TeethFit(F, S)
    rgb, a = np.full((40, 40, 3), 180, np.float32), np.ones((40, 40), np.float32)
    out = fit.apply(0, rgb, a, 0.4)
    assert out.mean() > 200 and fit.log[0]["gain"] > 1
    bright = np.full((40, 40, 3), 250, np.float32)
    assert fit.apply(1, bright, a, 0.4).mean() >= 249 and IT.FIT_MAX >= fit.log[1]["gain"] >= 1


def test_clean_atlas_cuts_past_the_bite_line_and_the_grille():
    class A:
        h0, texel = -40.0, 4

    A.tex = np.zeros((400, 200, 4), np.float32)
    A.tex[..., :3], A.tex[..., 3] = 200, 1
    r_e = (0 - A.h0) * A.texel - 0.5
    A.tex[int(r_e) - 8 : int(r_e) - 6] = [90, 90, 90, 1]  # the bite line
    A.tex[int(r_e) - 6 :: 1, ::2, :3] = 120  # the grille in the strip below it
    log = IT.clean_atlas(A, 100.0, True)
    assert log["cut_cols"] > 150
    assert A.tex[int(r_e) - 3, 100, 3] < 0.1 and A.tex[int(r_e) - 30, 100, 3] > 0.9


def test_plate_removes_the_object_and_restore_puts_it_back():
    from lsdub_engine.core import occluders as OC

    f = np.full((60, 60, 3), 120, np.uint8)
    f[20:40, 25:35] = 10
    occ = np.zeros((1, 60, 60), np.float32)
    occ[0, 20:40, 25:35] = 1
    p = OC.plate([f], (occ * 255).astype(np.uint8), 1)
    assert abs(int(p[0][30, 30, 0]) - 120) < 12
    outs = [p[0].copy()]
    OC.restore(outs, [f], occ, 1)
    assert outs[0][30, 30, 0] < 30


def test_the_cap_reads_the_core_as_the_curve_does(monkeypatch):
    """pink only on the rim of the opening: the core has none, so a low take curve asks for no sink at all."""
    monkeypatch.setattr(IT, "SOFT", 0.0)
    F = [frame((90, 90, 90)) for _ in range(8)]
    for f in F:
        cv2.ellipse(f, (100, 100), (40, 20), 0, 0, 360, (60, 60, 170), 5)
    S = np.array([face() for _ in range(8)])
    feat = np.tile([40.0, 100.0], (8, 1))
    T = IT.TakeInterior(MouthBank(0, feat, np.zeros((8, 3, 2)), np.zeros((8, 3)), S[:, :, :2]), F, S, feat, np.array([10.0, 10.0]))
    assert T.frame_interior(0)[0] < 0.02  # the rim is left out of the take's reading ...
    T.curve[:] = 0.1
    T.paint(0, S[0][ll.LIP_IN][:, :2], (0, 0, 200, 200), 0.4, np.zeros((200, 200), np.float32))
    assert T.log[0]["pink"] < 0.08 and T.log[0]["sank"] == 0  # ... and from the fill's, so nothing sinks


def test_the_take_share_is_read_inside_the_rim():
    F = [frame((90, 90, 90)) for _ in range(8)]
    for f in F:
        cv2.ellipse(f, (100, 100), (40, 20), 0, 0, 360, (60, 60, 170), 3)
    S = np.array([face() for _ in range(8)])
    feat = np.tile([40.0, 100.0], (8, 1))
    T = IT.TakeInterior(MouthBank(0, feat, np.zeros((8, 3, 2)), np.zeros((8, 3)), S[:, :, :2]), F, S, feat, np.array([10.0, 10.0]))
    m = IT.poly(S[0][ll.LIP_IN][:, :2], (200, 200))
    assert IT.inner(m).sum() < m.sum()
    lab = cv2.cvtColor(F[0], cv2.COLOR_BGR2LAB).astype(np.float32)
    assert IT.pink(lab)[m].mean() > 0.05 and T.frame_interior(0)[0] == pytest.approx(float(IT.pink(lab)[IT.inner(m)].mean()))


def test_the_bisection_is_what_the_cap_runs(monkeypatch):
    F, S, Fe, bank = clip()
    T = IT.TakeInterior(bank, F, S, Fe, np.array([10.0, 10.0]))
    T.curve[:] = 0.2
    seen = []
    real = IT.least_sink
    monkeypatch.setattr(IT, "least_sink", lambda *a: seen.append(real(*a)) or seen[-1])
    T.paint(0, S[0][ll.LIP_IN][:, :2], (0, 0, 200, 200), 0.4, np.zeros((200, 200), np.float32))
    assert seen and T.log[0]["sank"] == round(seen[0], 3)


def _fit():
    return IT.TeethFit.__new__(IT.TeethFit)


def test_teeth_fit_state_round_trips():
    t = _fit()
    t.gain, t.log = {3: 1.1}, {3: dict(gain=1.1)}
    st = t.state(3)
    t.put(3, t.EMPTY)
    assert t.state(3) == t.EMPTY
    t.put(3, st)
    assert t.gain[3] == 1.1 and t.log[3] == dict(gain=1.1)


def test_a_rejected_gate_candidate_leaves_no_teeth_fit_state():
    import types
    from lsdub_engine.core import frame_gate as FG

    t = _fit()
    t.gain, t.log = {2: 1.1}, {2: dict(gain=1.1)}
    c = types.SimpleNamespace(comp=types.SimpleNamespace(take_interior=None, teeth_fit=t))
    held = FG.interior_hold(c, [2])
    assert 2 not in t.gain  # cleared: a candidate that paints none leaves none
    t.gain[2], t.log[2] = 1.3, dict(gain=1.3)
    got = FG.interior_release(c, held)
    assert t.gain[2] == 1.1
    FG.interior_keep(c, got)
    assert t.gain[2] == 1.3


class _Spy:
    """a stand-in for the state holders: try_run and repair_window must clear them before a candidate and put them back after."""

    EMPTY = (None,)

    def __init__(self, frames):
        self.v = {f: "kept" for f in frames}
        self.during = []

    def state(self, i):
        return (self.v.get(i),)

    def put(self, i, st):
        self.v.pop(i, None) if st[0] is None else self.v.__setitem__(i, st[0])

    def paint(self, i):
        self.during.append(self.v.get(i))
        self.v[i] = "cand"


def _render_clip(spy):
    import types

    def render_frame(t, ov, fold=False):
        spy.paint(t)
        return "o", "m", dict(fold_det=1.0), np.zeros(2)

    return types.SimpleNamespace(
        comp=types.SimpleNamespace(take_interior=spy),
        render_frame=render_frame,
        gate_ov={},
        outs={},
        masks={},
        P_out={},
        log={},
        n=9,
    )


def _ck():
    import types

    return types.SimpleNamespace(
        update=lambda *a, **k: None,
        metrics=lambda: dict(),
        verdict=lambda M, ref: dict(frames=[dict(verdict="PASS", sev=0.0)] * 9),
    )


def test_try_run_and_commit_run_carry_the_state(monkeypatch):
    from lsdub_engine.core import frame_gate as FG

    monkeypatch.setattr(FG, "snapshot", lambda ck, run: None)
    monkeypatch.setattr(FG, "plan_entry", lambda *a, **k: {})
    spy = _Spy([2, 3])
    c, ck = _render_clip(spy), _ck()
    r = FG.try_run(c, ck, [2, 3], {2: ({}, {}), 3: ({}, {})}, {})
    assert spy.during == [None, None]  # the candidate started clean
    assert spy.v == {2: "kept", 3: "kept"}  # rejected until committed
    FG.commit_run(c, ck, r, "x", 0)
    assert spy.v == {2: "cand", 3: "cand"}


def test_repair_window_carries_the_state(monkeypatch):
    import types
    from lsdub_engine.core import frame_gate as FG

    frames = [3, 4]
    spy = _Spy(frames)
    c, ck = _render_clip(spy), _ck()
    c.P_out = {f: np.zeros(2) for f in range(9)}
    c.patched = lambda *a: __import__("contextlib").nullcontext()
    c.pl = types.SimpleNamespace(rigid3=lambda f, C: C)
    c.S = np.zeros((9, 2))
    C = np.zeros((9, 478, 3))
    for k, v in dict(
        closures=lambda clip: set(),
        window_tracks=lambda clip: dict(C=C),
        interp_params=lambda clip, fr, x, y, tr, deg: {f: dict(Cres=np.zeros((len(FG.mw.CTRL), 3))) for f in fr},
        snapshot=lambda ck, fr: None,
        restore=lambda ck, s: None,
        model_shape=lambda clip, X: np.zeros((1, len(FG.mw.CTRL), 3)),
        fold_guard=lambda *a, **k: 0,
        residual_for=lambda *a, **k: 0,
        plan_entry=lambda *a, **k: {},
        structure_ok=lambda *a, **k: True,
    ).items():
        monkeypatch.setattr(FG, k, v)
    e = iter([1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    monkeypatch.setattr(FG, "slosh_energy", lambda *a: next(e))
    out = FG.repair_window(
        c, ck, dict(frames=(2, 5)), {}, dict(frames=[dict(verdict="PASS", sev=0.0)] * 9), {"slosh": 0}, 0, [], ctx={}, cands=()
    )
    assert out is not None
    assert spy.during == [None, None, None, None]  # each try (deg 1, 2) started clean
    assert spy.v == {3: "cand", 4: "cand"}  # the kept try's state stayed


def test_boil_all_and_common_read_the_same_frames_across_renders(monkeypatch):
    n = 12
    T = np.zeros((n, 478, 2))
    T[:, 61], T[:, 291] = [50, 100], [150, 100]
    monkeypatch.setattr(MR, "frame_stats", lambda f, t: dict(w=100.0, ap=float(f[0, 0, 0])))
    monkeypatch.setattr(MR, "boil", lambda a, b, c, pa, pb, pc: (float(b[0, 0, 1]), float(b[0, 0, 1]), 0.0))

    def clip_(aps, val):
        F = [np.zeros((4, 4, 3), np.uint8) for _ in range(n)]
        for i, f in enumerate(F):
            f[0, 0, 0], f[0, 0, 1] = aps[i], val
        return F, T

    a = clip_([0] * 6 + [50] * 6, 5)  # held early
    b = clip_([0] * 3 + [50] * 3 + [0] * 6, 7)  # held elsewhere as well
    got = MR.boil_common([a, b])
    assert got["n"] == len({1, 2, 3, 4, 7, 8, 9, 10} & {1, 4, 7, 8, 9, 10})
    assert got["cav"] == [5.0, 7.0] and got["lip"] == [5.0, 7.0]
    assert MR.hold_frames(T, np.full(n, 100.0), np.r_[np.zeros(6), np.full(6, 50.0)]) == {1, 2, 3, 4, 7, 8, 9, 10}
    assert MR.boil_common([a, clip_([0, 50] * 6, 1)])["n"] == 0
