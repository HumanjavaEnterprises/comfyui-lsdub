# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v12 --warp-reg single (mouth_layers.LayerCompositor.plan_single): ONE reference for the mouth's interior -- the upper arch on
the skull pose only, the lower arch + tongue on the skull pose + one low-passed hinge angle, the tongue a slow function of
it, no pins, no per-frame exposure kick, gum guards; the default (v11) placement untouched. Synthetic, CPU only."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lsdub_engine.core import teeth_layer as tl, lip_layers as ll, mouth_paint
from test_mouth_layers import _comp
from test_teeth_layer import _P, _frame


def _seq(n, rng, gap=20.0, lo_noise=1.5, base=None):
    """n planned outlines: the same mouth, the lower inner lip (14 + the lower inner loop) jittered up / down by lo_noise px."""
    P = []
    for i in range(n):
        g = gap if base is None else base[i]
        p = _P(gap=g)
        dy = rng.normal(0, lo_noise)
        low = [k for k in ll.LIP_IN if p[k, 1] > p[13, 1] + 0.5 * g] + [14]
        p[low, 1] += dy
        P.append(p)
    return P


def test_upper_arch_rides_the_skull_only():
    n = 12
    comp, layer, frames, S = _comp(gaps=tuple([20.0] * n))
    rng = np.random.default_rng(0)
    layer.pin[:] = rng.normal(0, 3.0, (n, 2))                  # a v11 lip-root pin would move the arches every frame
    P = _seq(n, rng, base=np.linspace(12, 30, n))
    comp.plan_single(P)
    assert comp.pin_mode == 'single' and np.all(comp.dth == 0) and np.all(comp.pin_lo == 0)
    layer.pin[:] = 0.0                                         # (the caller never sets the pin in this mode: setup_layers skips set_pin)
    ups = []
    for i in range(n):
        comp.layers(frames[i], i, S[i], P[i], float(np.linalg.norm(P[i][13, :2] - P[i][14, :2])))
        ups.append(comp.reg[i]['up'])
    ups = np.array(ups)
    # identity skull pose: the upper incisal edge does not move at all in the skull frame, whatever the lips' plan does
    assert np.abs(ups - ups[0]).max() < 1e-9
    assert all(np.allclose(comp.reg[i]['kick'], 0) for i in range(n)) and all(np.allclose(comp.reg[i]['chin'], 0) for i in range(n))


def test_lower_arch_on_one_low_passed_hinge_angle():
    n = 40
    comp, layer, frames, S = _comp(gaps=tuple([20.0] * n))
    rng = np.random.default_rng(1)
    base = 20.0 + 8.0 * np.sin(np.arange(n) / 4.0)            # a slow opening / closing ...
    P = _seq(n, rng, base=base, lo_noise=2.0)                  # ... with a jittery planned lower lip
    comp.plan_single(P, sigma=1.0)
    tgt, th = np.asarray(layer.jaw_target), np.asarray(layer.jaw_plan)
    j = lambda x: np.percentile(np.abs(np.diff(x, 2)), 95)
    assert j(th) < 0.6 * j(tgt)                                # the jaw is heavy: the frame-to-frame jitter is filtered out
    assert np.corrcoef(th, tgt)[0, 1] > 0.8                    # ... but it still follows the planned opening
    los = []
    for i in range(n):
        comp.layers(frames[i], i, S[i], P[i], float(base[i]))
        r = comp.reg[i]
        assert abs(r['jaw'] - np.degrees(th[i])) < 1e-2 and abs(r['dth']) < 1e-9 and np.allclose(r['kick'], 0)
        los.append(r['lo'])
    los = np.array(los)
    # a rotation about the lateral hinge axis: the lower incisal edge moves on the arc (no lateral motion at the midline)
    assert np.ptp(los[:, 1, 0]) < 1e-6 and np.ptp(los[:, 1, 1]) > 0.02
    # the bite: never through the upper teeth
    assert (th >= comp.bite_angle() - 1e-9).all()


def test_tongue_is_a_slow_function_of_the_jaw():
    n = 30
    comp, layer, frames, S = _comp(gaps=tuple([20.0] * n))
    rng = np.random.default_rng(2)
    base = 6.0 + 10.0 * (np.arange(n) % 10 >= 5)               # steps: open / closed every 5 frames (the take's p95 gap: 20)
    P = _seq(n, rng, base=base, lo_noise=1.0)
    comp.plan_single(P, sigma=1.0, tongue_sigma=2.0)
    assert comp.prepared and np.all(comp.tg == 0)              # no per-frame vowel re-fit
    from scipy.ndimage import gaussian_filter1d
    op = np.asarray(comp.open_s)
    tht = gaussian_filter1d(np.asarray(layer.jaw_plan), 2.0, mode='nearest')
    o = np.argsort(tht)
    assert (np.diff(op[o]) >= -1e-9).all()  # the opening share is a (monotone) function of the jaw angle ...
    assert np.ptp(op) > 0.05 and np.percentile(np.abs(np.diff(op)), 95) < 0.5     # ... low-passed (no per-frame jumps)
    tips = []
    for i in range(n):
        comp.layers(frames[i], i, S[i], P[i], float(base[i]))
        r = comp.reg[i]
        tips.append(np.subtract(r['tip'], r['lo'][1]))
    tips = np.array(tips)
    # the tongue tip rides the jaw: its offset from the lower incisal edge changes only with the slow opening share
    assert np.percentile(np.abs(np.diff(tips, axis=0)), 95) < 0.01


def test_gum_floor_and_guard():
    n = 20
    comp, layer, frames, S = _comp(gaps=tuple([20.0] * n))
    rng = np.random.default_rng(3)
    P = _seq(n, rng, lo_noise=0.0)
    io = layer.iod
    P[10] = P[10].copy()
    low = [k for k in ll.LIP_IN if P[10][k, 1] > P[10][13, 1] + 5] + [14]
    P[10][low, 1] += 0.25 * io  # one frame: the lip drops far
    comp.plan_single(P, sigma=1.0, lo_limit=0.09)
    th = np.asarray(layer.jaw_plan)
    # the lower lip never sits more than lo_limit IOD below the lower incisal edge (no gum-less gap under the lower teeth)
    for i in range(n):
        R = tl.rot_axis(layer.u, th[i])
        e2, _ = comp.proj(i, True, R)(comp.lo.o[None])
        assert float(P[i][14, 1] - e2[0, 1]) <= 0.09 * io + 0.5, i
    assert comp.single_info()['raised_frames'] >= 0 and comp.single['floor_frames'] >= 1
    # the upper gum guard: a planned upper lip lifted past the upper arch's band extends the band (gum) upward
    comp2, layer2, frames2, S2 = _comp(gaps=tuple([20.0] * n))
    P2 = _seq(n, rng, lo_noise=0.0)
    P2[5] = P2[5].copy()
    upl = [k for k in ll.LIP_IN if P2[5][k, 1] < P2[5][14, 1] - 5] + [13]
    P2[5][upl, 1] -= 0.45 * io
    h0, Th = comp2.up.h0, comp2.up.Th
    comp2.plan_single(P2)
    assert comp2.up.h0 < h0 and comp2.up.Th > Th and comp2.up.tex.shape[0] == comp2.up.Th
    assert comp2.single['up_band_ext_iod'] > 0.1
    assert len(comp2.up.h) == comp2.up.Th and abs(comp2.up.h[0] - (comp2.up.h0 + 0.5 / comp2.up.texel)) < 1e-9


def test_default_is_v11_and_flags_parse(capsys):
    comp, layer, frames, S = _comp()
    P = _P(gap=20.0)
    assert getattr(comp, 'single', None) is None              # nothing changes unless plan_single is called
    comp.layers(frames[2], 2, S[2], P, 20.0)
    assert 'single' not in comp.info()
    try: mouth_paint.main(['--help'])
    except SystemExit: pass
    o = capsys.readouterr().out
    for f in ('--warp-reg', '--warp-reg-jaw-sigma', '--warp-reg-tongue-sigma', '--warp-reg-lo-limit'):
        assert f in o


def test_one_global_residual_moves_the_whole_interior_together(capsys):
    n = 10
    comp, layer, frames, S = _comp(gaps=tuple([20.0] * n))
    rng = np.random.default_rng(4)
    pin = layer.set_pin(frames, S, idx=tl.NOSE)
    assert pin.shape == (n, 2)          # (the nose region's own flow; a still synthetic take: ~0)
    assert tl.ROOT != tl.NOSE and not set(tl.NOSE) & {0, 13, 14, 17, 37, 267}         # no lip point in the nose residual
    P = _seq(n, rng, lo_noise=1.0)
    comp.plan_single(P, pin='nose')
    assert comp.single['pin'] == 'nose'
    X_up = comp.up.points(np.array([0.0]), np.zeros(1))
    X_lo = comp.lo.points(np.array([0.0]), np.zeros(1))
    def edges(i):
        R = tl.rot_axis(layer.u, layer.jaw_plan[i])
        return comp.proj(i, False)(X_up)[0][0], comp.proj(i, True, R)(X_lo)[0][0]
    layer.pin[:] = 0.0
    e0 = [edges(i) for i in range(n)]
    layer.pin[:] = rng.normal(0, 2.0, (n, 2))                                         # one residual per frame ...
    for i in range(n):
        u1, l1 = edges(i)
        # ... moves both arches by exactly that residual: their registration to each other (the bite) is the hinge's alone
        assert np.allclose(u1 - e0[i][0], layer.pin[i]) and np.allclose(l1 - e0[i][1], layer.pin[i])
        comp.layers(frames[i], i, S[i], P[i], 20.0)
        assert np.allclose(comp.reg[i]['pin_up'], np.round(layer.pin[i], 2))
    try: mouth_paint.main(['--help'])
    except SystemExit: pass
    assert '--warp-reg-pin' in capsys.readouterr().out


# ------------------------------------------------- v12 --warp-arch-build v12: the plate build ("teeth inside teeth") --
def _stacked(rows2=True, Th=60, Tw=80, edge=50, band=14):
    """a synthetic upper plate (BGRA): gum above, a tooth band ending at the incisal edge row; rows2 = a SECOND tooth band
    under a dark bite line above the edge (what a clenched-teeth harvest put on the plate)."""
    T = np.zeros((Th, Tw, 4), np.float32)
    T[..., :3] = (60, 60, 120)
    T[..., 3] = 1.0          # gum (BGR)
    T[edge - 2 * band - 3:edge, :, :3] = 235.0                                                    # teeth
    if rows2: T[edge - band - 3:edge - band, :, :3] = 40.0                                         # the bite line between two rows
    T[edge:, :, 3] = 0.0                                                                           # below the incisal edge: the cavity
    return T


def test_double_edge_ruler_counts_stacked_rows():
    from lsdub_engine.audits import layer_reg as lr
    one = lr.plate_edges(_stacked(False), True, iod=100.0, texel=1.0, h0=-50.0)
    two = lr.plate_edges(_stacked(True), True, iod=100.0, texel=1.0, h0=-50.0)
    assert one['all']['one'] == 1.0 and one['all']['two_plus'] == 0.0
    assert two['all']['two_plus'] == 1.0 and two['all']['mean'] >= 2.0
    lo = lr.plate_edges(_stacked(True)[::-1].copy(), False, iod=100.0, texel=1.0, h0=-10.0)      # the lower plate reads bottom-up
    assert lo['all']['two_plus'] == 1.0


def test_v12_build_harvests_an_open_bite_not_a_clenched_smile():
    # frames: 0 / 1 clenched smiles (the largest tooth area, both rows, no dark gap), 2 an open bite (the upper row, a gap)
    n = 4
    S = np.array([_P(gap=g) for g in (20.0, 20.0, 20.0, 20.0)])
    frames = []
    for k in range(n):
        img = _frame(_P(gap=20.0), tooth_h=9)
        if k in (0, 1, 3):           # clenched: the lower row of teeth right under the upper one (no cavity between)
            P = _P(gap=20.0)
            c = P[[13, 14], :2].mean(0)
            for j in range(-3, 3):
                x0 = int(round(c[0] + j * 10))
                img[int(c[1]) - 2:int(c[1]) + 10, x0:x0 + 8] = (235, 240, 245)
        frames.append(img)
    from test_teeth_layer import StubPose
    poses = StubPose(np.ones(n), np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)))
    F = np.stack([np.full(n, 20.0), np.full(n, 64.0)], 1)
    v11 = tl.TeethLayer(frames, S, poses, S, F)
    v12 = tl.TeethLayer(frames, S, poses, S, F, build='v12')
    assert v12.build_log['up']['h_up'] == 2 and v12.build_log['up']['apart_dark_iod'] > 0.02
    assert v11.build == 'v11' and v11.build_log is None


def test_v12_lip_band_takes_the_lip_rim_colour_not_the_gum():
    comp, layer, frames, S = _comp(gaps=(4.0, 20.0, 28.0, 12.0))
    io = layer.iod
    band = comp.up
    th = getattr(band, 'tooth_h_c', 0.11 * io)
    hm = np.array([[-0.5 * th, -th - 0.05 * io, -th - 0.2 * io, np.nan]], np.float32)       # tooth | just past the crown | deep | no band
    rgb = np.full((1, 4, 3), 200.0, np.float32)
    col = np.array([40.0, 50.0, 180.0])
    out = comp.lip_band(rgb, hm, band, True, col)
    assert np.allclose(out[0, 0], 200.0) and np.allclose(out[0, 3], 200.0)                    # the crown and no-band pixels untouched
    assert np.abs(out[0, 2] - col).max() < 1e-3                                              # deep in the band: the lip's own colour
    assert np.all(out[0, 1] <= out[0, 2] + 1e-3) and np.all(out[0, 1] >= 0.8 * out[0, 2] - 1)      # a little darker toward the teeth
    assert comp.lip_band(rgb, hm, band, True, None) is rgb                                   # no rim colour measured: unchanged
    # the rim colour of a frame: its planned lips' inner edge (here the synthetic red lip)
    P = _P(gap=20.0)
    x0, y0, w, h = comp._box(2, S[2], P, frames[2].shape[1], frames[2].shape[0])
    sl = np.s_[y0:y0 + h, x0:x0 + w]
    rims = comp.rim_colours(frames[2][sl], P[ll.LIP_IN, :2] - [x0, y0], P[ll.LIP_OUT, :2] - [x0, y0], io)
    from test_teeth_layer import LIP
    for k in ('up', 'lo'):
        r = np.asarray(rims[k], float)
        assert np.all(r <= np.array(LIP, float) + 1) and np.all(r >= (1 - comp.rim[0]) * np.array(LIP, float) - 1), (k, r)
    # the build logs the ruler per frame: the band right under the lips vs their rim
    v12, *_ = _comp(gaps=(4.0, 20.0, 28.0, 12.0), build='v12')
    v12.composite(frames[2], 2, S[2], P, 20.0)
    lb = v12.reg[2].get('lipband')
    assert lb is not None and set(lb) == {'up', 'lo'}
