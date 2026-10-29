# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v11 inter-layer registration: the per-frame log of the layer stack (mouth_layers.reg_entry -> warp.json frames[i].reg) and
the ruler on it (lsdub/audits/layer_reg.py: offsets of the teeth / cavity / tongue vs the lips in the skull-stabilised
frame; jaw residual, jitter, the v9 gate's windowed slosh, the texture drift), plus the interior clip
(--warp-interior-clip: no cavity or tooth pixel outside the warped lips). Synthetic, CPU only."""
import os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
from lsdub_engine.audits import layer_reg as lr
from lsdub_engine.core import teeth_layer as tl, lip_layers as ll
from test_mouth_layers import _comp
from test_teeth_layer import _P, _frame, SKIN, LIP


def _frames(off):
    """warp.json-like frames whose reg anchors produce the given up_lip offsets (n x 2), everything else static."""
    F = []
    for o in off:
        F.append(dict(reg=dict(up=[[-0.18, 0.0], [o[0], o[1]], [0.18, 0.0]], lo=[[-0.18, 0.2], [0.0, 0.2], [0.18, 0.2]], cav=[0.0, 0.1], open=[0.0, 0.1],
                               inner=[[0.0, 0.0], [0.0, 0.2], [-0.25, 0.1], [0.25, 0.1]], tip=[0.0, 0.25], jaw=5.0, pin_up_iod=[0.0, 0.0])))
    return F


def test_slosh_counts_back_and_forth_not_net_motion():
    n = 40
    t = np.arange(n)
    still = np.zeros((n, 2))
    drift = np.stack([0.002 * t, 0 * t], 1)
    wobble = np.stack([0.01 * (-1) ** t, 0 * t], 1)
    E0, Ed, Ew = (lr.slosh(x, 4) for x in (still, drift, wobble))
    assert np.nanmax(E0) == 0 and np.nanmax(Ed) < 1e-12                # one direction: no slosh
    assert np.nanmean(Ew) > 0.015                                        # +-0.01 IOD every frame: ~0.02 IOD / frame of slosh
    r = lr.measure(_frames(wobble), texture=False)
    assert r['offsets']['up_lip']['slosh_mean'] > 0.015 and r['offsets']['lo_lip']['slosh_mean'] == 0
    # the wobbling upper edge sloshes against the upper lip AND against the lower edge (the bite); nothing else moves
    assert abs(r['summary']['slosh_sum'] - r['offsets']['up_lip']['slosh_mean'] - r['offsets']['bite']['slosh_mean']) < 1e-9
    assert r['offsets']['cav_open']['slosh_mean'] == 0 and r['offsets']['tip_lo']['slosh_mean'] == 0


def test_jaw_residual_removes_what_the_jaw_explains():
    n = 30
    jaw = np.linspace(0, 12, n)
    o = np.stack([0.01 * jaw, -0.02 * jaw + 0.003], 1)
    assert np.nanmax(np.abs(lr.jaw_residual(o, jaw))) < 1e-9
    o2 = o.copy()
    o2[10, 1] += 0.05
    res = lr.jaw_residual(o2, jaw)
    assert np.nanargmax(np.linalg.norm(res, axis=1)) == 10


def test_gate_report_flags_a_sloshing_window_only():
    n = 40
    off = np.zeros((n, 2))
    off[15:23, 1] = 0.03 * (-1) ** np.arange(8)
    g = lr.gate_report(_frames(off), 4, floor=0.01)
    assert g['repaired'] is False and len(g['windows']) == 1 and g['windows'][0]['top'] == 'up_lip'
    f0, f1 = g['windows'][0]['frames']
    assert f0 <= 15 and f1 >= 22


def test_ncc_shift_recovers_a_known_texture_shift():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random((80, 120)).astype(np.float32) * 255, (0, 0), 1.2)
    M = np.float32([[1, 0, 2.0], [0, 1, -1.0]])
    moved = cv2.warpAffine(base, M, (120, 80), borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((80, 120), bool)
    mask[25:55, 30:90] = True
    d, q = lr.ncc_shift(moved, base, mask, 4)
    assert np.allclose(d, [2.0, -1.0], atol=0.15) and q > 0.9


def test_reg_entry_follows_every_layer_where_it_was_rendered():
    comp, layer, frames, S = _comp()
    P = _P(gap=20.0)
    take = _frame(P)
    comp.layers(take, 2, S[2], P, 20.0)
    r0 = comp.reg[2]
    for k in ('up', 'lo', 'cav', 'open', 'inner', 'contour', 'tip', 'jaw', 'pin_up', 'pin_up_iod', 'chin', 'kick', 'pose', 'stab'):
        assert k in r0, k
    io = layer.iod
    # the inner-lip top / bottom in the skull frame = the landmarks 13 / 14 (identity pose: canonical = frame px)
    org = np.asarray(comp.ref[:2], float)
    assert np.allclose(r0['inner'][0], (P[13, :2] - org) / io, atol=1e-3) and np.allclose(r0['inner'][1], (P[14, :2] - org) / io, atol=1e-3)
    # a lip-root pin of +3 px down moves the upper edge (and the lower: the pin carries both arches) by 3 px in the skull
    # frame, and nothing of the lips; pin_up_iod logs it
    layer.pin[2] = np.array([0.0, 3.0])
    comp.layers(take, 2, S[2], P, 20.0)
    r1 = comp.reg[2]
    assert np.allclose(np.subtract(r1['up'], r0['up'])[:, 1], 3.0 / io, atol=1e-3) and np.allclose(r1['inner'], r0['inner'])
    assert np.allclose(r1['pin_up_iod'], [0.0, 3.0 / io], atol=1e-4)
    layer.pin[2] = np.zeros(2)
    # an opening jaw drops the lower incisal edge (the bite grows)
    comp.layers(take, 2, S[2], _P(gap=28.0), 28.0)
    r2 = comp.reg[2]
    assert r2['jaw'] > r0['jaw'] and r2['lo'][1][1] > r0['lo'][1][1]


def test_interior_clip_keeps_the_interior_inside_the_warped_lips():
    comp, layer, frames, S = _comp(interior_clip=True)
    base, *_ = _comp()
    P = _P(gap=20.0)
    take = _frame(P)
    io = layer.iod
    # 1. a correct placement: the clip changes nothing that matters (the opening is enclosed by the lips)
    L1, box = comp.layers(take, 2, S[2], P, 20.0)
    L0, _ = base.layers(take, 2, S[2], P, 20.0)
    x0, y0, w, h = box
    tl.poly_mask(P[ll.LIP_IN, :2] - [x0, y0], (h, w), 0.0) & ~tl.poly_mask(P[ll.LIP_IN, :2] - [x0, y0], (h, w), -0.15)
    core = tl.poly_mask(P[ll.LIP_IN, :2] - [x0, y0], (h, w), -0.15)
    assert np.abs(L1['L2'][1][core] - L0['L2'][1][core]).max() < 0.02 and comp.reg[2]['clip'] < 0.05
    out1, _ = comp.composite(take, 2, S[2], P, 20.0)
    out0, _ = base.composite(take, 2, S[2], P, 20.0)
    assert np.abs(out1.astype(int) - out0.astype(int)).mean() < 0.5
    # 2. the planned mouth placed 0.45 IOD above the real lips (what a false face fit did to mv30 S3): the warped frame
    # (here: the take itself) has no lips around that opening, so no cavity / tooth pixel is drawn there; the face shows
    Pu = P.copy()
    Pu[:, 1] -= 0.45 * io
    Lu, bu = comp.layers(take, 2, S[2], Pu, 20.0)
    xu, yu, wu, hu = bu
    lips_real = tl.poly_mask(P[ll.LIP_OUT, :2] - [xu, yu], (hu, wu), 0.0)
    for k in ('L2', 'L3', 'L4'): assert Lu[k][1][~lips_real].max() < 0.02, k
    assert comp.reg[2]['clip'] > 0.9
    outu, _ = comp.composite(take, 2, S[2], Pu, 20.0)
    bu_out = base.composite(take, 2, S[2], Pu, 20.0)[0]
    dark = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0] < 60
    false_open = tl.poly_mask(Pu[ll.LIP_IN, :2], take.shape, 0.0)
    assert dark(bu_out)[false_open].mean() > 0.5 and dark(outu)[false_open].mean() < 0.02       # v9: a cavity there; v11: none


# ------------------------------------------------------------------------------ v11: the corner cover (step 4) --
def test_corner_points_push_only_the_corners():
    P = _P(gap=10.0)
    q = P[ll.LIP_OUT, :2]
    c = q.mean(0)
    q2 = ll.corner_points(q, c, (10.0, 4.0))
    iL, iR, iT, iB = ll.LIP_OUT.index(61), ll.LIP_OUT.index(291), ll.LIP_OUT.index(0), ll.LIP_OUT.index(17)
    assert np.allclose(q2[iL] - q[iL], [-10.0, 0.0], atol=1e-6) and np.allclose(q2[iR] - q[iR], [4.0, 0.0], atol=1e-6)
    assert np.allclose(q2[iT], q[iT]) and np.allclose(q2[iB], q[iB])
    m0 = ll.lip_erase_mask(_frame(P), P, 0.22, key=False)
    m1 = ll.lip_erase_mask(_frame(P), P, 0.22, key=False, corner=(10.0, 4.0))
    assert m1.sum() > m0.sum() and (m1 | m0).sum() == m1.sum()              # the cover only grows, at the corners


def test_corner_cover_fills_old_mouth_moved_past_a_wider_corner():
    P_take = _P(gap=8.0, hw=34)
    P_new = _P(gap=20.0, hw=46)                # the plan: wider (corners +12 px) and more open
    _frame(P_take)
    warped = _frame(P_new)
    blob = (int(round(P_new[61, 0] - 6)), int(round(P_new[61, 1] + 3)))     # old lip the warp dragged past the new left corner
    cv2.circle(warped, blob, 4, LIP, -1)
    lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    lip_lab = lab(np.uint8([[LIP]]))[0, 0]
    skin_lab = lab(np.uint8([[SKIN]]))[0, 0]
    near = lambda im, ref: np.linalg.norm(lab(im)[blob[1] - 1:blob[1] + 2, blob[0] - 1:blob[0] + 2] - ref, axis=2).max()
    off, layer, frames, S = _comp(cover_mode='fill')
    out0, _ = off.composite(warped, 2, P_take, P_new, 20.0)
    on, *_ = _comp(cover_mode='fill', cover_corners=1.0)
    out1, _ = on.composite(warped, 2, P_take, P_new, 20.0)
    assert near(out0, lip_lab) < 20                                         # v9: the old lip stays at the corner
    assert near(out1, skin_lab) < 25 and on.corner_log[2] > 0               # v11: filled with skin
    gl, gr = on.corner_growth(2, P_take[:, :2], P_new[:, :2], layer.iod)
    assert gl > 12 and gr > 12                                              # grows with the corners' push (+ half the opening)
    # off by default: the same frame as without the option
    base, *_ = _comp(cover_mode='fill')
    assert np.array_equal(base.composite(warped, 2, P_take, P_new, 20.0)[0], out0)


def test_frames_shown_as_the_take_are_not_measured():
    n = 30
    off = np.zeros((n, 2))
    off[10:20, 1] = 0.03 * (-1) ** np.arange(10)       # a slosh inside a skipped vocalisation
    F = _frames(off)
    for i in range(10, 20): F[i]['vocal_s'] = 0.0                                       # (the take's own frames: no layers shown)
    r = lr.measure(F, texture=False)
    assert r['offsets']['up_lip']['slosh_mean'] == 0 and r['offsets']['up_lip']['n'] == n - 10
