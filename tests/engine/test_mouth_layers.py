# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v7 layered mouth compositor (mouth_layers.py, --warp-compose layers). Synthetic, CPU only."""
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot

from lsdub_engine.core import mouth_layers as ml, teeth_layer as tl, lip_layers as ll, mouth_paint
from test_teeth_layer import _P, _frame, _layer, _skin_seq, SKIN, LIP


def _comp(gaps=(4.0, 20.0, 28.0, 12.0), poses=None, **kw):
    layer, frames, S = _layer(gaps, poses)
    st = ll.lip_stats(frames[2], S[2][:, :2])
    return ml.LayerCompositor(layer, frames, S, st[1:3], st[3:5], **kw), layer, frames, S


def _bright(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0] > 200


def test_lips_always_occlude_teeth():
    comp, layer, frames, S = _comp()
    for gap_t in (6.0, 20.0, 30.0):
        P = _P(gap=gap_t)
        take = _frame(P)                         # the warped frame: its own lips at the target shape
        L, box = comp.layers(take, 2, S[2], P, gap_t)
        x0, y0, w, h = box
        opening = tl.poly_mask(P[ll.LIP_IN, :2] - [x0, y0], (h, w), 0.0)
        lips = tl.poly_mask(P[ll.LIP_OUT, :2] - [x0, y0], (h, w), 0.0) & ~tl.poly_mask(P[ll.LIP_IN, :2] - [x0, y0], (h, w), 0.0, 2.0)
        # the arches themselves are bigger than the opening (the whole band is drawn) ...
        assert (comp._full['L4'] > 0.5).sum() > (L['L4'][1] > 0.5).sum()
        # ... but only the part inside the expanded opening reaches the composite, and on the lips the lips layer is opaque
        far = ~tl.poly_mask(P[ll.LIP_IN, :2] - [x0, y0], (h, w), 0.0, 0.03 * layer.iod + 4)
        for k in ('L2', 'L3', 'L4'): assert (L[k][1][far] < 1e-3).all(), k
        assert (L['L5'][1][lips] > 0.99).all()
        out, m = comp.composite(take, 2, S[2], P, gap_t)
        sub = out[y0:y0 + h, x0:x0 + w]
        # no tooth pixel ever lands on the lips: they are the warped frame's own lip pixels, unchanged
        assert not _bright(sub)[lips].any()
        body = lips & ~tl.poly_mask(P[ll.LIP_IN, :2] - [x0, y0], (h, w), 0.0, 0.05 * layer.iod)    # away from the inner rim
        assert np.abs(sub[body].astype(int) - take[y0:y0 + h, x0:x0 + w][body].astype(int)).max() <= 1
        # the rim only ever darkens the lip (it is the lip turning inward), never brightens it toward a tooth
        assert (sub[lips].astype(int).sum(1) <= take[y0:y0 + h, x0:x0 + w][lips].astype(int).sum(1) + 3).all()
        if gap_t >= 20: assert _bright(sub)[opening].sum() > 20      # and the opening shows teeth


def test_cover_layer_removes_the_old_mouth():
    comp, layer, frames, S = _comp()
    P_old = _P(gap=28.0)
    take = _frame(P_old)                      # the take: a big open mouth
    L, box = comp.layers(take, 2, P_old, P_old, 28.0)
    x0, y0, w, h = box
    rgb, a = L['L1']
    take[y0:y0 + h, x0:x0 + w]
    lipm = tl.poly_mask(P_old[ll.LIP_OUT, :2] - [x0, y0], (h, w), 0.0)
    assert (a[lipm] > 0.99).all()                                   # covers the whole old mouth (lips + interior)
    lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    skin = cv2.cvtColor(np.uint8([[SKIN]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    assert np.abs(lab[lipm] - skin).max(1).max() < 25               # with skin: no lip red, no tooth, no cavity left
    # the warp could not shrink the old lips (warped frame = the take), the target mouth is small: outside the new outer
    # lip the old lips are dropped and the cover shows there
    P_new = _P(gap=6.0, hw=30)
    out, _ = comp.composite(take, 2, P_old, P_new, 6.0)
    ring = tl.poly_mask(P_old[ll.LIP_OUT, :2], take.shape, 0.0) & ~tl.poly_mask(P_new[ll.LIP_OUT, :2], take.shape, 0.0, 0.03 * layer.iod + 3)
    lab_o = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    lip_lab = cv2.cvtColor(np.uint8([[LIP]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    left = np.linalg.norm(lab_o[ring] - lip_lab, axis=1) < 20
    assert left.mean() < 0.05, left.mean()


def test_arch_shows_more_teeth_at_wider_opening_and_on_turns():
    comp, layer, frames, S = _comp()
    band = comp.up
    c = layer.S[2][[13, 14], :2].mean(0)
    def visible(i, half_w, half_h=18.0, pose=None):
        box = (0, 0, 320, 260)
        f = comp.proj(i, False) if pose is None else pose
        rot = layer.pl.R[i] if pose is None else pose.rot
        _, al, am = band.render(f, box, rot, amap=True)
        yy, xx = np.mgrid[0:260, 0:320]
        op = ((xx - c[0]) / half_w) ** 2 + ((yy - (c[1] + 8)) / half_h) ** 2 <= 1
        a = am[op & np.isfinite(am)]
        return a
    # wider opening: a wider section of the arch, more of its length, is exposed (incisors -> canines -> premolars)
    span = [np.ptp(visible(2, hw)) for hw in (15.0, 30.0, 45.0)]
    assert span[0] < span[1] < span[2], span
    # a turned head: within the same opening, the side turning toward the camera shows teeth further back along the arch
    R = Rot.from_euler('y', 25, degrees=True).as_matrix()
    cen = layer.Cm[[13, 14]].mean(0)
    class P_:      # the head yawed 25 degrees about the mouth centre (orthographic)
        rot = R
        def __call__(self, X):
            Y = (np.asarray(X) - cen) @ R.T + cen
            return Y[:, :2], Y[:, 2]
    a0, a1 = visible(2, 30.0), visible(2, 30.0, pose=P_())
    assert max(a1.max(), -a1.min()) > max(a0.max(), -a0.min()) + 2.0, (a0.min(), a0.max(), a1.min(), a1.max())


def test_full_arch_is_built_beyond_the_harvested_teeth():
    comp, layer, frames, S = _comp()
    for b in (comp.up, comp.lo):
        assert (b.tex[..., 3].max(0) > 0.5).mean() > 0.8              # the whole arch has teeth, not the middle 2-4
    assert (comp.up.src == 3).any() or (comp.up.src == 2).any()      # completed beyond what the harvest showed
    # the extended back teeth recede into shadow
    L = cv2.cvtColor(np.clip(comp.up.tex[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    r = int((0 - comp.up.h0) * comp.up.texel) - 3
    cols = np.arange(comp.up.Tw)
    assert L[r, cols[np.abs(comp.up.a) > 0.85 * comp.up.A]].mean() < L[r, cols[np.abs(comp.up.a) < 0.1 * comp.up.A]].mean()


def test_lower_teeth_share_the_chin_motion():
    sh = [(0, 0), (2, 1), (0, -1), (3, 2), (1, 0), (4, 2), (2, 1), (5, 3), (3, 1), (6, 2)]
    n = len(sh)
    comp, layer, frames, S = _comp(gaps=tuple([20.0] * n))
    seq, Sq = _skin_seq(sh)
    a = np.linspace(0, 2 * np.pi, len(ml.CHIN), endpoint=False)
    for k, (dx, dy) in enumerate(sh):                # the soft chin region of the jittering skin patch
        Sq[k][ml.CHIN, :2] = np.stack([160 + dx + 30 * np.cos(a), 120 + dy + 20 * np.sin(a)], 1)
        Sq[k][14, :2] = (160 + dx, 100 + dy)
    layer.jaw_plan = np.zeros(n)                     # a still jaw: the chin's own motion is all the pin can add
    pin = comp.set_chin_pin(seq, Sq, sigma=0, noise=0)
    assert np.abs(pin - (np.array(sh, float) - sh[min(comp.h_lt, n - 1)])).max() < 0.35, pin
    # the lower arch moves by exactly the chin pin (on top of the skull + upper pin), the upper arch does not
    X = comp.lo.points(np.array([-10.0, 0.0, 10.0]), np.zeros(3))
    R = np.eye(3)
    for i in range(n):
        lo, _ = comp.proj(i, True, R)(X)
        up, _ = comp.proj(i, False)(X)
        assert np.allclose(lo - up, pin[i])


def test_bite_floor_and_flags_parse(capsys):
    comp, layer, frames, S = _comp()
    th = comp.bite_angle()
    T2 = (comp.lo.o - layer.H) @ tl.rot_axis(layer.u, th).T + layer.H     # the lower incisal edge at the closed bite
    assert abs((T2 - comp.lo.o) @ layer.d - ((comp.e_up - comp.overbite) - comp.e_lo)) < 1e-6
    noisy = np.radians(np.r_[np.full(10, -20.0), np.full(10, 5.0)])
    assert (tl.plan_jaw(noisy, floor=th) >= th - np.radians(0.3)).all()
    try: mouth_paint.main(['--help'])
    except SystemExit: pass
    o = capsys.readouterr().out
    for f in ('--warp-compose', '--warp-arch-k', '--warp-teeth-chin-pin', '--warp-layers-debug', '--warp-cover', '--warp-tongue'):
        assert f in o
