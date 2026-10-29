"""v8 polish of the layered mouth (mouth_layers.py): the chin pin split + projected onto the jaw hinge's arc, the
modelled tongue's near-closed constraint, modelled side teeth, multi-frame harvest, the lip-aware ghost check, flags.
Synthetic, CPU only."""
import numpy as np, cv2

from lsdub_engine.core import mouth_layers as ml, lip_layers as ll, mouth_paint
from lsdub_engine.audits import lip_edge as gc
from test_teeth_layer import _P, _frame, _layer, _skin_seq, LIP, TOOTH
from test_mouth_layers import _comp


def _chin_seq(sh, y14=100):
    """a jittering skin sequence whose soft chin + lower lip (at y14) move by sh[k] (px)."""
    seq, Sq = _skin_seq(sh)
    a = np.linspace(0, 2 * np.pi, len(ml.CHIN), endpoint=False)
    for k, (dx, dy) in enumerate(sh):
        Sq[k][ml.CHIN, :2] = np.stack([160 + dx + 30 * np.cos(a), 120 + dy + 20 * np.sin(a)], 1)
        Sq[k][14, :2] = (160 + dx, y14 + dy)
    return seq, Sq


def test_arc_pin_stays_on_the_hinge_arc_and_attenuates_the_fast_residual():
    # a slow drift of the chin (+ the take's own fast jaw jitter on top of it)
    n = 16
    slow = np.round(np.linspace(0, 6, n)).astype(int)
    fast = 3 * (-1) ** np.arange(n)
    sh = [(0, int(a + b)) for a, b in zip(slow, fast)]
    comp, layer, frames, S = _comp(gaps=tuple([20.0] * n), pin_mode='arc', pin_gain=0.4, pin_split=1.5)
    seq, Sq = _chin_seq(sh)
    layer.jaw_plan = np.zeros(n)
    comp.expose = 5.0        # (no exposure bound in this test)
    comp.set_chin_pin(seq, Sq, sigma=0, noise=0)
    assert not comp.pin_lo.any()                                                      # nothing translates off the arc
    H = layer.H
    for i in range(n):
        R = comp.jaw_R(i)
        T = (comp.lo.o - H) @ R.T + H
        assert abs(np.linalg.norm(T - H) - np.linalg.norm(comp.lo.o - H)) < 1e-9     # the incisor stays on its circle
        v = comp.arc_tangent(i, R)
        d = comp.pin_disp[i]                               # and its picture motion is along the arc
        assert abs(d[0] * v[1] - d[1] * v[0]) < 1e-6 * max(1.0, np.linalg.norm(v) * np.linalg.norm(d))
    # the slow part is followed in full, the frame-alternating part only by the gain (and eased)
    raw = comp.pin_raw[:, 1]
    got = comp.pin_disp[:, 1]
    hf = lambda x: np.abs(np.diff(x, 2))[2:-2].mean()
    assert hf(got) < 0.6 * hf(raw), (hf(got), hf(raw))
    from scipy.ndimage import gaussian_filter1d
    assert np.abs(gaussian_filter1d(got, 2.0)[3:-3] - gaussian_filter1d(raw, 2.0)[3:-3]).max() < 1.0


def test_arc_pin_honours_the_exposure_limit_so_the_clamp_never_fires():
    n = 10
    sh = [(0, -3 * k) for k in range(n)]                                      # the chin rising fast: the pin would lift the lower teeth
    comp, layer, frames, S = _comp(gaps=tuple([20.0] * n), pin_mode='arc', pin_gain=1.0, pin_split=0.0)
    seq, Sq = _chin_seq(sh, y14=150)
    layer.jaw_plan = np.zeros(n)
    comp.expose = 0.0   # the lower lip 12 px below the incisal edge
    comp.set_chin_pin(seq, Sq, sigma=0, noise=0)
    for i in range(n):
        over, rate, _ = comp.exposure_over(i, comp.jaw_R(i), Sq[i][14])
        assert over <= 0.05, (i, over)
    assert comp.dth.max() > 0.05                                                        # (the bound did act: the jaw opened to it)
    # (and the skin mode, v7, would have had to clamp)
    comp7, layer7, _, _ = _comp(gaps=tuple([20.0] * n))
    layer7.jaw_plan = np.zeros(n)
    comp7.expose = 0.0
    comp7.set_chin_pin(seq, Sq, sigma=0, noise=0)
    assert max(comp7.exposure_over(i, comp7.jaw_R(i), Sq[i][14])[0] for i in range(n)) > 1.0


def test_wedge_tongue_never_above_the_lower_teeth_when_near_closed():
    comp, layer, frames, S = _comp(tongue='wedge')
    comp.prepare(np.array([4.0, 20.0, 28.0, 12.0]), ['sil', 'ih', 'aa', 'E'])        # a narrow vowel raises the tongue ...
    for op in (0.0, 0.05, 0.12):
        for c, ax in comp.tongue_pose(1, op): assert c[1] - ax[1] >= 0.0, (op, c, ax)   # ... but never above the edge when near closed
    tops = [c[1] - ax[1] for c, ax in comp.tongue_pose(1, 1.0)]
    assert min(tops) < -0.02                                                           # wide open: the dorsum shows over the lower teeth
    # rendered: on a near-closed frame no tongue pixel sits above the lower incisal edge (frontal pose: the edge is a row)
    comp._open = 0.0
    comp.prepared = False
    box = (0, 0, 320, 260)
    rgb, a = comp.tongue_wedge(1, np.eye(3), box, layer.iod)
    e_y = comp.proj(1, True, np.eye(3))(comp.lo.o[None])[0][0, 1]
    assert a[: int(np.floor(e_y)) - 1].max() < 0.02
    assert a.max() > 0.5                                                                # (there is a tongue)


def test_modelled_side_teeth_and_multi_frame_harvest():
    layer, frames, S = _layer((20.0, 20.0, 20.0, 20.0, 20.0), tooth_w=4)              # a narrow row of teeth: most of the arch is built
    st = ll.lip_stats(frames[2], S[2][:, :2])
    comp = ml.LayerCompositor(layer, frames, S, st[1:3], st[3:5], harvest_n=3, synth=True, desat=0.5, texel=2.0)
    assert comp.harvest_used['up'] >= 2                                                 # the median of several frames
    assert comp.up.texel == 2.0 and comp.up.Tw > 2 * comp.up.A
    b = comp.up
    assert b.synth_cols > 0 and (b.src == 3).any()
    r = int((0 - b.h0) * b.texel) - int(0.02 * layer.iod * b.texel)                    # a row just inside the incisal edge
    lab = cv2.cvtColor(np.clip(b.tex[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(float)
    cols = np.nonzero(b.src == 3)[0]
    right = cols[cols > b.Tw // 2]
    # modelled teeth: a periodic dark interproximal line every tooth (not one repeated column)
    L = lab[r, right, 0]
    dark = np.nonzero((L[1:-1] < L[:-2] - 4) & (L[1:-1] < L[2:] - 4))[0]
    assert len(dark) >= 2, L.round(0)
    # and the arch recedes: darker and less saturated at the far end than next to the harvest
    near, far = right[: len(right) // 4], right[-len(right) // 4:]
    chroma = np.hypot(lab[r, :, 1] - 128, lab[r, :, 2] - 128)
    assert lab[r, far, 0].mean() < lab[r, near, 0].mean() and chroma[far].mean() <= chroma[near].mean() + 1e-6




def test_lip_aware_lip_edge_counts_lips_not_teeth():
    P = _P(gap=20.0)
    f = _frame(P)
    p = P[:, :2].copy()
    p[2] = (160, 40)
    p[152] = (160, 220)                                              # nose base / chin for the walks
    p[[187, 207, 205, 50]] = [(100, 60), (100, 80), (90, 70), (95, 90)]
    p[[411, 427, 425, 280]] = [(220, 60), (220, 80), (230, 70), (225, 90)]
    # (a) a clean frame whose tracked inner lower lip sits above a visible lower-teeth row + tongue: teeth, a dark gap,
    #     the tongue, then the lip -- bands of luminance, but one lip
    f1 = f.copy()
    y14 = int(P[14, 1])
    f1[y14 - 16:y14 - 10, 130:190] = TOOTH
    f1[y14 - 10:y14 - 7, 130:190] = (20, 15, 40)
    f1[y14 - 7:y14, 130:190] = (110, 100, 170)
    p1 = p.copy()
    p1[14, 1] = y14 - 16
    assert not gc.frame_ghost_lip(f1, p1)[0]
    # (c) two dark grey creases under the lip (luminance valleys, no lip colour): the old check counts a second band, the
    #     lip-aware one does not
    f3 = f.copy()
    y17 = int(P[17, 1])
    f3[y17 + 5:y17 + 8, 120:200] = (70, 70, 70)
    f3[y17 + 14:y17 + 17, 120:200] = (70, 70, 70)
    assert gc.frame_ghost(f3, p)[0] and not gc.frame_ghost_lip(f3, p)[0]
    # (b) a real ghost: a second lip-coloured band below the lower lip, over skin
    f2 = f.copy()
    y17 = int(P[17, 1])
    f2[y17 + 12:y17 + 18, 125:195] = LIP
    assert gc.frame_ghost_lip(f2, p)[0]
    assert not gc.frame_ghost_lip(f, p)[0]


def test_v8_flags_parse(capsys):
    try: mouth_paint.main(['--help'])
    except SystemExit: pass
    o = capsys.readouterr().out
    for s in ('arc', 'wedge', 'fill', '--warp-arch-synth', '--warp-arch-harvest', '--warp-lip-wet', '--warp-lip-rim', '--warp-layers-smooth',
              '--warp-chin-pin-gain', '--warp-arch-tex-from'):
        assert s in o, s
