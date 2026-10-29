"""v13 teeth scale (mouth_layers ANTHRO / anthro_map / synth_slots / the corridor): --warp-arch-layout anthro puts the plate's
crowns on the anthropometric slots (per the 63 mm IOD = the face's inter-pupil distance); --warp-arch-corridor K fades the
plates into the cavity at the corners of the planned opening; both opt-in, the defaults untouched. Synthetic, CPU only."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lsdub_engine.core import mouth_layers as ml, lip_layers as ll, mouth_paint
from test_mouth_layers import _comp
from test_teeth_layer import _P


def _band(iod=100.0, texel=2.0, arc=0.40):
    """a frontal upper ArchBand (canonical = frame), iod canonical px per IOD."""
    return ml.ArchBand(np.array([160.0, 130.0, 0.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                       arc * iod, 3.2 / iod, -0.32 * iod, 0.03 * iod, texel)


def _teeth_tex(band, period, run, line=0.35, colour=(205.0, 215.0, 225.0), offset=0.0):
    """a harvested upper texture: teeth of `period` canonical px between dark interdental lines (one on the midline + offset),
    opaque over the 0.06 IOD above the incisal edge, only within |a| <= run (canonical px); the rest transparent."""
    T = np.zeros((band.Th, band.Tw, 4), np.float32)
    e = int(round((0.0 - band.h0) * band.texel - 0.5))
    top = e - int(round(0.06 * 100.0 * band.texel))
    for c in range(band.Tw):
        a = band.a[c]
        if abs(a) > run: continue
        d = ((a - offset) / period) % 1.0
        d = min(d, 1 - d) * period * band.texel        # texels to the nearest line
        k = line if d < 0.6 else 1.0
        T[top:e + 1, c, :3] = np.array(colour) * k
        T[top:e + 1, c, 3] = 1.0
    band.tex = T
    band.src = (T[..., 3].max(0) > 0.5).astype(np.int8)
    return band


def test_anthro_table_and_bounds():
    assert ml.ANTHRO['up'][:6] == (0.1375, 0.105, 0.12, 0.11, 0.11, 0.16) and ml.ANTHRO['lo'][:5] == (0.087, 0.095, 0.11, 0.11, 0.11)
    b = ml.anthro_bounds(ml.ANTHRO['up'], 70.0, 40.0)
    assert b[0] == 0.0 and np.allclose(np.diff(b)[:3], np.array([0.1375, 0.105, 0.12]) * 70.0) and b[-1] > 40.0
    far = ml.anthro_bounds(ml.ANTHRO['up'], 1.0, 3.0)              # past the table: its last width repeats
    assert np.allclose(np.diff(far)[-2:], ml.ANTHRO['up'][-1])


def test_lines_map_onto_the_slots_and_spurious_lines_are_skipped():
    band = _band()
    u = 69.0
    b = ml.anthro_bounds(ml.ANTHRO['up'], u, 1.02 * band.A)
    tx = band.texel
    A = band.A
    c_of = lambda a: (np.asarray(a, float) + A) * tx - 0.5
    src_lines = np.array([-12.0, 0.0, 11.0, 19.5, 26.0])           # canonical px: narrow harvest teeth (0.11-0.12 IOD centrals)
    det = dict(x0=int(c_of(-25)), x1=int(c_of(30)), lines=c_of(np.sort(np.append(src_lines, 15.0))), prom=np.ones(6))   # + a spurious one
    sc, lg = ml.anthro_map(det, band.Tw, tx, A, b)
    assert lg['mode'] == 'lines' and 0.6 <= min(lg['stretch']) and max(lg['stretch']) <= 1.8
    assert np.isclose(lg['knots_slot'][lg['knots_src'].index(0.0)], 0.0)            # the midline line stays on the midline
    a_of = lambda c: (np.asarray(c, float) + 0.5) / tx - A
    for x, s in zip(lg['knots_src'], lg['knots_slot']):           # every assigned line lands on its slot boundary
        assert abs(a_of(np.interp(c_of(x), sc, np.arange(band.Tw))) - s) < 0.6 / tx
    assert any(abs(s - b[1]) < 1e-6 for s in lg['knots_slot']) and any(abs(s + b[1]) < 1e-6 for s in lg['knots_slot'])
    assert len(lg['skipped']) >= 1                                 # the spurious line (a 4 px "tooth") is skipped
    assert np.all(np.diff(sc) > 0)                                 # monotone


def test_uniform_stretch_with_few_lines_is_capped():
    band = _band()
    b = ml.anthro_bounds(ml.ANTHRO['up'], 69.0, 1.02 * band.A)
    tx = band.texel
    A = band.A
    c_of = lambda a: (np.asarray(a, float) + A) * tx - 0.5
    det = dict(x0=int(c_of(-15)), x1=int(c_of(15)), lines=c_of(np.array([-7.5, 0.5])), prom=np.ones(2))
    sc, lg = ml.anthro_map(det, band.Tw, tx, A, b)                 # 2 lines, 8 px apart: the period becomes the central slot
    assert lg['mode'] == 'uniform' and lg['period_from'] == 'detected' and np.isclose(lg['stretch_med'], 0.1375 * 69.0 / 8.0)
    one = dict(x0=int(c_of(-11)), x1=int(c_of(11)), lines=c_of(np.array([0.2])), prom=np.ones(1))
    sc1, lg1 = ml.anthro_map(one, band.Tw, tx, A, b)               # one line mid-run: the run is the two teeth it separates
    assert lg1['period_from'] == 'run'
    tiny = dict(x0=int(c_of(-3)), x1=int(c_of(3)), lines=c_of(np.array([-1.5, 1.5])), prom=np.ones(2))
    _, lg2 = ml.anthro_map(tiny, band.Tw, tx, A, b)                # 3 px teeth would need x3.2: capped at x1.8, logged
    assert lg2['stretch_med'] == 1.8 and lg2['capped']


def test_complete_puts_the_crowns_on_the_slots_and_models_the_rest():
    band = _teeth_tex(_band(), period=7.5, run=16.0)              # harvest teeth 0.075 IOD wide (v12's default period)
    u = 69.0
    lay = dict(bounds=ml.anthro_bounds(ml.ANTHRO['up'], u, 1.02 * band.A))
    band.complete(100.0, upper=True, synth=True, crown=True, layout=lay)
    lines, (cl, cr) = band.lines_now(100.0, None, True)
    assert cl is not None and cr is not None
    assert abs(cl - 0.1375 * u / 100.0) < 0.012 and abs(cr - 0.1375 * u / 100.0) < 0.012, (cl, cr)
    # beyond the run every modelled tooth fills one slot: its interproximal line sits on a slot boundary
    bnd = lay['bounds'] / 100.0
    side = [x for x in lines if abs(x) > 0.2 and abs(x) < band.A / 100.0 - 0.03]
    assert side and all(np.min(np.abs(np.abs(x) - bnd)) < 0.01 for x in side), (side, bnd)
    assert band.layout_log['mode'] in ('lines', 'uniform') and band.synth_cols > 0


def test_missed_lines_skip_boundaries_and_slivers_are_modelled_whole():
    band = _band()
    b = ml.anthro_bounds(ml.ANTHRO['up'], 69.0, 1.02 * band.A)
    tx = band.texel
    A = band.A
    c_of = lambda a: (np.asarray(a, float) + A) * tx - 0.5
    # the harvest's lateral line was missed on the right: its next line sits on the lateral / canine boundary
    det = dict(x0=int(c_of(-30)), x1=int(c_of(30)), lines=c_of(np.array([-b[1], 0.0, b[1], b[2]])) + 0.0, prom=np.ones(4))
    det['lines'] = c_of(np.array([-b[1], 0.0, b[2]]))
    _, lg = ml.anthro_map(det, band.Tw, tx, A, b)
    assert lg['mode'] == 'lines' and np.isclose(lg['knots_slot'][-1], b[2])       # two slots for the long segment, not x0.5
    det['lines'] = c_of(np.array([-b[1], 0.0, b[3]]))                             # two lines missed: three slots
    _, lg = ml.anthro_map(det, band.Tw, tx, A, b)
    assert np.isclose(lg['knots_slot'][-1], b[3])
    # a run ending 30 % into the canine slot is cut back to the slot's start (synth_slots then models the canine whole);
    # 70 % in: kept; the centrals are never cut
    for frac, cut in ((0.3, True), (0.7, False)):
        a_end = b[2] + frac * (b[3] - b[2])
        hi = int(np.floor(c_of(a_end)))
        lo = band.Tw - 1 - hi
        lo2, hi2, tr = ml.trim_run(lo, hi, b, A, tx)
        assert (hi2 < hi) == cut and (lo2 > lo) == cut and (tr[1] > 0) == cut
        if cut: assert abs((hi2 + 1 + 0.5) / tx - A - b[2]) <= 1.0 / tx                # the slot's first column follows the run
    lo3, hi3, tr3 = ml.trim_run(int(c_of(-0.3 * b[1])), int(c_of(0.3 * b[1])), b, A, tx)
    assert tr3 == [0, 0]


def test_slot_boundaries_read_as_lines_on_the_run_and_beyond():
    # a harvest with NO visible lines: after complete(layout) every slot boundary on the run is darker than its slot's middle
    band = _teeth_tex(_band(), period=1000.0, run=22.0, line=1.0)
    u = 69.0
    lay = dict(bounds=ml.anthro_bounds(ml.ANTHRO['up'], u, 1.02 * band.A))
    band.complete(100.0, upper=True, synth=True, crown=True, layout=lay)
    e = int(round((0.0 - band.h0) * band.texel - 0.5))
    row = band.tex[e - int(0.03 * 100 * band.texel), :, :3].mean(-1)
    col = lambda a: int(round((a + band.A) * band.texel - 0.5))
    bb = lay['bounds']
    for k, lim in ((1, 0.9), (2, 0.9), (3, 0.7)):                 # the lateral / canine boundaries on the run (the groove), the
        for sg in (1, -1):                                         # canine / premolar one beyond it (synth_slots' line)
            mid = sg * 0.5 * (bb[k] + bb[k + 1])
            c = col(sg * bb[k])
            edge = row[c - 1:c + 2].min()
            assert edge < lim * row[col(mid)], (k, sg, edge, row[col(mid)])


def test_corridor_fade_and_dim():
    comp, layer, frames, S = _comp(corridor=0.12)
    P = _P(gap=20.0)
    Q = P[ll.LIP_IN, :2] - [100.0, 100.0]
    h, w = 80, 140
    f = comp.corridor_fade(Q, h, w)
    W = comp._cor['W']
    s = comp._cor['s']
    assert np.all(f[np.abs(s) <= (0.5 - 0.12) * W] == 0.0) and np.all(f[np.abs(s) >= 0.5 * W] > 0.999)
    rgb = np.full((h, w, 3), 200.0, np.float32)
    cav = np.full((h, w, 3), 20.0, np.float32)
    out = comp.corridor_dim(rgb, cav, f)
    assert np.array_equal(out[f == 0], rgb[f == 0])                                  # untouched inside, bit for bit
    full = f > 0.999
    assert np.allclose(out[full], 20.0 + 0.25 * 180.0, atol=0.5)                     # a faint crown at the corners, not black
    mid = (f > 0.3) & (f < 0.7)
    assert np.all(out[mid] < 200.0) and np.all(out[mid] > 20.0 + 0.25 * 180.0 - 0.5)


def test_corridor_in_the_stack_dims_only_the_plates_at_the_corners():
    base, *_ = _comp(gaps=(4.0, 20.0, 28.0, 12.0))
    cor, layer, frames, S = _comp(gaps=(4.0, 20.0, 28.0, 12.0), corridor=0.12)
    P = _P(gap=28.0, hw=46)
    take = frames[2]
    L0, box = base.layers(take, 2, S[2], P, 28.0)
    L1, box1 = cor.layers(take, 2, S[2], P, 28.0)
    assert box == box1
    x0, y0, w, h = box
    Q = P[ll.LIP_IN, :2] - [x0, y0]
    f = cor.corridor_fade(Q, h, w)
    for k in ('L5', 'L6', 'L0', 'L1'):                                                # the lips, the face, the cover: identical
        assert np.array_equal(L0[k][0], L1[k][0]) and np.array_equal(L0[k][1], L1[k][1]), k
    a4 = L0['L4'][1] > 0.5
    assert np.allclose(L0['L4'][0][a4 & (f == 0)], L1['L4'][0][a4 & (f == 0)])         # the arch inside the span: unchanged
    cm = a4 & (f > 0.5)
    if cm.any(): assert (L1['L4'][0][cm].sum(1) < L0['L4'][0][cm].sum(1) - 1).mean() > 0.9   # at the corners: darker
    r = cor.reg[2]['corridor']
    assert r['K'] == 0.12 and r['W'] > 0 and len(r['start']) == 2 and r['dark_up'] is not None
    info = cor.info()['corridor']
    assert info['K'] == 0.12 and 'arc_short_up' in info and info['frames'] >= 1


def test_corridor_dims_both_arches_but_never_the_tongue():
    base, *_ = _comp(gaps=(4.0, 20.0, 28.0, 12.0), tongue='wedge')
    cor, layer, frames, S = _comp(gaps=(4.0, 20.0, 28.0, 12.0), tongue='wedge', corridor=0.35)
    P = _P(gap=34.0, hw=46)
    take = frames[2]
    L0, box = base.layers(take, 2, S[2], P, 34.0)
    L1, _ = cor.layers(take, 2, S[2], P, 34.0)
    x0, y0, w, h = box
    f = cor.corridor_fade(P[ll.LIP_IN, :2] - [x0, y0], h, w)
    # the lower arch: L3 = the arch over the tongue; where the arch is opaque in the corridor it is darker ...
    lo_rgb, lo_a = cor.lo.render(cor.proj(2, True, cor.jaw_R(2, 34.0, P[14])), box, layer.pl.R[2], facing=cor.facing)
    arch = (lo_a > 0.98) & (f > 0.6) & (L0['L3'][1] > 0.5)
    assert arch.sum() > 3 and (L1['L3'][0][arch].sum(1) < L0['L3'][0][arch].sum(1) - 1).mean() > 0.9
    assert np.allclose(L0['L3'][0][f == 0], L1['L3'][0][f == 0])
    # ... and the tongue is never dimmed: with the lower arch transparent, L3 is the tongue alone, with or without the corridor
    for c in (base, cor): c.lo.tex[..., 3] = 0.0
    T0, _ = base.layers(take, 2, S[2], P, 34.0)
    T1, _ = cor.layers(take, 2, S[2], P, 34.0)
    tongue = (T0['L3'][1] > 0.05) & (f > 0.02)
    assert tongue.sum() > 5 and np.allclose(T0['L3'][0][tongue], T1['L3'][0][tongue])


def test_corridor_keeps_the_v12_lip_band():
    # v12's lip band (the plate beyond the crown takes the planned lip's inner rim colour) is the lips' own inner surface: the
    # corridor dims the crowns at the corners, never the band
    a, layer, frames, S = _comp(gaps=(4.0, 20.0, 28.0, 12.0), build='v12')
    b, *_ = _comp(gaps=(4.0, 20.0, 28.0, 12.0), build='v12', corridor=0.12)
    P = _P(gap=40.0, hw=46)
    take = frames[2]
    L0, box = a.layers(take, 2, S[2], P, 40.0)
    L1, _ = b.layers(take, 2, S[2], P, 40.0)
    g = b._gw
    x0, y0, w, h = box
    f = b.corridor_fade(P[ll.LIP_IN, :2] - [x0, y0], h, w)
    band = (g > 0.999) & (f > 0.5)
    crown = (g < 0.001) & (f > 0.5) & (b._full['L4'] > 0.5)
    assert band.sum() > 20 and crown.sum() > 5
    assert np.allclose(L0['L4'][0][band], L1['L4'][0][band])
    assert (L1['L4'][0][crown].sum(1) < L0['L4'][0][crown].sum(1) - 1).all()


def test_unit_is_the_face_inter_pupil_distance():
    comp, layer, *_ = _comp(layout='anthro')
    assert comp.unit_info['source'] == 'fallback' and np.isclose(comp.unit_c, ml.IPD_OCD * layer.iod)    # no irises in the fixture
    Cm = layer.Cm.copy()
    c = Cm[[33, 263]].mean(0)
    Cm[468] = c - [0.35 * layer.iod, 0, 0]
    Cm[473] = c + [0.35 * layer.iod, 0, 0]
    layer.Cm = Cm
    u, inf = comp._anthro_unit('ipd')
    assert inf['source'] == 'iris' and np.isclose(u, 0.70 * layer.iod)
    u2, inf2 = comp._anthro_unit('iod')
    assert u2 == layer.iod and inf2['mode'] == 'iod'
    lay = comp.info()['layout']
    assert lay['table']['unit'] == 'IPD (63 mm)' and lay['slots_up'][0] < 0 < lay['slots_up'][-1] and 0.0 in lay['slots_up']


def test_defaults_untouched_and_flags_parse(capsys):
    comp, layer, frames, S = _comp()
    P = _P(gap=20.0)
    assert comp.lay is None and comp.corridor == 0.0
    comp.layers(frames[2], 2, S[2], P, 20.0)
    inf = comp.info()
    assert 'layout' not in inf and 'corridor' not in inf and 'corridor' not in comp.reg[2]
    try: mouth_paint.main(['--help'])
    except SystemExit: pass
    o = capsys.readouterr().out
    for fl in ('--warp-arch-layout', '--warp-arch-corridor', '--warp-arch-corridor-floor', '--warp-arch-anthro-unit'):
        assert fl in o
