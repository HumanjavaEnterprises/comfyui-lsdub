# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v15 teeth (mouth_layers): Feature 1 = pink gingiva above / below the crowns with a scalloped gum line (--warp-arch-gums),
Feature 2 = the tongue covering the lower crowns at mid-open (--warp-tongue-cover). Both opt-in; the v14 defaults are bit for
bit. Synthetic, CPU only, our own classical graphics."""
import os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
from lsdub_engine.core import mouth_layers as ml
from test_arch_v13 import _band, _teeth_tex
from test_arch_v14 import _grey_crown
from test_mouth_layers import _comp


def _lab(T):
    return cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


# ------------------------------------------------------------------- Feature 1: pink gingiva (gum_pass / complete) --
def test_gum_pass_paints_a_pinker_band_above_the_crown():
    """a flat grey crown atlas + a pink gum colour: the band just above the crown (d in [th, th+gh]) ends up with a higher
    Lab a* (pinker) than it had, while the crown itself (d < th) is untouched."""
    band = _band(texel=2.0)
    T, r_e, th = _grey_crown(band)
    iod = 100.0
    gh = int(round(0.04 * iod * band.texel))                         # ~8 texel rows of gum
    rows = np.arange(band.Th)[:, None]
    d = (r_e - rows)
    gum_rows = np.broadcast_to((d >= th) & (d <= th + gh), T.shape[:2]) & (T[..., 3] > 0.5)
    crown_rows = np.broadcast_to((d >= 0) & (d <= th - 2), T.shape[:2]) & (T[..., 3] > 0.5)
    a_gum0 = _lab(T)[..., 1][gum_rows].mean()
    crown0 = T[..., :3][crown_rows].copy()
    pink = np.array([150.0, 150.0, 210.0], np.float32)               # BGR warm pink (a* well above 128)
    band.gum_pass(T, np.ones(band.Tw, bool), iod, True, r_e, th, gh, 1.0, pink, None)
    a_gum1 = _lab(T)[..., 1][gum_rows].mean()
    assert a_gum1 > a_gum0 + 5.0, (a_gum0, a_gum1)                   # the gum band is pinker
    assert np.allclose(T[..., :3][crown_rows], crown0, atol=1.0)      # the crown itself is untouched


def test_gum_pass_zero_strength_is_a_noop():
    band = _band(texel=2.0)
    T, r_e, th = _grey_crown(band)
    T0 = T.copy()
    band.gum_pass(T, np.ones(band.Tw, bool), 100.0, True, r_e, th, 8, 0.0, np.array([150.0, 150.0, 210.0], np.float32), None)
    assert np.array_equal(T, T0)                                     # strength 0 touches nothing


def test_gum_margin_is_scalloped_at_the_interdental_papillae():
    """with a slot layout, the gum reaches further toward the incisal edge (a lower margin, smaller d) at a slot boundary
    (the papilla) than over the middle of a tooth."""
    band = _band(texel=2.0)
    T, r_e, th = _grey_crown(band)
    iod = 100.0
    lay = dict(bounds=ml.anthro_bounds(ml.ANTHRO['up'], 69.0, 1.02 * band.A))
    pink = np.array([150.0, 150.0, 210.0], np.float32)
    band.gum_pass(T, np.ones(band.Tw, bool), iod, True, r_e, th, int(round(0.045 * iod * band.texel)), 1.0, pink, lay, drop=0.0, pap=0.3 * th)
    a = _lab(T)[..., 1]                                              # a* is high where gum was painted
    ac = (np.arange(band.Tw) + 0.5) / band.texel - band.A
    bnd = float(lay['bounds'][1])                                    # the central/lateral boundary (a papilla)
    col_pap = int(np.argmin(np.abs(ac - bnd)))
    col_mid = int(np.argmin(np.abs(ac - 0.5 * bnd)))   # a tooth centre
    rows = np.arange(band.Th)
    d = (r_e - rows)
    def lowest_gum_d(c):                                             # the smallest d (closest to the incisal edge) that is pink
        pink_rows = (a[:, c] > 136) & (T[:, c, 3] > 0.5) & (d >= 0) & (d <= th + 20)
        return float(d[pink_rows].min()) if pink_rows.any() else 1e9
    assert lowest_gum_d(col_pap) < lowest_gum_d(col_mid) - 1.0, (lowest_gum_d(col_pap), lowest_gum_d(col_mid))


def test_gum_margin_drops_onto_the_crown_top():
    """with drop > 0 the gumline comes DOWN onto the cervical enamel: the top of the crown (just below th) is pink, while the
    body of the crown lower down stays enamel. The founder's 'bring the gums down the teeth' ask."""
    band = _band(texel=2.0)
    T, r_e, th = _grey_crown(band)
    iod = 100.0
    gh = int(round(0.07 * iod * band.texel))
    drop = 0.2 * th
    rows = np.arange(band.Th)[:, None]
    d = (r_e - rows)
    cervical = np.broadcast_to((d >= th - drop + 1) & (d <= th - 2), T.shape[:2]) & (T[..., 3] > 0.5)   # the top slice of enamel
    body = np.broadcast_to((d >= 2) & (d <= th - drop - 3), T.shape[:2]) & (T[..., 3] > 0.5)            # the crown body below it
    a0_cerv = _lab(T)[..., 1][cervical].mean()
    body0 = T[..., :3][body].copy()
    band.gum_pass(T, np.ones(band.Tw, bool), iod, True, r_e, th, gh, 1.0, np.array([150.0, 150.0, 210.0], np.float32), None, drop=drop, pap=0.0)
    assert _lab(T)[..., 1][cervical].mean() > a0_cerv + 5.0            # the crown top is now gum (pinker)
    assert np.allclose(T[..., :3][body], body0, atol=1.0)              # the crown body is untouched (teeth keep their size)


def test_upper_gum_band_is_taller_than_the_lower():
    comp, *_ = _comp(build='v12', layout='anthro', gums=0.85)
    assert comp.gum_h_up > comp.gum_h_lo and comp.gum_h_up >= 0.055   # a real upper band, the lower subtle
    off, *_ = _comp(build='v12', layout='anthro')
    assert getattr(off, 'gum_h_up', 0.0) == 0.0 and off.gum_rgb is None   # off = no gum state


def test_complete_with_gums_is_pinker_and_off_is_identical():
    """end to end through complete(): a harvested crown built with gums > 0 has a pinker band above the crown than the same
    build with gums off; and gums off is byte-identical to a build that never passed the gum arguments."""
    lay = dict(bounds=ml.anthro_bounds(ml.ANTHRO['up'], 69.0, 1.02 * _band().A))
    pink = np.array([150.0, 150.0, 210.0], np.float32)
    b_off = _teeth_tex(_band(), period=7.5, run=16.0)
    b_off.complete(100.0, upper=True, synth=True, crown=True, layout=dict(lay), gums=0.0, gum_rgb=pink, gum_h=0.04)
    b_none = _teeth_tex(_band(), period=7.5, run=16.0)
    b_none.complete(100.0, upper=True, synth=True, crown=True, layout=dict(lay))
    assert np.array_equal(b_off.tex, b_none.tex)                     # gums=0 path == the path with no gum args at all
    assert not hasattr(b_off, 'gum_h_c') or b_off.gum_h_c == 0.0 or getattr(b_off, 'gum_h_c', 0.0) == 0.0
    b_on = _teeth_tex(_band(), period=7.5, run=16.0)
    b_on.complete(100.0, upper=True, synth=True, crown=True, layout=dict(lay), gums=1.0, gum_rgb=pink, gum_h=0.04)
    assert getattr(b_on, 'gum_h_c', 0.0) > 0.0                       # lip_band will start the lip above the gum
    r_e = (0.0 - b_on.h0) * b_on.texel - 0.5
    th = 0.11 * 100.0 * b_on.texel
    gh = 0.04 * 100.0 * b_on.texel
    rows = np.arange(b_on.Th)[:, None]
    d = (r_e - rows)
    reg = np.broadcast_to((d >= th + 1) & (d <= th + gh), b_on.tex.shape[:2])
    m_on = reg & (b_on.tex[..., 3] > 0.5)
    m_off = reg & (b_off.tex[..., 3] > 0.5)
    assert m_on.sum() > 50 and _lab(b_on.tex)[..., 1][m_on].mean() > _lab(b_off.tex)[..., 1][m_off].mean() + 4.0


def test_lip_band_leaves_the_gum_band_for_the_gum():
    """lip_band paints lip colour only ABOVE the gum: with gum_h_c set, the band's gum rows keep the gum weight 0 (the lip
    does not reach them), whereas without a gum they were lip."""
    class _B:  # a tiny stand-in band with the two attrs lip_band reads
        tooth_h_c = 0.11 * 100.0
        tr_c = 0.012 * 100.0
    comp, layer, *_ = _comp(build='v12', layout='anthro')
    io = comp.t.iod
    hm = np.full((5, 5), -(0.11 * io + 0.03 * io), np.float32)        # a band sitting 0.03 IOD into the gum/lip zone above the crown
    rgb = np.zeros((5, 5, 3), np.float32)
    b0 = _B()
    comp.lip_band(rgb.copy(), hm, b0, True, np.array([10.0, 20.0, 180.0], np.float32))
    gw0 = comp._gw.copy()
    b1 = _B()
    b1.gum_h_c = 0.05 * io
    comp.lip_band(rgb.copy(), hm, b1, True, np.array([10.0, 20.0, 180.0], np.float32))
    gw1 = comp._gw.copy()
    assert gw0.mean() > gw1.mean()                                   # the gum offset pulls the lip band up, off the gum rows
    assert gw1.max() < 0.5                                           # at 0.03 into a 0.05 gum, this row is still gum, not lip


# ---------------------------------------------------------------- Feature 2: the tongue covering the lower crowns --
def test_cover_amt_peaks_mid_open_and_is_zero_at_the_extremes():
    comp, *_ = _comp(tongue='wedge', tongue_cover=1.0)
    assert comp._cover_amt(0, op=0.05) < 0.1                          # near-closed: nothing to cover
    assert comp._cover_amt(0, op=0.35) > 0.6                          # mid-open: a real tongue fills the floor
    assert comp._cover_amt(0, op=0.95) < 0.1                          # wide-open 'ah': the lowers flash


def test_tongue_pose_rises_more_under_cover_at_mid_open_only():
    comp, layer, *_ = _comp(tongue='wedge', tongue_cover=1.0)
    base, *_ = _comp(tongue='wedge')                                  # cover off (default envelope)
    for c in (comp, base): c.prepare(np.array([4.0, 20.0, 28.0, 12.0]), ['sil', 'ih', 'aa', 'E'])
    # mid-open: the cover tongue's tip reaches further down / its top rises higher than the default
    op = 0.35
    tip_c = comp.tongue_pose(1, op)[0]
    tip_b = base.tongue_pose(1, op)[0]
    assert (tip_c[0][1] + tip_c[1][1]) > (tip_b[0][1] + tip_b[1][1]) + 1e-3     # the tip's bottom reaches lower (covers the crown)
    # wide-open: cover ~ 0, the pose is the default envelope
    tip_cw = comp.tongue_pose(1, 0.95)[0]
    tip_bw = base.tongue_pose(1, 0.95)[0]
    assert np.allclose(tip_cw[0], tip_bw[0]) and np.allclose(tip_cw[1], tip_bw[1])


def _exposed_lower(comp, i):
    """the count of bright (lower-crown) pixels that survive in L3 of frame i -- L3's alpha is already clipped to the mouth
    opening, so these are the lower teeth the tongue should hide at mid-open."""
    from test_teeth_layer import _frame
    P = comp.S[i]
    take = _frame(P)
    L, box = comp.layers(take, i, P, P, float(comp.t.F[i, 0]))
    rgb3, a3 = L['L3']
    lab = cv2.cvtColor(np.clip(rgb3, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0]
    return int(((a3 > 0.5) & (lab > 180)).sum())


def test_tongue_cover_hides_lower_crowns_mid_open_and_off_is_identical():
    gaps = np.array([4.0, 22.0, 30.0, 16.0])
    vis = ['sil', 'aa', 'aa', 'E']
    off, *_ = _comp(tongue='wedge')
    on, *_ = _comp(tongue='wedge', tongue_cover=0.9)
    for c in (off, on):
        c.prepare(gaps, vis)
        c.open_s = np.array([0.05, 0.35, 0.95, 0.35])     # frame 1,3 mid-open; frame 2 wide-open
    off2, *_ = _comp(tongue='wedge')
    off2.prepare(gaps, vis)
    off2.open_s = off.open_s.copy()
    # (a) cover off is identical to a second cover-off build on the same frame
    e_off = _exposed_lower(off, 1)
    e_off2 = _exposed_lower(off2, 1)
    assert e_off == e_off2
    # (b) mid-open: cover hides lower crowns (fewer exposed bright pixels)
    e_on = _exposed_lower(on, 1)
    assert e_on < e_off, (e_off, e_on)
    # (c) wide-open teeth-baring frame: cover ~ 0, the lowers still flash (within a small tolerance)
    w_off = _exposed_lower(off, 2)
    w_on = _exposed_lower(on, 2)
    assert w_on >= w_off - max(3, int(0.05 * max(1, w_off))), (w_off, w_on)
