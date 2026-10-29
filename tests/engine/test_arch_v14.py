"""v14 teeth cleanup (mouth_layers): Part 1 = supersample + area-correct minification of the arch atlas (--warp-arch-mipmap, a
trilinear INTER_AREA mip pyramid sampled per column-quad). Part 3 = the enamel material pass (--warp-arch-enamel). Both opt-in;
the v13 defaults are bit for bit. Synthetic, CPU only."""
import os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
from lsdub_engine.core import mouth_layers as ml
from test_arch_v13 import _band


def _ortho(scale, box):
    """a frontal orthographic proj3(X Nx3) -> (px Nx2, depth N); scale < 1 minifies (texels per output px > 1). The band
    (canonical centre ~ (160, 130)) is mapped to the centre of box (x0, y0, w, h)."""
    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
    off = np.array([cx - 160.0 * scale, cy - 130.0 * scale])
    def proj3(X):
        X = np.asarray(X, float).reshape(-1, 3)
        return X[:, :2] * scale + off, X[:, 2].copy()
    return proj3


def _grating(band, period_tx=2):
    """an opaque atlas whose columns alternate bright / dark every `period_tx` texels (a worst-case interdental grating)."""
    T = np.zeros((band.Th, band.Tw, 4), np.float32)
    cols = (np.arange(band.Tw) // period_tx) % 2
    T[:, :, :3] = np.where(cols[None, :, None] > 0, 230.0, 60.0)
    T[:, :, 3] = 1.0
    band.tex = T.astype(np.float32)
    band.src = np.ones(band.Tw, np.int8)
    band._mip = None
    return band


def test_mip_pyramid_is_premultiplied_and_area_correct():
    """a half-opaque-bright / half-transparent atlas: the downsampled crown colour stays bright (no dark bleed from the
    transparent texels), and the alpha area-averages toward 0.5 at the boundary."""
    b = _band(texel=2.0)
    T = np.zeros((b.Th, b.Tw, 4), np.float32)
    half = b.Tw // 2
    T[:, :half, :3] = 200.0
    T[:, :half, 3] = 1.0                      # left: opaque bright; right: transparent (0)
    b.tex = T
    b._mip = None
    lv = b.mip_pyramid()
    assert len(lv) >= 3 and lv[0].shape == T.shape
    # premultiplied: level 0 rgb == rgb * alpha
    assert np.allclose(lv[0][..., :3], T[..., :3] * T[..., 3:4])
    # a deep level, well inside the opaque half: straight colour recovered ~200 (not pulled toward 0 by transparent texels)
    deep = lv[2]
    a = deep[..., 3:4]
    straight = np.where(a > 1e-4, deep[..., :3] / np.maximum(a, 1e-4), 0.0)
    q = straight.shape[1] // 4                                        # a column a quarter in: firmly opaque
    assert 190.0 <= straight[:, q, 0].mean() <= 205.0
    # the alpha has area-averaged (a soft boundary), not stayed a hard 0/1 step
    mid = deep[:, deep.shape[1] // 2 - 1: deep.shape[1] // 2 + 1, 3]
    assert 0.1 < float(mid.mean()) < 0.9


def test_mipmap_reduces_minification_aliasing_and_keeps_the_mean():
    """rendering a fine grating minified: the mip path's column-to-column variance (the picket-fence aliasing) is far lower
    than raw bilinear's, while the mean brightness over the patch is preserved (area average, not a dark/bright bias)."""
    scale = 0.25                                                      # 4 texels per output px -> heavy minification
    box = (0, 0, 120, 70)
    rot = np.eye(3)
    raw = _grating(_band(texel=2.0))
    raw.mipmap = False
    mip = _grating(_band(texel=2.0))
    mip.mipmap = True
    rgb_r, a_r = raw.render(_ortho(scale, box), box, rot)
    rgb_m, a_m = mip.render(_ortho(scale, box), box, rot)
    m = (a_r > 0.5) & (a_m > 0.5)
    assert m.sum() > 80
    # high-frequency energy along x (the aliasing) collapses under the mip
    hf = lambda g: float(np.mean((np.abs(np.diff(g[..., 0], axis=1)))[m[:, 1:] & m[:, :-1]]))
    hf_raw, hf_mip = hf(rgb_r), hf(rgb_m)
    assert hf_mip < 0.5 * hf_raw, (hf_raw, hf_mip)
    # the mean is preserved (both average the 60/230 grating toward ~145)
    assert abs(float(rgb_m[..., 0][m].mean()) - float(rgb_r[..., 0][m].mean())) < 20.0
    assert 120.0 < float(rgb_m[..., 0][m].mean()) < 170.0


def test_mipmap_is_temporally_stable_across_subpixel_scales():
    """two near-identical minification scales (a sub-pixel head move) produce near-identical mip renders -- no shimmer --
    whereas raw bilinear flickers. Measured as the max per-pixel change between the two scales."""
    box = (0, 0, 120, 70)
    rot = np.eye(3)
    def render(mipmap, scale):
        b = _grating(_band(texel=2.0))
        b.mipmap = mipmap
        rgb, a = b.render(_ortho(scale, box), box, rot)
        return rgb[..., 0], a > 0.5
    r0, m0 = render(False, 0.250)
    r1, m1 = render(False, 0.258)
    g0, n0 = render(True, 0.250)
    g1, n1 = render(True, 0.258)
    mr = m0 & m1
    mg = n0 & n1
    flick_raw = float(np.abs(r0 - r1)[mr].mean())
    flick_mip = float(np.abs(g0 - g1)[mg].mean())
    assert flick_mip < flick_raw, (flick_raw, flick_mip)


def test_magnification_falls_back_to_level_zero():
    """on magnification (ratio < 1 texel per px) the mip path samples level 0: the render matches raw bilinear closely (the
    pyramid must not blur a close-up that is already super-resolved)."""
    scale = 3.0
    box = (0, 0, 220, 90)
    rot = np.eye(3)              # 1/3 texel per px
    raw = _grating(_band(texel=2.0), period_tx=4)
    raw.mipmap = False
    mip = _grating(_band(texel=2.0), period_tx=4)
    mip.mipmap = True
    rgb_r, a_r = raw.render(_ortho(scale, box), box, rot)
    rgb_m, a_m = mip.render(_ortho(scale, box), box, rot)
    m = (a_r > 0.5) & (a_m > 0.5)
    assert m.sum() > 300 and float(np.abs(rgb_r[..., 0] - rgb_m[..., 0])[m].mean()) < 2.0


# ----------------------------------------------------------------------------- Part 3: the enamel material pass --
def _grey_crown(band, L=170.0, a=128.0, b=132.0):
    """a flat matte crown atlas: constant Lab (L, a*, b*) over the crown rows (within th of the incisal edge), opaque; the
    gum rows above it a different flat colour; the rest transparent. Returns (T, r_e, th)."""
    iod = 100.0
    th = int(round(0.11 * iod * band.texel))
    r_e = (0.0 - band.h0) * band.texel - 0.5
    lab = np.zeros((band.Th, band.Tw, 3), np.uint8)
    lab[..., 0] = 90
    lab[..., 1] = int(a)
    lab[..., 2] = 150   # gum-ish
    rows = np.arange(band.Th)[:, None]
    d = (r_e - rows)
    crown = (d >= -0.5) & (d <= th)
    crown = np.broadcast_to(crown, (band.Th, band.Tw))
    lab[..., 0] = np.where(crown, int(L), lab[..., 0])
    lab[..., 1] = np.where(crown, int(a), lab[..., 1])
    lab[..., 2] = np.where(crown, int(b), lab[..., 2])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    T = np.zeros((band.Th, band.Tw, 4), np.float32)
    T[..., :3] = bgr
    gum = (d >= -0.5) & (d <= th + int(round(0.06 * iod * band.texel)))
    T[..., 3] = np.where(np.broadcast_to(gum, (band.Th, band.Tw)), 1.0, 0.0)
    return T, r_e, th


def _lab_of(T, mask):
    lab = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[mask].mean(0)


def test_enamel_warms_the_hue_and_adds_gloss_and_ao():
    band = _band(texel=2.0)
    band.x_hl = 0.6
    T, r_e, th = _grey_crown(band)
    iod = 100.0
    fb = np.zeros(band.Tw, np.float32)
    rows = np.arange(band.Th)[:, None]
    d = (r_e - rows)
    crown = np.broadcast_to((d >= -0.5) & (d <= th), T.shape[:2]) & (T[..., 3] > 0.5)
    L0, a0, b0 = _lab_of(T, crown)
    T2 = T.copy()
    band.enamel_pass(T2, np.ones(band.Tw, bool), iod, True, r_e, th, fb, dict(spec=1.0, transl=0.8, ao=0.7, warm=1.0, a_target=128.0), None)
    L1, a1, b1 = _lab_of(T2, crown)
    assert b1 > b0 + 2.0, (b0, b1)                       # warmer (yellower)
    assert a1 <= a0 + 1.0                                 # a* not pushed red
    # gloss: the brightest crown pixel is much brighter than the flat input (a real highlight exists)
    lab2 = cv2.cvtColor(np.clip(T2[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    assert lab2[..., 0][crown].max() > L0 + 25.0
    # AO: the gum-side quarter of the crown is darker than the mid-crown (roundness / occlusion)
    u = np.clip((r_e - rows) / max(1.0, th), 0, 1)
    gum_q = np.broadcast_to(u > 0.8, T.shape[:2]) & crown
    mid_q = np.broadcast_to((u > 0.35) & (u < 0.55), T.shape[:2]) & crown
    assert lab2[..., 0][gum_q].mean() < lab2[..., 0][mid_q].mean() - 5.0


def test_enamel_leaves_the_gum_and_transparent_texels_untouched():
    band = _band(texel=2.0)
    band.x_hl = 0.0
    T, r_e, th = _grey_crown(band)
    iod = 100.0
    rows = np.arange(band.Th)[:, None]
    d = (r_e - rows)
    gum = np.broadcast_to((d > th + 1), T.shape[:2]) & (T[..., 3] > 0.5)
    trans = T[..., 3] <= 0.5
    before_gum = T[..., :3][gum].copy()
    before_trans = T[..., :3][trans].copy()
    band.enamel_pass(T, np.ones(band.Tw, bool), iod, True, r_e, th, np.zeros(band.Tw, np.float32),
                     dict(spec=1.0, transl=1.0, ao=1.0, warm=1.0, a_target=128.0), None)
    assert np.array_equal(T[..., :3][gum], before_gum) and np.array_equal(T[..., :3][trans], before_trans)


def test_enamel_zero_strength_is_a_noop():
    band = _band(texel=2.0)
    T, r_e, th = _grey_crown(band)
    T0 = T.copy()
    band.enamel_pass(T, np.ones(band.Tw, bool), 100.0, True, r_e, th, np.zeros(band.Tw, np.float32),
                     dict(spec=0.0, transl=0.0, ao=0.0, warm=0.0, a_target=128.0), None)
    assert np.allclose(T, T0, atol=1.0)                   # LAB<->BGR round-trip within 1 level, no material change


# --------------------------------------------------------------- Part 2: one sharp atlas shared across a face's shots --
import types, tempfile


def _filled(band, L=200.0):
    T = np.zeros((band.Th, band.Tw, 4), np.float32)
    T[..., :3] = L
    T[..., 3] = 1.0
    band.tex = T
    band.src = np.ones(band.Tw, np.int8)
    return band


def _share_stub(up, lo, lo_synth=True):
    s = types.SimpleNamespace(t=types.SimpleNamespace(iod=100.0), up=up, lo=lo, lo_synth=lo_synth, fallback=[])
    s._teeth_L = types.MethodType(ml.LayerCompositor._teeth_L, s)
    s._tex_detail = types.MethodType(ml.LayerCompositor._tex_detail, s)
    s._fallback_tex = types.MethodType(ml.LayerCompositor._fallback_tex, s)
    return s


def _atlas_npz(src_up, src_lo, up_h_iod, lo_h_iod, path, io=100.0, up_detail=0.0, lo_detail=0.0):
    np.savez(path, up_tex=src_up.tex, up_a=src_up.a / io, up_h=src_up.h / io, lo_tex=src_lo.tex, lo_a=src_lo.a / io,
             lo_h=src_lo.h / io, iod=io, up_L=200.0, lo_L=200.0, up_h_iod=up_h_iod, lo_h_iod=lo_h_iod,
             up_detail=up_detail, lo_detail=lo_detail)


def test_tex_share_uses_the_sharp_atlas_when_the_own_harvest_is_shorter_even_if_above_min_h():
    """the shared atlas (taller exposed teeth, 0.11 IOD) drives a shot whose own harvest is only 0.07 IOD -- ABOVE the 0.05
    min-h that the v8 short-harvest fallback needs. Without share the shot keeps its own (shorter) harvest; with share it
    loads the sharp atlas (src 4)."""
    with tempfile.TemporaryDirectory() as d:
        src_up = _filled(_band())
        src_lo = _filled(_band(arc=0.36))
        p = os.path.join(d, 'a.npz')
        _atlas_npz(src_up, src_lo, 0.11, 0.11, p)
        # share OFF, own 0.07 >= min_h 0.05 -> the UPPER is NOT replaced
        up = _filled(_band())
        up.harvest_h = 0.07
        lo = _filled(_band(arc=0.36))
        lo.harvest_h = 0.11
        s = _share_stub(up, lo, lo_synth=False)
        s._fallback_tex(p, 0.05, share=False)
        assert not any(f['arch'] == 'up' for f in s.fallback) and not (up.src == 4).any()
        # share ON, own 0.07 < src 0.11 -> the UPPER is replaced by the sharp atlas
        up = _filled(_band())
        up.harvest_h = 0.07
        lo = _filled(_band(arc=0.36))
        lo.harvest_h = 0.11
        s = _share_stub(up, lo, lo_synth=False)
        s._fallback_tex(p, 0.05, share=True)
        assert any(f['arch'] == 'up' and f['share'] for f in s.fallback) and (up.src == 4).any()


def test_tex_share_keeps_the_own_harvest_when_it_is_already_the_sharper():
    """a shot whose own harvest is TALLER than the atlas keeps its own teeth under share (never trades detail down)."""
    with tempfile.TemporaryDirectory() as d:
        src_up = _filled(_band())
        src_lo = _filled(_band(arc=0.36))
        p = os.path.join(d, 'a.npz')
        _atlas_npz(src_up, src_lo, 0.08, 0.08, p)
        up = _filled(_band())
        up.harvest_h = 0.12
        lo = _filled(_band(arc=0.36))
        lo.harvest_h = 0.11
        s = _share_stub(up, lo, lo_synth=False)
        s._fallback_tex(p, 0.05, share=True)
        assert not (up.src == 4).any() and not any(f['arch'] == 'up' for f in s.fallback)


def test_tex_share_pulls_the_sharper_atlas_even_at_equal_exposed_height():
    """the key Part-2 case: a close-up atlas is SHARPER (higher gradient energy) than a mid-shot whose exposed teeth are the
    same IOD height. Share loads it on the sharpness criterion (why='sharper'), not height."""
    with tempfile.TemporaryDirectory() as d:
        src_up = _grating(_band(), period_tx=3)
        src_lo = _grating(_band(arc=0.36), period_tx=3)   # crisp detail
        p = os.path.join(d, 'a.npz')
        _atlas_npz(src_up, src_lo, 0.08, 0.08, p, up_detail=40.0, lo_detail=40.0)
        up = _filled(_band())
        up.harvest_h = 0.08
        lo = _filled(_band(arc=0.36))
        lo.harvest_h = 0.08   # flat (detail ~0), same height
        s2 = _share_stub(up, lo, lo_synth=False)
        s2._fallback_tex(p, 0.05, share=True)
        e = [f for f in s2.fallback if f['arch'] == 'up']
        assert e and e[0]['why'] == 'sharper' and (up.src == 4).any()


# ------------------------------------------------------------------------- Part 4: the edge-aware detail pass --
def test_detail_pass_boosts_high_frequency_and_preserves_the_mean():
    """the detail pass sharpens the crown's structure (raises column-to-column L variation) while preserving the mean L over
    the teeth -- so, unlike brightening, it does not re-widen a brightness-thresholded span."""
    band = _band(texel=2.0)
    # a soft (blurred) grating over the crown: fine structure that the detail pass should crisp up
    T = np.zeros((band.Th, band.Tw, 4), np.float32)
    r_e = (0.0 - band.h0) * band.texel - 0.5
    th = int(round(0.11 * 100.0 * band.texel))
    rows = np.arange(band.Th)[:, None]
    d = (r_e - rows)
    crown = np.broadcast_to((d >= -0.5) & (d <= th), T.shape[:2])
    base = 150.0 + 40.0 * np.sin(np.arange(band.Tw) * 0.6)[None, :]           # a smooth ripple
    T[..., 0] = T[..., 1] = T[..., 2] = np.where(crown, base, 0.0)
    T[..., 3] = np.where(crown, 1.0, 0.0)
    Lb = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    m = crown & (T[..., 3] > 0.5)
    hf0 = float(np.abs(np.diff(Lb, axis=1))[m[:, 1:] & m[:, :-1]].mean())
    mean0 = float(Lb[m].mean())
    band.detail_pass(T, 100.0, True, r_e, th, amount=1.0)
    La = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    hf1 = float(np.abs(np.diff(La, axis=1))[m[:, 1:] & m[:, :-1]].mean())
    mean1 = float(La[m].mean())
    assert hf1 > 1.05 * hf0, (hf0, hf1)              # crisper
    assert abs(mean1 - mean0) < 3.0                   # mean preserved (no span-widening brightening)


def test_detail_pass_only_crown_texels_and_zero_is_noop():
    band = _band(texel=2.0)
    T, r_e, th = _grey_crown(band)
    rows = np.arange(band.Th)[:, None]
    d = (r_e - rows)
    gum = np.broadcast_to((d > th + 1), T.shape[:2]) & (T[..., 3] > 0.5)
    trans = T[..., 3] <= 0.5
    bg = T[..., :3][gum].copy()
    bt = T[..., :3][trans].copy()
    T0 = T.copy()
    band.detail_pass(T, 100.0, True, r_e, th, amount=0.0)
    assert np.allclose(T, T0, atol=1.0)               # zero amount = no-op (within LAB round-trip)
    band.detail_pass(T, 100.0, True, r_e, th, amount=1.0)
    assert np.array_equal(T[..., :3][gum], bg) and np.array_equal(T[..., :3][trans], bt)
