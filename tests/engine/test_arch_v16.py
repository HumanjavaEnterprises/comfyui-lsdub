"""v16 / v16.1 teeth (mouth_layers): the lower arch as the dim, half-hidden, soft-bottomed sibling of the uppers, not their
mirror (--warp-lower-real). Opt-in; the v15 lowers are bit for bit when off. v16.1: the lower-teeth atlas is built v15-BRIGHT
and the shadow is applied PER FRAME, gated by the take's own lower display -- so a bright smile row is not dragged down by a
baked darkening, while the grill still recedes where the take hides the lowers. Synthetic, CPU only, our own classical graphics."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_mouth_layers import _comp


def _lo_frame(comp, h=40, w=40, L=200.0):
    """a frame-space lower arch: a flat bright crown (lo_rgb), fully opaque (lo_a), and a height map lo_h running 0 (the biting
    edge, top rows) to 0.15 IOD (toward the gum, bottom rows)."""
    io = comp.t.iod
    lo_rgb = np.full((h, w, 3), L, np.float32)
    lo_a = np.ones((h, w), np.float32)
    lo_h = np.tile(np.linspace(0.0, 0.15 * io, h)[:, None], (1, w)).astype(np.float32)
    return lo_rgb, lo_a, lo_h, io


# --------------------------------------------------------- v16.1: the per-frame lower shadow is gated by the take-show --
def test_lower_shadow_is_a_noop_where_the_take_shows_a_row():
    """the founder's 'match the take': on a frame where the take bares a full lower row (show = 1), the per-frame shadow does
    NOTHING -- the crown stays v15-bright, no darkening, no alpha feather."""
    comp, *_ = _comp(lower_real=0.9)
    lo_rgb, lo_a, lo_h, io = _lo_frame(comp)
    comp._lshow[3] = 1.0
    r, a = comp._lower_shadow(lo_rgb.copy(), lo_a.copy(), lo_h, 3, io)
    assert np.allclose(r, lo_rgb) and np.allclose(a, lo_a)


def test_lower_shadow_darkens_and_feathers_where_the_take_hides_the_lowers():
    comp, *_ = _comp(lower_real=0.9)
    lo_rgb, lo_a, lo_h, io = _lo_frame(comp)
    comp._lshow[4] = 0.0                                              # the take hides the lowers -> full shadow
    r, a = comp._lower_shadow(lo_rgb.copy(), lo_a.copy(), lo_h, 4, io)
    assert r[:3].mean() < lo_rgb[:3].mean() - 10.0                   # the biting tips (top rows, small d) darken into the cavity
    assert a[0].mean() < 0.9                                         # the top edge alpha feathers (no hard opaque line)
    assert np.allclose(a[-1], lo_a[-1])                              # the crown body / gum (large d) keeps its alpha


def test_lower_crown_is_brighter_when_the_take_shows_a_row_than_when_it_hides_it():
    """the v16.1 recovery, as a unit: the incisal zone renders BRIGHTER on a high-take-show frame than on a low-take-show one."""
    comp, *_ = _comp(lower_real=0.9)
    lo_rgb, lo_a, lo_h, io = _lo_frame(comp)
    comp._lshow[1] = 1.0
    hi, _ = comp._lower_shadow(lo_rgb.copy(), lo_a.copy(), lo_h, 1, io)
    comp._lshow[2] = 0.0
    lo, _ = comp._lower_shadow(lo_rgb.copy(), lo_a.copy(), lo_h, 2, io)
    assert hi[:5].mean() > lo[:5].mean() + 10.0


def test_lower_shadow_scales_smoothly_with_the_take_show():
    comp, *_ = _comp(lower_real=0.9)
    lo_rgb, lo_a, lo_h, io = _lo_frame(comp)
    tips = []
    for k, s in enumerate((1.0, 0.5, 0.0)):                          # show 1 -> 0.5 -> 0 (shadow 0 -> 0.5 -> 1)
        comp._lshow[10 + k] = s
        r, _ = comp._lower_shadow(lo_rgb.copy(), lo_a.copy(), lo_h, 10 + k, io)
        tips.append(r[:3].mean())
    assert tips[0] > tips[1] > tips[2]                               # darker as the take shows less (take-show-gated, not constant)


# ------------------------------------------------------------------------------------------- off = byte-identical --
def test_lower_atlas_is_v15_bright_whether_or_not_lower_real_is_on():
    """v16.1: --warp-lower-real no longer bakes any darkening into the lower atlas -- it is identical on and off (the shadow is
    a per-frame pass)."""
    off, *_ = _comp()
    on, *_ = _comp(lower_real=0.9)
    assert np.array_equal(off.lo.tex, on.lo.tex)
    assert off.lower_p is None and off.lower_target is None          # off = no v16 state at all


# --------------------------------------------------------- the per-take ratio drives the dim target (founder's insight) --
def test_per_take_ratio_is_measured_and_maps_to_a_target():
    comp, *_ = _comp(lower_real=0.9)
    assert comp.lower_ratio_src == 'auto' and 0.05 <= comp.lower_ratio <= 1.2
    assert 0.40 <= comp.lower_target <= 0.78


def test_higher_lower_ratio_gives_a_brighter_target():
    lo, *_ = _comp(lower_real=0.9, lower_ratio='0.15')               # a top-teeth talker
    hi, *_ = _comp(lower_real=0.9, lower_ratio='0.90')               # a bottom-teeth talker
    assert lo.lower_ratio_src == 'override' and abs(lo.lower_ratio - 0.15) < 1e-6
    assert hi.lower_target > lo.lower_target


def test_lower_display_ratio_follows_the_takes_own_stats():
    comp, *_ = _comp(lower_real=0.9)
    st = comp.t.stats
    io2 = float(comp.t.iod) ** 2
    top = [dict(up_area=0.02 * io2, lo_area=0.002 * io2, apart=0.10, in_area=0.05 * io2, sharp=1.0, apart_dark=0.08) for _ in st]
    bot = [dict(up_area=0.02 * io2, lo_area=0.016 * io2, apart=0.10, in_area=0.05 * io2, sharp=1.0, apart_dark=0.08) for _ in st]
    comp.t.stats = top
    r_top = comp._lower_display_ratio()
    comp.t.stats = bot
    r_bot = comp._lower_display_ratio()
    assert r_bot > r_top + 0.3


# ------------------------------------------------------------ per-frame depth: dim open, keep smile, cast shadow --
def test_lower_depth_dims_frames_where_the_take_hides_the_lowers():
    comp, *_ = _comp(lower_real=0.9)
    io = comp.t.iod
    h = w = 40
    lo_rgb = np.full((h, w, 3), 200.0, np.float32)
    lo_a = np.ones((h, w), np.float32)
    comp._lshow[0] = 0.0                                             # the take hides the lowers at this frame (an 'ah' / 'oo')
    out = comp._lower_depth(lo_rgb.copy(), lo_a, 0, h, w, io)
    assert out.mean() < lo_rgb.mean() * 0.65                         # the take hides them -> the lowers recede (clearly dimmer)


def test_lower_depth_leaves_the_lowers_bright_where_the_take_shows_a_row():
    comp, *_ = _comp(lower_real=0.9)
    io = comp.t.iod
    h = w = 40
    lo_rgb = np.full((h, w, 3), 200.0, np.float32)
    lo_a = np.ones((h, w), np.float32)
    comp._lshow[0] = 1.0                                             # the take shows a full lower row here (a smile)
    comp._open = 0.95
    out = comp._lower_depth(lo_rgb.copy(), lo_a, 0, h, w, io)
    assert abs(out.mean() - lo_rgb.mean()) < 2.0                     # kept bright (matched), not over-dimmed


def test_lower_depth_casts_a_shadow_on_the_biting_edge():
    comp, *_ = _comp(lower_real=0.9)
    io = comp.t.iod
    h = w = 40
    lo_rgb = np.full((h, w, 3), 200.0, np.float32)
    lo_a = np.zeros((h, w), np.float32)
    lo_a[10:30, :] = 1.0        # the lowers occupy rows 10..29; row 10 is the biting edge
    comp._lshow[0] = 0.0
    comp._open = 0.95
    out = comp._lower_depth(lo_rgb.copy(), lo_a, 0, h, w, io)
    assert out[11, :, :].mean() < out[28, :, :].mean() - 3.0         # the top (biting edge) is darker than the base (cast shadow)


# ------------------------------------------------------------------- the tongue occludes the lowers wide-open --
def test_cover_amt_keeps_the_tongue_over_the_lowers_wide_open_where_the_take_hides_them():
    on, *_ = _comp(tongue='wedge', tongue_cover=0.9, lower_real=0.9)
    off, *_ = _comp(tongue='wedge', tongue_cover=0.9)
    on._lshow[0] = 0.0                                               # the take hides the lowers wide-open ('ah' / 'oo')
    assert on._cover_amt(0, op=0.95) > 0.15                          # the tongue stays partly over the lowers (no grill)
    assert off._cover_amt(0, op=0.95) < 0.1                          # v15: drops to ~0 (the lowers flash)
    assert on._cover_amt(0, op=0.08) < 0.1                           # near-closed: still a flash (wide = 0 there)
    on._lshow[0] = 1.0                                               # but where the take bares a real smile row, do NOT occlude it
    assert on._cover_amt(0, op=0.95) < 0.1
