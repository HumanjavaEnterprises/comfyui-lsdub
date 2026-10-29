#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_layers.py: v7 layered mouth compositor for --mode warp (--warp-compose layers). All CPU.

Founder on v6: "uncanny valley ... the lower teeth look like just the middle 2 or 4 ... build it up as layers". So the
mouth is an explicit stack of RGBA layers, composited back to front; depth / occlusion come from the ORDER and the
target inner-lip mask, never from blending one mouth into another:

  L0 frame       the take's own frame
  L1 cover       the take's whole mouth (lips + teeth + interior, its own landmarks) erased to its own skin
                 (lip_layers.erase_lips, a harmonic membrane + grain): nothing of the old mouth survives under a later
                 layer; it shows wherever no later layer is opaque (e.g. old lips the warp left outside the new ones)
  L2 cavity      the back of the mouth: the take's dark interior tone with depth (darker at the top / back, a faint warm
                 throat low in the centre, darker toward the cheeks); outline = the target inner-lip contour expanded
                 slightly behind the lips
  L3 lower       the tongue (a soft ellipsoid behind / under the lower incisors) and the FULL lower arch: a curved 3D
                 band (parabolic arch in head space: incisors, canines, premolars curving back on each side), textured
                 from the take's harvested lower teeth, mirrored / extended to the sides, darkened toward the back. Rides
                 the skull pose + the planned jaw (rotation about the condyle hinge) + the upper-lip-root pin + the CHIN
                 pin (the lower teeth take the chin skin's own frame-to-frame motion)
  L4 upper       the full upper arch, the same kind of band, fixed to the skull + the upper-lip-root pin (v6)
  L5 lips        the warped lips only (outer lip .. target inner-lip contour), with a darker inner rim (the lip turning
                 inward: it visibly sits in front of the teeth); soft contact shadows of the lips on the teeth and the
                 cavity edge are applied to L2..L4
  L6 face        the surrounding warped skin (the warp's C2 falloff to the untouched frame is in it); old-lip pixels the
                 warp left outside the new outer lip are dropped so L1 shows there

The arches and the cavity are defined in the head (canonical) frame and rasterised straight into the frame through the
rigid pose (+ jaw, + pins): one resampling, the same geometry as a head-stabilised composite without a second resample
of the frame. The debug strip shows every layer head-stabilised (--warp-layers-debug).

v8 (polish pass on the v7 stack; founder: "the closest one yet, keep grinding"). Every change opt-in, v7 is the default:
  PIN      --warp-teeth-chin-pin arc: the chin flow is split into its slow part (consistent with the planned jaw) and
           the fast residual (the take's own fast jaw motion the warp leaves in the chin); the lower arch follows the
           slow part fully and the fast part by --warp-chin-pin-gain (zero-phase smoothed), and the pin is PROJECTED
           ONTO THE JAW HINGE'S ARC (an extra rotation about the condyle, never a translation off the arc). The
           exposure limit is folded into that angle, so the per-frame clamp has nothing left to catch.
  TEETH    --warp-arch-harvest N: the arch texture is the median of the N best harvest frames in the same head-pose
           bin (registered on the canonical plane), at --warp-arch-texel texels per canonical px; --warp-arch-synth:
           side teeth beyond the harvest are MODELLED (rounded crown, dark interproximal line, enamel highlight on the
           shot's key-light side, premolar / molar widths) instead of column-repeated, and darken + desaturate toward
           the back; --warp-arch-tex-from: a fallback arch texture from another shot of the same character (scaled by
           IOD, colour-matched) when this shot's own harvest is only a few px tall; --warp-arch-tex-out writes one.
  TONGUE   --warp-tongue wedge: a modelled tongue (tip behind the lower incisors + a dorsum further back, riding the
           jaw), rising on narrow vowels (E / ih), dropping and retracting on open ones (aa / oh / ou), a wet highlight;
           never above the lower incisal edge on near-closed frames.
  LIPS     --warp-lip-rim S,W: the inner rim's strength / width; --warp-lip-wet: a 1-2 px soft wet edge where the lips
           occlude the teeth instead of a hard cut.
  COVER    --warp-cover fill: an old-lip blob the warp left outside the new lips is filled with skin IN THE WARPED
           FRAME (harmonic membrane + grain from around it), so no seam between the bald cover and the warped face.
  SMOOTH   --warp-layers-smooth: zero-phase ease (frames) of the per-frame layer parameters (opening, tongue).

v11 (opt-in unless noted):
  CORNERS  --warp-cover-corners K: the cover grows at the two mouth corners with the plan (how far it moves each outer
           corner past the take's, half the opening increase, 0.15 IOD x |sin yaw| -- the far corner in full), x K: the
           L1 cover's hull (lip_layers.corner_points), and the L6 fill also takes the old-mouth material the warp MOVED
           onto what was skin in the take there (corner_residue). v10c's wider openings showed the take's old lip / cavity
           at turned-head corners (mv30 NC f74-76, S8): the search stopped at the take's own (unwarped) lip hull.
  CLIP     --warp-interior-clip: the interior (L2 cavity, L3 lower arch + tongue, L4 upper arch) is only ever seen THROUGH
           the warped lips: the planned opening is kept where the warped frame has lip pixels above AND below it along the
           mouth's own vertical (enclosed), feathered <= 1 px; an opening the warped lips do not enclose shows the warped
           face (L6) instead. No cavity or tooth pixel can sit outside the lips, whatever placed the opening there.
  REG      (always, logging only) the inter-layer registration of every frame, in the skull-stabilised frame (canonical
           plane of the upper teeth, the skull pose only, IOD units): upper / lower incisal edge (centre + two ends),
           cavity centroid, tongue tip, the warped inner-lip contour (top / bottom / corners) and its centroid, the jaw
           angle, the upper (lip-root skin) pin, the chin pin, the exposure kick and the head pose -> warp.json frames[i].reg
           (lsdub layer-reg reads it).

v12 (opt-in):
  SINGLE   --warp-reg single: ONE reference for the interior (plan_single): the upper arch on the skull pose only, the lower
           arch + tongue on the skull pose + one low-passed hinge angle from the planned opening, the tongue a slow function
           of that angle, the cavity the planned opening; no pins, no per-frame exposure kick; gum guards. The lips, cover
           and face are unchanged. --warp-reg v11 (the default) is the v11 placement, bit for bit.

v13 (opt-in; founder: "the teeth are wider by 1/5th or so in comparison to the head, only on close ups"):
  LAYOUT   --warp-arch-layout anthro: the crowns' widths along the arc are the anthropometric table's (ANTHRO, per the 63 mm
           IOD = the face's own inter-pupil distance), symmetric about the midline; the harvested run is resampled onto those
           slots (its interdental lines -> the slot boundaries, a monotone piecewise-linear x-warp; one capped uniform stretch
           when fewer than 3 lines are found), then completed as before (crown / gum / mirror; the teeth beyond the run fill the
           slots: synth_slots). The arc lengths are unchanged. Logged: teeth.layers.layout.
  CORRIDOR --warp-arch-corridor K: the buccal corridor. Per frame, the plates' colour beyond +-(0.5 - K) W_f of the planned
           opening's centre (W_f its width corner to corner) recedes into the cavity's dark with a smootherstep, to a faint
           crown at the corners; never inside that span, never L5 / L6. Logged per frame (frames[i].reg.corridor) and as
           teeth.layers.corridor (incl. the frames whose arch ends short of the corridor: arc_short).
"""
import os
from ..io import ff
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

from . import lip_layers as ll
from . import teeth_layer as tl

LIP_IN, LIP_OUT = ll.LIP_IN, ll.LIP_OUT
CHIN = [18, 200, 199, 175, 83, 313, 201, 421, 182, 406]          # soft chin below the lower lip (rides the mandible)
def aa_poly(pts, shape, px=0.0, ss=4):
    """v17e: anti-aliased polygon coverage 0..1 (rasterised at ss x and area-averaged down), dilated by px; a plain
    fillPoly + small blur left a stair-step along the thin corner wedges of the opening."""
    H, W = shape[:2]
    m = np.zeros((H * ss, W * ss), np.uint8)
    cv2.fillPoly(m, [np.round((np.asarray(pts, float)[:, :2] + 0.5) * ss - 0.5).astype(np.int32)], 255, cv2.LINE_AA)
    k = int(round(abs(px) * ss))
    if k > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return cv2.resize(m.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)


NAMES = ('L0 frame', 'L1 mouth cover', 'L2 back of mouth', 'L3 lower teeth + tongue', 'L4 upper teeth', 'L5 lips', 'L6 face area')
REG_END = 0.18           # v11 reg: the incisal edge "ends" logged at +-0.18 IOD of arc from the midline (the canines)
smootherstep = tl.smootherstep
# vowel classes for the tongue: (dorsum rise, tip rise, retraction) in IOD; narrow vowels raise the tongue body, open
# ones drop and retract it
TONGUE_VOWEL = {'E': (0.035, 0.012, -0.01), 'ih': (0.045, 0.015, -0.015), 'aa': (-0.03, -0.01, 0.03), 'oh': (-0.02, -0.005, 0.035),
                'ou': (-0.01, 0.0, 0.03), 'RR': (0.01, 0.015, 0.0), 'DD': (0.02, 0.02, 0.0), 'nn': (0.02, 0.02, 0.0), 'TH': (0.0, 0.02, -0.02),
                'kk': (0.03, -0.01, 0.03)}


# ------------------------------------------------------------------------------------------------- the arch --
def arch_table(half_arc, k, n=2000):
    """parabolic arch z = k x^2 (x lateral, z back) tabulated by ARC LENGTH a >= 0 (so a tooth keeps its width along the
    curve). Returns (A, X, Z) arrays for np.interp."""
    xmax = half_arc * 1.05
    x = np.linspace(0, xmax, n)
    ds = np.sqrt(1 + (2 * k * x) ** 2)
    A = np.concatenate([[0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(x))])
    return A, x, k * x ** 2


def tooth_period(T, tk, r_e, th, iod, texel, upper, default):
    """(period in texels, highlight position -1..1 across a crown, n teeth) from the interdental minima of the L profile
    along the band just inside the incisal edge over the harvested tooth columns; the default period when fewer than 3
    gaps are found. The highlight position = where across a tooth it is brightest (the shot's key-light side)."""
    from scipy.signal import find_peaks
    Th, Tw = T.shape[:2]
    e = int(round(r_e))
    a0, a1 = (max(0, e - int(0.05 * iod * texel)), max(1, e - int(0.012 * iod * texel))) if upper else (min(Th - 1, e + int(0.012 * iod * texel)), min(Th, e + int(0.05 * iod * texel)))
    if a1 - a0 < 1: return default, 0.0, 0
    L = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    band = L[a0:a1]
    kb = tk[a0:a1]
    cnt = kb.sum(0)
    prof = np.where(cnt > 0, (band * kb).sum(0) / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(prof)
    if ok.sum() < 0.1 * iod * texel: return default, 0.0, 0
    xs = np.nonzero(ok)[0]
    x0, x1 = xs.min(), xs.max()
    pr = prof[x0:x1 + 1]
    o2 = np.isfinite(pr)
    pr = np.interp(np.arange(len(pr)), np.nonzero(o2)[0], pr[o2])
    pr = gaussian_filter1d(pr, max(0.5, 0.006 * iod * texel))
    hp = pr - gaussian_filter1d(pr, 0.05 * iod * texel)
    pk, _ = find_peaks(-hp, distance=max(2, int(0.04 * iod * texel)), prominence=max(1.5, 0.35 * hp.std()))
    if len(pk) < 3: return default, 0.0, len(pk) + 1
    d = np.diff(pk)
    p = float(np.median(d))
    if not (0.05 * iod * texel <= p <= 0.12 * iod * texel): p = default
    hl = []
    for i in range(len(pk) - 1):
        seg = pr[pk[i]:pk[i + 1] + 1]
        if len(seg) >= 4: hl.append((np.argmax(seg) / max(1, len(seg) - 1)) * 2 - 1)
    x_hl = float(np.clip(np.median(hl), -0.5, 0.5)) if hl else 0.0
    return p, x_hl, len(pk) + 1


def synth_teeth(T, src, lo, hi, p, iod, texel, upper, r_e, th, x_hl=0.0, line=0.75, hl=0.10):
    """model the teeth beyond the harvested run [lo, hi] (texel columns) instead of tiling it: each tooth is a crown of
    width w (canine 0.95 p, premolars 0.85 p, molars 1.15 p, p = the incisor period) built on the harvested run's own
    per-row colour (gum line, body shading and the tips' tone are its), shaded as a rounded crown (darker toward its
    sides), with a dark interproximal line at its front edge, an enamel highlight on the key-light side (x_hl) and a
    rounded incisal corner. Returns the number of modelled columns."""
    Th, Tw = T.shape[:2]
    rows = np.arange(Th, dtype=np.float32)
    run = T[:, lo:hi + 1]
    tk = run[..., 3] > 0.5
    base = np.array([np.median(run[r, tk[r], :3], 0) if tk[r].sum() >= 3 else np.median(run[r, :, :3], 0) for r in range(Th)], np.float32)
    abase = np.median(run[..., 3], 1).astype(np.float32)
    # the body rows: the column's alpha profile; a faint vertical gradient (darker toward the root) as the harvest has
    widths = [0.95, 0.85, 0.85, 1.15, 1.15, 1.15]
    done = 0
    for side in (1, -1):
        c = hi + 1 if side > 0 else lo - 1
        k = 0
        while 0 <= c < Tw:
            wi = max(3, int(round(widths[min(k, len(widths) - 1)] * p)))
            back = min(1.0, k / 4.0)
            for j in range(wi):
                cc = c + side * j
                if not (0 <= cc < Tw): break
                x = (j + 0.5) / wi * 2 - 1
                x = x if side > 0 else -x
                crown = 0.78 + 0.22 * np.sqrt(max(0.0, 1 - x * x))
                h_ = hl * (1 - 0.6 * back) * np.exp(-((x - x_hl) / 0.35) ** 2)
                col = base * (crown + h_)
                if j == 0: col = col * line                                       # the interproximal line
                a = abase.copy()
                d = 0.18 * p * (1 - np.sqrt(max(0.0, 1 - x * x)))   # rounded incisal corner
                if upper: a[(rows > r_e - d) & (rows <= r_e + 1)] = 0.0
                else: a[(rows < r_e + d) & (rows >= r_e - 1)] = 0.0
                T[:, cc, :3] = np.clip(col, 0, 255)
                T[:, cc, 3] = a
                src[cc] = 3
                done += 1
            c += side * wi
            k += 1
    return done


# ---------------------------------------------------------------------- v13: the anthropometric layout (--warp-arch-layout) --
# mesiodistal crown widths ALONG THE ARC, per the anthropometric IOD = the inter-PUPIL distance (63 mm; the founder's table,
# mv30 v13): upper central, lateral, canine, premolar 1, premolar 2, molar 1 (+ molar 2, 9 mm: beyond the table, only reached
# by a widened arc); lower central, lateral, canine, premolars (+ molars 1 / 2, 11 / 10 mm: beyond the table). lsdub's own IOD
# (|Cm[263] - Cm[33]|, the outer eye corners) is ~1.45 x that on these faces, so the table is scaled by the face's own
# inter-pupil distance (the iris centres 468 / 473 of the median canonical face), not by lsdub's IOD.
ANTHRO = dict(up=(0.1375, 0.105, 0.12, 0.11, 0.11, 0.16, 0.143), lo=(0.087, 0.095, 0.11, 0.11, 0.11, 0.175, 0.159))
ANTHRO_NAMES = ('central', 'lateral', 'canine', 'premolar 1', 'premolar 2', 'molar 1', 'molar 2')
IPD_OCD = 0.69           # the inter-pupil / outer-canthal ratio (adults; these faces measure 0.68-0.71): the fallback unit


def anthro_bounds(widths, unit, extent):
    """the slot boundaries from the midline (canonical px, ascending, starting at 0): the table's widths x unit, the last width
    repeated until extent is passed."""
    b = [0.0]
    k = 0
    while b[-1] <= extent:
        b.append(b[-1] + widths[min(k, len(widths) - 1)] * unit)
        k += 1
    return np.array(b)


def edge_lines(T, tk, r_e, iod, texel, upper, prom=(1.5, 0.35)):
    """v13: the interdental lines of a plate texture -- the minima of the L profile along the band just inside the incisal edge
    over the tooth columns (tooth_period's profile; its peak criteria: prominence >= max(prom[0], prom[1] x the high-passed
    profile's std)), sub-texel (a parabola through each minimum) -> dict(x0, x1 = the profiled run's columns, lines = texel
    columns (float), prom = their prominences) or None."""
    from scipy.signal import find_peaks
    Th, Tw = T.shape[:2]
    e = int(round(r_e))
    a0, a1 = (max(0, e - int(0.05 * iod * texel)), max(1, e - int(0.012 * iod * texel))) if upper else (min(Th - 1, e + int(0.012 * iod * texel)), min(Th, e + int(0.05 * iod * texel)))
    if a1 - a0 < 1: return None
    L = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    band = L[a0:a1]
    kb = tk[a0:a1]
    cnt = kb.sum(0)
    prof = np.where(cnt > 0, (band * kb).sum(0) / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(prof)
    if ok.sum() < 0.1 * iod * texel: return None
    xs = np.nonzero(ok)[0]
    x0, x1 = int(xs.min()), int(xs.max())
    pr = prof[x0:x1 + 1]
    o2 = np.isfinite(pr)
    pr = np.interp(np.arange(len(pr)), np.nonzero(o2)[0], pr[o2])
    pr = gaussian_filter1d(pr, max(0.5, 0.006 * iod * texel))
    hp = pr - gaussian_filter1d(pr, 0.05 * iod * texel)
    pk, pp = find_peaks(-hp, distance=max(2, int(0.04 * iod * texel)), prominence=max(prom[0], prom[1] * hp.std()))
    lines = []
    for k in pk:
        d = 0.0
        if 0 < k < len(hp) - 1:
            den = hp[k - 1] - 2 * hp[k] + hp[k + 1]
            if den > 1e-9: d = float(np.clip(0.5 * (hp[k - 1] - hp[k + 1]) / den, -0.5, 0.5))
        lines.append(x0 + k + d)
    return dict(x0=x0, x1=x1, lines=np.array(lines, float), prom=np.asarray(pp.get('prominences', np.zeros(len(pk))), float))


def anthro_map(det, Tw, texel, A, bounds, cap=(0.6, 1.8), default_p=None):
    """v13: the horizontal resample that puts the harvested run's teeth on the anthropometric slots. det = edge_lines(); bounds =
    anthro_bounds() (canonical px from the midline). Lines -> slot boundaries: the line nearest the midline takes its nearest
    boundary, then outward on each side every next line takes the next boundary (or the one after it, when the line between was
    missed, or the one after that: the stretch closer to 1 wins); a line whose segment would stretch beyond cap is skipped (spurious). Between the knots
    a monotone piecewise-linear map of arc length, outside them the median stretch of the segments. With fewer than 3 assigned
    lines: ONE uniform stretch about the midline, so the run's detected period (the median line spacing; default_p (texels)
    when fewer than 2 lines) becomes the slot width at the run's arc position, capped. -> (src columns (Tw,) float: the source
    column of each target column, log dict)."""
    a_of = lambda c: (np.asarray(c, float) + 0.5) / texel - A                 # texel column -> arc (canonical px)
    c_of = lambda a: (np.asarray(a, float) + A) * texel - 0.5
    Bs = np.concatenate([-bounds[:0:-1], bounds])
    at = a_of(np.arange(Tw))
    log = dict(cap=list(cap))
    lines = np.sort(a_of(det['lines'])) if det is not None and len(det['lines']) else np.zeros(0)
    log['lines'] = lines.tolist()
    knots = []
    if len(lines):
        k0 = int(np.argmin(np.abs(lines)))
        j0 = int(np.argmin(np.abs(Bs - lines[k0])))
        knots = [(lines[k0], Bs[j0], j0)]
        skipped = []
        for side in (1, -1):
            xc, bc, jc = knots[0]
            ks = range(k0 + 1, len(lines)) if side > 0 else range(k0 - 1, -1, -1)
            for k in ks:
                best = None
                for j in (jc + side, jc + 2 * side, jc + 3 * side):
                    if not (0 <= j < len(Bs)): continue
                    s = (Bs[j] - bc) / (lines[k] - xc)
                    if cap[0] <= s <= cap[1] and (best is None or abs(np.log(s)) < abs(np.log(best[1]))): best = (j, s)
                if best is None:
                    skipped.append(float(lines[k]))
                    continue
                xc, bc, jc = lines[k], Bs[best[0]], best[0]
                knots.append((xc, bc, jc))
        log['skipped'] = skipped
    knots.sort(key=lambda q: q[0])
    if len(knots) >= 3:
        xs = np.array([q[0] for q in knots])
        bs = np.array([q[1] for q in knots])
        st = np.diff(bs) / np.diff(xs)
        sm = float(np.median(st))
        src_a = np.interp(at, bs, xs)                                               # inside the knots: piecewise linear
        src_a = np.where(at < bs[0], xs[0] + (at - bs[0]) / sm, src_a)
        src_a = np.where(at > bs[-1], xs[-1] + (at - bs[-1]) / sm, src_a)
        log.update(mode='lines', knots_src=xs.tolist(), knots_slot=bs.tolist(), stretch=st.tolist(), stretch_med=sm)
    else:
        r0, r1 = (float(a_of(det['x0'])), float(a_of(det['x1']))) if det is not None else (0.0, 0.0)
        if len(lines) >= 2:
            p = float(np.median(np.diff(lines)))
            pk = 'detected'
        elif len(lines) == 1 and r0 < lines[0] < r1 and abs(lines[0] - 0.5 * (r0 + r1)) < 0.25 * (r1 - r0):
            p = 0.5 * (r1 - r0)
            pk = 'run'       # one line in the middle of the run: the run is the two teeth it separates
        else:
            p = float(default_p) / texel if default_p else None
            pk = 'default'
        if p is None or p <= 0:
            log.update(mode='none')
            return c_of(at), log
        pos = float(np.mean(lines)) if len(lines) else float(a_of(det['x0']) + a_of(det['x1'])) / 2 if det is not None else 0.0
        k = int(np.clip(np.searchsorted(bounds, abs(pos), side='right') - 1, 0, len(bounds) - 2))
        w = float(bounds[k + 1] - bounds[k])
        s = float(w / p)
        sc = float(np.clip(s, *cap))
        src_a = at / sc
        log.update(mode='uniform', period=p, period_from=pk, at=pos, slot=k, slot_w=w, stretch_raw=s, stretch_med=sc, capped=bool(sc != s))
    return c_of(src_a), log


def synth_slots(T, src, lo, hi, bounds, A, iod, texel, upper, r_e, x_hl=0.0, line=0.75, hl=0.10):
    """v13 (--warp-arch-layout anthro): synth_teeth with the layout's widths -- the teeth beyond the covered run [lo, hi] fill the
    anthropometric slots (bounds: canonical px from the midline; the left side their mirror) instead of 0.95 / 0.85 / 1.15 x
    the harvested period. Each slot is one modelled crown (synth_teeth's crown, highlight and rounded incisal corner over the
    WHOLE slot); a slot the run already partly covers is completed beyond it without a second interproximal line. Returns the
    number of modelled columns."""
    Th, Tw = T.shape[:2]
    rows = np.arange(Th, dtype=np.float32)
    run = T[:, lo:hi + 1]
    tk = run[..., 3] > 0.5
    base = np.array([np.median(run[r, tk[r], :3], 0) if tk[r].sum() >= 3 else np.median(run[r, :, :3], 0) for r in range(Th)], np.float32)
    abase = np.median(run[..., 3], 1).astype(np.float32)
    done = 0
    c_of = lambda a: (a + A) * texel - 0.5
    for side in (1, -1):
        k_s = 0
        for k in range(len(bounds) - 1):
            e0, e1 = sorted((c_of(side * bounds[k]), c_of(side * bounds[k + 1])))       # the slot's image-left / right column edges
            mesial = int(np.ceil(e0)) if side > 0 else int(np.floor(e1))
            cols = [cc for cc in range(int(np.ceil(e0)), int(np.floor(e1)) + 1) if 0 <= cc < Tw and (cc > hi if side > 0 else cc < lo)]
            if not cols: continue
            back = min(1.0, k_s / 4.0)
            w = max(2.0, e1 - e0)
            k_s += 1
            for cc in cols:
                x = float(np.clip((cc - e0) / max(1e-6, e1 - e0) * 2 - 1, -1, 1))
                crown = 0.78 + 0.22 * np.sqrt(max(0.0, 1 - x * x))
                h_ = hl * (1 - 0.6 * back) * np.exp(-((x - x_hl) / 0.35) ** 2)
                col = base * (crown + h_)
                if cc == mesial: col = col * line                                    # the interproximal line (a fresh tooth)
                a = abase.copy()
                d = 0.18 * w * (1 - np.sqrt(max(0.0, 1 - x * x)))   # rounded incisal corner
                if upper: a[(rows > r_e - d) & (rows <= r_e + 1)] = 0.0
                else: a[(rows < r_e + d) & (rows >= r_e - 1)] = 0.0
                T[:, cc, :3] = np.clip(col, 0, 255)
                T[:, cc, 3] = a
                src[cc] = 3
                done += 1
    return done


def trim_run(lo, hi, bounds, A, texel, keep=0.5):
    """v13 (anthro): the covered run [lo, hi] (texel columns) cut back to a slot boundary on a side where it covers less than
    `keep` of its outermost slot (a sliver -- the harvest's feathered, see-through edge columns -- would print a two-tone crown
    between two lines); synth_slots then models that slot whole. -> (lo, hi, trimmed texels (left, right))."""
    c_of = lambda a: (a + A) * texel - 0.5
    a_of = lambda c: (c + 0.5) / texel - A
    b = np.asarray(bounds, float)
    out = [0, 0]
    for side in (1, -1):
        e = hi if side > 0 else lo
        a = abs(a_of(e)) + 0.5 / texel          # the run's outer edge, arc from the midline
        k = int(np.searchsorted(b, a, side='right') - 1)
        if k < 1 or k + 1 >= len(b): continue                               # (never into the centrals; past the table: as it is)
        if (a - b[k]) / (b[k + 1] - b[k]) < keep:           # the run then ends right before the slot's first column (its line)
            c = int(np.ceil(c_of(b[k]))) - 1 if side > 0 else int(np.floor(c_of(-b[k]))) + 1
            if side > 0 and c < hi:
                out[1] = hi - c
                hi = c
            elif side < 0 and c > lo:
                out[0] = c - lo
                lo = c
    return lo, hi, out


class ArchBand:
    """one dental arch as a curved band in the canonical head frame: X(a, h) = origin + u x(a) + b z(a) + d h, a = arc
    length from the midline (- = the face's image-left), h = along the head's down axis from the incisal edge (+ down).
    The band is ruled (vertical lines along d), so every column quad is planar and its projection under the orthographic
    rigid pose is exactly affine. Texture: BGRA float (Th x Tw), texel = canonical px / texel."""

    def __init__(self, origin, u, d, b, half_arc, k, h0, h1, texel=1.0):
        self.o, self.u, self.d, self.b = (np.asarray(v, float) for v in (origin, u, d, b))
        self.A, self.k, self.h0, self.h1, self.texel = float(half_arc), float(k), float(h0), float(h1), float(texel)
        self.tab = arch_table(self.A, self.k)
        self.Tw = max(4, int(round(2 * self.A * texel)))
        self.Th = max(4, int(round((h1 - h0) * texel)))
        self.a = (np.arange(self.Tw) + 0.5) / texel - self.A
        self.h = (np.arange(self.Th) + 0.5) / texel + h0
        self.tex = np.zeros((self.Th, self.Tw, 4), np.float32)
        self.src = np.zeros(self.Tw, np.int8)   # 1 harvested, 2 mirrored, 3 extended
        self.harvest_h = 0.0
        self.period = None
        self.x_hl = 0.0
        self.n_teeth = 0
        self.synth_cols = 0
        self.mipmap = False
        self._mip = None
        self._mip_of = None    # v14: trilinear minification pyramid (opt-in, --warp-arch-mipmap)

    def mip_pyramid(self):
        """v14: a premultiplied-alpha mip pyramid of self.tex -- level 0 = the atlas itself, each next a 2x INTER_AREA
        downsample (area-correct: the premultiplied RGBA stays premultiplied under an area average, so transparent texels do
        not bleed dark into the crowns). Rebuilt only when self.tex changes (identity by id + shape). Returns [BGRA_premult, ...]."""
        key = (id(self.tex), self.tex.shape)
        if self._mip is not None and self._mip_of == key: return self._mip
        pm = self.tex.astype(np.float32).copy()
        a = np.clip(pm[..., 3:4], 0.0, 1.0)
        pm[..., :3] = pm[..., :3] * a
        pm[..., 3:4] = a
        levels = [pm]
        cur = pm
        for _ in range(6):                                    # <= 6 halvings is plenty (24 : 1)
            h, w = cur.shape[:2]
            if min(h, w) <= 4: break
            cur = cv2.resize(cur, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA).astype(np.float32)
            levels.append(cur)
        self._mip, self._mip_of = levels, key
        return levels

    def _warp_quad_mip(self, M, out_wh):
        """v14: warpAffine the atlas into a column quad with trilinear mip minification. M (2x3) maps dst px -> texel
        (WARP_INVERSE_MAP). The local minification = sqrt(|det of M's 2x2 linear part|) texels per dst px; level = log2 of it,
        two adjacent levels blended. Each level is premultiplied; the result is un-premultiplied to straight BGRA (what the
        raw-bilinear path returns), so the caller's 'over' composite is unchanged. Magnification (ratio < 1) -> level 0."""
        lv = self.mip_pyramid()
        A = np.asarray(M, np.float32)[:, :2]
        ratio = float(np.sqrt(max(abs(float(np.linalg.det(A))), 1e-12)))
        lvl = max(0.0, float(np.log2(max(ratio, 1e-6))))
        l0 = int(min(np.floor(lvl), len(lv) - 1))
        frac = float(lvl - l0)
        def warp(level):
            s = 0.5 ** level
            Ms = np.asarray(M, np.float32).copy()
            Ms[:2, :] *= s     # texel -> mip-level coord
            return cv2.warpAffine(lv[level], Ms, out_wh, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pm = warp(l0)
        if frac > 1e-3 and l0 + 1 < len(lv): pm = (1.0 - frac) * pm + frac * warp(l0 + 1)
        a = pm[..., 3:4]
        rgb = np.where(a > 1e-4, pm[..., :3] / np.maximum(a, 1e-4), 0.0)
        return np.dstack([rgb, pm[..., 3]]).astype(np.float32)

    def xz(self, a):
        a = np.asarray(a, float)
        A, X, Z = self.tab
        m = np.abs(a)
        return np.sign(a) * np.interp(m, A, X), np.interp(m, A, Z)

    def normal(self, a):
        """outward (front-facing) unit normal of the arch at a, canonical."""
        x, _ = self.xz(a)
        nx, nz = 2 * self.k * x, -np.ones_like(x)
        L = np.sqrt(nx ** 2 + nz ** 2)
        return (nx / L)[..., None] * self.u + (nz / L)[..., None] * self.b

    def points(self, a, h):
        x, z = self.xz(a)
        a = np.asarray(a, float)
        h = np.asarray(h, float)
        return self.o + x[..., None] * self.u + z[..., None] * self.b + h[..., None] * self.d

    def sample(self, img, alpha, proj):
        """(BGRA texture, src) sampled from one frame: proj(X (N x 3)) -> that frame's px (N x 2)."""
        aa, hh = np.meshgrid(self.a, self.h)
        X = self.points(aa.ravel(), hh.ravel())
        p = proj(X).reshape(self.Th, self.Tw, 2).astype(np.float32)
        rgb = cv2.remap(img, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        al = cv2.remap(np.asarray(alpha, np.float32), p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderValue=0)
        return np.dstack([rgb, al]).astype(np.float32), (al.max(0) > 0.5).astype(np.int8)

    def harvest(self, img, alpha, proj):
        """sample the band's texture from one frame."""
        self.tex, self.src = self.sample(img, alpha, proj)
        return self

    def harvest_many(self, sources, iod, max_shift=0.02, min_ncc=0.6, log=None):
        """v8: the texture = per-texel MEDIAN over several harvest frames [(img, alpha, proj, gain)], each registered to
        the first on the canonical plane (phase correlation of L over the shared tooth pixels; a frame that does not
        register is dropped); gain scales a frame's colour to the first's light. Returns the frames used."""
        texs = []
        T0 = None
        for n_, (img, alpha, proj, gain) in enumerate(sources):
            T, _ = self.sample(img, alpha, proj)
            T[..., :3] *= gain
            if T0 is None:
                T0 = T
                texs.append(T)
                continue
            L0 = cv2.cvtColor(np.clip(T0[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (T0[..., 3] > 0.5)
            L1 = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (T[..., 3] > 0.5)
            if (T[..., 3] > 0.5).sum() < 30: continue
            # registration: the best NCC over the shared tooth texels for integer shifts within max_shift (a phase
            # correlation of two mostly-empty textures locks onto the mask edges)
            r = max(1, int(round(max_shift * iod * self.texel)))
            best = (-2.0, 0, 0)
            m0 = T0[..., 3] > 0.5
            m1 = T[..., 3] > 0.5
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    L1s = cv2.warpAffine(L1, M, (self.Tw, self.Th))
                    m1s = cv2.warpAffine(m1.astype(np.uint8), M, (self.Tw, self.Th)) > 0
                    ov = m0 & m1s
                    if ov.sum() < 30: continue
                    c = float(np.corrcoef(L0[ov], L1s[ov])[0, 1])
                    if c > best[0]: best = (c, dx, dy)
            ncc, sx, sy = best
            if ncc < -1: 
                if log is not None:
                    log.append(dict(k=n_, why='no overlap'))
                    continue
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            Ts = cv2.warpAffine(T, M, (self.Tw, self.Th), flags=cv2.INTER_LINEAR, borderValue=0)
            if log is not None: log.append(dict(k=n_, shift=(round(float(sx), 1), round(float(sy), 1)), ncc=round(ncc, 3), why='ok' if ncc >= min_ncc else 'ncc'))
            if ncc < min_ncc: continue
            texs.append(Ts)
        if len(texs) == 1:
            self.tex, self.src = T0, (T0[..., 3].max(0) > 0.5).astype(np.int8)
            return 1
        S = np.stack(texs)
        A = S[..., 3] > 0.5
        cnt = A.sum(0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rgb = np.where(A[..., None], S[..., :3], np.nan)
            med = np.nanmedian(rgb, 0)
        med = np.where(np.isfinite(med), med, T0[..., :3])
        al = S[..., 3].max(0)
        al = np.where(cnt >= 2, al, np.where(T0[..., 3] > 0.5, T0[..., 3], al * 0.5 + T0[..., 3] * 0.5))
        self.tex = np.dstack([med, al]).astype(np.float32)
        self.src = (al.max(0) > 0.5).astype(np.int8)
        return len(texs)

    def complete(self, iod, upper=True, period=0.075, back=0.2, shade=0.5, fade=0.04, grow=0.012, skin_L=None, tooth_h=0.11,
                 gum=None, straighten=True, synth=False, desat=0.0, crown=False, layout=None, enamel=None, detail=0.0,
                 gums=0.0, gum_rgb=None, gum_h=0.0, gum_drop=0.15, gum_pap=0.10):
        """make the band a whole arch. (0) the harvested incisal edge is straightened onto the band's edge row (h = 0): a
        smooth (quadratic) fit of the edge across the arch is removed column by column, so the slant / smile curve of the
        2D harvest does not ride the 3D arch twice (the tips' own shapes stay); (1) along each column the teeth run
        tooth_h IOD from the edge (the part the harvest frame's lip covered repeats the teeth just inside it), then the
        gum; (2) columns the harvest frame did not show take their mirror image about the midline; (3) beyond that the
        outermost tooth period is tiled outward (premolars) -- or, v8 synth, modelled teeth (synth_teeth) -- and darkened
        (+ desaturated, v8) toward the back (shade at `back` IOD past the last real column); the arch's ends feather out.
        v12 crown (--warp-arch-build v12): (1) continues each column beyond its visible tooth as a CROWN BODY: the median
        colour of the visible tooth's root-side rows (0.02 IOD; never the incisal edge's row band), low-passed across the
        arch (0.06 IOD: no interdental or edge structure) and darkened toward the root, then the gum."""
        T = self.tex
        Th, Tw = T.shape[:2]
        al = T[..., 3]
        g = max(1, int(round(grow * iod * self.texel)))
        r_e = (0.0 - self.h0) * self.texel - 0.5                                   # the edge row (h = 0)
        tk = (al > 0.5)
        if skin_L is not None:          # the teeth themselves (not the harvest frame's lip line)
            lab = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
            tk = tl.teeth_key(lab, skin_L, 0.8) & (al > 0.5)
            if not upper:               # lower arch: only tooth pixels ride it (a tongue / lip highlight between them is
                k1 = cv2.dilate(tk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)    # see-through)
                T[..., 3] = al = al * cv2.GaussianBlur(k1, (0, 0), 0.7)
        ok = tk.sum(0) >= max(2, int(0.01 * iod * self.texel))
        cs = np.nonzero(ok)[0]
        self.harvest_h = float(np.median(tk.sum(0)[ok])) / (iod * self.texel) if ok.any() else 0.0
        if straighten and len(cs) >= 5:
            e = np.array([np.nonzero(tk[:, c])[0][-1 if upper else 0] for c in cs], float)
            f = np.polyfit(cs, e, 2)
            r = e - np.polyval(f, cs)
            keep = np.abs(r) <= 2.5 * max(1.0, 1.4826 * np.median(np.abs(r)))
            if keep.sum() >= 5: f = np.polyfit(cs[keep], e[keep], 2)
            fit = np.polyval(f, np.clip(np.arange(Tw), cs.min(), cs.max()))
            sh = (fit - r_e).astype(np.float32)                                    # texel rows to move each column up
            mx, my = np.meshgrid(np.arange(Tw, dtype=np.float32), np.arange(Th, dtype=np.float32))
            my = my + sh[None, :]
            T = cv2.remap(T, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            tk = cv2.remap(tk.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0) > 0.5
            al = T[..., 3]
            self.edge_fit = f.tolist()
        if layout is not None:          # v13 (--warp-arch-layout anthro): the harvested teeth resampled onto the anthropometric
            cap_ = tuple(layout.get('cap', (0.6, 1.8)))
            p0 = max(2, int(round(period * iod * self.texel)))    # slots (a monotone
            det = edge_lines(T, tk, r_e, iod, self.texel, upper)                                                # x-warp of the columns)
            sc, lg = anthro_map(det, Tw, self.texel, self.A, layout['bounds'], cap_, default_p=p0)
            lg['pass'] = 1
            if lg.get('mode') != 'lines':           # a faint run: the weaker minima too (the slot assignment skips spurious ones)
                det2 = edge_lines(T, tk, r_e, iod, self.texel, upper, prom=(0.75, 0.15))
                sc2, lg2 = anthro_map(det2, Tw, self.texel, self.A, layout['bounds'], cap_, default_p=p0)
                if lg2.get('mode') == 'lines': sc, lg = sc2, dict(lg2, **{'pass': 2, 'pass1': dict(mode=lg.get('mode'), n_lines=len(lg.get('lines', [])))})
            mx, my = np.meshgrid(sc.astype(np.float32), np.arange(Th, dtype=np.float32))
            T = cv2.remap(T, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            tk = cv2.remap(tk.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0) > 0.5
            al = T[..., 3]
            self.layout_log = lg
        th = tooth_h * iod * self.texel
        tr = max(1.0, 0.012 * iod * self.texel)
        rows = np.arange(Th, dtype=np.float32)
        p = max(2, int(round(period * iod * self.texel)))
        if synth: p, self.x_hl, self.n_teeth = tooth_period(T, tk, r_e, th, iod, self.texel, upper, float(p))
        self.period = float(p) / (iod * self.texel)
        gum = None if gum is None else np.asarray(gum, np.float32)
        rep = np.zeros((Th, Tw), np.float32)
        GW = np.zeros((Th, Tw), np.float32)
        for c in range(Tw):
            rr = np.nonzero(tk[:, c])[0]
            if len(rr) < 2: rr = np.nonzero(al[:, c] > 0.5)[0]
            if len(rr) < 2: continue
            if upper and crown:          # v12: a crown body above the visible tooth (the median colour of its root-side rows,
                g2 = max(g, int(round(0.02 * iod * self.texel)))    # below the harvest lip's contact line / shadow)
                r = int(min(rr[0] + g2, rr[-1]))
                r1 = min(int(rr[-1]) + 1, r + max(1, int(round(0.02 * iod * self.texel))))
                T[:r, c, :3] = np.median(T[r:r1, c, :3], 0)
                T[:r, c, 3] = np.maximum(T[:r, c, 3], T[r, c, 3])
                GW[:, c] = smootherstep(((r_e - th) - rows) / tr)
                rep[:r, c] = np.clip((r - rows[:r]) / max(1.0, 0.6 * th), 0, 1)
            elif crown:                  # v12, lower: the same below it
                g2 = max(g, int(round(0.02 * iod * self.texel)))
                r = int(max(rr[-1] - g2, rr[0]))
                r0 = max(int(rr[0]), r + 1 - max(1, int(round(0.02 * iod * self.texel))))
                T[r + 1:, c, :3] = np.median(T[r0:r + 1, c, :3], 0)
                T[r + 1:, c, 3] = np.maximum(T[r + 1:, c, 3], T[r, c, 3])
                GW[:, c] = smootherstep((rows - (r_e + th)) / tr)
                rep[r + 1:, c] = np.clip((rows[r + 1:] - r) / max(1.0, 0.6 * th), 0, 1)
            elif upper:                  # up toward the gum
                r = int(min(rr[0] + g, rr[-1]))
                T[:r, c, :3] = T[r, c, :3]
                T[:r, c, 3] = np.maximum(T[:r, c, 3], T[r, c, 3])
                GW[:, c] = smootherstep(((r_e - th) - rows) / tr)                    # 1 above the tooth's root line
                rep[:r, c] = np.clip((r - rows[:r]) / max(1.0, 0.6 * th), 0, 1)
            else:                        # down toward the gum
                r = int(max(rr[-1] - g, rr[0]))
                T[r + 1:, c, :3] = T[r, c, :3]
                T[r + 1:, c, 3] = np.maximum(T[r + 1:, c, 3], T[r, c, 3])
                GW[:, c] = smootherstep((rows - (r_e + th)) / tr)
                rep[r + 1:, c] = np.clip((rows[r + 1:] - r) / max(1.0, 0.6 * th), 0, 1)
        # the repeated part (a tooth's body toward its root) is not a row of identical pixels: softened across the arch
        # and a little darker toward the root, so it reads as enamel in shadow, not as vertical stripes
        Tb = cv2.GaussianBlur(T[..., :3], (0, 0), sigmaX=max(2.0, 0.06 * iod * self.texel) if crown else max(1.5, 0.03 * iod * self.texel), sigmaY=0.1)
        rw = np.sqrt(rep)[..., None]
        T[..., :3] = (T[..., :3] * (1 - rw) + Tb * rw) * (1 - 0.25 * rep[..., None])
        if not upper:                   # the lower incisal edge is the part the lip leaves visible: solid, not see-through,
            body = (rows[:, None] >= r_e + 1.5) & (T[..., 3] > 0.3)
            T[..., 3] = np.where(body, 1.0, T[..., 3])
            # and lit like the tooth below it (the harvest's tips carry the dark bite line between the arches). v16.1: the atlas
            # stays BRIGHT (v15); the lower-teeth shadow is applied per frame (LayerCompositor._lower_shadow), gated by the take's
            # own per-frame lower display -- bright where the take bares a row, dark/receded where it hides them (no baked darkening)
            k0 = int(np.ceil(r_e + max(2.0, 0.015 * iod * self.texel)))
            r0_ = int(np.ceil(r_e + 0.5))
            if k0 < Th: T[r0_:k0, :, :3] = np.maximum(T[r0_:k0, :, :3], 0.92 * T[k0:k0 + 1, :, :3])
        if gum is not None:
            gcol = gum[None, None, :] * np.ones((Th, Tw, 1), np.float32)
            if synth:                   # v8: the gum recedes into the lip's shadow away from the teeth (a pale gum line
                far = (r_e - th - rows) if upper else (rows - (r_e + th))     # printed through a see-through lip edge)
                gcol = gcol * (1 - 0.45 * smootherstep(far / max(1.0, 0.12 * iod * self.texel)))[:, None, None]
            T[..., :3] = T[..., :3] * (1 - GW[..., None]) + gcol * GW[..., None]
        # a SOLID column: opaque over the tooth's body (a harvest plate's feathered side is not; tiling or mirroring it
        # would print see-through stripes)
        r0, r1 = (int(max(0, r_e - th)), int(max(1, r_e - 2))) if upper else (int(r_e + 2), int(min(Th, r_e + th)))
        cov = np.median(T[r0:max(r0 + 1, r1), :, 3], 0) > 0.85
        src = np.where(cov, 1, 0).astype(np.int8)
        cs = np.nonzero(cov)[0]
        for c in range(Tw):             # a lone uncovered column between covered ones (an interdental gap): its neighbour
            if cov[c] or len(cs) == 0 or c < cs.min() or c > cs.max(): continue
            j = cs[np.argmin(np.abs(cs - c))]
            if abs(j - c) <= max(2, int(0.02 * iod * self.texel)):
                T[:, c] = T[:, j]
                src[c] = 1
        cov = src > 0
        for c in range(Tw):                                                       # mirror
            m = Tw - 1 - c
            if not cov[c] and cov[m]:
                T[:, c] = T[:, m]
                src[c] = 2
        if layout is not None:          # v13: every slot boundary on the harvested / mirrored run reads as an interproximal line
            self.groove(T, src > 0, layout['bounds'], iod, upper, r_e, th)     # over the whole crown (the gap fill above and the
        cov = src > 0
        mid = Tw // 2                                               # crown body's blur remove the harvest's own)
        dark = np.ones(Tw, np.float32)
        fb = np.zeros(Tw, np.float32)
        if cov.any():
            # contiguous covered run through the midline (or the covered run nearest it)
            cs = np.nonzero(cov)[0]
            c0 = cs[np.argmin(np.abs(cs - mid))]
            lo = c0
            hi = c0
            while lo - 1 >= 0 and cov[lo - 1]: lo -= 1
            while hi + 1 < Tw and cov[hi + 1]: hi += 1
            if layout is not None and hi - lo + 1 >= 2:          # v13: beyond the run, modelled teeth fill the anthropometric slots
                lo, hi, self.trimmed = trim_run(lo, hi, layout['bounds'], self.A, self.texel)   # (a run ending in a sliver of a slot:
                self.synth_cols = synth_slots(T, src, lo, hi, layout['bounds'], self.A, iod, self.texel, upper, r_e, self.x_hl)  # modelled whole)
            elif synth and hi - lo + 1 >= p:
                self.synth_cols = synth_teeth(T, src, lo, hi, p, iod, self.texel, upper, r_e, th, self.x_hl)
            else:
                for c in range(hi + 1, Tw):                                           # extend right
                    s = hi - p + 1 + ((c - hi - 1) % p)
                    s = int(np.clip(s, lo, hi))
                    T[:, c] = T[:, s]
                    src[c] = 3
                for c in range(lo - 1, -1, -1):                                       # extend left
                    s = lo + p - 1 - ((lo - 1 - c) % p)
                    s = int(np.clip(s, lo, hi))
                    T[:, c] = T[:, s]
                    src[c] = 3
            for c in range(hi + 1, Tw): fb[c] = smootherstep((c - hi) / max(1.0, back * iod * self.texel))
            for c in range(lo - 1, -1, -1): fb[c] = smootherstep((lo - c) / max(1.0, back * iod * self.texel))
            dark = 1 - shade * fb
            for c in range(Tw):                                                   # mirrored columns recede a little too
                if src[c] == 2: dark[c] = min(dark[c], 1 - 0.5 * shade * smootherstep((abs(c - mid) - 0.1 * iod * self.texel) / max(1.0, back * iod * self.texel)))
        T[..., :3] *= dark[None, :, None]
        if desat > 0:                   # v8: the back of the mouth loses its colour too (the teeth there are in shadow)
            grey = T[..., :3].mean(2, keepdims=True)
            s_ = (1 - desat * fb)[None, :, None]
            T[..., :3] = grey + (T[..., :3] - grey) * s_
        if detail > 0:                  # v14: edge-aware detail enhancement of the crown structure
            self.detail_pass(T, iod, upper, r_e, th, float(detail))
        if enamel:                      # v14: the enamel material pass (gloss, incisal translucency, interdental / gumline AO, warm hue)
            self.enamel_pass(T, src > 0, iod, upper, r_e, th, fb, enamel, layout)
        if gums > 0 and crown and gum_rgb is not None and gum_h > 0:      # v15: pink gingiva above / below the crowns, scalloped margin
            gh = gum_h * iod * self.texel                                 # the pink band above the crown top (texel rows)
            drop = float(gum_drop) * th
            pap = float(gum_pap) * th        # the margin drops onto the cervical enamel; papillae dip between teeth
            self.gum_pass(T, src > 0, iod, upper, r_e, th, gh, float(gums), gum_rgb, layout, drop, pap)
            self.gum_h_c = gum_h * iod                                    # (canonical px; lip_band starts the lip above the gum band)
        endf = smootherstep((self.A - np.abs(self.a)) / max(1e-6, fade * iod)).astype(np.float32)
        T[..., 3] *= endf[None, :]
        self.tex, self.src = T, src
        self.tooth_h_c, self.tr_c = th / self.texel, tr / self.texel     # (v12 lip band: canonical px)
        return self

    def detail_pass(self, T, iod, upper, r_e, th, amount, radius=0.004, clip=18.0):
        """v14 (--warp-arch-detail): an edge-aware, MEAN-PRESERVING detail enhancement of the crown's own structure in atlas
        space. The L channel's high-frequency residual over a small radius (sub-tooth: surface + groove texture, not the
        tooth-vs-gap macro edges) is soft-clipped (tanh, no ringing / halos) and added back, so a blurry mid-shot's harvested
        teeth get their crispness restored -- it only sharpens what the harvest already has. Because the unsharp residual is
        zero-mean, the visible-tooth span (a brightness threshold) is not re-widened. Crown tooth texels only (alpha > 0.5).
        In place."""
        Th, Tw = T.shape[:2]
        rows = np.arange(Th, dtype=np.float32)[:, None]
        d = (r_e - rows) if upper else (rows - r_e)
        m = ((d >= -0.5) & (d <= th)) & (T[..., 3] > 0.5)
        m = np.broadcast_to(m, (Th, Tw))
        if m.sum() < 30: return
        lab = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0]
        sig = max(0.6, radius * iod * self.texel)
        hp = L - cv2.GaussianBlur(L, (0, 0), sig)                          # the high-frequency residual (zero-mean)
        boost = amount * clip * np.tanh(hp / max(1e-3, clip))              # soft-clipped: no overshoot / ringing
        lab[..., 0] = np.where(m, np.clip(L + boost, 0, 255), L)
        T[..., :3] = np.where(m[..., None], cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32), T[..., :3])

    def enamel_pass(self, T, run, iod, upper, r_e, th, fb, p, layout=None):
        """v14 (--warp-arch-enamel): a material pass on the crown rows in atlas space -- kill the flat grey enamel. Four
        components, each a scalar in `p` (0 = off): `spec` a glossy specular highlight (a soft vertical band across the mid-
        crown, strongest on the key-light side, self.x_hl in -1..1); `transl` incisal translucency (the biting edge lighter and
        a touch warmer -- thin enamel); `ao` interdental + gum-line ambient occlusion (darker in the grooves and where the
        crown meets the gum / the back of the arch); `warm` the enamel hue toward the take's yellowish teeth (Lab b* up, a*
        held low -- the v13 render read greyer / bluer). Operates only on crown texels (within th of the incisal edge, tooth
        alpha > 0.5), leaving the gum, the cavity and the lip band untouched. In place. `fb` = the back-of-arch fade (0 front,
        1 back) so the gloss eases out toward the dark corners."""
        Th, Tw = T.shape[:2]
        rows = np.arange(Th, dtype=np.float32)[:, None]
        d = (r_e - rows) if upper else (rows - r_e)                       # 0 at the incisal edge, + toward the gum
        u = np.clip(d / max(1.0, th), 0.0, 1.0)                           # normalised crown depth
        crown = (d >= -0.5) & (d <= th)
        tk = (T[..., 3] > 0.5)
        m = crown & tk
        if not m.any(): return
        spec = float(p.get('spec', 0.0))
        transl = float(p.get('transl', 0.0))
        ao = float(p.get('ao', 0.0))
        warm = float(p.get('warm', 0.0))
        a_tgt = float(p.get('a_target', 128.0))
        ac = (np.arange(Tw) + 0.5) / self.texel - self.A                  # arc from the midline (canonical px)
        side = np.clip(ac / max(1.0, 0.02 * iod), -1.0, 1.0)                                  # smooth across the midline (no 1-col seam)
        keyside = (0.5 + 0.5 * np.clip(self.x_hl, -1, 1) * side).astype(np.float32)           # 1 on the key-light side, 0 the other
        keyside = (0.6 + 0.4 * keyside)[None, :]                          # never fully extinguished across the arch
        front = (1.0 - 0.85 * np.clip(fb, 0, 1))[None, :]                 # the gloss eases out toward the dark corners
        gloss = np.exp(-((u - 0.42) / 0.26) ** 2).astype(np.float32) * keyside * front   # a mid-crown vertical highlight band
        incisal = smootherstep((0.20 - u) / 0.14).astype(np.float32)      # the biting edge
        grooveAO = np.zeros((1, Tw), np.float32)
        if layout is not None and ao > 0:                                 # darken toward each interdental slot boundary
            for b in np.concatenate([-np.asarray(layout['bounds'][1:], float), np.asarray(layout['bounds'], float)]):
                grooveAO = np.maximum(grooveAO, np.exp(-((ac - b) / max(0.6 / self.texel, 0.006 * iod)) ** 2).astype(np.float32)[None, :])
        gumAO = smootherstep((u - 0.72) / 0.2).astype(np.float32)         # occlusion where the crown meets the gum
        aoamt = np.clip(ao * (0.6 * gumAO + 0.7 * grooveAO), 0.0, 0.9)
        lab = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a_, b_ = lab[..., 0], lab[..., 1], lab[..., 2]
        dL = (spec * 48.0 * gloss + transl * 26.0 * incisal - 60.0 * aoamt) * m          # lighten / darken
        db = (warm * 10.0 + transl * 4.0 * incisal) * m                                   # warm the hue (Lab b*, 0..255 -> +yellow)
        da = (-warm * 3.0 * (a_ - a_tgt) / 20.0) * m                                      # pull a* gently toward neutral-warm
        lab[..., 0] = np.clip(L + dL, 0, 255)
        lab[..., 2] = np.clip(b_ + db, 0, 255)
        lab[..., 1] = np.clip(a_ + da, 0, 255)
        bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        # a touch of true white in the specular core (wet enamel), premultiplied by the crown mask so edges do not glow
        core = (spec * 0.22 * np.exp(-((u - 0.42) / 0.16) ** 2) * keyside * front * m)[..., None].astype(np.float32)
        bgr = bgr * (1 - core) + 255.0 * core
        T[..., :3] = np.where(m[..., None], bgr, T[..., :3])

    def gum_pass(self, T, run, iod, upper, r_e, th, gh, strength, col, layout=None, drop=0.0, pap=0.0):
        """v15 (--warp-arch-gums): a pink gingival band baked into the arch atlas over the cervical top of the crowns and up
        toward the lip, so the crowns emerge from gum, not from lip colour. `gh` = the pink band's height ABOVE the crown top
        (texel rows, scaled by the face's IPD through the anthro unit); `col` = BGR gum colour (the take's own harvested
        gingiva, or a synthesized warm-pink fallback); `strength` = the blend opacity. The gum MARGIN is scalloped and DROPS
        onto the enamel: at a tooth's body it covers the top `drop` of the crown; at the interdental slot boundaries (the
        papillae) it dips a further `pap` toward the incisal edge, so a scalloped gumline sits on the crowns -- the crown
        stays mostly intact, the teeth keep their size. The gum rolls a touch darker toward its margin (it curves under to the
        tooth neck) and is built on the opaque crown-body rows, so it warps with the crowns. Crown columns only (`run`). In place."""
        Th, Tw = T.shape[:2]
        rows = np.arange(Th, dtype=np.float32)[:, None]
        d = (r_e - rows) if upper else (rows - r_e)                       # 0 at the incisal edge, + toward the gum
        ac = (np.arange(Tw) + 0.5) / self.texel - self.A                  # arc from the midline (canonical px)
        scal = np.zeros((1, Tw), np.float32)                              # 1 at an interdental boundary (the papilla), 0 over a tooth
        if layout is not None:
            for b in np.concatenate([-np.asarray(layout['bounds'][1:], float), np.asarray(layout['bounds'], float)]):
                scal = np.maximum(scal, np.exp(-((ac - b) / max(0.8 / self.texel, 0.010 * iod)) ** 2).astype(np.float32)[None, :])
        scal = scal * np.asarray(run, np.float32)[None, :]
        dm = (th - drop) - pap * scal                                     # the gum margin per column: dropped onto the crown, papillae dipping between teeth
        span = max(1.0, gh + drop)                                        # the full visible band: cervical enamel + above the crown
        lo_edge = smootherstep((d - dm) / max(1.0, 0.22 * span))          # 0 below the margin -> 1 inside the gum
        hi_edge = 1.0 - smootherstep((d - (th + gh)) / max(1.0, 0.30 * gh))   # fade out toward the lip band above it
        w = (lo_edge * hi_edge).astype(np.float32) * (T[..., 3] > 0.5).astype(np.float32) * np.asarray(run, np.float32)[None, :]
        w = np.clip(w * float(strength), 0.0, 1.0)
        # the gum shades a little darker toward its margin (it curves under to the tooth neck) and holds its colour up the band
        g_u = np.clip((d - dm) / span, 0.0, 1.0)                          # 0 at the margin .. 1 up the band
        rollunder = (0.80 + 0.20 * smootherstep(g_u))[..., None]
        gcol = (np.asarray(col, np.float32)[None, None, :] * rollunder).astype(np.float32)
        T[..., :3] = T[..., :3] * (1 - w[..., None]) + gcol * w[..., None]
        T[..., 3] = np.maximum(T[..., 3], (w > 0.05).astype(np.float32))  # opaque where gum was painted (no cavity shows through)

    def groove(self, T, run, bounds, iod, upper, r_e, th, depth=0.15, width=0.005):   # v17e: 0.3 -> 0.15 (founder: gaps too busy)
        """v13 (anthro): the slot boundaries (bounds: canonical px from the midline, both sides) on the run's columns (run: bool per
        column) as interproximal lines: each column near a boundary is scaled as a whole (every row by one factor: no new
        horizontal edge, the double-edge ruler is unaffected) to (1 - depth x a gaussian of `width` IOD about the boundary) of
        the run's local tooth brightness (a horizontal median over ~0.03 IOD, over the crown's rows within th of the incisal
        edge); a column already that dark there (the harvest's own line) is left as it is. In place."""
        from scipy.ndimage import median_filter
        Th, Tw = T.shape[:2]
        ac = (np.arange(Tw) + 0.5) / self.texel - self.A
        g = np.zeros(Tw, np.float32)
        for b in np.concatenate([-np.asarray(bounds[1:], float), np.asarray(bounds, float)]):
            g = np.maximum(g, np.exp(-((ac - b) / max(0.6 / self.texel, width * iod)) ** 2).astype(np.float32))
        g = g * run
        if not g.any(): return
        rows = np.arange(Th)
        d = (r_e - rows) if upper else (rows - r_e)
        crown = (d >= 0) & (d <= th)   # the crown rows
        if not crown.any(): return
        Y = T[crown][..., :3].mean(-1)
        med = median_filter(Y, size=(1, max(3, int(round(0.03 * iod * self.texel)) | 1)), mode='nearest')
        ok = T[crown][..., 3] > 0.5
        r = np.where(ok, Y / np.maximum(med, 1e-3), np.nan)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                r = np.nanmedian(r, 0)
        r = np.where(np.isfinite(r), r, 1.0)
        want = 1 - depth * g
        f = np.clip(want / np.maximum(r, 1e-3), want, 1.0).astype(np.float32)
        T[..., :3] *= f[None, :, None]

    def lines_now(self, iod, skin_L=None, upper=True):
        """v13: the interdental lines of the band's CURRENT texture (edge_lines over its opaque tooth pixels, the teeth key when
        skin_L is given) -> (lines: arc positions in IOD, ascending; the two central widths (image-left, image-right) in IOD
        around the line nearest the midline (within 0.08 IOD), None where a neighbour is missing)."""
        T = self.tex
        tk = T[..., 3] > 0.5
        r_e = (0.0 - self.h0) * self.texel - 0.5
        if skin_L is not None:
            lab = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
            tk = tk & tl.teeth_key(lab, skin_L, 0.8)
        det = edge_lines(T, tk, r_e, iod, self.texel, upper)
        if det is None or not len(det['lines']): return [], (None, None)
        a = np.sort(((det['lines'] + 0.5) / self.texel - self.A) / iod)
        k = int(np.argmin(np.abs(a)))
        if abs(a[k]) > 0.08: return a.tolist(), (None, None)
        return a.tolist(), (float(a[k] - a[k - 1]) if k > 0 else None, float(a[k + 1] - a[k]) if k + 1 < len(a) else None)

    def render(self, proj3, box, rot, step=None, lingual=0.45, amap=False, facing=0.0, hmap=False):
        """rasterise the band into box (x0, y0, w, h): proj3(X) -> (frame px N x 2, depth N (larger = farther));
        rot: the 3x3 linear part canonical -> frame (for facing). Column quads far -> near (painter's order); a quad whose
        normal faces away from the camera shows the arch's inner (lingual) side: drawn darker. Returns (BGR, alpha[, a])."""
        x0, y0, w, h = box
        out = np.zeros((h, w, 4), np.float32)
        am = np.full((h, w), np.nan, np.float32) if amap else None
        hm = np.full((h, w), np.nan, np.float32) if hmap else None       # v12: the band's h (canonical px) of the front-most quad
        step = step or max(2.0, 0.01 * self.Tw)
        nq = max(4, int(np.ceil(2 * self.A * self.texel / step)))
        ac = np.linspace(-self.A, self.A, nq + 1)
        P0 = self.points(ac, np.full(nq + 1, self.h0))
        P1 = self.points(ac, np.full(nq + 1, self.h1))
        q0, z0 = proj3(P0)
        q1, z1 = proj3(P1)
        q0 = q0 - [x0, y0]
        q1 = q1 - [x0, y0]
        nz = (self.normal(0.5 * (ac[1:] + ac[:-1])) @ np.asarray(rot, float).T)[:, 2]
        tx = (ac + self.A) * self.texel - 0.5
        order = np.argsort(-(z0[1:] + z0[:-1] + z1[1:] + z1[:-1]))
        for j in order:
            dst = np.float32([q0[j], q0[j + 1], q1[j + 1], q1[j]])
            bx0, by0 = np.floor(dst.min(0)).astype(int)
            bx1, by1 = np.ceil(dst.max(0)).astype(int) + 1
            bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(w, bx1), min(h, by1)
            if bx1 - bx0 < 1 or by1 - by0 < 1: continue
            if abs(cv2.contourArea(dst)) < 0.3: continue
            srcp = np.float32([[tx[j], -0.5], [tx[j + 1], -0.5], [tx[j + 1], self.Th - 0.5]])
            M = cv2.getAffineTransform(dst[:3] - np.float32([bx0, by0]), srcp)          # box px -> texel
            if self.mipmap:                                                            # v14: area-correct minification
                patch = self._warp_quad_mip(M, (bx1 - bx0, by1 - by0))
            else:
                patch = cv2.warpAffine(self.tex, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
            cv2.fillConvexPoly(m, np.round(dst - [bx0, by0]).astype(np.int32), 1)
            m = (m > 0) & (patch[..., 3] > 0.002)
            if nz[j] > 0: patch[..., :3] *= lingual                               # back-facing: the lingual side
            elif facing > 0:                                                      # v8: a crown seen edge-on is in shadow
                patch[..., :3] *= (1 - facing) + facing * float(np.clip(-nz[j], 0, 1)) ** 0.7
            sub = out[by0:by1, bx0:bx1]
            a = patch[..., 3:4]                     # premultiplied "over"
            sub[m] = np.concatenate([patch[..., :3] * a + sub[..., :3] * (1 - a), a + sub[..., 3:4] * (1 - a)], -1)[m]
            if amap:
                av = cv2.warpAffine(np.tile(self.a[None, :].astype(np.float32), (self.Th, 1)), M, (bx1 - bx0, by1 - by0),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                am[by0:by1, bx0:bx1][m & (patch[..., 3] > 0.5)] = av[m & (patch[..., 3] > 0.5)]
            if hmap:
                hv = cv2.warpAffine(np.tile(self.h[:, None].astype(np.float32), (1, self.Tw)), M, (bx1 - bx0, by1 - by0),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                hm[by0:by1, bx0:bx1][m] = hv[m]
        # out holds premultiplied colour; back to straight colour
        rgb = out[..., :3] / np.maximum(out[..., 3:4], 1e-4)
        if hmap: return (rgb, out[..., 3], am, hm) if amap else (rgb, out[..., 3], hm)
        return (rgb, out[..., 3], am) if amap else (rgb, out[..., 3])

    def load_from(self, Z, key, iod, L_scale=1.0):
        """v8 fallback: take the arch texture of another shot (np.load'ed Z, keys <key>_tex / <key>_a / <key>_h in IOD
        units), resampled onto this band's (a, h) grid by IOD, its L scaled by L_scale."""
        Ts = np.asarray(Z[f'{key}_tex'], np.float32)
        a_s = np.asarray(Z[f'{key}_a'], float)
        h_s = np.asarray(Z[f'{key}_h'], float)
        ca = np.interp(self.a / iod, a_s, np.arange(len(a_s)), left=-1, right=-1).astype(np.float32)
        rh = np.interp(self.h / iod, h_s, np.arange(len(h_s)), left=-1, right=-1).astype(np.float32)
        mx, my = np.meshgrid(ca, rh)
        T = cv2.remap(Ts, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lab = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] = np.clip(lab[..., 0] * L_scale, 0, 255)
        T[..., :3] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        self.tex = T.astype(np.float32)
        self.src = np.where(T[..., 3].max(0) > 0.5, 4, 0).astype(np.int8)
        return self


def ellipsoid_layer(c, L, box, colour, light=(0.0, -0.55, -0.83), groove=0.08, spec=0.0, ambient=0.45, diffuse=0.6):
    """a soft ellipsoid (the tongue) in box: centre c (frame px, 2), L = the 2x3 linear map canonical -> frame applied to
    the ellipsoid's semi-axes (columns = axis vectors x semi-axis, rotated). Silhouette = the projected ellipse
    (shape L L^T); shading = a sphere impostor lit from the front-above, a faint median groove, (v8) a wet specular
    highlight of strength spec. -> (BGR, alpha)."""
    x0, y0, w, h = box
    Mx = np.asarray(L, float)
    Sg = Mx @ Mx.T
    try: Si = np.linalg.inv(Sg)
    except np.linalg.LinAlgError: return np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    p = np.stack([xx + x0 - c[0], yy + y0 - c[1]], -1)
    r2 = np.einsum('...i,ij,...j->...', p, Si, p)
    al = smootherstep((1 - r2) / 0.3).astype(np.float32)                       # a soft silhouette (no cut-out pill)
    # impostor normal: the unit-disc coordinate (whitened) gives the sphere normal; light from the front-above
    ev, V = np.linalg.eigh(Sg)
    W = V @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-9))) @ V.T
    q = p @ W.T
    nz = np.sqrt(np.clip(1 - (q ** 2).sum(-1), 0, 1))
    lv = np.asarray(light, float)
    lv = lv / np.linalg.norm(lv)
    ndl = np.clip(q[..., 0] * lv[0] + q[..., 1] * lv[1] - nz * lv[2], 0, 1)
    sh = ambient + diffuse * ndl
    sh = sh * (1 - groove * np.exp(-(q[..., 0] / 0.12) ** 2) * (q[..., 1] < 0.2))
    rgb = (np.asarray(colour, np.float32)[None, None, :] * sh[..., None]).astype(np.float32)
    if spec > 0:
        hv = lv + np.array([0.0, 0.0, -1.0])
        hv = hv / np.linalg.norm(hv)
        ndh = np.clip(q[..., 0] * hv[0] + q[..., 1] * hv[1] - nz * hv[2], 0, 1) ** 24
        rgb = rgb + (spec * 255 * ndh)[..., None] * np.array([0.9, 0.95, 1.0], np.float32)
    return np.clip(rgb, 0, 255).astype(np.float32), al


def fill_blob(img, mask, seed=7, grain=1.0):
    """v8 cover: fill mask (bool) of img (BGR uint8) with the harmonic membrane of the pixels around it plus static
    grain matched to the surrounding skin's high-pass (the same recipe as lip_layers.erase_lips, on any mask). The fill
    meets the surrounding pixels exactly at the boundary, so there is no seam against the image it is cut into."""
    img = np.asarray(img)
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    if len(ys) == 0: return img.copy()
    y0, y1, x0, x1 = max(0, ys.min() - 3), min(img.shape[0], ys.max() + 4), max(0, xs.min() - 3), min(img.shape[1], xs.max() + 4)
    sub = img[y0:y1, x0:x1].astype(np.float64)
    ms = m[y0:y1, x0:x1]
    fill = ll.harmonic_fill(sub, ms)
    if grain > 0:
        f32 = img.astype(np.float32)
        hp = f32 - cv2.GaussianBlur(f32, (0, 0), 1.5)
        ring = cv2.dilate(ms.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~ms
        rs = hp[y0:y1, x0:x1][ring] if ring.sum() > 20 else hp[y0:y1, x0:x1].reshape(-1, 3)
        sd = float(rs.std()) if len(rs) > 1 else 1.0
        k = 1.5 * 1.4826 * float(np.median(np.abs(rs - np.median(rs)))) + 1e-3
        g = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, sub.shape[:2]).astype(np.float32), (0, 0), 0.6)
        tex = np.clip((g / (g.std() + 1e-6) * sd)[..., None] * np.ones((1, 1, 3)), -k, k)
        fill = fill + grain * tex * cv2.GaussianBlur(ms.astype(np.float32), (0, 0), 1.0)[..., None]
    out = img.copy()
    reg = out[y0:y1, x0:x1]
    reg[ms] = np.clip(fill[ms] + 0.5, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------------------------------- the compositor --
class LayerCompositor:
    """v7 layered mouth. teeth: the TeethLayer (pose, pins, jaw plan, harvest stats, cavity tone, light) -- its plates
    are not used; frames: the take; S: its smoothed track. v8 options (all default to v7's behaviour): pin_mode
    'skin' | 'arc', pin_gain / pin_split (the arc pin's fast-residual gain and split sigma), harvest_n, texel, synth,
    desat, tex_from / tex_out, tongue 'on' | 'off' | 'wedge', rim (strength, width IOD), wet, cover 'on' | 'off' |
    'fill', smooth (frames)."""

    def __init__(self, teeth, frames, S, lip_ab=None, skin_ab=None, arch_k=3.2, up_arc=0.40, lo_arc=0.36, overjet=0.03,
                 overbite=0.03, tongue=True, cover=True, debug=None, debug_frames=None, chin_sigma=4.0, expose=0.03,
                 pin_mode='skin', pin_gain=0.4, pin_split=1.5, harvest_n=1, texel=1.0, synth=False, desat=0.0, tex_from=None,
                 tex_out=None, tex_min_h=0.05, tex_share=False, rim=(0.22, 0.012), wet=0.0, cover_mode=None, smooth=0.0, shade=0.5, facing=0.0, tongue_lit=1.0,
                 harvest_lo=None, interior_clip=False, dbg_stab='pin', cover_corners=0.0, build='v11', layout='harvest', corridor=0.0,
                 corridor_floor=0.25, anthro_unit='ipd', mipmap=False, enamel=None, detail=0.0, gums=0.0, tongue_cover=0.0,
                 lower_real=0.0, lower_mix=None, lower_ratio='auto', cavity_fill=0.0):
        self.t, self.frames, self.S = teeth, frames, S
        t = teeth
        io = t.iod
        self.cavity_fill = float(cavity_fill or 0.0)     # v17 (--warp-cavity fill): tongue + lower row painted in the opening
        self.build = build                    # v12 (--warp-arch-build v12): crown-body continuation, open-bite lower harvest
        # v13 (--warp-arch-layout anthro, --warp-arch-corridor K): the teeth on anthropometric slots; the buccal corridor
        self.layout, self.corridor, self.corridor_floor, self.lay = layout, float(corridor or 0.0), float(corridor_floor), None
        self.cor_frames, self._cor = {}, None
        if layout == 'anthro' or self.corridor > 0:
            self.unit_c, self.unit_info = self._anthro_unit(anthro_unit)
        if layout == 'anthro':
            self.lay = {k: dict(bounds=anthro_bounds(ANTHRO[k], self.unit_c, 1.02 * arc * io)) for k, arc in (('up', up_arc), ('lo', lo_arc))}
        self.interior_clip, self.dbg_stab, self.reg = bool(interior_clip), dbg_stab, {}     # v11
        self.cover_corners = float(cover_corners or 0.0)
        self.corner_log = {}
        self.face_of = None             # v12 (--warp-yaw-clamp): (i, box) -> the corner cover's face test over the box, or None
        self.lip_ab, self.skin_ab = lip_ab, skin_ab
        self.tongue_mode = tongue if isinstance(tongue, str) else ('on' if tongue else 'off')
        self.tongue_on = self.tongue_mode != 'off'
        self.cover_mode = cover_mode or (cover if isinstance(cover, str) else ('on' if cover else 'off'))
        self.cover_on = self.cover_mode != 'off'
        self.overbite = overbite * io
        self.k = arch_k / io
        n = len(frames)
        self.n = n
        self.pin_lo = np.zeros((n, 2))
        self.chin_sigma = chin_sigma
        self.clamp = 0.0
        self.interior = {}
        self.tongue_y = 0.01    # tongue centre below (+) the lower incisal edge, IOD
        self.pin_mode, self.pin_gain, self.pin_split = pin_mode, pin_gain, pin_split
        self.dth = np.zeros(n)
        self.pin_disp = np.zeros((n, 2))
        self.pin_raw = np.zeros((n, 2))
        self.clamps = []
        self.rim, self.wet, self.smooth_f = tuple(rim), wet, smooth
        self.texel = texel
        self.synth = synth
        self.desat = desat
        self.shade = shade
        self.facing = facing
        self.tongue_lit = tongue_lit
        self.harvest_log = []
        self.prepared = False
        self.cover_lip = []
        self.harvest_used = dict(up=1, lo=1)
        self.fallback = []
        self.mipmap = bool(mipmap)          # v14 (--warp-arch-mipmap): area-correct minification of the arch atlas
        self.enamel = enamel if enamel else None    # v14 (--warp-arch-enamel): the crown material pass dict, or None (off)
        self.detail = float(detail or 0.0)          # v14 (--warp-arch-detail): edge-aware crown detail enhancement
        self.gums = float(gums or 0.0)              # v15 (--warp-arch-gums): pink gingiva above / below the crowns (0 = off, the v14 band)
        self.tongue_cover = float(tongue_cover or 0.0)   # v15 (--warp-tongue-cover): the tongue rises over the lower crowns at mid-open (0 = the v8 envelope)
        # v16 (--warp-lower-real): the lower teeth as the dim, half-hidden, soft-bottomed sibling of the uppers (0 = the v15 lowers)
        self.lower_real = float(lower_real or 0.0)
        self.lower_p = None
        self.lower_target = None
        self.lower_ratio = None
        self.lower_ratio_src = None
        self._lshow = {}
        self._lshow_arr = None
        if self.lower_real > 0:
            keys = ('dim', 'feather', 'cast', 'ao', 'occlude', 'cool', 'relief')
            if lower_mix:
                vals = [float(x) for x in str(lower_mix).split(',')]
                base = dict(zip(keys, (vals + [1.0] * len(keys))[:len(keys)]))
            else:
                base = {k: 1.0 for k in keys}
            self.lower_p = {k: self.lower_real * v for k, v in base.items()}
            # the per-take lower-vs-upper tooth-display ratio (the founder's insight): some faces show mostly top teeth, some
            # a blend, some mostly bottom; drive the lower dim target from the take's OWN tendency (own-pixels -> own-behaviour)
            if isinstance(lower_ratio, str) and lower_ratio.strip().lower() == 'auto':
                self.lower_ratio = self._lower_display_ratio()
                self.lower_ratio_src = 'auto'
            else:
                self.lower_ratio = float(lower_ratio)
                self.lower_ratio_src = 'override'
            # map the ratio to the lower-arch luminance fraction at wide-open: a top-teeth talker (ratio << REF) -> dim/hide
            # the lowers (~0.40 of the uppers); a bottom-teeth talker (ratio >> REF) -> let them show more (~0.75). The
            # anatomy default (~0.55) lands at REF. Clamped to a sane band. (openness scales between 1.0 near-closed and this.)
            REF = 0.45
            self.lower_target = float(np.clip(0.55 * (self.lower_ratio / REF) ** 0.5, 0.40, 0.78))
            self.lower_log = dict(real=self.lower_real, ratio=round(self.lower_ratio, 4), ratio_src=self.lower_ratio_src,
                                  target=round(self.lower_target, 4), preset={k: round(v, 3) for k, v in self.lower_p.items()})
            print(f"lower-real: ratio {self.lower_ratio:.3f} ({self.lower_ratio_src}) -> lower target {self.lower_target:.3f} "
                  f"of the upper-arch exposure (wide-open); preset dim/feather/cast/ao/occlude/cool/relief x{self.lower_real:.2f}")
        self.gum_h_up = 0.0
        self.gum_h_lo = 0.0
        self.gum_rgb = None
        self.gum_src = None   # set below when gums are on
        if self.gums > 0:
            if not hasattr(self, 'unit_c'): self.unit_c, self.unit_info = self._anthro_unit(anthro_unit)
            # the visible pink band above the crown, scaled by the face's IPD: the UPPER is taller (a real gum band when the
            # crowns are exposed), the LOWER stays subtle (the founder: gums at least on the top teeth)
            self.gum_h_up = float(np.clip(0.135 * self.unit_c / io, 0.055, 0.090))
            self.gum_h_lo = float(np.clip(0.070 * self.unit_c / io, 0.030, 0.055))   # (gum_rgb set after tongue_c, below)
        self.shadow = (0.3, 0.1, 0.15)          # contact-shadow strength: upper lip, lower lip, corners / all round
        self.ref = np.array(t.ref, float)
        self.z_up = t.z_up
        self.z_lof = t.z_up + overjet * io
        # ---- upper arch: the v6 harvest frame, its (unmirrored) upper plate as the texture source
        up_img, up_al = t._upper_plate(t.h_up, mirror=False)
        e_up = self._edge(t.h_up, up_al, self.z_up, 'bottom')
        self.e_up = e_up if e_up is not None else float(t.Cm[13, 1] + 0.06 * io)
        o_up = np.array([self.ref[0], self.e_up, self.z_up])
        self.up = ArchBand(o_up, t.u, t.d, t.b, up_arc * io, self.k, -0.32 * io, 0.03 * io, texel)
        self.tongue_c = self._tongue_colour()
        self.gum = self.tongue_c * 0.78
        if self.gums > 0: self.gum_rgb, self.gum_src = self._gum_colour()   # v15 (needs tongue_c for the synth fallback)
        if harvest_n > 1:
            srcs = [(up_img, up_al, (lambda X, h=t.h_up: t.pl.rigid(h, X)), 1.0)]
            for h in self._upper_candidates(harvest_n - 1):
                im, al = t._upper_plate(h, mirror=False)
                srcs.append((im, al, (lambda X, h=h: t.pl.rigid(h, X)), float(t.light[t.h_up]) / max(1e-3, float(t.light[h]))))
            self.harvest_cands = dict(up=[int(h) for h in self._upper_candidates(harvest_n - 1)])
            self.harvest_used['up'] = self.up.harvest_many(srcs, io, log=self.harvest_log)
        else: self.up.harvest(up_img, up_al, lambda X: t.pl.rigid(t.h_up, X))
        v12b = build == 'v12'
        up_th = max(0.11, float(((getattr(t, 'build_log', None) or {}).get('exposed_max_iod') or 0.0))) if v12b else 0.11
        self.up.complete(io, upper=True, skin_L=t.skin_med, gum=self.gum, synth=synth, desat=desat, shade=shade, tooth_h=up_th, crown=v12b,
                         enamel=self.enamel, detail=self.detail, gums=self.gums, gum_rgb=self.gum_rgb, gum_h=self.gum_h_up,
                         **({'layout': self.lay['up']} if self.lay else {}))
        # ---- lower arch: the frame showing the most lower teeth with the arches apart
        cands = self._lower_harvest(k=max(1, harvest_n if harvest_lo is None else harvest_lo))
        self.h_lt, lo_img, lo_al = cands[0] if cands else (int(t.h_lo), None, None)
        self.lo_synth = lo_img is None
        if not self.lo_synth:
            e_lo = self._edge(self.h_lt, lo_al, self.z_lof, 'top')
        if self.lo_synth or e_lo is None:
            self.lo_synth = True
            self.h_lt = t.h_lo
            e_lo = float(t.C[t.h_lo, 14, 1] - 0.015 * io)
        self.e_lo = e_lo
        o_lo = np.array([self.ref[0], self.e_lo, self.z_lof])
        self.lo = ArchBand(o_lo, t.u, t.d, t.b, lo_arc * io, self.k, -0.03 * io, 0.30 * io, texel)
        if not self.lo_synth:
            if len(cands) > 1:
                srcs = [(im, al, (lambda X, h=h: t.pl.rigid(h, X)), float(t.light[self.h_lt]) / max(1e-3, float(t.light[h]))) for h, im, al in cands]
                self.harvest_used['lo'] = self.lo.harvest_many(srcs, io, log=self.harvest_log)
            else: self.lo.harvest(lo_img, lo_al, lambda X: t.pl.rigid(self.h_lt, X))
            # v16.1: the lower arch atlas is built BRIGHT (v15); --warp-lower-real applies its shadow per frame (_lower_shadow /
            # _lower_depth), gated by the take's own per-frame lower display -- so a bright smile row is NOT dragged down by a
            # baked darkening, while the grill still recedes where the take hides the lowers.
            self.lo.complete(io, upper=False, skin_L=t.skin_med, gum=self.gum, tooth_h=0.09, synth=synth, desat=desat, shade=shade, crown=v12b,
                             enamel=self.enamel, detail=self.detail, gums=self.gums, gum_rgb=self.gum_rgb, gum_h=self.gum_h_lo,
                             **({'layout': self.lay['lo']} if self.lay else {}))
        else:
            self._synth_lower()
        if tex_from:
            self._fallback_tex(tex_from, tex_min_h, share=bool(tex_share))
        self.up.mipmap = self.lo.mipmap = self.mipmap                 # v14: both arch atlases minify through the mip pyramid
        if self.lay: self.layout_info = self._layout_info()          # v13: the slots, the resample, the widths as rendered
        if tex_out: self._save_tex(tex_out)
        # the jaw is now referenced to the lower harvest frame: angle 0 = that frame's jaw
        t.T_lo = t.C[self.h_lt, 14].copy()
        # the lower incisors sit at most `expose` IOD above the lower lip line (a generated take often shows a whole row
        # of tall lower teeth over its lip; a real mouth shows their edges): the band is carried down along the head's
        # down axis when its harvested edge is higher than that (the texture is unchanged, the lip simply covers more)
        sh = float((t.T_lo - self.lo.o) @ t.d) - expose * io
        if sh > 0:
            self.lo.o = self.lo.o + sh * t.d
            self.e_lo = float(self.lo.o[1])
        self.lo_shift = max(0.0, sh) / io
        self.expose = expose
        dd = t.d[:2] / max(1e-6, np.linalg.norm(t.d[:2]))
        t.e_lo = float(max(0.0, (t.T_lo[:2] - self.lo.o[:2]) @ dd))
        self.light_up, self.light_lo = float(t.light[t.h_up]), float(t.light[self.h_lt])
        self.debug, self.debug_frames, self.dbg = debug, debug_frames, {}
        self.open_s = None
        self.tg = None

    # ---------------------------------------------------------------------------------------------- harvest --
    def _edge(self, h, alpha, z, which):
        """canonical y of the incisal edge at the arch centre: the lowest (upper arch) / highest (lower arch) alpha > 0.5
        row over the central columns of harvest frame h, back-projected onto the plane z."""
        t = self.t
        io = t.iod * t.pl.s[h]
        P = t.S[h][:, :2]
        cx = P[[13, 14], 0].mean()
        cols = [x for x in range(int(cx - 0.08 * io), int(cx + 0.08 * io) + 1) if 0 <= x < alpha.shape[1] and (alpha[:, x] > 0.5).any()]
        if len(cols) < 3: return None
        ys = [np.nonzero(alpha[:, x] > 0.5)[0][-1 if which == 'bottom' else 0] for x in cols]
        Ai = tl.inv_aff(tl.plane_affine(t.pl, h, z, ref=t.ref))
        q = Ai[:, :2] @ np.array([cx, float(np.median(ys))]) + Ai[:, 2]
        return float(q[1])

    def _upper_candidates(self, k, bin_deg=8.0):
        """v8: up to k more upper-harvest frames in the same head-pose bin as h_up (out-of-plane pose within bin_deg),
        best-scoring first, at least 2 frames apart."""
        t = self.t
        sc = t._up_score * t._ok * np.exp(-(np.linalg.norm(t.pl.par[:, 1:4] - np.median(t.pl.par[:, 1:4], 0), axis=1) / 0.25) ** 2)
        op = t.pl.par[:, 1:3]
        dist = np.degrees(np.linalg.norm(op - op[t.h_up], axis=1))
        out = []
        for h in np.argsort(-sc):
            h = int(h)
            if len(out) >= k: break
            if h == t.h_up or dist[h] > bin_deg or sc[h] < 0.3 * sc[t.h_up] or h in out: continue
            out.append(h)
        return out

    def _lower_display_ratio(self):
        """v16, the founder's insight measured: this face's lower-vs-upper tooth-display tendency over its own talking frames.
        From the per-frame teeth stats (up_area / lo_area = the upper / lower arch tooth-pixel counts inside the inner lip,
        apart = how far open), the openness-weighted ratio sum(lo_area)/sum(up_area) over frames that actually bare the teeth.
        -> a scalar (~0.2 a top-teeth talker .. ~0.8 a bottom-teeth talker); REF (0.45) when nothing is shown."""
        st = self.t.stats
        io2 = float(self.t.iod) ** 2
        up = np.array([s['up_area'] for s in st], float)
        lo = np.array([s['lo_area'] for s in st], float)
        apart = np.array([s['apart'] for s in st], float)
        w = smootherstep((apart - 0.03) / 0.06) * (up > 0.0015 * io2).astype(float)   # open AND the uppers visibly bared
        su = float((up * w).sum())
        if su < 1.0: return 0.45
        return float(np.clip((lo * w).sum() / su, 0.05, 1.2))

    def _lower_harvest(self, k=1):
        """[(frame, image, alpha)] of the lower teeth, best first (up to k): teeth-coloured pixels inside the inner lip
        below the upper arch, per column from their top down to the lower lip line (teeth + the gaps between them), lip
        pixels dropped. Empty when no frame shows a usable lower arch."""
        t = self.t
        st = t.stats
        len(st)
        pose = t.pl.par[:, 1:4]
        front = np.exp(-(np.linalg.norm(pose - np.median(pose, 0), axis=1) / 0.25) ** 2)
        sc = t.pl.s ** 2
        lo = np.array([s['lo_area'] for s in st]) / sc
        apart = np.array([s['apart_dark' if self.build == 'v12' else 'apart'] for s in st])
        score = lo * front * np.sqrt(np.maximum(t._sharp, 1e-3)) * smootherstep((apart - 0.02) / 0.05)
        cand = np.argsort(-score)[:max(5, 3 * k)]
        out = []
        for h in cand:
            if score[h] <= 0 or lo[h] < 0.004 * t.iod ** 2: break
            if out and (score[h] < 0.3 * score[out[0][0]] or any(int(h) == j for j, _, _ in out)): continue
            if out and np.degrees(np.linalg.norm(pose[h, :2] - pose[out[0][0], :2])) > 8.0: continue
            img, pin, upper, _, topy, io = t._region(int(h))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            P = t.S[h][:, :2]
            kd = max(1, int(round(0.02 * io)))
            up_d = cv2.dilate(upper.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd + 1, 2 * kd + 1))) > 0
            mid = (P[13, 1] + P[14, 1]) / 2
            rows = np.arange(img.shape[0])[:, None]
            tk = pin & tl.teeth_key(lab, t.skin_med, 0.8) & ~up_d & (rows > mid)
            tk = cv2.morphologyEx(tk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
            if tk.sum() < 0.002 * io ** 2: continue
            boty = np.where(pin.any(0), img.shape[0] - 1 - pin[::-1].argmax(0), -1)
            m = np.zeros_like(tk)
            for x in np.nonzero(tk.any(0))[0]:
                y0 = np.nonzero(tk[:, x])[0].min()
                m[y0:boty[x] + 1, x] = True
            if self.lip_ab is not None:
                lp = tl.lipness_map(img, self.lip_ab, self.skin_ab)
                m &= ~((lp > 0.5) & ~tl.teeth_key(lab, t.skin_med, 0.8))
            f = max(1.0, 0.006 * t.iod)
            ramp = np.clip((boty[None, :] - rows) / max(1.0, 0.02 * io), 0, 1).astype(np.float32)   # fades at the lip line
            al = cv2.GaussianBlur(m.astype(np.float32), (0, 0), f) * ramp
            out.append((int(h), img, al.astype(np.float32)))
            if len(out) >= k: break
        return out

    def _synth_lower(self):
        """no frame shows a usable lower arch: the upper arch's texture, flipped (incisal edge up), lower incisors
        narrower (x 0.8 along the arch) and a little darker (in the upper teeth's shadow)."""
        U = self.up.tex
        _Th, Tw = self.lo.Th, self.lo.Tw
        aa = self.lo.a / 0.8
        if self.lay:            # v13 anthro: the lower slots onto the upper ones (each lower tooth from the upper tooth of its rank)
            lo_b, up_b = self.lay['lo']['bounds'], self.lay['up']['bounds']
            m = min(len(lo_b), len(up_b))   # the up/lo arches can truncate to different tooth counts; pair them by rank
            aa = np.sign(self.lo.a) * np.interp(np.abs(self.lo.a), lo_b[:m], up_b[:m])
        cu = (aa + self.up.A) * self.up.texel - 0.5
        hh = self.lo.h
        ru = (-hh - self.up.h0) * self.up.texel - 0.5                     # mirror about the incisal edge
        mx, my = np.meshgrid(cu.astype(np.float32), ru.astype(np.float32))
        T = cv2.remap(U, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        T[..., :3] *= 0.85
        self.lo.tex = T.astype(np.float32)
        self.lo.src = np.full(Tw, 3, np.int8)

    def _anthro_unit(self, mode='ipd'):
        """v13: the canonical length of the anthropometric table's IOD (63 mm). 'ipd' (default) = the face's own inter-pupil
        distance: the iris centres (468 / 473) of the median canonical face, or IPD_OCD x lsdub's IOD when they are missing or
        implausible (outside 0.6..0.8 of it). 'iod' = lsdub's own IOD (the outer eye corners, ~1.45 x the IPD): the table read
        literally. Logged with the denture rule's cross-check (bizygomatic 234 / 454 width / 2.2)."""
        t = self.t
        io = t.iod
        Cm = np.asarray(t.Cm, float)
        zyg = round(float(np.linalg.norm(Cm[454, :2] - Cm[234, :2])) / 2.2 / io, 4) if len(Cm) > 454 else None
        if mode == 'iod': return io, dict(mode='iod', unit_iod=1.0, cheek_rule_iod=zyg)
        r = float(np.linalg.norm(Cm[473, :2] - Cm[468, :2])) / io if len(Cm) > 473 else None
        ok = r is not None and 0.6 <= r <= 0.8
        u = (r if ok else IPD_OCD) * io
        return u, dict(mode='ipd', source='iris' if ok else 'fallback', unit_iod=round(u / io, 4), iris_iod=None if r is None else round(r, 4),
                       cheek_rule_iod=zyg)

    def _layout_info(self):
        """v13: teeth.layers.layout -- the table, the unit, per arch the resample (detected lines, period, stretch, knots; IOD), the
        slot boundaries (IOD, both sides, within the arc), the lines and central widths of the texture AS RENDERED (IOD and
        table units)."""
        io = self.t.iod
        u = self.unit_c / io
        r4 = lambda x: None if x is None else round(float(x), 4)
        out = dict(table=dict(names=list(ANTHRO_NAMES), up=list(ANTHRO['up']), lo=list(ANTHRO['lo']), unit='IPD (63 mm)'), unit=self.unit_info,
                   fallback=[f['arch'] for f in self.fallback])
        for k, band in (('up', self.up), ('lo', self.lo)):
            b = [x / io for x in self.lay[k]['bounds'] if x <= band.A + 1e-6]
            slots = sorted({round(-x, 4) for x in b} | {round(x, 4) for x in b})
            lg = dict(getattr(band, 'layout_log', None) or {})
            for key in ('lines', 'knots_src', 'knots_slot', 'skipped'):
                if key in lg: lg[key] = [r4(x / io) for x in lg[key]]
            for key in ('period', 'at', 'slot_w'):
                if key in lg: lg[key] = r4(lg[key] / io)
            if 'stretch' in lg: lg['stretch'] = [r4(x) for x in lg['stretch']]
            for key in ('stretch_med', 'stretch_raw'):
                if key in lg: lg[key] = r4(lg[key])
            lg['n_lines'] = len(lg.get('lines', []))
            lines, (cl, cr) = band.lines_now(io, self.t.skin_med, k == 'up')
            cw = [x for x in (cl, cr) if x is not None]
            out[k] = dict(resample=lg, slots_iod=slots, central_slot_iod=r4(self.lay[k]['bounds'][1] / io), lines_iod=[r4(x) for x in lines],
                          trimmed_iod=[r4(x / band.texel / io) for x in getattr(band, 'trimmed', [0, 0])],
                          central_iod=dict(left=r4(cl), right=r4(cr), mean=r4(np.mean(cw)) if cw else None),
                          central_table=r4(np.mean(cw) / u) if cw else None)
            out[f'slots_{k}'] = slots
        return out

    def _teeth_L(self, band):
        """median L of a band's harvested tooth texels (its own light)."""
        T = band.tex
        m = (T[..., 3] > 0.5) & (np.isin(band.src, (1, 4))[None, :])
        if m.sum() < 10: m = T[..., 3] > 0.5
        if m.sum() < 10: return None
        lab = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        return float(np.median(lab[..., 0][m]))

    def _tex_detail(self, band):
        """v14 (--warp-arch-tex-share): a sharpness score of a band's HARVESTED teeth (src 1 / 4 only, not the modelled /
        mirrored / extended columns): the mean gradient magnitude of L over those tooth texels, in units of IOD-pixels so
        close-ups (more take pixels per tooth) score higher than blurry mid / wide shots. The shared atlas is kept for a shot
        only when it is sharper than the shot's own harvest by this measure."""
        T = band.tex
        m = (T[..., 3] > 0.5) & (np.isin(band.src, (1, 4))[None, :])
        if m.sum() < 30: return 0.0
        L = cv2.cvtColor(np.clip(T[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy)
        mm = m & cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)      # drop the mask edge (a false gradient)
        if mm.sum() < 20: mm = m
        return float(np.mean(g[mm]))                 # compared at equal texel (the orchestration renders source + targets alike)

    def _save_tex(self, path):
        io = self.t.iod
        np.savez(path, up_tex=self.up.tex, up_a=self.up.a / io, up_h=self.up.h / io, lo_tex=self.lo.tex, lo_a=self.lo.a / io, lo_h=self.lo.h / io,
                 iod=io, up_L=self._teeth_L(self.up) or 0.0, lo_L=self._teeth_L(self.lo) or 0.0, up_h_iod=self.up.harvest_h, lo_h_iod=self.lo.harvest_h,
                 up_detail=self._tex_detail(self.up), lo_detail=self._tex_detail(self.lo))

    def _fallback_tex(self, path, min_h, share=False):
        """v8: when this shot's own harvested teeth are shorter than min_h IOD (a few px tall), take the arch texture of
        another shot of the same character (teeth are the same bone), scaled by IOD and L-matched to the own harvest.
        v14 (--warp-arch-tex-share): share = the shared SHARP atlas is the detail source for EVERY shot of that face (the
        close-ups feed the blurry mid-shots), so it is loaded whenever it is SHARPER than this shot's own harvest (taller
        exposed teeth OR the own harvest too short), not only when the own harvest is a few px tall. Teeth identity is
        constant per face, so the same bone is simply shown at this shot's IOD; the per-frame light still scales it."""
        Z = np.load(path)
        io = self.t.iod
        for key, band in (('up', self.up), ('lo', self.lo)):
            own_h = band.harvest_h if (key == 'up' or not self.lo_synth) else 0.0
            src_h = float(Z.get(f'{key}_h_iod', 1.0))
            own_d = self._tex_detail(band)
            src_d = float(Z.get(f'{key}_detail', 0.0))
            if share:          # v14: take the shared atlas when it is SHARPER (gradient energy), or taller AND at least as sharp
                sharper = src_d > 1.05 * own_d
                taller = src_h > own_h + 1e-4 and src_d >= own_d   # never trade detail DOWN for height
                take = sharper or taller or (own_h < min_h)
                why = 'sharper' if sharper else ('taller' if taller else 'short')
            else:              # v8: only the short-harvest fallback
                take = (own_h < min_h or src_h <= own_h)
                why = 'short'
            if not take: continue
            L_own = self._teeth_L(band)
            L_src = float(Z.get(f'{key}_L', 0.0))
            sc = (L_own / L_src) if (L_own and L_src > 1) else 1.0
            band.load_from(Z, key, io, float(np.clip(sc, 1.0, 1.3)))        # (never darker: the per-frame light scales it)
            self.fallback.append(dict(arch=key, own_h=round(own_h, 4), src_h=round(src_h, 4), own_detail=round(own_d, 3), src_detail=round(src_d, 3),
                                      L_scale=round(float(np.clip(sc, 1.0, 1.3)), 3), share=bool(share), why=why))

    def _tongue_colour(self):
        t = self.t
        g = t.F[:, 0]
        idx = np.argsort(-g)[:max(3, len(g) // 5)]
        px = []
        for i in idx:
            img = self.frames[i]
            P = t.S[i][:, :2]
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            m = tl.poly_mask(P[LIP_IN], img.shape, -0.12) & ~tl.teeth_key(lab, t.skin_med) & (lab[..., 0] > 0.45 * t.skin_med) & (lab[..., 1] > 140)
            if m.sum() >= 20: px.append(img[m].astype(float))
        if not px:
            if self.lip_ab is not None:       # v8: the lip colour, darkened
                lab = np.uint8([[[int(np.clip(0.7 * t.skin_med, 0, 255)), int(self.lip_ab[0]), int(self.lip_ab[1])]]])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0].astype(float)
            return np.array([150.0, 135.0, 205.0]) * t.skin_med / 180.0
        return np.median(np.vstack(px), 0)

    def _gum_colour(self):
        """v15 (--warp-arch-gums): the take's own gingival (pink gum) colour, harvested from the widest-smile frames -- the
        frames showing the most upper teeth, where the gum above them is exposed. In each such frame the gum pixels are inside
        the inner-lip polygon, NOT teeth (teeth_key false), reddish (Lab a* raised), of mid lightness (brighter than the dark
        cavity, not as light as enamel), and ABOVE the row band of the visible teeth (so the tongue / lower lip is excluded).
        Returns (BGR median, 'harvest'). If no frame shows enough gum, a warm-pink is SYNTHESIZED from the take's own inner-
        mouth colour statistics (the tongue colour, lightened and pushed pinker) -> (BGR, 'synth'). Our own classical graphics,
        the take's own pixels."""
        t = self.t
        # rank frames by the upper-tooth span they show (the widest smiles)
        sc = np.zeros(len(self.frames))
        for i in range(len(self.frames)):
            img = self.frames[i]
            P = t.S[i][:, :2]
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            m = tl.poly_mask(P[LIP_IN], img.shape, 0.0) & tl.teeth_key(lab, t.skin_med)
            sc[i] = float(m.sum())
        order = np.argsort(-sc)[:max(3, len(self.frames) // 8)]
        px = []
        for i in order:
            if sc[i] <= 0: continue
            img = self.frames[i]
            P = t.S[i][:, :2]
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            inner = tl.poly_mask(P[LIP_IN], img.shape, -0.05)
            tk = tl.teeth_key(lab, t.skin_med) & inner
            if tk.sum() < 20: continue
            rows = np.nonzero(tk.any(1))[0]
            int(rows.min())
            midr = int(np.median(np.nonzero(tk)[0]))   # the teeth's row band
            yy = np.arange(img.shape[0])[:, None]
            above = (yy < midr)                                 # the upper part of the mouth (over the teeth): the upper gum
            gum = inner & ~tl.teeth_key(lab, t.skin_med) & above & (lab[..., 1] > 143) \
                & (lab[..., 0] > 0.42 * t.skin_med) & (lab[..., 0] < 0.92 * t.skin_med)
            if gum.sum() >= 20: px.append(img[gum].astype(float))
        if px:
            return self._gum_pinken(np.median(np.vstack(px), 0).astype(np.float32)), 'harvest'
        # fallback: a warm pink synthesized from the take's own inner-mouth colour (lighter + pinker than the tongue)
        base = np.clip(np.asarray(self.tongue_c, np.float32).reshape(1, 1, 3), 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        lab[0] = min(255.0, lab[0] * 1.15 + 22.0)
        lab[1] = min(255.0, lab[1] + 9.0)
        lab[2] = max(0.0, lab[2] - 2.0)
        bgr = cv2.cvtColor(np.clip(lab, 0, 255).reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)
        return self._gum_pinken(bgr), 'synth'

    def _gum_pinken(self, bgr):
        """v15: nudge a harvested / synthesized gum colour toward a healthy gum pink so it reads as gum in the render even
        though the baked band sits in shadow (and then takes the per-frame light + corridor dimming): lift L off a floor and
        push a* toward pink, gently (a blend, the take's own hue kept). In Lab (0..255). Our own classical colour math."""
        lab = cv2.cvtColor(np.clip(np.asarray(bgr, np.float32).reshape(1, 1, 3), 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        Lf = 0.60 * float(self.t.skin_med)                     # a lightness floor: shadowed gums are lifted toward it
        lab[0] = max(lab[0], Lf) if lab[0] < Lf else lab[0] * 0.5 + Lf * 0.5
        lab[1] = lab[1] + 0.45 * (160.0 - lab[1])              # push a* toward pink
        lab[2] = lab[2] + 0.20 * (146.0 - lab[2])              # a touch warmer, not orange
        return cv2.cvtColor(np.clip(lab, 0, 255).reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)

    # ------------------------------------------------------------------------------------------------ motion --
    def jaw_R(self, i, gap_t=None, p14=None, extra=True):
        th = self.t.jaw_at(i, gap_t, p14) + (self.dth[i] if extra else 0.0)
        return tl.rot_axis(self.t.u, th)

    def proj(self, i, lower=False, R=None):
        """X (canonical, N x 3) -> (frame px N x 2, depth N): pose + upper-lip-root pin (+ jaw + chin pin)."""
        t = self.t
        def f(X):
            X = np.asarray(X, float)
            if lower: X = (X - t.H) @ R.T + t.H
            Y = t.pl.rigid3(i, X)
            xy = Y[:, :2] + t.pin[i] + (self.pin_lo[i] if lower else 0.0)
            return xy, Y[:, 2]
        return f

    def bite_angle(self):
        """the most-closed jaw: the lower incisal edge may rise at most `overbite` above the upper incisal edge (behind
        it: the overjet), never through it."""
        t = self.t
        T = self.lo.o
        delta = (self.e_up - self.overbite) - self.e_lo      # along d (canonical y ~ d)
        return tl.jaw_angle(T, t.H, t.u, t.d, delta)

    def jaw_floor(self, P14):
        """per-frame lower bound on the jaw angle: the bite (bite_angle) and the exposure limit -- the lower incisal edge
        sits at most `expose` IOD above the target lower lip (the lip covers the rest of the arch; a jaw plan that lags a
        closing lip would otherwise show a whole row of lower teeth over it)."""
        t = self.t
        b = self.bite_angle()
        ex = np.array([t.jaw(None, i, P14[i], lim=(-60, 60), extra=t.e_lo - self.expose * t.iod) for i in range(len(P14))])
        return np.maximum(b, ex)

    def arc_tangent(self, i, R):
        """d(frame px of the lower incisal edge) / d(jaw angle) at frame i with the lower rotation R: the tangent of the
        hinge arc in the picture (px / rad). The projection of a 2D pin onto it is the only motion the lower teeth may
        take from the chin (a rotation about the condyle, never a translation off the arc)."""
        t = self.t
        Ti = (self.lo.o - t.H) @ R.T + t.H
        tan = np.cross(t.u, Ti - t.H)
        return (t.pl.s[i] * t.pl.R[i] @ tan)[:2]

    def exposure_over(self, i, R, p14):
        """px the lower incisal edge sits above the exposure line (target lower lip - expose) along the head's down axis
        in frame i (> 0: too much lower tooth shows), and the rate at which opening the jaw lowers it (px / rad)."""
        t = self.t
        io = t.iod * t.pl.s[i]
        e2, _ = self.proj(i, True, R)(self.lo.o[None])
        dn = (t.pl.R[i] @ t.d)[:2]
        dn = dn / max(1e-6, np.linalg.norm(dn))
        over = float((np.asarray(p14, float)[:2] - e2[0]) @ dn) - self.expose * io
        return over, float(self.arc_tangent(i, R) @ dn), dn

    def set_chin_pin(self, outs, P_out, sigma=None, noise=0.5):
        """the lower teeth share the chin skin's motion: per frame, the high-pass of (DIS flow track of the soft chin in
        the warped frames - where the skull pose + upper pin + planned jaw put the chin), added to the lower arch only.
        The plan keeps the slow placement, the skin gives every frame-to-frame step (as the upper pin does at the lip
        root). v8 pin_mode 'arc': that correction is split into its slow part (kept) and its fast residual (gain
        pin_gain, zero-phase smoothed), then projected onto the jaw hinge's arc as an extra jaw angle per frame (dth),
        bounded by the exposure limit; pin_lo stays zero (nothing translates off the arc)."""
        t = self.t
        sigma = self.chin_sigma if sigma is None else sigma
        n = len(outs)
        Xc = t.Cm[CHIN].mean(0)[None]
        self.dth = np.zeros(self.n)
        model = np.array([self.proj(i, True, self.jaw_R(i, None, P_out[i][14]))(Xc)[0][0] for i in range(n)])
        iod = t.iod * t.pl.s[:n]
        c = tl.skin_pin(outs, np.asarray(P_out), model, min(self.h_lt, n - 1), CHIN, sigma, noise, iod)
        self.pin_raw = c
        if self.pin_mode != 'arc':
            self.pin_lo = c
            self.pin_disp = c.copy()
            return self.pin_lo
        self.pin_lo = np.zeros((n, 2))
        slow = gaussian_filter1d(c, self.pin_split, axis=0, mode='nearest') if self.pin_split > 0 else c
        fast = c - slow
        if self.pin_gain > 0 and self.pin_split > 0: fast = gaussian_filter1d(fast, 1.0, axis=0, mode='nearest')
        c2 = slow + self.pin_gain * fast
        self._pin_slow, self._pin_fast = slow, fast          # (v9: the maker's teeth_plan re-takes one frame at gain 0)
        dth = np.zeros(n)
        disp = np.zeros((n, 2))
        for i in range(n):
            R = self.jaw_R(i, None, P_out[i][14], extra=False)
            v = self.arc_tangent(i, R)
            vv = float(v @ v)
            dth[i] = float(c2[i] @ v) / vv if vv > 1e-9 else 0.0
        # the exposure limit, folded into the angle: the pin never lifts the lower incisal edge past the line the clamp
        # would catch (two Newton steps: the arc is nearly straight at these angles)
        for i in range(n):
            for _ in range(2):
                self.dth[i] = dth[i]
                R = self.jaw_R(i, None, P_out[i][14])
                over, rate, _ = self.exposure_over(i, R, P_out[i][14])
                if over > 1e-3 and rate > 1e-6: dth[i] += over / rate
                else: break
            self.dth[i] = dth[i]
            disp[i] = dth[i] * self.arc_tangent(i, self.jaw_R(i, None, P_out[i][14], extra=False))
        self.dth = dth
        self.pin_disp = disp
        return self.pin_lo

    def dth_for(self, i, gain=0.0, P14=None):
        """v9 (the gate's teeth_plan): the arc pin's extra jaw angle of frame i with another fast-residual gain (0 = the
        lower teeth follow the slow chin motion + the jaw plan only), the exposure limit folded in as set_chin_pin does."""
        if getattr(self, '_pin_slow', None) is None: return float(self.dth[i])
        c2 = self._pin_slow[i] + gain * self._pin_fast[i]
        R = self.jaw_R(i, None, P14, extra=False)
        v = self.arc_tangent(i, R)
        vv = float(v @ v)
        d = float(c2 @ v) / vv if vv > 1e-9 else 0.0
        old = float(self.dth[i])
        try:
            for _ in range(2):
                self.dth[i] = d
                R = self.jaw_R(i, None, P14)
                over, rate, _ = self.exposure_over(i, R, P14)
                if over > 1e-3 and rate > 1e-6: d += over / rate
                else: break
        finally:
            self.dth[i] = old
        return float(d)

    def cover_lip_of(self, i):
        d = dict(self.cover_lip)
        return round(float(d.get(i, 0.0)), 4)

    def pin_log(self, i):
        """the chin pin as applied at the lower incisal edge, frame px (for the log / the p95 metric)."""
        return self.pin_disp[i] if i < len(self.pin_disp) else np.zeros(2)

    def prepare(self, gaps, vis=None):
        """v8: per-frame layer parameters over the whole clip, zero-phase eased (smooth frames): the opening share (the
        cavity's throat and the tongue's lift) and the tongue's vowel pose (dorsum rise, tip rise, retraction)."""
        t = self.t
        n = self.n
        g = np.asarray(gaps, float)[:n]
        op = smootherstep((g / max(1e-6, t.g_hi) - 0.4) / 0.6)
        tg = np.zeros((n, 3))
        if vis is not None:
            for i in range(n): tg[i] = TONGUE_VOWEL.get(vis[min(i, len(vis) - 1)], (0.0, 0.0, 0.0))
        if self.smooth_f > 0:
            op = gaussian_filter1d(op, self.smooth_f, mode='nearest')
            tg = gaussian_filter1d(tg, max(self.smooth_f, 2.0), axis=0, mode='nearest')
        else: tg = gaussian_filter1d(tg, 2.0, axis=0, mode='nearest')
        self.open_s, self.tg, self.prepared = op, tg, True
        return self

    def plan_single(self, P_out, sigma=1.0, tongue_sigma=2.0, lo_limit=0.09, pin='none'):
        """v12 --warp-reg single: ONE reference for the whole interior (founder on v10/v11: the sub-elements' registrations
        "slosh" across frames; v11 measured that the placement rule, not the pins, drives it). Physically the upper teeth
        are bone fixed to the skull, the lower teeth rotate with the jaw about the condyle hinge (one angle), the lips slide
        over both, and the cavity is whatever the lips enclose that is not teeth or tongue. So:
          L4  the upper arch: the skull pose only (no lip-root skin pin; the caller skips set_pin), its texture fixed in
              the skull frame
          L3  the lower arch + tongue: the skull pose + ONE hinge rotation theta (the hinge geometry = the shot's own,
              teeth_layer.hinge_point, fixed once per shot). theta = the clip's jaw plan (teeth_layer.plan: the hinge
              angle that puts the lower harvest frame's lower lip at the PLANNED lower inner lip -- the planned opening
              below the skull-fixed upper lip --, solved once with smoothness / speed limits / the bite and exposure
              floors), low-passed zero-phase (sigma frames: the jaw is heavy), then the hard floors: never through the
              upper teeth (the bite) and the lower lip never more than lo_limit IOD below the lower incisal edge (the
              lower arch's limit: no gum-less gap under the lower teeth), absorbed smoothly (smooth-max). No chin pin, no
              per-frame exposure kick: nothing per frame but theta
          tongue  a slow function of theta (its own low-pass, tongue_sigma): the opening share theta implies drives the
              wedge tongue's drop / retraction and the cavity's throat; no per-frame vowel re-fit (tg = 0)
          L2  the planned inner-lip polygon (as v11; behind the arches), its throat shading from the same slow share
          guard   the planned upper inner lip never lifts above the upper arch's band: the band's gum is extended upward
              when the plan reaches past it (no cavity where gum should be)
        pin 'global' / 'nose' (--warp-reg-pin): the caller DID set one skin pin (teeth_layer.set_pin: the upper lip-root
        skin of the warped frames / the nose of the take) and it moves the WHOLE interior as one residual (both arches + the
        tongue ride proj(), which adds it; the jaw angle is taken with it removed, as v9): the sub-elements stay registered
        to each other, the dentition follows the real face where the skull pose misses it. 'none' (default): the skull
        pose alone, as specified.
        The lips (L5), the cover (L1) and the face (L6) are untouched."""
        t = self.t
        n = self.n
        io = t.iod
        self.pin_mode = 'single'
        self.pin_lo = np.zeros((n, 2))
        self.dth = np.zeros(n)
        self.pin_disp = np.zeros((n, 2))
        P14 = [np.asarray(P_out[i], float)[14] for i in range(n)]
        if t.jaw_plan is None:          # (--warp-teeth-jaw frame: the per-frame hinge angle of the planned lower lip)
            t.jaw_target = np.array([t.jaw(None, i, P14[i], lim=(-60, 60)) for i in range(n)])
            t.jaw_hide = None
            plan = t.jaw_target.copy()
        else: plan = np.asarray(t.jaw_plan, float).copy()
        bite = float(self.bite_angle())
        gum = np.zeros(n)               # the jaw angle that puts the lower incisal edge exactly lo_limit above the planned lip
        for i in range(n):              # (frame px along the head's down axis, as exposure_over; Newton on the hinge arc)
            th_ = t.jaw(None, i, P14[i], lim=(-60, 60), extra=t.e_lo - lo_limit * io)
            dn = (t.pl.R[i] @ t.d)[:2]
            dn = dn / max(1e-6, np.linalg.norm(dn))
            for _ in range(4):
                R = tl.rot_axis(t.u, th_)
                e2, _ = self.proj(i, True, R)(self.lo.o[None])
                ov = float((P14[i][:2] - e2[0]) @ dn) - lo_limit * io * t.pl.s[i]
                rate = float(self.arc_tangent(i, R) @ dn)
                if abs(ov) < 1e-3 or rate <= 1e-6: break
                th_ += ov / rate
            gum[i] = th_
        floor = np.maximum(bite, gum)
        lp = (lambda x: gaussian_filter1d(x, sigma, mode='nearest')) if sigma > 0 else (lambda x: x)
        th = lp(plan)
        for _ in range(3): th = lp(np.maximum(th, floor))         # smooth-max: a floor is met by a smooth detour
        raised = th < floor - 1e-9
        th = np.clip(np.maximum(th, floor), *np.radians((-25.0, 30.0)))
        t.jaw_plan = th
        # the tongue / throat: the opening share the (separately low-passed) jaw angle implies, mapped as prepare() maps a gap
        tht = gaussian_filter1d(th, tongue_sigma, mode='nearest') if tongue_sigma > 0 else th
        g0 = float(t.F[self.h_lt, 0])
        jg = t.jaw_gain
        a0 = tl.jaw_angle(t.T_lo, t.H, t.u, t.d, jg * (0.0 - g0))
        a1 = tl.jaw_angle(t.T_lo, t.H, t.u, t.d, jg * (t.g_hi - g0))
        g_eq = (tht - a0) / (a1 - a0) * t.g_hi if abs(a1 - a0) > 1e-9 else np.full(n, g0)
        self.open_s = smootherstep((g_eq / max(1e-6, t.g_hi) - 0.4) / 0.6)
        self.tg = np.zeros((n, 3))
        self.prepared = True
        # the upper gum guard: how far the planned upper inner lip lifts above the upper incisal edge (skull frame, IOD)
        up_pts = [LIP_IN.index(k) for k in (80, 81, 82, 13, 312, 311, 310)]
        lift = np.zeros(n)
        for i in range(n):
            Ai = tl.inv_aff(tl.plane_affine(t.pl, i, self.z_up, ref=t.ref))
            q = np.asarray(P_out[i], float)[LIP_IN][up_pts, :2]
            lift[i] = (self.e_up - float((q @ Ai[:, :2].T + Ai[:, 2])[:, 1].min())) / io
        band = -self.up.h0 / io
        ext = 0.0
        if lift.max() > band - 0.03:
            ext = float(lift.max() + 0.05 - band)
            add = int(np.ceil(ext * io * self.up.texel))
            self.up.tex = np.concatenate([np.repeat(self.up.tex[:1], add, 0), self.up.tex], 0)      # the gum's top row, upward
            self.up.h0 -= add / self.up.texel
            self.up.Th += add
            self.up.h = (np.arange(self.up.Th) + 0.5) / self.up.texel + self.up.h0
        d1 = lambda x: np.abs(np.diff(np.degrees(x)))
        self.single = dict(sigma=sigma, tongue_sigma=tongue_sigma, lo_limit=lo_limit, pin=pin, bite_deg=round(np.degrees(bite), 3),
                           floor_frames=int((np.degrees(floor - lp(plan)) > 0.05).sum()), raised_frames=int(raised.sum()),
                           jaw_step_p95_deg=dict(plan=round(float(np.percentile(d1(plan), 95)), 3), single=round(float(np.percentile(d1(th), 95)), 3)),
                           up_lift_max_iod=round(float(lift.max()), 4), up_band_iod=round(band, 4), up_band_ext_iod=round(ext, 4),
                           over={})
        return self

    # ------------------------------------------------------------------------- v12: the lip band (--warp-arch-build v12) --
    def rim_colours(self, img, Q, Qo, io, width=0.025):
        """v12: the colour of the PLANNED lips' inner rim in this frame (img = the warped frame's box: L5's own pixels), upper and
        lower: the lip pixels (lipness > 0.5) within `width` IOD outside the planned inner contour Q (and inside the outer lip),
        darkened as L5 darkens its inner rim (lip_rim strength x its falloff); median BGR per side (None: too few pixels)."""
        h, w = img.shape[:2]
        opening = tl.poly_mask(Q, (h, w), 0.0)
        dout = cv2.distanceTransform((~opening).astype(np.uint8), cv2.DIST_L2, 3)
        band = (dout > 0.5) & (dout <= max(1.5, width * io)) & tl.poly_mask(Qo, (h, w), 0.0, 0.02 * io)
        if self.lip_ab is not None: band &= tl.lipness_map(img, self.lip_ab, self.skin_ab) > 0.5
        yy, xx = np.mgrid[0:h, 0:w]
        cl, cr = Q[LIP_IN.index(78)], Q[LIP_IN.index(308)]
        e = cr - cl
        nrm = np.array([-e[1], e[0]]) / max(1e-6, float(np.linalg.norm(e)))                 # down the mouth
        below = ((xx - 0.5 * (cl[0] + cr[0])) * nrm[0] + (yy - 0.5 * (cl[1] + cr[1])) * nrm[1]) > 0
        dark = 1 - self.rim[0] * np.exp(-dout / max(0.8, self.rim[1] * io))
        out = {}
        for k, m in (('up', band & ~below), ('lo', band & below)):
            out[k] = np.median(img[m].astype(np.float32) * dark[m][:, None], 0) if m.sum() >= 12 else None
        return out

    def lip_band(self, rgb, hm, band, upper, col):
        """v12: the plate's band beyond the crown (where v8..v11 drew the gum: the harvest lip's contact line and a tongue-
        derived pink, read by the founder as the old mouth's lip bleeding through / a second row of teeth) takes the colour of
        the planned lip's inner rim (col, this frame), a little darker toward the teeth (0.9 at the crown .. 1.0 at 0.06 IOD);
        hm: the band's h per pixel (canonical px, ArchBand.render(hmap=True)). The gum guard's extended rows are in it too."""
        self._gw = None
        if col is None or hm is None: return rgb
        io = self.t.iod
        th = getattr(band, 'tooth_h_c', 0.11 * io) + getattr(band, 'gum_h_c', 0.0)
        tr = max(1e-3, getattr(band, 'tr_c', 0.012 * io))
        d = np.where(np.isfinite(hm), (-hm - th) if upper else (hm - th), -1e9)
        gw = smootherstep(d / tr)[..., None]
        shade = (0.9 + 0.1 * smootherstep(d / (0.06 * io)))[..., None]
        self._gw = gw[..., 0].astype(np.float32)                # (the lips' contact shadow is not cast on their own inner surface)
        return (rgb * (1 - gw) + np.asarray(col, np.float32)[None, None, :] * shade * gw).astype(np.float32)

    def lipband_ruler(self, comp, b, width=0.02):
        """v12 ruler (logged per frame, frames[i].reg.lipband): the composited plate band right under the planned lip (inside the
        opening, within `width` IOD of the inner contour, where the arch is opaque) vs the lip's inner rim right outside it,
        upper and lower: the mean Lab distance (dE) of the two bands, their pixel counts, and the share of the plate band that
        is the band beyond the crown (gum / lip band) rather than tooth (gum)."""
        Q, io = b['Q'], b['io']
        h, w = comp.shape[:2]
        opening = tl.poly_mask(Q, (h, w), 0.0)
        din = cv2.distanceTransform(opening.astype(np.uint8), cv2.DIST_L2, 3)
        dout = cv2.distanceTransform((~opening).astype(np.uint8), cv2.DIST_L2, 3)
        lab = cv2.cvtColor(comp, cv2.COLOR_BGR2LAB).astype(np.float32)
        W = max(1.5, width * io)
        yy, xx = np.mgrid[0:h, 0:w]
        cl, cr = Q[LIP_IN.index(78)], Q[LIP_IN.index(308)]
        e = cr - cl
        nrm = np.array([-e[1], e[0]]) / max(1e-6, float(np.linalg.norm(e)))
        below = ((xx - 0.5 * (cl[0] + cr[0])) * nrm[0] + (yy - 0.5 * (cl[1] + cr[1])) * nrm[1]) > 0
        lp = tl.lipness_map(comp, self.lip_ab, self.skin_ab) > 0.5 if self.lip_ab is not None else np.ones((h, w), bool)
        out = {}
        for k, side, a_, hm, band in (('up', ~below, b['a4'], b['up_h'], self.up), ('lo', below, b['a3'], b['lo_h'], self.lo)):
            plate = opening & (din > 0.5) & (din <= W) & side & (a_ > 0.5)
            rim = ~opening & (dout > 0.5) & (dout <= W) & side & lp
            if plate.sum() < 12 or rim.sum() < 12:
                out[k] = None
                continue
            th = getattr(band, 'tooth_h_c', 0.11 * self.t.iod)
            g = np.isfinite(hm) & (((-hm) if k == 'up' else hm) > th)
            out[k] = dict(dE=round(float(np.linalg.norm(lab[plate].mean(0) - lab[rim].mean(0))), 2), n=int(plate.sum()), n_rim=int(rim.sum()),
                          gum=round(float((g & plate).sum()) / float(plate.sum()), 3))
        return out

    # --------------------------------------------------------------- v13: the buccal corridor (--warp-arch-corridor K) --
    def corridor_fade(self, Q, h, w):
        """v13: the buccal corridor of this frame's PLANNED opening (Q: the warped inner contour the layers use, box px): W = its
        width corner to corner (78 -> 308), c = their mid-point, s = a pixel's position along the corner line from c. The fade is
        0 within |s| <= (0.5 - K) W (the plate is never touched there) and rises with a smootherstep to 1 at |s| = 0.5 W (the
        corners) and beyond -> (h x w float32); the geometry is kept for the log."""
        K = self.corridor
        cl, cr = np.asarray(Q[LIP_IN.index(78)], float), np.asarray(Q[LIP_IN.index(308)], float)
        e = cr - cl
        W = float(np.linalg.norm(e))
        if W < 1e-6:
            self._cor = None
            return None
        e = e / W
        c = 0.5 * (cl + cr)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        s = (xx - c[0]) * e[0] + (yy - c[1]) * e[1]
        f = smootherstep((np.abs(s) - (0.5 - K) * W) / max(1e-6, K * W)).astype(np.float32)
        self._cor = dict(c=c, e=e, W=W, s=s)
        return f

    def corridor_dim(self, rgb, cav, f):
        """v13: a plate's colour toward the cavity (L2's own dark at that pixel) by the corridor's fade, down to a faint crown at
        the corners (corridor_floor of the crown's contrast over the cavity stays: the back teeth are dark, not black); pixels
        with fade 0 are returned unchanged. (The caller leaves v12's lip band out: f x (1 - its weight).)"""
        keep = (1.0 - (1.0 - self.corridor_floor) * f)[..., None]
        return np.where(f[..., None] > 0, cav + (rgb - cav) * keep, rgb).astype(np.float32)

    def corridor_log(self, i, f, box, io, up_a, lo_a, clip, f_up=None, f_lo=None):
        """v13: frame i's corridor (frames[i].reg.corridor): W_f (lsdub IOD and the table's IPD unit), the corridor's start
        points (+-(0.5 - K) W_f from c) in the skull-stabilised frame (reg's frame: x, y IOD from the upper arch's origin) and
        the upper arch's arc position there (IOD; the anthro slot it falls in), the dark fraction of each plate's visible part
        (the share of its clipped alpha weighted by the fade as applied: f_up / f_lo, v12's lip band left out), and arc_short:
        the rendered arch ending before the corridor starts on
        a side (its unclipped alpha > 0.5 does not reach +-(0.5 - K) W_f along the corner line)."""
        g = self._cor
        if g is None: return dict(W=0.0)
        t = self.t
        K = self.corridor
        W, c, e, s = g['W'], g['c'], g['e'], g['s']
        io_c = t.iod
        r4 = lambda x: round(float(x), 4)
        A = tl.plane_affine(t.pl, i, self.z_up, ref=t.ref)
        Ai = tl.inv_aff(A)
        org = np.asarray(self.ref[:2], float)
        st = lambda xy: (np.asarray(xy, float) @ Ai[:, :2].T + Ai[:, 2] - org) / io_c
        x0, y0 = box[0], box[1]
        starts, arcs, slots = [], [], []
        Ax, Xx = self.up.tab[0], self.up.tab[1]                       # the upper arch: arc length <-> lateral x (canonical px)
        for sg in (-1, 1):
            p = c + sg * (0.5 - K) * W * e + [x0, y0]
            q = st(p[None])[0]
            starts.append([r4(q[0]), r4(q[1])])
            a_ = float(np.interp(abs(q[0]) * io_c, Xx, Ax)) / io_c
            arcs.append(r4(np.sign(q[0]) * a_))
            if self.lay: slots.append(ANTHRO_NAMES[min(len(ANTHRO_NAMES) - 1, int(np.searchsorted(self.lay['up']['bounds'], a_ * io_c, side='right') - 1))])
        out = dict(W=r4(W / io), W_ipd=r4(W / io / (self.unit_c / io_c)), K=K, start=starts, start_arc=arcs, start_slot=slots or None)
        for k, al, fk in (('up', up_a, f if f_up is None else f_up), ('lo', lo_a, f if f_lo is None else f_lo)):
            vis = al * clip
            tot = float(vis.sum())
            out[f'dark_{k}'] = r4(float((vis * fk).sum()) / tot) if tot > 1e-3 else None
            m = al > 0.5
            if m.any():
                lo_s, hi_s = float(s[m].min()), float(s[m].max())
                lim = (0.5 - K) * W
                out[f'short_{k}'] = [bool(lo_s > -lim + 1.0), bool(hi_s < lim - 1.0)]
                out[f'extent_{k}'] = [r4(lo_s / io), r4(hi_s / io)]
            else: out[f'short_{k}'] = None
        self.cor_frames[int(i)] = out
        return out

    def corridor_info(self, frames=None):
        """v13: teeth.layers.corridor -- K, the floor, W_f (IOD) p50 / p95 / max, the dark fractions p50 / p95, the frames whose
        upper / lower arch ends short of the corridor (arc_short). frames: the delivered frames' own per-frame entries (the
        warp's final log; a gate candidate rendered and rejected leaves its values in cor_frames), else cor_frames."""
        F = self.cor_frames if frames is None else {i: c for i, c in enumerate(frames) if c}
        ks = sorted(F)
        q = lambda v, p: round(float(np.percentile(v, p)), 4) if len(v) else None
        W = [F[i]['W'] for i in ks if F[i].get('W')]
        out = dict(K=self.corridor, floor=self.corridor_floor, frames=len(ks),
                                                               W_iod=dict(p50=q(W, 50), p95=q(W, 95), max=round(max(W), 4) if W else None))
        for k in ('up', 'lo'):
            d = [F[i][f'dark_{k}'] for i in ks if F[i].get(f'dark_{k}') is not None]
            out[f'dark_{k}'] = dict(p50=q(d, 50), p95=q(d, 95))
            out[f'arc_short_{k}'] = [i for i in ks if F[i].get(f'short_{k}') and any(F[i][f'short_{k}'])]
        sa = [abs(x) for i in ks for x in (F[i].get('start_arc') or [])]
        out['start_arc_iod'] = dict(p50=q(sa, 50), min=round(min(sa), 4) if sa else None)
        return out

    # ------------------------------------------------------------------------------------------------ layers --
    def _box(self, i, P_take, P_out, W, H):
        io = self.t.iod * self.t.pl.s[i]
        q = np.vstack([np.asarray(P_take)[LIP_OUT, :2], np.asarray(P_out)[LIP_OUT, :2]])
        x0, y0 = np.floor(q.min(0) - 0.3 * io).astype(int)
        x1, y1 = np.ceil(q.max(0) + 0.3 * io).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        return x0, y0, x1 - x0, y1 - y0

    def cavity(self, i, Q, box, gap_t, io):
        """L2: (BGR, alpha) in box; Q = target inner contour (box px)."""
        t = self.t
        h, w = box[3], box[2]
        g = float(t.light[i])
        cav = np.clip(np.asarray(t.cavity, np.float32) * g / max(1e-3, t.skin_med), 0, 255)
        e = 0.03 * io
        al = aa_poly(Q, (h, w), e)                                     # v17e: supersampled (the corner stair-step)
        al = cv2.GaussianBlur(al, (0, 0), 0.9)
        c = Q.mean(0)
        rx = max(2.0, np.ptp(Q[:, 0]) / 2)
        ry = max(2.0, np.ptp(Q[:, 1]) / 2)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        vx = (xx - c[0]) / rx
        vy = (yy - c[1]) / ry
        o = float(smootherstep((gap_t / max(1e-6, t.g_hi) - 0.3) / 0.6))       # how wide open: the throat shows more
        if self.prepared: o = float(np.clip(self.open_s[i] * 1.2, 0, 1))
        top = 0.55 + 0.45 * smootherstep((vy + 1.0) / 1.6)                        # darker at the top (under the palate)
        side = 1 - 0.35 * smootherstep((np.abs(vx) - 0.55) / 0.45)                # and toward the cheeks
        thr = np.exp(-((vx / 0.45) ** 2 + ((vy - 0.25) / 0.55) ** 2))             # faint throat, low in the centre
        warm = 0.6 * cav + np.array([14, 18, 52], np.float32)
        k = (0.35 + 0.25 * o) * thr
        rgb = (cav * (1 - k[..., None]) + warm * k[..., None]) * (top * side)[..., None]
        if self.lower_real > 0:          # v16: an AO ramp deepening the cavity floor (where the lowers emerge), scaled by openness,
            floor_ao = self.lower_p['ao'] * float(smootherstep((o - 0.2) / 0.6))   # so the lowers sit in FRONT of a receding dark, not on a lit floor
            low = smootherstep((vy - 0.1) / 0.9)
            rgb = rgb * (1.0 - 0.35 * floor_ao * low)[..., None]
        # v17e: the corner wedges shade toward the lip crease (a dark-red fold, not a black hole poking past the lips)
        kc = np.clip((np.abs(vx) - 0.80) / 0.20, 0.0, 1.0)[..., None]
        rgb = rgb * (1 - 0.7 * kc) + np.array([46.0, 50.0, 112.0], np.float32) * (0.7 * kc)
        return np.clip(rgb, 0, 255).astype(np.float32), al

    def _compute_show(self, i):
        """v16.1 (--warp-lower-real): how much of a real BRIGHT LOWER tooth row the TAKE itself bares at frame i -- the AREA
        (IOD^2 units) of bright, near-neutral enamel pixels (Lab L* > 55, low chroma) in the LOWER half of the take's own
        inner-mouth polygon. An ABSOLUTE area (not a fraction of the growing opening), so a real smile row reads full from
        barely to fully open; a faint edge or tongue/lip reads near 0. ~1 = a full lower row, 0 = the lowers hidden."""
        take = self.frames[i]
        P = np.asarray(self.S[i], float)[:, :2]
        inner = tl.poly_mask(P[LIP_IN], take.shape[:2], 0.0)
        if inner.sum() < 20: return 0.0
        ys = np.nonzero(inner.any(1))[0]
        cy = int(0.5 * (ys[0] + ys[-1]))       # the mouth-opening mid row
        lower = inner.copy()
        lower[:cy] = False
        if lower.sum() < 10: return 0.0
        lab = cv2.cvtColor(take, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0] * 100.0 / 255.0
        a_ = lab[..., 1] - 128.0
        b_ = lab[..., 2] - 128.0
        teeth = (L > 55.0) & (np.sqrt(a_ * a_ + b_ * b_) < 22.0) & lower            # bright, near-neutral = exposed enamel
        io2 = (self.t.iod * float(self.t.pl.s[i])) ** 2
        return float(np.clip(float(teeth.sum()) / max(1.0, 0.0090 * io2), 0.0, 1.0))

    def _lower_show_at(self, i):
        """v16.1 (--warp-lower-real): the founder's 'match the take' at the FRAME level. 0 = the take hides the lowers (wide-open
        'ah', rounded 'oo'): dim / recede / de-grill them; 1 = the take bares a real lower row (a smile / laugh): keep them BRIGHT
        (v15). The whole-clip signal is computed once from the take and TEMPORALLY SMOOTHED (a small gaussian) so the per-frame
        gating of the lower shadow does not flicker. A per-frame override in self._lshow (tests) wins."""
        if i in self._lshow: return float(self._lshow[i])
        if self._lshow_arr is None:
            self._lshow_arr = gaussian_filter1d(np.array([self._compute_show(j) for j in range(self.n)], np.float32), 1.5)
        return float(self._lshow_arr[i])

    def _lower_depth(self, lo_rgb, lo_a, i, h, w, io):
        """v16 (--warp-lower-real): per-frame, light the lower arch BELOW the uppers. A depth attenuation scaled by how wide
        the mouth is open (wider -> dimmer) toward self.lower_target -- derived from this take's OWN lower/upper tooth-display
        ratio (a top-teeth talker dims harder, a bottom-teeth talker less) -- and a soft cast shadow of the cavity / upper lip
        onto the TOP (biting edge) of the lowers. Near-closed / smile frames stay bright (the take shows a real lower row
        there). Returns the attenuated lo_rgb."""
        p = self.lower_p
        hide = 1.0 - self._lower_show_at(i)                       # how much the take hides the lowers at this frame (the primary gate)
        floor = min(float(self.lower_target), 0.48)               # the darkest the crown goes when fully hidden (dark enough to de-grill
        dim = 1.0 - p['dim'] * hide * (1.0 - floor)               # any face); bright+matched where the take bares a row
        out = lo_rgb * dim
        cast = p['cast'] * hide
        if cast > 1e-3:
            a = lo_a > 0.3
            cb = a.any(0)
            top = np.where(cb, a.argmax(0), h)   # the top (biting edge) row of the lowers per column
            rows = np.arange(h)[:, None]
            dd = (rows - top[None, :]).astype(np.float32)
            sh = 1.0 - 0.5 * cast * np.exp(-np.maximum(dd, 0.0) / max(1.0, 0.045 * io))
            out = out * np.where(a, sh, 1.0).astype(np.float32)[..., None]
        return out

    def _lower_shadow(self, lo_rgb, lo_a, lo_h, i, io):
        """v16.1 (--warp-lower-real): the lower-teeth incisal shadow, applied PER FRAME (the atlas is v15-bright) and gated by
        the take's own per-frame lower display, so a bright smile row is NOT dragged down by a baked darkening. Where the take
        HIDES the lowers (show -> 0, shadow -> 1): the biting tips darken into the dark cavity and their top alpha feathers (no
        bright flat strip, no hard line -- the grill fix); where the take BARES a real row (show -> 1, shadow -> 0): nothing is
        applied, the crown stays v15-bright (the smile recovers). lo_h = the band's canonical height per pixel (0 at the incisal
        edge, + toward the gum). Returns (lo_rgb, lo_a)."""
        p = self.lower_p
        shadow = 1.0 - self._lower_show_at(i)
        if shadow <= 1e-3 or lo_h is None: return lo_rgb, lo_a
        d = np.nan_to_num(lo_h, nan=1e6).astype(np.float32)          # canonical px from the incisal edge (lower: + toward the gum)
        feather = max(1.0, 0.030 * io)
        tk = lo_a > 0.3
        # (a) darken the biting tips toward the cavity (floor at the tip -> full crown brightness a feather down), x shadow x ao
        floor = 1.0 - 0.45 * float(p['ao'])
        ramp = floor + (1.0 - floor) * smootherstep(np.clip(d / (1.6 * feather), 0.0, 1.0))
        dark = 1.0 - shadow * (1.0 - ramp)
        zone = tk & (d >= -feather) & (d < 1.6 * feather)
        lo_rgb = np.where(zone[..., None], lo_rgb * dark[..., None], lo_rgb)
        # (b) feather the top alpha into the dark cavity (no hard opaque line); the crown body stays solid
        aramp = smootherstep(np.clip(d / feather, 0.0, 1.0))
        topm = tk & (d >= -feather) & (d < feather)
        lo_a = np.where(topm, lo_a * (1.0 - shadow * float(p['feather']) * (1.0 - aramp)), lo_a)
        # (c) the shadowed enamel reads a touch cooler / desaturated vs the warm interior (x shadow x cool)
        c = shadow * float(p['cool'])
        if c > 1e-3:
            crown = tk & (d >= -feather) & (d <= 0.11 * io)
            if crown.any():
                lab = cv2.cvtColor(np.clip(lo_rgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                a_, b_ = lab[..., 1], lab[..., 2]
                lab[..., 1] = np.where(crown, np.clip(a_ - 0.30 * c * (a_ - 128.0), 0, 255), a_)
                lab[..., 2] = np.where(crown, np.clip(b_ - 0.40 * c * (b_ - 128.0) - 4.0 * c, 0, 255), b_)
                lo_rgb = np.where(crown[..., None], cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32), lo_rgb)
        return lo_rgb, lo_a

    def tongue(self, i, R, box, io_c, kick=0.0):
        """L3 tongue: an ellipsoid behind / under the lower incisors, riding the jaw."""
        t = self.t
        o = self.lo.o
        # the tongue's front sits a little higher in a wide-open mouth (more of it shows over the lower teeth)
        Cc = o + t.b * (0.12 * io_c) + t.d * ((self.tongue_y - 0.025 * self._open) * io_c)
        ax = np.stack([t.u * 0.21 * io_c, t.d * 0.04 * io_c, t.b * 0.13 * io_c], 1)   # wide: its sides run under the teeth        # columns: semi-axis vectors
        f = self.proj(i, True, R)
        c2, _ = f(Cc[None])
        c2 = c2 + kick
        Lm = (t.pl.s[i] * t.pl.R[i] @ R @ ax)[:2]
        col = np.clip(self.tongue_c * float(t.light[i]) / max(1e-3, t.skin_med), 0, 255)
        return ellipsoid_layer(c2[0] - [box[0], box[1]], Lm, (0, 0, box[2], box[3]), col)

    def _cover_amt(self, i, op=None):
        """v15 (--warp-tongue-cover): how much the tongue should cover the lower crowns at frame i, from the mouth geometry:
        0 near-closed (the lowers are hidden by the lip anyway), rising to 1 at mid-open (a real tongue fills the floor and
        the lowers are naturally occluded), back to 0 wide-open (the teeth-baring 'ah' / smile poses that actually show the
        lower teeth). The master knob --warp-tongue-cover scales it; 0 = the v8 envelope, untouched."""
        if op is None: op = float(self.open_s[i]) if (self.prepared and self.open_s is not None) else float(self._open)
        rise = smootherstep((op - 0.12) / 0.16)
        fall = 1.0 - smootherstep((op - 0.55) / 0.30)
        amt = rise * fall
        if self.lower_real > 0:          # v16: where the take hides the lowers wide-open ('ah' / 'oo' show tongue + red lip, NOT a
            wide = smootherstep((op - 0.55) / 0.35)                # row), keep the tongue partly over the crowns (exposing only the tips);
            amt = max(amt, min(1.0, self.lower_p['occlude']) * 0.55 * wide * (1.0 - self._lower_show_at(i)))   # a real smile row still flashes
        return float(np.clip(amt, 0.0, 1.0))

    def tongue_pose(self, i, open_=None):
        """v8 wedge tongue in the lower-arch frame (IOD units along u, d, b from the lower incisal edge): [(centre, semi-
        axes)] for the tip (behind the lower incisors) and the dorsum (further back, higher), after the vowel pose and the
        near-closed constraint: on a near-closed frame nothing of it sits above the lower incisal edge (h < margin); as
        the mouth opens the dorsum may rise above the edge by up to 0.06 IOD (it shows over the lower teeth). v15
        (--warp-tongue-cover): at mid-open (_cover_amt) the tongue body rises and its front face grows DOWN over the lower
        crowns so it occludes them; on the wide-open teeth-baring frames cover -> 0 and the lowers flash as before."""
        op = float(self._open if open_ is None else open_)
        rise, tip, retr = (self.tg[i] if (self.prepared and self.tg is not None) else (0.0, 0.0, 0.0))
        closed = 1 - smootherstep((op - 0.15) / 0.3)
        cov = self.tongue_cover * self._cover_amt(i, op) if getattr(self, 'tongue_cover', 0.0) > 0 else 0.0
        blobs = [(np.array([0.0, 0.04 - tip - 0.015 * op, 0.09 + retr]), np.array([0.12, 0.03, 0.07])),       # tip
                 (np.array([0.0, 0.035 - rise - 0.03 * op, 0.25 + retr]), np.array([0.18, 0.06, 0.16]))]      # dorsum
        out = []
        for k, (c, ax) in enumerate(blobs):
            if cov > 0:                           # v15: grow the front face down over the crowns and lift the body to the edge
                c = c.copy()
                ax = ax.copy()
                if k == 0:
                    ax[1] = ax[1] + 0.030 * cov
                    c[1] = c[1] - 0.010 * cov      # the tip reaches down over the lower crown
                else: c[1] = c[1] - 0.020 * cov                                        # the dorsum lifts behind it
            allow = 0.006 * closed + (-0.035 if k == 1 else -0.01) * (1 - closed)      # the highest point allowed (h, + = down)
            allow = allow - 0.050 * cov                                               # v15: the tongue may rise further over the edge
            top = c[1] - ax[1]
            if top < allow:
                c = c.copy()
                c[1] = allow + ax[1]
            out.append((c, ax))
        return out

    def tongue_wedge(self, i, R, box, io_c, kick=0.0):
        """L3 tongue, v8: the modelled tongue (tongue_pose) rendered as two soft ellipsoids (tip in front of the dorsum),
        a wet highlight; colour from the take's own tongue pixels."""
        t = self.t
        o = self.lo.o
        f = self.proj(i, True, R)
        col = np.clip(self.tongue_c * float(t.light[i]) / max(1e-3, t.skin_med), 0, 255)
        acc_rgb = np.zeros((box[3], box[2], 3), np.float32)
        acc_a = np.zeros((box[3], box[2]), np.float32)
        for k, (c, ax) in enumerate(reversed(self.tongue_pose(i))):             # dorsum first, the tip over it
            Cc = o + t.u * (c[0] * io_c) + t.d * (c[1] * io_c) + t.b * (c[2] * io_c)
            A = np.stack([t.u * ax[0] * io_c, t.d * ax[1] * io_c, t.b * ax[2] * io_c], 1)
            c2, _ = f(Cc[None])
            c2 = c2 + kick
            Lm = (t.pl.s[i] * t.pl.R[i] @ R @ A)[:2]
            rgb, a = ellipsoid_layer(c2[0] - [box[0], box[1]], Lm, (0, 0, box[2], box[3]), col * self.tongue_lit * (0.8 if k == 0 else 1.0),
                                     spec=0.12 if k == 0 else 0.08, groove=0.06 if k == 1 else 0.1, ambient=0.3, diffuse=0.5)
            acc_rgb = acc_rgb * (1 - a[..., None]) + rgb * a[..., None]
            acc_a = acc_a + a * (1 - acc_a)
        rgb = acc_rgb / np.maximum(acc_a, 1e-4)[..., None]
        return np.clip(rgb, 0, 255).astype(np.float32), acc_a

    def layers(self, warped, i, P_take, P_out, gap_t):
        """all seven layers of frame i: dict name -> (BGR float box, alpha box), the box (x0, y0, w, h)."""
        t = self.t
        H, W = warped.shape[:2]
        io = t.iod * t.pl.s[i]
        io_c = t.iod
        box = self._box(i, P_take, P_out, W, H)
        x0, y0, w, h = box
        sl = np.s_[y0:y0 + h, x0:x0 + w]
        take = self.frames[i]
        Pt = np.asarray(P_take, float)[:, :2] - [x0, y0]
        Po = np.asarray(P_out, float)[:, :2] - [x0, y0]
        Q = Po[LIP_IN]
        Qo = Po[LIP_OUT]
        L = {}
        L['L0'] = (take[sl].astype(np.float32), np.ones((h, w), np.float32))
        # v11 (--warp-cover-corners): how far the plan pushes each mouth corner past the take's own (+ the opening, + yaw)
        cg = self.corner_growth(i, Pt, Po, io) if self.cover_corners > 0 else None
        fz = self.face_of(i, box) if self.face_of is not None else None     # v12 (--warp-yaw-clamp): the take's skin / lips
        cm = self.corner_masks(take[sl], Pt, Po, io, cg, face=fz) if cg is not None and self.lip_ab is not None else None
        # L1: the take's own mouth (inner + outer lip region) erased to its skin
        if self.cover_on:
            bald, m1 = ll.erase_lips(take[sl], Pt, getattr(self, 'cover_grow', None) or 0.22, lip_ab=self.lip_ab if self.cover_mode == 'fill' else None,
                                     skin_ab=self.skin_ab, corner=cg)
            a1 = cv2.GaussianBlur(m1.astype(np.float32), (0, 0), max(1.0, 0.01 * io))
            L['L1'] = (bald.astype(np.float32), np.maximum(a1, m1.astype(np.float32)))
            if self.lip_ab is not None and m1.sum() > 0:        # the cover alone must read as plain skin
                lp1 = tl.lipness_map(bald, self.lip_ab, self.skin_ab)
                self.cover_lip.append((i, float((lp1[m1] > 0.5).mean())))
        else: L['L1'] = (take[sl].astype(np.float32), np.zeros((h, w), np.float32))
        # L2: the cavity
        self._open = float(smootherstep((gap_t / max(1e-6, t.g_hi) - 0.4) / 0.6)) if not self.prepared else float(self.open_s[i])
        cav, a2 = self.cavity(i, Q, box, gap_t, io)
        enc = None
        clip_frac = None
        v12b = getattr(self, 'build', 'v11') == 'v12'
        band_log = v12b or getattr(self, 'band_log', False)
        rims = self.rim_colours(warped[sl], Q, Qo, io) if band_log else None      # v12: the planned lips' inner rim colour
        if self.interior_clip and self.lip_ab is not None:
            # v11: the interior is seen only THROUGH the warped lips (enclosed); elsewhere the warped face shows (L6)
            enc = self.enclosed(warped[sl], Q, Qo, io)
            a0_ = float(a2.sum())
            a2 = a2 * enc
            clip_frac = round(1.0 - float(a2.sum()) / a0_, 4) if a0_ > 1e-3 else 0.0
        L['L2'] = (cav, a2)
        clip = a2
        # L3: tongue + lower arch (skull + jaw + pins); L4: upper arch (skull + pin)
        R = self.jaw_R(i, gap_t, P_out[14])
        rot_lo = t.pl.R[i] @ R
        rot_up = t.pl.R[i]
        # the exposure limit, per frame AFTER the chin pin: the lower incisal edge never shows more than `expose` IOD above
        # the target lower lip (the planned jaw honours it; the chin skin's own fast motion can push past it)
        over, _, dn = self.exposure_over(i, R, P_out[14])
        self.clamp = max(0.0, over)
        kick = self.clamp * dn
        if getattr(self, 'single', None) is not None:      # v12 single: no per-frame kick (the jaw plan's floors hold the line)
            self.single['over'][int(i)] = round(self.clamp / max(1e-6, io), 4)
            self.clamp = 0.0
            kick = np.zeros(2)
        self.clamps.append(self.clamp)
        f0 = self.proj(i, True, R)
        def f_lo(X):
            xy, z = f0(X)
            return xy + kick, z
        if band_log or self.lower_real > 0: lo_rgb, lo_a, lo_h = self.lo.render(f_lo, box, rot_lo, facing=self.facing, hmap=True)
        else:
            lo_rgb, lo_a = self.lo.render(f_lo, box, rot_lo, facing=self.facing)
            lo_h = None
        k = float(t.light[i]) / max(1e-3, self.light_lo)
        lo_rgb = lo_rgb * k
        if self.lower_real > 0:          # v16.1: the per-frame lower shadow (atlas is v15-bright), gated by the take's own display
            lo_rgb = self._lower_depth(lo_rgb, lo_a, i, box[3], box[2], io)       # dim below the uppers + cast where the take hides them
            lo_rgb, lo_a = self._lower_shadow(lo_rgb, lo_a, lo_h, i, io)          # incisal darken + feathered top alpha + cool-shift
        if v12b:
            lo_rgb = self.lip_band(lo_rgb, lo_h, self.lo, False, rims['lo'])
            gw_lo = self._gw   # v12: the lower lip's colour
        cf = self.corridor_fade(Q, h, w) if self.corridor > 0 else None          # v13: the buccal corridor of this frame's opening
        if cf is not None:                            # (the lower arch only, not the tongue; v12's lip band is the lips' own inner
            lo_rgb = self.corridor_dim(lo_rgb, cav, cf if not v12b or gw_lo is None else cf * (1 - gw_lo))   # surface: kept)
        if self.tongue_on:
            tg, ta = (self.tongue_wedge if self.tongue_mode == 'wedge' else self.tongue)(i, R, box, io_c, kick)
            a3 = lo_a + ta * (1 - lo_a)                 # the union alpha (same whichever way the two are ordered)
            # default (v8): the lower arch composites OVER the tongue (the tongue sits behind the teeth)
            rgb3 = (lo_rgb * lo_a[..., None] + tg * (ta * (1 - lo_a))[..., None]) / np.maximum(a3, 1e-4)[..., None]
            cov = (self.tongue_cover * self._cover_amt(i)) if self.tongue_mode == 'wedge' and getattr(self, 'tongue_cover', 0.0) > 0 else 0.0
            if cov > 0:        # v15: at mid-open the tongue rides IN FRONT of the lower crowns (covering them), eased by cov
                rgb_front = (tg * ta[..., None] + lo_rgb * (lo_a * (1 - ta))[..., None]) / np.maximum(a3, 1e-4)[..., None]
                rgb3 = (1 - cov) * rgb3 + cov * rgb_front
        else: rgb3, a3 = lo_rgb, lo_a
        L['L3'] = (rgb3, a3 * clip)
        if band_log: up_rgb, up_a, up_h = self.up.render(self.proj(i, False), box, rot_up, facing=self.facing, hmap=True)
        else: up_rgb, up_a = self.up.render(self.proj(i, False), box, rot_up, facing=self.facing)
        if v12b:                                      # v12: the band above the crown takes the upper lip's colour (no gum)
            L['L4'] = (self.lip_band(up_rgb * float(t.light[i]) / max(1e-3, self.light_up), up_h, self.up, True, rims['up']), up_a * clip)
            gw_up = self._gw
        else: L['L4'] = (up_rgb * float(t.light[i]) / max(1e-3, self.light_up), up_a * clip)
        if cf is not None:                            # v13: the upper arch recedes into the corridor too; logged per frame
            fu = cf if not v12b or gw_up is None else cf * (1 - gw_up)
            fl = cf if not v12b or gw_lo is None else cf * (1 - gw_lo)
            L['L4'] = (self.corridor_dim(L['L4'][0], cav, fu), L['L4'][1])
            cor = self.corridor_log(i, cf, box, io, up_a, lo_a, clip, fu, fl)
        if self.cavity_fill > 0:                      # v17: fill the opening (tongue + lower row) instead of a dark void
            cf_rgb, cf_a = self.fill_cavity(i, Q, h, w, io, L['L4'][0], L['L4'][1])
            L['L2'] = (L['L2'][0] * (1 - cf_a[..., None]) + cf_rgb * cf_a[..., None], L['L2'][1])
            om = tl.poly_mask(Q, (h, w), 0.0, 0.03 * io).astype(np.float32)
            L['L3'] = (L['L3'][0], L['L3'][1] * (1 - np.maximum(cf_a, om)))   # the arch lowers yield to the fill (corners too: no grey line)
        if band_log: self._band = dict(i=i, up_h=up_h, lo_h=lo_h, rims=rims)
        self._full = dict(L3=a3, L4=up_a)          # unclipped arches (the debug view shows the whole band)
        # contact shadows of the lips on L2..L4: below the upper lip (strongest), above the lower lip, the corners
        opening = tl.poly_mask(Q, (h, w), 0.0)
        rows = np.arange(h)[:, None]
        cb = opening.any(0)
        top = np.where(cb, opening.argmax(0), h)
        bot = np.where(cb, h - 1 - opening[::-1].argmax(0), -1)
        dt_ = (rows - top[None, :]).astype(np.float32)
        db_ = (bot[None, :] - rows).astype(np.float32)
        dd = cv2.distanceTransform(opening.astype(np.uint8), cv2.DIST_L2, 3)
        sh = (1 - self.shadow[0] * np.exp(-np.maximum(dt_, 0) / max(1.0, 0.018 * io))) \
            * (1 - self.shadow[1] * np.exp(-np.maximum(db_, 0) / max(1.0, 0.012 * io))) * (1 - self.shadow[2] * np.exp(-dd / max(1.0, 0.012 * io)))
        sh = np.where(opening, sh, 1.0).astype(np.float32)[..., None]     # (outside: only seen where a stray lip pixel is see-through)
        if self.cavity_fill > 0:
            # v17c: the fill's tongue rests ON the lower lip / into the corners -- no gap for a contact shadow there. The
            # lower-lip + corner terms darkened the pink floor into a grey-mauve 1-3 px line along the bottom / corners
            # (the grey outline); on the fill they ease out by its alpha (the upper-lip shadow under the teeth stays)
            s_up = 1 - self.shadow[0] * np.exp(-np.maximum(dt_, 0) / max(1.0, 0.018 * io))
            s_lo = sh[..., 0] / np.maximum(s_up, 1e-4)
            s_lo = s_lo + (1 - s_lo) * cf_a
            sh2 = np.where(opening, s_up * s_lo, 1.0).astype(np.float32)[..., None]
        else: sh2 = sh
        if v12b:            # v12: the lip band is the lips' own inner surface: no contact shadow of the lips on it
            sh4 = sh if gw_up is None else sh + (1 - sh) * gw_up[..., None]
            sh3 = sh if gw_lo is None else sh + (1 - sh) * gw_lo[..., None]
            L['L2'] = (L['L2'][0] * sh2, L['L2'][1])
            L['L3'] = (L['L3'][0] * sh3, L['L3'][1])
            L['L4'] = (L['L4'][0] * sh4, L['L4'][1])
        else:
            L['L2'] = (L['L2'][0] * sh2, L['L2'][1])
            for key in ('L3', 'L4'): L[key] = (L[key][0] * sh, L[key][1])
        # L5: the warped lips only, outer lip .. target inner contour; a darker inner rim; non-lip pixels at the inner
        # edge (a stretched take tooth / cavity the warp left) are see-through, deeper ones filled from the lip around
        reg = warped[sl].astype(np.float32)
        lab = cv2.cvtColor(warped[sl], cv2.COLOR_BGR2LAB)
        outer = tl.poly_mask(Qo, (h, w), 0.0, 0.02 * io)
        inner_aa = cv2.GaussianBlur(opening.astype(np.float32), (0, 0), 0.6 if self.wet <= 0 else max(0.6, 0.5 + self.wet))
        a5 = outer.astype(np.float32) * (1 - inner_aa)
        rgb5 = reg.copy()
        lp = None
        if self.lip_ab is not None:
            lp = tl.lipness_map(warped[sl], self.lip_ab, self.skin_ab)
            chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
            # not lip: a tooth (bright, low a*), the cavity (dark) or a stretched grey tooth / shadow (no chroma: lips and
            # skin always carry some) the warp squeezed into the lip band
            nonlip = (lp < 0.35) & (tl.teeth_key(lab, t.skin_med) | (lab[..., 0] < 0.5 * t.skin_med) | (chroma < 10))
            band = tl.poly_mask(Q, (h, w), 0.0, 0.03 * io) & ~opening
            see = cv2.GaussianBlur((nonlip & band).astype(np.float32), (0, 0), 0.8)
            a5 = a5 * (1 - see)
            deep = nonlip & outer & ~tl.poly_mask(Q, (h, w), 0.0, 0.03 * io)
            if deep.any():
                wk = (~nonlip).astype(np.float32)
                sg = max(2.0, 0.03 * io)
                fill = cv2.GaussianBlur(reg * wk[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(wk, (0, 0), sg), 1e-3)[..., None]
                df = cv2.GaussianBlur(deep.astype(np.float32), (0, 0), 1.0)[..., None]
                rgb5 = rgb5 * (1 - df) + fill * df
        dout = cv2.distanceTransform((~opening).astype(np.uint8), cv2.DIST_L2, 3)
        if self.cavity_fill > 0:
            # v17c: L5's inner anti-alias edge (a5 ~0.2-0.6 on the last ~2 px) carried the WARPED frame's own pixels there -- the
            # take's grey teeth / dark cavity squeezed to the lip line -- which over the pink fill read as a grey outline (bottom
            # + corners). Those edge pixels take the lip colour from just outside them (normalised blur of the lip ring)
            src = ((dout >= 2.5) & outer).astype(np.float32)
            sg = max(1.5, 0.015 * io)
            lipc = cv2.GaussianBlur(rgb5 * src[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(src, (0, 0), sg), 1e-3)[..., None]
            ew = np.clip((3.0 - dout) / 1.5, 0, 1)[..., None]
            rgb5 = rgb5 * (1 - ew) + lipc * ew
        rim = np.exp(-dout / max(0.8, self.rim[1] * io)) * outer
        rgb5 = rgb5 * (1 - self.rim[0] * rim)[..., None]
        if self.wet > 0:        # v8: the lip's wet inner edge -- a faint 1-2 px lift where it turns over the teeth, only on
            wl = np.exp(-((dout - 1.3) / 0.9) ** 2) * outer                        # lip pixels and scaled by their own light
            if lp is not None: wl = wl * np.clip(lp * 1.5, 0, 1)
            wl = wl * np.clip(lab[..., 0].astype(np.float32) / 255.0, 0, 1)
            rgb5 = rgb5 + (0.25 * self.wet * wl)[..., None] * (255.0 - rgb5) * 0.6
        nf5 = None
        if cm is not None:
            # v11 corner face guard: where the planned lip ring reaches past the take's face at a corner (hair / background
            # in the take, and not lip in the warped frame), no lip is drawn: the warped frame (L6) shows there
            nf5 = cm['region'] & outer & ~opening & ~cm['face_t'] & (lp < 0.5)
            nf5 = cv2.GaussianBlur(nf5.astype(np.float32), (0, 0), 0.8)
            a5 = a5 * (1 - nf5)
        L['L5'] = (np.clip(rgb5, 0, 255), a5)
        self.interior[i] = (box, (opening & (a5 < 0.5)))         # the rendered interior (protected from later lip passes)
        # L6: the warped face around the lips; old-lip pixels outside the new outer lip are dropped (L1 shows there).
        # Hard complement of L5's outer edge, one px inside it: both layers are the same warped pixels there, so the
        # seam is exact (a soft complement would let L1 leak through the "over" of two half-transparent layers)
        a6 = 1 - cv2.erode(outer.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        reg6 = reg
        if self.lip_ab is not None and self.cover_on:
            # an old-lip pixel: clearly lip-coloured, outside the new outer lip, inside the take's OWN lip region (where
            # an old lip can come from), in a blob (not a speck of rosy skin)
            ring = tl.poly_mask(Qo, (h, w), 0.0, 0.12 * io) & ~tl.poly_mask(Qo, (h, w), 0.0, 0.03 * io) & tl.poly_mask(Pt[LIP_OUT], (h, w), 0.12)
            thr = 0.75 if self.cover_mode != 'fill' else 0.5
            st = (ring & (lp > thr)).astype(np.uint8)
            nl, lbl, cc, _ = cv2.connectedComponentsWithStats(st)
            big = np.isin(lbl, [k for k in range(1, nl) if cc[k, cv2.CC_STAT_AREA] >= (0.03 * io) ** 2])
            if fz is not None: big &= fz            # v12: old lip only on the face (auburn hair reads lip-coloured)
            if cg is not None:
                # v11: at the mouth corners, as far as the plan pushed them (corner_growth), old-mouth material the warp
                # MOVED onto what was skin in the take -- lip-coloured, cavity-dark or tooth in the warped frame where the
                # take shows skin at the same pixel (hair / shadow dark in both is left alone) -- outside the new outer lip
                mv = self.corner_residue(warped[sl], outer, lp, cm, io) if cm is not None else np.zeros_like(big)
                big = big | mv
                self.corner_log[i] = int(mv.sum())
            big = cv2.dilate(big.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(0.01 * io)))
            if self.cover_mode == 'fill':
                # v8: the blob is filled with skin in the WARPED frame itself (membrane + grain from around it): L6 stays
                # opaque, nothing of L1 shows, no texture seam between the bald cover and the warped face
                if big.any():
                    hole = cv2.dilate(big, np.ones((3, 3), np.uint8), iterations=max(1, int(0.015 * io))) > 0
                    hole &= ~cv2.erode(outer.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    filled = fill_blob(warped[sl], hole, seed=7 + i % 3).astype(np.float32)
                    sf = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), max(1.0, 0.01 * io))[..., None]
                    sf = np.maximum(sf, hole.astype(np.float32)[..., None])
                    reg6 = reg * (1 - sf) + filled * sf
            else:
                stray = np.clip(cv2.GaussianBlur(big.astype(np.float32), (0, 0), max(1.0, 0.02 * io)) * 1.5, 0, 1) * L['L1'][1]
                a6 = a6 * (1 - stray)
        if enc is not None:         # v11: an opening the warped lips do not enclose is no mouth: the warped face shows there
            a6 = np.maximum(a6, inner_aa * (1.0 - enc))
        if nf5 is not None: a6 = np.maximum(a6, nf5)       # v11: the corner face guard (above)
        L['L6'] = (reg6, a6.astype(np.float32))
        self.reg[i] = self.reg_entry(i, box, np.asarray(P_out, float), gap_t, R, kick, a2, clip_frac)
        if cf is not None: self.reg[i]['corridor'] = cor                    # v13
        if band_log: self._band.update(Q=Q, io=io, a4=L['L4'][1], a3=L['L3'][1])
        return L, box

    def fill_cavity(self, i, Q, h, w, io, up_rgb, up_a):
        """v17 (--warp-cavity fill): what a real open mouth shows under the upper teeth instead of a dark void, painted in the
        2D opening (Q = target inner contour, box px) so it always sits INSIDE the lips whatever the 3D arches did:
          tongue  a shaded dark-pink dome rising from the floor of the opening (higher in the centre, lower at the corners),
                  the take's own tongue colour, a soft dorsum highlight + a faint median groove; grows in with the opening;
          lowers  a dim row of lower incisors along the floor, in FRONT of the tongue, bottoms tucked under the lower lip,
                  colour from this frame's upper crowns; only past ~35 % open (hidden on near-closed frames);
          back    the cavity's own dark keeps the top (under the upper teeth) and the corners.
        -> (BGR float, alpha) in the box; alpha already feathered and x the fill strength."""
        t = self.t
        k = self.cavity_fill
        g = float(t.light[i]) / max(1e-3, t.skin_med)
        opening = aa_poly(Q, (h, w), 0.03 * io) > 0.5              # grown as L2's own alpha (the 3D lower arch must not peek past it)
        z = np.zeros((h, w), np.float32)
        if opening.sum() < 12: return np.zeros((h, w, 3), np.float32), z
        cols = np.nonzero(opening.any(0))[0]
        xl, xr = float(cols[0]), float(cols[-1])
        xc = 0.5 * (xl + xr)
        hw = max(2.0, 0.5 * (xr - xl))
        top = np.where(opening.any(0), opening.argmax(0), h).astype(np.float32)
        bot = np.where(opening.any(0), h - 1 - opening[::-1].argmax(0), -1).astype(np.float32)
        hc = np.maximum(bot - top, 0.0)
        hmax = max(0.0, float(hc.max()) - 0.06 * io)    # (minus the growth)
        # openness from THIS frame's opening height in IOD (absolute, not the take's range: a take that opens wide would
        # otherwise read our moderate openings as closed): 0 at <= 0.05 IOD, 1 at >= 0.17 IOD
        op = float(smootherstep((hmax / max(1.0, io) - 0.05) / 0.12))
        pv = getattr(self, '_cf_op', None)                                     # temporal calm: no frame-to-frame pop of the fill
        if pv is not None and i == pv[0] + 1: op = 0.5 * op + 0.5 * pv[1]
        self._cf_op = (i, op)
        # the talker's own lower-teeth habit (--warp-lower-ratio, auto = measured on the take: ~0.2 top-teeth .. ~0.8
        # bottom-teeth talker): a top-teeth talker barely shows the lower row, a bottom-teeth talker shows it early + taller
        lr = getattr(self, 'lower_ratio', None)
        lr = 0.5 if lr is None else lr
        talk = float(np.clip((lr - 0.2) / 0.6, 0.0, 1.0))                    # 0 top-teeth .. 1 bottom-teeth
        sm = lambda v: cv2.GaussianBlur(v.reshape(1, -1).astype(np.float32), (0, 0), max(1.0, 0.03 * io)).ravel()
        bot, hc = sm(bot), sm(hc)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        vx = (xx - xc) / hw
        dome = np.sqrt(np.clip(1.0 - (vx[0] / 0.92) ** 2, 0.0, 1.0))
        fe = max(0.8, 0.012 * io)                                              # edge feather, px
        rgb = np.zeros((h, w, 3), np.float32)
        a = z.copy()
        # tongue: its dorsum line per column, a share of that column's opening (never above ~half of it)
        tw = float(smootherstep(op / 0.35))
        if tw > 0:
            # v17b: the vowel's tongue pose (prepare(): eased dorsum rise, tip rise, retraction, IOD) shapes the fill --
            # E / ih ride high and forward, DD / nn / TH / RR lift the tip toward the teeth, aa / oh / ou sit low and back
            tgi = self.tg[min(i, len(self.tg) - 1)] if getattr(self, 'tg', None) is not None and len(self.tg) else (0.0, 0.0, 0.0)
            rise = float(np.clip(tgi[0] / 0.045, -1, 1))
            tip = float(np.clip(tgi[1] / 0.02, 0, 1))
            ret = float(np.clip(tgi[2] / 0.035, 0, 1))
            share = (0.42 + 0.14 * op + 0.16 * rise - 0.10 * ret) * dome ** (0.8 + 0.6 * ret)
            share = share + 0.22 * tip * np.exp(-(vx[0] / 0.28) ** 2)          # the tip, centred, up behind the incisors
            tk = float(getattr(self, 'tongue_fill', 0.45))                    # v17d: tongue presence (1 = v17c; founder: 'too present, a lisp')
            share = np.clip(share * tk, 0.0, 0.72 * tk) * np.clip(dome, 0, 1) ** 0.6   # v17e: low at the corners (a full-width band read as a 2nd lower lip)
            yt = bot - share * hc                                              # the dorsum line (per column)
            d = yy - yt[None, :]                                               # >0 below the dorsum
            ta = np.clip(d / fe + 0.5, 0, 1) * np.clip((dome[None, :] - 0.25) / 0.35, 0, 1) ** 1.5   # v17c: fades into the dark corners (a hard cut left a thin grey tongue sliver along the corner's floor)
            base = 0.3 * np.asarray(self.tongue_c, np.float32) + 0.7 * np.array([58.0, 64.0, 128.0], np.float32)   # v17e: toward a warm tongue red (the +blue lift read lilac, a 2nd lower lip)
            base = np.clip(base * g * (0.70 + 0.30 * tk), 0, 255)               # less present = deeper in the shadow
            th = np.maximum(share * hc, 1.0)[None, :]
            depth = np.clip(d / th, 0, 1)                                      # 0 at the dorsum .. 1 at the floor
            lit = (1.0 - 0.30 * depth - 0.40 * (1 - dome[None, :]) ** 1.5) * (1.0 - 0.28 * ret + 0.08 * max(rise, 0.0))   # back = in shadow            # lit on top, darker down and to the corners
            hl = 0.7 * tk * np.exp(-((vx / 0.30) ** 2) - ((d - 0.18 * th) / np.maximum(0.12 * th, 1.0)) ** 2)    # soft wet dorsum light
            gro = 1 - 0.08 * np.exp(-(vx * hw / max(1.5, 0.025 * io)) ** 2) * np.clip(1.0 - depth / 0.6, 0, 1)        # the median groove
            trgb = base[None, None, :] * (lit * gro)[..., None] + (0.22 * hl)[..., None] * (np.array([150.0, 160.0, 215.0]) - base)[None, None, :]
            rgb = trgb
            a = ta * tw
        # lower incisors: a dim row along the floor, in front of the tongue
        lw = float(smootherstep((op - (0.55 - 0.30 * talk)) / 0.35)) * (0.10 + 1.10 * talk) * (1.0 - 0.45 * float(smootherstep((hmax / max(1.0, io) - 0.20) / 0.10)))   # very wide: tongue over them
        if lw > 0:
            m = up_a > 0.6
            if m.sum() > 20:                                                   # the upper crowns' lit enamel (not their seams / grey)
                px = up_rgb[m].astype(np.float32)
                lum = px.mean(1)
                tc = px[lum >= np.percentile(lum, 70)].mean(0)
            else: tc = np.array([200.0, 205.0, 215.0]) * g
            tc = tc.mean() * np.array([0.90, 0.98, 1.06]) * 0.4 + 0.6 * np.array([200.0, 220.0, 238.0]) * g   # neutralised (the uppers read grey-green) + warm ivory   # warm ivory (the lower lip's contact shadow dims it)
            lh = np.minimum((0.032 + 0.020 * talk) * io, 0.20 * hc) * min(1.0, lw) * np.clip(1.0 - (vx[0] / 0.62) ** 2, 0, 1) ** 0.6   # row height per column
            yl = bot - 0.03 * io - lh                                        # (above the grown edge: the lip covers that)
            d = yy - yl[None, :]
            la = np.clip(d / fe + 0.5, 0, 1) * (lh[None, :] > 0.6)
            p = max(3.0, 0.062 * io)                                           # incisor pitch
            ph = (xx - xc) / p
            seam = np.exp(-((ph - np.round(ph)) / 0.09) ** 2)
            ld = np.clip(d / np.maximum(lh[None, :], 1.0), 0, 1)
            shade = (0.74 + 0.20 * np.clip(1 - np.abs(vx) / 0.8, 0, 1)) * (1 - 0.22 * seam) * (1.0 - 0.20 * ld)   # brightest at the tips
            lrgb = tc[None, None, :] * shade[..., None]
            lw = min(1.0, lw)
            la = la * 0.85 * lw
            # premultiplied "over": the row's COLOUR weighs by its own alpha (la x lw), not la alone -- a near-off row (top-teeth
            # talker, lw ~0.05) used to paint a thin grey-ivory band at full colour weight along the floor (the grey outline)
            an = la + a * (1 - la)
            rgb = (lrgb * la[..., None] + rgb * (a * (1 - la))[..., None]) / np.maximum(an, 1e-4)[..., None]
            a = an
        a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.6) * opening * k
        return np.clip(rgb, 0, 255).astype(np.float32), a.astype(np.float32)

    def corner_growth(self, i, Pt, Po, io, yaw_k=0.15):
        """v11: per mouth corner (left, right), px, x cover_corners: how far the plan moves the outer corner (61 / 291)
        outward past the take's own along the corner line, + half the planned opening increase (13-14), + yaw_k IOD x
        |sin yaw| (the far corner -- the one closer to the nose in the picture -- in full, the near one a third)."""
        e = Po[291] - Po[61]
        L = float(np.linalg.norm(e))
        if L < 1e-6: return (0.0, 0.0)
        e = e / L
        dl = max(0.0, float(-(Po[61] - Pt[61]) @ e))
        dr = max(0.0, float((Po[291] - Pt[291]) @ e))
        op = max(0.0, float(np.linalg.norm(Po[13] - Po[14]) - np.linalg.norm(Pt[13] - Pt[14])))
        R = np.asarray(self.t.pl.R[i], float)
        sy = abs(float(R[2, 0])) if R.shape == (3, 3) else 0.0      # |sin yaw|
        nose = Pt[1]
        far_left = abs(float((Pt[61] - nose) @ e)) < abs(float((Pt[291] - nose) @ e))
        yl, yr = (1.0, 1 / 3) if far_left else (1 / 3, 1.0)
        k = self.cover_corners
        return (k * (dl + 0.5 * op + yaw_k * io * sy * yl), k * (dr + 0.5 * op + yaw_k * io * sy * yr))

    def corner_masks(self, take, Pt, Po, io, cg, r0=0.06, face=None):
        """v11: the two corner regions (box): discs of r0 IOD + the corner's growth around the NEW outer corners (61 / 291);
        and face_t: where the TAKE shows face at a pixel -- skin (mouth_morph.skin_px), lip (lipness >= 0.35) or its own
        mouth (its outer-lip hull grown 0.12) -- vs hair / background (never painted over, never drawn as lip). v12 face
        (a bool box, --warp-yaw-clamp on a frame it clamps, mouth_warp.YawClamp.face_box): the take's own skin / lips instead
        (on mv30 NC the auburn hair passes lipness >= 0.35, so v11's face_t let the residue fill paint skin over the hair)."""
        from . import mouth_morph as mm
        h, w = take.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        reg_ = np.zeros((h, w), bool)
        for p, g in ((Po[61], cg[0]), (Po[291], cg[1])):
            r = r0 * io + g
            reg_ |= (xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= r * r
        lp_t = tl.lipness_map(take, self.lip_ab, self.skin_ab)
        face_t = mm.skin_px(take) | (lp_t >= 0.35) | tl.poly_mask(np.asarray(Pt, float)[LIP_OUT], (h, w), 0.12) if face is None else face
        return dict(region=reg_, face_t=face_t, lp_t=lp_t)

    def corner_residue(self, img, outer, lp, cm, io, min_area=0.015):
        """v11: bool mask (box) of the old-mouth residue at the corners (see layers()): in the corner regions, outside the
        new outer lip, old-mouth-like in the warped frame (lipness > 0.5, L < 0.45 skin, or tooth) on a pixel where the
        take shows face (corner_masks' face_t: the take's own lips left in place, or old-mouth material the warp moved
        onto its skin; hair / background are left alone); blobs of at least (min_area IOD)^2."""
        t = self.t
        lab_w = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        old_w = (lp > 0.5) | (lab_w[..., 0] < 0.45 * t.skin_med) | tl.teeth_key(lab_w, t.skin_med)
        mv = (cm['region'] & ~outer & old_w & cm['face_t']).astype(np.uint8)
        nl, lbl, cc, _ = cv2.connectedComponentsWithStats(mv)
        return np.isin(lbl, [k for k in range(1, nl) if cc[k, cv2.CC_STAT_AREA] >= max(4.0, (min_area * io) ** 2)])

    def enclosed(self, img, Q, Qo, io, lip_thr=0.35, grow=0.15, side=0.04):
        """v11: float mask over the box (img = the warped frame's box, BGR): the part of the planned opening (inner contour Q,
        box px) that the warped lips enclose -- for each pixel, lip-coloured pixels of the frame (lipness vs the shot's own
        lip / skin colour > lip_thr, L5's own non-lip line) both above AND below it along the mouth's vertical axis
        (perpendicular to the inner corners' line), searched within the planned outer lip grown by `grow` IOD and over
        +-`side` IOD along the corner line (the corners' last pixels); feathered <= 1 px (Gaussian sigma 0.5)."""
        h, w = img.shape[:2]
        Q = np.asarray(Q, float)
        Qo = np.asarray(Qo, float)
        lip = (tl.lipness_map(img, self.lip_ab, self.skin_ab) > lip_thr) & tl.poly_mask(Qo, (h, w), 0.0, grow * io)
        cl, cr = Q[LIP_IN.index(78)], Q[LIP_IN.index(308)]
        e = cr - cl
        L_ = float(np.linalg.norm(e))
        e = e / L_ if L_ > 1e-6 else np.array([1.0, 0.0])
        nrm = np.array([-e[1], e[0]])          # nrm: down the mouth
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        c = 0.5 * (cl + cr)
        sa = (xx - c[0]) * e[0] + (yy - c[1]) * e[1]
        va = (xx - c[0]) * nrm[0] + (yy - c[1]) * nrm[1]
        if not lip.any(): return np.zeros((h, w), np.float32)
        b = np.round(sa).astype(np.int64)
        b0 = int(b.min())
        nb = int(b.max()) - b0 + 1
        bi = b - b0
        vmin = np.full(nb, np.inf)
        vmax = np.full(nb, -np.inf)
        np.minimum.at(vmin, bi[lip], va[lip])
        np.maximum.at(vmax, bi[lip], va[lip])
        k = max(1, int(round(side * io)))
        from scipy.ndimage import minimum_filter1d, maximum_filter1d
        big = 1e9
        vmin = minimum_filter1d(np.where(np.isfinite(vmin), vmin, big), 2 * k + 1, mode='nearest')
        vmax = maximum_filter1d(np.where(np.isfinite(vmax), vmax, -big), 2 * k + 1, mode='nearest')
        ok = (va > vmin[bi]) & (va < vmax[bi])
        return cv2.GaussianBlur(ok.astype(np.float32), (0, 0), 0.5)

    def reg_entry(self, i, box, P_out, gap_t, R, kick, a2, clip_frac=None):
        """v11: frame i's inter-layer registration (see the module doc, REG), every layer's anchor where it was rendered,
        mapped into the skull-stabilised frame: canonical (x, y) on the upper-teeth plane through frame i's skull pose
        only (no pin), origin the upper arch's reference point, IOD units."""
        t = self.t
        io_c = t.iod
        A = tl.plane_affine(t.pl, i, self.z_up, ref=t.ref)
        Ai = tl.inv_aff(A)
        org = np.asarray(self.ref[:2], float)
        st = lambda xy: (np.asarray(xy, float).reshape(-1, 2) @ Ai[:, :2].T + Ai[:, 2] - org) / io_c
        rd = lambda v, k=4: np.round(np.asarray(v, float), k).tolist()
        aa = np.array([-REG_END * io_c, 0.0, REG_END * io_c])
        hh = np.zeros(3)
        up = st(self.proj(i, False)(self.up.points(aa, hh))[0])
        f0 = self.proj(i, True, R)
        lo = st(f0(self.lo.points(aa, hh))[0] + kick)
        x0, y0 = box[0], box[1]
        tot = float(a2.sum())
        cav = None
        if tot > 1e-3:
            yy, xx = np.mgrid[0:a2.shape[0], 0:a2.shape[1]]
            cav = rd(st([float((xx * a2).sum()) / tot + x0, float((yy * a2).sum()) / tot + y0])[0])
        inner = st(P_out[LIP_IN, :2])
        m = cv2.moments(P_out[LIP_IN, :2].astype(np.float32))
        oc = rd(st([m['m10'] / m['m00'], m['m01'] / m['m00']])[0]) if abs(m['m00']) > 1e-6 else rd(inner.mean(0))
        tip = None
        if self.tongue_on:
            o = self.lo.o
            if self.tongue_mode == 'wedge':
                cc, _ = self.tongue_pose(i)[0]
                Cc = o + t.u * (cc[0] * io_c) + t.d * (cc[1] * io_c) + t.b * (cc[2] * io_c)
            else:
                Cc = o + t.b * (0.12 * io_c) + t.d * ((self.tongue_y - 0.025 * self._open) * io_c)
            tip = rd(st(f0(Cc[None])[0] + kick)[0])
        from scipy.spatial.transform import Rotation
        ypr = Rotation.from_matrix(np.asarray(t.pl.R[i], float)).as_euler('yxz', degrees=True)      # yaw, pitch, roll
        p14 = P_out[14]
        th = t.jaw_at(i, gap_t, p14) + float(self.dth[i])
        pin = np.asarray(t.pin[i], float)
        return dict(up=rd(up), lo=rd(lo), cav=cav, open=oc, inner=rd(inner[[LIP_IN.index(13), LIP_IN.index(14), LIP_IN.index(78), LIP_IN.index(308)]]),
                    contour=rd(inner, 4), tip=tip, jaw=round(float(np.degrees(th)), 3), dth=round(float(np.degrees(self.dth[i])), 3),
                    pin_up=rd(pin, 2), pin_up_iod=rd((Ai[:, :2] @ pin) / io_c), chin=rd(self.pin_disp[i] if i < len(self.pin_disp) else np.zeros(2), 2),
                    kick=rd(kick, 2), pose=rd(ypr, 2), stab=rd(A, 5), clip=clip_frac)

    def composite(self, warped, i, P_take, P_out, gap_t):
        """frame i through the stack -> (image, changed mask float 0..1)."""
        H, W = warped.shape[:2]
        L, box = self.layers(warped, i, P_take, P_out, gap_t)
        x0, y0, w, h = box
        if w < 4 or h < 4: return warped, np.zeros((H, W), np.float32)
        acc = L['L0'][0].copy()
        for k in ('L1', 'L2', 'L3', 'L4', 'L5', 'L6'):
            rgb, a = L[k]
            acc = acc * (1 - a[..., None]) + rgb * a[..., None]
        out = warped.copy()
        new = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
        if getattr(self, '_band', None) is not None and self._band.get('i') == i and i in self.reg:
            self.reg[i]['lipband'] = self.lipband_ruler(new, self._band)      # v12: the plate's band under the lips vs their rim
        diff = np.abs(new.astype(np.int16) - warped[y0:y0 + h, x0:x0 + w].astype(np.int16)).max(2) > 0
        out[y0:y0 + h, x0:x0 + w] = np.where(diff[..., None], new, warped[y0:y0 + h, x0:x0 + w])
        M = np.zeros((H, W), np.float32)
        M[y0:y0 + h, x0:x0 + w] = np.maximum(L['L2'][1], cv2.GaussianBlur(diff.astype(np.float32), (0, 0), 1.0))
        if self.debug and (self.debug_frames is None or i in self.debug_frames): self._keep(i, L, box, new)
        return out, M

    # ------------------------------------------------------------------------------------------------- debug --
    def _stab(self, i, img, box, cw=200, ch=150):
        """box image -> the head-stabilised mouth crop (canonical plane at the upper teeth, + the lip-root pin)."""
        t = self.t
        io = t.iod
        c = self.ref
        sx = cw / (1.0 * io)
        x0c, y0c = c[0] - 0.5 * io, c[1] - 0.33 * io
        pin = t.pin[i] if self.dbg_stab == 'pin' else np.zeros(2)        # v11 'skull': the skull pose only (the pin's motion shows)
        A = tl.shift_aff(tl.plane_affine(t.pl, i, self.z_up, ref=t.ref), pin - np.array(box[:2], float))
        M = tl.compose(A, np.array([[1 / sx, 0, x0c], [0, 1 / sx, y0c]]))
        return cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _keep(self, i, L, box, comp):
        h, w = box[3], box[2]
        yy, xx = np.mgrid[0:h, 0:w]
        chk = np.where(((yy // 6 + xx // 6) % 2)[..., None] > 0, 70.0, 110.0) * np.ones(3)
        pan = []
        for k in ('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6'):
            rgb, a = L[k]
            if k in self._full: a = np.maximum(a, 0.4 * self._full[k])     # the whole arch; the part the lips hide at 40 %
            pan.append(self._stab(i, np.clip(chk * (1 - a[..., None]) + rgb * a[..., None], 0, 255).astype(np.uint8), box))
        pan.append(self._stab(i, comp, box))
        self.dbg[i] = pan

    def write_debug(self, d, fps=25.0, strip_frames=None):
        """d/layers.mp4 (every kept frame: L0..L6 + composite, head-stabilised, labels in a bottom band) and
        d/layers.jpg (a strip of strip_frames, one row each)."""
        if not self.dbg: return
        os.makedirs(d, exist_ok=True)
        lab = list(NAMES) + ['composite']
        ks = sorted(self.dbg)
        def row(i):
            cells = []
            for p, nm in zip(self.dbg[i], lab):
                c = cv2.copyMakeBorder(p, 0, 22, 0, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                cv2.putText(c, nm, (4, p.shape[0] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                cells.append(c)
            r = np.hstack(cells)
            r = cv2.copyMakeBorder(r, 0, 18, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(r, f'frame {i}', (4, r.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            return r
        r0 = row(ks[0])
        Hh, Ww = r0.shape[:2]
        pr = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{Ww}x{Hh}', '-r', str(fps), '-i', '-',
                               '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                               os.path.join(d, 'layers.mp4')], stdin=ff.PIPE)
        for i in ks: pr.stdin.write(np.ascontiguousarray(row(i)).tobytes())
        pr.stdin.close()
        pr.wait()
        sf = [i for i in (strip_frames or []) if i in self.dbg] or ks[:: max(1, len(ks) // 4)][:4]
        cv2.imwrite(os.path.join(d, 'layers.jpg'), np.vstack([row(i) for i in sf]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    def bite_overlap(self, P_out, gaps, close_thr=0.2):
        """v8 check: on near-closed frames (target opening below close_thr of the take's p95), how far the lower incisal
        edge sits above the upper one along the head's down axis (IOD; the overbite it should be) -> summary."""
        t = self.t
        g = np.asarray(gaps, float)
        u_ = g / max(1e-6, t.g_hi)
        vals = []
        for i in range(self.n):
            if u_[i] >= close_thr: continue
            R = self.jaw_R(i, g[i], P_out[i][14])
            Ti = (self.lo.o - t.H) @ R.T + t.H
            vals.append(float((np.array([self.ref[0], self.e_up, self.z_up]) - Ti) @ t.d) / t.iod)
        if not vals: return dict(n=0)
        v = np.array(vals)
        return dict(n=len(v), med=round(float(np.median(v)), 4), p5=round(float(np.percentile(v, 5)), 4), p95=round(float(np.percentile(v, 95)), 4),
                                        within=round(float(((v >= -0.005) & (v <= self.overbite / t.iod + 0.01)).mean()), 3))

    def info(self):
        pin = np.linalg.norm(self.pin_disp, axis=1) if len(self.pin_disp) else np.zeros(1)
        cl = np.array(self.clamps) if self.clamps else np.zeros(1)
        return dict(h_up=int(self.t.h_up), h_lt=int(self.h_lt), lower_synth=bool(self.lo_synth), lower_shift_iod=round(self.lo_shift, 3), e_up=round(self.e_up, 1), e_lo=round(self.e_lo, 1),
                    arch_k=round(self.k * self.t.iod, 2), up_arc=round(self.up.A / self.t.iod, 3), lo_arc=round(self.lo.A / self.t.iod, 3),
                    up_cols=dict(harvested=int((self.up.src == 1).sum()), mirrored=int((self.up.src == 2).sum()), extended=int((self.up.src == 3).sum()), fallback=int((self.up.src == 4).sum())),
                    lo_cols=dict(harvested=int((self.lo.src == 1).sum()), mirrored=int((self.lo.src == 2).sum()), extended=int((self.lo.src == 3).sum()), fallback=int((self.lo.src == 4).sum())),
                    bite_deg=round(float(np.degrees(self.bite_angle())), 2), tongue=np.round(self.tongue_c, 1).tolist(),
                    chin_pin_px_p95=round(float(np.percentile(pin, 95)), 2), chin_pin_raw_px_p95=round(float(np.percentile(np.linalg.norm(self.pin_raw, axis=1), 95)), 2),
                    pin_mode=self.pin_mode, dth_deg_p95=round(float(np.degrees(np.percentile(np.abs(self.dth), 95))), 3),
                    clamp_fires=int((cl > 0.05).sum()), clamp_px_max=round(float(cl.max()), 2),
                    harvest=dict(up=self.harvest_used['up'], lo=self.harvest_used['lo'], up_h_iod=round(self.up.harvest_h, 4), lo_h_iod=round(self.lo.harvest_h, 4),
                                 up_period_iod=None if self.up.period is None else round(self.up.period, 4), up_x_hl=round(self.up.x_hl, 2), up_n_teeth=self.up.n_teeth,
                                 lo_period_iod=None if self.lo.period is None else round(self.lo.period, 4), synth_cols=dict(up=self.up.synth_cols, lo=self.lo.synth_cols)),
                    fallback=self.fallback, texel=self.texel, harvest_log=self.harvest_log, harvest_cands=getattr(self, 'harvest_cands', None),
                    cover_lip_frames=int(sum(1 for _, f in self.cover_lip if f > 0.005)), cover_lip_max=round(max([f for _, f in self.cover_lip], default=0.0), 4),
                    tongue_mode=self.tongue_mode, cover_mode=self.cover_mode, rim=list(self.rim), wet=self.wet, smooth=self.smooth_f,
                    # v11: what frames[i].reg is measured in (the skull-stabilised frame: canonical plane z, origin, IOD)
                    reg_meta=dict(origin=[round(float(x), 4) for x in self.ref[:2]], iod=round(float(self.t.iod), 4), z=round(float(self.z_up), 4),
                                  end_arc_iod=REG_END), interior_clip=self.interior_clip,
                    **({'single': self.single_info()} if getattr(self, 'single', None) is not None else {}),   # v12 --warp-reg single
                    **({'plate_build': self.plate_build_info()} if getattr(self, 'build', 'v11') == 'v12' else {}),  # v12 --warp-arch-build v12
                    **({'layout': self.layout_info} if getattr(self, 'lay', None) else {}),       # v13 --warp-arch-layout anthro
                    **({'corridor': self.corridor_info()} if getattr(self, 'corridor', 0.0) > 0 else {}),   # v13 --warp-arch-corridor
                    **({'mipmap': True} if getattr(self, 'mipmap', False) else {}),                # v14 --warp-arch-mipmap
                    **({'enamel': self.enamel} if getattr(self, 'enamel', None) else {}),          # v14 --warp-arch-enamel
                    **({'detail': self.detail} if getattr(self, 'detail', 0.0) > 0 else {}))       # v14 --warp-arch-detail

    def plate_build_info(self):
        """v12 --warp-arch-build v12: the harvest choice (teeth_layer.TeethLayer.build_log), the lower harvest frame, the
        crown heights, and the double-edge ruler (lsdub/audits/layer_reg.plate_edges) of the built upper / lower plates."""
        from ..audits import layer_reg as lr_
        io = self.t.iod
        return dict(log=getattr(self.t, 'build_log', None), h_lt=int(self.h_lt), lower_synth=bool(self.lo_synth),
                    edges=dict(up={k: v for k, v in lr_.plate_edges(self.up.tex, True, io, self.up.texel, self.up.h0).items() if k != 'counts'},
                               lo={k: v for k, v in lr_.plate_edges(self.lo.tex, False, io, self.lo.texel, self.lo.h0).items() if k != 'counts'}))

    def single_info(self):
        """v12: the single-reference placement's log (plan_single): settings, how often the floors bound, the jaw's step
        p95 before / after the low-pass, the upper gum guard, the per-frame exposure the kick would have taken (IOD)."""
        s = dict(self.single)
        ov = np.array(list(s.pop('over').values()) or [0.0])
        s.update(reg='single', over_iod_p95=round(float(np.percentile(ov, 95)), 4), over_iod_max=round(float(ov.max()), 4))
        return s
