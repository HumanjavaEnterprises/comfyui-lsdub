#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""frame_gate.py: the CHECKER of the v9 per-frame gate (the maker / confirmation loop is lsdub/core/frame_gate.py). CPU.

Every frame of a render is checked against its neighbours and against the source take, in the SKULL-STABILISED mouth
crop: the take's own rigid skull pose (RigidPlacement on the strict skull points: the pose the warp itself rides) maps a
canonical window around the mouth into every frame, so head motion is not counted; geometric numbers are in IOD units.
The take is the calibration: its own frame-to-frame change is what "allowed change" means, so the continuity rulers are
EXCESS over the take at the same instant, and the thresholds (TH) are set so the takes themselves pass.

Rulers (per frame t; * = two-sided, uses t+1):
  pop*        excess interpolation error: mean |out[t] - (out[t-1] + out[t+1]) / 2| in the mouth ROI minus the take's
              (a frame that is not "between" its neighbours: a one-frame pop / flicker the take does not have)
  step        excess raw change t-1 -> t in the mouth ROI
  mc          excess motion-compensated residual t-1 -> t (DIS flow, forward-backward consistent pixels): the part of
              the change flow cannot explain (a texture swap)
  geom*       jerk of the OUTPUT's own landmarks (IMAGE-mode MediaPipe on the output frame, outside-in) minus the jerk
              of the outline the pipeline planned (IOD): a geometric jump in the picture
  lm_in*      jerk of the pipeline's INPUT track (the take's raw landmarks) at t: a MediaPipe jump the warp consumed
  drift       output mouth (upper-lip mid, mouth centre) vs where the pipeline placed it, in the skull frame (IOD):
              the per-frame offset beyond the shot's median offset, and the slow slide (smoothed)
  fold        min Jacobian determinant of the frame's applied warp; < 0 = the map folds (a smeared second edge)
  teeth       incisal edge jump vs t-1 (upper vs the skull + pin, lower vs the planned jaw) and 1 - NCC of the visible
              teeth t vs t-1 (a texture swap)
  ghost       the lip-aware ghost check (a second lip-coloured edge)
  seam        mask-edge gradient ratio vs the take |r - 1|, and the luminance step across the edge vs the take (L)
  colour      lip Lab jump vs t-1, excess over the take's own jump (dE); lip a*b* vs the lip-lock reference (logged)
  outside     changed pixels (> 8) outside the dilated mask, %
  leak        blobs of interior-coloured pixels (cavity-dark or tooth) outside the planned opening, and (lipleak) of
              clearly lip-coloured pixels outside the planned outer lip, in the mouth region (1/1000 IOD^2), excess over
              the take's own (its pixels vs its own track): the old mouth / old lip showing (the cover not erasing it)
  lm_err      the pipeline's INPUT landmarks vs an independent face-box re-detection of the take (IOD, shot bias
              removed): a held, lost or wrong input track (the warp then moves the wrong pixels)
  viseme      output gap / width (measured) vs the audio-driven target, shot bias removed (units of the take's range)
  slosh_win   windowed (W frames each side): the lip TEXTURE flow vs the lip OUTLINE motion, both projected on the lip
              tangent / normal, upper and lower lip: (i) reversals the audio did not ask for, (ii) oscillatory energy
              sum |v| - |net| (texture and outline), (iii) texture-vs-outline mismatch energy, per step of the window;
              excess over the take's own numbers in the same window (the take's lips = its own re-detection: with the
              pipeline's input track there, input jitter would make the take as sloshy as the output that inherits it)
              -> SLOSH windows (frames listed)
Verdict: a ruler flags when its robust z (median / MAD of the shot's own series, scale floored) is >= Z AND its value is
>= its absolute floor (SOFT); FAIL when it is also >= its fail level, or a hard gate fires (fold < 0, a lip-aware ghost,
outside > 2 %, drift slide > 0.05 IOD). Cause codes: CAUSES.

  lsdub frame-gate --video out.mp4 --take take.mp4 --take-trk trk.npz [--warp-json w.json] [--masks m.npz]
      [--frames N] [--json gate.json] [--csv gate.csv] [--strip strip.png] [--take-mode] [--slosh-w 4]
      [--out-trk cache.npz] [--take-lm cache.npz]
"""
import argparse, csv, json, os, sys
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d, median_filter

from ..core import teeth_layer as tl, lip_layers as ll, mouth_morph as mm, mouth_paint as mp_
from . import lip_edge as gc

LIP_OUT, LIP_IN = ll.LIP_OUT, ll.LIP_IN
UP_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
LO_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
UP_LINE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LO_LINE = [146, 91, 181, 84, 17, 314, 405, 321, 375, 324, 318, 402, 317, 14, 87, 178, 88, 95]
GEOM = [0, 13, 14, 17, 61, 291]                                  # the outline points whose jerk / drift is measured
CTRL = LIP_OUT + LIP_IN + [18, 200, 199, 175]                    # mouth_warp.CTRL
# cause codes (the taxonomy of the founder's "random frames where we error out")
CAUSES = {'lm_jump': '(a) MediaPipe landmark jump on the input, consumed by the warp',
          'fold': '(b) warp fold / tear', 'drift': '(c) mouth moves relative to the skull (drift)',
          'teeth': '(d) teeth jump / texture swap', 'colour': '(e) lip luminance / colour flicker', 'ghost': '(f) ghost second lip',
          'residue': '(g) cover / erase residue showing', 'seam': '(h) seam halo at the mask edge',
          'donor': '(i) donor interior swap', 'viseme': '(j) wrong for the audio (smooth but off the viseme target)',
          'slosh': '(k) slosh (windowed: texture moving against the outline)', 'pop': 'one-frame photometric pop, no other ruler',
          'geom': 'output geometry jump (landmarks of the output)', 'outside': 'change outside the mask', 'lost': 'no face found on the output'}

# thresholds, calibrated on the 7 takes (Stage 0, v9/work/calibration.md): floor (SOFT, with robust z >= Z) = the
# takes' own p99.5 of the same ruler (the take checked against itself: its series minus its running median), fail
# (FAIL) = just above the takes' own MAXIMUM, so no take frame reaches FAIL; scale = the floor of the robust scale (MAD)
# so a near-constant series does not turn noise into a huge z. Units: pop / step / mc in 8-bit levels (mean over the
# mouth ROI), geometry in IOD, colour in dE, leak in 1/1000 IOD^2, slosh in IOD (summed over the window).
TH = dict(pop=dict(floor=18.0, fail=28.0, scale=2.0), step=dict(floor=25.0, fail=36.0, scale=3.0), mc=dict(floor=7.0, fail=12.0, scale=0.8),
          geom=dict(floor=0.045, fail=0.085, scale=0.01), drift=dict(floor=0.06, fail=0.10, scale=0.01),
          teeth=dict(floor=0.06, fail=0.14, scale=0.01), teeth_ncc=dict(floor=1.0, fail=1.5, scale=0.2),
          seam=dict(floor=1.3, fail=2.0, scale=0.2), seam_step=dict(floor=80.0, fail=120.0, scale=10.0),
          colour=dict(floor=17.0, fail=21.0, scale=2.0), viseme=dict(floor=0.35, fail=0.6, scale=0.08),
          lm_in=dict(floor=0.09, fail=0.14, scale=0.01), lm_err=dict(floor=0.08, fail=0.11, scale=0.012),
          leak=dict(floor=17.0, fail=25.0, scale=2.0), lipleak=dict(floor=12.0, fail=19.0, scale=1.5),
          slosh=dict(floor=0.0135, fail=0.019, scale=0.002))
HARD = dict(fold=0.0, outside=0.02, drift_slide=0.05, held_lm_err=0.06)
# which rulers make a frame's verdict (the others are evidence for the cause attribution / logged)
VERDICT_KEYS = ('pop', 'step', 'mc', 'geom', 'drift', 'teeth', 'colour', 'leak', 'lipleak', 'viseme', 'lm_err')   # seam / teeth_ncc: logged
# v11: a re-detection is only used when it is a plausible fit of the face the take's track follows (redet_ok /
# redet_check). MediaPipe's IMAGE mode can lock onto a false, smaller face on a single frame (mv30 v10d S3 s1234 f50 and
# s2718 f82: the whole mesh 0.8 IOD off at a third of the size). Trusted, such a fit became the gate's "repaired" input
# track (lm_hold through held_fix, k5-smoothed over 5 frames) and the warp drew the new mouth at the nostrils; it also
# raised false lm_err FAILs on the frames around it. Bounds, from v9's 7 takes and their renders (the take side and the
# output side): the skull points' median distance to the track reaches 0.22 IOD, the spread ratio 0.76..1.04, the jump of
# the skull offset vs the neighbouring re-detections 0.14 IOD; the two false fits: 0.78 / 0.84, 0.31 / 0.46, 0.82 / 0.84.
REDET = dict(skull=0.4, spread=(0.6, 1.6), jump=0.3)
SKULL_PTS = [33, 133, 263, 362, 6, 168, 1, 4, 10, 151, 9, 8, 234, 454]   # eye corners, nose bridge / tip, forehead, face sides


def skull_offset(lm, ref):
    """landmarks lm vs the track's ref at the same frame (both 478 x >= 2, frame px): (median distance of the skull points,
    IOD; their median offset vector, IOD; the ratio of their spreads about the centroid)."""
    lm, ref = np.asarray(lm, float), np.asarray(ref, float)
    io = max(1e-6, iod_of(ref))
    e = (lm[SKULL_PTS, :2] - ref[SKULL_PTS, :2]) / io
    sp = lambda P: float(np.sqrt(((P[SKULL_PTS, :2] - P[SKULL_PTS, :2].mean(0)) ** 2).sum(1).mean()))
    return float(np.median(np.linalg.norm(e, axis=1))), np.median(e, 0), sp(lm) / max(1e-6, sp(ref))


def redet_ok(lm, ref):
    """v11: one re-detection is a plausible fit of the tracked face (the single-frame part of redet_check)."""
    if lm is None or not np.isfinite(lm).all(): return False
    d, _, r = skull_offset(lm, ref)
    return d <= REDET['skull'] and REDET['spread'][0] <= r <= REDET['spread'][1]


def redet_check(R, T):
    """v11: which re-detections R (n x 478 x 3, nan = none) to trust, given the take's track T: redet_ok on each, and
    the skull offset within REDET['jump'] IOD of the median offset of the plausible re-detections at t-2..t+2 (a one-frame
    flip to another face). -> (bool mask, [dict(i, skull, spread, jump)] of the rejected ones)"""
    R = np.asarray(R, float)
    n = min(len(R), len(T))
    ok = np.zeros(len(R), bool)
    off = np.full((len(R), 2), np.nan)
    info = {}
    for t in range(n):
        if not np.isfinite(R[t]).all(): continue
        d, o, r = skull_offset(R[t], T[t])
        off[t] = o
        info[t] = dict(i=int(t), skull=round(d, 3), spread=round(r, 3))
        ok[t] = d <= REDET['skull'] and REDET['spread'][0] <= r <= REDET['spread'][1]
    keep = ok.copy()
    for t in np.nonzero(ok)[0]:
        js = [j for j in range(t - 2, t + 3) if j != t and 0 <= j < n and ok[j]]
        if js:
            jmp = float(np.linalg.norm(off[t] - np.median(off[js], 0)))
            info[t]['jump'] = round(jmp, 3)
            if jmp > REDET['jump']: keep[t] = False
    rej = [info[t] for t in sorted(info) if not keep[t]]
    return keep, rej


def filter_redet(R, T):
    """R with the implausible re-detections (redet_check) set to nan -> (R', rejected list)."""
    R = np.array(R, float, copy=True)
    keep, rej = redet_check(R, T)
    for x in rej: R[x['i']] = np.nan
    return R, rej


def iod_of(P):
    P = np.asarray(P, float)
    return float(np.linalg.norm(P[33, :2] - P[263, :2]))


def poly(P, idx, shape, grow=0.0, px=0.0):
    return tl.poly_mask(np.asarray(P, float)[idx, :2], shape, grow, px)


class Landmarker:
    """IMAGE-mode MediaPipe face landmarks: every frame measured on its own (a re-made frame is re-measured alone),
    upscaled x2 as lsdub's tracker does. __call__(img) -> (478, 3) frame px (z x W) or None."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        from .._models import face_landmarker_path
        self.mp = mp
        self.lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_landmarker_path()), running_mode=vision.RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))

    def __call__(self, img, box=None, size=640):
        """box = (cx, cy, side) frame px: the face is cropped there and scaled to size x size (a medium shot's small
        face is found as reliably as a close-up's); None = the whole frame x2."""
        H, W = img.shape[:2]
        if box is None:
            big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ox = oy = 0.0
            w_, h_ = W, H
        else:
            cx, cy, sd = box
            sd = int(round(sd))
            x0, y0 = int(round(cx - sd / 2)), int(round(cy - sd / 2))
            pad = cv2.copyMakeBorder(img, sd, sd, sd, sd, cv2.BORDER_REPLICATE)
            big = cv2.resize(pad[y0 + sd:y0 + 2 * sd, x0 + sd:x0 + 2 * sd], (size, size), interpolation=cv2.INTER_CUBIC)
            w_ = h_ = sd
            ox, oy = x0, y0
        r = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(big[:, :, ::-1])))
        if not r.face_landmarks: return None
        return np.array([[q.x * w_ + ox, q.y * h_ + oy, q.z * w_] for q in r.face_landmarks[0]])


def face_box(P, k=1.7):
    """(cx, cy, side) of a face crop around landmarks P (the take's track: the head is the take's in every version)."""
    P = np.asarray(P, float)[:, :2]
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2
    return c[0], c[1], k * float((hi - lo).max())


class Stab:
    """the canonical mouth window of a take: crop px <-> frame i px through the take's skull pose (the plane through the
    lips, orthographic: one affine per frame). Crop scale: IOD = at most max_iod crop px."""

    def __init__(self, pl, C, max_iod=160.0):
        Cm = np.median(C, 0)
        self.iod_c = float(np.linalg.norm(Cm[263, :2] - Cm[33, :2]))
        c = C[:, [13, 14, 61, 291], :2].mean((0, 1))
        self.c = c
        self.z0 = float(np.median(C[:, [13, 14], 2]))
        self.k = min(1.0, max_iod / self.iod_c)
        io = self.iod_c
        self.x0, self.y0 = c[0] - 0.75 * io, c[1] - 0.45 * io
        self.w, self.h = int(round(1.5 * io * self.k)), int(round(1.05 * io * self.k))
        self.iod = io * self.k
        B = np.array([[1 / self.k, 0, self.x0], [0, 1 / self.k, self.y0]])
        self.M = [tl.compose(tl.plane_affine(pl, i, self.z0, ref=tuple(c)), B) for i in range(len(pl.s))]
        self.Mi = [tl.inv_aff(M) for M in self.M]

    def crop(self, i, img):
        return cv2.warpAffine(img, self.M[i], (self.w, self.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    def to_crop(self, i, P):
        P = np.asarray(P, float)[..., :2]
        return P @ self.Mi[i][:, :2].T + self.Mi[i][:, 2]

    def vec_to_crop(self, i, v):
        return np.asarray(v, float) @ self.Mi[i][:, :2].T


def robust(v, scale_floor):
    """(median, scale) of a series (nan ignored): scale = max(1.4826 MAD, floor)."""
    x = np.asarray(v, float)
    x = x[np.isfinite(x)]
    if not len(x): return 0.0, scale_floor
    med = float(np.median(x))
    return med, max(1.4826 * float(np.median(np.abs(x - med))), scale_floor)


def osc(v):
    """oscillatory energy of a velocity sequence: sum |v| - |sum v| (0 for motion in one direction)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.abs(v).sum() - abs(v.sum())) if len(v) else 0.0


def reversals(v, dead):
    """sign changes between consecutive significant (|v| > dead) values."""
    v = np.asarray(v, float)
    s = np.sign(v[np.isfinite(v) & (np.abs(v) > dead)])
    return int((s[1:] * s[:-1] < 0).sum()) if len(s) > 1 else 0


class Checker:
    """the per-frame checker of one shot. take: the take's frames; S3raw: its raw MediaPipe 3D track (the pipeline's
    input); plan: per-frame pipeline internals (dict of arrays / lists, any may be missing): P_out (where the pipeline
    put the lips, 478 x 2 frame px), Fe (n x 2 effective target gap / width, canonical px), F (observed), fold (min det),
    jaw (deg, planned + chin-pin extra), pin (upper teeth pin px); rng: the take's feature range (warp.json 'range');
    S: the pipeline's smoothed track (k5); pl: the pipeline's RigidPlacement (rebuilt from S3raw when None)."""

    def __init__(self, take, S3raw, plan=None, rng=None, S=None, pl=None, lip_ref=None, z=3.0, slosh_w=4, lmk=None, take_lm=None,
                 th=None):
        self.take = take
        n = len(take)
        self.n = n
        self.S3raw = np.asarray(S3raw, float)[:n]
        self.S = mp_.smooth_track(self.S3raw, 'k5') if S is None else np.asarray(S, float)[:n]
        if pl is None:
            ref = self.S3raw[int(np.argmin(np.abs(np.arange(n) - n // 2)))]
            pl = mm.RigidPlacement(ref, self.S3raw, mm.MESH_ANCH, smooth='sg7', lam=0.0)
        self.pl = pl
        self.C = np.array([pl.to_lib(i, self.S[i]) for i in range(n)])
        self.stab = Stab(pl, self.C)
        self.plan = plan or {}
        self.rng = rng
        self.Z = z
        self.W = slosh_w
        self.th = {k: dict(v) for k, v in TH.items()}
        if th: [self.th.setdefault(k, {}).update(v) for k, v in th.items()]
        self.lmk = lmk
        self.lip_ref = None if lip_ref is None else np.asarray(lip_ref, float)
        st = [ll.lip_stats(take[i], self.S[i][:, :2]) for i in range(n)]
        ok = [x for x in st if x is not None]
        md = np.median(ok, 0) if ok else np.array([120, 160, 150, 140, 145.0])
        self.lip_ab, self.skin_ab = md[1:3], md[3:5]
        self.skin_L = float(np.nanmedian([tl.skin_ring_L(take[i], self.S[i]) for i in range(0, n, 3)]))
        self.take_lm = take_lm                                   # IMAGE-mode landmarks of the take (cache), n x 478 x 3
        self.obs_k = [None] * n
        self.rel_k = [None] * n
        self.obs = [None] * n
        self.rel = [None] * n
        self.ref_stats = None

    # ------------------------------------------------------------------------------------------- observation --
    def landmarks(self, img, i=None):
        if self.lmk is None: self.lmk = Landmarker()
        return self.lmk(img, None if i is None else face_box(self.S3raw[i]))

    def observe(self, i, img, lm=None, mask=None, take_side=False, redetect=True):
        """per-frame observation of frame i (an output frame, or the take's own when take_side). lm: its landmarks when
        already measured (a cache: nan / None = no usable face, not measured again unless redetect)."""
        sb = self.stab
        H, W = img.shape[:2]
        o = dict(i=i, det=True)
        if lm is None and redetect:
            lm = self.landmarks(img, i)
            if lm is not None and not redet_ok(lm, self.S3raw[i]):   # v11: a false face fit is no detection
                o['redet_rejected'] = True
                lm = None
        if lm is None or not np.isfinite(lm).all():             # no face found: the masks from the planned / take outline,
            o['det'] = False                                     # the geometric rulers unmeasured at this frame
            P = None if take_side else self._plan_P(i)
            lm = np.asarray(self.S[i], float).copy()
            if P is not None: lm[:, :2] = P
        lm = np.asarray(lm, float)
        o['lm'] = lm
        crop = sb.crop(i, img)
        o['crop'] = crop
        o['gray'] = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        q = sb.to_crop(i, lm)
        o['q'] = q
        Cq = self.pl.to_lib(i, lm) if lm.shape[1] == 3 else None
        if Cq is not None:
            o['gap'] = float(np.linalg.norm(Cq[13, :2] - Cq[14, :2]))
            o['width'] = float(np.linalg.norm(Cq[61, :2] - Cq[291, :2]))
        # lip colour: the median of the clearly lip-coloured pixels of the frame's lip tissue (robust to a landmark that
        # puts the tissue mask on a tooth or the skin)
        o['lab'] = self._lip_lab(img, lm[:, :2])
        # teeth edges in the crop (IOD from the crop's top): upper = lowest teeth row of the arch hanging from the upper
        # lip in the central columns, lower = highest teeth row below the lips' mid-line
        o.update(self._teeth(crop, q))
        # the lip-aware ghost check on the frame itself (with its own landmarks, and with the planned outline)
        P_plan = None if take_side else self._plan_P(i)
        g, lo_, up_ = gc.frame_ghost_lip(img, lm[:, :2])
        g2 = gc.frame_ghost_lip(img, P_plan)[0] if P_plan is not None else False
        o['ghost'] = bool(g or g2)
        # leak: interior-coloured / old-lip pixels outside the lips the frame is supposed to show (the plan; the take's
        # own track for the take)
        # (the take's own baseline uses its re-detected landmarks: where its lips really are, not the input track)
        o['leak'], o['lipleak'] = self._leak(img, P_plan if P_plan is not None else (lm[:, :2] if o['det'] else self.S[i][:, :2]), i)
        if not take_side:
            o.update(self._outside_seam(img, self.take[i], mask))
            # swim (warp2/tools/swim.py, the earlier slosh ruler): the edit displacement output -> take at the same
            # instant, median over the lip core (planned outer lip), crop px
            ok_ = self.obs_k[i]
            if ok_ is not None:
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                fw = dis.calc(o['gray'], ok_['gray'], None)
                core = poly(sb.to_crop(i, P_plan if P_plan is not None else lm[:, :2]), LIP_OUT, o['gray'].shape, 0.0)
                o['edit'] = np.median(fw[core], 0) if core.sum() >= 15 else np.full(2, np.nan)
        return o

    def _lip_lab(self, img, P):
        w = mm.lip_tissue_mask(P, img.shape)
        if w.sum() < 30: return np.full(3, np.nan)
        ys, xs = np.nonzero(w)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = img[y0:y1, x0:x1]
        lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB).astype(np.float32)
        m = w[y0:y1, x0:x1]
        lp = ll.lipness(lab, self.lip_ab, self.skin_ab) > 0.5
        k = m & lp
        if k.sum() < 20: k = m
        return np.array([float(np.median(lab[..., c][k])) for c in range(3)])

    def _leak(self, img, P, i):
        """(interior, old-lip) leak of frame i around the outline P (478 x 2 frame px), in 1/1000 IOD^2: blobs of
        interior-coloured pixels (cavity-dark or tooth) outside the opening P[LIP_IN] (+0.03 IOD) and blobs of clearly
        lip-coloured pixels outside the outer lip P[LIP_OUT] (+0.03 IOD), inside the mouth region (outer lip x 1.35)."""
        P = np.asarray(P, float)[:, :2]
        io = iod_of(self.S[i])
        q = P[LIP_OUT]
        x0, y0 = np.maximum(0, np.floor(q.min(0) - 0.4 * io).astype(int))
        x1, y1 = np.ceil(q.max(0) + 0.4 * io).astype(int)
        x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
        sub = img[y0:y1, x0:x1]
        sh = sub.shape
        Pl = P - [x0, y0]
        lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)
        roi = poly(Pl, LIP_OUT, sh, 0.35)
        inter = ((lab[..., 0] < 0.45 * self.skin_L) | tl.teeth_key(lab, self.skin_L)) & roi & ~poly(Pl, LIP_IN, sh, 0.0, 0.03 * io)
        if getattr(self, 'leak_scene', False) and i is not None:
            # v11 (opt-in): the SCENE is not a leak -- a pixel that is just as dark / tooth-like in the take at the same
            # place, outside the take's own mouth (its outer lip grown 25 %), is hair / a nostril / background the bigger
            # ROI of a wider planned mouth reaches (mv30 NC f74-76: ~80 % of the "leak" was the hair at the far corner)
            tk = self.take[i][y0:y1, x0:x1]
            labk = cv2.cvtColor(tk, cv2.COLOR_BGR2LAB)
            Sk = np.asarray(self.S[i], float)[:, :2] - [x0, y0]
            scene = ((labk[..., 0] < 0.45 * self.skin_L) | tl.teeth_key(labk, self.skin_L)) & ~poly(Sk, LIP_OUT, sh, 0.25)
            inter = inter & ~scene
        lipx = (tl.lipness_map(sub, self.lip_ab, self.skin_ab) > 0.6) & roi & ~poly(Pl, LIP_OUT, sh, 0.0, 0.03 * io)
        a_min = max(4.0, (0.02 * io) ** 2)
        def blobs(m):
            nl, lbl, cc, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
            return float(sum(cc[k, cv2.CC_STAT_AREA] for k in range(1, nl) if cc[k, cv2.CC_STAT_AREA] >= a_min))
        return 1000 * blobs(inter) / io ** 2, 1000 * blobs(lipx) / io ** 2

    def _plan_P(self, i):
        P = self.plan.get('P_out')
        tk = self.plan.get('take')
        if tk is not None and i < len(tk) and tk[i]: return None    # v10a: the take's own frame: no plan, its own lips (as the take side)
        return None if P is None or P[i] is None else np.asarray(P[i], float)

    def _teeth(self, crop, q):
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        io = self.stab.iod
        h, w = crop.shape[:2]
        inner = tl.poly_mask(q[LIP_IN], crop.shape, 0.05)
        t = inner & tl.teeth_key(lab, self.skin_L)
        mid = (q[13, 1] + q[14, 1]) / 2
        cx = q[[13, 14], 0].mean()
        yy, xx = np.mgrid[0:h, 0:w]
        cen = np.abs(xx - cx) < 0.12 * io
        r = dict(e_up=np.nan, e_lo=np.nan, t_up=None, t_lo=None)
        up = t & (yy < mid) & (yy <= q[13, 1] + 0.06 * io + (mid - q[13, 1]))
        lo = t & (yy >= mid)
        if (up & cen).sum() >= max(6, 0.0015 * io * io):
            r['e_up'] = float(np.percentile(yy[up & cen], 90)) / io
            r['t_up'] = up
        if (lo & cen).sum() >= max(6, 0.0015 * io * io):
            r['e_lo'] = float(np.percentile(yy[lo & cen], 10)) / io
            r['t_lo'] = lo
        r['L'] = lab[..., 0].astype(np.float32)
        return r

    def _outside_seam(self, img, tk, mask):
        r = dict(outside=np.nan, seam=np.nan, seam_step=np.nan, halo=np.nan)
        if mask is None: return r
        m = np.asarray(mask, np.float32)
        if m.shape != img.shape[:2]: m = cv2.resize(m, (img.shape[1], img.shape[0]))
        md = cv2.dilate((m > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        d = np.abs(img.astype(np.int16) - tk.astype(np.int16)).max(2)
        r['outside'] = float((d[~md] > 8).mean()) if (~md).any() else 0.0
        if (m > 0.5).sum() < 50: return r
        ys, xs = np.nonzero(m > 0.03)
        y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
        ms = m[y0:y1, x0:x1]
        Ls = [cv2.cvtColor(z[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) for z in (img, tk)]
        # skin only: the warp's falloff ring is skin; lips (re-coloured by the lip lock, moved by the warp) are not a seam
        skin = mm.skin_px(tk[y0:y1, x0:x1]) & (tl.lipness_map(tk[y0:y1, x0:x1], self.lip_ab, self.skin_ab) < 0.3) & \
            (tl.lipness_map(img[y0:y1, x0:x1], self.lip_ab, self.skin_ab) < 0.3)
        G = [np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)) for L in Ls]
        yb, xb = np.nonzero(ms > 0.5)
        cx, cy, wx = xb.mean(), yb.mean(), np.ptp(xb)
        yy, xx = np.mgrid[0:ms.shape[0], 0:ms.shape[1]]
        sides = {'L': xx < cx - wx / 6, 'R': xx > cx + wx / 6, 'chin': (np.abs(xx - cx) <= wx / 6) & (yy > cy)}
        gr, stp = [], []
        for s, sd in sides.items():
            ins, out, band = (ms > 0.55) & (ms < 0.85) & sd & skin, (ms > 0.03) & (ms < 0.2) & sd & skin, (ms > 0.25) & (ms < 0.75) & sd & skin
            if ins.sum() > 20 and out.sum() > 20:
                st = [L[ins].mean() - L[out].mean() for L in Ls]
                stp.append(abs(st[0] - st[1]))
            if band.sum() > 20: gr.append(abs(G[0][band].mean() / (G[1][band].mean() + 1e-3) - 1))
        hb = (ms > 0.03) & (ms < 0.25) & skin
        r['seam'] = float(max(gr)) if gr else np.nan
        r['seam_step'] = float(max(stp)) if stp else np.nan
        r['halo'] = float((np.abs(Ls[0] - Ls[1])[hb] > 12).mean()) if hb.sum() > 20 else np.nan
        return r

    # --------------------------------------------------------------------------------------------- relations --
    def relate(self, A, B):
        """t-1 -> t on the stabilised crops: raw change, motion-compensated residual, lip texture velocity (upper /
        lower lip, crop px), teeth texture NCC."""
        if A is None or B is None: return None
        ga, gb = A['gray'], B['gray']
        h, w = ga.shape
        io = self.stab.iod
        roi = poly(A['q'], LIP_OUT, ga.shape, 0.35) | poly(B['q'], LIP_OUT, ga.shape, 0.35)
        r = dict(raw=float(np.abs(B['crop'].astype(np.float32) - A['crop'].astype(np.float32))[roi].mean()) if roi.any() else np.nan)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        fw = dis.calc(ga, gb, None)
        bk = dis.calc(gb, ga, None)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        bw = cv2.remap(bk, xx + fw[..., 0], yy + fw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ok = np.linalg.norm(fw + bw, axis=2) < 0.7
        wb = cv2.remap(gb.astype(np.float32), xx + fw[..., 0], yy + fw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        res = np.abs(wb - ga.astype(np.float32))
        m = roi & ok
        r['mc'] = float(res[m].mean()) if m.sum() > 30 else np.nan
        r['inc'] = float(1 - ok[roi].mean()) if roi.any() else np.nan
        for name, idx in (('up', UP_LIP), ('lo', LO_LIP)):
            lm_ = poly(A['q'], idx, ga.shape, -0.05) & ok
            r['v_' + name] = np.median(fw[lm_], 0) if lm_.sum() >= 12 else np.full(2, np.nan)
        # teeth NCC t-1 vs t (upper arch as is; lower shifted by its measured edge change)
        nccs = []
        for key, e in (('t_up', None), ('t_lo', 'e_lo')):
            ta, tb = A.get(key), B.get(key)
            if ta is None or tb is None: continue
            La, Lb = A['L'], B['L']
            dy = 0
            if e and np.isfinite(A[e]) and np.isfinite(B[e]): dy = int(round((B[e] - A[e]) * io))
            Mt = np.float32([[1, 0, 0], [0, 1, -dy]])
            Lbs = cv2.warpAffine(Lb, Mt, (w, h))
            tbs = cv2.warpAffine(tb.astype(np.uint8), Mt, (w, h)) > 0
            ov = ta & tbs
            if ov.sum() >= 25:
                x, y = La[ov], Lbs[ov]
                if x.std() > 1 and y.std() > 1: nccs.append(float(np.corrcoef(x, y)[0, 1]))
        r['ncc'] = min(nccs) if nccs else np.nan
        return r

    # ----------------------------------------------------------------------------------------------- loading --
    def observe_take(self, masks=None):
        """the take's own observations / relations (the calibration side)."""
        n = self.n
        for i in range(n):
            lm = None if self.take_lm is None else self.take_lm[i]
            if lm is not None and not np.isfinite(lm).all(): lm = None
            # (a cached re-detection that is missing or was rejected as implausible is not measured again)
            self.obs_k[i] = self.observe(i, self.take[i], lm=lm, take_side=True, redetect=self.take_lm is None)
        for i in range(1, n): self.rel_k[i] = self.relate(self.obs_k[i - 1], self.obs_k[i])
        return self

    def observe_all(self, frames, masks=None, lms=None):
        n = self.n
        for i in range(n):
            self.obs[i] = self.observe(i, frames[i], None if lms is None else lms[i], None if masks is None else masks[i], redetect=lms is None)
        for i in range(1, n): self.rel[i] = self.relate(self.obs[i - 1], self.obs[i])
        return self

    def update(self, i, img, mask=None, lm=None, plan_i=None):
        """frame i was re-made: re-observe it, re-relate it to both neighbours. plan_i: its new plan entries."""
        if plan_i:
            for k, v in plan_i.items():
                if k in self.plan and self.plan[k] is not None: self.plan[k][i] = v
        self.obs[i] = self.observe(i, img, lm, mask)
        if i >= 1: self.rel[i] = self.relate(self.obs[i - 1], self.obs[i])
        if i + 1 < self.n: self.rel[i + 1] = self.relate(self.obs[i], self.obs[i + 1])

    # ------------------------------------------------------------------------------------------------ metrics --
    def _series(self, obs, rel, outline, slosh_outline=None):
        """the per-frame raw series of one side (output or take). slosh_outline: the outline the lip texture is compared
        with in the slosh ruler (default: outline)."""
        n = self.n
        io = self.stab.iod
        ser = {}
        crops = [o['crop'].astype(np.float32) for o in obs]
        pop = np.full(n, np.nan)
        for t in range(1, n - 1):
            roi = poly(obs[t]['q'], LIP_OUT, crops[t].shape, 0.35)
            if roi.any(): pop[t] = float(np.abs(crops[t] - 0.5 * (crops[t - 1] + crops[t + 1]))[roi].mean())
        ser['pop'] = pop
        g = lambda k: np.array([np.nan if (r is None or not np.isfinite(r.get(k, np.nan))) else r[k] for r in rel])
        ser['raw'], ser['mc'], ser['inc'], ser['ncc'] = g('raw'), g('mc'), g('inc'), g('ncc')
        # measured outline jerk (crop coords, IOD)
        Q = np.array([o['q'][GEOM] if o['det'] else np.full((len(GEOM), 2), np.nan) for o in obs]) / io
        jk = np.full(n, np.nan)
        jk[1:-1] = np.linalg.norm(Q[1:-1] - 0.5 * (Q[:-2] + Q[2:]), axis=2).max(1)
        ser['jerk'] = jk
        # the outline the frame is supposed to show (the plan for outputs, the take's own smoothed track for the take)
        Qp = np.array([self.stab.to_crop(t, outline[t])[GEOM] for t in range(n)]) / io
        jp = np.full(n, np.nan)
        jp[1:-1] = np.linalg.norm(Qp[1:-1] - 0.5 * (Qp[:-2] + Qp[2:]), axis=2).max(1)
        ser['jerk_plan'] = jp
        # drift: measured upper-lip mid and mouth centre vs the outline, beyond the shot's median offset
        def mids(Qx): return np.stack([Qx[:, [0, 1]].mean(1), Qx[:, [4, 5]].mean(1)], 1)        # (0,13) mid, (61,291) mid
        off = mids(Q) - mids(Qp)
        med = np.nanmedian(off, 0)
        dv = np.linalg.norm(off - med, axis=2).max(1)
        ser['drift'] = dv
        offs = gaussian_filter1d(np.nan_to_num(off - med), 2.0, axis=0, mode='nearest')
        ser['drift_slide'] = np.linalg.norm(offs, axis=2).max(1)
        # teeth edges
        e_up = np.array([o['e_up'] for o in obs])
        e_lo = np.array([o['e_lo'] for o in obs])
        ser['e_up'], ser['e_lo'] = e_up, e_lo
        lab = np.array([o['lab'] for o in obs])
        dE = np.full(n, np.nan)
        dE[1:] = np.linalg.norm(lab[1:] - lab[:-1], axis=1)
        ser['dE'] = dE
        ser['lab'] = lab
        ser['ghost'] = np.array([float(o['ghost']) for o in obs])
        ser['leak'] = np.array([o.get('leak', np.nan) for o in obs])
        ser['lipleak'] = np.array([o.get('lipleak', np.nan) for o in obs])
        ser['Q'] = Q
        ser['Qp'] = Qp
        for k in ('outside', 'seam', 'seam_step', 'halo'): ser[k] = np.array([o.get(k, np.nan) for o in obs])
        ser['det'] = np.array([float(o['det']) for o in obs])
        ser['gap_m'] = np.array([o.get('gap', np.nan) if o['det'] else np.nan for o in obs])
        ser['width_m'] = np.array([o.get('width', np.nan) if o['det'] else np.nan for o in obs])
        # lip velocities: texture (flow) and outline (the outline's own lip points), crop px -> IOD, on the lip
        # tangent (corner 61 -> 291) / normal (down)
        vt = np.full((n, 2, 2), np.nan)
        vo = np.full((n, 2, 2), np.nan)
        so = outline if slosh_outline is None else slosh_outline
        for t in range(1, n):
            r = rel[t]
            qa = self.stab.to_crop(t - 1, so[t - 1])
            qb = self.stab.to_crop(t, so[t])
            u = qa[291] - qa[61]
            u = u / max(1e-6, np.linalg.norm(u))
            nn_ = np.array([-u[1], u[0]])
            if nn_[1] < 0: nn_ = -nn_
            for k, (name, line) in enumerate((('up', UP_LINE), ('lo', LO_LINE))):
                if r is not None:
                    v = r['v_' + name]
                    vt[t, k] = (v @ u / io, v @ nn_ / io)
                d = (qb[line] - qa[line]).mean(0)
                vo[t, k] = (d @ u / io, d @ nn_ / io)
        ser['v_tex'], ser['v_out'] = vt, vo
        return ser

    def _input_jerk(self):
        """jerk of the pipeline's INPUT track (the take's raw landmarks, skull frame, IOD) at the mouth points."""
        n = self.n
        io = self.stab.iod
        Q = np.array([self.stab.to_crop(t, self.S3raw[t])[GEOM + [78, 308, 82, 87, 312, 317]] for t in range(n)]) / io
        j = np.full(n, np.nan)
        j[1:-1] = np.linalg.norm(Q[1:-1] - 0.5 * (Q[:-2] + Q[2:]), axis=2).max(1)
        return j

    def _input_err(self):
        """the pipeline's INPUT landmarks (the take's raw track) vs an independent face-box re-detection of the take
        (IMAGE mode), at the mouth points, beyond the shot's median offset (IOD): a held / lost / wrong input track."""
        n = self.n
        io = self.stab.iod
        pts = [13, 14, 61, 291, 0, 17, 78, 308]
        S_in = self.plan.get('S_in')
        S_in = self.S if S_in is None else S_in        # the (k5) landmarks each frame used
        R = np.array([o['lm'] if o['det'] else np.full((478, 3), np.nan) for o in self.obs_k], float)
        ok = np.isfinite(R).all((1, 2))
        if ok.any():
            for k in range(R.shape[1]):
                for d_ in range(2): R[~ok, k, d_] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], R[ok, k, d_])
        Rk = mp_.smooth_track(R, 'k5')
        Qi = np.array([self.stab.to_crop(t, S_in[t])[pts] for t in range(n)]) / io
        Qr = np.array([self.stab.to_crop(t, Rk[t])[pts] for t in range(n)]) / io
        d = Qr - Qi
        med = np.nanmedian(d, 0)
        e = np.linalg.norm(d - med, axis=2).max(1)
        held = np.zeros(n, bool)
        held[1:] = np.abs(np.diff(self.S3raw[:, :, :2], axis=0)).max((1, 2)) < 1e-6
        if n > 1 and held[1]: held[0] = True
        if self.plan.get('held_fixed') is not None: held &= ~np.asarray(self.plan['held_fixed'], bool)
        self.held = held
        return e

    def metrics(self, take_mode=False):
        """the per-frame rulers (dict of arrays) from the current observations. take_mode: the take checked against
        itself = each continuity series minus its own running median (the calibration of the floors)."""
        n = self.n
        plan = self.plan
        outline_k = [self.S[t][:, :2] for t in range(n)]
        # the take's lips in the slosh ruler: its own re-detection (where they really are). With the pipeline's input
        # track here instead, a jittery input track would make the take look as sloshy as the output that inherits
        # the jitter, and the excess would hide exactly that slosh
        redet_k = [o['lm'][:, :2] if o['det'] else self.S[t][:, :2] for t, o in enumerate(self.obs_k)]
        if take_mode:
            K = self._series(self.obs_k, self.rel_k, outline_k, redet_k)
            base = lambda x: x - median_filter(np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(x).any() else 0.0), 9, mode='nearest')
            M = dict(pop=base(K['pop']), step=base(K['raw']), mc=base(K['mc']), geom=base(K['jerk'] - K['jerk_plan']))
            M['drift'] = np.full(n, np.nan)
            M['drift_slide'] = np.full(n, np.nan)        # (the take vs itself: 0)
            M['colour'] = base(K['dE'])
            M['ncc'] = K['ncc']
            M['ghost'] = np.zeros(n)        # (the take vs itself)
            M['leak'] = base(K['leak'])
            M['lipleak'] = base(K['lipleak'])
            M['lm_err'] = np.full(n, np.nan)
            M.update(self._teeth_metrics(K, None, take=True))
            M['teeth_ncc'] = base(1 - np.nan_to_num(K['ncc'], nan=1.0))
            M['lm_in'] = self._input_jerk()
            M['viseme'] = np.full(n, np.nan)
            M['fold'] = np.full(n, np.nan)
            for k in ('outside', 'seam', 'seam_step', 'halo'): M[k] = np.full(n, np.nan)
            M['slosh_win'], M['slosh_parts'] = self._slosh(K, None, take_mode=True)
            self.K = K
            return M
        Pp = plan.get('P_out')
        outline_o = [np.asarray(Pp[t], float) if (Pp is not None and Pp[t] is not None) else self.S[t][:, :2] for t in range(n)]
        if not hasattr(self, 'K') or self.K is None or getattr(self, '_K_mode', None) != 'out':
            self.K = self._series(self.obs_k, self.rel_k, outline_k, redet_k)
            self._K_mode = 'out'
        O = self._series(self.obs, self.rel, outline_o)
        K = self.K
        self.O = O
        M = dict(pop=O['pop'] - K['pop'], step=O['raw'] - K['raw'], mc=O['mc'] - K['mc'])
        M['geom'] = O['jerk'] - O['jerk_plan']
        # drift, same landmark mode on both sides: (output - take, both re-detected) minus (plan - the take's track), at the
        # upper-lip mid and the mouth centre, beyond the shot's median offset (IOD); + its slow slide
        def mids(Qx): return np.stack([Qx[:, [0, 1]].mean(1), Qx[:, [4, 5]].mean(1)], 1)
        S_in = self.plan.get('S_in')
        S_in = self.S if S_in is None else S_in        # the take landmarks each frame used
        Qs = np.array([self.stab.to_crop(t, np.asarray(S_in[t], float)[:, :2])[GEOM] for t in range(n)]) / self.stab.iod
        off = (mids(O['Q']) - mids(K['Q'])) - (mids(O['Qp']) - mids(Qs))
        med = np.nanmedian(off, 0)
        M['drift'] = np.linalg.norm(off - med, axis=2).max(1)
        offs = gaussian_filter1d(np.nan_to_num(off - med), 2.0, axis=0, mode='nearest')
        M['drift_slide'] = np.linalg.norm(offs, axis=2).max(1)
        M['leak'] = O['leak'] - np.nan_to_num(K['leak'], nan=0.0)
        M['lipleak'] = O['lipleak'] - np.nan_to_num(K['lipleak'], nan=0.0)
        M['lm_err'] = self._input_err()
        M['held'] = self.held.astype(float)
        M['colour'] = O['dE'] - np.nan_to_num(K['dE'], nan=0.0)
        if self.lip_ref is not None: M['colour_ref'] = np.linalg.norm(O['lab'][:, 1:] - self.lip_ref[1:], axis=1)
        M['teeth_ncc'] = 1 - np.nan_to_num(O['ncc'], nan=1.0)            # the output's teeth are fixed textures: absolute
        M.update(self._teeth_metrics(O, K))
        M['ghost'] = O['ghost'] * (1 - K['ghost'])        # a second lip edge the take itself does not show at that frame
        for k in ('outside', 'seam', 'seam_step', 'halo'): M[k] = O[k]
        M['fold'] = np.asarray(plan['fold'], float) if plan.get('fold') is not None else np.full(n, np.nan)
        M['lm_in'] = self._input_jerk()
        M['det_fail'] = (1 - O['det']) * K['det']        # lost on the output, found on the take
        M['viseme'] = self._viseme(O)
        M['slosh_win'], M['slosh_parts'] = self._slosh(O, K)
        # swim core_hf per frame (logged; the windowed slosh ruler's cross-check): |E - 5-frame mean of E| / IOD
        from scipy.ndimage import uniform_filter1d
        E = np.array([o.get('edit', np.full(2, np.nan)) for o in self.obs], float)
        okE = np.isfinite(E).all(1)
        if okE.sum() >= 5:
            tt = np.arange(n)
            E2 = E.copy()
            for k in range(2): E2[~okE, k] = np.interp(tt[~okE], tt[okE], E[okE, k])
            hf = np.linalg.norm(E2 - uniform_filter1d(E2, 5, axis=0, mode='nearest'), axis=1) / self.stab.iod
            hf[~okE] = np.nan
        else: hf = np.full(n, np.nan)
        M['swim_hf'] = hf
        return M

    def _teeth_metrics(self, O, K, take=False):
        """incisal edge jumps (IOD): upper vs the skull (+ the upper pin in the crop), lower vs the planned jaw."""
        n = self.n
        plan = self.plan
        e_up, e_lo = O['e_up'].copy(), O['e_lo'].copy()
        if not take and plan.get('pin') is not None:
            pin = np.asarray(plan['pin'], float)
            e_up = e_up - np.array([self.stab.vec_to_crop(t, pin[t])[1] for t in range(n)]) / self.stab.iod
        if not take and plan.get('jaw') is not None:
            th = np.radians(np.asarray(plan['jaw'], float))
            okm = np.isfinite(e_lo) & np.isfinite(th)
            if okm.sum() >= 5:
                A = np.stack([np.ones(okm.sum()), th[okm]], 1)
                c, *_ = np.linalg.lstsq(A, e_lo[okm], rcond=None)
                e_lo = e_lo - (c[0] + c[1] * th)
        elif take:      # the take's own lower teeth ride its own lower lip
            Q14 = np.array([self.stab.to_crop(t, self.S[t])[14, 1] for t in range(n)]) / self.stab.iod
            e_lo = e_lo - Q14
        jt = np.full(n, np.nan)
        for t in range(1, n):
            v = [abs(x[t] - x[t - 1]) for x in (e_up, e_lo) if np.isfinite(x[t]) and np.isfinite(x[t - 1])]
            if v: jt[t] = max(v)
        if take:
            ok = np.isfinite(jt)
            base = jt.copy()
            base[ok] = jt[ok] - np.median(jt[ok]) if ok.any() else jt[ok]
            return dict(teeth=base)
        return dict(teeth=jt)

    def _viseme(self, O):
        """|measured - target| in units of the take's range (gap, width), shot bias removed; max of the two."""
        Fe = self.plan.get('Fe')
        r = self.rng
        if Fe is None or r is None: return np.full(self.n, np.nan)
        Fe = np.asarray(Fe, float)
        gs = max(1e-6, r['g_hi'] - r['g_lo'])
        ws = max(1e-6, r['w_hi'] - r['w_lo'])
        rg = (O['gap_m'] - Fe[:, 0]) / gs
        rw = (O['width_m'] - Fe[:, 1]) / ws
        sk = self.plan.get('skip')               # v10a: a skipped vocalisation has no audio target (not scored, not in the bias)
        if sk is not None:
            sk = np.asarray(sk, bool)[:self.n]
            rg = np.where(sk, np.nan, rg)
            rw = np.where(sk, np.nan, rw)
        return np.maximum(np.abs(rg - np.nanmedian(rg)), np.abs(rw - np.nanmedian(rw)))

    # -------------------------------------------------------------------------------------------------- slosh --
    def _slosh(self, O, K, take_mode=False):
        """slosh_win per window centre c (frames c-W .. c+W): for both lips and both axes (tangent, normal),
        mismatch m = v_tex - v_outline; energy = oscillatory part of m (sum |m| - |sum m|) + the excess of the
        texture's own reversals over the audio's (x a reversal unit); the take's own energy in the same window is
        subtracted (take_mode: the take's energy minus its running median). Returns (per-centre excess, parts)."""
        n, W = self.n, self.W
        def energy(S, Fe=None):
            vt, vo = S['v_tex'], S['v_out']
            E = np.full(n, np.nan)
            parts = np.full((n, 5), np.nan)
            tv = None
            if Fe is not None:
                tv = np.full((n, 2), np.nan)
                tv[1:] = np.diff(np.asarray(Fe, float), axis=0) / (self.stab.iod_c)
            for c in range(n):
                a0, a1 = max(1, c - W + 1), min(n - 1, c + W)
                if a1 - a0 + 1 < max(3, W): continue
                mis = osc_t = osc_o = rev = 0.0
                for k in range(2):
                    for ax in range(2):
                        v = vt[a0:a1 + 1, k, ax]
                        vv = vo[a0:a1 + 1, k, ax]
                        if np.isfinite(v).sum() < 3: continue
                        m = v - vv
                        mis += osc(m)
                        osc_t += osc(v)
                        osc_o += osc(vv)
                        rt = reversals(v, 0.002)
                        ra = reversals(tv[a0:a1 + 1, 1 - ax], 0.0005) if tv is not None else reversals(vv, 0.002)
                        rev += max(0, rt - ra)
                steps = a1 - a0 + 1               # per step of the window: one threshold serves W = 3, 4, 6
                E[c] = (mis + 0.004 * rev) / steps
                parts[c] = (mis, osc_t, osc_o, rev, 0)
            return E, parts
        if take_mode:
            Ek, Pk = energy(O)
            base = Ek - median_filter(np.nan_to_num(Ek, nan=np.nanmedian(Ek) if np.isfinite(Ek).any() else 0), 2 * W + 1, mode='nearest')
            return base, Pk
        Eo, Po = energy(O, self.plan.get('Fe'))
        Ek, Pk = energy(K)
        ex = Eo - np.nan_to_num(Ek, nan=0.0)
        Po[:, 4] = Ek
        return ex, Po

    # ------------------------------------------------------------------------------------------------ verdict --
    def stats(self, M):
        return {k: robust(M[k], self.th[k]['scale']) for k in self.th if k in M and k != 'slosh'} | \
               {'slosh': robust(M['slosh_win'], self.th['slosh']['scale'])}

    def verdict(self, M, ref=None, z=None):
        """per frame: verdict (PASS / SOFT / FAIL), causes (codes, worst first), severity (0 = clean; >= 1 = FAIL
        level), the robust z per ruler; plus the SLOSH windows. ref: fixed (median, scale) per ruler (the round-0
        distribution, so a candidate is judged on the same scale)."""
        n = self.n
        Z = self.Z if z is None else z
        ref = ref or self.stats(M)
        th = self.th
        V = ['PASS'] * n
        causes = [[] for _ in range(n)]
        sev = np.zeros(n)
        zs = {}
        def flag(i, code, s, fail):
            causes[i].append((code, round(float(s), 3)))
            sev[i] = max(sev[i], s)
            if fail: V[i] = 'FAIL'
            elif V[i] == 'PASS': V[i] = 'SOFT'
        for k in VERDICT_KEYS:
            if k not in M: continue
            v = np.asarray(M[k], float)
            med, sc = ref[k]
            z_ = (v - med) / sc
            zs[k] = z_
            for i in np.nonzero(np.isfinite(v) & (z_ >= Z) & (v >= th[k]['floor']))[0]:
                f = v[i] >= th[k]['fail']
                flag(i, k, v[i] / th[k]['fail'], f)
        # hard gates
        for i in np.nonzero(np.nan_to_num(np.asarray(M.get('fold', np.full(n, np.nan)), float), nan=1.0) < HARD['fold'])[0]: flag(i, 'fold', 2.0, True)
        for i in np.nonzero(np.nan_to_num(np.asarray(M.get('outside', np.full(n, np.nan)), float), nan=0.0) > HARD['outside'])[0]: flag(i, 'outside', 2.0, True)
        for i in np.nonzero(np.nan_to_num(np.asarray(M.get('drift_slide', np.full(n, np.nan)), float), nan=0.0) > HARD['drift_slide'])[0]: flag(i, 'drift', 1.5, True)
        for i in np.nonzero(np.asarray(M.get('ghost', np.zeros(n))) > 0)[0]: flag(i, 'ghost', 1.5, True)
        if 'held' in M:     # the input track is held (no detection) and disagrees with the re-detection
            for i in np.nonzero((np.asarray(M['held']) > 0) & (np.nan_to_num(np.asarray(M.get('lm_err', np.zeros(n)), float)) >= HARD['held_lm_err']))[0]:
                flag(i, 'lm_err', 1.5, True)
        if 'det_fail' in M:
            for i in np.nonzero(np.asarray(M['det_fail']) > 0)[0]: flag(i, 'lost', 0.8, False)
        # slosh windows
        sw = np.asarray(M['slosh_win'], float)
        med, sc = ref['slosh']
        zsl = (sw - med) / sc
        zs['slosh'] = zsl
        hit = np.isfinite(sw) & (zsl >= Z) & (sw >= th['slosh']['floor'])
        wins = []
        for c in np.nonzero(hit)[0]:
            a0, a1 = int(max(0, c - self.W)), int(min(n - 1, c + self.W))
            f = sw[c] >= th['slosh']['fail']
            if wins and a0 <= wins[-1]['frames'][1]:
                wins[-1]['frames'][1] = a1
                wins[-1]['centres'].append(int(c))
                wins[-1]['energy'] = max(wins[-1]['energy'], float(sw[c]))
                wins[-1]['fail'] |= bool(f)
            else: wins.append(dict(frames=[a0, a1], centres=[int(c)], energy=float(sw[c]), fail=bool(f)))
        slosh_frames = np.zeros(n, bool)
        for w in wins:
            for i in range(w['frames'][0], w['frames'][1] + 1): slosh_frames[i] = True
        # attribution: the founder's cause list (a)..(k)
        lmz = (np.asarray(M['lm_in'], float) - ref['lm_in'][0]) / ref['lm_in'][1] if 'lm_in' in M and 'lm_in' in ref else np.zeros(n)
        lm_hit = np.nan_to_num(lmz, nan=0) >= Z
        lm_hit &= np.nan_to_num(np.asarray(M.get('lm_in', np.zeros(n)), float), nan=0) >= th['lm_in']['floor']
        out = []
        for i in range(n):
            codes = [c for c, _ in causes[i]]
            attr = []
            vis = {'pop', 'step', 'mc', 'geom', 'teeth', 'teeth_ncc', 'colour', 'leak', 'lipleak', 'seam', 'seam_step', 'drift', 'viseme'}
            if 'lm_err' in codes or (any(c in vis for c in codes) and any(lm_hit[max(0, i - 2):i + 3])): attr.append('lm_jump')
            for c, a in (('fold', 'fold'), ('drift', 'drift'), ('teeth', 'teeth'), ('teeth_ncc', 'teeth'), ('colour', 'colour'),
                         ('ghost', 'ghost'), ('leak', 'residue'), ('lipleak', 'residue'), ('seam', 'seam'), ('seam_step', 'seam'), ('viseme', 'viseme'),
                         ('outside', 'outside'), ('lost', 'lost'), ('geom', 'geom')):
                if c in codes and a not in attr: attr.append(a)
            if any(c in ('pop', 'step', 'mc') for c in codes) and not attr: attr.append('pop')
            if slosh_frames[i]: attr.append('slosh')
            out.append(dict(i=i, verdict=V[i], causes=sorted(causes[i], key=lambda x: -x[1]), attr=attr, sev=round(float(sev[i]), 3)))
        return dict(frames=out, windows=wins, ref={k: [round(float(a), 5), round(float(b), 5)] for k, (a, b) in ref.items()},
                    z={k: [None if not np.isfinite(x) else round(float(x), 2) for x in v] for k, v in zs.items()})


# ------------------------------------------------------------------------------------------- offline helpers --
def replay(warp_json, S3raw, shape, lam=0.01, pose_smooth='sg7', track_smooth='k5'):
    """the pipeline's own placement of a finished render, rebuilt from its warp.json and the take's raw track: per
    frame P_out (where the lips were put, frame px), the applied warp's min Jacobian determinant, Fe, F, jaw, pin.
    (The warp is TPS through dst -> src with the HeadEllipse ring; dst = src + J dFe in the skull frame.)"""
    from ..core import mouth_warp as mw
    w = json.load(open(warp_json)) if isinstance(warp_json, str) else warp_json
    fr = w['frames']
    n = min(len(fr), len(S3raw))
    S3raw = np.asarray(S3raw, float)[:n]
    S = mp_.smooth_track(S3raw, track_smooth)
    ref = S3raw[int(np.argmin(np.abs(np.arange(n) - n // 2)))]
    pl = mm.RigidPlacement(ref, S3raw, mm.MESH_ANCH, smooth=pose_smooth, lam=0.0)
    C = np.array([pl.to_lib(i, S[i]) for i in range(n)])
    J = np.asarray(w['J'], float)
    F = np.array([[f['gap'], f['width']] for f in fr[:n]])
    Fe = np.array([[f['egap'], f['ewidth']] for f in fr[:n]])
    ix = {k: j for j, k in enumerate(mw.CTRL)}
    iin = [ix[k] for k in LIP_IN]
    ell = None
    if w.get('ellipse'):
        e = w['ellipse']
        ell = object.__new__(mw.HeadEllipse)
        ell.c = C[:, [13, 14, 61, 291]].mean((0, 1))
        ell.ax, ell.ay, ell.r_in, ell.r_out = tuple(e['ax']), tuple(e['ay']), e['r_in'], e['r_out']
    P_out, fold = [], []
    for i in range(n):
        src = S[i][mw.CTRL, :2]
        dst = src + (np.tensordot(Fe[i] - F[i], J, axes=(0, 0)) @ (pl.s[i] * pl.R[i]).T)[:, :2]
        if 'yaw_clamp_dst' in fr[i]: dst = np.asarray(fr[i]['yaw_clamp_dst'], float)          # v12: --warp-yaw-clamp's own outline
        P = S[i][:, :2].copy()
        P[mw.CTRL] = dst
        P_out.append(P)
        if ell is None or np.abs(dst - src).max() < 0.25:
            fold.append(1.0)
            continue
        fold.append(mw.fold_det(src, dst, ell.anchors(pl, i), ell.weight(pl, i, shape), lam, inner=dst[iin]))
    jaw = [f.get('jaw_deg', np.nan) + f.get('jaw_dth_deg', 0.0) for f in fr[:n]]
    pin = [f.get('teeth_pin', [0.0, 0.0]) for f in fr[:n]]
    sk_ = (w.get('vocal') or {}).get('mode') == 'skip'                                                # v10a
    skip = [bool(f.get('vocal_skip', False)) for f in fr[:n]] if sk_ else None
    take = [float(f.get('vocal_s', 1.0)) <= 0.0 for f in fr[:n]] if sk_ else None
    return dict(P_out=P_out, fold=fold, Fe=Fe, F=F, jaw=jaw, pin=pin, S=S, pl=pl, range=w.get('range'), n=n, skip=skip, take=take)


def read_frames(path, n=None):
    c = cv2.VideoCapture(path)
    F = []
    while n is None or len(F) < n:
        ok, f = c.read()
        if not ok: break
        F.append(f)
    return F


def landmark_cache(frames, path=None, lmk=None, boxes_from=None, check=True):
    """IMAGE-mode landmarks of every frame (n x 478 x 3, nan when no face), cached in path (raw). boxes_from: a track whose
    per-frame face box crops the frame first (the take's raw track); v11 (check): the re-detections that are not a
    plausible fit of that track's face (redet_check) are returned as nan (the cache keeps them raw)."""
    out = None
    if path and os.path.exists(path):
        z = np.load(path)['S3']
        if len(z) == len(frames): out = z
    if out is None:
        lmk = lmk or Landmarker()
        out = []
        for i, f in enumerate(frames):
            p = lmk(f, None if boxes_from is None else face_box(boxes_from[i]))
            out.append(p if p is not None else np.full((478, 3), np.nan))
        out = np.array(out)
        if path: np.savez(path, S3=out)
    if check and boxes_from is not None:
        out, rej = filter_redet(out, boxes_from)
        if rej: print(f"frame-gate: {len(rej)} implausible re-detection(s) not used: " + ', '.join(
            f"f{x['i']} (skull {x['skull']} IOD, spread {x['spread']}" + (f", jump {x['jump']}" if 'jump' in x else '') + ')' for x in rej))
    return out


def strip_image(verdicts, width, height=12, cursor=None):
    """the VERDICT STRIP: one tick per frame (green PASS, amber SOFT, red FAIL; a SLOSH frame gets a violet underline),
    for a bottom label band (never over the picture). cursor: the current frame (a white mark)."""
    n = len(verdicts)
    img = np.zeros((height, width, 3), np.uint8)
    col = {'PASS': (60, 170, 60), 'SOFT': (0, 175, 255), 'FAIL': (40, 40, 230)}
    for i, v in enumerate(verdicts):
        x0, x1 = int(round(i * width / n)), max(int(round(i * width / n)) + 1, int(round((i + 1) * width / n)) - 1)
        img[:height - 3, x0:x1] = col[v['verdict'] if isinstance(v, dict) else v]
        if isinstance(v, dict) and 'slosh' in v.get('attr', []): img[height - 3:, x0:x1] = (200, 60, 160)
    if cursor is not None:
        x = int(round((cursor + 0.5) * width / n))
        img[:, max(0, x - 1):x + 2] = 255
    return img


def report(V, M, path_json=None, path_csv=None, extra=None):
    """gate.json (per-frame metrics, z, verdict, causes + windows + counts) and the CSV."""
    n = len(V['frames'])
    keys = [k for k in M if k != 'slosh_parts']
    cnt = dict(PASS=0, SOFT=0, FAIL=0)
    for f in V['frames']: cnt[f['verdict']] += 1
    rows = []
    for f in V['frames']:
        i = f['i']
        r = dict(f)
        r['m'] = {k: (None if not np.isfinite(np.asarray(M[k], float)[i]) else round(float(np.asarray(M[k], float)[i]), 5)) for k in keys}
        rows.append(r)
    out = dict(n=n, counts=cnt, windows=V['windows'], ref=V['ref'], frames=rows, thresholds=dict(TH=TH, HARD=HARD))
    if extra: out.update(extra)
    if path_json: json.dump(out, open(path_json, 'w'))
    if path_csv:
        with open(path_csv, 'w', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['frame', 'verdict', 'sev', 'attr', 'causes'] + keys)
            for r in rows: wr.writerow([r['i'], r['verdict'], r['sev'], ' '.join(r['attr']), ' '.join(f'{c}:{s}' for c, s in r['causes'])] + [r['m'][k] for k in keys])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsdub frame-gate', description=__doc__.split('\n')[0])
    ap.add_argument('--video', help='the render to check (omit with --take-mode)')
    ap.add_argument('--take', required=True)
    ap.add_argument('--take-trk', required=True, help='npz with S3 = the take\'s raw MediaPipe 3D track (the pipeline\'s input)')
    ap.add_argument('--warp-json', help='the render\'s warp.json (targets, J, ellipse, jaw, pins): the pipeline\'s placement is replayed')
    ap.add_argument('--masks', help='the render\'s mask npz (half res, as --mask-out writes)')
    ap.add_argument('--frames', type=int)
    ap.add_argument('--lam', type=float, default=0.01)
    ap.add_argument('--lip-ref', default='107,169.7,157.6')
    ap.add_argument('--json')
    ap.add_argument('--csv')
    ap.add_argument('--strip')
    ap.add_argument('--take-mode', action='store_true')
    ap.add_argument('--slosh-w', type=int, default=4)
    ap.add_argument('--z', type=float, default=3.0)
    ap.add_argument('--out-trk', help='cache of the render\'s IMAGE-mode landmarks')
    ap.add_argument('--take-lm', help='cache of the take\'s IMAGE-mode landmarks')
    ap.add_argument('--plan', help='a gated render\'s final plan (<gate report>_plan.npz): the frames the gate re-made keep '
                                   'their own placement / input landmarks (the warp.json replay only knows round 0)')
    ap.add_argument('--leak-scene', action='store_true', help='v11: the leak ruler ignores the scene (pixels as dark / tooth-like '
                                   'in the take at the same place, outside its own mouth: hair, nostrils, background)')
    ap.add_argument('--reg', action='store_true', help='v11: also the inter-layer registration ruler (lsdub layer-reg) on the '
                                   'warp.json reg series (reported in the json under "reg")')
    a = ap.parse_args(argv)
    S3raw = np.load(a.take_trk)['S3']
    n = a.frames or len(S3raw)
    take = read_frames(a.take, n)
    n = min(n, len(take), len(S3raw))
    take = take[:n]
    plan, rng, S, pl = {}, None, None, None
    if a.warp_json and not a.take_mode:
        rp = replay(a.warp_json, S3raw[:n], take[0].shape[:2], a.lam)
        n = min(n, rp['n'])
        plan = {k: rp[k] for k in ('P_out', 'fold', 'Fe', 'F', 'jaw', 'pin', 'skip', 'take')}
        rng, S, pl = rp['range'], rp['S'], rp['pl']
        if a.plan:
            z = np.load(a.plan)
            for k in ('P_out', 'S_in', 'held_fixed', 'fold', 'jaw', 'pin'):
                if k in z.files: plan[k] = [x for x in z[k][:n]]
    lmk = Landmarker()
    tlm = landmark_cache(take, a.take_lm, lmk, S3raw[:n])
    ck = Checker(take[:n], S3raw[:n], plan, rng, S, pl, ll.parse_lip_ref(a.lip_ref), a.z, a.slosh_w, lmk, tlm)
    ck.leak_scene = a.leak_scene
    ck.observe_take()
    if a.take_mode:
        M = ck.metrics(take_mode=True)
    else:
        out = read_frames(a.video, n)
        masks = None
        if a.masks:
            mz = np.load(a.masks)['masks'].astype(np.float32) / 255
            masks = [mz[i] for i in range(min(n, len(mz)))]
        olm = landmark_cache(out, a.out_trk, lmk, S3raw[:n])
        ck.observe_all(out, masks, [None if not np.isfinite(x).all() else x for x in olm])
        M = ck.metrics()
    extra = dict(video=a.video, take=a.take, take_mode=a.take_mode, slosh_w=a.slosh_w)
    if a.reg and a.warp_json:
        from . import layer_reg
        extra['reg'] = layer_reg.gate_report(json.load(open(a.warp_json))['frames'][:n], a.slosh_w)
    V = ck.verdict(M)
    r = report(V, M, a.json, a.csv, extra)
    if a.strip: cv2.imwrite(a.strip, strip_image(V['frames'], 1280, 14))
    print(f"frame-gate {'take' if a.take_mode else os.path.basename(os.path.dirname(a.video or '')) or a.video}: {r['counts']} "
          f"slosh windows {[w['frames'] for w in V['windows']]}; FAIL {[f['i'] for f in V['frames'] if f['verdict'] == 'FAIL']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
