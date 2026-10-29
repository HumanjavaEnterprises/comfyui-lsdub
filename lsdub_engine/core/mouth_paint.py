#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_paint.py — paint viseme mouths (lsdub mouths) onto a calm body take, frame by frame.

Per frame: MediaPipe FaceLandmarker on the take (gap-filled, temporally smoothed) -> a similarity transform from the
library's source crop to the frame, fitted on RIGID points (nose bridge/tip, nostrils, cheekbones) — not on the lips,
which must keep the library's shape. The library mouth for the track's viseme (cross-faded with the runner-up viseme by
its share) is warped in, inside a feathered lips+lower-jaw mask (the union of the library mouth and the take's own
mouth, so the take's lips never show through), with a per-frame Lab colour match to the take around the mouth.
Open vowels get their jaw drop scaled by the track's openness (A2F jawOpen or the vocal envelope).
A light temporal pass then blends each painted region 25% with its neighbours to kill shimmer.

  lsdub paint --video body.mp4 --track vis_018.json \
      --lib mouthlib --out painted.mp4 [--frames 200] [--temporal 0.25]
"""
import argparse, json, os
from ..io import ff
import numpy as np, cv2

from .._models import face_landmarker_path
MODEL = face_landmarker_path()
# points around the mouth that ride the head, not the lips: nose base, nostrils, nasolabial folds, cheeks, jaw sides
RIGID = [1, 2, 4, 5, 94, 98, 327, 164, 167, 393, 206, 426, 203, 423, 50, 280, 187, 411, 172, 397, 136, 365, 129, 358]
LIP_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
CHIN = [17, 84, 314, 18, 200]          # stop above the chin tip: the mic sits just below it
LIBMAP = {'kk': 'DD', 'nn': 'DD'}
LAST_OK = None


def _crop_detect(lm, fr, c, s, z, out=640):
    """MediaPipe (IMAGE mode) on a square crop of side s around c, edge-padded so a face cut by the frame edge still
    reads as a face, scaled to `out` px -> 478 x (2|3) landmarks in frame px, or None."""
    import mediapipe as mp
    P = int(s)
    big = cv2.copyMakeBorder(fr, P, P, P, P, cv2.BORDER_REPLICATE)
    x0, y0 = int(c[0] - s / 2) + P, int(c[1] - s / 2) + P
    cr = big[y0:y0 + int(s), x0:x0 + int(s)]
    if cr.shape[0] < 8 or cr.shape[1] < 8: return None
    k = out / cr.shape[0]
    cr = cv2.resize(cr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cr[:, :, ::-1])))
    if not r.face_landmarks: return None
    w = cr.shape[1]
    return np.array([[q.x * w / k + x0 - P, q.y * cr.shape[0] / k + y0 - P] + ([q.z * w / k] if z else []) for q in r.face_landmarks[0]])


def _box(p):
    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    return (lo + hi) / 2, float(max(hi - lo))


def track_chain(video, n, z=False, grow=2.2, jump=0.6):
    """v18 tracker for medium / moving shots. The full-frame pass can lock onto something face-like (a sleeve) and
    lose a face cut by the frame edge. Here every full-frame detection is re-checked on its own padded, enlarged
    crop; the anchor is a frame where both agree; from it the face is followed frame to frame, forward and back,
    each frame detected inside a crop around the previous face (grown x `grow`) and refused if its centre jumps more
    than `jump` x the face size. -> (list of arrays | None, fps, full-frame list)."""
    from mediapipe.tasks.python import vision, BaseOptions
    full, fps, _ = track(video, n, z=z, smooth='none', raw=True)
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.1, min_face_presence_confidence=0.1))
    cap = cv2.VideoCapture(video)
    F = [cap.read()[1] for _ in range(len(full))]
    cap.release()
    n = len(full)
    agree = []
    for i, p in enumerate(full):
        if p is None: continue
        c, s = _box(p)
        q = _crop_detect(lm, F[i], c, s * grow, z)
        if q is not None and np.linalg.norm(_box(q)[0] - c) < 0.15 * s: agree.append(i)
    if not agree: return full, fps, full
    a0 = agree[len(agree) // 2]
    out = [None] * n
    out[a0] = full[a0]
    for rng in (range(a0 + 1, n), range(a0 - 1, -1, -1)):
        c, s = _box(out[a0])
        miss = 0
        for i in rng:
            q = _crop_detect(lm, F[i], c, s * grow * (1.3 ** min(miss, 3)), z)
            if q is not None and np.linalg.norm(_box(q)[0] - c) > jump * s * (1 + miss): q = None
            if q is None and full[i] is not None and np.linalg.norm(_box(full[i])[0] - c) < jump * s * (1 + miss): q = full[i]
            if q is None:
                miss += 1
                continue
            out[i] = q
            c, s = _box(q)
            miss = 0
    return out, fps, full


def track(video, n, z=False, smooth='k5', raw=False, mode='full'):
    if mode == 'chain' and not raw:
        P, fps, _ = track_chain(video, n, z=z)
        return _finish(P, fps, z, smooth)
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.2, min_face_presence_confidence=0.2, min_tracking_confidence=0.2))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    P = []
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        H, W = fr.shape[:2]
        # upscale x2 for detection: the face is ~250 px in a 1344x768 take
        big = cv2.resize(fr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(big[:, :, ::-1])), int(i * 1000 / fps))
        P.append(np.array([[q.x * W, q.y * H] + ([q.z * W] if z else []) for q in r.face_landmarks[0]]) if r.face_landmarks else None)
    cap.release()
    if raw: return P, fps, None
    return _finish(P, fps, z, smooth)


def _finish(P, fps, z, smooth):
    ok = np.array([p is not None for p in P])
    D = 3 if z else 2
    A = np.array([p if p is not None else np.full((478, D), np.nan) for p in P])
    ii = np.arange(len(P))
    for k in range(A.shape[1]):
        for d in range(D): A[:, k, d] = np.interp(ii, ii[ok], A[ok, k, d])
    S = smooth_track(A, smooth)
    global LAST_OK
    LAST_OK = ok   # which frames had a real detection (the rest are interpolated)
    return S, fps, float(ok.mean())


def smooth_track(A, mode='k5'):
    """temporal filter on a landmark track A (T x K x D), offline so centred (zero-lag).
    k5 = [1,2,3,2,1] (default, light); k9 = [1,2,3,4,5,4,3,2,1]; sg9 = Savitzky-Golay window 9, order 2 (wide but keeps
    accelerations, i.e. real head moves); none = raw."""
    if mode == 'none' or len(A) < 3: return A.copy()
    if mode == 'sg9':
        from scipy.signal import savgol_filter
        w = min(9, len(A) if len(A) % 2 else len(A) - 1)
        return savgol_filter(A, w, min(2, w - 1), axis=0, mode='nearest') if w >= 3 else A.copy()
    k = np.array({'k5': [1, 2, 3, 2, 1], 'k9': [1, 2, 3, 4, 5, 4, 3, 2, 1]}[mode], float)
    k /= k.sum()
    h = len(k) // 2
    P = np.concatenate([np.repeat(A[:1], h, 0), A, np.repeat(A[-1:], h, 0)])
    return sum(k[i] * P[i:i + len(A)] for i in range(len(k)))


def hull_mask(pts, shape, grow, blur):
    m = np.zeros(shape[:2], np.uint8)
    c = pts.mean(0)
    q = c + (pts - c) * (1 + grow)
    cv2.fillConvexPoly(m, cv2.convexHull(q.astype(np.int32)), 255)
    k = max(3, int(blur) | 1)
    return cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255


OPEN = {'aa': 1.0, 'E': 0.6, 'ih': 0.4, 'oh': 0.75, 'ou': 0.3, 'RR': 0.25, 'DD': 0.22, 'TH': 0.2, 'CH': 0.2}


def jaw_scale(img, lm, s):
    """Scale the library mouth's opening by s: the lower lip + jaw slide toward the upper lip (s<1) or away (s>1),
    the displacement ramps in from the upper-lip line, holds over the lower lip and fades out by the chin."""
    if abs(s - 1) < 0.03: return img
    H, W = img.shape[:2]
    y13, y14, yc = lm[13, 1], lm[14, 1], lm[152, 1]
    gap = max(1.0, y14 - y13)
    x0, x1 = lm[61, 0], lm[291, 0]
    cx, hw = (x0 + x1) / 2, (x1 - x0) / 2 * 1.6
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    up = np.clip((yy - y13) / gap, 0, 1)
    down = np.clip((yc - yy) / max(1.0, yc - y14), 0, 1)
    wy = np.where(yy < y14, up, down)
    wy = wy * wy * (3 - 2 * wy)
    wx = np.clip(1 - np.abs(xx - cx) / hw, 0, 1)
    wx = wx * wx * (3 - 2 * wx)
    dy = (1 - s) * gap * wy * wx            # sample from further down when closing
    return cv2.remap(img, xx, (yy + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def lab_match(src, ref, w):
    a, b = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32), cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    if w.sum() < 50: return src
    for c in range(3):
        s, r = a[..., c][w], b[..., c][w]
        a[..., c] = (a[..., c] - s.mean()) * np.clip(r.std() / (s.std() + 1e-6), 0.7, 1.4) ** 0.5 + r.mean()
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--track', required=True)
    ap.add_argument('--lib', help='mouth library (legacy/morph; not used by --mode warp)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--grow', type=float, default=0.30, help='mask growth around the lips (fraction of lip-hull size)')
    ap.add_argument('--no-open-scale', dest='open_scale', action='store_false')
    ap.add_argument('--open-min', type=float, default=0.45)
    ap.add_argument('--temporal', type=float, default=0.25)
    ap.add_argument('--debug-dir')
    ap.add_argument('--mode', choices=['legacy', 'morph', 'cutout', 'warp'], default='morph',
                    help='legacy = v4 paint_a (affine + cross-fade); morph = mesh warp; '
                         'cutout = fixed-region paint (no face tracking, for non-human faces); '
                         'warp = no library mouth: warp the take\'s OWN lips toward the audio\'s shape (opt-in, see mouth_warp.py)')
    # --- warp-mode options ---
    ap.add_argument('--warp-interior', choices=['donor', 'dark', 'stretch'], default='donor', help='warp: fill of a mouth opened '
                    'wider than the take: donor = the take\'s own frame nearest in (openness, pose), dark = darkened gap, stretch = warp only')
    ap.add_argument('--warp-gain', type=float, default=1.0, help='warp: share of the residual (target - take) applied')
    ap.add_argument('--warp-open-gain', type=float, default=1.0)
    ap.add_argument('--warp-width-gain', type=float, default=1.0)
    ap.add_argument('--warp-deadband', type=float, default=0.04, help='warp: residual (fraction of the take\'s range) below '
                    'which the take is left untouched (<0 = always warp)')
    ap.add_argument('--warp-sigma', type=float, default=1.0, help='warp: zero-phase gaussian (frames) on the targets')
    ap.add_argument('--warp-lam', type=float, default=0.01, help='warp: TPS regularisation of the dense field')
    ap.add_argument('--warp-ring', type=float, default=None, help='warp: skin ring (zero displacement) at this x the mouth '
                    'envelope (ellipse falloff, default 2.6: wide enough that a full jaw move does not fold) or x the control hull (hull falloff, default 1.6)')
    ap.add_argument('--warp-falloff', choices=['ellipse', 'hull'], default='ellipse', help='warp: ellipse = falloff + ring fixed '
                    'in the canonical head frame, carried by the strict skull pose, C2 smootherstep (r2 default); hull = r1 per-frame '
                    'control hull')
    ap.add_argument('--warp-core', type=float, default=1.0, help='warp (ellipse): full-weight core radius, x the mouth envelope')
    ap.add_argument('--warp-temporal', type=float, default=0.8, help='warp: zero-phase gaussian (frames) on the effective '
                    'residual = on the warp field itself (0 = off, r1)')
    ap.add_argument('--warp-feat-smooth', type=float, default=1.5, help='warp: zero-phase gaussian (frames) on the OBSERVED '
                    'shape before the residual is taken, so landmark noise is not re-injected into the field (0 = off, r1)')
    ap.add_argument('--warp-class-width', type=float, default=0.5, help='warp: spread visemes (ih/E/SS) target at least this '
                    'far above the take\'s median width, round ones (ou/oh/CH/RR) this far below (units of its p10-p90 half-range; '
                    '0 = library bands as-is, r1)')
    ap.add_argument('--warp-fold-guard', type=float, default=0.15, help='warp (ellipse): per frame, scale the residual down '
                    'until the warp map\'s Jacobian determinant stays >= this (no fold = no smeared second lip); 0 = off')
    ap.add_argument('--warp-feather', type=float, default=0.12, help='warp: feather of the donor interior (x the target '
                    'opening); r1 0.35 half-blended the donor cavity over the take\'s own teeth (grey teeth)')
    ap.add_argument('--warp-donor-hold', choices=['segment', 'frame'], default='segment', help='warp: one interior donor per '
                    'opening segment (r2) or per frame with stickiness + cross-fade (frame, r1-like)')
    ap.add_argument('--warp-temporal-width', type=float, default=0.0, help='warp: total zero-phase gaussian (frames) on the WIDTH '
                    'residual, if larger than --warp-temporal (corner moves spread into the cheeks; 0 = same as the gap)')
    ap.add_argument('--warp-centre', type=float, default=0.8, help='warp: a shape outside its viseme band goes this far past '
                    'the band edge toward the band centre (0 = edge, r1)')
    ap.add_argument('--no-warp-jaw', dest='warp_jaw', action='store_false', help='warp: do not re-reference the opening to the '
                    'upper lip (r1: the gap is split between both lips as the take did it)')
    ap.add_argument('--warp-xfade', type=float, default=1.0, help='warp: gaussian (frames) cross-fade between bank sources and '
                    'interior donors (0 = hard switch, r1)')
    ap.add_argument('--warp-donor-stick', type=float, default=0.3, help='warp: stickiness bonus of the previous interior donor')
    ap.add_argument('--warp-ramp', type=float, default=0.08, help='warp: confidence ramp width above the dead-band')
    ap.add_argument('--warp-interp', choices=['lanczos', 'cubic'], default='lanczos', help='warp: resampling of the moved pixels')
    ap.add_argument('--warp-track-mode', choices=['full', 'chain'], default='full', help='warp (v18): full = MediaPipe on the whole '
                    'frame (x2); chain = verified anchor, then each frame detected in a padded, enlarged crop around the previous '
                    'face with a jump check (medium / moving shots, faces cut by the frame edge)')
    ap.add_argument('--warp-yaw-width', type=float, default=0.0, help='warp (v18): P > 0 = the planned width is scaled by the '
                    'head turn\'s foreshortening (eye balance about the nose bridge vs the clip median, ^P), never below the take\'s own '
                    'width: a near-profile frame no longer asks for the frontal width (the far corner on the cheek); 0 = off')
    ap.add_argument('--warp-profile-plain', type=float, default=0.0, help='warp (v18, layers): T > 0 = a frame whose head turn '
                    '(eye balance about the nose bridge, 1 frontal .. 0 profile) is below T keeps the own-pixel warp without the '
                    'synthetic teeth / cavity layers; 0 = off')
    ap.add_argument('--warp-track-cache', help='warp: npz to save / reuse the MediaPipe track of --video')
    ap.add_argument('--warp-pose-smooth', default='sg7', help='warp: zero-phase filter on the skull pose (canonical frame)')
    ap.add_argument('--warp-pose-w', type=float, default=2.0, help='warp: pose weight (per radian) in the donor choice')
    ap.add_argument('--warp-bank', dest='warp_bank', action='store_true', help='warp: allow mode b (pull the nearest mouth from '
                    'elsewhere in the clip, cross-faded by --warp-xfade). Off by default since r2: a pulled mouth is another '
                    'frame\'s texture, and its swaps / cross-fades read as slosh or a double mouth')
    ap.add_argument('--no-warp-bank', dest='warp_bank', action='store_false', help='warp: only correct each frame\'s own mouth '
                    '(mode a; the default)')
    ap.add_argument('--warp-interior-mode', choices=['invented', 'more'], default='invented', help='warp: donor-fill strength '
                    'by the share of the opening the take does not show (invented, r2) or by the opening added (more, r1)')
    ap.add_argument('--warp-bank-thr', type=float, default=0.5, help='warp: residual (fraction of range) above which a bank '
                    'entry is considered')
    ap.add_argument('--warp-width-mix', choices=['switch', 'primary'], default='primary', help='warp (v3 default primary): width band on a vis/vis2 '
                    'mix. switch (r2) = vis2\'s band once mix >= .5 even if vis2 has none; primary = vis\'s band unless vis2 has one and mix > .5')
    ap.add_argument('--warp-class-ref', choices=['half', 'wide'], default='wide', help='warp (v3 default wide): --warp-class-width in units of '
                    'the matching half-range (half, r2) or of the take\'s wider width half-range (wide: a take parked near p90 '
                    'still gets a spread floor clearly above its median)')
    ap.add_argument('--warp-close-depth', type=float, default=0.0, help='warp: PP closures target this far below the shut '
                    'gap (x the take\'s range): a lip press that survives the field smoothing; 0 = touch (r2)')
    ap.add_argument('--warp-jaw-gain', type=float, default=1.0, help='warp: gain on the soft chin\'s share of an opening / '
                    'closing (jaw drop / raise)')
    ap.add_argument('--warp-donor-min', type=float, default=0.0, help='warp: no donor interior on frames whose target '
                    'opening (x the take\'s range) is below this: the take\'s own stretched lips / interior are held')
    ap.add_argument('--warp-seal', type=float, default=0.0, help='warp: target opening (x the take\'s range) below which '
                    'the slit a compressed mouth still shows is inpainted from the same frame\'s lips (closure seal); 0 = off')
    ap.add_argument('--warp-seal-grow', type=float, default=1.6, help='warp: vertical growth of the sealed slit (x the target inner-lip polygon)')
    ap.add_argument('--warp-json', help='warp: per-frame log (observed / target features, confidence, donor)')
    # --- v4 layering (opt-in; lip_layers.py) ---
    ap.add_argument('--warp-size-cap', type=float, default=0.0, help='warp (v4): the target mouth never goes past the take\'s '
                    'own p95 opening and p5..p95 width (x this) unless the take\'s own frame already does; 0 = off')
    ap.add_argument('--warp-erase', choices=['off', 'closures', 'all'], default='off', help='warp (v4): lip-erase layer. '
                    'closures = at a closure over an open take, the take\'s lips are erased to its own skin (bald plate from its '
                    'most-closed frame, carried by the head pose) and its own closed lips go on top in the target shape; all = also '
                    'frames whose target is smaller than the take (the warped lips on top of the bald plate)')
    ap.add_argument('--warp-erase-thr', type=float, default=0.12, help='warp: target opening (x the take\'s range) below which '
                    'a frame is a closure for --warp-erase')
    ap.add_argument('--warp-erase-grow', type=float, default=0.22, help='warp: growth of the erased lip region (x the lip hull)')
    ap.add_argument('--warp-lip-lock', help='warp (v4): one lip colour for the character: "L,a,b" (OpenCV 8-bit Lab) or a PNG '
                    'of a reference face. a*/b* of the lips are locked to it (lip mask x lip-ness, feathered), L follows the shot')
    ap.add_argument('--warp-lip-lock-sigma', type=float, default=4.0, help='warp: zero-phase gaussian (frames) on the lip colour shift')
    ap.add_argument('--warp-lip-lock-l', type=float, default=0.0, help='warp: share (0..1) of the lip lightness also moved to the reference')
    ap.add_argument('--warp-teeth', choices=['off', 'rigid'], default='off', help='warp (v5): rigid = teeth are bone: the '
                    'interior inside the target inner lip comes from rigid teeth plates harvested from the take (upper at the '
                    'skull pose, lower at skull + jaw rotation about the condyle hinge), never from stretched pixels or donors')
    ap.add_argument('--warp-teeth-hinge', default='0.25,0.6', help='warp: jaw hinge "down,back" from the outer-canthi mid-point, x IOD')
    ap.add_argument('--warp-teeth-jaw-gain', type=float, default=1.0, help='warp: lower-teeth drop per unit of target opening change')
    # --- v6 (opt-in; teeth_layer.py): teeth anchored to the skin, the jaw planned over the clip ---
    ap.add_argument('--warp-teeth-anchor', choices=['pose', 'skin'], default='pose', help='warp (v6): skin = the teeth plates take '
                    'the upper-lip root skin\'s own motion (DIS flow of philtrum + nose base in the warped frames, high-passed onto '
                    'the skull pose) so teeth and upper lip share one motion source; pose = the skull pose alone (v5)')
    ap.add_argument('--warp-teeth-pin-sigma', type=float, default=4.0, help='warp: frames; the skin pin is high-passed at this '
                    'sigma (slow placement from the pose, every frame-to-frame step from the skin)')
    ap.add_argument('--warp-teeth-jaw', choices=['frame', 'plan'], default='frame', help='warp (v6): plan = the jaw (lower teeth) '
                    'solved once over the whole clip: target + smoothness + speed / acceleration limits + holds on sustained '
                    'vowels + lower teeth hidden below the lower lip on near-closed frames; frame = per frame from the target lip (v5)')
    ap.add_argument('--warp-teeth-jaw-lead', type=float, default=0.0, help='warp: extra jaw anticipation in frames (0 = the '
                    'lips\' own lead, which the target already carries)')
    ap.add_argument('--warp-teeth-jaw-vmax', type=float, default=3.0, help='warp: jaw speed limit, degrees / frame')
    ap.add_argument('--warp-teeth-jaw-amax', type=float, default=1.0, help='warp: jaw acceleration limit, degrees / frame^2')
    ap.add_argument('--warp-teeth-jaw-smooth', type=float, default=1.5, help='warp: weight of the jaw plan\'s second-difference '
                    'term (the third-difference term is 0.43x it)')
    ap.add_argument('--warp-teeth-hide', type=float, default=0.2, help='warp: target opening (x the take\'s range) below which '
                    'the planned jaw keeps the lower plate hidden below the lower lip; 0 = off')
    ap.add_argument('--warp-teeth-tongue-drop', type=float, default=0.0, help='warp (v6): the lower plate\'s tongue is carried '
                    'this far down (x IOD) behind the lower teeth')
    ap.add_argument('--warp-teeth-plates', type=int, default=1, help='warp (v6): up to N upper plates harvested across the '
                    'head\'s yaw range, blended by pose (1 = one frontal plate, v5)')
    ap.add_argument('--warp-teeth-mirror', action='store_true', help='warp (v6): upper-plate columns the harvest frame does not '
                    'show are filled from the arch\'s mirror image about the face midline; the arch\'s ends recede into shadow and '
                    'feather out (no rectangular cut where the plate ends)')
    ap.add_argument('--warp-teeth-throat', type=float, default=0.0, help='warp (v6): soft throat / palate shading of the cavity '
                    'on wide-open targets (0 = flat cavity tone, v5; 1 = full)')
    # --- v7 (opt-in; mouth_layers.py): the mouth as an explicit layer stack ---
    ap.add_argument('--warp-compose', choices=['plates', 'layers'], default='plates', help='warp (v7): layers = the mouth is '
                    'composited back to front from explicit RGBA layers: L0 frame, L1 mouth cover (the take\'s mouth erased to '
                    'its skin), L2 back of mouth, L3 lower arch + tongue, L4 upper arch, L5 lips, L6 face; full curved 3D arches '
                    '(needs --warp-teeth rigid); plates = the v5/v6 plate render')
    ap.add_argument('--warp-arch-k', type=float, default=3.2, help='warp (v7): arch curvature, z = k x^2 in IOD units')
    ap.add_argument('--warp-arch-upper', type=float, default=0.40, help='warp (v7): upper arch half arc length, x IOD')
    ap.add_argument('--warp-arch-lower', type=float, default=0.36, help='warp (v7): lower arch half arc length, x IOD')
    ap.add_argument('--warp-arch-overjet', type=float, default=0.03, help='warp (v7): lower arch this far behind the upper, x IOD')
    ap.add_argument('--warp-arch-overbite', type=float, default=0.03, help='warp (v7): the lower incisal edge may rise this far '
                    'above the upper one (behind it) on a closed bite, x IOD (the jaw plan\'s floor)')
    ap.add_argument('--warp-arch-expose', type=float, default=0.03, help='warp (v7): the lower incisal edge sits at most this far '
                    'above the lower lip line (x IOD) at the lower harvest frame\'s jaw')
    ap.add_argument('--warp-teeth-chin-pin', choices=['skin', 'arc', 'off'], default='skin', help='warp (v7, layers): skin = the lower '
                    'arch takes the soft chin\'s own frame-to-frame motion (DIS flow, high-passed onto the jaw plan); arc (v8) = that '
                    'motion split into its slow part (kept) + fast residual (x --warp-chin-pin-gain), projected onto the jaw hinge\'s arc '
                    'as an extra jaw angle (the teeth never translate off the arc), the exposure limit folded in')
    ap.add_argument('--warp-chin-pin-gain', type=float, default=0.4, help='warp (v8, arc pin): share of the fast chin residual the lower teeth follow')
    ap.add_argument('--warp-chin-pin-split', type=float, default=1.5, help='warp (v8, arc pin): frames; gaussian sigma that splits the chin '
                    'pin into its slow part and the fast residual')
    ap.add_argument('--warp-tongue', choices=['on', 'off', 'wedge'], default='on', help='warp (v7, layers): the tongue layer; wedge (v8) = '
                    'a modelled tongue (tip behind the lower incisors + dorsum, rises on narrow vowels, drops / retracts on open ones, '
                    'wet highlight; never above the lower incisal edge on near-closed frames)')
    ap.add_argument('--warp-cover', choices=['on', 'off', 'fill'], default='on', help='warp (v7, layers): the mouth cover layer (L1); fill '
                    '(v8) = an old-lip blob outside the new lips is filled with skin in the warped frame itself (membrane + grain), no seam')
    ap.add_argument('--warp-arch-harvest', type=int, default=1, help='warp (v8): median arch texture over up to N harvest frames in the same head-pose bin')
    ap.add_argument('--warp-arch-harvest-lower', type=int, help='warp (v8): harvest frames for the LOWER arch (default = --warp-arch-harvest)')
    ap.add_argument('--warp-arch-texel', type=float, default=1.0, help='warp (v8): arch texture texels per canonical px (2 = super-resolved; v14: 4 with --warp-arch-mipmap = render 4x, shrink)')
    ap.add_argument('--warp-arch-mipmap', action='store_true', help='warp (v14): area-correct minification of the arch atlas into the frame -- a trilinear mip '
                    'pyramid (INTER_AREA 2x levels) sampled per column-quad by the local minification ratio, instead of raw bilinear. Groove / slot lines '
                    'become soft sub-pixel lines, not hard 1 px aliasing, and a super-resolved atlas (--warp-arch-texel 4) is shrunk with its detail kept. '
                    'Off (default) = the v13 raw-bilinear path, bit for bit')
    ap.add_argument('--warp-arch-synth', action='store_true', help='warp (v8): side teeth beyond the harvest are modelled (rounded crown, '
                    'interproximal line, enamel highlight on the key-light side, premolar / molar widths) instead of column-repeated')
    ap.add_argument('--warp-arch-enamel', type=float, default=0.0, help='warp (v14): S > 0 = the enamel material pass on the crowns -- a '
                    'glossy specular highlight on the key-light side, incisal translucency (the biting edge lighter / warmer), interdental + '
                    'gum-line ambient occlusion, and the hue warmed toward the take\'s yellowish teeth (the v13 render read flat grey / blue). '
                    'S scales a balanced preset (spec 1.0, transl 0.8, ao 0.7, warm 1.0); 0 (default) = off, the v13 enamel bit for bit')
    ap.add_argument('--warp-arch-enamel-mix', default=None, help='warp (v14): override the enamel preset ratio "spec,transl,ao,warm" (each x S)')
    ap.add_argument('--warp-arch-detail', type=float, default=0.0, help='warp (v14): S > 0 = a mean-preserving edge-aware detail pass on the '
                    'crown in atlas space (the L high-frequency residual over a sub-tooth radius, soft-clipped and added back) -- restores a blurry '
                    'shot\'s harvested crispness without re-widening the visible span; 0 (default) = off')
    ap.add_argument('--warp-arch-gums', type=float, default=0.0, help='warp (v15): S > 0 = pink gingiva baked above the upper crowns '
                    'and below the lower crowns (the crowns emerge from gum, not from lip colour). The gum colour is harvested from the take\'s '
                    'own widest-smile frames (a warm-pink synthesized from its inner-mouth colour when no frame shows enough gum); the gum line is '
                    'scalloped at the interdental papillae; the band height scales with the face\'s IPD (the anthro unit, --warp-arch-anthro-unit). '
                    'S is the blend opacity. 0 (default) = off, the v14 band bit for bit')
    ap.add_argument('--warp-tongue-cover', type=float, default=0.0, help='warp (v15, wedge tongue): S > 0 = the tongue rises and rides IN '
                    'FRONT of the lower crowns at mid-open vowel-ish poses (covering them, so the lowers no longer read as a permanent grille), '
                    'while still dropping / retracting to flash the lower teeth on the wide-open teeth-baring frames. S eases the effect; 0 '
                    '(default) = off, the v8 tongue envelope and compositing bit for bit')
    ap.add_argument('--warp-cavity', choices=['off', 'fill'], default='off', help='warp (v17, layers): fill = paint a shaded '
                    'tongue + a dim lower-incisor row (past ~35%% open) into the opening instead of the dark void; off = v16')
    ap.add_argument('--warp-cavity-strength', type=float, default=1.0, help='warp: opacity of the --warp-cavity fill (0..1)')
    ap.add_argument('--warp-lower-real', type=float, default=0.0, help='warp (v16): S > 0 = the lower-teeth realism family -- the lower arch '
                    'is the dim, half-hidden, soft-bottomed sibling of the uppers, not their mirror. Replaces the force-brightened flat incisal '
                    'strip + hard opaque top edge (which read as a grill / braces on wide-open frames) with a softly DARKER, feathered incisal '
                    'gradient; attenuates the lower-arch luminance below the uppers scaled by how wide the mouth is open (wider = dimmer) toward '
                    'a target DERIVED from this take\'s own lower-vs-upper tooth-display ratio (--warp-lower-ratio); casts the cavity / upper-lip '
                    'shadow onto the top of the lowers + an AO ramp so the dark cavity fades softly into them (no hard line); occludes the lower '
                    'crowns more behind the tongue on wide-open frames (keeping the smile-frame flash); and desaturates / cool-shifts the shadowed '
                    'enamel with per-tooth value variation (cool tooth in a warm interior reads deep). S scales a balanced preset (dim 1.0, feather '
                    '1.0, cast 1.0, ao 1.0, occlude 1.0, cool 1.0, relief 1.0); 0 (default) = off, the v15 lowers bit for bit')
    ap.add_argument('--warp-lower-mix', default=None, help='warp (v16): override the --warp-lower-real preset ratio '
                    '"dim,feather,cast,ao,occlude,cool,relief" (each x S)')
    ap.add_argument('--warp-lower-ratio', default='auto', help='warp (v16): this face\'s lower-vs-upper tooth-display tendency driving the dim '
                    'target -- "auto" (default) measures it from the take\'s own frames (some people show mostly top teeth, some a blend, some '
                    'mostly bottom; a top-teeth talker gets dimmer/more-hidden lowers, a bottom-teeth talker gets more), or a float overrides it '
                    '(lo_area / up_area, ~0.2 top-teeth .. ~0.8 bottom-teeth)')
    ap.add_argument('--warp-arch-shade', type=float, default=0.5, help='warp (v7/v8): how dark the arch gets toward the back (0..1)')
    ap.add_argument('--warp-arch-facing', type=float, default=0.0, help='warp (v8): 0..1; a crown seen edge-on (the arch curving away) is darkened by this share')
    ap.add_argument('--warp-tongue-lit', type=float, default=1.0, help='warp (v8, wedge tongue): the tongue\'s light vs the take\'s own tongue pixels')
    ap.add_argument('--warp-arch-desat', type=float, default=0.0, help='warp (v8): how much colour the arch loses toward the back (0..1)')
    ap.add_argument('--warp-arch-tex-out', help='warp (v8): write this shot\'s completed arch textures (npz, IOD units) for other shots of the same character')
    ap.add_argument('--warp-arch-tex-from', help='warp (v8): fallback arch textures (npz from --warp-arch-tex-out) used when this shot\'s own '
                    'harvest is shorter than --warp-arch-tex-min-h')
    ap.add_argument('--warp-arch-tex-min-h', type=float, default=0.05, help='warp (v8): IOD; own harvested tooth height below which the fallback is used')
    ap.add_argument('--warp-arch-tex-share', action='store_true', help='warp (v14): the --warp-arch-tex-from atlas is the SHARED SHARP atlas for a '
                    'face -- used whenever it is sharper (taller exposed teeth) than this shot\'s own harvest, not only when the own harvest is a few '
                    'px tall. The sharpest close-up of a face (its detail) then drives every shot of that face, the blurry mid-shots included')
    ap.add_argument('--warp-lip-rim', default='0.22,0.012', help='warp (v7/v8): the lips\' darker inner rim "strength,width" (width x IOD)')
    ap.add_argument('--warp-lip-wet', type=float, default=0.0, help='warp (v8): a soft 1-2 px wet edge where the lips occlude the teeth (0 = hard cut, v7)')
    ap.add_argument('--warp-layers-smooth', type=float, default=0.0, help='warp (v8): frames; zero-phase ease of the per-frame layer parameters (opening share, tongue pose)')
    ap.add_argument('--warp-layers-debug', help='warp (v7, layers): directory for layers.mp4 / layers.jpg (each layer '
                    'head-stabilised, one panel per layer + the composite)')
    ap.add_argument('--warp-layers-debug-frames', default='', help='warp (v7): frames for layers.jpg, "f1,f2,..."')
    # --- v10c (opt-in; core/coart.py = the planner, audits/naturalness.py = its ruler): coarticulated targets ---
    ap.add_argument('--warp-target', choices=['v9', 'coart'], default='v9', help='warp (v10c): v9 = per-frame step targets (each frame '
                    'aimed at its own viseme band); coart = one planned trajectory: phone targets blended by dominance (closures / '
                    'sibilants strong and narrow, vowels broad, DD/kk/nn/TH weak), short segments undershoot, the plan fitted to '
                    'the take\'s own motion statistics and kept inside the shapes the face makes; closures hard')
    ap.add_argument('--coart-dom-width', type=float, default=80.0, help='warp (v10c, coart): ms; how long a segment\'s dominance takes '
                    'to rise from 10 %% to its plateau before it (and to fall after it), x a per-class factor (closures 0.6, sibilants 0.7, '
                    'weak consonants 0.8, vowels 1.3); two equal neighbours hand over in about twice this')
    ap.add_argument('--coart-undershoot', type=float, default=0.6, help='warp (v10c, coart): 0..1; a segment shorter than its class\'s '
                    'reach time (from the take\'s own transition speed) keeps 1 - U (1 - dur / reach) of its dominance')
    ap.add_argument('--coart-hull-grow', type=float, default=1.05, help='warp (coart): the take\'s own (gap, width) shape hull is '
                    'grown this x about its centre before targets are projected into it (larger = the audio may ask for shapes the take never made)')
    ap.add_argument('--coart-vmax-scale', type=float, default=1.15, help='warp (coart): the planned mouth may move at most this x the '
                    'take\'s own p99 frame-to-frame speed (raise when the take\'s mouth is sluggish: the plan then follows the audio)')
    ap.add_argument('--coart-prior', type=float, default=0.3, help='warp (v10c, coart): 0..1; how far the plan\'s velocity / '
                    'acceleration p95 are pulled toward the take\'s own (0 = no dynamics prior, 1 = the take\'s statistics)')
    ap.add_argument('--coart-anticipate', type=float, default=100.0, help='warp (v10c, coart): ms; lip rounding (ou / oh / RR / CH) '
                    'starts up to this far before its segment')
    ap.add_argument('--coart-jaw-share', default='auto', help='warp (v10c, coart): auto | 0..1; the share of an opening carried by '
                    'the jaw (lower lip + chin; the rest by the upper lip). 1 = v9; auto = the take\'s own share, biased toward the jaw')
    ap.add_argument('--coart-range', type=float, default=1.0, help='warp (v10c, coart): 0..1; how far the planned gap / width '
                    'DISTRIBUTION is matched to the take\'s own (percentile map of the plan\'s active frames onto the take\'s): '
                    'its full range of openings and its share of shut frames, not a compressed middle')
    ap.add_argument('--coart-rest-ms', type=float, default=120.0, help='warp (v10c, coart): ms; an aligner silence at least this long '
                    'is a pause: the mouth eases to the face\'s own rest shape (the take\'s closed mode, at most the library\'s resting '
                    'opening) at the take\'s own closing / opening speed (0 = off)')
    ap.add_argument('--coart-dyn', help='warp (v10c, coart): an external mouth-dynamics CSV on the song timeline (e.g. Audio2Face-3D, '
                    'v10b a2f_cut_25fps.csv: time_s + geo_gap_cm / geo_width_cm), mapped onto the take\'s range by percentile')
    ap.add_argument('--coart-dyn-mix', type=float, default=0.0, help='warp (v10c, coart): 0..1; weight of --coart-dyn in the target '
                    '(0 = off); the aligner\'s closures stay hard and the pauses keep the rest shape')
    # --- v9 (opt-in; audits/frame_gate.py = the checker, core/frame_gate.py = the maker): the per-frame gate ---
    ap.add_argument('--warp-gate', choices=['off', 'report', 'repair'], default='off', help='warp (v9): per-frame maker / checker / '
                    'confirmation. report = every frame checked against its neighbours and the source take (continuity, drift, fold, '
                    'teeth, ghost, seam, colour, outside-mask change, fit to the audio, windowed slosh) and logged; repair = a failing '
                    'frame is re-made from alternatives (parameters, never pixels) and a candidate is kept only when it passes and its '
                    'neighbours still pass; off = v8')
    ap.add_argument('--occluder-mask', help='warp: masks npz from `lsdub occluders` (a mic, a hand in front of the mouth); '
                    'the take\'s own pixels are kept under it, after the gate')
    ap.add_argument('--occluder-grow', type=int, default=3, help='warp: px the occluder mask is grown before the feathered blend')
    ap.add_argument('--warp-gate-rounds', type=int, default=3, help='warp (v9): repair rounds (fixing frame t changes t+1\'s continuity)')
    ap.add_argument('--warp-gate-cands', default='a,b,c,d,e,f,g', help='warp (v9): candidates: a lm_hold, b warp_soft, c donor_hold, '
                    'd teeth_plan, e layers_hold, f param_interp (last resort), g slosh_fit (windows); v11: h layers_off (the plain '
                    'own-pixel warp without the layer stack; tried after a..e, before f)')
    ap.add_argument('--warp-gate-z', type=float, default=3.0, help='warp (v9): robust z (median / MAD of the shot) above which a metric flags')
    ap.add_argument('--warp-gate-soft', action='store_true', help='warp (v9): also repair SOFT frames (default: FAIL only)')
    ap.add_argument('--warp-gate-slosh-w', type=int, default=4, help='warp (v9): half-window (frames each side) of the slosh ruler')
    ap.add_argument('--warp-gate-report', help='warp (v9): write the gate report (gate.json + gate.csv) here (a path ending in .json)')
    ap.add_argument('--warp-gate-unresolved', choices=['best', 'none'], default='best', help='warp (v9): a frame no candidate fixes: '
                    'best = keep the best-scoring candidate when it beats the original (logged UNRESOLVED), none = leave it as round 0')
    # --- v11 (opt-in): the interior seen only through the warped lips; layer debug stabilisation; the registration ruler ---
    ap.add_argument('--warp-interior-clip', action='store_true', help='warp (v11, layers): the cavity / teeth / tongue are only '
                    'seen THROUGH the warped lips: the planned opening is kept where the warped frame has lip pixels above and '
                    'below it (enclosed), feathered <= 1 px; elsewhere the warped face shows')
    ap.add_argument('--warp-cover-corners', type=float, default=0.0, help='warp (v11, layers): the cover grows at the two mouth '
                    'corners with the plan (how far it moves each corner past the take\'s, half the opening increase, 0.15 IOD x '
                    '|sin yaw|) x this; the L6 fill there also takes old-mouth material the warp moved onto the take\'s skin '
                    '(0 = off = v9)')
    ap.add_argument('--warp-yaw-clamp', type=float, default=0.0, help='warp (v12): M > 0 = the PLAN keeps the mouth inside the '
                    'take\'s face under yaw: where the far half\'s planned outer lip comes closer than 0.02 + M IOD to the face\'s edge '
                    '(the face mesh outline + the take\'s own skin pixels), that half is clamped toward the take\'s own outline, eased '
                    'over the clip (mouth_warp.YawClamp); a frame inside the face is untouched (0 = off)')
    ap.add_argument('--warp-gate-h-guard', action='store_true', help='warp (v12): gate candidate h (layers_off) only on near-'
                    'closed frames (planned opening <= the take\'s p5 + --warp-erase-thr) or between frames that show the take\'s own '
                    'interior too; and a candidate that switches the interior\'s look against both neighbours ranks a FAIL lower')
    ap.add_argument('--warp-layers-debug-stab', choices=['pin', 'skull'], default='pin', help='warp (v11): the layer debug '
                    'view stabilised on the skull pose + the upper lip-root pin (pin, v8) or on the skull pose only (skull: '
                    'the pin\'s own motion shows)')
    ap.add_argument('--warp-gate-reg', action='store_true', help='warp (v11): the gate also runs the inter-layer registration '
                    'ruler (lsdub/audits/layer_reg.py: the windowed slosh of the teeth / cavity / tongue offsets vs the lips) '
                    'and reports its windows in gate.json (reported, not repaired)')
    ap.add_argument('--warp-gate-leak-scene', action='store_true', help='warp (v11): the gate\'s leak ruler ignores the scene '
                    '(pixels as dark / tooth-like in the take at the same place, outside its own mouth: hair, nostrils)')
    ap.add_argument('--warp-gate-reg-floor', type=float, default=None, help='warp (v11): the registration ruler\'s window '
                    'energy floor (IOD / frame); default layer_reg.FLOOR')
    # --- v12 reg (opt-in; mouth_layers.LayerCompositor.plan_single): one reference for the mouth's sub-elements ---
    ap.add_argument('--warp-reg', choices=['v11', 'single'], default='v11', help='warp (v12, layers): how the interior layers are '
                    'placed. v11 (default) = each with its own per-frame correction (upper teeth: the lip-root skin pin; lower '
                    'teeth: the chin pin on the hinge arc + the exposure kick; tongue: its vowel pose); single = ONE reference: '
                    'the upper arch on the skull pose only, the lower arch + tongue on the skull pose + one jaw angle about the '
                    'hinge (the planned opening, low-passed), the tongue a slow function of that angle, the cavity the planned '
                    'opening, no pins; gum guards; the lips unchanged')
    ap.add_argument('--warp-reg-jaw-sigma', type=float, default=1.0, help='warp (v12, single): frames; zero-phase low-pass of the '
                    'jaw angle on top of the jaw plan (the jaw is heavy)')
    ap.add_argument('--warp-reg-tongue-sigma', type=float, default=2.0, help='warp (v12, single): frames; the tongue\'s own '
                    'low-pass of the jaw angle (its drop / retraction and the cavity\'s throat follow it)')
    ap.add_argument('--warp-reg-lo-limit', type=float, default=0.09, help='warp (v12, single): IOD; the lower lip never sits more '
                    'than this below the lower incisal edge (the lower arch\'s limit: a hard floor on the jaw angle)')
    ap.add_argument('--warp-arch-build', choices=['v11', 'v12'], default='v11', help='warp (v12, layers): how the arch plates are '
                    'built. v11 (default) = as v8..v11; v12 = the upper plate harvested from the frame with the most EXPOSED upper tooth '
                    'height whose arches are apart (a dark gap below the incisal edge; a smile over clenched teeth put the lower row '
                    'on the upper plate: two stacked rows of incisors), the lower from an open bite too, and each column continued '
                    'beyond its visible tooth as a smooth crown body (no repeated edge or interdental structure), then the gum')
    ap.add_argument('--warp-arch-band-log', action='store_true', help='warp (v12, layers): log the lip-band ruler (frames[i].reg.lipband: '
                    'the plate band right under the planned lips vs their inner rim, Lab dE) without the v12 build (on by itself with '
                    '--warp-arch-build v12); the pixels are unchanged')
    # --- v13 (opt-in; mouth_layers ANTHRO / synth_slots / corridor_*): the teeth's scale vs the head ---
    ap.add_argument('--warp-arch-layout', choices=['harvest', 'anthro'], default='harvest', help='warp (v13, layers): harvest '
                    '(default) = the plate\'s tooth widths are the harvest\'s own (beyond it: 0.95 / 0.85 / 1.15 x its period); '
                    'anthro = the crowns along the arc take the anthropometric widths (upper central 0.1375, lateral 0.105, canine '
                    '0.12, premolars 0.11, molar 0.16; lower 0.087 / 0.095 / 0.11 / 0.11; x the 63 mm IOD = the face\'s inter-pupil '
                    'distance), symmetric about the midline: the harvested run is resampled onto those slots (its interdental lines '
                    'mapped to the slot boundaries; one uniform stretch with fewer than 3 lines; stretches capped x0.6..1.8) and the '
                    'teeth beyond it fill the slots')
    ap.add_argument('--warp-arch-anthro-unit', choices=['ipd', 'iod'], default='ipd', help='warp (v13, anthro): the table\'s IOD '
                    '(63 mm) is the face\'s inter-pupil distance (ipd: the iris centres of the median canonical face; the default) '
                    'or lsdub\'s own IOD (iod: the outer eye corners, ~1.45 x the IPD: the table read literally)')
    ap.add_argument('--warp-arch-corridor', type=float, default=0.0, help='warp (v13, layers): K > 0 = the buccal corridor: per '
                    'frame, the plates beyond +-(0.5 - K) x the planned opening\'s width from its centre fade into the cavity\'s '
                    'dark (smootherstep) to a faint crown at the corners; untouched inside, the lips / face unchanged (0 = off; '
                    'recommended 0.12)')
    ap.add_argument('--warp-arch-corridor-floor', type=float, default=0.25, help='warp (v13, corridor): the share of the crown\'s '
                    'contrast over the cavity that stays at the corners (the back teeth are dark, not black)')
    ap.add_argument('--warp-reg-pin', choices=['none', 'global', 'nose'], default='none', help='warp (v12, single): none = the '
                    'interior on the skull pose alone; global = the upper lip-root skin pin (v6, needs --warp-teeth-anchor skin) '
                    'moves the WHOLE interior (both arches + tongue) as one residual; nose = the same with the NOSE\'s own flow in '
                    'the take (the head motion the skull pose misses, no lip in it): still one reference, following the real face')
    # --- v10a (lsdub/core/vocalisations.py): non-lexical vocalisations / ad-libs (uh uh uh, ah, da da, mm, hey, ...) ---
    ap.add_argument('--vocalisations', choices=['auto', 'skip', 'rest', 'mouth'], default='skip', help='warp (v10a; default '
                    'skip since v11: the platform skips ad-libs): what the mouth does on a vocalisation span (uh uh, ah, da da, mm, '
                    'hey: the track\'s skip fields from `lsdub track skip`, else the same detection now). skip (default) = the '
                    'take\'s own frame (no warp, no layers, no erase), the strength eased zero-phase over --vocal-ease frames at '
                    'each edge; rest = driven to the library\'s rest mouth; mouth = mouthed like words (v9: a v9 command plus '
                    '--vocalisations mouth renders v9 bit for bit); auto = v10a\'s default (skip on a track marked by `lsdub track '
                    'skip`, mouth with a hint on an unmarked one). A track without vocalisations renders the same in every mode')
    ap.add_argument('--skip-words', help='warp (v10a): extra tokens to treat as vocalisations, one per line')
    ap.add_argument('--skip-lyrics', help='warp (v10a): the lyric sheet; words in ( ... ) or after a [skip] tag are skipped')
    ap.add_argument('--vocal-ease', type=int, default=3, help='warp (v10a, skip): frames; zero-phase ease of the warp strength at a span edge')
    ap.add_argument('--vocal-merge-gap', type=float, default=0.75, help='warp (v10a): s; vocalisation spans are merged across a '
                    'silence this short that holds no lexical word')
    ap.add_argument('--vocal-align', help='warp (v10a): the word/phone alignment JSON (default: the track\'s "align" path, from '
                    'the working directory or the track\'s directory)')
    ap.add_argument('--lip-erase', action='store_true', help='morph (v4): the clean plate\'s reference frame has its lips '
                    'erased to its own skin first, so the plate under the paint is bald and a smaller painted mouth never '
                    'shows the take\'s lips around it')
    ap.add_argument('--mouth-scale', choices=['none', 'take'], default='none', help='morph (v4): take = scale the library mouth '
                    'about its centre to the take\'s own rest width (x) and lip thickness (y), measured on its track')
    ap.add_argument('--mouth-open-cap', type=float, default=0.0, help='morph (v4): cap the painted opening at the take\'s own '
                    'p95 opening x this (inter-ocular units); 0 = off')
    ap.add_argument('--mouth', help='cutout mode: fixed mouth region "x,y,w,h" (pixels) or '
                                    'per-frame JSON list of [x,y,w,h]')
    # --- morph-mode options (each one is a A/B-able improvement) ---
    ap.add_argument('--smooth', choices=['none', 'sg', 'euro'], default='sg', help='temporal filter on the warped mouth landmarks')
    ap.add_argument('--weight-sigma', type=float, default=1.0, help='frames; gaussian ease of the viseme weights (0 = off)')
    ap.add_argument('--mask', choices=['hull', 'face'], default='face', help='hull = v4 lip/chin hull; face = face-mesh lower face')
    ap.add_argument('--blend', choices=['alpha', 'laplacian', 'feather', 'poisson', 'mixed'], default='laplacian',
                    help='morph: feather = alpha over a 2x wider-feathered mask; poisson/mixed = cv2.seamlessClone NORMAL/MIXED (gradient domain)')
    ap.add_argument('--relight', choices=['none', 'lowfreq'], default='none', help='morph: replace the paint\'s low-frequency '
                    'luminance GRADIENT (Lab L, sigma = mouth width/4) with the take\'s at the same place: the take\'s light direction carries onto the lips')
    ap.add_argument('--grain', type=float, default=1.0, help='re-add the take\'s grain/sharpness (0 = off)')
    ap.add_argument('--motion-blur', type=float, default=0.5, help='shutter fraction for head-motion blur on the paint (0 = off)')
    ap.add_argument('--no-mic-guard', dest='mic_guard', action='store_false', help='keep the library frame\'s mic pixels')
    ap.add_argument('--skin', choices=['lib', 'take'], default='take', help='skin around the lips: lib = library low-freq + take texture (v4); take = the take itself mesh-warped to the mouth shape')
    ap.add_argument('--tame', default='ih=0.5,ou=0.7,TH=0.4', help='v=g,...: keep g of viseme v, give the rest to --tame-to (morph mode), e.g. ih=0.5,ou=0.7')
    ap.add_argument('--tame-to', default='sil', help='viseme that receives the tamed weight (must be in the track)')
    ap.add_argument('--open-max', type=float, default=0.85, help='cap on the openness scale (morph mode); <1 stops over-wide gapes')
    ap.add_argument('--pose3d', action='store_true', help='morph mode: carry each viseme as a 3D landmark delta, rotated by the '
                    'head pose of the library base vs each frame (rigid fit on the face mesh), instead of a 2D image-plane offset')
    ap.add_argument('--clean-plate', action='store_true', default=True, help='(DEFAULT ON) morph mode: first rebuild the take\'s lower face from its '
                    'most-closed frame (warped into each frame\'s head pose), so the take\'s own open mouth / dropped chin never '
                    'shows under the paint; the paint then moves that closed jaw to each viseme')
    ap.add_argument('--no-clean-plate', dest='clean_plate', action='store_false', help='paint straight over the take (its own lips can show through)')
    ap.add_argument('--plate-debug', help='write the clean plates (no paint) to this mp4')
    ap.add_argument('--transition', choices=['crossfade', 'lipclose'], default='crossfade',
                    help='morph mode: how mouth pixels change between shapes. crossfade = weighted dissolve (v7/v8); lipclose = '
                         'no dissolve: the more-open mouth owns the middle, the more-closed mouth\'s lips sweep in from the top and '
                         'bottom edges toward the mouth line by its weight share')
    ap.add_argument('--jaw-lock', action='store_true', help='morph mode: chin/jaw line ride rigidly with the lower lip '
                    '(lip->chin distance held at the face\'s natural value; the jaw never collapses upward)')
    ap.add_argument('--track-smooth', choices=['k5', 'k9', 'sg9', 'none'], default='k5',
                    help='temporal filter on the take\'s face track (head pose that places the paint); centred, zero-lag')
    ap.add_argument('--lip-ref', help='morph mode: PNG of a reference face whose lip colour every painted mouth is locked to '
                    '(Lab a/b mean+std of the lip tissue, MediaPipe lip landmarks); lightness still follows the take')
    ap.add_argument('--lip-ref-l', type=float, default=0.0, help='with --lip-ref: share (0..1) of the lip LIGHTNESS also taken from the reference')
    ap.add_argument('--anchors', choices=['anch', 'strict', 'strict+iris'], default='anch',
                    help='morph mode: what places the mouth. anch = TPS on 43 upper-face/nose/cheek/oval points (default, v1/v2); '
                         'strict = one rigid 3D skull pose per frame from nose bridge/tip + eye corners only, pose parameters '
                         'smoothed zero-phase, plate + mask edge placed by the same pose (the take\'s own smile/singing never moves it)')
    ap.add_argument('--anchor-smooth', default='sg13', help='--anchors strict: zero-phase filter on the pose parameters: sgNN '
                    '(Savitzky-Golay, order 2, window NN), ffNN (Butterworth filtfilt), none')
    ap.add_argument('--anchor-lam', type=float, default=10.0, help='--anchors strict: TPS regularisation for the residual the '
                    'rigid pose leaves on the anchors (large = near-affine; <= 0 = rigid only)')
    ap.add_argument('--mouth-render', choices=['tps2d', 'mesh3d'], default='tps2d',
                    help='morph mode: how the library mouth is put into the frame. tps2d = 2D frontal texture warped by the '
                         'anchor TPS (default). mesh3d = the library face mesh lifted to 3D (MediaPipe depth + the viseme\'s 3D '
                         'lip shape), projected by one smoothed rigid head pose per frame (strict skull points + forehead), '
                         'texture-mapped per triangle far->near with back-facing triangles culled (implies --pose3d)')
    ap.add_argument('--mesh-smooth', default='sg7', help='--mouth-render mesh3d: zero-phase filter on the rigid pose parameters '
                    '(as --anchor-smooth). sg7 measured best on the mv30 nose-circle and S7 swings; sg13 is ~equal on slow S1')
    ap.add_argument('--mesh-lam', type=float, default=0.0, help='--mouth-render mesh3d: TPS residual on the pose anchors '
                    '(<= 0 = pure rigid projection, the default; large = near-affine correction)')
    ap.add_argument('--mask-out',help='save the per-frame paint masks (npz, half res) for lsdub jerk')
    # --- v12 (opt-in; core/mouth_card.py): the mouth card as a toggle, next to the clean output (which is only read) ---
    ap.add_argument('--annotate', action='store_true', help='v12: after the render also write <out>_annotated.mp4 (the clean '
                    'frames + the lsdub annotate card: word, viseme glyph, openness, 1 s of visemes, weak-alignment marks, heard: '
                    'diffs) and <out>_lyric_check.json / .md + <out>_lyrics_suggested.txt; the clean output is the same either way')
    ap.add_argument('--sync-strip', action='store_true', help='QA: also write <out>_strip.mp4, the vocal waveform scrolling '
                    'under the render past a fixed playhead (on the annotated video when --annotate); clean output unchanged')
    ap.add_argument('--sync-strip-t0', type=float, help='--sync-strip: song time of frame 0 for the audio (default: the track t0; '
                    'set it when the track carries a lip-sync offset)')
    ap.add_argument('--sync-strip-audio', help='--sync-strip: the vocal to draw, read from the track t0 (default: the video audio)')
    ap.add_argument('--annotate-corner', choices=['auto', 'lower-right', 'lower-left', 'band'], default='auto', help='v12: the '
                    "card's place: auto = lower-right, else lower-left, else a band below the picture (never over a face)")
    ap.add_argument('--annotate-faces', help='v12: extra keep-out boxes for the card (host-studio lyric_faces.py JSON)')
    a = ap.parse_args(argv)
    ret = _render(a, ap)
    if a.annotate:
        from . import mouth_card
        mouth_card.after_paint(a)
    if getattr(a, 'sync_strip', False):
        from . import mouth_card
        mouth_card.strip_after_paint(a)
    return ret


def _render(a, ap):
    """the render (v12: split from main so that --annotate runs after it)."""
    if a.mode == 'warp':
        from . import mouth_warp
        return mouth_warp.run(a)
    if not a.lib and a.mode != 'cutout':
        ap.error('--lib is required for --mode legacy|morph')
    if a.mode == 'morph':
        from . import mouth_morph
        return mouth_morph.run(a)
    if a.mode == 'cutout':
        if not a.mouth:
            ap.error('--mode cutout requires --mouth (fixed "x,y,w,h" or per-frame JSON)')
        from ..anchors import mouth_paint_cutout
        return mouth_paint_cutout.run(a)
    T = json.load(open(a.track))
    fr_t = T['frames']
    n = a.frames or len(fr_t)
    json.load(open(f'{a.lib}/prep.json'))
    pb = np.load(f'{a.lib}/base_lm.npy')
    lib, liblm = {}, {}
    for v in set(LIBMAP.get(f['vis'], f['vis']) for f in fr_t) | set(LIBMAP.get(f['vis2'], f['vis2']) for f in fr_t):
        f = f'{a.lib}/{v}.png'
        if not os.path.exists(f):
            v2 = 'sil'
            f = f'{a.lib}/sil.png'
        lib[v] = cv2.imread(f)
        liblm[v] = np.load(f.replace('.png', '_lm.npy'))
    S, fps, found = track(a.video, n)
    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(3)), int(cap.get(4))
    frames, regions, masks = [], [], []
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        p = S[i]
        f = fr_t[i]
        v1, v2 = LIBMAP.get(f['vis'], f['vis']), LIBMAP.get(f['vis2'], f['vis2'])
        M, _ = cv2.estimateAffine2D(   # full affine: absorbs yaw foreshortening
            pb[RIGID].astype(np.float32), p[RIGID].astype(np.float32), method=cv2.LMEDS)
        mix = f['mix'] if v2 != v1 else 0.0
        src = lib[v1] if mix < 0.02 else cv2.addWeighted(lib[v1], 1 - mix, lib[v2], mix, 0)
        if a.open_scale and v1 in OPEN and OPEN[v1] >= 0.4:   # open vowels: the track's openness sets the jaw drop
            src = jaw_scale(src, liblm[v1], float(np.clip(f['open'] / OPEN[v1], a.open_min, 1.1)))
        warped = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        # mask: union of the library mouth (mapped into the frame) and the take's own mouth, lips + the soft chin area
        lm_lib = cv2.transform(liblm[v1][None, LIP_OUT + CHIN].astype(np.float32), M)[0]
        pts = np.vstack([lm_lib, p[LIP_OUT + CHIN]])
        lip_w = np.linalg.norm(p[61] - p[291])
        m = hull_mask(pts, fr.shape, a.grow, lip_w * 0.35)
        ring = (m > 0.05) & (m < 0.6)
        warped = lab_match(warped, fr, ring)
        # skin around the lips: low frequencies from the library (jaw/cheek shape), fine texture from the take (Qwen
        # invents freckles/pores that would shimmer); the lips themselves come from the library in full
        core = np.maximum(hull_mask(lm_lib[:len(LIP_OUT)], fr.shape, 0.12, lip_w * 0.12),     # library lips
                          hull_mask(p[LIP_OUT], fr.shape, 0.18, lip_w * 0.12))[..., None]   # + the take's own lips (no ghost edges)
        sg = max(1.5, lip_w * 0.03)
        lo = cv2.GaussianBlur(warped.astype(np.float32), (0, 0), sg)
        hi = fr.astype(np.float32) - cv2.GaussianBlur(fr.astype(np.float32), (0, 0), sg)
        warped = np.clip(core * warped + (1 - core) * (lo + hi), 0, 255).astype(np.uint8)
        frames.append(fr)
        regions.append(warped)
        masks.append(m)
    cap.release()
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                          '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', a.out], stdin=ff.PIPE)
    N = len(frames)
    for i in range(N):
        reg = regions[i].astype(np.float32)
        if a.temporal > 0 and 0 < i < N - 1:   # 1-2-1 blend of the painted mouth only where neighbours share the viseme
            same = [j for j in (i - 1, i + 1) if fr_t[j]['vis'] == fr_t[i]['vis']]
            if same: reg = reg * (1 - a.temporal) + sum(regions[j].astype(np.float32) for j in same) / len(same) * a.temporal
        m = masks[i][..., None]
        out = (reg * m + frames[i].astype(np.float32) * (1 - m)).astype(np.uint8)
        if a.mask_out: masks[i] = cv2.resize(masks[i], None, fx=0.5, fy=0.5)
        if a.debug_dir and i % 25 == 0:
            os.makedirs(a.debug_dir, exist_ok=True)
            cv2.imwrite(f'{a.debug_dir}/{i:04d}_{fr_t[i]["vis"]}.png', out)
        p.stdin.write(out.tobytes())
    p.stdin.close()
    p.wait()
    if a.mask_out: np.savez_compressed(a.mask_out, masks=(np.array(masks) * 255).astype(np.uint8))
    print(f'face tracked {found:.0%}; {N} frames -> {a.out}')


def paint(video, track, out, mode='warp', show_mouth_card=False, card_corner='auto', watermark=True, lib=None, frames=0,
          args=(), card_faces=None):
    """v12: the one call a ComfyUI node wraps (docs/node-contract.md): `lsdub paint` exactly as the CLI runs it (the render,
    then the invisible watermark unless watermark=False), and with show_mouth_card the mouth card next to it.
    args: further `lsdub paint` flags (a list, e.g. the warp settings). Returns {clean, annotated, lyric_check,
    lyric_check_md, lyrics_suggested, ret}; the card paths are None when show_mouth_card is off."""
    from .. import cli
    from . import mouth_card
    argv = ['paint', '--mode', mode, '--video', video, '--track', track, '--out', out]
    if lib: argv += ['--lib', lib]
    if frames: argv += ['--frames', str(int(frames))]
    argv += [str(x) for x in args]
    if show_mouth_card:
        argv += ['--annotate', '--annotate-corner', card_corner] + (['--annotate-faces', card_faces] if card_faces else [])
    if not watermark: argv.append('--no-watermark')
    ret = cli.main(argv)
    side = dict(zip(('annotated', 'lyric_check', 'lyric_check_md', 'lyrics_suggested'), mouth_card.side_paths(out)))
    res = {k: (p if show_mouth_card and os.path.exists(p) else None) for k, p in side.items()}
    return dict(clean=out if os.path.exists(out) else None, ret=ret, **res)


if __name__ == '__main__':
    main()
