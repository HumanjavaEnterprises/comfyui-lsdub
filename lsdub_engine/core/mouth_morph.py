#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_morph.py: the --mode morph engine of lsdub paint (v5 viseme paint). Called through mouth_paint.py.

What it changes vs the v4 (legacy) paint, one option per problem:
  SNAPPING  the viseme weights are eased in time (--weight-sigma); the frame's mouth SHAPE is the weighted blend of the
            library landmarks (all 478 MediaPipe points), filtered over time (--smooth sg|euro). Every contributing
            library image is mesh-warped (Delaunay piecewise-affine) to that SAME shape before blending, so the lips
            move between visemes instead of dissolving. Jaw openness scales the lower-face displacement, not the pixels.
  SEAMS     pose-aware placement: a thin-plate spline fitted on upper-face/nose/cheek anchors maps library space to the
            frame (absorbs yaw/roll and foreshortening on head turns, which the old single affine could not). The
            paint mask is the face-mesh lower-face region (--mask face), warped with the mouth; --blend laplacian does a
            pyramid blend; per-frame Lab colour stats are smoothed over time; --grain matches sharpness and re-adds grain.
            The mic in the library frame is masked out (--mic-guard), which killed the grey blob on the chin.
  COLLAGE   --motion-blur blurs the paint along the head's motion (shutter fraction). The v2v pass is
            lsdub refine.
"""
import json, os
from ..io import ff
import numpy as np, cv2
from scipy.signal import savgol_filter
from scipy.spatial import Delaunay
from . import mouth_paint as mp_

LIP_OUT = mp_.LIP_OUT
LIBMAP = mp_.LIBMAP
OPEN = mp_.OPEN
LIP_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
# anchors that follow the HEAD, not the mouth: face oval above the jaw angle, nose, cheekbones, eye corners
JAW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 175, 199, 200, 18]
FACE_LOW = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 425, 327, 2, 98, 205]
ANCH = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 21, 54, 103, 67, 109, 162, 127, 234, 93, 132,
        1, 2, 4, 5, 6, 168, 197, 195, 98, 327, 50, 280, 187, 411, 123, 352, 33, 133, 362, 263, 205, 425]


# --anchors strict: skull-rigid points only (nose bridge/tip + eye corners). Measured on mv30 takes: residual after a
# similarity ~2 % IOD and r(lip gap) ~0.0, vs ANCH 5-11 % / r ~0.5 (cheeks, alar base and the oval move with the mouth).
ANCH_STRICT = [168, 6, 197, 195, 5, 4, 1, 33, 133, 362, 263]
IRIS = [468, 473]   # optional (--anchors strict+iris): stable detections but they follow gaze, not the skull
# --mouth-render mesh3d: the rigid pose is fitted on the strict skull points + the upper forehead/brow line (seen above the
# shades even when the eye corners are hidden behind them); pure rigid by default, so the projection foreshortens
MESH_ANCH = ANCH_STRICT + [10, 151, 9, 8, 109, 338, 67, 297, 108, 337]


def umeyama3(src, dst):
    """similarity dst ~= s R src + t (N x D, D = 2 or 3), least squares. Returns (s, R, t)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S_, Vt = np.linalg.svd(b.T @ a)
    D = np.eye(src.shape[1])
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S_) @ D) / (a ** 2).sum())
    return s, R, md - s * R @ ms


def zero_phase(X, mode='sg13'):
    """centred (zero-lag) temporal filter on a parameter series X (T x ...): sgNN = Savitzky-Golay window NN, order 2
    (passes constant velocity AND constant acceleration exactly, i.e. real head moves); ffNN = 2nd-order Butterworth
    filtfilt, cutoff 25/NN Hz-ish (fs=25); none = raw."""
    if mode in (None, 'none') or len(X) < 5: return X.copy()
    w = int(mode[2:])
    T = len(X)
    if mode.startswith('sg'):
        w = min(w, T if T % 2 else T - 1)
        return savgol_filter(X, w, 2, axis=0, mode='interp') if w >= 5 else X.copy()
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, min(0.95, 2.0 / w), output='sos')
    return sosfiltfilt(sos, X, axis=0, padlen=min(3 * w, T - 1))


def plane_lift(xy, P3):
    """give 2D points a z from the least-squares plane z = ax + by + c through the 3D points P3 (box rings etc.)."""
    A = np.hstack([P3[:, :2], np.ones((len(P3), 1))])
    c, *_ = np.linalg.lstsq(A, P3[:, 2], rcond=None)
    return np.hstack([xy, (np.hstack([xy, np.ones((len(xy), 1))]) @ c)[:, None]])


class RigidPlacement:
    """--anchors strict: library space -> frame i as ONE rigid pose per frame (3D similarity on the strict skull points,
    projected orthographically: MediaPipe x/y px + z), not a TPS bent by expression-driven points.
      - fitted on the RAW (unsmoothed) take track; the 7 pose parameters (log s, rotation vector about the shot median,
        t) are smoothed zero-phase; nothing causal touches placement
      - what the rigid model leaves on the anchors (yaw foreshortening, detector bias) may be taken up by a near-affine
        TPS (lam large) on the SAME strict anchors, its targets smoothed with the same filter (lam <= 0: off)
    src3: library points (N x 3), S3: take track (T x N x 3, raw)."""

    def __init__(self, src3, S3, idx=ANCH_STRICT, smooth='sg13', lam=10.0):
        from scipy.spatial.transform import Rotation as Rot
        self.src3, self.S3, self.idx, self.lam = src3, S3, list(idx), lam
        fit = [umeyama3(src3[self.idx], S3[i][self.idx]) for i in range(len(S3))]
        Rm = Rot.from_matrix(np.array([f[1] for f in fit]))
        R0 = Rot.from_matrix(Rm.mean().as_matrix())
        par = np.hstack([np.log([[f[0]] for f in fit]), (R0.inv() * Rm).as_rotvec(), np.array([f[2] for f in fit])])
        self.raw = par
        self.par = zero_phase(par, smooth)
        self.s = np.exp(self.par[:, 0])
        self.R = (R0 * Rot.from_rotvec(self.par[:, 1:4])).as_matrix()
        self.t = self.par[:, 4:]
        self.tm = [None] * len(S3)
        if lam > 0:
            a = np.array([self.rigid(i, src3[self.idx]) for i in range(len(S3))])
            r = zero_phase(S3[:, self.idx, :2] - a, smooth)          # what the rigid pose leaves on the anchors
            self.tm = [tps_fit(a[i], a[i] + r[i], lam=lam) for i in range(len(S3))]

    def rigid3(self, i, X3): return self.s[i] * X3 @ self.R[i].T + self.t[i]
    def rigid(self, i, X3): return self.rigid3(i, X3)[:, :2]

    def apply(self, i, X3):
        """library-space points (N x 3) -> frame i pixels (N x 2)."""
        y = self.rigid(i, X3)
        return y if self.tm[i] is None else tps_apply(self.tm[i], y)

    def to_lib(self, i, Y3):
        """frame-i 3D points back into library space (inverse rigid pose)."""
        return (Y3 - self.t[i]) @ self.R[i] / self.s[i]


def tps_fit(src, dst, lam=1e-3):
    c = src.mean(0)
    s = src.std() + 1e-6
    X = (src - c) / s
    n = len(X)
    d = np.linalg.norm(X[:, None] - X[None], axis=2)
    K = d ** 2 * np.log(d + 1e-12)
    P = np.hstack([np.ones((n, 1)), X])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K + lam * np.eye(n)
    A[:n, n:] = P
    A[n:, :n] = P.T
    B = np.zeros((n + 3, 2))
    B[:n] = dst
    return (np.linalg.solve(A, B), X, c, s)


def tps_apply(m, pts):
    W, X, c, s = m
    Y = (pts - c) / s
    d = np.linalg.norm(Y[:, None] - X[None], axis=2)
    K = d ** 2 * np.log(d + 1e-12)
    return K @ W[:len(X)] + np.hstack([np.ones((len(Y), 1)), Y]) @ W[len(X):]


def one_euro(x, fps, mincut=1.5, beta=0.3, dcut=1.0):
    def alpha(cut):
        r = 2 * np.pi * cut / fps
        return r / (r + 1)
    y = x.copy()
    dx = np.zeros_like(x[0])
    for i in range(1, len(x)):
        dx = dx + alpha(dcut) * ((x[i] - y[i - 1]) * fps - dx)
        cut = mincut + beta * np.abs(dx)
        a = alpha(cut)
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def pwa(src, sp, dp, tris, box):
    """piecewise-affine warp of src (points sp) onto destination points dp, rendered in box=(x0,y0,w,h)."""
    x0, y0, w, h = box
    tid = np.full((h, w), -1, np.int32)
    M = np.zeros((len(tris), 2, 3), np.float32)
    for k, t in enumerate(tris):
        d = (dp[t] - [x0, y0]).astype(np.float32)
        cv2.fillConvexPoly(tid, np.round(d).astype(np.int32), int(k))
        M[k] = cv2.getAffineTransform(d, sp[t].astype(np.float32))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ok = tid >= 0
    Mt = M[np.maximum(tid, 0)]
    mx = Mt[..., 0, 0] * xx + Mt[..., 0, 1] * yy + Mt[..., 0, 2]
    my = Mt[..., 1, 0] * xx + Mt[..., 1, 1] * yy + Mt[..., 1, 2]
    out = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, ok


def tri_area(P, tris):
    """signed 2D area of each triangle (tris: M x 3 indices into P)."""
    a, b, c = P[tris[:, 0], :2], P[tris[:, 1], :2], P[tris[:, 2], :2]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def pwa3d(src, sp, dp, dz, tris, box, ref_sign, min_area=0.5):
    """--mouth-render mesh3d: piecewise-affine texture map of a projected 3D mesh. Like pwa, but triangles are drawn
    far -> near (painter's order on the mean projected depth dz, larger = farther) so the near lip covers the far one, and
    a triangle whose projected winding flips vs the frontal library (ref_sign) or collapses below min_area px^2 is
    back-facing (the far lip corner on a big turn): it is culled, i.e. left to the take."""
    dp = np.asarray(dp)[:, :2]
    x0, y0, w, h = box
    tid = np.full((h, w), -1, np.int32)
    M = np.zeros((len(tris), 2, 3), np.float32)
    ar = tri_area(dp, tris)
    vis = (np.sign(ar) == ref_sign) & (np.abs(ar) >= min_area)
    for k in np.argsort(-dz[tris].mean(1)):
        if not vis[k]: continue
        t = tris[k]
        d = (dp[t] - [x0, y0]).astype(np.float32)
        cv2.fillConvexPoly(tid, np.round(d).astype(np.int32), int(k))
        M[k] = cv2.getAffineTransform(d, sp[t].astype(np.float32))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ok = tid >= 0
    Mt = M[np.maximum(tid, 0)]
    mx = Mt[..., 0, 0] * xx + Mt[..., 0, 1] * yy + Mt[..., 0, 2]
    my = Mt[..., 1, 0] * xx + Mt[..., 1, 1] * yy + Mt[..., 1, 2]
    return cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), ok


def lap_blend(a, b, m, levels=4):
    """Laplacian-pyramid blend of a over b with mask m (float, HxW)."""
    ga, gb, gm = [a.astype(np.float32)], [b.astype(np.float32)], [m.astype(np.float32)]
    for _ in range(levels):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))
    out = ga[-1] * gm[-1][..., None] + gb[-1] * (1 - gm[-1][..., None])
    for i in range(levels - 1, -1, -1):
        sz = (ga[i].shape[1], ga[i].shape[0])
        la = ga[i] - cv2.pyrUp(ga[i + 1], dstsize=sz)
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=sz)
        out = cv2.pyrUp(out, dstsize=sz) + la * gm[i][..., None] + lb * (1 - gm[i][..., None])
    return out


def lm3d(img, l):
    """478 x 3 landmarks of a still (x, y px; z = MediaPipe relative depth * width) or None."""
    import mediapipe as mpp
    r = l.detect(mpp.Image(image_format=mpp.ImageFormat.SRGB, data=np.ascontiguousarray(img[:, :, ::-1])))
    if not r.face_landmarks: return None
    H, W = img.shape[:2]
    return np.array([[q.x * W, q.y * H, q.z * W] for q in r.face_landmarks[0]])


def rigid3(src, dst):
    """Umeyama similarity src->dst (N x 3): returns s*R, the part that rotates a library-space delta into the frame."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    U, S_, Vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    return (np.trace(np.diag(S_) @ D) / (a ** 2).sum()) * R


def mic_mask(imgs, pb):
    """library-space mask of the microphone (grey, low saturation) below the lower lip, union over the library."""
    m = np.zeros(imgs[0].shape[:2], np.uint8)
    y_lip = int(pb[17, 1])
    for im in imgs:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        g = ((hsv[..., 1] < 70) & (hsv[..., 2] > 70)).astype(np.uint8)
        g[:y_lip] = 0
        m |= g
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 2000: keep[lab == i] = 1
    return cv2.GaussianBlur(cv2.dilate(keep, np.ones((25, 25), np.uint8)).astype(np.float32), (0, 0), 8)


def skin_px(im):
    """skin-ish pixels (orange hues, not too dark/grey): keeps earrings, hair, lipstick and mic out of colour stats."""
    h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H_, S_, V_ = h[..., 0], h[..., 1], h[..., 2]
    return (H_ <= 22) & (S_ > 40) & (S_ < 200) & (V_ > 70)


MIC_HSV = lambda im: (lambda h: ((h[..., 1] < 70) & (h[..., 2] > 60)))(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))


class CleanPlate:
    """--clean-plate: erase the take's own mouth before painting.

    The take's lower face is replaced, every frame, by the SAME source pixels: the take's most-closed well-tracked frame
    (the reference), mesh-warped into this frame's head pose (TPS on head-rigid anchors -> closed-mouth landmarks q,
    then a piecewise-affine warp over the lower-face mesh). One source = no flicker; same take = same skin, light, grain.
    A per-frame Lab match (eased in time) absorbs lighting drift; the frame's own mic stays on top. The paint then warps
    THIS plate from q (closed) to each viseme's shape, so the chin follows the new mouth instead of the take's gape."""

    def __init__(self, video, S, ok, n, place=None, erase=False):
        self.place = place   # RigidPlacement (--anchors strict): the plate rides the same smoothed skull pose as the paint
        op = np.linalg.norm(S[:, 13] - S[:, 14], axis=1) / np.linalg.norm(S[:, 10] - S[:, 152], axis=1)
        good = np.where(ok[:n])[0] if ok is not None and ok[:n].any() else np.arange(n)
        # most closed, preferring a frame whose head pose is near the shot's median (fewer warp extremes)
        ctr = np.median(S[good][:, ANCH].mean(1), 0)
        dev = np.linalg.norm(S[good][:, ANCH].mean(1) - ctr, axis=1)
        fh = np.median(np.linalg.norm(S[good, 10] - S[good, 152], axis=1))
        self.r = int(good[np.argmin(op[good] + 0.5 * dev / fh)])
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.r)
        _, self.ref = cap.read()
        cap.release()
        self.sr = S[self.r].copy()
        self.open_ref = float(op[self.r])
        if erase:   # v4 --lip-erase: the plate's reference loses its lips to its own skin (one static bald frame,
            # carried by the pose like the rest of the plate), so no take lip ever shows around a smaller painted mouth
            from . import lip_layers as ll
            self.ref, _ = ll.erase_lips(self.ref, self.sr[:, :2])
        # mesh over the lower face (nose base down, past the jaw line) + a box ring that maps by the same TPS
        lowk = [i for i in range(478) if self.sr[i, 1] > self.sr[2, 1] - 0.15 * fh]
        x0, y0 = self.sr[lowk].min(0) - 0.25 * fh
        x1, y1 = self.sr[lowk].max(0) + 0.25 * fh
        self.ring = np.array([[x, y] for x in np.linspace(x0, x1, 7) for y in (y0, y1)] + [[x, y] for y in np.linspace(y0, y1, 7)[1:-1] for x in (x0, x1)], float)
        self.K = lowk
        self.tris = Delaunay(np.vstack([self.sr[lowk], self.ring])).simplices
        if place is not None:   # reference landmarks + ring, carried back into library space once (via the ref frame's pose)
            sr3 = place.S3[self.r]
            self.sr_lib = place.to_lib(self.r, sr3)
            self.ring_lib = place.to_lib(self.r, plane_lift(self.ring, sr3[lowk]))
        self.stats = []
        # the reference's earrings / hair / collar would ride into the plate when the head pose differs. Mark them ONCE,
        # in reference space (a per-frame test on warped pixels flickers in shadows): clearly foreign colours only,
        # super-saturated reds (earring, not lips or skin) and any green / blue / purple; the lips are allowed back in
        h = cv2.cvtColor(self.ref, cv2.COLOR_BGR2HSV)
        H_, S_ = h[..., 0].astype(int), h[..., 1].astype(int)
        al = (((S_ > 205) & ((H_ <= 8) | (H_ >= 170))) | ((H_ > 25) & (H_ < 165) & (S_ > 70))).astype(np.uint8)
        al[mp_.hull_mask(self.sr[mp_.LIP_OUT], self.ref.shape, 0.30, 3) > 0.3] = 0
        al = cv2.dilate(cv2.morphologyEx(al, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((5, 5), np.uint8))
        self.alien = al.astype(np.float32)

    def closed_lm(self, p, i=None):
        """the reference's (closed) landmarks (and the box ring) carried into this frame's head pose."""
        if self.place is not None and i is not None:
            return self.place.apply(i, self.sr_lib), self.place.apply(i, self.ring_lib)
        tm = tps_fit(self.sr[ANCH], p[ANCH], lam=1e-3)
        return tps_apply(tm, self.sr), tps_apply(tm, self.ring)

    def plate(self, fr, p, i=None):
        """(clean frame, closed landmarks q, plate mask) for frame fr (index i) with take landmarks p."""
        H, W = fr.shape[:2]
        q, qring = self.closed_lm(p, i)
        sp = np.vstack([self.sr[self.K], self.ring])
        dp = np.vstack([q[self.K], qring])
        x0, y0 = np.maximum(0, np.floor(dp.min(0)).astype(int))
        x1, y1 = np.minimum([W, H], np.ceil(dp.max(0)).astype(int))
        box = (x0, y0, x1 - x0, y1 - y0)
        warped, okm = pwa(self.ref, sp, dp, self.tris, box)
        reg = fr[y0:y1, x0:x1]
        lip_w = np.linalg.norm(p[61] - p[291])
        # where the plate goes: the take's lower face AND the reference's, plus the take's whole mouth grown (the gape
        # drops the lower lip and chin below the face mesh), never on the frame's mic or outside the warped mesh
        rel = lambda P: P - [x0, y0]
        m = np.maximum(mp_.hull_mask(rel(p[FACE_LOW]), reg.shape, -0.04, 3), mp_.hull_mask(rel(q[FACE_LOW]), reg.shape, -0.04, 3))
        m = np.maximum(m, mp_.hull_mask(rel(np.vstack([p[mp_.LIP_OUT], p[[152, 148, 377]]])), reg.shape, 0.25, 3))
        m = (m > 0.5).astype(np.uint8)
        mic = MIC_HSV(reg).astype(np.uint8)
        mic[:max(0, int(p[13, 1] - y0))] = 0
        mic = cv2.morphologyEx(mic, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nl, lab_, st_, _ = cv2.connectedComponentsWithStats(mic)
        keep = np.zeros_like(mic)
        for k in range(1, nl):
            if st_[k, cv2.CC_STAT_AREA] > 150: keep[lab_ == k] = 1
        rmic = MIC_HSV(warped).astype(np.uint8)   # the reference's own mic, carried by the warp: never into the frame
        rmic[:max(0, int(q[13, 1] - y0))] = 0
        rmic = cv2.morphologyEx(rmic, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # teeth and pale lips read as "mic" (grey, low saturation): nothing inside either mouth is ever a mic
        lips = np.maximum(mp_.hull_mask(rel(p[mp_.LIP_OUT]), reg.shape, 0.30, 3), mp_.hull_mask(rel(q[mp_.LIP_OUT]), reg.shape, 0.30, 3)) > 0.3
        keep[lips] = 0
        rmic[lips] = 0
        # the reference's earrings / hair / collar, carried in by the warp when the head pose differs: only skin (and the
        # lips) may come from the plate; anything else stays the frame's own pixels
        alien = (pwa(self.alien, sp, dp, self.tris, box)[0] > 0.5).astype(np.uint8)   # static, from the reference
        alien[lips] = 0
        bad = cv2.dilate(np.maximum(np.maximum(keep, rmic), alien), np.ones((5, 5), np.uint8))
        m = m * okm.astype(np.uint8) * (1 - bad)
        feather = max(2.0, lip_w * 0.08)
        mf = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), feather)
        # colour: plate vs frame on the skin band just inside the mask edge (both are skin there)
        band = (mf > 0.15) & (mf < 0.7) & (bad == 0) & skin_px(reg) & skin_px(warped)
        self.stats.append(lab_stats(warped.astype(np.float32), reg.astype(np.float32), band))
        return dict(box=box, warped=warped, m=mf), q

    def finish(self, items):
        """apply eased colour stats and composite; returns the clean frames."""
        st = np.array(self.stats, float)
        good = st[:, 2] > 0
        N = len(st)
        if good.sum() > 3:
            from scipy.ndimage import gaussian_filter1d
            for c in range(9): st[:, c] = gaussian_filter1d(np.interp(np.arange(N), np.where(good)[0], st[good, c]), 3.0, mode='nearest')
        out = []
        for i, (fr, d) in enumerate(items):
            x0, y0, w, h = d['box']
            reg = fr[y0:y0 + h, x0:x0 + w].astype(np.float32)
            pl = apply_lab(d['warped'].astype(np.float32), st[i]) if st[i, 2] > 0 else d['warped'].astype(np.float32)
            # the take's grain on top of the plate's (reference) grain is the same take: keep plate texture as is
            o = fr.copy()
            o[y0:y0 + h, x0:x0 + w] = np.clip(lap_blend(pl, reg, d['m'], 3), 0, 255).astype(np.uint8)
            out.append(o)
        return out


def clean_frames(video, S, n, plate_debug=None, place=None, erase=False):
    """all frames of the take with the clean plate applied, plus the closed landmarks Q (n x 478 x 2)."""
    cp = CleanPlate(video, S, mp_.LAST_OK, n, place, erase)
    cap = cv2.VideoCapture(video)
    items, Q = [], []
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        d, q = cp.plate(fr, S[i], i)
        items.append((fr, d))
        Q.append(q)
    cap.release()
    frames = cp.finish(items)
    print(f'clean plate: reference frame {cp.r} (openness {cp.open_ref:.3f})')
    if plate_debug:
        H, W = frames[0].shape[:2]
        pr = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                               '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', plate_debug], stdin=ff.PIPE)
        for f in frames: pr.stdin.write(f.tobytes())
        pr.stdin.close()
        pr.wait()
    return frames, np.array(Q)


def jaw_lock(L, base):
    """Keep the lower lip -> chin (menton, 152) distance at the base face's natural value in every frame: the chin and
    jaw line translate (along y) with the lower lip, so an opening jaw drops the chin by the same amount and the chin
    never rides up toward the lip. Weight ramps 0 at the lower lip line -> 1 at the chin, feathered out across the
    cheeks/jaw corners, so the lips themselves are untouched. Works on 2D (T x 478 x 2) or 3D landmark tracks."""
    D0 = base[152, 1] - base[17, 1]
    ylip, ych = base[17, 1], base[152, 1]
    cx = base[[61, 291], 0].mean()
    hw = abs(base[291, 0] - base[61, 0]) / 2
    wy = np.clip((base[:, 1] - ylip) / max(1e-6, ych - ylip), 0, 1)
    wx = np.clip(1 - (np.abs(base[:, 0] - cx) - 1.6 * hw) / (1.6 * hw), 0, 1)   # full across the chin, fades by the jaw corners
    w = wy * wy * (3 - 2 * wy) * wx
    sh = (L[:, 17, 1] + D0) - L[:, 152, 1]          # how far the chin must move so lip->chin = D0
    L = L.copy()
    L[:, :, 1] += sh[:, None] * w[None, :]
    return L, w


def run(a):
    MESH = getattr(a, 'mouth_render', 'tps2d') == 'mesh3d'
    if MESH: a.pose3d = True   # mesh3d carries the viseme as a 3D shape; placement is a rigid pose (below)
    T = json.load(open(a.track))
    fr_t = T['frames']
    n = a.frames or len(fr_t)
    pb = np.load(f'{a.lib}/base_lm.npy')
    base = cv2.imread(f'{a.lib}/base.png')
    qmask = cv2.imread(f'{a.lib}/mask.png', 0).astype(np.float32) / 255
    names = sorted(set(LIBMAP.get(f[k], f[k]) for f in fr_t for k in ('vis', 'vis2')))
    lib, liblm = {}, {}
    for v in names:
        f = f'{a.lib}/{v}.png'
        if not os.path.exists(f): f = f'{a.lib}/sil.png'
        lib[v] = cv2.imread(f)
        liblm[v] = np.load(f.replace('.png', '_lm.npy'))
    # outside the region Qwen could change, every library image IS the base: pin those landmarks to the base
    inside = qmask[np.clip(pb[:, 1].astype(int), 0, 1023), np.clip(pb[:, 0].astype(int), 0, 1023)] > 0.02
    for v in names: liblm[v] = np.where(inside[:, None], liblm[v], pb)
    if getattr(a, 'pose3d', False):   # the same viseme shapes as 3D deltas (library camera space), pinned outside Qwen's region
        from . import mouth_library as ml_
        _, l3 = ml_.landmarker()
        pb3 = lm3d(base, l3)
        lib3 = {}
        for v in names:
            q = lm3d(lib[v], l3)
            q = pb3 if q is None else q
            lib3[v] = np.where(inside[:, None], q - q[ANCH].mean(0) + pb3[ANCH].mean(0), pb3)
    vi = {v: i for i, v in enumerate(names)}
    V = len(names)
    # --- viseme weights, eased in time; closures/rounds keep their peak ---
    Wt = np.zeros((n, V))
    for i, f in enumerate(fr_t[:n]):
        v1, v2 = LIBMAP.get(f['vis'], f['vis']), LIBMAP.get(f['vis2'], f['vis2'])
        mix = f['mix'] if v2 != v1 else 0
        Wt[i, vi[v1]] += 1 - mix
        Wt[i, vi[v2]] += mix
    # --tame v=g: keep g of viseme v and hand the rest to --tame-to (Qwen's ih is a toothy grin, ou a full kiss;
    # on singing those read as grimace / pucker pops). Shape AND pixels follow, since both come from the weights.
    for kv in filter(None, (getattr(a, 'tame', '') or '').split(',')):
        v, g = kv.split('=')
        g = float(g)
        to = getattr(a, 'tame_to', 'sil')
        if v in vi and to in vi:
            Wt[:, vi[to]] += Wt[:, vi[v]] * (1 - g)
            Wt[:, vi[v]] *= g
    W0 = Wt.copy()
    if a.weight_sigma > 0:
        from scipy.ndimage import gaussian_filter1d as g1
        Wt = g1(Wt, a.weight_sigma, axis=0, mode='nearest')
        for j, v in enumerate(names):
            if v in ('PP', 'ou', 'FF'): Wt[:, j] = np.maximum(Wt[:, j], W0[:, j])
        Wt /= Wt.sum(1, keepdims=True)
    # openness: scale the lower face's vertical displacement by track open / expected open of the mix
    oexp = Wt @ np.array([OPEN.get(v, 0.0) for v in names])
    op = np.array([f['open'] for f in fr_t[:n]])
    s = np.where(oexp >= 0.35, np.clip(op / np.maximum(oexp, 1e-3), a.open_min, getattr(a, 'open_max', 1.1)), 1.0) if a.open_scale else np.ones(n)
    from scipy.ndimage import gaussian_filter1d as g1
    s = g1(s.astype(float), 1.0, mode='nearest')
    LM = np.stack([liblm[v] for v in names])                  # V x 478 x 2
    L = np.einsum('tv,vkd->tkd', Wt, LM)
    low = pb[:, 1] > pb[13, 1]
    L[:, low, 1] = pb[None, low, 1] + (L[:, low, 1] - pb[None, low, 1]) * s[:, None]
    JL = getattr(a, 'jaw_lock', False)
    L0 = L.copy()
    if a.smooth == 'sg': L = savgol_filter(L, 5, 2, axis=0, mode='nearest')
    elif a.smooth == 'euro': L = one_euro(L, T['fps'])
    # closures are contacts, not targets: frames whose track says PP keep the unfiltered (fully shut) shape
    if 'PP' in vi: L[W0[:, vi['PP']] >= 0.5] = L0[W0[:, vi['PP']] >= 0.5]
    P3D = getattr(a, 'pose3d', False)
    STRICT = getattr(a, 'anchors', 'anch') in ('strict', 'strict+iris') or MESH
    if STRICT and not P3D:   # strict placement needs the library base in 3D (z of every point) even for 2D viseme shapes
        from . import mouth_library as ml_
        _, l3 = ml_.landmarker()
        pb3 = lm3d(base, l3)
        if pb3 is None: raise RuntimeError('--anchors strict: no face found on the library base image')
    if P3D:   # same weights / openness / smoothing / closures on the 3D shape, kept as a delta from the base
        L3 = np.einsum('tv,vkd->tkd', Wt, np.stack([lib3[v] for v in names]))
        L3[:, low] = pb3[None, low] + (L3[:, low] - pb3[None, low]) * s[:, None, None]
        L30 = L3.copy()
        if a.smooth == 'sg': L3 = savgol_filter(L3, 5, 2, axis=0, mode='nearest')
        elif a.smooth == 'euro': L3 = one_euro(L3, T['fps'])
        if 'PP' in vi: L3[W0[:, vi['PP']] >= 0.5] = L30[W0[:, vi['PP']] >= 0.5]
        D3 = L3 - pb3[None]
    # --- take tracking (head pose) ---
    if STRICT:   # raw track for the pose fit (smoothing happens on the pose PARAMETERS); k5 copy for masks/stats
        S3raw, fps, found = mp_.track(a.video, n, z=True, smooth='none')
        S = mp_.smooth_track(S3raw, getattr(a, 'track_smooth', 'k5'))
    else: S, fps, found = mp_.track(a.video, n, z=P3D, smooth=getattr(a, 'track_smooth', 'k5'))
    LREF = lip_ref_stats(cv2.imread(a.lip_ref)) if getattr(a, 'lip_ref', None) else None
    if a.smooth == 'euro' and not STRICT: S = one_euro(S, fps, mincut=3.0, beta=0.5)   # never a causal filter on strict placement
    if P3D or STRICT: S3, S = S, S[..., :2]
    if getattr(a, 'mouth_scale', 'none') == 'take' or getattr(a, 'mouth_open_cap', 0.0) > 0:
        # v4 size calibration: the painted mouth is scaled to THIS take's own mouth (rest width / lip thickness, in
        # inter-ocular units, library base vs the take's track), and its opening is capped at the take's own p95
        from . import lip_layers as ll
        ts = ll.take_size(S)
        bd = ll.mouth_dims(pb)
        if getattr(a, 'mouth_scale', 'none') == 'take':
            sx = float(np.clip(ts['rest'][0] / max(1e-6, bd[0]), 0.6, 1.25))
            sy = float(np.clip(ts['rest'][3] / max(1e-6, bd[3]), 0.6, 1.25))
            L = ll.mouth_scale(L, pb, sx, sy)
            if P3D:
                L3 = ll.mouth_scale(L3, pb3, sx, sy)
                D3 = L3 - pb3[None]
            print(f'mouth scale (take): x {sx:.2f} (rest width {ts["rest"][0]:.3f} vs library {bd[0]:.3f}), '
                  f'y {sy:.2f} (lip thickness {ts["rest"][3]:.3f} vs {bd[3]:.3f}) [inter-ocular units]')
        oc = getattr(a, 'mouth_open_cap', 0.0)
        if oc > 0:
            g = ll.mouth_dims(L)[:, 1]
            lim = oc * ts['p95'][1]
            f = np.where(g > lim, lim / np.maximum(g, 1e-6), 1.0)
            # the opening shrinks as the lower face's vertical displacement from the base shrinks (as the openness scale does)
            L[:, low, 1] = pb[None, low, 1] + (L[:, low, 1] - pb[None, low, 1]) * f[:, None]
            if P3D:
                L3[:, low] = pb3[None, low] + (L3[:, low] - pb3[None, low]) * f[:, None, None]
                D3 = L3 - pb3[None]
            print(f'mouth open cap: <= {lim:.3f} (take p95 x {oc}); {int((f < 1).sum())} frames capped')
    if JL:   # the take's natural lower-lip -> chin length (face-height units), and per-landmark ride weights
        def _u(P):
            ax = P[..., 291, :] - P[..., 61, :]
            u = np.stack([-ax[..., 1], ax[..., 0]], -1)
            u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-6)
            return u * np.sign(u[..., 1:2] + 1e-9)
        _fh = np.linalg.norm(S[:, 152] - S[:, 10], axis=1)
        D_nat = float(np.median(((S[:, 152] - S[:, 17]) * _u(S)).sum(1) / _fh))
        _, jl_w = jaw_lock(pb[None], pb)
    # --- library prescale to ~1.3x frame face size (no aliasing on the way down) ---
    fh_lib = np.linalg.norm(pb[10] - pb[152])
    fh_fr = np.median(np.linalg.norm(S[:, 10] - S[:, 152], axis=1))
    k = min(1.0, 1.3 * fh_fr / fh_lib)
    rs = lambda im: cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    PL = None
    if STRICT:   # one smoothed rigid skull pose per frame places paint, plate and mask edge alike
        base3k = np.hstack([pb, pb3[:, 2:]]) * k     # the library's 2D landmarks + MediaPipe depth of the base
        idx = ANCH_STRICT + (IRIS if a.anchors == 'strict+iris' else [])
        if MESH: idx = (ANCH_STRICT if a.anchors in ('strict', 'strict+iris') else MESH_ANCH) + (IRIS if a.anchors == 'strict+iris' else [])
        lam = getattr(a, 'mesh_lam', 0.0) if MESH else getattr(a, 'anchor_lam', 10.0)
        sm = getattr(a, 'mesh_smooth', 'sg7') if MESH else getattr(a, 'anchor_smooth', 'sg13')
        PL = RigidPlacement(base3k, S3raw[:len(S)], idx, sm, lam)
        ph = np.array([PL.apply(i, base3k[[0, 164]]).mean(0) for i in range(len(S))])
        print(f'anchors strict: {len(idx)} pts; placed philtrum step p95 {np.percentile(np.linalg.norm(np.diff(ph, axis=0), axis=1), 95):.2f} px')
    CP = getattr(a, 'clean_plate', False)
    if CP: clean, Q = clean_frames(a.video, S, n, getattr(a, 'plate_debug', None), PL, getattr(a, 'lip_erase', False))
    elif getattr(a, 'lip_erase', False): print('--lip-erase needs the clean plate (on by default); ignored with --no-clean-plate')
    libk = {v: rs(lib[v]) for v in names}
    pbk, LMk, Lk = pb * k, {v: liblm[v] * k for v in names}, L * k
    mic = rs(mic_mask([base] + list(lib.values()), pb)) if a.mic_guard else None
    # paint mask in library space: Qwen's region, grown + feathered so its edge sits where library == base
    pm = cv2.dilate((qmask > 0.4).astype(np.uint8), np.ones((61, 61), np.uint8)).astype(np.float32)
    pm = rs(cv2.GaussianBlur(pm, (0, 0), 28) * (cv2.GaussianBlur(pm, (0, 0), 28) > 0.02))
    if mic is not None: pm = pm * (1 - np.clip(mic * 1.3, 0, 1))
    # mesh: all landmarks in/near the mask + a box of outer points
    ys, xs = np.where(pm > 0.01)
    x0, x1, y0, y1 = xs.min() - 30, xs.max() + 30, ys.min() - 30, ys.max() + 30
    if JL:   # the chin + jaw line must be IN the mesh so they can ride with the lower lip; box runs past the chin into the neck
        jx = pbk[JAW, 0]
        x0, x1 = min(x0, jx.min() - 30), max(x1, jx.max() + 30)
        y1 = max(y1, pbk[152, 1] + 0.35 * (pbk[152, 1] - pbk[17, 1]) + 30)
    K = [i for i in range(478) if x0 <= pbk[i, 0] <= x1 and y0 <= pbk[i, 1] <= y1]
    ring = np.array([[x, y] for x in np.linspace(x0, x1, 6) for y in (y0, y1)] + [[x, y] for y in np.linspace(y0, y1, 6)[1:-1] for x in (x0, x1)], float)
    tris = Delaunay(np.vstack([pbk[K], ring])).simplices
    ring3 = plane_lift(ring, base3k[K]) if PL is not None else None
    ref_sign = np.sign(tri_area(np.vstack([pbk[K], ring]), tris)) if MESH else None
    LC = getattr(a, 'transition', 'crossfade') == 'lipclose'
    if LC:
        np.gradient(g1(Wt, 1.0, axis=0, mode='nearest'), axis=0)


        gapv = np.array([liblm[v][14, 1] - liblm[v][13, 1] for v in names])
        gapf = L[:, 14, 1] - L[:, 13, 1]

        def lc_pair(i):
            # (centre mouth, edge mouth or None, edge share 0..1). Never a two-mouth dissolve (it ghosts the teeth/gap
            # line in from the centre). Of the two strongest mouths, the more OPEN one owns the middle (teeth, cavity)
            # and the more CLOSED one's lips come in from the top and bottom edges toward the mouth line by its weight
            # share: as a mouth closes the lips sweep in top-down / bottom-up and meet at the line, like real lips.
            o = np.argsort(-Wt[i])
            a_, b_ = int(o[0]), int(o[1])
            if Wt[i, b_] < 0.03: return a_, None, 0.0
            op_, cl_ = (a_, b_) if gapv[a_] >= gapv[b_] else (b_, a_)
            t = float(Wt[i, cl_] / (Wt[i, cl_] + Wt[i, op_]))
            # a thin gap can't show the open mouth's teeth (squashed they read as a jagged line): the lips meet instead
            r = float(gapf[i] / max(1e-6, gapv[op_]))
            q = float(np.clip((0.45 - r) / 0.20, 0, 1))
            return op_, cl_, max(t, q * q * (3 - 2 * q))

        def lc_wipe(P, shape, t, soft=0.25):
            # 0 at the mouth line, 1 at the outer lip edges (separately above / below, along the face's own vertical);
            # the incoming mouth shows where that distance exceeds a threshold that sweeps from the edges to the line
            c = P[[13, 14]].mean(0)
            ax = P[291] - P[61]
            u = np.array([-ax[1], ax[0]]) / (np.linalg.norm(ax) + 1e-6)
            if u[1] < 0: u = -u
            yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
            v = (xx - c[0]) * u[0] + (yy - c[1]) * u[1]
            top = max(2.0, -float((P[0] - c) @ u))
            bot = max(2.0, float((P[17] - c) @ u))
            d = np.minimum(np.where(v < 0, -v / top, v / bot), 1.0)
            thr = (1 + soft) - t * (1 + 2 * soft)
            w = np.clip((d - thr) / soft, 0, 1)
            return (w * w * (3 - 2 * w)).astype(np.float32)
    cap = cv2.VideoCapture(a.video)
    Wd, Hd = int(cap.get(3)), int(cap.get(4))
    outs, _masks, stats, items = [], [], [], []
    centers = []
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        fr0 = fr
        if CP:
            if i >= len(clean): break
            fr = clean[i]
        p = S[i]
        ps = Q[i] if CP else p   # ps = where the (plate's) lips are: the skin warp starts from there
        if PL is not None:   # rigid skull pose (+ near-affine residual): the take's own expression never moves the mouth
            X3 = base3k + D3[i] * k if P3D else np.hstack([Lk[i], base3k[:, 2:]])
            Lf = PL.apply(i, X3)
            dst = np.vstack([Lf[K], PL.apply(i, ring3)])
            dz = np.concatenate([PL.rigid3(i, X3[K])[:, 2], PL.rigid3(i, ring3)[:, 2]])
        elif P3D:   # base mouth placed by the TPS, viseme delta rotated by the library->frame head pose (3D rigid fit)
            tm = tps_fit(pbk[ANCH], p[ANCH], lam=1e-3)
            sR = rigid3(pb3[ANCH] * k, S3[i][ANCH])
            dk = (D3[i] * k) @ sR.T
            Lf = tps_apply(tm, pbk) + dk[:, :2]
            dst = np.vstack([Lf[K], tps_apply(tm, ring)])
        else:
            tm = tps_fit(pbk[ANCH], p[ANCH], lam=1e-3)
            dst = tps_apply(tm, np.vstack([Lk[i][K], ring]))
            Lf = tps_apply(tm, Lk[i])
        if JL:   # chin/jaw ride with the lower lip: lip->chin held at the take's natural length (never collapses upward)
            uu = _u(Lf)
            fh_i = np.linalg.norm(p[152] - p[10])
            sh = (float((Lf[17] - Lf[152]) @ uu) + D_nat * fh_i)
            Lf = Lf + sh * jl_w[:, None] * uu[None]
            dst = np.vstack([Lf[K], dst[len(K):]])
        bx0, by0 = np.floor(dst.min(0)).astype(int) - 4
        bx1, by1 = np.ceil(dst.max(0)).astype(int) + 4
        bx0, by0 = max(0, bx0), max(0, by0)
        bx1, by1 = min(Wd, bx1), min(Hd, by1)
        box = (bx0, by0, bx1 - bx0, by1 - by0)
        acc = np.zeros((box[3], box[2], 3), np.float32)
        okm = None
        tex = (lambda src_, sp_: pwa3d(src_, sp_, dst, dz, tris, box, ref_sign)) if MESH else (lambda src_, sp_: pwa(src_, sp_, dst, tris, box))
        if LC:   # lip-close: no pixel dissolve of two mouths (that ghosts the teeth/gap line in from the centre);
            # the dominant mouth's pixels ride the blended shape, and a changing mouth is revealed from the outer
            # lip edges (top down, bottom up) into the mouth line, like real lips meeting
            a_, b_, m_ = lc_pair(i)
            wa, okm = tex(libk[names[a_]], np.vstack([LMk[names[a_]][K], ring]))
            acc = wa.astype(np.float32)
            if b_ is not None and m_ > 1e-3:
                wb, _ = tex(libk[names[b_]], np.vstack([LMk[names[b_]][K], ring]))
                wm = lc_wipe(Lf - [bx0, by0], acc.shape[:2], m_)[..., None]
                acc = acc * (1 - wm) + wb.astype(np.float32) * wm
        else:
            for j, v in enumerate(names):
                if Wt[i, j] < 0.03: continue
                sp = np.vstack([LMk[v][K], ring])
                wv, okv = tex(libk[v], sp)
                acc += Wt[i, j] * wv.astype(np.float32)
                okm = okv
            acc /= max(1e-6, Wt[i][Wt[i] >= 0.03].sum())
        # mask follows the mouth: library-space paint mask warped base-shape -> this frame's shape
        sp0 = np.vstack([pbk[K], ring])
        mw, _ = tex(pm, sp0)
        mw = mw * okm
        if a.mask == 'hull':
            m_full = mp_.hull_mask(np.vstack([Lf[LIP_OUT + mp_.CHIN], p[LIP_OUT + mp_.CHIN]]), fr.shape, a.grow, np.linalg.norm(p[61] - p[291]) * 0.35)
            mw = m_full[by0:by1, bx0:bx1]
        reg = fr[by0:by1, bx0:bx1].astype(np.float32)
        reg0 = fr0[by0:by1, bx0:bx1].astype(np.float32)
        lip_w = np.linalg.norm(p[61] - p[291])
        skin = None
        if a.skin == 'take':
            # the take's OWN lower face, mesh-warped from its landmarks to the target mouth shape: jaw/chin/cheeks move
            # with the viseme but keep the take's skin, light and grain; identity at the outer ring -> no seam
            skin, _ = pwa(fr, np.vstack([ps[K], dst[len(K):]]), dst, tris, box)
            skin = np.where(okm[..., None], skin, fr[by0:by1, bx0:bx1]).astype(np.float32)
            core = mp_.hull_mask(Lf[LIP_OUT] - [bx0, by0], reg.shape, 0.10, lip_w * 0.10)
            # colour ref = lip TISSUE only (outer-lip hull minus inner-lip hull), library vs the take's own lips
            ann = lambda P: (mp_.hull_mask(P[LIP_OUT] - [bx0, by0], reg.shape, -0.03, 3) > 0.5) & ~(mp_.hull_mask(P[LIP_IN] - [bx0, by0], reg.shape, 0.08, 3) > 0.3)
            if LREF is not None:   # --lip-ref: one lip colour for the character across shots
                stats.append(lab_stats_lipref(acc, ann(Lf), LREF, reg0, ann(p), getattr(a, 'lip_ref_l', 0.0)))
            else: stats.append(lab_stats2(acc, ann(Lf), reg0, ann(p)))   # lip colour: the TAKE's own lips (truth), not the plate's
        else:
            core = np.maximum(mp_.hull_mask(Lf[LIP_OUT] - [bx0, by0], reg.shape, 0.14, lip_w * 0.12),
                              mp_.hull_mask(p[LIP_OUT] - [bx0, by0], reg.shape, 0.18, lip_w * 0.12))
            stats.append(lab_stats(acc, reg, (mw > 0.05) & (mw < 0.6) & (core < 0.2)))
        if a.mic_guard:   # the TAKE's mic (grey, below the lower lip) stays on top: never painted or warped
            hsv = cv2.cvtColor(fr[by0:by1, bx0:bx1], cv2.COLOR_BGR2HSV)
            g = ((hsv[..., 1] < 70) & (hsv[..., 2] > 60)).astype(np.uint8)
            g[:max(0, int(p[17, 1] - by0))] = 0
            g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            nl, lab_, st_, _ = cv2.connectedComponentsWithStats(g)
            keep = np.zeros_like(g)
            for q in range(1, nl):
                if st_[q, cv2.CC_STAT_AREA] > 150: keep[lab_ == q] = 1
            km = cv2.dilate(keep, np.ones((7, 7), np.uint8)).astype(np.float32)
            if skin is not None:   # also where the warp would carry mic pixels somewhere else (ghost mic)
                full = np.zeros(fr.shape[:2], np.float32)
                full[by0:by1, bx0:bx1] = km
                km = np.maximum(km, pwa(full, np.vstack([ps[K], dst[len(K):]]), dst, tris, box)[0])
            mw = mw * (1 - np.clip(cv2.GaussianBlur(km, (0, 0), 2.5) * 1.5, 0, 1))
        if a.mask == 'face':   # never outside the take's lower face: jaw line below, nose base above (glasses, hair, mic safe)
            # strict: the edge comes from the library's lower face placed by the rigid pose (never the take's sliding
            # silhouette / cheek points, whose crawl on the shaded side read as a pasted-on edge)
            lim = mp_.hull_mask((PL.apply(i, base3k[FACE_LOW]) if PL is not None else p[FACE_LOW]) - [bx0, by0], reg.shape, -0.10, lip_w * 0.30)
            if MESH:   # on turns the projected library face can overhang the real silhouette (hair/neck behind the
                # far cheek): the take's own lower-face outline is the occluding contour there, so the paint stays inside it
                lim = np.minimum(lim, mp_.hull_mask(p[FACE_LOW] - [bx0, by0], reg.shape, -0.04, lip_w * 0.15))
            mw = np.minimum(mw, lim)
        if JL:   # + the jaw band between the source and target chin lines, so the moved chin is actually shown
            jm = mp_.hull_mask(np.vstack([Lf[JAW + [17]], ps[JAW + [17]]]) - [bx0, by0], reg.shape, 0.06, lip_w * 0.25)
            mw = np.maximum(mw, jm * (okm > 0))
        items.append((box, acc, mw, core, skin))
        centers.append(Lf[[13, 14, 61, 291]].mean(0))
        outs.append(fr)
    cap.release()
    N = len(outs)
    from scipy.ndimage import gaussian_filter1d
    st = np.array(stats, float)
    good = st[:, 2] > 0
    if good.sum() > 3:   # colour stats eased over ~2 frames: no per-frame colour flicker
        for c in range(9): st[:, c] = gaussian_filter1d(np.interp(np.arange(N), np.where(good)[0], st[good, c]), 2.0, mode='nearest')
    C = np.array(centers)
    vel = np.gradient(C, axis=0)
    proc = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{Wd}x{Hd}', '-r', str(fps), '-i', '-',
                             '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', a.out], stdin=ff.PIPE)
    rng = np.random.default_rng(7)
    mk = []
    for i in range(N):
        (bx0, by0, bw, bh), acc, mw, core, skin = items[i]
        fr = outs[i]
        reg = fr[by0:by1_(by0, bh), bx0:bx0 + bw].astype(np.float32)
        acc = apply_lab(acc, st[i])
        sg = max(1.5, np.linalg.norm(S[i][61] - S[i][291]) * 0.03)
        if a.grain > 0:   # sharpness: match the take's lip-region Laplacian energy, outside the lips keep the take's texture
            lt, lp = cv2.Laplacian(cv2.cvtColor(reg.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F), cv2.Laplacian(cv2.cvtColor(np.clip(acc, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F)
            sel = core > 0.5
            if sel.sum() > 30:
                r = lt[sel].std() / (lp[sel].std() + 1e-6)
                if r < 0.85: acc = cv2.GaussianBlur(acc, (0, 0), float(np.clip((1 / r - 1) * 0.8, 0.3, 1.5)))
                elif r > 1.2: acc = cv2.addWeighted(acc, 1 + min(0.8, (r - 1)) * 0.6, cv2.GaussianBlur(acc, (0, 0), 1.2), -min(0.8, (r - 1)) * 0.6, 0)
        c3 = core[..., None]
        if skin is not None: paint = c3 * acc + (1 - c3) * skin
        else:
            lo = cv2.GaussianBlur(acc, (0, 0), sg)
            hi = reg - cv2.GaussianBlur(reg, (0, 0), sg)
            paint = c3 * acc + (1 - c3) * (lo + hi)
        if a.grain > 0:   # re-add grain on the library lips (fresh, luminance only, std from the take's flat skin)
            ring_ = (mw > 0.3) & (core < 0.1)
            if ring_.sum() > 30:
                gs = float(np.std((reg - cv2.GaussianBlur(reg, (0, 0), 0.8))[ring_])) * 0.7 * a.grain
                paint += (rng.normal(0, gs, paint.shape[:2])[..., None] * c3)
        if getattr(a, 'relight', 'none') == 'lowfreq':
            paint = relight_lowfreq(paint, reg, mw, np.linalg.norm(S[i][61] - S[i][291]) / 4)
        m = mw.astype(np.float32)
        if a.motion_blur > 0:
            v = vel[i]
            L_ = float(np.linalg.norm(v) * a.motion_blur)
            if L_ > 1.0:
                ksz = int(np.ceil(L_)) * 2 + 1
                ker = np.zeros((ksz, ksz), np.float32)
                c = ksz // 2
                d = v / (np.linalg.norm(v) + 1e-6)
                for t in np.linspace(-L_ / 2, L_ / 2, ksz * 2): ker[int(round(c + t * d[1])), int(round(c + t * d[0]))] = 1
                ker /= ker.sum()
                paint = cv2.filter2D(paint, -1, ker, borderType=cv2.BORDER_REFLECT)
                m = cv2.filter2D(m, -1, ker)
        out = fr.copy()
        if a.blend == 'laplacian': blended = lap_blend(paint, reg, m, 3)
        elif a.blend == 'feather':
            mf = cv2.GaussianBlur(m, (0, 0), max(2.0, np.linalg.norm(S[i][61] - S[i][291]) * 0.12))
            mf = np.minimum(mf * 1.15, 1.0) * (m > 0.01)
            m = cv2.GaussianBlur(mf, (0, 0), 1.5)
            blended = paint * m[..., None] + reg * (1 - m[..., None])
        elif a.blend in ('poisson', 'mixed'): blended = seamless(paint, reg, m, a.blend == 'mixed')
        else: blended = paint * m[..., None] + reg * (1 - m[..., None])
        out[by0:by0 + bh, bx0:bx0 + bw] = np.clip(blended, 0, 255).astype(np.uint8)
        if a.mask_out:
            full = np.zeros((Hd, Wd), np.float32)
            full[by0:by0 + bh, bx0:bx0 + bw] = m
            mk.append(cv2.resize(full, None, fx=0.5, fy=0.5))
        if a.debug_dir and i % 25 == 0:
            os.makedirs(a.debug_dir, exist_ok=True)
            cv2.imwrite(f'{a.debug_dir}/{i:04d}.png', out)
        proc.stdin.write(out.tobytes())
    proc.stdin.close()
    proc.wait()
    if a.mask_out: np.savez_compressed(a.mask_out, masks=(np.array(mk) * 255).astype(np.uint8))
    print(f'face tracked {found:.0%}; {N} frames -> {a.out}')


def by1_(by0, bh): return by0 + bh


def lab_stats(src, ref, w):
    if w.sum() < 50: return [0.0] * 9   # g=0 -> skip
    a = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    out = []
    for c in range(3):
        s_, r_ = a[..., c][w], b[..., c][w]
        out += [s_.mean(), r_.mean(), np.clip(r_.std() / (s_.std() + 1e-6), 0.7, 1.4) ** 0.5]
    return out


def apply_lab(src, st):
    a = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        sm, rm, g = st[3 * c:3 * c + 3]
        if g == 0: continue
        a[..., c] = (a[..., c] - sm) * g + rm
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def lab_stats2(src, ws, ref, wr):
    """colour stats: src region ws matched to ref region wr (library lips -> the take's own lips)."""
    if ws.sum() < 30 or wr.sum() < 30: return [0.0] * 9
    a = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    out = []
    for c in range(3):
        s_, r_ = a[..., c][ws], b[..., c][wr]
        # lightness: match the mean of the brighter (lip-skin) half only; open mouths have dark interiors
        out += [s_.mean(), r_.mean(), np.clip(r_.std() / (s_.std() + 1e-6), 0.8, 1.25) ** 0.5]
    return out


def lip_tissue_mask(P, shape):
    """bool mask of the lip tissue (outer-lip hull minus the inner-lip hull) for 478 landmarks P (x, y px)."""
    return (mp_.hull_mask(P[LIP_OUT], shape, -0.03, 3) > 0.5) & ~(mp_.hull_mask(P[LIP_IN], shape, 0.08, 3) > 0.3)


def lip_ref_stats(img, P=None):
    """--lip-ref: Lab (mean, std) per channel of the lip tissue of a reference face image -> 3 x 2 array.
    P = its landmarks (detected with MediaPipe when None)."""
    if img is None: raise RuntimeError('--lip-ref: cannot read the image')
    if P is None:
        from . import mouth_library as ml_
        P = ml_.landmarks(img)
        if P is None: raise RuntimeError('--lip-ref: no face found in the reference image')
        P = np.asarray(P)[:, :2]
    w = lip_tissue_mask(P, img.shape)
    if w.sum() < 30: raise RuntimeError('--lip-ref: lip region too small')
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.array([[lab[..., c][w].mean(), lab[..., c][w].std()] for c in range(3)])


def lab_stats_lipref(src, ws, ref_stats, take, wr, l_share=0.0):
    """--lip-ref colour stats (same layout as lab_stats2: src mean, target mean, gain per L/a/b): the painted lips' a/b
    (lipstick hue/saturation) go to the REFERENCE lip colour's mean and spread; L keeps the take's own lip lightness
    (this shot's lighting), blended toward the reference by l_share."""
    if ws.sum() < 30: return [0.0] * 9
    a = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    have_take = wr is not None and wr.sum() >= 30
    b = cv2.cvtColor(np.clip(take, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32) if have_take else None
    out = []
    for c in range(3):
        s_ = a[..., c][ws]
        rm, rs = ref_stats[c]
        if c == 0 and have_take:
            t_ = b[..., 0][wr]
            rm = (1 - l_share) * t_.mean() + l_share * rm
            rs = (1 - l_share) * t_.std() + l_share * rs
        out += [s_.mean(), rm, np.clip(rs / (s_.std() + 1e-6), 0.8, 1.25) ** 0.5]
    return out


def relight_lowfreq(paint, take, m, sigma):
    """--relight lowfreq: frequency-separate the paint in Lab L; keep its high frequencies (lip detail) and its overall
    level (set by the colour match), replace the low-frequency luminance GRADIENT with the take's at the same place
    (offset = blur(L_take) - blur(L_paint), minus its mask-weighted mean), so the take's light direction carries onto
    the lips. paint/take: float BGR same shape; m: float mask (weights the mean); sigma in px."""
    sigma = float(max(1.5, sigma))
    lp = cv2.cvtColor(np.clip(paint, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lt = cv2.cvtColor(np.clip(take, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    off = cv2.GaussianBlur(lt[..., 0], (0, 0), sigma) - cv2.GaussianBlur(lp[..., 0], (0, 0), sigma)
    w = np.clip(m, 0, 1).astype(np.float32)
    off -= float((off * w).sum() / max(1e-6, w.sum())) if w.sum() > 1 else float(off.mean())
    lp[..., 0] = np.clip(lp[..., 0] + off, 0, 255)
    return cv2.cvtColor(lp.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def seamless(paint, reg, m, mixed=False):
    """gradient-domain composite of paint into reg inside mask m (cv2.seamlessClone NORMAL or MIXED)."""
    mk = (m > 0.1).astype(np.uint8) * 255
    mk[:2] = 0
    mk[-2:] = 0
    mk[:, :2] = 0
    mk[:, -2:] = 0
    if mk.sum() < 255 * 30: return paint * m[..., None] + reg * (1 - m[..., None])
    ys, xs = np.where(mk > 0)
    c = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))
    out = cv2.seamlessClone(np.clip(paint, 0, 255).astype(np.uint8), np.clip(reg, 0, 255).astype(np.uint8), mk, c,
                            cv2.MIXED_CLONE if mixed else cv2.NORMAL_CLONE).astype(np.float32)
    w = cv2.GaussianBlur((mk > 0).astype(np.float32), (0, 0), 1.5)[..., None]   # soften the clone's hard mask edge
    return out * w + reg * (1 - w)
