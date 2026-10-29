# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""skull_lock.py — put the mouth back on the skull: one midline, mouth line parallel to the pupils, upper lip rigid with
the nose, lips in order, the mouth wrapped on the face's own cylinder.

Every target is the TAKE's own geometry on the same frame (same head pose), from face_constraints (docs/FACE_CONSTRAINTS.md):
  C3  philtrum (0) height held within 0.75 x the character's own upper-lip tolerance of its REST height (skull frame)
  C1  mouth centre's lateral offset from the nose midline -> the take's (within 1 % of mouth width)
  C2  mouth-corner line angle vs the pupil line -> the take's (within 1 deg); the inner opening shares that axis
  C15 near/far half-width ratio (philtrum -> corners) -> within 7 % of the take's (yaw foreshortening)
  C10 inner lips never cross (13 above 14, 0 above 17)
  --jaw-rules adds C4 lip/jaw ratio (own band), C5 corners, C7 speed limits, C8 chin yaw (always CHECKED either way)
Each correction is capped (CAP); anything bigger is treated as a tracker failure, not anatomy.
Corrections are control-point displacements (nose, eyes, cheekbones and the face oval pinned at zero), smoothed over
time, spread by a thin-plate RBF and applied with a feathered lower-face mask. Small by construction (a few px).

  lsdub skull-lock --video in.mp4 --take take.mp4 --frames N --out out.mp4 [--json corrections.json]"""
import argparse, json, os
from ..io import ff
import cv2, numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter1d
from .mouth_paint import track
from . import face_constraints as fc

UPPER = [0, 37, 267, 39, 269, 40, 270, 13, 82, 312, 81, 311]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LOWER_IN = [14, 87, 178, 88, 95, 317, 402, 318, 324]
INNER_RING = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
CAP = dict(upper_mm=3.0, mid_mm=2.5, axis_deg=4.0, chin_mm=3.0, total_mm=4.0)    # bigger "errors" are tracker failures, not anatomy
CHIN = [152, 148, 377, 176, 400, 199, 200, 18, 175, 171, 396]
SIDE_A = [61, 146, 91, 185, 40, 78, 95, 191]          # corner A side
SIDE_B = [291, 375, 321, 409, 270, 308, 324, 415]     # corner B side
PIN = [1, 2, 4, 5, 94, 98, 327, 195, 197, 6, 168, 33, 133, 263, 362, 468, 473, 50, 280, 123, 352, 187, 411,
       234, 454, 93, 323, 132, 361, 58, 288, 172, 397, 136, 365, 116, 345, 147, 376]
CTRL = sorted(set(LIPS + CHIN))


def rot(v, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return v @ np.array([[c, -s], [s, c]]).T


def corrections(Po, Pt, ipd_mm, L=fc.LIMITS, cal=None, jaw_rules=False):
    """per frame: target displacement (mm, skull frame of the OUTPUT frame) for every CTRL point.
    cal (face_constraints.calibrate of the take) adds the jaw rules: C4 lower lip rides the jaw (the character's own
    ratio band), C5 corners ~half the chin, C7 chin / lower-lip speed limits."""
    n = min(len(Po), len(Pt))
    D = np.zeros((n, len(CTRL), 2))
    ix = {k: i for i, k in enumerate(CTRL)}
    log = []
    XS, SO = [], []
    for t in range(n):
        po, pt = Po[t], Pt[t]
        So = fc.to_skull(po, CTRL, ipd_mm)
        fc.to_skull(pt, CTRL, ipd_mm)
        X = So.copy()
        rec = {}
        # C3 upper lip rigid with the skull: its midline height pulled to within half the tolerance of the character's
        # REST upper lip (skull frame, vertical; lateral is C1). The take's own per-frame lip is not the target: a
        # big open in the take lifts its upper lip, which the new mouth may not be doing.
        if cal is not None:
            e = X[ix[0]][1] - cal['rest_phil_y']
            tol = cal['upper_tol'] * 0.75
            if abs(e) > tol:
                dy = float(np.clip(-(e - np.sign(e) * tol), -CAP['upper_mm'], CAP['upper_mm']))
                for k in UPPER: X[ix[k], 1] += dy
                rec['upper_mm'] = float(e)
        # C1 one midline: the whole mouth (and the chin chain) shifts sideways onto the take's offset
        w = np.linalg.norm(X[ix[291]] - X[ix[61]])
        mc = np.mean([X[ix[k]] for k in (0, 13, 14, 17)], 0)
        mid_o = mc[0] - fc.midline_x(po, mc[1], ipd_mm)
        q = fc.quantities(pt, ipd_mm)
        err = mid_o - q['mid']
        tol = L['midline_frac'] / 2 * w
        if abs(err) > tol:
            dx = float(np.clip(-(err - np.sign(err) * tol), -CAP['mid_mm'], CAP['mid_mm']))
            for k in LIPS: X[ix[k], 0] += dx
            for k in CHIN: X[ix[k], 0] += 0.6 * dx
            rec['mid_mm'] = float(err)
        # C2 mouth line parallel to the pupil line (relative to the take's own angle): rotate the lips about the centre
        cv = X[ix[291]] - X[ix[61]]
        ang = np.degrees(np.arctan2(cv[1], cv[0]))
        err = ((ang - q['axis']) + 90) % 180 - 90
        if abs(err) > L['axis_deg'] / 2:
            da = float(np.clip(-(err - np.sign(err) * L['axis_deg'] / 2), -CAP['axis_deg'], CAP['axis_deg']))
            c = np.mean([X[ix[k]] for k in (61, 291)], 0)
            for k in LIPS: X[ix[k]] = c + rot(X[ix[k]] - c, da)
            rec['axis_deg'] = float(err)
        # the inner opening shares the mouth's axis (a wedge-shaped opening reads as teeth on another plane)
        iv = X[ix[308]] - X[ix[78]]
        ov = X[ix[291]] - X[ix[61]]
        ie = ((np.degrees(np.arctan2(iv[1], iv[0])) - np.degrees(np.arctan2(ov[1], ov[0]))) + 90) % 180 - 90
        if abs(ie) > 1.0:
            da = -float(np.clip(ie - np.sign(ie) * 1.0, -6, 6))
            c = np.mean([X[ix[k]] for k in INNER_RING], 0)
            for k in INNER_RING: X[ix[k]] = c + rot(X[ix[k]] - c, da)
            rec['inner_axis_deg'] = float(ie)
        # C15 cylinder: near/far half widths keep the take's ratio (far half foreshortens with yaw)
        ra = np.linalg.norm(X[ix[61]] - X[ix[0]]) / max(np.linalg.norm(X[ix[291]] - X[ix[0]]), 1e-6)
        rt = q['half_ratio']
        tol = L['cylinder_tol'] / 2
        if abs(ra / rt - 1) > tol:
            want = rt * (1 + np.sign(ra / rt - 1) * tol)
            scale = want / ra   # scale side A about the philtrum
            for k in SIDE_A:
                X[ix[k]] = X[ix[0]] + (X[ix[k]] - X[ix[0]]) * np.array([scale, 1.0])
            rec['cyl'] = float(ra / rt - 1)
        # C8 jaw yaw: chin lateral swing vs the take, clamped to half the limit
        fc.to_skull(po, [fc.MENTON], ipd_mm)[0]
        cx_o = X[ix[152]][0] - fc.midline_x(po, X[ix[152]][1], ipd_mm)
        lim = np.tan(np.radians(L['jaw_yaw_deg'] / 2)) * q['jaw_len']
        err = cx_o - q['chin_x']
        if jaw_rules and abs(err) > lim:
            dx = float(np.clip(-(err - np.sign(err) * lim), -CAP['chin_mm'], CAP['chin_mm']))
            for k in CHIN: X[ix[k], 0] += dx
            for k in (17, 84, 314, 181, 405): X[ix[k], 0] += 0.5 * dx
            rec['chin_yaw_mm'] = float(err)
        # C10 lips never cross
        g = X[ix[14]][1] - X[ix[13]][1]
        if g < 0:
            for k in LOWER_IN: X[ix[k], 1] += -g + 0.3
            rec['cross_mm'] = float(g)
        # C4 lower lip rides the jaw: chin brought to the character's own lip/jaw ratio band (moves the chin, never the lip)
        if cal is not None and jaw_rules:
            ll = X[ix[17]][1] - X[ix[0]][1] - cal['rest_ll']
            cd = X[ix[152]][1] - X[ix[0]][1] - cal['rest_chin']
            lo, hi = cal['ll_band']
            if cd > 3 and not lo <= ll / cd <= hi:
                want = ll / (hi if ll / cd > hi else lo)
                dy = float(np.clip(want - cd, -CAP['chin_mm'], CAP['chin_mm']))
                for k in CHIN: X[ix[k], 1] += dy
                rec['ll_jaw'] = float(ll / cd)
            cd = X[ix[152]][1] - X[ix[0]][1] - cal['rest_chin']
            kd = (X[ix[61]][1] + X[ix[291]][1]) / 2 - X[ix[0]][1] - cal['rest_corner']
            lo, hi = cal['corner_band']
            if jaw_rules and cd > 4 and not lo <= kd / cd <= hi:   # C5: corners follow half way to the band (soft tissue)
                dy = 0.5 * ((hi if kd / cd > hi else lo) * cd - kd)
                for k in (61, 291, 78, 308, 146, 375, 185, 409): X[ix[k], 1] += dy
                rec['corners'] = float(kd / cd)
        XS.append(X)
        SO.append(So)
        log.append(rec)
    # C7 speed limits in the skull frame (25 fps): chin <= 4.5 mm/frame, lower lip <= 5 mm/frame. Non-accumulating:
    # each frame is pulled toward the previous one by at most CAP chin_mm from its own position; > 3x the limit is a
    # tracker failure, left alone.
    for t in range(1, n if jaw_rules else 1):
        for k, lim, group in ((152, L['chin_mm_per_frame'], CHIN), (17, L['ll_mm_per_frame'], [17, 84, 314, 181, 405])):
            v = XS[t][ix[k]] - XS[t - 1][ix[k]]
            m = np.linalg.norm(v)
            if lim < m < 3 * lim:
                fix = -(v - v / m * lim)
                fix *= min(1.0, CAP['chin_mm'] / max(np.linalg.norm(fix), 1e-6))
                for g in group: XS[t][ix[g]] += fix
                log[t]['speed_' + str(k)] = float(m)
    for t in range(n):
        D[t] = XS[t] - SO[t]
        mag = np.linalg.norm(D[t], axis=1, keepdims=True)
        D[t] *= np.minimum(1.0, CAP['total_mm'] / np.maximum(mag, 1e-6))
    return D, log


def warp_frame(f, p, d_px, ctrl_pts, ipd_mm):
    """thin-plate RBF of control displacements (pins at zero) over a lower-face ROI; backward map src = x - d(x)"""
    H, W = f.shape[:2]
    pins = p[PIN, :2]
    src = np.vstack([ctrl_pts, pins])
    dsp = np.vstack([d_px, np.zeros((len(pins), 2))])
    if np.abs(d_px).max() < 0.15: return f
    lo = np.vstack([p[CTRL, :2], p[PIN, :2]])
    x0, y0 = np.floor(lo.min(0) - 20).astype(int)
    x1, y1 = np.ceil(lo.max(0) + 20).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    rb = RBFInterpolator(src, dsp, kernel='thin_plate_spline', smoothing=1.0)
    st = 4
    gy, gx = np.mgrid[y0:y1:st, x0:x1:st].astype(np.float32)
    dd = rb(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gx.shape + (2,)).astype(np.float32)
    dd = cv2.resize(dd, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    # feathered lower-face mask: hull of lips + chin, grown, blurred; nose/eyes stay untouched by construction (pins)
    m = np.zeros((H, W), np.float32)
    hull = cv2.convexHull(np.vstack([p[CTRL, :2], p[[234, 454, 132, 361], :2]]).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1.0)
    k = int(max(9, 0.15 * np.linalg.norm(p[61, :2] - p[291, :2]))) | 1
    m = cv2.GaussianBlur(m, (k * 2 + 1, k * 2 + 1), 0)[y0:y1, x0:x1, None]
    dd *= m
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    roi = cv2.remap(f, xx - dd[..., 0], yy - dd[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    out = f.copy()
    out[y0:y1, x0:x1] = roi
    return out


def _write(frames, fps, path, audio_src):
    H, W = frames[0].shape[:2]
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-']
    cmd += (['-i', audio_src, '-map', '0:v', '-map', '1:a?', '-c:a', 'copy'] if audio_src else []) + ['-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p', path]
    p = ff.popen(cmd, stdin=ff.PIPE)
    for o in frames: p.stdin.write(o.tobytes())
    p.stdin.close()
    p.wait()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--take', required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json')
    ap.add_argument('--ipd-mm', type=float, default=fc.IPD_MM)
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--jaw-rules', action='store_true', help='also ENFORCE the jaw rules C4 lip/jaw ratio, C5 corners, C7 speed, C8 chin yaw (always '
                    'CHECKED by face_constraints). Off by default: jaw-lock already holds lip->chin; on a pitched head  '
                    'the 2D chin-yaw estimate is biased and pulling the chin smears the lower lip')
    ap.add_argument('--passes', type=int, default=2, help='re-track and correct again (warp + re-measure)')
    a = ap.parse_args(argv)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    F = []
    while len(F) < a.frames:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    from . import mouth_paint
    Pt, _, _ = track(a.take, len(F))
    n = min(len(F), len(Pt))
    out = F[:n]
    Dsum = np.zeros((n, len(CTRL), 2))
    cal = fc.calibrate(Pt[:n], getattr(mouth_paint, 'LAST_OK', None), a.ipd_mm)
    for w in cal['warnings']: print('  calib WARN', w)
    import tempfile
    for it in range(a.passes):
        if it == 0: Po, _, _ = track(a.video, n)
        else:
            tmp = tempfile.mktemp(suffix='.mp4')
            _write(out, fps, tmp, None)
            Po, _, _ = track(tmp, n)
            os.remove(tmp)
        D, log = corrections(Po[:n], Pt[:n], a.ipd_mm, cal=cal, jaw_rules=a.jaw_rules)
        if a.sigma > 0: D = gaussian_filter1d(D, a.sigma, axis=0, mode='nearest')    # no frame-to-frame jitter
        out = [warp_frame(out[t], Po[t], fc.from_skull_vec(Po[t], D[t], a.ipd_mm), Po[t][CTRL, :2], a.ipd_mm) for t in range(n)]
        Dsum += D
        print(f'  pass {it + 1}: max {np.linalg.norm(D, axis=2).max():.2f} mm')
    D = Dsum
    _write(out, fps, a.out, a.video)
    mag = np.linalg.norm(D, axis=2).max(1)
    touched = int((mag > 0.1).sum())
    print(f'skull-lock: corrected {touched}/{n} frames, max {mag.max():.2f} mm, median (touched) '
          f'{np.median(mag[mag > 0.1]) if touched else 0:.2f} mm')
    if a.json: json.dump({'per_frame': log, 'max_mm': mag.round(3).tolist()}, open(a.json, 'w'))


if __name__ == '__main__':
    main()
