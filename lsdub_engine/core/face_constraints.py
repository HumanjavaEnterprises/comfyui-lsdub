# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""face_constraints.py — the anatomy rules in docs/FACE_CONSTRAINTS.md, as code (check + targets for --skull-lock/--jaw-lock).

PRINCIPLE: calibrate the character's OWN proportions from the take (median of well-tracked frames, or the take's own
value on the same frame where head pose matters). The canon is only a sanity band that flags a bad calibration; it is
never a target.

Coordinates: the skull frame. Origin = subnasale (2); x = pupil line (468 -> 473); y = perpendicular, pointing down
the face; units = mm, scaled by the interpupillary distance (IPD_MM, 62 mm female default).

  from face_constraints import calibrate, check, summarize
  cal = calibrate(P_take)                 # P: (frames, 478, 2) MediaPipe landmarks in pixels
  rows = check(P_out, cal, frames=F_out)  # per-frame dict {rule: magnitude} for each violated rule
  print(summarize(rows))

  lsdub constraints --take body.mp4 --frames N name=video.mp4 [...]   # violation counts per rule"""
import argparse, json
import numpy as np

IPD_MM = 62.0
PUPIL_A, PUPIL_B = 468, 473
SUBNASALE, NASION = 2, 168
NOSE = [1, 2, 4, 94]
PHIL, UL_IN, LL_IN, LL_OUT, MENTON = 0, 13, 14, 17, 152
CORNER_A, CORNER_B = 61, 291
JAW_A, JAW_B = 172, 397
HINGE_A, HINGE_B = (234, 172), (454, 397)
INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
RULES = ['C1_midline', 'C2_axis', 'C3_upper_rigid', 'C3_upper_speed', 'C4_lowerlip_jaw', 'C5_corners', 'C7_chin_speed',
         'C7_lowerlip_speed', 'C8_jaw_roll', 'C8_jaw_yaw', 'C9_open_max', 'C10_lip_order', 'C11_closure', 'C12_sticky',
         'C13_rounding', 'C14_incisor_show', 'C15_cylinder']
LIMITS = dict(midline_frac=0.02, axis_deg=2.0, upper_mm=2.5, upper_mm_per_frame=1.0, ll_ratio=(0.6, 1.1), corner_ratio=(0.2, 0.8),
              chin_mm_per_frame=4.5, ll_mm_per_frame=5.0, jaw_roll_deg=2.0, jaw_yaw_deg=3.0, open_max_mm=44.0, open_flag_mm=35.0,
              cross_mm=0.5, closed_mm=1.0, closure_frames=1, sticky_frac=0.25, rounding_min=0.6, incisor_mm=9.0, cylinder_tol=0.15)
SANITY = dict(mouth_ipd=(0.70, 0.90), sn_sto_over_sto_me=(0.33, 0.75), lip_chin_mm=(30, 60), rest_gap_mm=(0, 4))


# ---------- skull frame ----------
def skull_frame(p, ipd_mm=IPD_MM):
    """origin, x unit, y unit (down the face), mm per px for one frame of landmarks"""
    ex = p[PUPIL_B, :2] - p[PUPIL_A, :2]
    d = np.linalg.norm(ex)
    ux = ex / max(d, 1e-6)
    uy = np.array([-ux[1], ux[0]])
    if np.dot(uy, p[MENTON, :2] - p[NASION, :2]) < 0: uy = -uy
    return p[SUBNASALE, :2].copy(), ux, uy, ipd_mm / max(d, 1e-6)


def to_skull(p, idx, ipd_mm=IPD_MM):
    o, ux, uy, s = skull_frame(p, ipd_mm)
    q = p[idx, :2] - o
    return np.stack([q @ ux, q @ uy], -1) * s


def from_skull_vec(p, v_mm, ipd_mm=IPD_MM):
    """a displacement in skull mm -> pixels for frame p"""
    _, ux, uy, s = skull_frame(p, ipd_mm)
    v = np.asarray(v_mm, float)
    return (v[..., :1] * ux + v[..., 1:2] * uy) / s


def midline_x(p, y_mm, ipd_mm=IPD_MM):
    """x (mm, skull frame) of the nose midline extended to height y_mm: the line nasion -> subnasale"""
    a, b = to_skull(p, [NASION, SUBNASALE], ipd_mm)
    t = (y_mm - b[1]) / (b[1] - a[1]) if abs(b[1] - a[1]) > 1e-6 else 0.0
    return b[0] + t * (b[0] - a[0])


def hinge(p):
    return np.array([0.8 * p[HINGE_A[0], :2] + 0.2 * p[HINGE_A[1], :2], 0.8 * p[HINGE_B[0], :2] + 0.2 * p[HINGE_B[1], :2]])


# ---------- per-frame quantities ----------
def quantities(p, ipd_mm=IPD_MM):
    k = [PHIL, UL_IN, LL_IN, LL_OUT, MENTON, CORNER_A, CORNER_B, JAW_A, JAW_B]
    S = dict(zip(k, to_skull(p, k, ipd_mm)))
    mc = (S[PHIL] + S[UL_IN] + S[LL_IN] + S[LL_OUT]) / 4
    width = np.linalg.norm(S[CORNER_B] - S[CORNER_A])
    cv = S[CORNER_B] - S[CORNER_A]
    jv = S[JAW_B] - S[JAW_A]
    h = hinge(p)
    _, _, _, s = skull_frame(p, ipd_mm)
    hm = (h[0] + h[1]) / 2
    jaw_len = np.linalg.norm(p[MENTON, :2] - hm) * s
    return dict(
        mid=mc[0] - midline_x(p, mc[1], ipd_mm), width=width,
        axis=np.degrees(np.arctan2(cv[1], cv[0])), jaw_roll=np.degrees(np.arctan2(jv[1], jv[0])),
        chin_x=S[MENTON][0] - midline_x(p, S[MENTON][1], ipd_mm), jaw_len=jaw_len,
        phil=S[PHIL], ul=S[UL_IN], ll=S[LL_OUT], ll_in=S[LL_IN], chin=S[MENTON],
        gap=S[LL_IN][1] - S[UL_IN][1], outer_order=S[LL_OUT][1] - S[PHIL][1],
        chin_drop=S[MENTON][1] - S[PHIL][1], ll_drop=S[LL_OUT][1] - S[PHIL][1],
        corner_drop=(S[CORNER_A][1] + S[CORNER_B][1]) / 2 - S[PHIL][1],
        half_ratio=np.linalg.norm(S[CORNER_A] - S[PHIL]) / max(np.linalg.norm(S[CORNER_B] - S[PHIL]), 1e-6),
        sn_sto=(S[UL_IN][1] + S[LL_IN][1]) / 2, sto_me=S[MENTON][1] - (S[UL_IN][1] + S[LL_IN][1]) / 2,
        ipd_px=1.0 / s * ipd_mm)


def _series(P, ipd_mm):
    Q = [quantities(p, ipd_mm) for p in P]
    return {k: np.array([q[k] for q in Q]) for k in Q[0]}


def calibrate(P_take, ok=None, ipd_mm=IPD_MM):
    """the character's own proportions. Per-frame take values (same head pose) + medians over well-tracked frames."""
    T = _series(P_take, ipd_mm)
    n = len(P_take)
    ok = np.ones(n, bool) if ok is None else np.asarray(ok, bool)[:n]
    rest = ok & (T['gap'] <= np.percentile(T['gap'][ok], 25))          # the take's most-closed quarter
    cal = dict(take=T, n=n, ipd_mm=ipd_mm,
               rest_width=float(np.median(T['width'][rest])), rest_chin=float(np.median(T['chin_drop'][rest])),
               rest_ll=float(np.median(T['ll_drop'][rest])), rest_corner=float(np.median(T['corner_drop'][rest])),
               rest_gap=float(np.median(T['gap'][rest])),
               rest_ul_y=float(np.median(T['ul'][rest][:, 1])), rest_phil_y=float(np.median(T['phil'][rest][:, 1])),
               max_chin=float(np.percentile(T['chin_drop'][ok], 98)),
               mouth_ipd=float(np.median(T['width'][ok]) / ipd_mm),
               sn_ratio=float(np.median(T['sn_sto'][ok] / np.maximum(T['sto_me'][ok], 1e-6))))
    cal['max_chin_drop'] = cal['max_chin'] - cal['rest_chin']
    # C3: the philtrum (0) is the rigid reference (the inner lip 13 everts on opening). Tolerance = the larger of the
    # canon 2.5 mm and what the character's own upper lip does in the take (p95), same for its per-frame speed.
    py = T['phil'][:, 1]
    cal['upper_tol'] = float(max(LIMITS['upper_mm'], np.percentile(np.abs(py[ok] - cal['rest_phil_y']), 95)))
    cal['upper_speed_tol'] = float(max(LIMITS['upper_mm_per_frame'], np.percentile(np.abs(np.diff(py[ok])), 95) if ok.sum() > 2 else 0))
    # ratio rules: the character's OWN range (take 5-95 %) widened to at least the canon band; canon never narrows it
    cd = T['chin_drop'] - cal['rest_chin']
    ld = T['ll_drop'] - cal['rest_ll']
    kd = T['corner_drop'] - cal['rest_corner']
    for key, num, mind, canon in (('ll_band', ld, 3.0, LIMITS['ll_ratio']), ('corner_band', kd, 4.0, LIMITS['corner_ratio'])):
        sel = ok & (cd > mind)
        if sel.sum() >= 5:
            q = num[sel] / cd[sel]
            lo, hi = np.percentile(q, [5, 95])
            cal[key] = (float(min(canon[0], lo)), float(max(canon[1], hi)))
        else:
            cal[key] = canon
    # sticky lips: allowed up to the larger of 25 % of the character's max drop and what the take itself does sealed
    sealed = ok & (T['gap'] <= LIMITS['closed_mm'])
    own = float(np.percentile(cd[sealed], 95)) if sealed.sum() >= 3 else 0.0
    cal['sticky_mm'] = max(LIMITS['sticky_frac'] * max(cal['max_chin_drop'], 1e-6), own, 2.0)
    warn = []
    for key, val in (('mouth_ipd', cal['mouth_ipd']), ('sn_sto_over_sto_me', cal['sn_ratio']),
                     ('lip_chin_mm', cal['rest_chin']), ('rest_gap_mm', cal['rest_gap'])):
        lo, hi = SANITY[key]
        if not lo <= val <= hi: warn.append(f'{key}={val:.2f} outside canon {lo}-{hi} (check the track; not a target)')
    cal['warnings'] = warn
    return cal


# ---------- pixel helpers ----------
def incisor_show_mm(f, p, ipd_mm=IPD_MM):
    """height of visible upper teeth (bright, low-saturation pixels hanging below the upper inner lip), mm"""
    import cv2
    m = np.zeros(f.shape[:2], np.uint8)
    cv2.fillPoly(m, [p[INNER, :2].astype(np.int32)], 1)
    if m.sum() < 60: return 0.0
    yy, xx = np.nonzero(m)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[yy, xx]
    tooth = (hsv[:, 1] < 80) & (hsv[:, 2] > 130)
    if tooth.sum() < 20: return 0.0
    o, ux, uy, s = skull_frame(p, ipd_mm)
    y = ((np.stack([xx[tooth], yy[tooth]], 1) - o) @ uy) * s
    return float(np.percentile(y, 95) - np.percentile(y, 5))


# ---------- check ----------
def check(P_out, cal, frames=None, closures=None, L=LIMITS):
    """per frame: {rule: magnitude} for every violated rule. closures = iterable of frame indices where a closure
    viseme (M/B/P) is scheduled (C11 checked only when given)."""
    T = cal['take']
    O = _series(P_out, cal['ipd_mm'])
    n = min(len(P_out), cal['n'])
    rows = [dict() for _ in range(n)]
    for t in range(n):
        r = rows[t]
        w = max(O['width'][t], 1e-6)
        d = abs(O['mid'][t] - T['mid'][t]) / w
        if d > L['midline_frac']: r['C1_midline'] = d
        a = abs(((O['axis'][t] - T['axis'][t]) + 90) % 180 - 90)
        if a > L['axis_deg']: r['C2_axis'] = a
        # upper lip vs the character's REST upper lip, vertically in the skull frame (lateral is C1)
        u = abs(O['phil'][t][1] - cal['rest_phil_y'])
        if u > cal.get('upper_tol', L['upper_mm']): r['C3_upper_rigid'] = u
        if t:
            du = abs(O['phil'][t][1] - O['phil'][t - 1][1])
            if du > cal.get('upper_speed_tol', L['upper_mm_per_frame']): r['C3_upper_speed'] = du
            v = np.linalg.norm(O['chin'][t] - O['chin'][t - 1])
            if v > L['chin_mm_per_frame']: r['C7_chin_speed'] = v
            v = np.linalg.norm(O['ll'][t] - O['ll'][t - 1])
            if v > L['ll_mm_per_frame']: r['C7_lowerlip_speed'] = v
        cd = O['chin_drop'][t] - cal['rest_chin']
        ld = O['ll_drop'][t] - cal['rest_ll']
        kd = O['corner_drop'][t] - cal['rest_corner']
        lb = cal.get('ll_band', L['ll_ratio'])
        kb = cal.get('corner_band', L['corner_ratio'])
        if cd > 3:
            q = ld / cd
            if not lb[0] <= q <= lb[1]: r['C4_lowerlip_jaw'] = q
        if cd > 4:
            q = kd / cd
            if not kb[0] <= q <= kb[1]: r['C5_corners'] = q
        jr = abs(((O['jaw_roll'][t] - T['jaw_roll'][t]) + 90) % 180 - 90)
        if jr > L['jaw_roll_deg']: r['C8_jaw_roll'] = jr
        jy = np.degrees(np.arctan2(abs(O['chin_x'][t] - T['chin_x'][t]), max(O['jaw_len'][t], 1e-6)))
        if jy > L['jaw_yaw_deg']: r['C8_jaw_yaw'] = jy
        if O['gap'][t] > L['open_max_mm']: r['C9_open_max'] = O['gap'][t]
        if O['gap'][t] < -L['cross_mm'] or O['outer_order'][t] < -L['cross_mm']: r['C10_lip_order'] = min(O['gap'][t], O['outer_order'][t])
        if O['gap'][t] <= L['closed_mm'] and cd > cal.get('sticky_mm', L['sticky_frac'] * max(cal['max_chin_drop'], 1e-6)): r['C12_sticky'] = cd
        if O['width'][t] < L['rounding_min'] * cal['rest_width']: r['C13_rounding'] = O['width'][t] / cal['rest_width']
        c = abs(O['half_ratio'][t] / max(T['half_ratio'][t], 1e-6) - 1)
        if c > L['cylinder_tol']: r['C15_cylinder'] = c
        if frames is not None and t < len(frames):
            s = incisor_show_mm(frames[t], P_out[t], cal['ipd_mm'])
            if s > L['incisor_mm']: r['C14_incisor_show'] = s
    if closures is not None:
        for t0 in closures:
            win = range(max(t0 - 1, 0), min(t0 + 2, n))
            if sum(O['gap'][t] <= L['closed_mm'] for t in win) < L['closure_frames']: rows[min(t0, n - 1)]['C11_closure'] = 1
    return rows


def summarize(rows):
    out = {k: 0 for k in RULES}
    for r in rows:
        for k in r: out[k] += 1
    out['frames_with_any'] = sum(1 for r in rows if r)
    out['total'] = sum(out[k] for k in RULES)
    out['frames'] = len(rows)
    return out


def _read(video, n):
    import cv2
    cap = cv2.VideoCapture(video)
    F = []
    while len(F) < n:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F


def main(argv=None):
    from .mouth_paint import track
    from . import mouth_paint
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--ipd-mm', type=float, default=IPD_MM)
    ap.add_argument('--json')
    ap.add_argument('videos', nargs='+')
    a = ap.parse_args(argv)
    Pt, _, _ = track(a.take, a.frames)
    cal = calibrate(Pt, getattr(mouth_paint, 'LAST_OK', None), a.ipd_mm)
    print(f"calibration: mouth/IPD {cal['mouth_ipd']:.2f}  sn:sto/sto:me {cal['sn_ratio']:.2f}  rest gap {cal['rest_gap']:.1f} mm  "
          f"max chin drop {cal['max_chin_drop']:.1f} mm  rest width {cal['rest_width']:.1f} mm  "
          f"upper-lip tol {cal['upper_tol']:.1f} mm  own LL/jaw {cal['ll_band'][0]:.2f}-{cal['ll_band'][1]:.2f}  corners/jaw {cal['corner_band'][0]:.2f}-{cal['corner_band'][1]:.2f}")
    for w in cal['warnings']: print('  WARN', w)
    res = {}
    for spec in a.videos:
        name, v = spec.split('=', 1)
        F = _read(v, a.frames)
        P, _, _ = track(v, len(F))
        s = summarize(check(P, cal, frames=F))
        res[name] = s
        print(f"{name:10s} violations {s['total']:4d} in {s['frames_with_any']}/{s['frames']} frames  " +
              ' '.join(f"{k.split('_', 1)[0]}={s[k]}" for k in RULES if s[k]))
    if a.json: json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
