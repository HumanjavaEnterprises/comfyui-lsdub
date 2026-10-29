#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""medium.py: `lsdub medium`, the tier-2 dub for a singer face of ~120-300 px (research "Sync the singers by shot size",
recommendation 4). CPU.

  track     YuNet + MediaPipe faces per frame (cast.Faces) -> IoU tracks (cast.link) -> the singer's track (--box hint,
            or the biggest/longest) -> gaps interpolated -> zero-phase smoothed centre and ONE crop size per shot
  crop      similarity only (translate + one fixed scale): a square of --crop-mult x the median face, Lanczos-upscaled so
            the face is --face-target px (the warp's working size). No face restorer, ever.
  warp      the normal `lsdub paint --mode warp` on the crop (every flag after `--` is passed through)
  paste     painted crop -> area downscale -> colour match in Lab on the unchanged skin of the mask -> grain re-added
            -> composited through a feathered NOSE-TO-CHIN mask only. Every pixel outside that mask is the source
            frame, so nothing else in the shot can move; the outside-mask difference is measured and reported.

  lsdub medium --video take.mp4 --track visemes.json --out dubbed.mp4 [--box x0,y0,x1,y1] [--face-target 330] \\
               [--feather 4] [--work dir] [--json report.json] -- --warp-target coart --coart-dyn dyn.csv ...
"""
import argparse, json, os, sys, tempfile
from ..io import ff
import cv2, numpy as np

# MediaPipe 478: the lower face oval (cheek to cheek through the chin) + the nose base -> the nose-to-chin mask
OVAL_LOW = [93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
NOSE_BASE = [425, 327, 2, 98, 205]


# ---------------------------------------------------------------------------------------------------- io
def read(video, n=0):
    cap = cv2.VideoCapture(video)
    F = []
    while not n or len(F) < n:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F, cap.get(cv2.CAP_PROP_FPS) or 25.0


def write(frames, path, fps=25.0, audio=None, crf=12):
    h, w = frames[0].shape[:2]
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
    if audio: cmd += ['-i', audio, '-map', '0:v', '-map', '1:a?', '-c:a', 'aac', '-shortest']
    cmd += ['-c:v', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p', path]
    p = ff.popen(cmd, stdin=ff.PIPE)
    for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


# ---------------------------------------------------------------------------------------------------- track
def _gauss(x, s):
    """Zero-phase gaussian along axis 0 (edge-padded); s <= 0 = identity."""
    x = np.asarray(x, np.float64)
    if s <= 0 or len(x) < 2: return x
    r = int(3 * s + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode='edge')
    return np.stack([np.convolve(pad[:, j], k, 'valid') for j in range(x.shape[1])], 1) if x.ndim == 2 else np.convolve(pad, k, 'valid')


def fill(vals, n):
    """{frame: vector} -> (n, d) array, missing frames linearly interpolated / held at the ends."""
    ks = sorted(vals)
    V = np.array([vals[k] for k in ks], np.float64)
    return np.stack([np.interp(np.arange(n), ks, V[:, j]) for j in range(V.shape[1])], 1)


def valid_frames(found, n, max_gap=3):
    """bool (n,): frames inside the detected span whose gap to a detection is bridgeable. Frames before the first /
    after the last detection, and every frame of a gap longer than max_gap, are False: the tier-2 paste is skipped
    there (the singer left the frame or is occluded), never pasted at a held stale position."""
    ok = np.zeros(n, bool)
    fs = sorted(int(f) for f in found if 0 <= f < n)
    for a, b in zip(fs, fs[1:] + [None]):
        ok[a] = True
        if b is not None and b - a - 1 <= max_gap: ok[a:b] = True
    return ok


def pick_track(tracks, box=None):
    """The singer's track: the best IoU with the --box hint over its frames, else the largest x longest."""
    from .cast import iou
    if box is not None:
        return max(tracks, key=lambda t: max(iou(box, f.get('det_box', f['box'])) for _, f in t))
    return max(tracks, key=lambda t: len(t) * np.median([f['face_h'] for _, f in t]))


def track_face(F, box=None, faces=None, smooth=2.0):
    """-> dict(cx, cy (n,), face_h (median px), lm {frame: (478, 2)}, found) for the singer in frames F."""
    from .cast import Faces, link
    faces = faces or Faces()
    per = [faces(f) for f in F]
    T = [t for t in link(per) if len(t) >= max(2, len(F) // 10)] or link(per)
    if not T: raise SystemExit('medium: no face track in the clip')
    t = pick_track(T, box)
    c = {i: [(f['box'][0] + f['box'][2]) / 2, (f['box'][1] + f['box'][3]) / 2] for i, f in t}
    C = _gauss(fill(c, len(F)), smooth)
    return {'cx': C[:, 0], 'cy': C[:, 1], 'face_h': float(np.median([f['face_h'] for _, f in t])),
            'lm': {i: f['lm'] for i, f in t if f['lm'] is not None}, 'found': len(t) / len(F), 'det': [i for i, _ in t]}


# ---------------------------------------------------------------------------------------------------- crop
def crop_geometry(face_h, crop_mult=2.2, face_target=330.0):
    """-> (S source side px, z zoom, O output side px, even)."""
    S = crop_mult * face_h
    z = face_target / face_h
    O = int(round(S * z / 2)) * 2
    return S, O / S, O


def crop_matrix(cx, cy, S, z):
    """Source -> crop affine: a pure scale z about the crop corner (cx - S/2, cy - S/2). Similarity only."""
    return np.float32([[z, 0, -z * (cx - S / 2)], [0, z, -z * (cy - S / 2)]])


def crop_frame(f, cx, cy, S, z, O):
    return cv2.warpAffine(f, crop_matrix(cx, cy, S, z), (O, O), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def uncrop_points(p, cx, cy, S, z):
    return p / z + np.float32([cx - S / 2, cy - S / 2])


def paste_back(P, cx, cy, S, z, shape):
    """Painted crop (O x O) -> the source frame geometry (area downscale). Any zoom above 1 pre-shrinks with
    INTER_AREA to an integer side r = round(S) (warpAffine's INTER_AREA is bilinear), then places it with the residual
    scale S / r folded into the matrix, so the paste sits exactly on the crop geometry (and on the mask's landmarks).
    Pixels outside the crop square are 0 (outside the mask by construction)."""
    H, W = shape[:2]
    if z > 1.0:
        r = max(1, int(round(S)))
        small = cv2.resize(P, (r, r), interpolation=cv2.INTER_AREA)
        h = 0.5 * (S / r - 1.0 / z)                                     # resize maps pixel CENTRES (half-px), warpAffine does not
        Ms = np.float32([[S / r, 0, cx - S / 2 + h], [0, S / r, cy - S / 2 + h]])
        return cv2.warpAffine(small, Ms, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    Minv = cv2.invertAffineTransform(crop_matrix(cx, cy, S, z))
    return cv2.warpAffine(P, Minv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------------------------------- mask
def lower_face_poly(lm):
    return np.concatenate([lm[OVAL_LOW], lm[NOSE_BASE[::-1]][::-1]])


def lower_face_mask(lm, shape, grow=2.0, feather=4.0):
    """Feathered nose-to-chin mask (float 0..1) in the frame of lm. grow = dilation px, feather = gaussian sigma px.
    The hard support is grow + ~2.5 sigma beyond the polygon; outside that the mask is exactly 0."""
    H, W = shape[:2]
    ss = 4                                           # 4x supersampled polygon edge
    m = np.zeros((H * ss, W * ss), np.uint8)
    cv2.fillPoly(m, [np.round(lower_face_poly(lm) * ss).astype(np.int32)], 255, cv2.LINE_AA)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if grow > 0:
        r = int(np.ceil(grow))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
        m[m < 1.0 / 255] = 0.0                                         # a true zero outside: no 8-bit leakage
    return np.clip(m, 0, 1)


def composite(orig, painted, m):
    """orig * (1 - m) + painted * m, rounded; pixels with m == 0 are orig bit-for-bit."""
    o = orig.astype(np.float32)
    out = o * (1 - m[..., None]) + painted.astype(np.float32) * m[..., None]
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    out[m <= 0] = orig[m <= 0]
    return out


def lab_match(painted, orig, m, unchanged=6.0):
    """Shift painted's Lab mean onto orig's over the mask pixels the warp did NOT move (|diff| < unchanged): the
    resample round trip's colour drift, without pulling the (legitimately darker) open mouth back."""
    sel = (m > 0.05) & (np.abs(painted.astype(np.int16) - orig.astype(np.int16)).max(-1) < unchanged)
    if sel.sum() < 20: return painted, [0.0, 0.0, 0.0]
    P = cv2.cvtColor(painted, cv2.COLOR_BGR2LAB).astype(np.float32)
    Q = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.clip((Q[sel] - P[sel]).mean(0), -6, 6)
    return cv2.cvtColor(np.clip(P + d, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), [round(float(x), 2) for x in d]


def grain(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 1.0)


def regrain(painted, orig, m, rng, cap=3.0):
    """Add gaussian grain so the mask's NOISE matches the source's (the up/down round trip smooths it). The noise is
    measured robustly (MAD of the 1-px high-pass) on the FLAT pixels only, so lost detail (edges, freckles) is not
    mistaken for grain at small faces; capped at `cap` (8-bit sigma)."""
    sel = m > 0.3
    if sel.sum() < 20: return painted, 0.0
    g = cv2.GaussianBlur(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = sel & (gm <= np.percentile(gm[sel], 40))
    if flat.sum() < 20: return painted, 0.0
    mad = lambda x: 1.4826 * float(np.median(np.abs(x - np.median(x))))
    so, sp = mad(grain(orig)[flat]), mad(grain(painted)[flat])
    s = min(cap, float(np.sqrt(max(0.0, so ** 2 - sp ** 2)) / 0.75))   # the 1-px high-pass keeps ~3/4 of white noise
    if s < 0.2: return painted, 0.0
    n = rng.normal(0, s, painted.shape[:2]).astype(np.float32)[..., None]
    return np.clip(painted.astype(np.float32) + n, 0, 255).astype(np.uint8), round(s, 2)


# ---------------------------------------------------------------------------------------------------- mouth opening
def crop_landmarks(crops, faces, face_px):
    """MediaPipe 478 on each (upscaled) crop, the face assumed near the centre at ~face_px. -> {i: (478, 2)}."""
    out = {}
    for i, c in enumerate(crops):
        O = c.shape[0]
        b = (O / 2 - face_px / 2, O / 2 - face_px / 2, face_px, face_px)
        p, _ = faces._landmarks(c, b)
        if p is not None: out[i] = p
    return out


def opening(lm):
    """Inner-lip gap (13-14) / face height (10-152): the landmark mouth-opening metric."""
    return float(np.linalg.norm(lm[13] - lm[14]) / (np.linalg.norm(lm[10] - lm[152]) + 1e-6))


def opening_series(F, cx, cy, face_h, faces, face_target=330.0, crop_mult=2.2):
    """Mouth opening per frame measured through the SAME crop-upscale path for every video (relative, resolution-fair).
    -> (n,) float with NaN where the landmarker found nothing."""
    S, z, O = crop_geometry(face_h, crop_mult, face_target)
    crops = [crop_frame(f, cx[i], cy[i], S, z, O) for i, f in enumerate(F)]
    L = crop_landmarks(crops, faces, face_target)
    return np.array([opening(L[i]) if i in L else np.nan for i in range(len(F))])


# ---------------------------------------------------------------------------------------------------- run
def run(video, track, out, paint_args=(), box=None, face_target=330.0, crop_mult=2.2, smooth=2.0, grow=None, feather=4.0,
        work=None, frames=0, crf=12, seed=0, report=None, grain_cap=3.0, max_gap=3, fps_mult=1, pick=None, zoom_note=True):
    """fps_mult > 1: the crop is painted at fps_mult x the frame rate (each frame repeated; `track` must be built at that
    rate) and `pick(warp_json_path, n, fps_mult) -> [internal frame per output frame]` chooses the frames back
    (default: frame k * fps_mult). The rest is the tier-2 composite."""
    from .cast import Faces
    F, fps = read(video, frames)
    n = len(F)
    work = work or tempfile.mkdtemp(prefix='lsdub_medium_')
    os.makedirs(work, exist_ok=True)
    faces = Faces()
    T = track_face(F, box, faces, smooth)
    fh = T['face_h']
    S, z, O = crop_geometry(fh, crop_mult, face_target)
    if zoom_note and not 1.0 <= z <= 2.6:
        print(f'medium: zoom x{z:.2f} for a {fh:.0f} px face (tier-2 is designed for x1.2-2.5)', file=sys.stderr)
    crops = [crop_frame(f, T['cx'][i], T['cy'][i], S, z, O) for i, f in enumerate(F)]
    cpath, ppath = os.path.join(work, 'crop.mp4'), os.path.join(work, 'crop_painted.mp4')
    m_ = max(1, int(fps_mult))
    write([c for c in crops for _ in range(m_)], cpath, fps * m_, crf=10)
    wj = os.path.join(work, 'crop_painted_warp.json')
    cmd = ['--mode', 'warp', '--video', cpath, '--track', track, '--frames', str(n * m_), *paint_args, '--out', ppath, '--warp-json', wj]
    print('paint ' + ' '.join(cmd), file=sys.stderr)
    from . import mouth_paint as MP
    from .._inproc import call
    call(MP.main, cmd, name='paint')
    PH, _ = read(ppath, n * m_)
    PH += [crops[-1]] * (n * m_ - len(PH))
    idx = (pick(wj, n, m_) if pick else None) or [k * m_ for k in range(n)]
    P = [PH[min(len(PH) - 1, max(0, int(j)))] for j in idx]
    # the mask comes from the ORIGINAL crop's landmarks (smoothed), mapped back to the source frame
    Lc = crop_landmarks(crops, faces, face_target)
    if not Lc: raise SystemExit('medium: the landmarker found no face in the crop')
    ks = sorted(Lc)
    A = np.stack([Lc[k] for k in ks])
    LM = np.stack([np.stack([np.interp(np.arange(n), ks, A[:, j, d]) for d in (0, 1)], 1) for j in range(A.shape[1])], 1)
    LM = _gauss(LM.reshape(n, -1), 1.0).reshape(n, -1, 2)
    grow = max(2.0, 0.04 * fh) if grow is None else grow
    pasted = valid_frames(T['det'], n, max_gap) & valid_frames(Lc.keys(), n, max_gap)   # detected AND landmarked
    rng = np.random.default_rng(seed)
    OUT, M, stats = [], [], []
    for i in range(n):
        cx, cy = T['cx'][i], T['cy'][i]
        lm = uncrop_points(LM[i].astype(np.float32), cx, cy, S, z)
        m = lower_face_mask(lm, F[i].shape, grow, feather) if pasted[i] else np.zeros(F[i].shape[:2], np.float32)
        down = paste_back(P[i], cx, cy, S, z, F[i].shape)          # painted crop back to the source frame
        down, shift = lab_match(down, F[i], m)
        thrown = np.abs(down.astype(np.int16) - F[i].astype(np.int16)).max(-1)   # what the mask throws away (pre-grain)
        down, gs = regrain(down, F[i], m, rng, grain_cap)
        o = composite(F[i], down, m)
        out_d = np.abs(o.astype(np.int16) - F[i].astype(np.int16))[m <= 0]
        ring = (m <= 0) & (cv2.dilate((m > 0).astype(np.uint8), np.ones((int(fh * 0.3) | 1,) * 2, np.uint8)) > 0)
        stats.append({'f': i, 'pasted': bool(pasted[i]), 'centre': [round(float(cx), 1), round(float(cy), 1)], 'mask_px': int((m > 0).sum()),
                      'outside_max': int(out_d.max()) if out_d.size else 0, 'lab_shift': shift, 'grain_sigma': gs,
                      'discarded_ring_p95': float(np.percentile(thrown[ring], 95)) if ring.any() else 0.0})
        OUT.append(o)
        M.append((m * 255).astype(np.uint8))
    write(OUT, out, fps, audio=video, crf=crf)
    np.savez_compressed(os.path.join(work, 'mask.npz'), mask=np.stack(M))
    post = outside_diff(video, out, np.stack(M) > 0, n)
    R = {'video': video, 'out': out, 'frames': n, 'face_h_px': round(fh, 1), 'track_found': round(T['found'], 3),
         'crop_src_px': round(S, 1), 'zoom': round(z, 3), 'crop_px': O, 'face_target': face_target, 'grow_px': round(grow, 2),
         'feather_px': feather, 'warp_found': round(len(Lc) / n, 3), 'max_gap': max_gap,
         'not_pasted': [i for i in range(n) if not pasted[i]],
         'outside_max_preencode': max(s['outside_max'] for s in stats),
         'outside_mean_postencode': post['mean'], 'outside_p99_postencode': post['p99'],
         'codec_floor_mean': post['codec_mean'], 'discarded_ring_p95_med': float(np.median([s['discarded_ring_p95'] for s in stats])),
         'paint_cmd': cmd, 'work': work, 'fps_mult': m_, 'picked': [int(j) for j in idx], 'per_frame': stats}
    if report: json.dump(R, open(report, 'w'), indent=1)
    print(f"medium {out}: face {fh:.0f}px x{z:.2f} -> crop {O}px; outside-mask diff pre-encode max {R['outside_max_preencode']}, "
          f"post-encode mean {post['mean']:.3f} (codec floor {post['codec_mean']:.3f}); warp landmarks {R['warp_found']:.0%}; "
          f"not pasted (track lost) {len(R['not_pasted'])} frames {R['not_pasted'][:20]}")
    return R


def outside_diff(src, out, M, n, crf=12):
    """Mean |out - src| outside the mask after decoding both, and the codec floor: the source re-encoded the same way."""
    A, _ = read(src, n)
    B, _ = read(out, n)
    tmp = tempfile.mkdtemp(prefix='lsdub_medium_floor_')
    fl = os.path.join(tmp, 'floor.mp4')
    write(A, fl, crf=crf)
    C, _ = read(fl, n)
    d = np.concatenate([np.abs(b.astype(np.int16) - a.astype(np.int16))[~m].ravel() for a, b, m in zip(A, B, M)])
    c = np.concatenate([np.abs(b.astype(np.int16) - a.astype(np.int16))[~m].ravel() for a, b, m in zip(A, C, M)])
    return {'mean': round(float(d.mean()), 4), 'p99': float(np.percentile(d, 99)), 'codec_mean': round(float(c.mean()), 4)}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog='lsdub medium')
    ap.add_argument('--video', required=True)
    ap.add_argument('--track', required=True, help='viseme track (lsdub visemes)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--box', help='x0,y0,x1,y1 hint: which face is the singer')
    ap.add_argument('--face-target', type=float, default=330.0, help='face height (px) in the warped crop')
    ap.add_argument('--crop-mult', type=float, default=2.2, help='crop side, x the median face height')
    ap.add_argument('--smooth', type=float, default=2.0, help='zero-phase gaussian (frames) on the crop centre')
    ap.add_argument('--grow', type=float, help='mask dilation, source px (default max(2, 4%% of the face))')
    ap.add_argument('--feather', type=float, default=4.0, help='mask feather sigma, source px (research: 3-6)')
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--crf', type=int, default=12)
    ap.add_argument('--work')
    ap.add_argument('--json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grain-cap', type=float, default=3.0, help='max re-added grain sigma (8-bit)')
    ap.add_argument('--max-gap', type=int, default=3, help='a track / landmark gap longer than this (frames) is not pasted')
    a = ap.parse_args(argv)
    run(a.video, a.track, a.out, extra, [float(v) for v in a.box.split(',')] if a.box else None, a.face_target, a.crop_mult,
        a.smooth, a.grow, a.feather, a.work, a.frames, a.crf, a.seed, a.json, a.grain_cap, a.max_gap)
    return 0


if __name__ == '__main__':
    sys.exit(main())
