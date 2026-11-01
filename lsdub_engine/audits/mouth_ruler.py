#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_ruler.py: `lsdub mouth-ruler`, the baseline rulers of the natural mouth build (step 0). CPU, no models beyond the
MediaPipe face landmarker. Each video is tracked on its own and regions come from THAT video's mesh: the render's lips are
not the take's lips. In every "A -> B" the left side is the take.

  stats     take vs render, per shot, over the frames where the video's own mesh is found (medians of per-frame values):
              pink_frac       inner-lip pixels with L* in (25, 75] and a* > 14, on open frames (aperture 13-14 > 0.2 mouth width)
              cav_L50         median L* of the inner-lip region, open frames
              cav_black_frac  share of that region with L* < 10
              teeth_L         mean L* of teeth pixels (L* > 55, chroma < 22), open frames with > 25 such pixels
              lip_a, lip_b    mean Lab of the lip ring, up_* / lo_* the upper / lower lip polygons eroded 3 px
              edge_inner      p75 of the Sobel gradient along the inner lip contour (edge_outer: the outer one)
              lip_hf, teeth_hf, ring_hf   fine detail: std of a 0.7 - 1.6 px difference of Gaussians in the region
              boil            |y[t-1] - 2 y[t] + y[t+1]| of luma, each neighbour shifted onto the mouth centre, in the lip
                              ring / inner lip / skin ring, on hold frames: the opening moves < 0.006 mouth width per frame.
                              boil_loose is the same at < 0.03, because a strict hold can be only a few frames long
                              boil_all is the same on every tracked frame: it does not change with which frames a render
                              happens to hold, so it compares two renders where the hold keys' medians can fall between modes
            a take with no render reports the take column only (a clip nobody has dubbed yet).
  stages    Laplacian variance of the mouth box per pipeline stage, each stage tracked on its own:
              lap_own   at the stage's own pixels
              lap_ref   after resampling the box so the mouth has the first stage's width, the only number comparable
                        across stages of different raster
  null      full-reference, a build of the take's own mouth (`--warp-target own`: the plan IS the take's opening and width,
            so the warp moves nothing and every difference comes from the stack and the codec) against the take:
            PSNR, SSIM, mean delta-E 1976 and the Laplacian ratio, in the mouth region (lip hull grown 0.25 mouth width) and
            on open-frame interiors; `floor` is the same score for the take re-encoded at the warp's own CRF, the codec alone
  spectrum  mouth vs cheek: square patches (side 0.44 mouth width) on the mouth centre and mid-cheek. Mean power per radial band
            (cycles / px), mouth / cheek per video, gap_db = 10 log10(render ratio / take ratio): 0 = the mouth sits against its
            cheek as the take's does; > 0 = the mouth carries more energy in that band than the take's did. Grain: robust
            (MAD) std of a 0.6 - 1.5 px difference of Gaussians, lip ring / cheek, grain_gap = render ratio / take ratio.

  lsdub mouth-ruler stats --take T.mp4 [--render R.mp4] [--take-t s] [--render-t s] [--frames n] [--mask full.npz] [--json o.json]
  lsdub mouth-ruler stages a=one.mp4 b=two.mp4[@start_frame[+frames]] ... [--json o.json]
  lsdub mouth-ruler null --reference T.mp4 --null N.mp4 [--frames n] [--json o.json]
  lsdub mouth-ruler source --take T.mp4 [--render R.mp4] [--take-t s] [--render-t s] [--frames n] [--json o.json]
  lsdub mouth-ruler run --manifest m.json --out dir [--labels a,b] [--jobs 4] [--reference audit.json]
  lsdub mouth-ruler cut --video cut.mp4 --shots shots.json --json o.json

`cut` is the ruler for a finished cut: each dubbed shot read from the delivered file, against its cell's take, because a cell's render.mp4
is not what gets delivered (sing or native, the encode, the paste).
"""

from ..core import landmarker
from ..io import ff
import argparse, concurrent.futures, hashlib, json, os, shlex, sys, tempfile
import cv2, numpy as np
from ..core.cpu_env import cpu_env

LIPO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPI = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
UP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
LO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
OPEN = 0.2  # a frame is open when aperture > this x mouth width
HOLD, HOLD_LOOSE = 0.006, 0.03
BANDS = ((0.02, 0.08), (0.08, 0.16), (0.16, 0.25), (0.25, 0.35), (0.35, 0.5))
CRF_WARP = 14  # the warp's own writer (mouth_warp.WarpClip.write)
STAT_KEYS = (
    "w",
    "pink_frac",
    "pink_core",
    "cav_L50",
    "cav_black_frac",
    "cav_mid_frac",
    "teeth_frac",
    "teeth_L",
    "teeth_C",
    "teeth_hf",
    "lip_L",
    "lip_a",
    "lip_b",
    "up_a",
    "up_b",
    "lo_a",
    "lo_b",
    "edge_inner",
    "edge_outer",
    "lip_hf",
    "ring_hf",
)


# ------------------------------------------------------------------------------------------------------------ io
def read_clip(path, start=0, n=None, t=None):
    """BGR frames from frame `start`, or from time `t` s when given, at most n."""
    c = cv2.VideoCapture(os.path.expanduser(path))
    if t is not None:
        c.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    elif start:
        c.set(cv2.CAP_PROP_POS_FRAMES, start)
    F = []
    while n is None or len(F) < n:
        ok, f = c.read()
        if not ok:
            break
        F.append(f)
    c.release()
    return F


def track(F, fps=25.0):
    """478 x 2 landmarks per frame of a clip (NaN where no face), VIDEO mode on the CPU delegate."""
    mp = landmarker.mp()
    L = landmarker.get("video", conf=0.2, presence=0.5)
    R = []
    for i, f in enumerate(F):
        h, w = f.shape[:2]
        r = L.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps) + 1)
        R.append(
            np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]][:478], np.float32)
            if r.face_landmarks
            else np.full((478, 2), np.nan, np.float32)
        )
    L.close()
    return np.array(R)


def cached_track(F, key, work=None):
    if not work:
        return track(F)
    os.makedirs(work, exist_ok=True)
    p = os.path.join(work, "trk_" + hashlib.sha1(key.encode()).hexdigest()[:16] + ".npy")
    if os.path.exists(p):
        return np.load(p)
    T = track(F)
    np.save(p, T)
    return T


# ------------------------------------------------------------------------------------------------------- regions
def poly(s, p):
    m = np.zeros(s, np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(p))], 1)
    return m.astype(bool)


def dog(y, a, b):
    return cv2.GaussianBlur(y, (0, 0), a) - cv2.GaussianBlur(y, (0, 0), b)


def erode(m, k):
    return cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def dil(m, k):
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def lab(f):
    l = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
    l[..., 0] *= 100 / 255
    l[..., 1] -= 128
    l[..., 2] -= 128
    return l


def contour_grad(g, pts):
    """p75 of the gradient along a closed contour."""
    p = np.vstack([np.linspace(pts[i], pts[(i + 1) % len(pts)], 6) for i in range(len(pts))])
    x = np.clip(p[:, 0].astype(int), 0, g.shape[1] - 1)
    y = np.clip(p[:, 1].astype(int), 0, g.shape[0] - 1)
    return float(np.percentile(g[y, x], 75))


def frame_stats(f, p):
    d = dict(w=np.nan, ap=np.nan)
    if not np.isfinite(p).all():
        return d
    d["w"] = float(np.linalg.norm(p[61] - p[291]))
    d["ap"] = float(np.linalg.norm(p[13] - p[14]))
    H, W = f.shape[:2]
    mo = poly((H, W), p[LIPO])
    mi = poly((H, W), p[LIPI])
    l = lab(f)
    y = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(3, int(d["w"] * 0.2)) | 1
    lip = erode(mo & ~mi, 3)
    ring = dil(mo, k) & ~dil(mo, 3)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3) / 8
    g = np.hypot(gx, gy)
    hh = dog(y, 0.7, 1.6)
    for nm, m in (("lip", lip), ("ring", ring)):
        if m.sum() > 30:
            d[nm + "_hf"] = float(hh[m].std())
            d[nm + "_L"] = float(l[..., 0][m].mean())
            d[nm + "_a"] = float(l[..., 1][m].mean())
            d[nm + "_b"] = float(l[..., 2][m].mean())
    mu = erode(poly((H, W), p[UP]), 3)
    ml = erode(poly((H, W), p[LO]), 3)
    if mu.sum() > 20 and ml.sum() > 20:
        d["up_a"], d["up_b"] = l[mu][:, 1:].mean(0).tolist()
        d["lo_a"], d["lo_b"] = l[ml][:, 1:].mean(0).tolist()
    d["edge_outer"] = contour_grad(g, p[LIPO])
    d["edge_inner"] = contour_grad(g, p[LIPI])
    if mi.sum() > max(60, 0.03 * d["w"] ** 2) and d["ap"] > OPEN * d["w"]:
        L, a, b = [l[..., c][mi] for c in range(3)]
        C = np.hypot(a, b)
        d["cav_black_frac"] = float((L < 10).mean())
        d["cav_L50"] = float(np.median(L))
        d["cav_mid_frac"] = float(((L >= 10) & (L <= 25)).mean())
        te = (L > 55) & (C < 22)
        d["teeth_frac"] = float(te.mean())
        d["pink_frac"] = float(((L > 25) & (a > 14) & (L <= 75)).mean())
        core = erode(mi, 7)  # 3 px clear of the tracked contour: the lip's own rim does not count
        if core.sum() > 30:
            pk = (l[..., 0] > 25) & (l[..., 1] > 14) & (l[..., 0] <= 75)
            d["pink_core"] = float(pk[core].mean())
        if te.sum() > 25:
            d["teeth_L"] = float(L[te].mean())
            d["teeth_C"] = float(C[te].mean())
            d["teeth_hf"] = float(hh[mi][te].std())
    return d


def boil(a, b, c, pa, pb, pc):
    """-> (lip ring, inner lip, skin ring): second temporal difference of luma, neighbours shifted onto b's mouth centre."""
    H, W = b.shape[:2]
    mo = poly((H, W), pb[LIPO])
    mi = poly((H, W), pb[LIPI])
    lip = mo & ~mi
    k = max(3, int(np.linalg.norm(pb[61] - pb[291]) * 0.2)) | 1
    ring = dil(mo, k) & ~dil(mo, 3)

    def sh(f, pf):
        s = np.mean(pb[[61, 291, 13, 14]] - pf[[61, 291, 13, 14]], 0)
        return cv2.warpAffine(
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), np.float32([[1, 0, s[0]], [0, 1, s[1]]]), (W, H), flags=cv2.INTER_CUBIC
        )

    d = np.abs(sh(a, pa) - 2 * sh(b, pb) + sh(c, pc))
    return tuple(float(d[m].mean()) if m.sum() > 20 else np.nan for m in (lip, mi, ring))


def boil_stats(F, T, W_, O, hold):
    q = []
    for i in range(1, len(F) - 1):
        if not np.isfinite(T[i - 1 : i + 2]).all():
            continue
        if abs(O[i + 1] - O[i]) > hold * W_[i] or abs(O[i] - O[i - 1]) > hold * W_[i]:
            continue
        q.append(boil(F[i - 1], F[i], F[i + 1], T[i - 1], T[i], T[i + 1]))
    if not q:
        return dict(n=0, lip=None, cav=None, ring=None)
    q = np.array(q)
    return dict(n=len(q), lip=_r(np.nanmedian(q[:, 0])), cav=_r(np.nanmedian(q[:, 1])), ring=_r(np.nanmedian(q[:, 2])))


def hold_frames(T, W_, O, hold=None):
    """the frames boil_stats reads at this hold (HOLD_LOOSE by default)."""
    hold = HOLD_LOOSE if hold is None else hold
    return {
        i
        for i in range(1, len(O) - 1)
        if np.isfinite(T[i - 1 : i + 2]).all() and abs(O[i + 1] - O[i]) <= hold * W_[i] and abs(O[i] - O[i - 1]) <= hold * W_[i]
    }


def boil_common(clips, hold=None):
    """boil on the frames held in EVERY render: clips = [(F, T), ...] of one take, compared on the same frames.
    -> dict(n, cav=[one per clip], lip=[one per clip], paired=[one per clip after the first]). paired reads the difference frame
    by frame against the first clip: dict(cav, lo, hi) = its median and 95 % bootstrap interval. Each clip's own median can swing
    by the mode it lands in where the paired difference does not."""
    meas = []
    for F, T in clips:
        S = [frame_stats(F[i], T[i]) for i in range(len(F))]
        meas.append((F, T, np.array([s["w"] for s in S], float), np.array([s["ap"] for s in S], float)))
    keep = sorted(set.intersection(*(hold_frames(T, W_, O, hold) for _, T, W_, O in meas)))
    if not keep:
        return dict(n=0, cav=[None] * len(clips), lip=[None] * len(clips), paired=[None] * (len(clips) - 1))
    q = [np.array([boil(F[i - 1], F[i], F[i + 1], T[i - 1], T[i], T[i + 1]) for i in keep]) for F, T, _, _ in meas]
    rng = np.random.default_rng(0)
    paired = []
    for x in q[1:]:
        d = (x[:, 1] - q[0][:, 1])[np.isfinite(x[:, 1] - q[0][:, 1])]
        if not len(d):
            paired.append(None)
            continue
        b = np.median(rng.choice(d, (2000, len(d))), axis=1)
        paired.append(dict(cav=_r(np.median(d)), lo=_r(np.percentile(b, 2.5)), hi=_r(np.percentile(b, 97.5))))
    return dict(n=len(keep), cav=[_r(np.nanmedian(x[:, 1])) for x in q], lip=[_r(np.nanmedian(x[:, 0])) for x in q], paired=paired)


def _r(v, k=3):
    return None if v is None or not np.isfinite(v) else round(float(v), k)


def video_stats(F, T):
    S = [frame_stats(F[i], T[i]) for i in range(len(F))]
    r = dict(frames=len(F), tracked=int(np.isfinite(T[:, 0, 0]).sum()), open_frames=sum("cav_L50" in s for s in S))
    for k in STAT_KEYS:
        v = [s[k] for s in S if k in s and s[k] is not None and np.isfinite(s[k])]
        r[k] = _r(np.median(v)) if v else None
    W_ = np.array([s["w"] for s in S], float)
    O = np.array([s["ap"] for s in S], float)
    r["ap_p95"] = _r(np.nanpercentile(O, 95)) if np.isfinite(O).any() else None
    r["boil"] = boil_stats(F, T, W_, O, HOLD)
    r["boil_loose"] = boil_stats(F, T, W_, O, HOLD_LOOSE)
    r["boil_all"] = boil_stats(F, T, W_, O, np.inf)
    return r


def stats(A, TA, B=None, TB=None):
    n = min(len(A), len(TA), *([len(B), len(TB)] if B is not None else []))
    r = dict(n=n, take=video_stats(A[:n], TA[:n]))
    if B is not None:
        r["render"] = video_stats(B[:n], TB[:n])
        r["frame_absdiff"] = (
            _r(np.median([np.abs(A[i].astype(float) - B[i]).mean() for i in range(n)]), 2) if A[0].shape == B[0].shape else None
        )
    return r


# -------------------------------------------------------------------------------------------------------- stages
def mouth_box(p, shape, grow=0.15):
    w = np.linalg.norm(p[61] - p[291])
    q = p[LIPO]
    H, W = shape[:2]
    g = grow * w
    return (int(max(0, q[:, 0].min() - g)), int(max(0, q[:, 1].min() - g)), int(min(W, q[:, 0].max() + g)), int(min(H, q[:, 1].max() + g)))


def lap_var(y):
    return float(cv2.Laplacian(y, cv2.CV_64F).var())


def stage_sharpness(F, T, ref_w=None):
    """-> dict(w, lap_own, lap_ref): medians over tracked frames. ref_w None = this stage's own median width."""
    ok = [i for i in range(min(len(F), len(T))) if np.isfinite(T[i]).all()]
    if not ok:
        return dict(frames=len(F), tracked=0, w=None, lap_own=None, lap_ref=None)
    ws = np.array([np.linalg.norm(T[i, 61] - T[i, 291]) for i in ok])
    w0 = float(np.median(ws))
    ref_w = ref_w or w0
    own, ref = [], []
    for i, w in zip(ok, ws):
        x0, y0, x1, y1 = mouth_box(T[i], F[i].shape)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        y = cv2.cvtColor(F[i][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        own.append(lap_var(y))
        s = ref_w / max(1e-6, w)
        ref.append(lap_var(cv2.resize(y, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)))
    return dict(frames=len(F), tracked=len(ok), w=_r(w0, 1), lap_own=_r(np.median(own), 1), lap_ref=_r(np.median(ref), 1))


def stages(items, work=None):
    """items: (label, path, start, n, t). The first stage sets the width lap_ref is read at."""
    out, ref = [], None
    for lab_, path, start, n, t in items:
        if not os.path.exists(os.path.expanduser(path)):
            out.append(dict(stage=lab_, missing=path))
            continue
        F = read_clip(path, start, n, t)
        T = cached_track(F, f"{path}:{start}:{n}:{t}", work)
        r = stage_sharpness(F, T, ref)
        ref = ref or r["w"]
        out.append(dict(stage=lab_, **r))
    return out


# ---------------------------------------------------------------------------------------------------- null / full-ref
def ssim_map(a, b):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    G = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = G(a), G(b)
    va, vb = G(a * a) - ma * ma, G(b * b) - mb * mb
    c = G(a * b) - ma * mb
    return ((2 * ma * mb + C1) * (2 * c + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))


def full_ref_frame(a, b, m):
    ya, yb = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (a, b)]
    e = float(((ya - yb)[m] ** 2).mean())
    return dict(
        psnr=99.0 if e < 1e-9 else 10 * float(np.log10(255**2 / e)),
        ssim=float(ssim_map(ya, yb)[m].mean()),
        de=float(np.linalg.norm(lab(a) - lab(b), axis=2)[m].mean()),
    )


def full_ref(R, X, T):
    """R reference frames, X test frames, T the reference's own track -> medians for the mouth and the open interior."""
    M, I, lr = [], [], []
    for i in range(min(len(R), len(X), len(T))):
        p = T[i]
        if not np.isfinite(p).all():
            continue
        s = R[i].shape[:2]
        w = np.linalg.norm(p[61] - p[291])
        mo = dil(poly(s, p[LIPO]), max(3, int(0.25 * w)) | 1)
        M.append(full_ref_frame(R[i], X[i], mo))
        x0, y0, x1, y1 = mouth_box(p, s)
        gy = lambda f: cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        lr.append(lap_var(gy(X[i])) / max(1e-6, lap_var(gy(R[i]))))
        mi = poly(s, p[LIPI])
        if np.linalg.norm(p[13] - p[14]) > OPEN * w and mi.sum() > max(60, 0.03 * w**2):
            I.append(full_ref_frame(R[i], X[i], mi))
    med = lambda L: {k: _r(np.median([d[k] for d in L])) for k in ("psnr", "ssim", "de")} if L else None
    return dict(frames=len(M), mouth=med(M), open_frames=len(I), interior=med(I), lap_ratio=_r(np.median(lr)) if lr else None)


def encode_floor(R, fps=25.0, crf=CRF_WARP):
    d = tempfile.mkdtemp(prefix="lsdub_ruler_")
    p = os.path.join(d, "floor.mp4")
    h, w = R[0].shape[:2]
    pr = ff.popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            p,
        ],
        stdin=ff.PIPE,
    )
    for f in R:
        pr.stdin.write(f.tobytes())
    pr.stdin.close()
    pr.wait()
    F = read_clip(p)
    os.remove(p)
    os.rmdir(d)
    return F


def null_args(tokens):
    """a logged paint command -> the same recipe as a null build: the plan, the run's outputs and the report are dropped,
    `--warp-target own` added. Everything else (layers, gate, lock, kit, track) stays as logged."""
    out, i, t = [], 0, list(tokens)
    while i < len(t):
        k = t[i]
        if k in ("--warp-target", "--warp-gate-report", "--out", "--warp-json", "--mask-out") or k.startswith("--coart-"):
            i += 2
            continue
        out.append(k)
        i += 1
    return out + ["--warp-target", "own"]


def render_null(spec, out):
    """run the null paint of one manifest entry -> out (mp4, no watermark). CPU."""
    args = null_args(
        spec["paint"] if isinstance(spec["paint"], list) else shlex.split(open(os.path.expanduser(spec["paint"])).read().strip())[1:]
    )
    args = [os.path.expanduser(a) for a in args + spec.get("extra", [])]
    if args and args[0] != "paint":
        args = ["paint"] + args
    env = cpu_env()
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".log", "w") as lg:
        ff.python(
            None,
            "lsdub_engine.cli",
            [*args, "--no-watermark", "--out", out, "--warp-json", out + ".warp.json"],
            cwd=os.path.expanduser(spec.get("cwd", ".")),
            env=env,
            stdout=lg,
            stderr=ff.STDOUT,
            check=True,
        )


def null_test(R, X, T, fps=25.0):
    n = min(len(R), len(X), len(T))
    R, X, T = R[:n], X[:n], T[:n]
    return dict(null=full_ref(R, X, T), floor=full_ref(R, encode_floor(R, fps), T))


# ------------------------------------------------------------------------------------------------------- spectrum
def band_power(y):
    """mean power per BANDS of a square patch, Hann windowed, mean removed."""
    n = y.shape[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    P = np.abs(np.fft.fft2((y - y.mean()) * w)) ** 2
    f = np.fft.fftfreq(n)
    r = np.hypot(f[:, None], f[None])
    return np.array([P[(r >= a) & (r < b)].mean() if ((r >= a) & (r < b)).any() else np.nan for a, b in BANDS])


def mad_std(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def patch(y, c, s):
    x0, y0 = int(round(c[0] - s / 2)), int(round(c[1] - s / 2))
    return None if x0 < 0 or y0 < 0 or x0 + s > y.shape[1] or y0 + s > y.shape[0] else y[y0 : y0 + s, x0 : x0 + s]


def video_spectrum(F, T, M=None):
    """-> dict(band powers of mouth and cheek, grain of lip ring and cheek); M = per-frame paint mask (frame size), a cheek
    patch more than 10 % inside it is not cheek."""
    ok = [i for i in range(min(len(F), len(T))) if np.isfinite(T[i]).all()]
    if not ok:
        return None
    s = max(16, int(0.44 * np.median([np.linalg.norm(T[i, 61] - T[i, 291]) for i in ok]))) // 2 * 2
    PM, PC, GM, GC = [], [], [], []
    for i in ok:
        p = T[i]
        y = cv2.cvtColor(F[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        c0 = p[[61, 291, 13, 14]].mean(0)
        m = patch(y, c0, s)
        ch = []
        for a, b in ((234, 61), (454, 291)):
            c = p[a] * 0.55 + p[b] * 0.45
            q = patch(y, c, s)
            if q is None:
                continue
            if M is not None:
                u = patch(M[min(i, len(M) - 1)].astype(np.float32), c, s)
                if u is None or u.mean() > 0.1:
                    continue
            ch.append((c, q))
        if m is None or not ch:
            continue
        PM.append(band_power(m))
        PC.extend(band_power(q) for _, q in ch)
        hf = dog(y, 0.6, 1.5)
        mo = poly(y.shape, p[LIPO])
        mi = poly(y.shape, p[LIPI])
        lip = erode(mo & ~mi, 3)
        if lip.sum() > 30:
            GM.append(mad_std(hf[lip]))
        GC.extend(mad_std(patch(hf, c, s).ravel()) for c, _ in ch)
    if not PM:
        return None
    return dict(
        side=s,
        frames=len(PM),
        mouth=np.nanmean(PM, 0).tolist(),
        cheek=np.nanmean(PC, 0).tolist(),
        grain_mouth=float(np.median(GM)) if GM else None,
        grain_cheek=float(np.median(GC)) if GC else None,
    )


def spectrum(A, TA, B, TB, M=None):
    a, b = video_spectrum(A, TA), video_spectrum(B, TB, M)
    if not a or not b:
        return dict(note="no usable mouth and cheek patches")
    Ra, Rb = np.array(a["mouth"]) / np.array(a["cheek"]), np.array(b["mouth"]) / np.array(b["cheek"])
    gr = lambda s: s["grain_mouth"] / s["grain_cheek"] if s["grain_mouth"] and s["grain_cheek"] else None
    ga, gb = gr(a), gr(b)
    return dict(
        bands=[list(x) for x in BANDS],
        frames=[a["frames"], b["frames"]],
        side=[a["side"], b["side"]],
        ratio_take=[_r(v) for v in Ra],
        ratio_render=[_r(v) for v in Rb],
        gap_db=[_r(v, 2) for v in 10 * np.log10(Rb / Ra)],
        grain_take=_r(ga),
        grain_render=_r(gb),
        grain_gap=_r(gb / ga) if ga and gb else None,
        grain_cheek=[_r(a["grain_cheek"]), _r(b["grain_cheek"])],
    )


STABLE = [33, 133, 263, 362, 1, 168, 6]


def canon(F, T, ref=None):
    """per frame a similarity onto the stable points (eyes, nose bridge) of ref, by default the first found frame: the skull,
    not the mouth."""
    ok = [i for i in range(len(T)) if np.isfinite(T[i]).all()]
    ref = T[ok[0]][STABLE] if ref is None else ref
    return {i: cv2.estimateAffinePartial2D(T[i][STABLE].astype(np.float32), ref.astype(np.float32), method=cv2.LMEDS)[0] for i in ok}


def tf(M, p):
    return p @ M[:, :2].T + M[:, 2]


def interior_residual(F, T, pairs=120):
    """the take's interior against itself: frame pairs at the same aperture (within 0.03 mouth width, at least 4 frames apart),
    the second carried onto the first by an affine of the inner contour plus a phase-correlation shift. Mean |L*| difference,
    all / teeth / tongue and cavity; `floor` = the same on neighbouring frames, the tracking and codec share."""
    ok = [i for i in range(len(T)) if np.isfinite(T[i]).all()]
    ap = {i: float(np.linalg.norm(T[i][13] - T[i][14]) / np.linalg.norm(T[i][61] - T[i][291])) for i in ok}
    op = [i for i in ok if ap[i] > OPEN]
    near = [(a, b) for k, a in enumerate(op) for b in op[k + 1 :] if b - a >= 4 and abs(ap[a] - ap[b]) < 0.03]
    near = [near[k] for k in np.linspace(0, len(near) - 1, min(pairs, len(near))).astype(int)] if near else []
    adj = [(a, a + 1) for a in op if a + 1 in ap and ap[a + 1] > OPEN][:pairs]

    def one(a, b):
        fa, fb = F[a], F[b]
        M = np.linalg.lstsq(np.c_[T[b][LIPI], np.ones(len(LIPI))], T[a][LIPI], rcond=None)[0].T
        w = cv2.warpAffine(fb, M.astype(np.float64), fa.shape[1::-1], flags=cv2.INTER_LINEAR)
        m = poly(fa.shape[:2], T[a][LIPI])
        m = erode(m, 2) if erode(m, 2).sum() >= 30 else m
        ys, xs = np.nonzero(m)
        if len(ys) < 30:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        ga, gb = [cv2.cvtColor(v[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32) for v in (fa, w)]
        if min(ga.shape) >= 8:
            (dx, dy), _ = cv2.phaseCorrelate(gb, ga)
            if max(abs(dx), abs(dy)) < 0.15 * max(ga.shape):
                w = cv2.warpAffine(w, np.array([[1, 0, dx], [0, 1, dy]], np.float64), fa.shape[1::-1], flags=cv2.INTER_LINEAR)
        la, lb = lab(fa), lab(w)
        tooth = (la[..., 0] > 55) & (np.hypot(la[..., 1], la[..., 2]) < 22)
        d = np.abs(la[..., 0] - lb[..., 0])
        return float(d[m].mean()), float(d[m & tooth].mean()) if (m & tooth).sum() > 25 else np.nan, float(d[m & ~tooth].mean())

    def med(ps):
        r = [x for x in (one(a, b) for a, b in ps) if x is not None]
        return [round(float(np.nanmedian(c)), 2) if len(c) and np.isfinite(c).any() else None for c in zip(*r)] if r else [None] * 3

    r, fl = med(near), med(adj)
    return dict(
        open_frames=len(op), pairs=len(near), res_all=r[0], res_teeth=r[1], res_interior=r[2], floor_all=fl[0], floor_interior=fl[2]
    )


def reveal(TA, CA, TB, CB, shape):
    """does the dub open past what the take showed? Inner contours in skull space: the union over the take is what was seen,
    `unseen` = the share of a render frame's opening outside it. Apertures in mouth widths."""
    S = 2
    H, W = shape[0] * S, shape[1] * S
    seen = np.zeros((H, W), bool)
    for i, M in CA.items():
        seen |= poly((H, W), tf(M, TA[i][LIPI]) * S)
    ap = lambda t: float(np.linalg.norm(t[13] - t[14]) / np.linalg.norm(t[61] - t[291]))
    a_t = [ap(TA[i]) for i in CA]
    u, a_r = [], []
    for i, M in CB.items():
        a_r.append(ap(TB[i]))
        m = poly((H, W), tf(M, TB[i][LIPI]) * S)
        if a_r[-1] > OPEN and m.sum() > 0:
            u.append(float((m & ~seen).sum() / m.sum()))
    u_ = np.array(u) if u else np.zeros(1)
    top = max(a_t)
    return dict(
        ap_take_max=round(top, 3),
        ap_render_max=round(max(a_r), 3),
        frames_past_take=int((np.array(a_r) > top).sum()),
        open_frames=len(u),
        unseen_mean=round(float(u_.mean()), 4),
        unseen_p95=round(float(np.percentile(u_, 95)), 4),
        unseen_max=round(float(u_.max()), 4),
        frames_over_5pct=int((u_ > 0.05).sum()),
    )


def source_test(take, render, take_t=None, render_t=None, frames=None):
    """the real interior's source test on one shot: the take's self-residual and, given a render, reveal against seen."""
    A = read_clip(take, t=take_t, n=frames)
    TA = track(A)
    out = dict(residual=interior_residual(A, TA))
    if render:
        B = read_clip(render, t=render_t, n=frames)
        TB = track(B)
        CA = canon(A, TA)
        ok = next(iter(CA))
        out["reveal"] = reveal(TA, CA, TB, canon(B, TB, TA[ok][STABLE]), A[0].shape)
    return out


def load_mask(path, size):
    if not path or not os.path.exists(os.path.expanduser(path)):
        return None
    m = np.load(os.path.expanduser(path))["masks"]
    return [cv2.resize(x, size, interpolation=cv2.INTER_LINEAR) > 64 for x in m]


# ---------------------------------------------------------------------------------------------------------- batch
def run_shot(s, work):
    """one manifest entry -> its result dict. Any missing input is recorded, never dropped."""
    r = {k: s.get(k) for k in ("label", "family", "group", "note")}
    j = lambda *a: os.path.join(work, *a)
    n = s.get("frames")
    A = read_clip(s["take"], s.get("take_start", 0), n, s.get("take_t"))
    if not A:
        r["error"] = f"no frames in {s['take']}"
        return r
    TA = cached_track(A, f"{s['take']}:{s.get('take_start', 0)}:{n}:{s.get('take_t')}", work)
    B = TB = None
    if s.get("render"):
        B = read_clip(s["render"], s.get("render_start", 0), n, s.get("render_t"))
        TB = cached_track(B, f"{s['render']}:{s.get('render_start', 0)}:{n}:{s.get('render_t')}", work)
    r["stats"] = stats(A, TA, B, TB)
    if B is not None:
        H, W = A[0].shape[:2]
        m = min(len(A), len(B))
        r["spectrum"] = spectrum(A[:m], TA[:m], B[:m], TB[:m], load_mask(s.get("mask"), (W, H)))
    if s.get("stages"):
        r["stages"] = stages([(x["stage"], x["video"], x.get("start", 0), x.get("frames"), x.get("t")) for x in s["stages"]], work)
    if s.get("null"):
        q = s["null"]
        out = j("null", s["label"].replace("/", "_") + ".mp4")
        try:
            if not os.path.exists(out):
                render_null(q, out)
            Rf = read_clip(q["reference"]) if q.get("reference") else A
            Tf = cached_track(Rf, q["reference"] + ":null", work) if q.get("reference") else TA
            X = read_clip(out)
            r["null_test"] = dict(
                reference=q.get("reference") or s["take"],
                fps=q.get("fps", 25.0),
                **({"ref_note": q["ref_note"]} if q.get("ref_note") else {}),
                **null_test(Rf, X, Tf, q.get("fps", 25.0)),
            )
            n2 = min(len(Rf), len(X), len(Tf))
            r["null_stats"] = video_stats(X[:n2], cached_track(X[:n2], out, work))
            r["null_stats_ref"] = video_stats(Rf[:n2], Tf[:n2])
        except ff.CalledProcessError as e:
            tail = open(out + ".log").read()[-400:]
            r["null_test"] = dict(error=f"paint failed ({e.returncode})", log_tail=tail)
    return r


NOTES = (  # take -> render changes worth a look; reported, they move no band
    ("cavity void", lambda t, r: _past(r, "cav_black_frac", t, 0.2)),
    ("dim teeth", lambda t, r: _past(t, "teeth_L", r, 10.0)),
    ("boil", lambda t, r: _past(r, "boil_loose", t, 0.0, 1.5)),
    ("over-sharp", lambda t, r: _past(r, "edge_inner", t, 0.0, 1.5)),
)
GRAIN_LOST = 0.7


def _past(a, k, b, add=0.0, mul=1.0):
    """a[k] above b[k] * mul + add, both measured. A boil dict reads its cavity value."""
    x, y = a.get(k), b.get(k)
    if isinstance(x, dict):
        x, y = x.get("cav"), (y or {}).get("cav")
    return x is not None and y is not None and x > y * mul + add


def shot_row(take, render, stage_items=(), work=None, frames=None):
    """one dubbed shot -> the node's row: [take, render] of each ruler, grain_gap, per-stage sharpness, notes."""
    A, B = read_clip(take, 0, frames), read_clip(render, 0, frames)
    if not A or not B:
        return dict(note="no frames")
    m = min(len(A), len(B))
    A, B = A[:m], B[:m]
    TA, TB = cached_track(A, f"{take}:{m}", work), cached_track(B, f"{render}:{m}", work)
    s = stats(A, TA, B, TB)
    t, r = s["take"], s["render"]
    row = {k: [t.get(k), r.get(k)] for k in ("pink_frac", "cav_L50", "teeth_L", "edge_inner")}
    row["boil_loose"] = [t["boil_loose"].get("cav"), r["boil_loose"].get("cav")]
    row["grain_gap"] = spectrum(A, TA, B, TB).get("grain_gap")
    row["stages"] = stages(stage_items, work)
    gap = row["grain_gap"]
    row["notes"] = [k for k, f in NOTES if f(t, r)] + (["grain lost"] if gap is not None and gap < GRAIN_LOST else [])
    return row


def rows_md(rows):
    """the node's table: one row per shot, take -> render."""
    L = [
        "# LSDub mouth ruler",
        "",
        "Take -> render, per dubbed shot. Reported only: nothing here moves a band.",
        "",
        "| shot | pink | cavity L* | teeth L* | inner edge | boil loose | grain gap | sharpness by stage | notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ar = lambda v: f"{f2(v[0])} → {f2(v[1])}"
    for r in rows:
        if "pink_frac" not in r:
            L.append(f"| {r['shot']} | {r.get('note', '-')} |" + " |" * 7)
            continue
        st = "; ".join(f"{x['stage']} {f2(x.get('lap_ref'), 0)}" for x in r["stages"] if "missing" not in x)
        L.append(
            f"| {r['shot']} | {ar(r['pink_frac'])} | {ar(r['cav_L50'])} | {ar(r['teeth_L'])} | {ar(r['edge_inner'])} | {ar(r['boil_loose'])} | "
            f"{f2(r['grain_gap'])} | {st} | {', '.join(r['notes'])} |"
        )
    return "\n".join(L) + "\n"


def f2(v, k=2):
    return "-" if v is None else f"{v:.{k}f}"


def pair(r, k, d=2):
    s = r.get("stats", {})
    a, b = s.get("take", {}).get(k), s.get("render", {}).get(k)
    return f"{f2(a, d)} → {f2(b, d)}" if "render" in s else f"{f2(a, d)} (take only)"


def summary(res, ref=None):
    L = [
        "# Mouth baseline",
        "",
        'Written by `lsdub mouth-ruler run`. Definitions: `lsdub/audits/mouth_ruler.py`. "A → B" is take → render.',
        "",
        "## Take vs render",
        "",
        "| shot | n open | pink | cavity L* | black | teeth L* | lip a* | lip b* | inner edge | lip detail | cavity boil (n) | boil loose (n) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def bl(r, k):
        s = r.get("stats", {})
        a, b = s.get("take", {}).get(k, {}), s.get("render", {}).get(k)
        return f"{f2(a.get('cav'), 1)} ({a.get('n')})" + (f" → {f2(b.get('cav'), 1)} ({b.get('n')})" if b else "")

    for r in res:
        if "stats" not in r:
            L.append(f"| {r['label']} | error: {r.get('error')} |" + " |" * 10)
            continue
        s = r["stats"]
        o = s["take"]["open_frames"]
        o = f"{o}/{s['render']['open_frames']}" if "render" in s else o
        L.append(
            f"| {r['label']} | {o} | {pair(r, 'pink_frac')} | {pair(r, 'cav_L50', 1)} | {pair(r, 'cav_black_frac')} | {pair(r, 'teeth_L', 1)} | {pair(r, 'lip_a', 1)} | "
            f"{pair(r, 'lip_b', 1)} | {pair(r, 'edge_inner')} | {pair(r, 'lip_hf')} | {bl(r, 'boil')} | {bl(r, 'boil_loose')} |"
        )
    L += [
        "",
        "## Per-stage sharpness",
        "",
        "Laplacian variance in the mouth box. `ref` is resampled to the first stage's mouth width, so read across a row with it.",
        "",
        "| shot | stage: mouth px, own → ref |",
        "|---|---|",
    ]
    for r in res:
        if r.get("stages"):
            L.append(
                f"| {r['label']} | "
                + "; ".join(
                    f"{x['stage']}: missing"
                    if "missing" in x
                    else f"{x['stage']}: {f2(x['w'], 0)} px, {f2(x['lap_own'], 0)} → {f2(x['lap_ref'], 0)}"
                    for x in r["stages"]
                )
                + " |"
            )
    L += [
        "",
        "## Null test",
        "",
        "Own mouth through the stack (`--warp-target own`, no watermark) vs the reference; floor = the reference re-encoded at CRF 14.",
        "",
        "| shot | PSNR mouth, null (floor) | SSIM mouth, null (floor) | ΔE mouth, null (floor) | PSNR interior | ΔE interior | Laplacian ratio | pink ref → null | cavity L* ref → null |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ref_notes = []
    for r in res:
        q = r.get("null_test")
        if not q:
            continue
        if "error" in q:
            L.append(f"| {r['label']} | {q['error']} |" + " |" * 7)
            continue
        if q.get("ref_note"):
            ref_notes.append(q["ref_note"])
        mark = " †" if q.get("ref_note") else ""
        a, b = q["null"], q["floor"]
        m, f = a["mouth"], b["mouth"]
        i = a["interior"] or {}
        x, y = r["null_stats_ref"], r["null_stats"]
        L.append(
            f"| {r['label']}{mark} | {f2(m['psnr'], 1)} ({f2(f['psnr'], 1)}) | {f2(m['ssim'], 3)} ({f2(f['ssim'], 3)}) | {f2(m['de'], 2)} ({f2(f['de'], 2)}) | {f2(i.get('psnr'), 1)} | {f2(i.get('de'), 2)} | "
            f"{f2(a['lap_ratio'], 2)} | {f2(x['pink_frac'])} → {f2(y['pink_frac'])} | {f2(x['cav_L50'], 1)} → {f2(y['cav_L50'], 1)} |"
        )
    if ref_notes:
        L += [""] + [f"† {t}" for t in ref_notes]
    L += [
        "",
        "## Mouth vs cheek",
        "",
        "gap_db per band (cycles/px); 0 = the mouth sits against its cheek as the take's does. grain gap = render ratio / take ratio.",
        "",
        "| shot | " + " | ".join(f"{a}-{b}" for a, b in BANDS) + " | grain take → render | grain gap | frames take/render |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res:
        q = r.get("spectrum")
        if not q:
            continue
        if "note" in q:
            L.append(f"| {r['label']} | {q['note']} |" + " |" * 7)
            continue
        L.append(
            f"| {r['label']} | "
            + " | ".join(f2(v, 1) for v in q["gap_db"])
            + f" | {f2(q['grain_take'])} → {f2(q['grain_render'])} | {f2(q['grain_gap'])} | {q['frames'][0]}/{q['frames'][1]} |"
        )
    if ref:
        L += [
            "",
            "## Against the audit",
            "",
            "Stored audit value vs this run, not adjusted. Δ = this run - audit.",
            "",
            "| shot | metric | audit take → render | this run | Δ take, Δ render |",
            "|---|---|---|---|---|",
        ]
        by = {r["label"]: r for r in res}
        for lab_, q in ref.items():
            if lab_ not in by or "stats" not in by[lab_]:
                continue
            s = by[lab_]["stats"]
            for k, v in q.items():
                a, b = s["take"].get(k), s["render"].get(k)
                if None in (a, b, v[0], v[1]):
                    L.append(f"| {lab_} | {k} | {v[0]} → {v[1]} | {a} → {b} | - |")
                    continue
                L.append(f"| {lab_} | {k} | {v[0]} → {v[1]} | {f2(a)} → {f2(b)} | {a - v[0]:+.2f}, {b - v[1]:+.2f} |")
    return "\n".join(L) + "\n"


def run_manifest(path, out, labels=None, jobs=4, reference=None):
    M = json.load(open(path))
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "work")
    shots = [s for s in M["shots"] if not labels or s["label"] in labels]
    part = os.path.join(work, "parts")
    os.makedirs(part, exist_ok=True)

    def one(s):
        p = os.path.join(part, s["label"].replace("/", "_") + ".json")
        ff.python(
            None,
            "lsdub_engine.audits.mouth_ruler",
            [
                "shot",
                "--manifest",
                path,
                "--label",
                s["label"],
                "--work",
                work,
                "--json",
                p,
            ],
            check=True,
            env=cpu_env(),
        )
        return json.load(open(p))

    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(one, shots))
    return res


def cut_stats(video, shots_json, work=None):
    """a Verify shots.json -> {shot: stats of the delivered frames against the cell's take}; a shot with no cell or no take is left out."""
    out = {}
    for s in json.load(open(shots_json))["shots"]:
        wj = s.get("warp_json")
        take = os.path.join(wj.split(os.sep + "work" + os.sep)[0], "take.mp4") if wj else ""
        if not os.path.exists(take):
            continue
        n = int(s["f1"]) - int(s["f0"]) + 1
        A = read_clip(take, 0, n)
        B = read_clip(video, int(s["f0"]), n)
        m = min(len(A), len(B))
        out[f"shot_{s['shot']}"] = stats(
            A[:m], cached_track(A[:m], f"{take}:{m}", work), B[:m], cached_track(B[:m], f"{video}:{s['f0']}:{m}", work)
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lsdub mouth-ruler")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a_ = sp.add_parser("stats")
    a_.add_argument("--take", required=True)
    a_.add_argument("--render")
    a_.add_argument("--take-t", type=float)
    a_.add_argument("--render-t", type=float)
    a_.add_argument("--frames", type=int)
    a_.add_argument("--mask")
    a_.add_argument("--work")
    a_.add_argument("--json")
    b_ = sp.add_parser("stages")
    b_.add_argument("items", nargs="+", help="label=video[@start[+frames]]")
    b_.add_argument("--work")
    b_.add_argument("--json")
    c_ = sp.add_parser("null")
    c_.add_argument("--reference", required=True)
    c_.add_argument("--null", required=True)
    c_.add_argument("--frames", type=int)
    c_.add_argument("--fps", type=float, default=25.0)
    c_.add_argument("--work")
    c_.add_argument("--json")
    d_ = sp.add_parser("run")
    d_.add_argument("--manifest", required=True)
    d_.add_argument("--out", required=True)
    d_.add_argument("--labels")
    d_.add_argument("--jobs", type=int, default=4)
    d_.add_argument("--reference")
    d_.add_argument("--publish", help="directory for baseline.json + baseline.md (default: --out)")
    f_ = sp.add_parser("source")
    f_.add_argument("--take", required=True)
    f_.add_argument("--render")
    f_.add_argument("--take-t", type=float)
    f_.add_argument("--render-t", type=float)
    f_.add_argument("--frames", type=int)
    f_.add_argument("--json")
    e_ = sp.add_parser("shot")
    e_.add_argument("--manifest", required=True)
    e_.add_argument("--label", required=True)
    e_.add_argument("--work", required=True)
    e_.add_argument("--json", required=True)
    g_ = sp.add_parser("cut")
    g_.add_argument("--video", required=True)
    g_.add_argument("--shots", required=True, help="a Verify shots.json")
    g_.add_argument("--work")
    g_.add_argument("--json")
    a = ap.parse_args(argv)

    def emit(r, p):
        if p:
            json.dump(r, open(p, "w"), indent=1)
        else:
            print(json.dumps(r, indent=1))

    if a.cmd == "cut":
        emit(cut_stats(a.video, a.shots, a.work), a.json)
        return 0
    if a.cmd == "stats":
        A = read_clip(a.take, 0, a.frames, a.take_t)
        TA = cached_track(A, f"{a.take}:{a.frames}:{a.take_t}", a.work)
        B = TB = None
        if a.render:
            B = read_clip(a.render, 0, a.frames, a.render_t)
            TB = cached_track(B, f"{a.render}:{a.frames}:{a.render_t}", a.work)
        r = stats(A, TA, B, TB)
        if B is not None:
            m = min(len(A), len(B))
            r["spectrum"] = spectrum(A[:m], TA[:m], B[:m], TB[:m], load_mask(a.mask, A[0].shape[1::-1]))
        emit(r, a.json)
    elif a.cmd == "stages":
        I = []
        for x in a.items:
            lab_, v = x.split("=", 1)
            st, n = 0, None
            if "@" in v:
                v, q = v.rsplit("@", 1)
                st, _, m = q.partition("+")
                st = int(st)
                n = int(m) if m else None
            I.append((lab_, v, st, n, None))
        emit(stages(I, a.work), a.json)
    elif a.cmd == "null":
        R = read_clip(a.reference, 0, a.frames)
        X = read_clip(a.null, 0, a.frames)
        emit(null_test(R, X, cached_track(R, a.reference + ":null", a.work), a.fps), a.json)
    elif a.cmd == "source":
        emit(source_test(a.take, a.render, a.take_t, a.render_t, a.frames), a.json)
    elif a.cmd == "shot":
        s = next(x for x in json.load(open(a.manifest))["shots"] if x["label"] == a.label)
        json.dump(run_shot(s, a.work), open(a.json, "w"), indent=1)
    else:
        res = run_manifest(a.manifest, a.out, set(a.labels.split(",")) if a.labels else None, a.jobs)
        p = a.publish or a.out
        os.makedirs(p, exist_ok=True)
        json.dump(res, open(os.path.join(p, "baseline.json"), "w"), indent=1)
        open(os.path.join(p, "baseline.md"), "w").write(summary(res, json.load(open(a.reference)) if a.reference else None))
        print("mouth-ruler:", len(res), "shots ->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
