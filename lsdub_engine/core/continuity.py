#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""continuity.py: `lsdub continuity`, the last-4-frames continuity maker / checker / confirmation for a painted mouth. CPU.

A single bad painted frame (an aliased, jumped or smeared mouth) ruins the shot even when its neighbours are fine. This
looks at each frame against the frames before it, only where the paint is (the region where the render differs from
the take, grown, over the window), and only for change the take itself does not have:

  checker       per frame i, the last K (4) output frames are each carried to frame i along the TAKE's own motion
                (DIS flow, take_j -> take_i), their median is the prediction; err_i = mean |render_i - prediction| minus
                the same for the take (the take's own change is allowed). Plus an aliasing ruler: the paint's high-frequency
                energy (Laplacian) over the take's, vs its K-frame median. Each is a robust z over the shot (median / MAD);
                a frame is FLAG above --z on either, with an absolute floor so a calm shot does not flag noise.
  maker         a flagged frame is re-made from its nearest unflagged neighbours (flow interpolation at its own time,
                core/retime.interp), blended in only over the paint region (feathered): the face stays the take's.
                Runs of more than --max-run flagged frames are left alone (a real change, not a glitch).
  coherence     two-sided: a frame or a run (up to 8) whose mouth is unlike BOTH sides while the sides agree is re-made:
                flow interpolation from the sides vs each --alt render, the most coherent wins; re-checked, up to
                --odd-rounds times.
  confirmation  the checker again on the fixed render; frames still flagged are reported.
  QA            --json report; --strip writes take | before | after crops of every fixed frame (png).

  lsdub continuity --take take.mp4 --render ours.mp4 --out fixed.mp4 [--k 4] [--z 3.5] [--json r.json] [--strip s.png]
"""
import argparse, json
from ..io import ff
import cv2, numpy as np
from .retime import _read, interp

DIS = None


def _dis():
    global DIS
    if DIS is None: DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return DIS


def _carry(img, g_from, g_to):
    """img (aligned with g_from) moved onto g_to's geometry along the take's flow."""
    fl = _dis().calc(g_to, g_from, None)                               # for each pixel of g_to: where it was in g_from
    h, w = g_to.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def paint_region(T, R, thr=10, grow=12, near=None, track=True):
    """per frame bool mask: where the render differs from the take (the paint), grown; unioned with its neighbours.
    The blob is TRACKED: each frame keeps the component nearest the shot's (median-filtered) paint centre, so a moving
    arm or a re-encode's noise never wins over the mouth. near: masks from the original render to pick inside."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    comps = []
    for i, (t, r) in enumerate(zip(T, R)):                            # the solid paint blobs, not scattered codec noise
        D = cv2.GaussianBlur(np.abs(r.astype(np.float32) - t.astype(np.float32)).max(-1), (0, 0), 2.0)
        d = D > thr
        if near is not None: d &= cv2.dilate(near[i].astype(np.uint8), k) > 0
        nl, lab, st, cen = cv2.connectedComponentsWithStats(d.astype(np.uint8))
        mass = np.bincount(lab.ravel(), weights=D.ravel(), minlength=nl)
        mass[0] = 0
        comps.append((lab, mass, cen))
    pick = [int(np.argmax(m)) if len(m) > 1 and m.max() > 0 else 0 for _, m, _ in comps]
    if track and len(T) > 2:
        c = np.array([cen[p] if p else (np.nan, np.nan) for (_, _, cen), p in zip(comps, pick)])
        ref = np.array([np.nanmedian(c[max(0, i - 6):i + 7], 0) for i in range(len(c))])
        for i, (lab, mass, cen) in enumerate(comps):
            if len(mass) < 2 or np.isnan(ref[i]).any(): continue
            big = np.where(mass > 0.15 * mass.max())[0]               # a real blob, the one nearest where the mouth is
            pick[i] = int(big[np.argmin(np.linalg.norm(cen[big] - ref[i], axis=1))])
    M = [cv2.dilate((lab == p).astype(np.uint8), k) if p else np.zeros(lab.shape, np.uint8) for (lab, _, _), p in zip(comps, pick)]
    return [np.maximum.reduce(M[max(0, i - 1):i + 2]) > 0 for i in range(len(M))]


def _box(m, pad=8):
    ys, xs = np.where(m)
    if len(ys) < 20: return None
    h, w = m.shape
    return max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)


def _hf(img):
    return float(np.abs(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)).mean())


BLUE_D, BLUE_SOFT, BLUE_BG = 12, 6, 8
# A ghost pixel: clearly blue (B - R > 12), OR faintly blue AND blue over green (B - R > 6 and B - G > 8: a true
# blue-grey, BGR ~155,143,142 on sky shot B). Cool-lit teeth / a chrome mic are neutral or cyan (B - G 1-7) and only
# count when clearly blue. (reco14: B - R > 6 alone flagged 11C's teeth and reverted good warp frames; B - R > 12 alone
# lost shot B f91, whose ghost sits entirely at B - R 7-10.)
DESYNC_FRAC = 0.25   # |take gap - planned egap| above this x the plan's p5-p95 range: the take is off the plan there
                     # and is no re-make candidate (calibrated on 11C only; on sky shot B it holds on 68% of frames)
BLUE_FLOOR = 12   # more ghost px than this in the paint region = a ghosted frame (check and the fix_alt fallback)


def ghost_mask(t, r, m):
    """bool: px of the paint region that read blue-grey in the render (see BLUE_*) and are not blue in the take."""
    t = t.astype(np.int16)
    r = r.astype(np.int16)
    bt = t[..., 0] - t[..., 2]
    br = r[..., 0] - r[..., 2]
    bg = r[..., 0] - r[..., 1]
    return ((br > BLUE_D) | ((br > BLUE_SOFT) & (bg > BLUE_BG))) & (bt <= 0) & m


def blue_px(t, r, m):
    """ghost px count: a lip / teeth layer ghost (sky B f79-81, f91-94) or a warped-in prop (a mic over the lips).
    Skin, lips and the cavity have R >= B. Teeth may sit a few levels blue under cool light, so a faintly blue px
    (B - R 7-12) counts only when it is also clearly bluer than green (blue-grey); see ghost_mask."""
    return int(ghost_mask(t, r, m).sum())


def measure(T, R, M, k=4):
    """-> per frame (excess prediction error, aliasing ratio); NaN where there is no paint or no history."""
    n = len(T)
    err = np.full(n, np.nan)
    hf = np.full(n, np.nan)
    for i in range(n):
        b = _box(M[i])
        if b is None: continue
        y0, y1, x0, x1 = b
        m = M[i][y0:y1, x0:x1]
        gt = lambda j: cv2.cvtColor(T[j][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        hf[i] = _hf(R[i][y0:y1, x0:x1]) / max(1e-3, _hf(T[i][y0:y1, x0:x1]))
        if i == 0: continue
        gi = gt(i)
        pr, pt = [], []
        for j in range(max(0, i - k), i):
            gj = gt(j)
            pr.append(_carry(R[j][y0:y1, x0:x1].astype(np.float32), gj, gi))
            pt.append(_carry(T[j][y0:y1, x0:x1].astype(np.float32), gj, gi))
        er = np.abs(R[i][y0:y1, x0:x1] - np.median(pr, 0)).mean(-1)[m].mean()
        et = np.abs(T[i][y0:y1, x0:x1] - np.median(pt, 0)).mean(-1)[m].mean()
        err[i] = er - et
    return err, hf


def _z(x, k=4):
    """robust z of each value against the shot; the aliasing ratio against its own K-frame past median first."""
    v = x[~np.isnan(x)]
    if len(v) < 5: return np.zeros_like(x)
    mad = np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-6
    return np.nan_to_num((x - np.median(v)) / mad)


def check(T, R, k=4, z=3.5, err_floor=4.0, hf_floor=0.25, grow_z=None, blue_floor=BLUE_FLOOR, M=None):
    M = paint_region(T, R) if M is None else M
    err, hf = measure(T, R, M, k)
    hfr = np.array([hf[i] - np.nanmedian(hf[max(0, i - k):i]) if i and not np.isnan(hf[i]) and not np.all(np.isnan(hf[max(0, i - k):i])) else 0.0
                    for i in range(len(hf))])
    ze, zh = _z(err), _z(hfr)
    blue = np.array([blue_px(T[i], R[i], M[i] > 0) for i in range(len(T))])
    motion = ((ze > z) & (np.nan_to_num(err) > err_floor)) | ((zh > z) & (hfr > hf_floor))   # glitch / aliasing
    flag = motion | (blue > blue_floor)
    core = flag.copy()
    if grow_z is not None:                                            # hysteresis: a flag takes in the whole bad stretch
        for i in np.where(core)[0]:
            for step in (-1, 1):
                j = i + step
                while 0 <= j < len(flag) and ze[j] > grow_z:
                    flag[j] = True
                    j += step
    return flag, dict(err=err, hf=hf, hfr=hfr, ze=ze, zh=zh, blue=blue, core=core, motion=motion), M


def _dev(T, X, M, i, L, a, b, cache=None):
    """how unlike its two sides frame i is: the frames a (before the run) and b (after) carried to i along the take's
    flow; -> (dev excess over the take's own, how much the two sides disagree, excess)."""
    bx = _box(M[i])
    if bx is None: return np.nan, np.nan
    y0, y1, x0, x1 = bx
    m = M[i][y0:y1, x0:x1]
    gt = lambda j: cv2.cvtColor(T[j][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gi = gt(i)
    c = lambda Y, j: _carry(Y[j][y0:y1, x0:x1].astype(np.float32), gt(j), gi)
    p90 = lambda D: float(np.percentile(np.abs(D).mean(-1)[m], 90))
    w = (i - a) / (b - a)
    ta, tb, xa, xb = c(T, a), c(T, b), c(X, a), c(X, b)
    dv = p90(X[i][y0:y1, x0:x1] - ((1 - w) * xa + w * xb)) - p90(T[i][y0:y1, x0:x1] - ((1 - w) * ta + w * tb))
    return dv, max(0.0, p90(xa - xb) - p90(ta - tb))


def odd_frames(T, R, M, skip=(), z=2.0, ratio=1.0, floor=3.5, runs=tuple(range(1, 11))):
    """the two-sided coherence ruler: a frame, or a run of up to 8, whose mouth is unlike BOTH sides while the sides
    agree with each other (a popped-in teeth slab, a washed-out mouth, a dropped-looking frame). For a run i..i+L-1 the
    frames before (i-1) and after (i+L) predict each frame; dev = p90 |render - prediction| minus the take's own.
    -> list of runs (start, length, mean dev), non-overlapping, worst first; and the per-frame single-step dev."""
    n = len(T)
    one = np.full(n, np.nan)
    for i in range(1, n - 1): one[i] = _dev(T, R, M, i, 1, i - 1, i + 1)[0]
    v = one[~np.isnan(one)]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    cand = []
    for L in runs:
        for i in range(1, n - L):
            if np.nanmax(one[i:i + L] if L > 1 else one[i:i + 1]) < med + 1.0 * mad and L > 1: continue
            if L == 1 and not (one[i] > floor and (one[i] - med) / mad > z): continue
            d = [_dev(T, R, M, t, L, i - 1, i + L) for t in range(i, i + L)]
            dv = np.array([x[0] for x in d])
            ag = np.nanmax([x[1] for x in d])
            if np.isnan(dv).any(): continue
            if dv.mean() > floor and (dv.mean() - med) / mad > z and dv.mean() > ratio * (ag + 1.0) and dv.min() > 0.2 * dv.mean():
                cand.append((i, L, float(dv.mean())))
    cand.sort(key=lambda r: (-r[2] * r[1] ** 0.5))
    used = np.zeros(n, bool)
    out = []
    for i, L, d in cand:
        if (i, L) in skip or used[i:i + L].any(): continue
        used[i:i + L] = True
        out.append((i, L, round(d, 2)))
    return out, one


def remake(T, R, runs, M, alts=(), feather=5, only=None, desync=None):
    """the maker for odd runs: candidates are the flow interpolation from the two sides and every alternate render;
    each is scored on the same two-sided ruler and the most coherent one is blended in over the mouth. -> (frames, log)."""
    out = list(R)
    log = []
    for i, L, d in runs:
        a, b = i - 1, i + L
        best = None
        cands = {}                                                    # a blip (<= 3) is blended from its sides; a longer run is
        if L <= 3 or not alts:                                        # a held state (teeth in a smile) unless a re-render beats it
            cands['interp'] = [interp(R[a], R[b], (t - a) / (b - a), _dis()) for t in range(i, i + L)]
        if L > 3 and not alts:
            log.append(dict(start=i, length=L, dev=d, used='kept', dev_after=d))
            continue
        for k, A in enumerate(alts): cands[f'alt{k}'] = [A[t] for t in range(i, i + L)]
        if L <= 3 and (desync is None or not desync[i:i + L].any()):  # last rung: her own mouth (never pops), but
            cands['take'] = [T[t] for t in range(i, i + L)]           # not where the take's opening is off the plan
        if only: cands = {only: cands[only]}
        for name, F in cands.items():
            X = list(R)
            X[i:i + L] = F
            sc = float(np.nanmean([_dev(T, X, M, t, L, a, b)[0] for t in range(i, i + L)]))
            if best is None or sc < best[1]: best = (name, sc, F)
        for t, f in zip(range(i, i + L), best[2]):
            m = cv2.GaussianBlur(M[t].astype(np.float32), (0, 0), feather)[..., None]
            out[t] = (R[t] * (1 - m) + f * m + 0.5).astype(np.uint8)
        log.append(dict(start=i, length=L, dev=d, used=best[0], dev_after=round(best[1], 2)))
    return out, log


def texture_pops(T, R, A, M, hi=6.0, lo=5.0, max_len=10):
    """a striped / aliased teeth slab that pops in: the render's mouth detail (98th pct |Laplacian|) over the plain
    alternate's, both minus the take's. A short burst (> hi, grown while > lo, <= max_len frames, not running into the
    shot's end) is a pop; a gradual ramp of real teeth is not. -> list of (start, length)."""
    n = len(R)
    def hf(img, m): return float(np.percentile(np.abs(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F))[m], 98))
    d = np.zeros(n)
    for i in range(n):
        b = _box(M[i])
        if b is None: continue
        y0, y1, x0, x1 = b
        m = M[i][y0:y1, x0:x1]
        d[i] = hf(R[i][y0:y1, x0:x1], m) - hf(A[i][y0:y1, x0:x1], m)
    out, i = [], 0
    while i < n:
        if d[i] <= hi:
            i += 1
            continue
        a = i
        while a > 0 and d[a - 1] > lo: a -= 1
        b = i
        while b + 1 < n and d[b + 1] > lo: b += 1
        if b - a + 1 <= max_len and b < n - 2: out.append((a, b - a + 1))
        i = b + 1
    return out


LIP_ZONE = 0.55   # the upper share of the paint region's rows: where a squeezed lip / teeth ghost lives (a prop the
                  # closing jaw uncovers, e.g. the 19 / 29 mic grille, sits lower)


def lipzone_ghost(t, r, m, frac=LIP_ZONE):
    """ghost px (ghost_mask) in the upper `frac` of the paint region's rows only."""
    g = ghost_mask(t, r, m)
    ys = np.where(m)[0]
    if not len(ys): return 0
    g[int(ys.min() + frac * (ys.max() - ys.min())):] = False
    return int(g.sum())


def squeeze_with_ghost(sq, T, R, M, floor=BLUE_FLOOR):
    """A squeeze run (take open, plan shut) is only a reason to put the take back when the render SHOWS the squeeze
    ghost somewhere in that run: ghost px in the lip zone over `floor`. Without it, the shut mouth is the sync the
    audio asks for (v3_4 cuts 19 / 29: the bare squeeze rule reverted exactly the sung closures, paint 0.96 -> cont
    0.09). -> (kept squeeze mask, per-frame lip-zone ghost px)."""
    n = len(sq)
    lz = np.array([lipzone_ghost(T[i], R[i], M[i] > 0) if sq[i] else 0 for i in range(n)])
    out = np.zeros(n, bool)
    i = 0
    while i < n:
        if not sq[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and sq[j + 1]: j += 1
        if (lz[i:j + 1] > floor).any(): out[i:j + 1] = True
        i = j + 1
    return out, lz


def squeeze_frames(warp_json, n, open_frac=0.4, keep=0.35):
    """from the paint's warp json: frames where the take's mouth is open (gap > open_frac x its range) and the plan shuts
    it to < keep x that. An open mouth (teeth showing) squeezed shut by the warp drags the teeth into the lip line: a
    grey ghost lip (sky B f78-95). Those frames are better with the take's own mouth."""
    J = json.load(open(warp_json))
    W = J['frames'][:n]
    hi = J['range']['g_hi']
    g = np.array([f['gap'] for f in W])
    e = np.array([f['egap'] for f in W])
    out = np.zeros(n, bool)
    out[:len(W)] = (g > open_frac * hi) & (e < keep * g)
    return out


def fix_alt(T, R, A, flag, M, ease=2, feather=5, take_frames=None, blue_floor=BLUE_FLOOR, keep=None):
    """flagged frames take the ALTERNATE render (e.g. the plain own-pixel warp) over the paint region, eased in and out
    over `ease` frames each side so the switch never pops. -> (frames, weights)."""
    n = len(R)
    out = list(R)
    MA = paint_region(T, A, near=M) if A is not None else M
    src = {}
    mmask = lambda t: np.maximum(M[t], MA[t]) > 0
    squeeze = lambda t: take_frames is not None and bool(take_frames[t])
    alt_ok = lambda t: not squeeze(t) and A is not None and blue_px(T[t], A[t], mmask(t)) <= blue_floor
    flag = np.asarray(flag, bool).copy()
    for t in np.where(flag)[0]:                                     # ladder: alternate -> our own clean frame (only grown
        if not alt_ok(t) and keep is not None and bool(keep[t]) and not squeeze(t):   # into the flag) -> the take. A kept
            src[int(t)] = 'render'
            flag[t] = False                  # frame is not replaced and spreads no ease (no blend ghost)
    w = flag.astype(np.float32)
    for i in np.where(flag)[0]:
        for d in range(1, ease + 1):
            for j in (i - d, i + d):
                if 0 <= j < n: w[j] = max(w[j], 1 - d / (ease + 1))
    for t in list(src):                                             # kept frames stay ours, and so do the unflagged frames
        for j in range(max(0, t - ease), min(n, t + ease + 1)):     # beside them: easing a different mouth in between two
            if not flag[j]: w[j] = 0.0                              # of our own frames only ghosts (11C f14)
    for t in np.where(w > 0)[0]:
        ok = alt_ok(t)
        S = A[t] if ok else T[t]
        src[int(t)] = 'alt' if ok else 'take'
        m = cv2.GaussianBlur(mmask(t).astype(np.float32), (0, 0), feather)[..., None] * w[t]
        out[t] = (R[t] * (1 - m) + S * m + 0.5).astype(np.uint8)
    return out, w, src


def fix(T, R, flag, M, max_run=3, feather=5):
    """flagged frames re-made from their unflagged neighbours inside the paint region. -> (frames, fixed idx, skipped runs)."""
    out = list(R)
    n = len(R)
    fixed, skipped = [], []
    i = 0
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1]: j += 1
        a, b = i - 1, j + 1
        if j - i + 1 > max_run or a < 0 and b >= n:
            skipped.append([i, j])
            i = j + 1
            continue
        for t in range(i, j + 1):
            if a < 0: new = R[b]
            elif b >= n: new = R[a]
            else: new = interp(R[a], R[b], (t - a) / (b - a), _dis())
            m = cv2.GaussianBlur(M[t].astype(np.float32), (0, 0), feather)[..., None]
            out[t] = (R[t] * (1 - m) + new * m + 0.5).astype(np.uint8)
            fixed.append(t)
        i = j + 1
    return out, fixed, skipped


def strip(T, R, F, idx, path, size=(220, 160)):
    rows = []
    for t in idx[:24]:
        M = paint_region([T[t]], [R[t]])[0]
        b = _box(M, 30)
        if b is None: continue
        y0, y1, x0, x1 = b
        r = np.hstack([cv2.resize(X[t][y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC) for X in (T, R, F)])
        cv2.putText(r, f'f{t}  take | before | after', (4, 16), 0, 0.45, (255, 255, 255), 1)
        rows.append(r)
    if rows: cv2.imwrite(path, np.vstack(rows))


def write(frames, fps, out, audio_src):
    h, w = frames[0].shape[:2]
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                          '-i', audio_src, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p',
                          '-c:a', 'copy', '-shortest', '-movflags', '+faststart', out], stdin=ff.PIPE)
    for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


def run(take, render, out, k=4, z=3.5, max_run=3, rounds=2, report=None, strip_png=None, alt=None, grow_z=1.5, warp_json=None, odd_rounds=3, squeeze=True, plan_tol=0.12, max_frac=0.10):
    T, fps = _read(take)
    R, _ = _read(render)
    n = min(len(T), len(R))
    T, R = T[:n], R[:n]
    flag0, s0, M = check(T, R, k, z, grow_z=grow_z if alt else None)
    sq = squeeze_frames(warp_json, n) if warp_json and squeeze else None
    sq_plan = None
    if sq is not None:
        sq_plan = sq
        sq, _ = squeeze_with_ghost(sq, T, R, M)          # only runs that show the ghost
        flag0 = flag0 | sq
    plan = None
    desync = None                                        # the mouth plan (egap): change it asks for is not a glitch
    if warp_json:
        J = json.load(open(warp_json))
        plan = np.array([f['egap'] for f in J['frames'][:n]], float)
        rng = max(1e-6, float(np.percentile(plan, 95) - np.percentile(plan, 5)))
        tgap = np.array([f['gap'] for f in J['frames'][:n]], float)
        desync = np.abs(tgap - plan) > DESYNC_FRAC * rng             # the take's own mouth is off the plan here
    cur, allfix, skipped = R, [], []
    if (alt or sq is not None) and flag0.any():                      # maker: alternate render / take over each bad stretch
        A = _read(alt)[0][:n] if alt else None
        keep = ~s0['core'] & (s0['blue'] <= BLUE_FLOOR)                     # grown-in frames whose own paint is clean
        if desync is not None:                                         # a PROP flag (ghost px only below the lip zone, no
            lz = np.array([lipzone_ghost(T[i], R[i], M[i] > 0) for i in range(n)])   # glitch) where the take is off the
            prop = s0['core'] & ~s0['motion'] & (lz <= BLUE_FLOOR) & desync           # plan: the take would kill the sync,
            if sq is not None: prop &= ~sq                                            # keep ours (occluder masks fix the
            keep = keep | prop                                                        # prop properly: GPU, lsdub occluders)
        cur, w, src = fix_alt(T, R, A, flag0, M, take_frames=sq, keep=keep)
        allfix = [t for t in np.where(w > 0)[0].tolist() if src.get(int(t)) != 'render']
        print('maker sources:', {k: v for k, v in src.items() if v != 'alt'} or 'all alt')
    elif flag0.any():                                                # maker: neighbour interpolation (short glitches)
        flag = flag0
        for r in range(rounds):
            if not flag.any(): break
            cur, fixed, sk = fix(T, cur, flag, M, max_run)
            allfix += fixed
            skipped += sk
            flag, _, _ = check(T, cur, k, z, M=M)
            flag &= ~np.isin(np.arange(n), [t for a, b in skipped for t in range(a, b + 1)])
    odd_log, odd_left = [], []                                        # coherence: a run unlike BOTH sides is re-made
    alts = [A0[:n] for A0 in ([_read(alt)[0]] if alt else [])]       # (interp / alternate render, the most coherent wins)
    tried = set()                                                     # a run already re-made is not retried: others get a turn
    touched, budget = set(), int(max_frac * n)                       # a checker touching >10% of a shot is wrong, not the shot
    def allowed(i, L):
        if plan is not None and i + L < len(plan):
            w = (np.arange(i, i + L) - (i - 1)) / (L + 1)
            if np.max(np.abs(plan[i:i + L] - ((1 - w) * plan[i - 1] + w * plan[i + L]))) > plan_tol * rng: return False
        new = set(range(i, i + L)) - touched
        if len(touched) + len(new) > budget: return False
        touched.update(new)
        return True
    for r in range(odd_rounds):                                       # then checked again, until none are left
        runs, _ = odd_frames(T, cur, M, tried)
        tried |= {(i, L) for i, L, _ in runs}
        runs = [r_ for r_ in runs if allowed(r_[0], r_[1])]
        if not runs: break
        print(f'coherence round {r + 1}: odd runs {[(i, L) for i, L, _ in runs]}')
        cur, lg = remake(T, cur, runs, M, alts, desync=desync)
        odd_log += lg
    else:
        odd_left = [[i, L] for i, L, _ in odd_frames(T, cur, M, tried)[0]] if odd_rounds else []
    if alts:                                                          # texture pops: the plain alternate over a slab burst
        for i, L in texture_pops(T, cur, alts[0], M):
            cur, lg = remake(T, cur, [(i, L, 0.0)], M, alts[:1], only='alt0')
            odd_log += lg
            print(f'texture pop: frames {i}-{i + L - 1} -> alt0')
    done = set()
    for r in range(odd_rounds):
      flag, _, _ = check(T, cur, k, z, M=M)                          # confirmation-driven: a frame the backward ruler still
      todo = [f for f in np.where(flag)[0] if f not in done]          # flags starts a run; its length is the one whose two
      if not todo: break
      for f0 in todo:                                                 # sides explain it worst (the whole bad stretch); after
        f, total = int(f0), 0                                         # a re-make, the stretch keeps going while the next frame
        while total < 10 and 1 <= f < n - 1 and f not in done:        # is still unlike its sides (a slab outlasting the run)
            done.add(f)
            best = None
            for L in range(1, 11 - total):
                if f + L >= n: break
                dv = np.nanmean([_dev(T, cur, M, t, L, f - 1, f + L)[0] for t in range(f, f + L)])
                if best is None or dv > best[1]: best = (L, dv)
            if not best or best[1] <= 3.5 or not allowed(f, best[0]): break
            cur, lg = remake(T, cur, [(f, best[0], round(float(best[1]), 2))], M, alts, desync=desync)
            odd_log += lg
            print(f'confirmation remake: frames {f}-{f + best[0] - 1} -> {lg[0]["used"]}')
            f += best[0]
            total += best[0]
            if f >= n - 1 or _dev(T, cur, M, f, 1, f - 1, f + 1)[0] <= 3.5: break
    flag, s1, _ = check(T, cur, k, z, M=M)                           # confirmation (same mouth region as the input)
    still = [int(t) for t in np.where(flag)[0] if not any(a <= t <= b for a, b in skipped)]
    write(cur, fps, out, render)
    e0, e1 = np.nan_to_num(s0['err']), np.nan_to_num(s1['err'])
    rep = dict(frames=n, k=k, z=z, alt=alt, coherence_frames=len(touched), coherence=odd_log, odd_left=odd_left, squeeze=[] if sq is None else np.where(sq)[0].tolist(), squeeze_plan_only=[] if sq_plan is None else np.where(sq_plan & ~sq)[0].tolist(), flagged=np.where(flag0)[0].tolist(), fixed=sorted(set(allfix)), skipped_runs=skipped,
               still_flagged=still, err_before_on_flagged=round(float(e0[flag0].mean()), 2) if flag0.any() else None,
               err_after_on_flagged=round(float(e1[flag0].mean()), 2) if flag0.any() else None,
               err=[None if np.isnan(v) else round(float(v), 2) for v in s0['err']], err_after=[None if np.isnan(v) else round(float(v), 2) for v in s1['err']],
               hf=[None if np.isnan(v) else round(float(v), 3) for v in s0['hf']], blue=s0['blue'].tolist(), blue_after=s1['blue'].tolist())
    print(f"continuity {render}: flagged {len(rep['flagged'])} {rep['flagged']}; fixed {len(rep['fixed'])} frames; "
          f"err on flagged {rep['err_before_on_flagged']} -> {rep['err_after_on_flagged']}; skipped runs {skipped}; "
          f"coherence: re-made {[(l['start'], l['length'], l['used']) for l in odd_log]}, left {odd_left}; "
          f"confirmation: {'PASS' if not still else 'FLAG ' + str(still)} -> {out}")
    if report: json.dump(rep, open(report, 'w'), indent=1)
    if strip_png and rep['fixed']: strip(T, R, cur, rep['flagged'] or rep['fixed'], strip_png)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', required=True)
    ap.add_argument('--render', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=4, help='frames of history the prediction uses')
    ap.add_argument('--z', type=float, default=3.5, help='robust z above which a frame is flagged')
    ap.add_argument('--max-run', type=int, default=3, help='longer flagged runs are a real change: left alone')
    ap.add_argument('--alt', help='an alternate render of the same shot (e.g. paint --warp-profile-plain 1.01, the plain own-pixel '
                    'warp): a bad stretch takes it over the mouth, eased in / out; without it short glitches are interpolated')
    ap.add_argument('--grow-z', type=float, default=1.5, help='with --alt: a flag grows over neighbours above this z (the whole stretch)')
    ap.add_argument('--warp-json', help='the paint\'s warp json: also flags SQUEEZE frames (an open take mouth planned shut, the '
                    'grey ghost lip); those take the take\'s own mouth')
    ap.add_argument('--odd-rounds', type=int, default=3, help='coherence passes: a frame unlike BOTH its neighbours is re-made '
                    'from them, then re-checked (0 = off)')
    ap.add_argument('--no-squeeze', action='store_true', help='with --warp-json: use it only as the motion plan (close-ups)')
    ap.add_argument('--plan-tol', type=float, default=0.12, help='a run whose planned opening moves more than this x the '
                    'shot\'s range is real singing: never re-made')
    ap.add_argument('--max-frac', type=float, default=0.10, help='the coherence pass touches at most this share of the shot')
    ap.add_argument('--json')
    ap.add_argument('--strip', help='png of take | before | after for every flagged frame')
    a = ap.parse_args(argv)
    run(a.take, a.render, a.out, a.k, a.z, a.max_run, report=a.json, strip_png=a.strip, alt=a.alt, grow_z=a.grow_z, warp_json=a.warp_json, odd_rounds=a.odd_rounds, squeeze=not a.no_squeeze, plan_tol=a.plan_tol, max_frac=a.max_frac)


if __name__ == '__main__':
    main()
