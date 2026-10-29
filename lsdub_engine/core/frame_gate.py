#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""frame_gate.py (core): the MAKER and the CONFIRMATION of the v9 per-frame gate (the CHECKER: audits/frame_gate.py).

Round 0 is the v8 render (mouth_warp.WarpClip.run_batch). The checker reads every frame against its neighbours and the
source take (PASS / SOFT / FAIL + causes, SLOSH windows). For each FAIL frame (worst first; SOFT too with
--warp-gate-soft) the maker re-makes the frame from ALTERNATIVES -- per-frame parameter overrides through
WarpClip.render_frame, never a blend of pixels from other frames -- and runs each through the checker against the
CURRENT neighbours:
  a lm_hold      the frame's input landmarks replaced: by an independent face-box re-detection of the take where the
                 input track is held / wrong (lm_err), else by the prediction from its neighbours (a one-frame MediaPipe
                 jump); the target features are kept (the residual is re-taken against the new landmarks). A held
                 input run is re-made as a whole (held_span), not only its failing frames
  b warp_soft    a softer warp for this frame: 0.85 of the residual, a wider falloff (core 0.7), 3x the TPS
                 regularisation, the fold guard at 0.3 (once: a softened frame is not softened again in a later round)
  c donor_hold   the interior source of t-1: its donors (donor fill) / its interior parameters (planned jaw, chin-pin
                 angle, opening share, tongue pose) for the layer stack
  d teeth_plan   the lower teeth from the jaw plan only (the chin pin's fast residual at gain 0)
  e layers_hold  the mouth cover with a wider hull (0.32) + the lip lock's colour state of t-1
  h layers_off   (v11, opt-in: add h to --warp-gate-cands) the plain own-pixel warp without the layer stack, tried after
                 a..e and before f
  f param_interp (last resort) the target features and every layer parameter interpolated between the nearest PASS
                 frames on both sides (parameters, never pixels)
  g slosh_fit    (a SLOSH window) the RESIDUAL tracks of the window (the take's landmark shape beyond the 2 target
                 features, the chin pin's extra angle, the upper-teeth pin, the lip lock state) replaced by a low-order
                 fit through its PASS anchors (linear and quadratic tried, the lower slosh kept); STRUCTURE (the target
                 gap / width, the planned jaw, the opening share) is never changed, and a closure frame (viseme PP)
                 inside the window is not re-made at all: it anchors the fit (piecewise between anchors)
CONFIRMATION: a candidate is kept only when frame t is no longer FAIL, passes every hard gate, and neither neighbour got
worse; the best gate score (severity over t-1, t, t+1) wins; none -> (f); (f) fails too -> the best is kept (only if it
beats the original; --warp-gate-unresolved none: never) and the frame is logged UNRESOLVED. Up to --warp-gate-rounds rounds. With --warp-gate-report the
final plan is saved beside the report (<report>_plan.npz) for the offline checker (lsdub frame-gate --plan)."""
import time
import numpy as np

from ..audits import frame_gate as chk
from . import mouth_warp as mw, mouth_paint as mp_, lip_layers as ll

RANK = {'PASS': 0, 'SOFT': 1, 'FAIL': 2}
HARD_CODES = ('fold', 'outside', 'ghost')


# ------------------------------------------------------------------------------------------- the checker's view --
def plan_of(clip, fold=True):
    n = clip.n
    jaw = [l.get('jaw_deg', np.nan) + l.get('jaw_dth_deg', 0.0) if 'jaw_deg' in l else np.nan for l in clip.log]
    pin = [np.asarray(clip.teeth.pin[i], float) if clip.teeth is not None else np.zeros(2) for i in range(n)]
    return dict(P_out=[np.asarray(p, float) for p in clip.P_out], fold=[clip.fold_det_of(i) if fold else 1.0 for i in range(n)],
                Fe=np.asarray(clip.Fe, float).copy(), F=np.asarray(clip.F, float).copy(), jaw=jaw, pin=pin,
                S_in=[np.asarray(x, float) for x in clip.S], held_fixed=[False] * n,
                skip=getattr(clip, 'vocal_skip', None) if getattr(clip, 'vocal_mode', None) == 'skip' else None,   # v10a: no audio target
                take=(np.asarray(clip.vocal_s) <= 0.0) if getattr(clip, 'vocal_mode', None) == 'skip' else None)    # v10a: the take's own


def is_take(clip, t):
    """v10a: frame t is the take's own (a skipped vocalisation at strength 0): nothing to re-make."""
    s = getattr(clip, 'vocal_s', None)
    return s is not None and t < len(s) and s[t] <= 0.0


def make_checker(clip, a, lmk=None):
    lip_ref = ll.parse_lip_ref(a.warp_lip_lock) if getattr(a, 'warp_lip_lock', None) else None
    lmk = lmk or chk.Landmarker()
    ck = chk.Checker(clip.frames, clip.S3raw, plan_of(clip), clip.model.rng, clip.S, clip.pl, lip_ref,
                     getattr(a, 'warp_gate_z', 3.0), getattr(a, 'warp_gate_slosh_w', 4), lmk)
    ck.leak_scene = bool(getattr(a, 'warp_gate_leak_scene', False))      # v11 (opt-in): the scene is not a leak
    raw = chk.landmark_cache(clip.frames, None, lmk, clip.S3raw, check=False)
    ck.take_lm, ck.redet_rejected = chk.filter_redet(raw, clip.S3raw)      # v11: a false face fit is never used
    if ck.redet_rejected: print('gate: implausible re-detection(s) of the take not used: ' + ', '.join(f"f{x['i']} (skull {x['skull']} IOD, spread {x['spread']})" for x in ck.redet_rejected))
    ck.observe_take()
    ck.observe_all(clip.outs, clip.masks)
    return ck


def snapshot(ck, idx):
    idx = sorted(set(i for i in idx if 0 <= i < ck.n))
    rel = sorted(set(j for i in idx for j in (i, i + 1) if 0 <= j < ck.n))
    return dict(obs={i: ck.obs[i] for i in idx}, rel={j: ck.rel[j] for j in rel},
                plan={k: {i: ck.plan[k][i] for i in idx} for k in ('P_out', 'fold', 'jaw', 'pin', 'S_in', 'held_fixed') if ck.plan.get(k) is not None})


def restore(ck, snap):
    for i, o in snap['obs'].items(): ck.obs[i] = o
    for j, r in snap['rel'].items(): ck.rel[j] = r
    for k, d in snap['plan'].items():
        for i, v in d.items(): ck.plan[k][i] = v


def score(V, idx):
    """gate score of frames idx: severity + 1 per FAIL (+ 0.25 per SOFT)."""
    s = 0.0
    for j in idx:
        f = V['frames'][j]
        s += f['sev'] + (1.0 if f['verdict'] == 'FAIL' else 0.25 if f['verdict'] == 'SOFT' else 0.0)
    return s


def hard_fail(f):
    return any(c in HARD_CODES for c, _ in f['causes'])


# ----------------------------------------------------------------------------------------------- candidates --
def held_fix(clip, ck):
    """the input track repaired where it is held / wrong: frames whose input landmarks disagree with the independent
    re-detection (lm_err >= floor, or a held run) take the re-detection (canonical shape, re-projected by the frame's
    pose); k5-smoothed like the pipeline's own track. -> (S_new n x 478 x 3, bad mask)"""
    n = clip.n
    M = getattr(ck, '_M', None)
    err = M['lm_err'] if M is not None and 'lm_err' in M else ck._input_err()
    held = getattr(ck, 'held', np.zeros(n, bool))
    bad = (np.nan_to_num(err) >= ck.th['lm_err']['floor']) | held
    R = np.asarray(ck.take_lm, float)
    raw = clip.S3raw.copy()
    for t in np.nonzero(bad)[0]:
        if np.isfinite(R[t]).all(): raw[t] = R[t]
    from . import mouth_morph as mm
    pl = clip.pl
    plf = mm.RigidPlacement(pl.src3, raw, mm.MESH_ANCH, smooth=clip.a.warp_pose_smooth, lam=0.0)
    return mp_.smooth_track(raw, clip.g('track_smooth', 'k5')), bad, plf


def residual_for(clip, i, Si, ov=None):
    """the residual that puts frame i's target features where they were, from landmarks Si (structure kept)."""
    with clip.patched(i, {k: v for k, v in (ov or {}).items() if k == 'pose'}):
        C = clip.pl.to_lib(i, Si)
        Fn = mw.features(C[None])[0]
    return np.asarray(clip.Fe[i], float) - Fn


def fold_guard(clip, i, dFe_i, Si=None, ell=None, lam=None, thr=None, pose=None):
    thr = clip.g('warp_fold_guard', 0.15) if thr is None else thr
    with clip.patched(i, {} if pose is None else {'pose': pose}):
        for k in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):
            if clip.fold_det_of(i, k * dFe_i, Si, ell, lam) >= thr: return k * dFe_i
    return 0.25 * dFe_i


def cand_lm_hold(clip, ck, t, ctx, base):
    n = clip.n
    if 'S' in base: return None                                         # (the frame's landmarks were already repaired)
    if 'held' not in ctx: ctx['held'] = held_fix(clip, ck)
    S_fix, bad, plf = ctx['held']
    if bad[max(0, t - 2):t + 3].any():
        Si = S_fix[t]
        pose = (plf.s[t], plf.R[t], plf.t[t])
        ov = dict(base, S=Si, pose=pose)
        ov['dFe'] = fold_guard(clip, t, residual_for(clip, t, Si, ov), Si, pose=pose)
        return ov, 'redetect', dict(S_in=Si, held_fixed=True)
    else:
        # a one-frame jump: the input frame with the largest jerk near t is treated as the outlier, predicted from its
        # neighbours in the skull frame; the k5 track re-made
        jk = ck._input_jerk()
        w = [j for j in range(max(1, t - 2), min(n - 1, t + 3))]
        if not w: return None
        s = max(w, key=lambda j: np.nan_to_num(jk[j]))
        raw = clip.S3raw.copy()
        pl = clip.pl
        raw[s] = pl.rigid3(s, 0.5 * (pl.to_lib(s - 1, raw[s - 1]) + pl.to_lib(s + 1, raw[s + 1])))
        Si = mp_.smooth_track(raw, clip.g('track_smooth', 'k5'))[t]
        how = f'pred@{s}'
    dFe = fold_guard(clip, t, residual_for(clip, t, Si), Si)
    return dict(base, S=Si, dFe=dFe), how, dict(S_in=Si)


def cand_warp_soft(clip, ck, t, ctx, base):
    if base.get('soft'): return None          # already softened (an unresolved frame, a later round): never compounded
    ell = clip.ell
    if ell is not None:
        e2 = object.__new__(mw.HeadEllipse)
        e2.c, e2.ax, e2.ay, e2.r_in, e2.r_out = ell.c, ell.ax, ell.ay, min(0.7, ell.r_in), ell.r_out
    else: e2 = None
    lam = 3 * clip.a.warp_lam
    dFe = fold_guard(clip, t, 0.85 * np.asarray(base.get('dFe', clip.dFe[t]), float), base.get('S'), e2, lam, thr=0.3, pose=base.get('pose'))
    return dict(base, dFe=dFe, ell=e2, lam=lam, soft=True), 'soft', {}


def cand_donor_hold(clip, ck, t, ctx, base):
    j = t - 1 if t > 0 else t + 1
    if not clip.rigid_teeth:
        return dict(base, donors=clip.mdon[j]), f'donors@{j}', {}
    ov = {}
    if clip.teeth is not None and clip.teeth.jaw_plan is not None: ov['jaw'] = clip.teeth.jaw_plan[j]
    c = clip.comp
    if c is not None:
        ov['dth'] = c.dth[j]
        if c.open_s is not None: ov['open_s'] = c.open_s[j]
        if c.tg is not None: ov['tg'] = c.tg[j]
    return (dict(base, **ov), f'interior@{j}', {}) if ov else None


def cand_teeth_plan(clip, ck, t, ctx, base):
    c = clip.comp
    if c is None or c.pin_mode != 'arc': return None
    return dict(base, dth=c.dth_for(t, gain=0.0, P14=clip.P_out[t][14])), 'gain0', {}


def cand_layers_hold(clip, ck, t, ctx, base):
    ov = dict(base, cover_grow=0.32)
    if clip.lock is not None and clip.lock.st is not None and t > 0: ov['lock_st'] = clip.lock.st[t - 1]
    return ov, 'cover0.32+lock@t-1', {}


def cand_layers_off(clip, ck, t, ctx, base):
    """v11 (h): the frame re-made as the plain own-pixel warp, without the layer stack (the take's own mouth moved to the
    target: its own teeth / interior, no cover, cavity, arches or lip layer); the erase layer and the lip lock as usual.
    v12 (--warp-gate-h-guard): only where it cannot switch the interior's look for a frame (h_allowed)."""
    if not clip.rigid_teeth or base.get('layers_off'): return None
    if clip.g('warp_gate_h_guard', False) and not h_allowed(clip, t, ctx): return None
    return dict(base, layers_off=True), 'plain warp', {}


# ---- v12 (opt-in, --warp-gate-h-guard): candidate h without one-frame switches of the interior's look -----------------
STYLE_PEN = 1.0           # the ranking penalty (a FAIL's worth) per frame whose interior style differs from both neighbours


def h_closed_thr(clip):
    """the planned opening (units of the take's range above its shut level) below which a frame is NEAR-CLOSED: the take's own
    p5 opening (its closure depth) + the pipeline's closure band (--warp-erase-thr, 0.12: the erase layer treats targets
    under it as closures). There the interior is a slit at most, so the layer stack and the take's own interior look alike."""
    r = clip.model.rng
    gs = max(1e-6, r['g_hi'] - r['g_lo'])
    return float(np.percentile((np.asarray(clip.F, float)[:, 0] - r['g_lo']) / gs, 5)) + float(clip.g('warp_erase_thr', 0.12))


def near_closed(clip, t):
    r = clip.model.rng
    gs = max(1e-6, r['g_hi'] - r['g_lo'])
    return (float(clip.Fe[t][0]) - r['g_lo']) / gs <= h_closed_thr(clip)


def own_style(clip, j):
    """frame j (as committed) shows the take's own interior: re-made without the layer stack, or shown as the take."""
    return bool(clip.log[j].get('layers_off')) or is_take(clip, j)


def h_allowed(clip, t, ctx):
    """h may re-make frame t when t is near-closed, or when both its neighbours show the take's own interior too (committed,
    or members of the run being re-made that qualify themselves: a fixed point from the near-closed ones); a clip edge
    counts as a matching neighbour."""
    run = list(ctx.get('run') or [])
    n = clip.n
    if t not in run: run = [t]                  # (a single-frame call, e.g. the slosh fallback: no run context)
    q = {j for j in run if near_closed(clip, j)}
    def ok_nb(j): return j < 0 or j >= n or (j in q if j in run else own_style(clip, j))
    while True:
        add = {j for j in run if j not in q and ok_nb(j - 1) and ok_nb(j + 1)}
        if not add: break
        q |= add
    return t in q


def style_switches(clip, ovs):
    """frames of a candidate run whose interior style (the take's own / the layer stack) differs from BOTH neighbours (the
    committed frames, or the candidate's own run-mates)."""
    n = clip.n
    def own(j): return bool(ovs[j][0].get('layers_off')) if j in ovs else own_style(clip, j)
    k = 0
    for t in ovs:
        nb = [own(j) for j in (t - 1, t + 1) if 0 <= j < n]
        if nb and all(x != own(t) for x in nb): k += 1
    return k


def interp_params(clip, frames, lo, hi, tracks, deg=1):
    """per frame of `frames`: each parameter track fitted through the anchors lo / hi (linear) or a quadratic through
    them least-squares on the frames between (deg 2). tracks: name -> (n x ...) array."""
    out = {f: {} for f in frames}
    for name, A in tracks.items():
        A = np.asarray(A, float)
        a0, a1 = A[lo], A[hi]
        for f in frames:
            u = (f - lo) / max(1, hi - lo)
            out[f][name] = (1 - u) * a0 + u * a1
        if deg == 2 and hi - lo >= 3:
            ts = np.arange(lo + 1, hi)
            U = (ts - lo) / (hi - lo)
            base = np.array([(1 - u) * a0 + u * a1 for u in U])
            res = A[lo + 1:hi] - base
            q = U * (1 - U)                          # the quadratic that keeps both anchors
            c = np.tensordot(q, res, axes=(0, 0)) / max(1e-9, float(q @ q))
            for f in frames:
                u = (f - lo) / (hi - lo)
                out[f][name] = out[f][name] + u * (1 - u) * c
    return out


def layer_tracks(clip):
    tr = {}
    t_, c = clip.teeth, clip.comp
    if t_ is not None:
        if t_.jaw_plan is not None: tr['jaw'] = t_.jaw_plan
        tr['pin'] = t_.pin
        tr['light'] = t_.light
    if c is not None:
        tr['dth'] = c.dth
        if c.open_s is not None: tr['open_s'] = c.open_s
        if c.tg is not None: tr['tg'] = c.tg
    if clip.lock is not None and clip.lock.st is not None: tr['lock_st'] = clip.lock.st
    if clip.er_al is not None: tr['erase_al'] = clip.er_al
    return tr


def cand_param_interp(clip, ck, t, ctx, V, run=None, base=None):
    run = run or [t]
    lo = next((j for j in range(run[0] - 1, -1, -1) if V['frames'][j]['verdict'] == 'PASS'), None)
    hi = next((j for j in range(run[-1] + 1, clip.n) if V['frames'][j]['verdict'] == 'PASS'), None)
    if lo is None and hi is None: return None
    one_sided = lo is None or hi is None                     # (a run at the clip's edge: its own audio target is kept)
    if lo is None: lo = hi
    if hi is None: hi = lo
    tracks = layer_tracks(clip)
    tracks['Fe'] = clip.Fe
    p = interp_params(clip, [t], lo, hi, tracks)[t] if lo != hi else {k: np.asarray(v, float)[lo] for k, v in tracks.items()}
    ov = {k: v for k, v in p.items() if k != 'Fe'}
    if 'held' not in ctx: ctx['held'] = held_fix(clip, ck)
    S_fix, bad, plf = ctx['held']
    extra = {}
    base = base or {}
    Si = base.get('S', S_fix[t] if bad[t] else clip.S[t])
    if 'pose' in base: ov['pose'] = base['pose']
    elif bad[t]: ov['pose'] = (plf.s[t], plf.R[t], plf.t[t])
    if bad[t] or 'S' in base: extra = dict(S_in=Si, held_fixed=True)
    Fe_t = np.asarray(clip.Fe[t], float) if one_sided else (p['Fe'] if lo != hi else np.asarray(clip.Fe[lo], float))
    with clip.patched(t, {k: v for k, v in ov.items() if k == 'pose'}):
        Fn = mw.features(clip.pl.to_lib(t, Si)[None])[0]
    ov['dFe'] = fold_guard(clip, t, Fe_t - Fn, Si, pose=ov.get('pose'))
    ov['S'] = Si
    return ov, f'interp {lo}..{hi}', extra


CANDS = dict(a=('lm_hold', cand_lm_hold), b=('warp_soft', cand_warp_soft), c=('donor_hold', cand_donor_hold),
             d=('teeth_plan', cand_teeth_plan), e=('layers_hold', cand_layers_hold), h=('layers_off', cand_layers_off))
ORDER = 'abcdeh'          # the order candidates are tried in (h after the others), then f (param_interp) as the last resort


# --------------------------------------------------------------------------------------------- confirmation --
def acceptable(V0, V1, run, n):
    """a run of re-made frames is kept when its FAIL count drops (to none, or at least by one), no frame of it fails a
    hard gate, and neither outer neighbour got worse."""
    a, b = run[0], run[-1]
    f0 = sum(V0['frames'][t]['verdict'] == 'FAIL' for t in run)
    f1 = sum(V1['frames'][t]['verdict'] == 'FAIL' for t in run)
    if f1 >= f0 and f0 > 0: return False
    if any(hard_fail(V1['frames'][t]) for t in run): return False
    for j in (a - 1, b + 1):
        if 0 <= j < n and RANK[V1['frames'][j]['verdict']] > RANK[V0['frames'][j]['verdict']]: return False
    return True


def plan_entry(clip, t, ov, ent, P, fd, extra=None):
    """the checker's plan for a re-made frame t: where its lips were put, its fold, planned jaw, teeth pin (+ extra:
    the input landmarks it used, S_in / held_fixed)."""
    jaw = ent.get('jaw_deg', np.nan) + ent.get('jaw_dth_deg', 0.0) if 'jaw_deg' in ent else np.nan
    pin = np.asarray(ov['pin'], float) if 'pin' in ov else (np.asarray(clip.teeth.pin[t], float) if clip.teeth is not None else np.zeros(2))
    plan_i = dict(P_out=np.asarray(P, float), fold=fd, jaw=jaw, pin=pin)
    plan_i.update(extra or {})
    return plan_i


def try_run(clip, ck, run, ovs, ref):
    """render the run's frames with their overrides (ovs: frame -> (ov, extra)) and check them against the current
    neighbours; the checker is left UPDATED (restore 'snap' to undo)."""
    snap = snapshot(ck, run)
    res = {}
    for t in run:
        ov, extra = ovs[t]
        out, mask, ent, P = clip.render_frame(t, ov, fold=True)
        fd = ent.pop('fold_det', 1.0)
        plan_i = plan_entry(clip, t, ov, ent, P, fd, extra)
        ck.update(t, out, mask, plan_i=plan_i)
        res[t] = dict(out=out, mask=mask, ent=ent, P=P, plan_i=plan_i)
    M = ck.metrics()
    V = ck.verdict(M, ref)
    return dict(res=res, V=V, M=M, snap=snap)


def commit_run(clip, ck, r, how, rnd):
    for t, (ov, extra) in r.get('ovs', {}).items(): clip.gate_ov[t] = ov
    for t, x in r['res'].items():
        ck.update(t, x['out'], x['mask'], plan_i=x['plan_i'])
        clip.outs[t], clip.masks[t] = x['out'], x['mask']
        clip.P_out[t] = np.asarray(x['P'], float)
        ent = dict(x['ent'])
        ent.update(gate_cand=how, gate_round=rnd)
        clip.log[t] = ent


def runs_of(V, want):
    fr = [f['i'] for f in V['frames'] if f['verdict'] in want]
    runs = []
    for t in fr:
        if runs and t == runs[-1][-1] + 1: runs[-1].append(t)
        else: runs.append([t])
    return sorted(runs, key=lambda r: -max(V['frames'][t]['sev'] for t in r))


def held_span(ck, run):
    """the whole HELD input run a failing run touches (frames whose input landmarks repeat the previous frame's: no
    detection, the first detection held): lm_hold re-makes it as a whole. Re-made frame by frame, the input would jump
    between the held and the re-detected landmarks inside the run -- a slosh the gate itself would create."""
    held = getattr(ck, 'held', None)
    held = np.zeros(ck.n, bool) if held is None else np.asarray(held, bool)
    a, b = run[0], run[-1]
    if not held[a:b + 1].any(): return list(run)
    while a > 0 and held[a - 1]: a -= 1
    while b + 1 < ck.n and held[b + 1]: b += 1
    return list(range(a, b + 1))


def repair_run(clip, ck, run, cands, ref, V, rnd, ctx, log):
    """re-make a run of consecutive flagged frames (often 1) from each candidate, confirm, keep the best. lm_hold
    re-makes the whole held input run around it (held_span); the confirmation then covers every frame it re-made."""
    n = clip.n
    hs = held_span(ck, run) if 'a' in cands else list(run)
    span = [j for j in range(min(run[0], hs[0]) - 1, max(run[-1], hs[-1]) + 2) if 0 <= j < n]
    s0 = score(V, span)
    tried = []
    best = None
    keys = sorted((k for k in cands if k in CANDS), key=ORDER.index) + (['f'] if 'f' in cands else [])
    guard = clip.g('warp_gate_h_guard', False)
    if guard: ctx['run'] = list(hs if 'a' in cands else run)        # v12: h_allowed sees the run being re-made
    for key in keys:
        if key == 'f' and best is not None and best[0]: break                 # (f) only as the last resort
        ovs = {}
        for t in (hs if key == 'a' else run):
            base = clip.gate_ov.get(t, {})
            got = cand_param_interp(clip, ck, t, ctx, V, run, base) if key == 'f' else CANDS[key][1](clip, ck, t, ctx, base)
            if got is None: continue
            ov, how, extra = got
            ovs[t] = (ov, extra)
        if not ovs: continue
        name = 'param_interp' if key == 'f' else CANDS[key][0]
        r = try_run(clip, ck, sorted(ovs), ovs, ref)
        sc = score(r['V'], span)
        ok = acceptable(V, r['V'], sorted(ovs), n)
        tried.append(dict(cand=f'{key}:{name}', frames=sorted(ovs), score=round(sc, 3), verdicts=[r['V']['frames'][t]['verdict'] for t in run], ok=ok))
        r['ovs'] = ovs
        if guard:       # v12: the style-continuity term: a one-frame switch of the interior's look ranks a FAIL's worth lower
            sw = style_switches(clip, ovs)
            tried[-1]['style_switch'] = sw
            sc = sc + STYLE_PEN * sw
        if best is None or (ok, -sc) > (best[0], -best[1]): best = (ok, sc, f'{key}:{name}', r)
        restore(ck, r['snap'])
    rec = dict(frames=[int(t) for t in run], round=rnd, before=[V['frames'][t]['verdict'] for t in run],
               causes=sorted(set(c for t in run for c in V['frames'][t]['attr'])), score_before=round(s0, 3), tried=tried)
    keep_best = getattr(clip.a, 'warp_gate_unresolved', 'best') == 'best'    # none: an unfixable frame stays round 0
    if best is not None and (best[0] or (keep_best and best[1] < s0 - 1e-6)):
        ok, sc, name, r = best
        commit_run(clip, ck, r, name, rnd)
        M = ck.metrics()
        V1 = ck.verdict(M, ref)
        rec.update(chosen=name, remade=[int(t) for t in sorted(r['ovs'])], after=[V1['frames'][t]['verdict'] for t in run], score_after=round(sc, 3),
                   resolved=bool(ok) and all(V1['frames'][t]['verdict'] != 'FAIL' for t in run))
        log.append(rec)
        return V1, True
    rec.update(chosen=None, after=rec['before'], resolved=False)
    log.append(rec)
    return V, False


# ------------------------------------------------------------------------------------------------ slosh fit --
def window_tracks(clip):
    """the RESIDUAL tracks the slosh fit may replace (the take's landmark shape beyond its 2 target features, in the
    skull frame; the chin pin's extra jaw angle; the upper-teeth pin; the lip lock state), from each frame's current
    (committed) parameters."""
    n = clip.n
    C = []
    for i in range(n):
        b = clip.gate_ov.get(i, {})
        with clip.patched(i, {k: v for k, v in b.items() if k == 'pose'}):
            C.append(clip.pl.to_lib(i, b.get('S', clip.S[i])))
    C = np.array(C)
    # the take's mouth SHAPE split by the mouth model: the part its own gap / width explain (the real opening /
    # spreading the warp must follow: kept per frame) and the residual beyond it (landmark noise, slides: fitted)
    tr = dict(C=C, Cres=C[:, mw.CTRL] - model_shape(clip, C))
    if clip.comp is not None: tr['dth'] = clip.comp.dth
    if clip.teeth is not None: tr['pin'] = clip.teeth.pin
    if clip.lock is not None and clip.lock.st is not None: tr['lock_st'] = clip.lock.st
    return tr


def model_shape(clip, C):
    """the 2-feature mouth model's control-point shape for canonical landmarks C (n x 478 x 3): coef0 + (gap, width) coef."""
    F = mw.features(C)
    K = len(mw.CTRL)
    co = clip.model.coef
    return (co[0][None] + F @ co[1:]).reshape(len(C), K, 3)


def slosh_energy(V, M, a0, a1):
    sw = np.asarray(M['slosh_win'], float)[a0:a1 + 1]
    return float(np.nansum(np.clip(sw, 0, None)))


def structure_ok(clip, ck, P_new, P_old, frames, tol=0.05):
    """the outline's gap / width (the skull frame) within tol x the take's range of before, on every frame (a range
    under 10 canonical px counts as 10: a take whose width never changes must not make any sub-pixel change 'structure')."""
    r = clip.model.rng
    gs, ws = max(10.0, r['g_hi'] - r['g_lo']), max(10.0, r['w_hi'] - r['w_lo'])
    for f in frames:
        qa, qb = ck.stab.to_crop(f, P_old[f]) / ck.stab.k, ck.stab.to_crop(f, P_new[f]) / ck.stab.k
        ga, gb = np.linalg.norm(qa[13] - qa[14]), np.linalg.norm(qb[13] - qb[14])
        wa, wb = np.linalg.norm(qa[61] - qa[291]), np.linalg.norm(qb[61] - qb[291])
        if abs(ga - gb) > tol * gs or abs(wa - wb) > tol * ws: return False
    return True


def closures(clip):
    """the frames whose audio target is a lip closure (viseme PP: M / B / P): mouth STRUCTURE the slosh repair never
    re-makes (the lips meet exactly as round 0 drew them)."""
    fr = getattr(clip, 'fr_t', None) or []
    return {i for i in range(clip.n) if i < len(fr) and fr[i].get('vis') == 'PP'}


def window_fallback(clip, ck, win, ref, V, M, rnd, ctx, cands):
    """(g) rejected: the single-frame candidates on the window's worst frame (its highest slosh centre, closures
    excluded); kept only when the window's slosh energy drops, no frame of the window fails and neither neighbour of
    the frame gets worse."""
    n = clip.n
    a0, a1 = win['frames']
    sw = np.nan_to_num(np.asarray(M['slosh_win'], float))
    pp = closures(clip)
    free = [f for f in range(a0, a1 + 1) if f not in pp]
    e0 = slosh_energy(V, M, a0, a1)
    best = None
    tried = []
    if not free: return None, e0, None, tried
    c = int(max(free, key=lambda f: sw[f]))
    for key in [k for k in cands if k in CANDS]:
        got = CANDS[key][1](clip, ck, c, ctx, clip.gate_ov.get(c, {}))
        if got is None: continue
        ov, how, extra = got
        r = try_run(clip, ck, [c], {c: (ov, extra)}, ref)
        r['ovs'] = {c: (ov, extra)}
        e1 = slosh_energy(r['V'], r['M'], a0, a1)
        ok = e1 < e0 - 1e-6 and all(r['V']['frames'][f]['verdict'] != 'FAIL' for f in range(a0, a1 + 1)) and \
            all(RANK[r['V']['frames'][j]['verdict']] <= RANK[V['frames'][j]['verdict']] for j in (c - 1, c, c + 1) if 0 <= j < n)
        tried.append(dict(cand=f'{key}:{CANDS[key][0]}', frame=c, energy=round(e1, 4), ok=ok))
        if ok and (best is None or e1 < best[0]): best = (e1, f'{key}:{CANDS[key][0]}', r)
        restore(ck, r['snap'])
    return c, e0, best, tried


def window_fallback_commit(clip, ck, win, ref, V, M, rnd, log, rec, ctx, cands):
    """(g) not possible / not kept: the single-frame fallback; committed when it helps."""
    a0, a1 = win['frames']
    c, e0, fb, tried = window_fallback(clip, ck, win, ref, V, M, rnd, ctx if ctx is not None else {}, cands)
    rec.update(chosen=None, fallback=dict(frame=c, tried=tried))
    if fb is not None:
        e1, name, r = fb
        commit_run(clip, ck, r, name + ' (slosh fallback)', rnd)
        M2 = ck.metrics()
        V2 = ck.verdict(M2, ref)
        rec['fallback'].update(chosen=name, energy_after=round(slosh_energy(V2, M2, a0, a1), 4))
        log.append(rec)
        return V2, M2, True
    log.append(rec)
    return V, M, False


def repair_window(clip, ck, win, ref, V, M, rnd, log, ctx=None, cands=()):
    n = clip.n
    a0, a1 = win['frames']
    lo = next((j for j in range(a0, -1, -1) if V['frames'][j]['verdict'] == 'PASS'), None)
    hi = next((j for j in range(a1, n) if V['frames'][j]['verdict'] == 'PASS'), None)
    rec = dict(window=[int(a0), int(a1)], round=rnd, energy_before=round(slosh_energy(V, M, a0, a1), 4))
    if lo is None and hi is not None and a0 == 0: lo = 0                   # the clip's first / last frame anchors a window
    if hi is None and lo is not None and a1 == n - 1: hi = n - 1            # that touches it
    if lo is None or hi is None or hi - lo < 2:
        rec.update(chosen=None, why='no PASS anchors')
        log.append(rec)
        return V, M, False
    # a closure inside the window is structure: it keeps its frame as it is and anchors the fit (piecewise between the
    # PASS anchors and the closures)
    pp = closures(clip)
    anc = [lo] + [f for f in range(lo + 1, hi) if f in pp] + [hi]
    segs = [(x, y) for x, y in zip(anc[:-1], anc[1:]) if y - x >= 2]
    frames = [f for x, y in segs for f in range(x + 1, y)]
    if len(anc) > 2: rec['closures'] = [int(f) for f in anc[1:-1]]
    if not frames:
        rec['why'] = 'closures only'
        return window_fallback_commit(clip, ck, win, ref, V, M, rnd, log, rec, ctx, cands)
    tracks = window_tracks(clip)
    vis = M.get('viseme')
    vis0 = {f: np.nan_to_num(np.asarray(vis, float)[f]) for f in frames} if vis is not None else {}
    best = None
    P_old = {f: np.asarray(clip.P_out[f], float) for f in frames}
    for deg in (1, 2):
        fit = {}
        for x, y in segs: fit.update(interp_params(clip, list(range(x + 1, y)), x, y, tracks, deg))
        snaps = snapshot(ck, frames)
        res = {}
        for f in frames:
            b = clip.gate_ov.get(f, {})
            ov = dict(b)
            ov.update({k: v for k, v in fit[f].items() if k not in ('C', 'Cres')})
            Cf = tracks['C'][f].copy()
            Cf[mw.CTRL] = model_shape(clip, Cf[None])[0] + fit[f]['Cres']   # structure kept, residual fitted
            with clip.patched(f, {k: v for k, v in b.items() if k == 'pose'}):
                Si = clip.pl.rigid3(f, Cf)
            ov['S'] = Si
            ov['S_take'] = b.get('S_take', b.get('S', clip.S[f]))
            ov['dFe'] = fold_guard(clip, f, residual_for(clip, f, Si, ov), Si, pose=ov.get('pose'))
            res_ov = ov
            out, mask, ent, P = clip.render_frame(f, ov, fold=True)
            fd = ent.pop('fold_det', 1.0)
            plan_i = plan_entry(clip, f, ov, ent, P, fd, dict(S_in=Si))      # (the fitted input landmarks it now uses)
            ck.update(f, out, mask, plan_i=plan_i)
            res[f] = (out, mask, ent, P, plan_i, res_ov)
        M1 = ck.metrics()
        V1 = ck.verdict(M1, ref)
        e1 = slosh_energy(V1, M1, a0, a1)
        okS = structure_ok(clip, ck, {f: res[f][3] for f in frames}, P_old, frames)
        # the measured fit to the audio (MediaPipe on the re-made frames) may not get worse beyond its own measurement
        # noise: the shot's robust scale of the viseme ruler (the outline's own gap / width is checked exactly above)
        tolV = max(0.05, float(ref.get('viseme', (0.0, 0.05))[1]))
        okV = all(np.nan_to_num(np.asarray(M1['viseme'], float)[f]) <= vis0.get(f, 0) + tolV for f in frames) if vis is not None else True
        okF = all(V1['frames'][f]['verdict'] != 'FAIL' for f in range(max(0, lo), min(n, hi + 1)))
        cand = dict(deg=deg, energy=round(e1, 4), structure=okS, viseme=okV, no_fail=okF)
        rec.setdefault('tried', []).append(cand)
        if okS and okV and okF and e1 < rec['energy_before'] - 1e-6 and (best is None or e1 < best[0]):
            best = (e1, deg, res, V1, M1)
        restore(ck, snaps)
    if best is None: return window_fallback_commit(clip, ck, win, ref, V, M, rnd, log, rec, ctx, cands)
    e1, deg, res, V1, M1 = best
    for f in frames:
        out, mask, ent, P, plan_i, ov = res[f]
        ck.update(f, out, mask, plan_i=plan_i)
        clip.gate_ov[f] = ov
        ent = dict(ent)
        ent.update(gate_cand=f'g:slosh_fit(deg {deg})', gate_round=rnd)
        clip.outs[f], clip.masks[f], clip.log[f] = out, mask, ent
        clip.P_out[f] = np.asarray(P, float)
    M2 = ck.metrics()
    V2 = ck.verdict(M2, ref)
    rec.update(chosen='g:slosh_fit', deg=deg, anchors=[int(x) for x in anc], frames=frames, energy_after=round(slosh_energy(V2, M2, a0, a1), 4))
    log.append(rec)
    return V2, M2, True


# ------------------------------------------------------------------------------------------------------ gate --
def summary(V):
    c = dict(PASS=0, SOFT=0, FAIL=0)
    for f in V['frames']: c[f['verdict']] += 1
    cz = {}
    for f in V['frames']:
        if f['verdict'] == 'PASS': continue
        for x in f['attr']: cz[x] = cz.get(x, 0) + 1
    return dict(counts=c, causes=cz, slosh_windows=[w['frames'] for w in V['windows']])


def gate(clip, a, lmk=None):
    """--warp-gate report | repair on a finished round 0 (clip.outs / masks / log, modified in place for repair).
    Returns the warp.json block {'gate': ...}; --warp-gate-report also writes gate.json + gate.csv. lmk: the landmark
    detector (default: IMAGE-mode MediaPipe on a face-box crop)."""
    t0 = time.time()
    mode = a.warp_gate
    clip.gate_ov = {}
    ck = make_checker(clip, a, lmk)
    M = ck.metrics()
    ref = ck.stats(M)
    V = ck.verdict(M, ref)
    ck._M = M
    V0, _M0 = V, M
    log = []
    wlog = []
    print(f'gate round 0: {summary(V)["counts"]} slosh windows {summary(V)["slosh_windows"]} ({time.time() - t0:.0f} s)')
    if mode == 'repair':
        cands = [c.strip() for c in str(getattr(a, 'warp_gate_cands', 'a,b,c,d,e,f,g')).split(',') if c.strip()]
        ctx = {}
        for rnd in range(1, max(1, getattr(a, 'warp_gate_rounds', 3)) + 1):
            want = ('FAIL', 'SOFT') if getattr(a, 'warp_gate_soft', False) else ('FAIL',)
            done_any = False
            for run in runs_of(V, want):
                run = [t for t in run if V['frames'][t]['verdict'] in want and not is_take(clip, t)]
                if not run: continue
                V, ok = repair_run(clip, ck, run, cands, ref, V, rnd, ctx, log)
                done_any |= ok
                M = ck.metrics()
                ck._M = M
            if 'g' in cands:
                for w in list(V['windows']):
                    if all(is_take(clip, t) for t in range(w['frames'][0], w['frames'][1] + 1)): continue
                    V, M, ok = repair_window(clip, ck, w, ref, V, M, rnd, wlog, ctx, cands)
                    done_any |= ok
            M = ck.metrics()
            V = ck.verdict(M, ref)
            ck._M = M
            print(f'gate round {rnd}: {summary(V)["counts"]} slosh windows {summary(V)["slosh_windows"]} ({time.time() - t0:.0f} s)')
            if not done_any: break
    # per-frame log: verdict before / after, causes, chosen candidate
    for f0, f1 in zip(V0['frames'], V['frames']):
        e = clip.log[f0['i']]
        e['gate_before'] = f0['verdict']
        e['gate_after'] = f1['verdict']
        e['gate_causes'] = f1['attr'] if f1['verdict'] != 'PASS' else f0['attr']
    chosen = {}
    for r in log + wlog:
        if r.get('chosen'): chosen[r['chosen']] = chosen.get(r['chosen'], 0) + len(r.get('remade', r.get('frames', [1])))
        elif (r.get('fallback') or {}).get('chosen'):
            k_ = r['fallback']['chosen'] + ' (slosh fallback)'
            chosen[k_] = chosen.get(k_, 0) + 1
    unresolved = [f['i'] for f in V['frames'] if f['verdict'] == 'FAIL']
    reg = None
    if getattr(a, 'warp_gate_reg', False):
        # v11: the inter-layer registration ruler on the frames as they leave the gate (reported, not repaired)
        from ..audits import layer_reg
        reg = layer_reg.gate_report(clip.log, ck.W, getattr(a, 'warp_gate_reg_floor', None))
        print(f"gate reg (reported, not repaired): slosh sum {reg['summary']['slosh_sum']:.4f} IOD/frame; windows >= {reg['floor']}: "
              f"{[(w['frames'], w['top']) for w in reg['windows']]}")
    out = dict(mode=mode, z=ck.Z, slosh_w=ck.W, before=summary(V0), after=summary(V), repairs=log, windows=wlog, chosen=chosen,
               unresolved=unresolved, seconds=round(time.time() - t0, 1), thresholds=dict(TH=ck.th, HARD=chk.HARD),
               redet_rejected=getattr(ck, 'redet_rejected', []))       # v11: the take's re-detections not used (false face fits)
    if reg is not None: out['reg'] = reg
    rp = getattr(a, 'warp_gate_report', None)
    if rp:
        base = rp[:-5] if rp.endswith('.json') else rp
        chk.report(V, M, base + '.json', base + '.csv', dict(gate=out, before=[dict(i=f['i'], verdict=f['verdict'], attr=f['attr'], sev=f['sev']) for f in V0['frames']]))
        # the final plan (where every frame's lips were put, from which input landmarks): the offline checker's
        # --plan, since a re-made frame's overrides are not in warp.json (its replay rebuilds round 0's plan)
        pz = ck.plan
        np.savez(base + '_plan.npz', P_out=np.array([np.asarray(p, float)[:, :2] for p in pz['P_out']]),
                 S_in=np.array([np.asarray(s, float) for s in pz['S_in']]), held_fixed=np.asarray(pz['held_fixed'], bool),
                 fold=np.asarray(pz['fold'], float), jaw=np.asarray(pz['jaw'], float), pin=np.array([np.asarray(p, float) for p in pz['pin']]))
    print(f'gate {mode}: before {out["before"]["counts"]} -> after {out["after"]["counts"]}; chosen {chosen}; unresolved {out["unresolved"]}; {out["seconds"]} s')
    return dict(gate=out)
