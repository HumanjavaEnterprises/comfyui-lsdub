# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""rise_loop.py: --warp-rise achieved. A sung syllable's onset rise, read on the rendered mouth and lifted only where it fell short.

  plan -> render -> [measure the rise across each onset on the pixels] -> lift the short ones with an attack shape -> render those frames

`floor` / `lift` (mouth_warp.rise_floor) raise the PLANNED rise of every onset under 0.30 of the take's range before anything is
rendered, on two frames, whether or not the picture would have lost it. This stage works on what the render delivered instead:

  1. every frame of the rendered clip is re-detected (inner-lip gap / eye distance, aperture.measure) and the reading's own p5..p95
     spread is its ruler;
  2. per onset the rise across it (the reading SPAN s after a centre minus SPAN s before it) is read at the centre within NEAR s of
     the onset whose rise is the largest peak (no larger rise one output frame either side). An onset whose rise reaches TARGET of
     that spread and is such a peak is left exactly as rendered;
  3. a short onset gets the missing part (never so much that a frame-to-frame step passes the clip's own STEP_Q-th percentile
     step) as an opening added on the gap residual with the shape of an attack: zero SPAN s before the centre, a straight ramp to the full amount SPAN s after it (the least per-frame step that carries that rise), held through the
     syllable's nucleus (at most HOLD s), then eased back over RELEASE s, done before the next onset's ramp, a kept-shut frame or a
     frame handed back to the take. No bump-and-drop on the frames after the onset;
  4. the touched frames are rendered again (never past the warp's fold limit, never past the take's widest) and re-read; an onset still
     short gets the rest once more, scaled by what the first lift delivered;
  5. a lift whose onset still does not rise KEEP x TARGET after that is taken back out: those frames are rendered as they first were.

Inputs: the clip's rendered frames, its residual, the event stage's onsets and nucleus spans, the kept-shut frames. Nothing here reads
the vocal's pitch, the judge, or another stage's internals; the lift is a function of the opening read on the pixels alone."""

import numpy as np

SPAN = 0.04  # s each side of the centre: the rise is read from t - SPAN to t + SPAN (two output frames at 25 fps)
NEAR = 0.06  # s: the centre is the frame within this of the onset whose rise is largest and a peak
TARGET = 0.30  # x the rendered opening's p5..p95 spread: an onset rising at least this much (as a peak) is left as rendered
# (this reading, one inner-lip pair on single images, puts a rise at about 2.4x the share of its own spread that a steadier mouth
# reading gives the same frames: 26 rows read both ways on b2b's paints, B2-report.md. 0.30 here is about 0.12 there.)
CAP = 0.30  # x the take's range (canonical): the most a lift ever adds
HOLD = 0.04  # s: the lifted opening is held through the nucleus for at most this long after the ramp
RELEASE = 0.16  # s: then eased back to the plan over this long (shorter when the next onset or a closure comes first)
ITERS = 2
MIN_GAIN = 0.2  # a lift that delivered under this share of itself is not pushed further
STEP_Q = 98.0  # the lift never makes a frame-to-frame step (one output frame) larger than this percentile of the clip's own steps
KEEP = 0.8  # x TARGET: a lift whose onset still does not rise this much (as a peak) after the last round is taken back out


def frames_of(sec, fps):
    return max(1, int(round(sec * fps)))


def events(fr_t, n, fps):
    """the event stage's nucleus onsets -> [(onset time, last frame index of its nucleus)] on the clip's clock, one per event; frame
    times are frame centres as in mouth_warp.onset_frames."""
    if not fr_t:
        return []
    t0 = float(fr_t[0].get("t", 0.0))
    on, last = {}, {}
    for i in range(min(n, len(fr_t))):
        e = fr_t[i].get("ev") or {}
        if e.get("kind") == "nucleus" and e.get("t_on") is not None:
            k = e.get("k")
            on.setdefault(k, t0 + float(e["t_on"]))
            last[k] = i
    for i, f in enumerate(fr_t[n:]):  # an event tagged only past the clip's frames still has an onset (as onset_frames reads it)
        e = f.get("ev") or {}
        if e.get("kind") == "nucleus" and e.get("t_on") is not None and e.get("k") not in on:
            on[e.get("k")] = t0 + float(e["t_on"])
            last[e.get("k")] = n + i
    return [(on[k], last[k]) for k in on]


def rise_at(m, c, h):
    """(the rise across centre c: m[c + h] - m[c - h], the larger rise across its neighbours one output frame away, c -+ h); NaN unread,
    -inf for a neighbour off the clip."""
    n = len(m)
    r = lambda j: m[j + h] - m[j - h] if h <= j < n - h else np.nan
    nb = [r(c + d) for d in (-h, h)]
    nb = [x for x in nb if np.isfinite(x)]
    return float(r(c)), (max(nb) if nb else -np.inf)


def best_rise(m, t, on, h, near=NEAR):
    """the rise across an onset on the reading m: (centre frame, m[c + h] - m[c - h]) for the centre within `near` of the onset with the
    largest rise (a tie to the frame nearest the onset); None when no centre has both ends read."""
    b = best_peak(m, t, on, h, near, peaks=False)
    return None if b is None else b[:2]


def best_peak(m, t, on, h, near=NEAR, peaks=True):
    """(centre frame, rise, neighbours' rise) for the centre within `near` of the onset whose rise is the largest that is also a peak
    (no larger than the rise one output frame either side); with no peak (or peaks=False), the largest rise. A tie goes to the frame
    nearest the onset. None when no centre has both ends read."""
    m, t = np.asarray(m, float), np.asarray(t, float)
    n = len(m)
    c = np.flatnonzero(np.abs(t - on) <= near + 1e-9)
    c = c[(c >= h) & (c < n - h)]
    best = None
    for j in c[np.argsort(np.abs(t[c] - on), kind="stable")]:
        r, nb = rise_at(m, int(j), h)
        if not np.isfinite(r):
            continue
        key = (peaks and r >= nb, r)
        if best is None or key > best[0]:
            best = (key, (int(j), r, nb))
    return None if best is None else best[1]


def need_of(r, nb, line):
    """what a rise r with neighbours nb lacks to read as an onset rise: up to `line`, and past its neighbours (the ramp adds half its
    amount to the rise after it, so twice the gap). 0 when it already does."""
    need = max(0.0, line - r, 2.0 * (nb - r) + 0.05 * line if nb > r else 0.0)
    return need if need > 1e-6 * max(line, 1e-12) else 0.0  # a rise on the line to rounding is on it


def spread(m):
    """the reading's p5..p95 spread (its unread frames left out); 0 with too few frames read."""
    m = np.asarray(m, float)
    m = m[np.isfinite(m)]
    return float(np.percentile(m, 95) - np.percentile(m, 5)) if len(m) >= 5 else 0.0


def attack(n, c, h, hold_to, stop, release):
    """the lift's shape (0..1 per frame) for a rise centred on frame c: 0 at c - h, a straight ramp to 1 at c + h, 1 until hold_to (the
    nucleus' last frame), then a raised-cosine release over `release` frames; every frame from `stop` on stays 0, and the hold gives
    way to the release when the room after the ramp is short. None when the ramp itself does not fit before `stop`."""
    if c - h < 0 or c + h >= min(stop, n):
        return None
    p = np.zeros(n)
    for i in range(c - h + 1, c + h + 1):
        p[i] = (i - (c - h)) / (2.0 * h)
    room = min(stop, n) - (c + h) - 1  # frames after the ramp's top that may carry lift
    r = min(release, room)
    hold = min(max(0, hold_to - (c + h)), room - r)
    p[c + h + 1 : c + h + 1 + hold] = 1.0
    for k in range(1, r + 1):
        p[c + h + hold + k] = 0.5 * (1.0 + np.cos(np.pi * k / (r + 1)))
    return p


def plan_lifts(m, t, evs, fps, scale, blocked, target=TARGET, near=NEAR, span=SPAN, hold=HOLD, release=RELEASE):
    """the onsets that fell short on the reading m and the shape each one's lift takes.

    m: the rendered opening per frame (NaN unread); t: frame times; evs: events(); scale: the reading's ruler (spread(m));
    blocked: bool per frame, frames a lift must never touch (kept shut, handed back to the take). -> (list of dicts per short onset
    {k, c, rise, short, p}, counts {onsets, adequate, short, unread, no_room})."""
    m = np.asarray(m, float)
    n = len(m)
    h = frames_of(span, fps)
    hold_n, rel_n = frames_of(hold, fps), frames_of(release, fps)
    blocked = np.asarray(blocked, bool)
    cnt = dict(onsets=len(evs), adequate=0, short=0, unread=0, no_room=0)
    read = []
    for k, (on, last) in enumerate(evs):
        b = best_peak(m, t, on, h, near)
        read.append(b)
        if b is None:
            cnt["unread"] += 1
        elif need_of(b[1], b[2], target * scale) <= 0:
            cnt["adequate"] += 1
    starts = sorted(b[0] - h for b in read if b is not None)
    out = []
    for k, ((on, last), b) in enumerate(zip(evs, read)):
        if b is None or need_of(b[1], b[2], target * scale) <= 0:
            continue
        c = b[0]
        cnt["short"] += 1
        stop = min([s for s in starts if s > c] + [n])  # the next onset's ramp start
        nb = np.flatnonzero(blocked[c - h + 1 :]) + c - h + 1
        if len(nb):
            stop = min(stop, int(nb[0]))
        p = attack(n, c, h, min(last, c + h + hold_n), stop, rel_n)
        if p is None:
            cnt["no_room"] += 1
            continue
        out.append(dict(k=k, c=c, rise=b[1], short=need_of(b[1], b[2], target * scale), p=p))
    return out, cnt


def step_room(m, c, h, q=STEP_Q):
    """how much a ramp centred on c may add (reading units, the whole amount: the ramp puts half of it on each output-frame step)
    before either of its two steps passes the q-th percentile of the reading's own output-frame steps; 0 when it already does."""
    m = np.asarray(m, float)
    st = np.abs(m[h:] - m[:-h])
    st = st[np.isfinite(st)]
    if not len(st):
        return 0.0
    line = float(np.percentile(st, q))
    steps = [m[c] - m[c - h], m[c + h] - m[c]]
    steps = [x for x in steps if np.isfinite(x)]
    return max(0.0, 2.0 * (line - max(steps))) if steps else 0.0


def lift_series(base, lifts, amount, ceiling):
    """the gap residual with the lifts on it: per frame base + the largest amount_k * p_k, never past `ceiling` (per frame; a frame
    already past it keeps its own value)."""
    base = np.asarray(base, float)
    add = np.zeros(len(base))
    for L in lifts:
        add = np.maximum(add, amount[L["k"]] * L["p"])
    return np.where(add > 0, np.minimum(base + add, np.maximum(base, ceiling)), base)


def close_rise(clip, log=print):
    """the stage on a rendered WarpClip (after run_batch and the vocalisation pass): read, lift the short onsets, render them again.
    -> a summary dict (also under 'rise' in each touched frame's log entry)."""
    from . import aperture
    from .mouth_warp import onset_frames

    n, F, rng = clip.n, clip.F, clip.model.rng
    gs = max(1e-6, float(rng["g_hi"]) - float(rng["g_lo"]))
    fps = float(getattr(clip, "track_fps", None) or clip.fps)
    t, _ = onset_frames(clip.fr_t, n, fps)
    evs = events(clip.fr_t, n, fps)
    keep = np.ones(n) if getattr(clip, "vocal_s", None) is None else np.asarray(clip.vocal_s[:n], float)
    shut = getattr(clip, "rise_shut", None)
    blocked = (keep < 1.0) | (np.zeros(n, bool) if shut is None else np.asarray(shut[:n], bool))
    summary = dict(mode="achieved", target=TARGET, onsets=len(evs), lifted=0, rerendered=0)
    if not evs:
        return summary
    m = aperture.measure(clip.outs[:n])
    scale = spread(m)
    cal = aperture.calibration(m, clip.Fe[:n, 0])  # the rendered opening per canonical px of the applied target
    summary.update(scale=round(scale, 4), cal=None if cal is None else round(cal, 5))
    if cal is None or scale <= 0:
        log("rise (achieved): the rendered mouth was not read; left as rendered")
        return summary
    lifts, cnt = plan_lifts(m, t, evs, fps, scale, blocked)
    summary.update(cnt)
    if not lifts:
        log(f"rise (achieved): {cnt['adequate']} of {len(evs)} onsets rise as rendered; none lifted")
        return summary
    thr = float(clip.a.warp_fold_guard)
    base = clip.dFe[:n, 0].copy()
    rows0 = clip.dFe[:n].copy()
    ceiling = float(rng["g_hi"]) - F[:n, 0]  # the take's widest, as a residual
    h = frames_of(SPAN, fps)
    room = {L["k"]: min(CAP * gs, step_room(m, L["c"], h) / cal) for L in lifts}  # no step past the clip's own steep ones
    amount = {L["k"]: min(room[L["k"]], L["short"] / cal) for L in lifts}
    done = []
    for it in range(ITERS):
        want = lift_series(base, lifts, amount, ceiling)
        moved = np.flatnonzero(np.abs(want - clip.dFe[:n, 0]) > 1e-6)
        for i in moved:
            cur = clip.dFe[i].copy()
            new = cur.copy()
            new[0] = want[i]
            if thr > 0 and abs(new[0]) > abs(cur[0]):
                s = aperture.fold_limited(lambda v, i=i: clip.fold_det_of(i, v), cur, new, thr)
                new = cur + s * (new - cur)
            if np.abs(new - cur).max() < 1e-6:
                continue
            out, mask, ent, P_out = clip.render_frame(int(i), {"dFe": new})
            ent["rise"] = dict(lift=round(float(new[0] - base[i]), 2))
            clip.outs[i], clip.masks[i], clip.P_out[i] = out, mask, P_out
            clip.log[i].update(ent)
            clip.dFe[i] = new
            clip.Fe[i] = F[i] + new
            summary["rerendered"] += 1
        if not len(moved):
            for L in lifts:  # nothing left to render: what was read last stands
                r, nb = rise_at(m, L["c"], h)
                L["after"] = (float(r), need_of(r, nb, TARGET * scale) <= 0) if np.isfinite(r) else None
            break
        m2 = m.copy()
        m2[moved] = aperture.measure(clip.outs[:n], moved)[moved]
        again = []
        for L in lifts:
            r, nb = rise_at(m2, L["c"], h)
            if it == 0:
                done.append(L)
            short = need_of(r, nb, TARGET * scale) if np.isfinite(r) else 0.0
            if not np.isfinite(r) or short <= 0 or it == ITERS - 1:
                L["after"] = (float(r), short <= 0) if np.isfinite(r) else None
                continue
            got = (r - L["rise"]) / max(1e-9, amount[L["k"]] * cal)  # the share of the asked lift the pixels carried
            if got < MIN_GAIN or amount[L["k"]] >= room[L["k"]] - 1e-9:
                L["after"] = (float(r), False)
                continue
            amount[L["k"]] = min(room[L["k"]], amount[L["k"]] + short / (cal * got))
            again.append(L)
        m = m2
        if not again:
            break
    # a lift that did not deliver is motion and nothing else: those onsets go back to the frames as first rendered
    back = [L for L in done if not np.isfinite(rise_at(m, L["c"], h)[0]) or need_of(*rise_at(m, L["c"], h), KEEP * TARGET * scale) > 0]
    if back:
        for L in back:
            amount[L["k"]] = 0.0
        want = lift_series(base, lifts, amount, ceiling)
        for i in np.flatnonzero(np.abs(want - clip.dFe[:n, 0]) > 1e-6):
            new = rows0[i].copy()
            new[0] = want[i]
            if thr > 0 and abs(new[0]) > abs(rows0[i][0]):
                s = aperture.fold_limited(lambda v, i=i: clip.fold_det_of(i, v), rows0[i], new, thr)
                new = rows0[i] + s * (new - rows0[i])
            out, mask, ent, P_out = clip.render_frame(int(i), {"dFe": new})
            clip.log[i].pop("rise", None)
            if abs(new[0] - base[i]) > 1e-6:
                ent["rise"] = dict(lift=round(float(new[0] - base[i]), 2))
            clip.outs[i], clip.masks[i], clip.P_out[i] = out, mask, P_out
            clip.log[i].update(ent)
            clip.dFe[i] = new
            clip.Fe[i] = F[i] + new
            summary["rerendered"] += 1
    after = [L.get("after") for L in done]
    summary.update(
        taken_back=len(back),
        lifted=len(done),
        reached=int(sum(a is not None and a[1] for a in after)),
        mean_added=round(float(np.mean([amount[L["k"]] for L in done]) / gs), 3),
        lifted_frames=int((np.abs(clip.dFe[:n, 0] - base) > 1e-6).sum()),
    )
    log(
        f"rise (achieved): {cnt['adequate']} of {len(evs)} onsets rise as rendered; {len(done)} short ones lifted "
        f"({summary['reached']} now reach x{TARGET} of the opening's spread), {cnt['no_room']} without room, {cnt['unread']} unread; "
        f"{len(back)} that did not deliver taken back; {summary['rerendered']} renders"
    )
    return summary
