# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""nucleus_hold.py: one held opening target per sung syllable. One stage of the drive chain, behind one paint flag, --warp-hold
(off | nucleus | nucleus-follow). Pure: track frames with the event stage's `ev` in, new frames out.

  nucleus-follow   the same, and inside each hold the opening follows the voice's level against the syllable's own peak, between
                   FOLLOW_FLOOR of the target and the target (a note that decays closes with it, slowly; no onset is made or lost)

  drive (the track's frames with `ev`) -> [hold] -> the closure stage -> the aperture stage -> the warp's targets

Why: the plain path's opening is the vowel class times a 60 ms loudness envelope, eased three times over; a syllable inside a held
phrase barely dips, so 22-36 sung onsets become 14-21 rises. Production rigs set the opening per syllable instead (a viseme class
times a coarse loudness, held) and shape the curve around the onset. The rule here:

  target   per nucleus: V[vowel] x (0.6 + 0.5 x J), V the viseme table below, J the event's loudness bin value (0.15 / 0.45 / 0.8)
  release  a syllable with an articulated onset (the event's `cons`: a consonant between it and the nucleus before, or `src` with the
           spectral flux: the onset was heard) first releases
           over RELEASE_S to DIP under the lower of the two levels (never under REST_OPEN) — the jaw's closing for the consonant,
           what a singer's own mouth shows before most syllables and what makes the onset a RISE; a re-articulation inside a held
           vowel (no consonant) releases nothing
  attack   starts ATTACK_S before the nucleus as a raised cosine from that level, apex ON the onset
  hold     the target holds to the next nucleus's attack or to a closure; a held note keeps its target
  decay    DECAY_S, raised cosine, only into a rest (the voice off >= the event stage's REST_S), down to REST_OPEN (lips apart: a
           pause does not shut the mouth); any shorter gap, and every tongue consonant (L N T D G K NG) and short `sil`, keeps the
           vowel's level
  closure  a PP frame (the alignment's P / B / M) keeps its `open` 0: the closure stage owns it; a closure event ends the hold
  minimum  events closer than MIN_EVENT_S were merged by the event stage toward the earlier one; nothing is dropped here

`open` is written per frame; `vis`, `vis2`, `mix` and every other key are kept. The rest level before the first event is REST_OPEN.
"""

import numpy as np

MODES = ("off", "nucleus", "nucleus-follow")
FOLLOW_FLOOR = 0.6  # nucleus-follow: inside a hold the opening sits between this share of the target and the target, by the voice's level
FOLLOW_DB = 12.0  # dB under the syllable's own peak at which the follow reaches its floor
FOLLOW_S = 0.06  # s: the level's smoothing before it drives the follow (no frame-rate pumping)
V = {"AA": 1.0, "AE": 0.9, "AY": 0.9, "AW": 0.9, "AH": 0.75, "AO": 0.7, "EH": 0.7, "EY": 0.6, "OY": 0.6, "ER": 0.6, "OW": 0.55,
     "IH": 0.45, "IY": 0.4, "UH": 0.4, "UW": 0.35}  # fmt: skip
BINS = (0.15, 0.45, 0.8)
ATTACK_S = 0.10  # s before the onset the opening starts to rise (apex on the onset, less APEX_LEAD_S)
APEX_LEAD_S = 0.0  # s the apex is placed ahead of the onset. Measured and REFUSED: the rendered rise reaches 90 % of its height one
# output frame after the plan's (median 1.0 frame, 106 onsets, ledger S5R-apex-lag.json), but a 0.04 s lead moved the judge's rise mark
# (the attack's steepest point, ~50 ms before the apex) outside its +-60 ms and the plan's own onset reading fell 0.93 -> 0.63 while
# the render followed it down (0.67 -> 0.58); the realisation stage closes the render-plan gap instead. Kept as a constant at 0.
RELEASE_S = 0.06  # s the pre-onset release takes, ending where the attack starts
DIP = 0.19  # the pre-onset release depth under the lower of the two syllables' levels: the upper quartile of the six takes' own pre-onset
# dips (peak in the 150 ms after a judge onset minus the trough in the 150 ms before, over the take's p5..p95 range: median 0.03, q75 0.19,
# one onset in six at or over 0.30; tests/ledger take_dip.py). A singer closes visibly before a quarter of syllables; 0.19 is that closing,
# and the smallest the judge's rise rule (0.10 of the range over two frames) reads reliably after the warp's easing.
DECAY_S = 0.12  # s the opening falls into a rest
REST_OPEN = 0.15  # the opening at rest: lips apart, jaw relaxed (never a shut)
MIN_EVENT_S = 0.08  # s: the event stage's spacing; stated here, enforced there
TONGUE = ("L", "N", "T", "D", "G", "K", "NG")


def target_of(vowel, bin_):
    """the held opening of a nucleus."""
    j = BINS[int(bin_) if bin_ is not None and 0 <= int(bin_) < len(BINS) else 1]
    return round(float(V.get(str(vowel or "AH").rstrip("012").upper(), 0.75) * (0.6 + 0.5 * j)), 3)


def _cos(x):
    """raised cosine 0 -> 1 over x in [0, 1]."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def curve(times, events, rest=REST_OPEN, attack=ATTACK_S, decay=DECAY_S, lead=APEX_LEAD_S):
    """the held opening at each time (clip clock) from the event list [{kind, t_on, (t_off), vowel, bin}]; the attack's apex sits `lead`
    before the onset."""
    t = np.asarray(times, float)
    nuc = [e for e in events if e["kind"] == "nucleus"]
    bounds = sorted((e["t_on"], e["kind"], e) for e in events if e["kind"] in ("nucleus", "closure", "rest"))
    out = np.full(len(t), float(rest))
    level = float(rest)  # the level the next attack rises from
    for i, e in enumerate(nuc):
        tgt = target_of(e["vowel"], e["bin"])
        t_on = e["t_on"] - lead  # the apex lands here; the hold runs from it
        nxt = next(((b, k, x) for b, k, x in bounds if b > e["t_on"] + 1e-9), None)  # the next event of any kind
        end = nxt[0] if nxt else t[-1] + 1.0 if len(t) else t_on
        start = level
        if i > 0 and (e.get("cons") or "flux" in str(e.get("src") or "")):  # the pre-onset release: an articulated onset, or one the
            # spectral flux heard (a re-articulation with acoustic evidence; a recogniser-only vowel change inside a note has none)
            start = max(float(rest), min(level, tgt) - DIP)
            m = (t >= t_on - attack - RELEASE_S) & (t < t_on - attack)
            out[m] = level + (start - level) * _cos((t[m] - (t_on - attack - RELEASE_S)) / RELEASE_S)
        # the attack: from `start` to the target over [t_on - attack, t_on]
        m = (t >= t_on - attack) & (t < t_on)
        out[m] = start + (tgt - start) * _cos((t[m] - (t_on - attack)) / attack)
        # the hold: [t_on, end)
        m = (t >= t_on) & (t < end)
        out[m] = tgt
        if nxt and nxt[1] == "rest":  # the decay into a rest, over [end, end + decay], then the rest level
            rest_end = nxt[2].get("t_off", end)
            m = (t >= end) & (t < end + decay)
            out[m] = tgt + (rest - tgt) * _cos((t[m] - end) / decay)
            m = (t >= end + decay) & (t < rest_end)
            out[m] = rest
            level = float(rest)
        else:
            level = tgt  # the next attack starts from this level (a closure between keeps the level around it)
    return out


def follow(times, events, level, hop, c, floor=FOLLOW_FLOOR, span_db=FOLLOW_DB, smooth_s=FOLLOW_S):
    """nucleus-follow: the held curve c scaled, inside each nucleus's hold (from its onset to the next event), by the voice's level
    against that syllable's own peak: 1 at the peak, `floor` at span_db under it, smoothed over smooth_s. The attack and the release
    before an onset are left as they are, so every onset keeps its rise."""
    from scipy.ndimage import uniform_filter1d

    lv = np.asarray(level, float)
    if not len(lv):
        return c
    lv = uniform_filter1d(lv, max(1, int(round(smooth_s / hop))), mode="nearest")
    t = np.asarray(times, float)
    out = np.asarray(c, float).copy()
    starts = sorted(e["t_on"] for e in events if e["kind"] in ("nucleus", "closure", "rest"))
    for e in events:
        if e["kind"] != "nucleus":
            continue
        a = e["t_on"]
        b = next((s for s in starts if s > a + 1e-9), t[-1] + 1.0 if len(t) else a)
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        idx = np.clip((t[m] / hop).astype(int), 0, len(lv) - 1)
        peak = float(lv[np.clip(int(a / hop), 0, len(lv) - 1) : np.clip(int(b / hop) + 1, 1, len(lv))].max())
        share = np.clip(1.0 - (peak - lv[idx]) / span_db, 0.0, 1.0)
        out[m] = out[m] * (floor + (1.0 - floor) * share)
    return out


def held(frames, fps, t0=None, mode="nucleus", level=None, hop=0.01):
    """-> (new frames with `open` held per nucleus, info). Frames need the event stage's `ev`; frames without it come back unchanged.
    t0: the clip's start in the frames' `t` clock (default: the first frame's t). mode nucleus-follow: `follow` on top, from the
    clip's level per hop (level, clip clock)."""
    if not frames or not any("ev" in f for f in frames):
        return list(frames), {"mode": "nucleus", "events": 0, "note": "no events on the track"}
    t0 = float(frames[0].get("t", 0.0)) if t0 is None else float(t0)
    times = np.array([float(f.get("t", t0 + i / fps)) - t0 for i, f in enumerate(frames)])
    seen, events = set(), []
    for f in frames:
        e = f.get("ev") or {}
        if e.get("k", -1) >= 0 and e["k"] not in seen:
            seen.add(e["k"])
            events.append(dict(e))
    c = curve(times, events)
    if mode == "nucleus-follow" and level is not None:
        c = follow(times, events, level, hop, c)
    out = []
    for f, o in zip(frames, c):
        if f.get("vis") == "PP":
            out.append(dict(f))  # the closure stage's frame
        else:
            out.append(dict(f, open=round(float(np.clip(o, 0.0, 1.0)), 3)))
    o = np.array([f["open"] for f in out])
    sgn = np.sign(np.diff(o))
    sgn = sgn[sgn != 0]  # a plateau is not a turn
    reversals = int(np.sum(np.diff(sgn) != 0)) if len(sgn) > 1 else 0
    info = dict(
        mode=mode,
        events=len(events),
        nuclei=sum(1 for e in events if e["kind"] == "nucleus"),
        reversals=reversals,
        attack_s=ATTACK_S,
        apex_lead_s=APEX_LEAD_S,
        decay_s=DECAY_S,
        rest_open=REST_OPEN,
        release_s=RELEASE_S,
        dip=DIP,
        released=sum(1 for e in events if e["kind"] == "nucleus" and (e.get("cons") or "flux" in str(e.get("src") or ""))),
        targets=[dict(t_on=e["t_on"], vowel=e["vowel"], bin=e["bin"], target=target_of(e["vowel"], e["bin"])) for e in events if e["kind"] == "nucleus"],
    )  # fmt: skip
    return out, info
