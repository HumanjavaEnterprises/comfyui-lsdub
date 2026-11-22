# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""closure_plan.py: lip closures only where the lyric has a p, b or m, timed by the alignment read off the vocal. One stage of the
drive chain, behind one paint flag, --warp-closures (plan | aligned). Pure: track frames with the event stage's `ev` in, new frames out.

  drive (the track's frames with `ev`) -> [closures] -> the hold stage -> the aperture stage -> the warp's targets

plan     the frames as the viseme track planned them: a PP frame wherever the aligner's P / B / M fell, however the word was placed,
         and a `sil` press between words (the 0.1.11 behaviour)
aligned  every closure EVENT (an aligned P / B / M onset, from the event stage) gets PP frames from LEAD_S before its onset (a bilabial
         closes ahead of its burst), no earlier than GAP_S after the nucleus before it, to the phone's release, at least MIN_S; every
         PP frame the track carries with no closure event within TOL_S is lifted back to the class under it (`vis2`), its opening the
         nearest open frame's. Tongue consonants and short silences never shut the lips here; the word-boundary press is the viseme
         track's own flag (--word-close-vocal) and is conditioned on the vocal there.

aligned-press  the same placement; the paint then applies its existing closure levers (the lip-erase layer: the take's own closed
         lips from its most-closed frame; the slit seal) on the frames this stage shut and on no other frame (`pp_aligned`): a press
         on an open take is realised as a slit by the warp alone (~a third of its depth), these make it a contact.

lead (--warp-closure-lead, under aligned / aligned-press): where a closure's shut frames start.
  plan   LEAD_S (0.12 s) before the aligned P / B / M onset (the 0.3.x behaviour)
  phone  lead_s (--warp-closure-lead-s, default LEAD_PHONE_S = 0.02 s, one internal frame; 0 .. LEAD_S) before it: at 0.02, with
         the shape stage's one-frame advance, the lips are shut about 0.04 s before the bilabial, and the voiced vowel ahead of it keeps
         its opening. The gap rule (GAP_S) and the minimum (MIN_S) stay; the release end is plan's (a closure too short for MIN_S from the
         phone lead starts earlier instead, never later at the release).
         Track PP frames ahead of that window, by where they sit and not by their phone class first:
           - further than TOL_S ahead of the onset: a stray, lifted by the stage's own stray rule, on a vowel or a consonant alike (under
             plan these frames were shut only because plan's 0.12 s window covered them, not because the track's PP was kept). Such a
             PP is the viseme track's own early placement of the bilabial or a tongue consonant / short silence the lips need not close
             for: the closure's timing is the aligned onset's, so it does not shut the lips ahead of the window.
           - within TOL_S of the onset: on a vowel phone, lifted (the vowel ahead keeps its opening); on a consonant, kept (that close to
             the onset it is the stop's own closure, e.g. the T of "it back" running into the B: the lips may close over it).

  placed(frames, fps, lead="plan", lead_s=LEAD_PHONE_S) -> (frames, info)
"""

import numpy as np

MODES = (
    "plan",
    "aligned",
    "aligned-press",
)  # aligned-press: aligned, and the engine's closure levers (lip erase, slit seal) on those frames
LEAD_S = 0.12  # s the lips close ahead of the aligned P / B / M onset
GAP_S = 0.04  # s after the nucleus before it that a closure may start at the earliest
MIN_S = 0.04  # s a closure lasts at least (two internal frames at 50 fps)
TOL_S = 0.06  # s: a planned PP frame this near a closure event is the event's; further, it is a stray
LEADS = ("plan", "phone")  # --warp-closure-lead
LEAD_PHONE_S = 0.02  # s: lead "phone", one internal frame at 50 fps
VOWELS = frozenset(("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"))


def _window(e, events, lead_s=LEAD_S):
    """[start, end) of a closure event's shut frames."""
    t_on, t_off = e["t_on"], e.get("t_off", e["t_on"])
    prev = [x["t_on"] for x in events if x["kind"] == "nucleus" and x["t_on"] < t_on]
    a = t_on - lead_s
    if prev:
        a = max(a, max(prev) + GAP_S)
    b = max(t_off, a + MIN_S, t_on)
    return a, b


def _phone_window(e, events, lead_s=LEAD_PHONE_S):
    """lead "phone": [start, end) of a closure's shut frames. The start is lead_s before the onset (no earlier than GAP_S after the
    nucleus before it); the end is plan's, so the release side never moves; a window that would be shorter than MIN_S starts earlier
    (never earlier than plan's start)."""
    a_plan, b = _window(e, events)
    a, _ = _window(e, events, lead_s)
    return max(a_plan, min(a, b - MIN_S)), b


def _is_vowel(f):
    return str(f.get("phone") or "").rstrip("012").upper() in VOWELS


def lead_seconds(x):
    """--warp-closure-lead-s: a float in [0, LEAD_S] (seconds before the aligned onset), else ValueError."""
    v = float(x)
    if not (0.0 <= v <= LEAD_S):  # also refuses nan
        raise ValueError(f"--warp-closure-lead-s {x!r}: 0 <= lead <= {LEAD_S}")
    return v


def placed(frames, fps, t0=None, lead="plan", lead_s=LEAD_PHONE_S):
    """-> (new frames, info). Frames need the event stage's `ev`; frames without it come back unchanged. lead (LEADS): plan = shut
    from LEAD_S before each onset (lead_s is not read); phone = from lead_s (default LEAD_PHONE_S) before it, the vowel ahead kept
    open."""
    if lead not in LEADS:
        raise ValueError(f"--warp-closure-lead {lead!r}")
    if lead == "phone":
        lead_s = lead_seconds(lead_s)
    if not frames or not any("ev" in f for f in frames):
        return list(frames), {"mode": "aligned", "closures": 0, "lifted": [], "note": "no events on the track"}
    t0 = float(frames[0].get("t", 0.0)) if t0 is None else float(t0)
    times = np.array([float(f.get("t", t0 + i / fps)) - t0 for i, f in enumerate(frames)])
    seen, events = set(), []
    for f in frames:
        e = f.get("ev") or {}
        if e.get("k", -1) >= 0 and e["k"] not in seen:
            seen.add(e["k"])
            events.append(dict(e))
    closures = [e for e in events if e["kind"] == "closure"]
    out = [dict(f) for f in frames]
    n = len(out)
    shut = np.zeros(n, bool)
    windows = []
    starts = []
    for e in closures:
        a, b = _window(e, events) if lead == "plan" else _phone_window(e, events, lead_s)
        starts.append(a)
        m = (times >= a - 1e-9) & (times < b - 1e-9)
        if not m.any():  # shorter than a frame: the nearest frame to the onset
            m[int(np.argmin(np.abs(times - e["t_on"])))] = True
        shut |= m
        windows.append((round(a, 3), round(b, 3)))
    near = np.zeros(n, bool)
    for e in closures:
        near |= np.abs(times - e["t_on"]) <= TOL_S
    near |= shut
    lead_side = np.zeros(n, bool)
    if lead == "phone":  # a track PP on a vowel ahead of the window is the vowel's, not the closure's: the stray rule lifts it
        vowel = np.array([_is_vowel(f) for f in out])
        own = np.zeros(n, bool)  # near an event at or after its window's start: still that event's
        for e, a in zip(closures, starts):
            ne = np.abs(times - e["t_on"]) <= TOL_S
            lead_side |= ne & (times < a - 1e-9)
            own |= ne & (times >= a - 1e-9)
        lead_side &= vowel & ~shut & ~own
        near = shut.copy()
        for e in closures:
            near |= (np.abs(times - e["t_on"]) <= TOL_S) & ~lead_side
    lifted = []
    for i, f in enumerate(out):
        if f.get("vis") == "PP" and not near[i]:
            under = f.get("vis2") if f.get("vis2") not in (None, "", "PP") else "sil"
            f.update(vis=under, vis2="sil" if under != "sil" else under, mix=0.0)
            lifted.append(i)
    opens = np.array([float(f.get("open", 0.0)) for f in out])
    ok = np.array([f.get("vis") != "PP" and not shut[i] and i not in set(lifted) for i, f in enumerate(out)])
    for i in lifted:  # the opening of the nearest frame that is not a closure
        j = np.flatnonzero(ok)
        if len(j):
            out[i]["open"] = round(float(opens[j[np.argmin(np.abs(j - i))]]), 3)
    added = []
    for i in np.flatnonzero(shut):
        f = out[i]
        if f.get("vis") != "PP":
            f.update(vis2=f.get("vis") if f.get("vis") != "PP" else f.get("vis2"), vis="PP", mix=0.0)
            added.append(int(i))
        f["open"] = 0.0
        f["pp_aligned"] = True  # the press mode's frames: the closure levers apply here and nowhere else
    info = dict(mode="aligned", closures=len(closures), windows=windows, frames_shut=int(shut.sum()), added=added, lifted=lifted)
    if lead == "phone":
        info.update(lead="phone", lead_s=lead_s, lifted_lead=[i for i in lifted if lead_side[i]])
    return out, info
