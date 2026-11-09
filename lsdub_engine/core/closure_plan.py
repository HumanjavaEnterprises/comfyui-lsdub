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

  placed(frames, fps) -> (frames, info)
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


def _window(e, events):
    """[start, end) of a closure event's shut frames."""
    t_on, t_off = e["t_on"], e.get("t_off", e["t_on"])
    prev = [x["t_on"] for x in events if x["kind"] == "nucleus" and x["t_on"] < t_on]
    a = t_on - LEAD_S
    if prev:
        a = max(a, max(prev) + GAP_S)
    b = max(t_off, a + MIN_S, t_on)
    return a, b


def placed(frames, fps, t0=None):
    """-> (new frames, info). Frames need the event stage's `ev`; frames without it come back unchanged."""
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
    for e in closures:
        a, b = _window(e, events)
        m = (times >= a - 1e-9) & (times < b - 1e-9)
        if not m.any():  # shorter than a frame: the nearest frame to the onset
            m[int(np.argmin(np.abs(times - e["t_on"])))] = True
        shut |= m
        windows.append((round(a, 3), round(b, 3)))
    near = np.zeros(n, bool)
    for e in closures:
        near |= np.abs(times - e["t_on"]) <= TOL_S
    near |= shut
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
    return out, info
