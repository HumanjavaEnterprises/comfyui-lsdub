# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""nucleus_hold: the target per vowel x bin is exact; the apex sits on the onset frame; no decay into a 140 ms gap, decay into a 150 ms
rest; a held 1.5 s vowel is flat; tongue-consonant frames keep the vowel's opening; a PP frame keeps its 0; the reversal count on a
20-nucleus line is the syllable count; frames without events come back unchanged."""

import numpy as np
import pytest

from lsdub_engine.core import nucleus_hold as NH

FPS = 50.0
LEAD = int(round(NH.APEX_LEAD_S * FPS))  # the apex sits this many frames before the onset (the measured realisation lag)


def frames(n, fps=FPS, t0=0.0, vis="aa", phone="AA"):
    return [dict(t=round(t0 + k / fps, 3), vis=vis, vis2="sil", mix=0.0, open=0.5, word="w", phone=phone) for k in range(n)]


def tag(frames, events, fps=FPS):
    """the event stage's annotation, by hand: closure / rest spans own their frames, else the latest nucleus."""
    out = []
    for i, f in enumerate(frames):
        t = f["t"] - frames[0]["t"]
        cur, k = None, -1
        for kk, e in enumerate(events):
            if e["kind"] in ("closure", "rest") and e["t_on"] <= t < e["t_off"]:
                cur, k = e, kk
        if cur is None:
            for kk, e in enumerate(events):
                if e["kind"] == "nucleus" and e["t_on"] <= t:
                    cur, k = e, kk
        ev = dict(k=k, kind=cur["kind"] if cur else "rest", t_on=cur["t_on"] if cur else None, vowel=cur.get("vowel") if cur else None,
                  db=0.0, bin=cur.get("bin") if cur else None, cons=cur.get("cons", False) if cur else False, src=cur.get("src") if cur else None)  # fmt: skip
        if cur and "t_off" in cur:
            ev["t_off"] = cur["t_off"]
        out.append(dict(f, ev=ev))
    return out


def nucleus(t, vowel="AA", b=1, cons=False):
    return dict(kind="nucleus", t_on=t, vowel=vowel, bin=b, cons=cons)


def rest(a, b):
    return dict(kind="rest", t_on=a, t_off=b)


def test_the_target_per_vowel_and_bin_is_exact():
    assert NH.target_of("AA", 2) == pytest.approx(1.0) and NH.target_of("AA", 0) == pytest.approx(0.675)
    assert NH.target_of("IY", 1) == pytest.approx(0.4 * (0.6 + 0.5 * 0.45)) and NH.target_of("UW1", 2) == pytest.approx(0.35)
    assert NH.target_of(None, None) == NH.target_of("AH", 1)  # an unknown vowel reads as AH, an unknown bin as the middle


def test_the_apex_sits_on_the_onset_and_the_attack_starts_attack_s_before_it():
    ev = [nucleus(1.0, "AA", 2)]
    out, info = NH.held(tag(frames(100), ev), FPS)
    o = [f["open"] for f in out]
    k = 50 - LEAD  # the apex, APEX_LEAD_S before the 1.0 s onset
    assert (
        o[k] == pytest.approx(1.0) and o[k - 5] == pytest.approx(NH.REST_OPEN) and NH.REST_OPEN < o[k - 2] < 1.0
    )  # 5 frames = 0.10 s attack
    assert all(v == pytest.approx(1.0) for v in o[k:]) and all(v == pytest.approx(NH.REST_OPEN) for v in o[: k - 5])
    assert info["nuclei"] == 1 and info["targets"][0]["target"] == 1.0


def test_no_decay_into_a_short_gap_but_decay_into_a_rest():
    short = [nucleus(0.5, "AA", 2), nucleus(1.5, "IH", 1)]  # a 140 ms unvoiced gap between them is no event: the hold carries through
    o = [f["open"] for f in NH.held(tag(frames(120), short), FPS)[0]]
    assert all(v == pytest.approx(1.0) for v in o[25 - LEAD : 70 - LEAD - 1])  # held from the apex to the next attack's start
    with_rest = [nucleus(0.5, "AA", 2), rest(1.0, 1.15), nucleus(1.5, "IH", 1)]  # a 150 ms rest: the opening decays over 120 ms into it
    o = [f["open"] for f in NH.held(tag(frames(120), with_rest), FPS)[0]]
    assert o[49] == pytest.approx(1.0) and o[56] == pytest.approx(NH.REST_OPEN) and NH.REST_OPEN < o[53] < 1.0  # 1.0 s .. 1.12 s decay
    assert o[57] == pytest.approx(NH.REST_OPEN)  # the rest level: lips apart, never 0


def test_a_held_vowel_is_flat_and_tongue_consonants_keep_it():
    fr = frames(100)
    for k in range(40, 44):  # a T inside the vowel, as the track names it
        fr[k].update(vis="DD", phone="T")
    out, info = NH.held(tag(fr, [nucleus(0.2, "AA", 2)]), FPS)
    o = [f["open"] for f in out]
    assert all(v == pytest.approx(1.0) for v in o[10:]) and info["reversals"] == 0  # 1.5 s held, flat; the T frames keep 1.0
    assert out[41]["vis"] == "DD" and out[41]["open"] == pytest.approx(1.0)


def test_a_pp_frame_keeps_its_zero_and_a_closure_event_ends_the_hold():
    fr = frames(100)
    for k in range(50, 53):
        fr[k].update(vis="PP", phone="M", open=0.0)
    ev = [nucleus(0.2, "AA", 2), dict(kind="closure", t_on=1.0, t_off=1.06), nucleus(1.1, "IY", 0)]
    out, _ = NH.held(tag(fr, ev), FPS)
    o = [f["open"] for f in out]
    assert o[51] == 0.0 and out[51]["vis"] == "PP"
    assert o[47] == pytest.approx(1.0) and o[60] == pytest.approx(
        NH.target_of("IY", 0)
    )  # 0.94 s: before the next attack (1.1 - lead - 0.10)


def test_a_sixty_ms_event_is_not_dropped_and_the_reversal_count_is_the_syllable_count():
    ev = [nucleus(0.3 + 0.2 * i, ["AA", "IY"][i % 2], 2) for i in range(20)]
    out, info = NH.held(tag(frames(250), ev), FPS)
    assert info["nuclei"] == 20 and info["reversals"] == 19  # alternating up / down: one turn at every syllable boundary
    late = [nucleus(0.3, "AA", 2), nucleus(0.36, "IY", 2)]  # 60 ms apart: the event stage would have merged them; here both are kept
    out, info = NH.held(tag(frames(50), late), FPS)
    assert info["nuclei"] == 2


def test_frames_without_events_come_back_unchanged_and_other_keys_are_kept():
    fr = frames(10)
    out, info = NH.held(fr, FPS)
    assert out == fr and info["events"] == 0
    tagged = tag(frames(30), [nucleus(0.2, "AE", 1)])
    out, _ = NH.held(tagged, FPS)
    assert all(set(f) == set(g) for f, g in zip(out, tagged)) and [f["vis"] for f in out] == [f["vis"] for f in tagged]
    assert all(0.0 <= f["open"] <= 1.0 for f in out)


def test_an_articulated_onset_releases_before_its_attack_and_a_rearticulation_does_not():
    """the second syllable, lower than the first and with a consonant between: a dip of DIP under its level over RELEASE_S, then the
    attack to its target, apex on the onset (a rise the judge can read); the same syllable without a consonant holds flat into it."""
    ev = [nucleus(0.5, "AA", 2), nucleus(1.5, "AH", 1, cons=True)]
    o = [f["open"] for f in NH.held(tag(frames(100), ev), FPS)[0]]
    tgt = NH.target_of("AH", 1)
    k = 75 - LEAD  # the apex, APEX_LEAD_S before the 1.5 s onset
    assert o[k] == pytest.approx(tgt, abs=1e-3) and o[k - 5] == pytest.approx(
        max(NH.REST_OPEN, tgt - NH.DIP), abs=1e-3
    )  # the attack rises from the dip
    assert o[k - 8] == pytest.approx(1.0)  # before the release: the AA level
    assert all(v == pytest.approx(1.0) for v in o[25 : k - 8])
    flat = [nucleus(0.5, "AA", 2), nucleus(1.5, "AH", 1, cons=False)]
    o2 = [f["open"] for f in NH.held(tag(frames(100), flat), FPS)[0]]
    assert (
        o2[k - 5] == pytest.approx(1.0) and o2[k] == pytest.approx(tgt, abs=1e-3) and min(o2[25:k]) >= tgt - 1e-3
    )  # no dip, a step down (25: after the first attack)
    assert NH.held(tag(frames(100), ev), FPS)[1]["released"] == 1


def test_nucleus_follow_scales_the_hold_by_the_voices_level_and_keeps_the_onset():
    """a 1.0 s AA held while the voice decays 12 dB over it: the opening falls from the target to FOLLOW_FLOOR of it inside the hold;
    the attack before the onset and the apex on it are untouched; a level that holds keeps the hold flat."""
    ev = [nucleus(0.5, "AA", 2)]
    fr = tag(frames(100), ev)
    lv = np.zeros(200)  # 10 ms rows over 2 s: -0 dB at 0.5 s falling to -12 dB at 1.5 s
    lv[50:150] = -12.0 * (np.arange(100) / 99.0)
    lv[150:] = -12.0
    o = [f["open"] for f in NH.held(fr, FPS, mode="nucleus-follow", level=lv)[0]]
    k = 25 - LEAD
    assert o[k] == pytest.approx(1.0, abs=0.02) and o[k - 5] == pytest.approx(NH.REST_OPEN) and NH.REST_OPEN < o[k - 2] < 1.0
    assert o[74] == pytest.approx(NH.FOLLOW_FLOOR, abs=0.05) and o[50] == pytest.approx(0.5 * (1 + NH.FOLLOW_FLOOR), abs=0.08)
    flat = [f["open"] for f in NH.held(fr, FPS, mode="nucleus-follow", level=np.zeros(200))[0]]
    assert all(v == pytest.approx(1.0) for v in flat[25 - LEAD :])
    assert NH.held(fr, FPS, mode="nucleus-follow", level=lv)[1]["mode"] == "nucleus-follow"


def test_a_flux_heard_nucleus_releases_without_a_consonant_and_a_recogniser_only_one_does_not():
    ev = [nucleus(0.5, "AA", 2), dict(kind="nucleus", t_on=1.5, vowel="AH", bin=1, cons=False, src="nucflux")]
    o = [f["open"] for f in NH.held(tag(frames(100), ev), FPS)[0]]
    k, tgt = 75 - LEAD, NH.target_of("AH", 1)
    assert o[k - 5] == pytest.approx(max(NH.REST_OPEN, tgt - NH.DIP), abs=1e-3) and o[k] == pytest.approx(tgt, abs=1e-3)
    ev2 = [nucleus(0.5, "AA", 2), dict(kind="nucleus", t_on=1.5, vowel="AH", bin=1, cons=False, src="nuc")]
    o2 = [f["open"] for f in NH.held(tag(frames(100), ev2), FPS)[0]]
    assert o2[k - 5] == pytest.approx(1.0)  # no consonant, not heard by the flux: held, a step down
