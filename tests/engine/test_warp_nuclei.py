# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""--warp-nuclei lyric (sync 0.4 step 6): only the detections the aligned lyric has a vowel for are syllable events."""

import numpy as np
import pytest

from lsdub_engine.audits import event_judge as EJ
from lsdub_engine.core import mouth_paint as MP
from lsdub_engine.core import nucleus_events as NE
from lsdub_engine.core import nucleus_hold as NH


def nuc(t, cons=False, src="nucflux", vowel="AA"):
    return dict(kind="nucleus", t_on=t, vowel=vowel, db=0.0, bin=1, cons=cons, src=src)


WORDS = [
    dict(w="so", phones=[dict(p="S", start=0.40, end=0.50), dict(p="OW1", start=0.50, end=1.60)]),
    dict(w="far", phones=[dict(p="F", start=1.60, end=1.70), dict(p="AA1", start=1.70, end=2.40), dict(p="R", start=2.40, end=2.5)]),
]
EV = [nuc(0.52, cons=True), nuc(0.80, src="flux"), nuc(1.10, src="flux"), nuc(1.35, cons=True, src="flux"), nuc(1.72), nuc(2.05, src="nuc")]


def test_lyric_vowels_are_the_vowel_phone_starts():
    assert NE.lyric_vowels(WORDS) == [0.50, 1.70]
    assert NE.lyric_vowels([dict(w="x")]) == [] and NE.lyric_vowels(None) == []


def test_one_detection_per_lyric_vowel_the_nearest():
    out, dropped = NE.confirmed(EV, WORDS)
    assert [e["t_on"] for e in out] == [0.52, 1.72] and dropped == 4


def test_a_dropped_detections_consonant_passes_to_the_next_kept_nucleus():
    out, _ = NE.confirmed(EV, WORDS)
    assert out[0]["cons"] is True and out[1]["cons"] is True  # 1.72 had none of its own; the dropped 1.35 did
    assert EV[4]["cons"] is False  # the input is not edited


def test_closures_and_rests_are_kept_and_no_lyric_confirms_nothing():
    ev = [*EV, dict(kind="closure", t_on=0.3, t_off=0.36), dict(kind="rest", t_on=2.6, t_off=3.0)]
    out, _ = NE.confirmed(ev, WORDS)
    assert sum(e["kind"] == "closure" for e in out) == 1 and sum(e["kind"] == "rest" for e in out) == 1
    assert NE.confirmed(EV, []) == (EV, 0)
    assert NE.confirmed(EV, [dict(w="mm", phones=[dict(p="M", start=0.5, end=0.9)])]) == (EV, 0)


def test_a_detection_outside_the_reach_is_no_syllable():
    out, dropped = NE.confirmed([nuc(0.52 + NE.LYRIC_S + 0.01), nuc(1.72)], WORDS)
    assert [e["t_on"] for e in out] == [1.72] and dropped == 1


def test_the_held_note_makes_one_rise_instead_of_four():
    fps = 50.0
    frames = [dict(t=i / fps, vis="aa", vis2="aa", mix=0.0, open=0.5) for i in range(150)]

    def rises(ev):
        f, _ = NE.annotate(frames, ev, fps)
        f, _ = NH.held(f, fps, mode="nucleus")
        return EJ.rises(np.array([x["open"] for x in f])[::2], 25.0)

    every, lyric = rises(EV), rises(NE.confirmed(EV, WORDS)[0])
    assert len(lyric) < len(every)
    assert EJ.match([0.52, 1.72], lyric, EJ.TOL)[0] == 2  # both sung syllables keep their rise


# the held vowel (round 2): `confirmed` alone left a sustained vowel with no event when its detections sat past the reach, and dropped
# the detections that follow a held note's level. `supported` is the flag's rule.
HELD = [
    dict(w="know", phones=[dict(p="N", start=0.00, end=0.08), dict(p="OW1", start=0.08, end=1.80)]),
    dict(w="it", phones=[dict(p="IH1", start=1.80, end=1.90), dict(p="T", start=1.90, end=2.00)]),
]
HELD_EV = [  # the note swells in: the first detection is 0.12 s past the aligned start; a breath after the word
    nuc(0.20, src="nuc", vowel="AH"),
    nuc(0.60, src="flux", vowel="AH"),
    nuc(1.10, cons=True, src="nucflux", vowel="UW"),
    nuc(1.82, vowel="OW"),
    nuc(2.30, src="flux", vowel="AH"),
]


def _open(ev, n=125, fps=50.0):
    frames = [dict(t=i / fps, vis="aa", vis2="aa", mix=0.0, open=0.5) for i in range(n)]
    f, _ = NE.annotate(frames, ev, fps)
    f, _ = NH.held(f, fps, mode="nucleus")
    return np.array([x["open"] for x in f])


def test_a_held_vowel_keeps_its_late_onset_and_its_sustain():
    out, dropped = NE.supported(HELD_EV, HELD)
    assert [(e["t_on"], e["src"], e["vowel"]) for e in out] == [
        (0.20, "nuc", "OW"),
        (0.60, "held", "OW"),
        (1.10, "held", "OW"),
        (1.82, "nucflux", "IH"),
    ]
    assert dropped == 1  # the breath at 2.30 is inside no sung vowel
    assert [e["t_on"] for e in NE.confirmed(HELD_EV, HELD)[0]] == [1.82]  # the onset rule alone: the 1.7 s vowel has no event at all


def test_a_held_detection_is_no_new_syllable():
    out, _ = NE.supported(HELD_EV, HELD)
    assert [e["cons"] for e in out] == [False, False, False, True]  # the held 1.10's consonant passes to the next onset
    assert HELD_EV[2]["cons"] is True and HELD_EV[2]["src"] == "nucflux"  # the input is not edited
    assert NE.supported(HELD_EV, []) == (HELD_EV, 0)


def test_one_onset_per_vowel_still_the_nearest_and_the_rest_of_the_note_is_held():
    out, dropped = NE.supported(EV, WORDS)
    assert [(e["t_on"], e["src"]) for e in out] == [
        (0.52, "nucflux"),
        (0.80, "held"),
        (1.10, "held"),
        (1.35, "held"),
        (1.72, "nucflux"),
        (2.05, "held"),
    ]
    assert dropped == 0 and [e["vowel"] for e in out] == ["OW"] * 4 + ["AA"] * 2
    assert out[4]["cons"] is True  # the held 1.35's consonant still reaches the next onset


def test_the_mouth_stays_open_through_the_held_vowel():
    fps = 50.0
    held = _open(NE.supported(HELD_EV, HELD)[0])
    onset_only = _open(NE.confirmed(HELD_EV, HELD)[0])
    span = slice(int(0.30 * fps), int(1.70 * fps))  # inside the sustained OW
    assert held[span].min() > NH.REST_OPEN + 0.1  # open for the whole vowel
    assert onset_only[span].max() <= NH.REST_OPEN + 1e-6  # what round 1 did: at rest until the next word
    every = EJ.rises(_open(HELD_EV)[::2], 25.0)
    assert len(EJ.rises(held[::2], 25.0)) < len(every)  # and still fewer rises than every detection makes


def test_events_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        NE.events(np.zeros(16000), [], 1.0, nuclei="some")


def test_flag_parses_and_classic_is_all():
    ap = MP._parser()
    base = ["--video", "v.mp4", "--track", "t.json", "--out", "o.mp4"]
    assert ap.parse_args(base).warp_nuclei == "all"
    assert ap.parse_args([*base, "--warp-nuclei", "classic"]).warp_nuclei == "all"
    assert ap.parse_args([*base, "--warp-nuclei", "lyric"]).warp_nuclei == "lyric"
