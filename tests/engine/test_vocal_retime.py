# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""word times read off the vocal: a held vowel, the consonants where the vocal has them, the lyric untouched."""

import copy, json, os

import numpy as np
import pytest

from lsdub_engine.core import vocal_retime as V


def tone(s, f0=220.0):
    t = np.arange(int(s * V.SR)) / V.SR
    return 0.3 * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))


def hiss(s):
    r = np.random.default_rng(1)
    x = np.diff(r.normal(0, 0.15, int(s * V.SR) + 1))
    return x


def word(w, ph):
    return dict(w=w, start=ph[0][1], end=ph[-1][2], phones=[dict(p=p, start=a, end=b) for p, a, b in ph])


def line():
    """ "a sat at": a held vowel, a hiss, a short vowel, a burst, a long vowel, a burst; the aligner spread them evenly."""
    x = np.r_[tone(1.2), hiss(0.1), tone(0.3), hiss(0.08), tone(0.6), hiss(0.08), tone(0.15)]
    W = [
        word("a", [("AH", 0.0, 0.5)]),
        word("sat", [("S", 0.5, 0.6), ("AE", 0.6, 0.9), ("T", 0.9, 1.0)]),
        word("at", [("AE", 1.0, 1.5), ("T", 1.5, 1.6)]),
    ]
    return x, dict(words=W, aligner="test")


@pytest.mark.parametrize(
    "p,c", [("AH0", "V"), ("S", "H"), ("T", "S"), ("M", "N"), ("Z", "W"), ("W", "G"), ("L", "G"), ("B", "N"), ("HH", "H")]
)
def test_phone_class(p, c):
    assert V.phone_class(p) == c


def test_features_tell_voiced_from_hiss_from_silence():
    x = np.r_[tone(0.4), hiss(0.2), np.zeros(int(0.3 * V.SR)), tone(0.3)]
    f = V.features(x)
    assert f["voiced"][10:30].mean() > 0.8 and f["voiced"][45:55].mean() < 0.3
    assert f["hiss"][45:55].mean() > 0.7 and f["hiss"][10:30].mean() < 0.2
    assert f["silent"][65:80].mean() > 0.8


def test_consonants_land_on_the_vocal_and_the_held_vowel_takes_the_rest():
    x, A = line()
    r = V.retime(A, "", x=x)
    S, T, T2 = r["words"][1]["phones"][0], r["words"][1]["phones"][2], r["words"][2]["phones"][1]
    assert (
        S["start"] == pytest.approx(1.2, abs=0.05)
        and T["start"] == pytest.approx(1.6, abs=0.05)
        and T2["start"] == pytest.approx(2.28, abs=0.05)
    )
    assert r["words"][0]["end"] > 1.1  # the held vowel holds


def test_lyric_order_and_fields_stay_and_the_old_times_are_kept():
    x, A = line()
    r = V.retime(A, "", x=x)
    assert [w["w"] for w in r["words"]] == ["a", "sat", "at"] and r["aligner"] == "test" and r["retimed"] == "vocal"
    assert r["words"][1]["start0"] == 0.5
    t = [p["start"] for w in r["words"] for p in w["phones"]]
    assert t == sorted(t)
    assert A["words"][1]["start"] == 0.5  # the caller's alignment is not touched


def test_what_the_vocal_never_reaches_stays_past_its_end():
    x, A = line()
    cut = x[: int(2.3 * V.SR)]  # stops on the last burst's first frames
    r = V.retime(A, "", x=cut)
    assert r["words"][-1]["end"] >= 2.25


def test_matching_times_stay_home():
    x = np.r_[tone(0.5), hiss(0.1), tone(0.5)]
    A = dict(words=[word("a", [("AH", 0.0, 0.5)]), word("as", [("S", 0.5, 0.6), ("AH", 0.6, 1.1)])])
    r = V.retime(A, "", x=x)
    assert r["words"][1]["start"] == pytest.approx(0.5, abs=0.04)


def test_word_without_phones_keeps_its_time():
    x, A = line()
    A["words"].insert(1, dict(w="uh", start=0.5, end=0.5, phones=[]))
    r = V.retime(A, "", x=x)
    assert r["words"][1]["start"] == 0.5 and "start0" not in r["words"][1]


def test_word_offsets_reports_median_and_worst():
    a = [dict(w="x", start=1.0, phones=[1]), dict(w="y", start=2.5, phones=[1]), dict(w="z", start=3.0, phones=[1])]
    b = [dict(w="x", start=1.0), dict(w="y", start=2.0), dict(w="z", start=3.05)]
    r = V.word_offsets(a, b, 0.0, 10.0)
    assert r == dict(vocal_off_ms=50, vocal_off_n=1, vocal_words=3, vocal_worst="y +500")
    assert V.word_offsets(a, b, 5.0, 6.0) == {}


# ---- the walk's ends, on a tone and on the real demo vocal -------------------------------------------------------------------------------
DATA = os.path.join(os.path.dirname(__file__), "data")


def demo():
    if not os.path.exists(os.path.join(DATA, "demo_align.json")):
        pytest.skip("the demo vocal is not in this tree")
    return json.load(open(os.path.join(DATA, "demo_align.json"))), V.read_mono(os.path.join(DATA, "demo_lead_16k.flac"))


def shifted(A, d):
    B = copy.deepcopy(A)
    for w in B["words"]:
        w["start"] += d
        w["end"] += d
        for p in w["phones"]:
            p["start"] += d
            p["end"] += d
    return B


def starts(r):
    return np.array([p["start"] for w in r["words"] for p in w["phones"]])


@pytest.mark.parametrize("lead,tail", [(0.5, 0.0), (0.9, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 0.5), (0.0, 2.0), (5.0, 3.0)])
def test_silence_at_either_end_leaves_the_real_vocal_where_it_was(lead, tail):
    A, x = demo()
    base = V.retime(A, "", x=x)
    y = np.r_[np.zeros(int(lead * V.SR)), x, np.zeros(int(tail * V.SR))]
    r = V.retime(shifted(A, lead), "", x=y)
    e = starts(r) - lead - starts(base)
    assert np.median(np.abs(e)) < 0.02 and np.abs(e[:-3]).max() < 0.05  # the last word's unsung phones excepted
    assert r["words"][0]["start"] >= lead
    assert r["words"][-1]["end"] < lead + 15.3 + 0.8  # no word stretched into the trailing silence


@pytest.mark.parametrize("lead", [0.5, 0.9, 1.0, 2.5])
def test_a_tone_line_after_silence_keeps_its_first_word_off_zero(lead):
    x, A = line()
    y = np.r_[np.zeros(int(lead * V.SR)), x, np.zeros(int(1.0 * V.SR))]
    r = V.retime(shifted(A, lead), "", x=y)
    assert r["words"][0]["start"] == pytest.approx(lead, abs=0.03)
    assert r["words"][1]["start"] == pytest.approx(lead + 1.2, abs=0.05)
    assert r["words"][2]["end"] < lead + 2.6 + 0.1  # the last word ends with the sound, not in the silence


def test_an_alignment_far_from_the_vocal_still_has_a_path():
    x, A = line()
    r = V.retime(shifted(A, 6.0), "", x=np.r_[x, np.zeros(V.SR)])  # every phone outside any band: no frame is out of reach
    assert r["words"][0]["start"] < 3.0 and all(w["end"] >= w["start"] for w in r["words"])


# hand reading of the demo vocal off its spectrogram: (word, phone, s)
EVENTS = [
    (0, 1, 0.23),
    (2, 3, 1.36),
    (3, 0, 1.61),
    (6, 3, 3.16),
    (7, 8, 4.15),
    (10, 0, 4.62),
    (10, 3, 5.10),
    (11, 0, 5.28),
    (11, 1, 5.48),
    (12, 0, 5.92),
    (13, 2, 7.10),
    (14, 0, 7.29),
    (14, 2, 7.45),
    (16, 3, 8.27),
    (17, 1, 8.45),
    (17, 2, 8.65),
    (17, 3, 8.91),
    (18, 1, 9.12),
    (18, 2, 9.34),
    (19, 1, 9.60),
    (19, 2, 9.93),
    (20, 0, 10.05),
    (21, 0, 10.53),
    (22, 1, 12.25),
    (22, 3, 12.42),
    (22, 4, 12.75),
    (23, 2, 13.16),
    (24, 1, 13.38),
    (24, 2, 13.67),
    (25, 0, 13.80),
    (25, 2, 13.94),
    (26, 0, 14.10),
    (27, 0, 14.19),
    (27, 1, 14.23),
    (27, 2, 14.34),
    (27, 3, 14.49),
    (28, 0, 14.52),
    (28, 2, 14.66),
    (29, 2, 15.09),
]  # the tail read with "time" sung inside the clip: its M on the dip at 15.02-15.10, where the take shuts


def misses(r, tol=0.1):
    return [(w, p, t) for w, p, t in EVENTS if abs(r["words"][w]["phones"][p]["start"] - t) > tol]


def test_the_demo_vocal_lands_on_the_hand_reading():
    A, x = demo()
    assert misses(V.retime(A, "", x=x)) == []


def test_winding_holds_the_glide_and_sits_the_vowel_on_the_loud_part():
    A, x = demo()
    w = V.retime(A, "", x=x)["words"][11]["phones"]
    W, AY, IH = w[0], w[1], w[4]
    assert W["end"] - W["start"] > 0.15  # the quiet /w/ is the glide's
    assert AY["start"] == pytest.approx(5.5, abs=0.05) and AY["end"] - AY["start"] > 0.2  # the loud part is the diphthong's
    assert IH["end"] - IH["start"] < 0.15


def test_a_sung_nasal_with_no_level_dip_is_found_by_its_band_ratio():
    A, x = demo()
    m = V.retime(A, "", x=x)["words"][19]["phones"][2]
    assert m["start"] == pytest.approx(9.93, abs=0.05) and m["end"] == pytest.approx(10.04, abs=0.05)
    f = V.features(x)
    k = lambda t: int(round((t - V.WIN / 2) / V.HOP))
    assert f["nasal"][k(9.97) : k(10.02)].mean() > 0.7 and f["quiet"][k(9.97) : k(10.02)].mean() < 0.2  # no dip: only the ratio sees it
    assert f["nasal"][k(9.62) : k(9.78)].mean() < 0.1  # the open vowel before it


def test_times_are_centred_on_the_frames():
    # a clean edge at a known time: the reported join must sit on it, not a window early or a hop late
    for edge in (1.203, 1.217):
        x = np.r_[tone(edge), hiss(0.25), tone(0.8)]
        A = dict(words=[word("a", [("AH", 0.0, 1.0)]), word("sa", [("S", 1.0, 1.3), ("AH", 1.3, 2.0)])])
        r = V.retime(A, "", x=x)
        assert r["words"][1]["start"] == pytest.approx(edge, abs=0.012)
        assert r["words"][1]["phones"][1]["start"] == pytest.approx(edge + 0.25, abs=0.012)


def test_a_hiss_the_vocal_never_sounds_is_stepped_over_and_a_blip_earns_nothing():
    x = np.r_[tone(1.0), tone(0.5), hiss(0.02), tone(0.6)]  # "if it": no /f/ anywhere, a 20 ms blip further on
    A = dict(words=[word("if", [("IH", 0.0, 0.8), ("F", 0.8, 0.9)]), word("it", [("IH", 0.9, 2.1)])])
    r = V.retime(A, "", x=x)
    f = r["words"][0]["phones"][1]
    assert f["end"] == f["start"]  # no time of its own, so no mouth shape


def test_walking_a_real_hiss_still_places_it():
    x = np.r_[tone(1.0), hiss(0.15), tone(0.6)]
    A = dict(words=[word("if", [("IH", 0.0, 0.8), ("F", 0.8, 0.9)]), word("it", [("IH", 0.9, 1.7)])])
    f = V.retime(A, "", x=x)["words"][0]["phones"][1]
    assert f["start"] == pytest.approx(1.0, abs=0.03) and f["end"] - f["start"] > 0.1


def test_words_run_on_with_no_gap_frame_between_them():
    x = tone(2.0)
    A = dict(words=[word("a", [("AH", 0.0, 1.0)]), word("o", [("OW", 1.0, 2.0)])])
    r = V.retime(A, "", x=x)
    assert r["words"][0]["end"] == r["words"][1]["start"]  # the gap between words may be skipped


def test_a_clip_cut_mid_word_leaves_what_it_never_reached_past_the_stop_each_with_its_length():
    x, A = line()
    r = V.retime(A, "", x=x[: int(1.3 * V.SR)])  # stops inside "sat"
    p = [q for w in r["words"] for q in w["phones"]]
    assert p[-1]["p"] == "T" and p[-1]["end"] - p[-1]["start"] == pytest.approx(0.1, abs=0.001)  # own length, not squeezed to nothing
    assert p[-2]["end"] - p[-2]["start"] == pytest.approx(0.5, abs=0.001)
    assert [q["start"] for q in p] == sorted(q["start"] for q in p)


def test_the_short_vowel_floor_keeps_a_phone_on_screen():
    x, A = line()
    r = V.retime(A, "", x=x)
    for w in r["words"]:
        for p in w["phones"]:
            if V.phone_class(p["p"]) == "V":
                assert p["end"] - p["start"] >= V.MIN_S["V"] - 0.011


def test_a_vowel_keeps_its_share_of_the_aligned_length():
    x = tone(1.5)
    A = dict(words=[word("o", [("OW", 0.0, 1.0)]), word("a", [("AH", 1.0, 1.1)]), word("o", [("OW", 1.1, 1.5)])])
    r = V.retime(A, "", x=x)
    assert r["words"][2]["end"] - r["words"][2]["start"] >= V.SHARE * 0.4 - 0.011
    assert r["words"][0]["end"] - r["words"][0]["start"] >= V.SHARE - 0.011


def test_an_aligner_spread_stop_lands_in_the_dip_with_only_the_level_saying_so():
    # no hiss at all: a voiced tone with a 16 dB dip; the /p/ has only the dip
    x = np.r_[tone(0.7), 0.15 * tone(0.12), tone(0.7)]
    A = dict(words=[word("up", [("AH", 0.0, 0.5), ("P", 0.5, 0.6)]), word("o", [("OW", 0.6, 1.52)])])
    p = V.retime(A, "", x=x)["words"][0]["phones"][1]
    assert p["start"] == pytest.approx(0.7, abs=0.04)


def test_a_voiced_closure_lands_in_the_dip_not_on_the_spread_time():
    x = np.r_[tone(0.7), 0.15 * tone(0.12), tone(0.7)]
    A = dict(words=[word("am", [("AH", 0.0, 0.5), ("M", 0.5, 0.6)]), word("o", [("OW", 0.6, 1.52)])])
    p = V.retime(A, "", x=x)["words"][0]["phones"][1]
    assert p["start"] == pytest.approx(0.7, abs=0.04)


def test_the_aligner_breaks_a_tie_and_only_a_tie():
    x = tone(2.0)  # nothing in the vocal between the two vowels
    A = dict(words=[word("a", [("AH", 0.0, 0.7)]), word("o", [("OW", 0.7, 2.0)])])
    assert V.retime(A, "", x=x)["words"][1]["start"] == pytest.approx(0.7, abs=0.05)
    x = np.r_[tone(1.2), hiss(0.12), tone(0.8)]  # the vocal says otherwise: the hiss wins over a home 500 ms away
    A = dict(words=[word("a", [("AH", 0.0, 0.7)]), word("so", [("S", 0.7, 0.8), ("OW", 0.8, 2.1)])])
    assert V.retime(A, "", x=x)["words"][1]["start"] == pytest.approx(1.2, abs=0.03)


def test_consonants_do_not_stretch_over_a_vowel_they_should_leave():
    x = np.r_[tone(0.5), hiss(0.1), tone(1.0)]
    A = dict(words=[word("a", [("AH", 0.0, 0.5)]), word("so", [("S", 0.5, 0.8), ("OW", 0.8, 1.6)])])
    s = V.retime(A, "", x=x)["words"][1]["phones"][0]
    assert s["end"] == pytest.approx(0.6, abs=0.03)  # staying costs: the hiss ends with the sound, not where the aligner ended it


def test_the_demo_holds_across_home_and_under_a_shifted_aligner(monkeypatch):
    A, x = demo()
    for h in (1.5, 2.0):
        monkeypatch.setattr(V, "HOME", h)
        assert misses(V.retime(A, "", x=x)) == []
    for h in (0.5, 1.0, 3.0):
        monkeypatch.setattr(V, "HOME", h)
        assert len(misses(V.retime(A, "", x=x))) <= 6
    monkeypatch.setattr(V, "HOME", 1.5)
    for d in (-0.3, -0.15, 0.15, 0.3):
        assert (
            len([m for m in misses(V.retime(shifted(A, d), "", x=x)) if m[0] < 27]) <= 7
        )  # the last three words, one vowel short of the lyric, hang on the aligner
