# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""word times read off the vocal: a held vowel, the consonants where the vocal has them, the lyric untouched."""

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


@pytest.mark.parametrize("p,c", [("AH0", "V"), ("S", "H"), ("T", "S"), ("M", "N"), ("Z", "W"), ("B", "N"), ("HH", "H")])
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
