# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""ctc_align: synthetic posteriors with known phone frames give exact boundaries; every boundary moves CORR_S earlier; the nucleus
anchor moves a word's first vowel only within ANCHOR_S and takes the leading consonants with it; a word with no nucleus near keeps
its CTC time; an empty window returns no words."""

import numpy as np
import pytest

from lsdub_engine.core import ctc_align as CA

V, BLANK = 8, 0
F = CA.FRAME


def logp_of(path, V=V):
    """(T, V) log-posteriors: each row near-certain on the id the path names (BLANK rows between tokens)."""
    lp = np.full((len(path), V), np.log(0.01 / (V - 1)))
    for t, k in enumerate(path):
        lp[t, k] = np.log(0.99)
    return lp


def path(spans, T):
    """token ids over frames from [(start row, end row, id)], BLANK elsewhere."""
    p = [BLANK] * T
    for a, b, k in spans:
        for t in range(a, b):
            p[t] = k
    return p


WORDS = [("ma", [("M", 1), ("AA", 2)]), ("pa", [("P", 3), ("AA", 2)]), ("tee", [("T", 4), ("IY", 5)])]
SPANS = [(5, 8, 1), (8, 20, 2), (24, 26, 3), (26, 40, 2), (45, 47, 4), (47, 60, 5)]


def test_known_frames_give_exact_boundaries_minus_the_correction():
    out = CA.align(logp_of(path(SPANS, 70)), WORDS, BLANK, corr=0.0)
    got = [(p["p"], p["start"], p["end"]) for w in out for p in w["phones"]]
    assert got == [(p, round(a * F, 3), round(b * F, 3)) for (a, b, k), p in zip(SPANS, ["M", "AA", "P", "AA", "T", "IY"])]
    gappy = [(5, 7, 1), (9, 12, 2), (24, 26, 3), (26, 40, 2), (45, 47, 4), (47, 60, 5)]  # blanks between ma's M and AA
    out = CA.align(logp_of(path(gappy, 70)), WORDS, BLANK, corr=0.0)
    assert out[0]["phones"][0]["end"] == out[0]["phones"][1]["start"] == round(9 * F, 3)  # the phones of a word abut
    assert out[0]["end"] == round(12 * F, 3) and out[1]["start"] == round(24 * F, 3)  # the gap between words stays
    assert [w["align"] for w in out] == ["ctc"] * 3 and out[0]["start"] == round(5 * F, 3) and out[2]["end"] == round(60 * F, 3)
    late = CA.align(logp_of(path(SPANS, 70)), WORDS, BLANK)  # the default correction: every boundary CORR_S earlier, floored at 0
    assert all(abs((a["start"] - b["start"]) - CA.CORR_S) < 1e-9 for a, b in zip(out, late))
    assert late[0]["phones"][0]["start"] == round(5 * F - CA.CORR_S, 3)


def test_a_repeated_token_needs_its_blank_and_t0_shifts_the_clock():
    words = [("aa", [("AA", 2), ("AA", 2)])]
    lp = logp_of(path([(2, 6, 2), (8, 12, 2)], 16))
    out = CA.align(lp, words, BLANK, t0=10.0, corr=0.0)
    assert [(p["start"], p["end"]) for p in out[0]["phones"]] == [(10.0 + 2 * F, 10.0 + 8 * F), (10.0 + 8 * F, 10.0 + 12 * F)]  # abutting


def test_the_anchor_moves_a_first_vowel_within_its_reach_and_the_consonants_with_it():
    out = CA.align(logp_of(path(SPANS, 70)), WORDS, BLANK, corr=0.0)
    v0 = out[1]["phones"][1]["start"]  # pa's AA at row 26
    near = CA.align(logp_of(path(SPANS, 70)), WORDS, BLANK, corr=0.0, nuclei=[v0 - 0.06])
    assert near[1]["phones"][1]["start"] == pytest.approx(v0 - 0.06) and near[1]["anchor_s"] == pytest.approx(-0.06)
    assert near[1]["phones"][0]["start"] == pytest.approx(out[1]["phones"][0]["start"] - 0.06)  # the P moved with its vowel
    assert near[1]["start"] == near[1]["phones"][0]["start"] and near[1]["end"] == out[1]["end"]  # the word's end did not move
    assert near[0] == out[0] and near[2] == out[2]  # no nucleus near the other words: their CTC times
    far = CA.align(logp_of(path(SPANS, 70)), WORDS, BLANK, corr=0.0, nuclei=[v0 - 0.2])
    assert far == out  # out of reach: nothing moves


def test_the_anchor_never_moves_a_word_before_the_one_before_it_ends():
    out = CA.align(logp_of(path(SPANS, 70)), WORDS, BLANK, corr=0.0)
    v0 = out[1]["phones"][1]["start"]
    tight = CA.align(logp_of(path(SPANS, 70)), WORDS, BLANK, corr=0.0, nuclei=[v0 - 0.08])
    assert tight[1]["start"] >= out[0]["end"] - 1e-9


def test_an_empty_window_or_no_frames_returns_no_words():
    assert CA.align(logp_of(path(SPANS, 70)), [], BLANK) == []
    assert CA.align(np.zeros((1, V)), WORDS, BLANK) == []
    assert CA.align(logp_of(path(SPANS, 70)), [("x", [])], BLANK) == []
    assert CA.viterbi(np.zeros((5, V)), [], BLANK) == []
