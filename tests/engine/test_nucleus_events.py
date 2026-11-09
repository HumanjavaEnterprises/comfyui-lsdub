# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""nucleus_events: hand-made posteriors and flux clicks give the event list; nuclei and flux merge at MIN_GAP; the aligned vowel wins
over the recogniser's label when the alignment was read off the vocal; a rest of REST_S is a rest and a shorter gap is not; closures
come from the alignment's P/B/M; the frames' `open` is byte-identical in and out; every frame carries the event it sits in."""

import json

import numpy as np

from lsdub_engine.core import nucleus_events as NE, phone_net as PN

SR = PN.SR


class Net:
    labels = ["<pad>", "<unk>", "m", "ɑ", "ɪ", "t", "p", "a"]
    blank, special = 0, {0, 1}
    index = {t: i for i, t in enumerate(labels)}


def logp_of(labs):
    lp = np.full((len(labs), len(Net.labels)), np.log(0.001))
    for t, lab in enumerate(labs):
        lp[t, Net.index[lab]] = np.log(0.9)
    return lp


def voice(spans, dur, f0=200.0):
    t = np.arange(int(dur * SR)) / SR
    x = 0.001 * np.random.default_rng(0).standard_normal(len(t))
    for a, b, g in spans:
        m = (t >= a) & (t < b)
        x[m] += 0.3 * g * np.sin(2 * np.pi * f0 * t[m])
    return x


def frames(n, fps, t0=0.0):
    return [
        dict(t=round(t0 + k / fps, 3), vis="aa", vis2="sil", mix=0.0, open=round(0.3 + 0.01 * k, 3), word="w", phone="AA") for k in range(n)
    ]


# m ɑ (0.10 s) t ɪ (0.40 s) over one note to 0.7 s, silence, p a (1.0 s) to 1.3 s
LABS = ["<pad>"] * 5 + ["m"] + ["ɑ"] * 14 + ["t"] + ["ɪ"] * 14 + ["<pad>"] * 15 + ["p"] + ["a"] * 14 + ["<pad>"] * 25
X = voice([(0.08, 0.7, 1.0), (0.98, 1.3, 0.5)], 2.0)
WORDS = [
    dict(w="matti", start=0.08, end=0.7, phones=[dict(p="M", start=0.08, end=0.1), dict(p="AE", start=0.1, end=0.4), dict(p="T", start=0.4, end=0.42), dict(p="IY", start=0.42, end=0.7)]),
    dict(w="pa", start=0.98, end=1.3, phones=[dict(p="P", start=0.98, end=1.0), dict(p="AA", start=1.0, end=1.3)]),
]  # fmt: skip


def test_the_event_list_has_the_nuclei_the_closures_and_the_rest():
    ev = NE.events(X, WORDS, 2.0, logp_of(LABS), Net, vocal_read=True)
    kinds = [(e["kind"], e["t_on"]) for e in ev]
    nuc = [t for k, t in kinds if k == "nucleus"]
    assert any(abs(t - 0.1) <= 0.03 for t in nuc) and any(abs(t - 1.0) <= 0.03 for t in nuc)  # ɑ at 0.10 and a at 1.00 (CORR_S earlier)
    assert all(b - a >= PN.MIN_GAP - 1e-9 for a, b in zip(nuc, nuc[1:]))  # merged MIN_GAP apart
    assert [(t) for k, t in kinds if k == "closure"] == [0.08, 0.98]  # the alignment's M and P onsets
    rest = [e for e in ev if e["kind"] == "rest"]
    assert len(rest) >= 1 and any(abs(r["t_on"] - 0.7) <= 0.05 and r["t_off"] >= 0.95 for r in rest)  # 0.7-0.98 is a rest (>= REST_S)
    assert not any(abs(t - 0.694) <= 0.01 or abs(t - 1.294) <= 0.01 for t in nuc)  # a note's release is not an onset
    assert all(e["bin"] in (0, 1, 2) for e in ev if e["kind"] == "nucleus") and all(
        e["db"] is not None for e in ev if e["kind"] == "nucleus"
    )


def test_the_aligned_vowel_wins_over_the_label_when_the_alignment_was_read_off_the_vocal():
    ev = NE.events(X, WORDS, 2.0, logp_of(LABS), Net, vocal_read=True)
    first = [e for e in ev if e["kind"] == "nucleus"][0]
    assert first["vowel"] == "AE"  # the lyric says AE over the ɑ label
    ev2 = NE.events(X, WORDS, 2.0, logp_of(LABS), Net, vocal_read=False)
    first2 = [e for e in ev2 if e["kind"] == "nucleus"][0]
    assert first2["vowel"] == "AA"  # the recogniser's ɑ
    assert (
        NE.vowel_at(WORDS, 1.1) == "AA" and NE.vowel_at(WORDS, 1.5) is None and NE.vowel_at(WORDS, 0.41) == "IY"
    )  # the nearest vowel start within reach


def test_a_rest_needs_rest_s_and_a_shorter_gap_is_not_one():
    m = np.ones(200, bool)
    m[50:65] = False  # 150 ms
    m[100:114] = False  # 140 ms
    assert NE.rests(m, 2.0) == [(0.5, 0.65)]
    assert NE.rests(np.ones(10, bool), 0.1) == []


def test_loudness_bins_read_the_peak_against_the_clips_nuclei():
    db = np.full(300, -30.0)
    db[100:115] = -10.0  # a loud syllable
    db[200:215] = -40.0  # a quiet one
    pk, bins = NE.loudness(db, [0.2, 1.0, 2.0])
    assert bins == [1, 2, 0] and pk[1] > pk[0] > pk[2]
    assert NE.loudness(db, []) == ([], [])


def test_annotate_leaves_open_byte_identical_and_tags_every_frame():
    fr = frames(50, 50.0)
    before = json.dumps([f["open"] for f in fr])
    ev = NE.events(X, WORDS, 1.0, logp_of(LABS), Net, vocal_read=True)
    out, info = NE.annotate(fr, ev, 50.0)
    assert json.dumps([f["open"] for f in out]) == before and [f["vis"] for f in out] == [f["vis"] for f in fr]
    assert all("ev" in f for f in out) and "ev" not in fr[0]
    assert out[0]["ev"]["kind"] == "rest" and out[0]["ev"]["k"] == -1  # before the first event
    assert out[4]["ev"]["kind"] == "closure" and out[4]["ev"]["t_on"] == 0.08  # 0.08 s: the M closure
    nuc_frames = [f for f in out if f["ev"]["kind"] == "nucleus"]
    assert nuc_frames and nuc_frames[0]["ev"]["vowel"] == "AE" and nuc_frames[0]["ev"]["t_on"] <= nuc_frames[0]["t"] == 0.1  # after the M
    assert out[40]["ev"]["kind"] == "rest" and out[40]["ev"]["k"] >= 0  # 0.8 s: inside the rest
    assert info["nuclei"] >= 2 and info["closures"] == 2 and info["rests"] >= 1 and info["mode"] == "vocal"


def test_annotate_honours_a_track_t0():
    fr = frames(10, 50.0, t0=12.0)
    ev = [dict(kind="nucleus", t_on=0.1, vowel="AA", db=0.0, bin=1)]
    out, _ = NE.annotate(fr, ev, 50.0, t0=12.0)
    assert [f["ev"]["kind"] for f in out] == ["rest"] * 5 + ["nucleus"] * 5


def test_without_the_model_the_events_are_flux_onsets_with_a_vowel_from_the_alignment():
    ev = NE.events(X, WORDS, 2.0, None, None, vocal_read=True)
    nuc = [e for e in ev if e["kind"] == "nucleus"]
    assert nuc and all(e["vowel"] in PN.ARPA_VOWELS for e in nuc)


def test_events_name_the_detector_that_found_each_nucleus():
    """`src` is written by events(): a nucleus the recogniser and the flux both found reads nucflux, one only the flux found reads flux."""
    ev = NE.events(X, WORDS, 2.0, logp_of(LABS), Net, vocal_read=True)
    nuc = [e for e in ev if e["kind"] == "nucleus"]
    assert nuc and all(e.get("src") in ("nuc", "flux", "nucflux") for e in nuc), nuc
    assert any("nuc" in e["src"] for e in nuc)
    flux_only = NE.events(X, WORDS, 2.0, None, None, vocal_read=True)  # no recogniser: every nucleus is the flux's
    assert all(e["src"] == "flux" for e in flux_only if e["kind"] == "nucleus")
    out, _ = NE.annotate(frames(50, 50.0), ev, 50.0)
    assert all(f["ev"].get("src") in (None, "nuc", "flux", "nucflux") for f in out) and any(f["ev"].get("src") for f in out)
