# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""phone_net, the pure parts: the nucleus gate on a hand-made label sequence (a vowel after a consonant or silence, a vowel after a vowel
only over a level dip, sung frames only, MIN_GAP apart, CORR_S earlier); flux onsets on a click train land on the clicks; the ARPAbet to
vocabulary map; vocal events merge the two detectors."""

import numpy as np

from lsdub_engine.core import phone_net as PN

SR = PN.SR


class Net:
    """a recogniser's vocabulary with no model: labels, blank and the special rows."""

    labels = ["<pad>", "<s>", "</s>", "<unk>", "m", "ɑ", "ɪ", "t", "p", "ə", "a", "ʌ", "b", "ɡ", "g", "ɹ", "r"]
    blank = 0
    special = {0, 1, 2, 3}
    index = {t: i for i, t in enumerate(labels)}


def logp_of(labels_per_frame, net=Net):
    lp = np.full((len(labels_per_frame), len(net.labels)), np.log(0.001))
    for t, lab in enumerate(labels_per_frame):
        lp[t, net.index[lab]] = np.log(0.9)
    return lp


def tone(spans, dur, f0=220.0, level=0.3):
    """a voice on over [a, b) spans at level, a faint floor elsewhere."""
    t = np.arange(int(dur * SR)) / SR
    x = 0.001 * np.random.default_rng(0).standard_normal(len(t))
    for a, b, g in spans:
        m = (t >= a) & (t < b)
        x[m] += level * g * np.sin(2 * np.pi * f0 * t[m])
    return x


def test_runs_merge_frames_of_one_label_and_skip_special_rows():
    lp = logp_of(["<pad>", "m", "m", "<pad>", "ɑ", "ɑ", "ɑ", "<pad>", "<pad>", "ɑ"])
    assert PN.runs(lp, Net) == [("m", 0.02, 0.06), ("ɑ", 0.08, 0.14), ("ɑ", 0.18, 0.2)]


def test_a_vowel_after_a_consonant_is_a_nucleus_and_a_vowel_after_a_vowel_is_not_without_a_dip():
    # m ɑ (0.1-0.3) ɪ (0.3-0.5) t ʌ (0.52-0.7) over one steady note: ɑ and ʌ are nuclei, ɪ (vowel after vowel, no dip) is not
    labs = ["<pad>"] * 4 + ["m"] + ["ɑ"] * 10 + ["ɪ"] * 10 + ["t"] + ["ʌ"] * 9 + ["<pad>"] * 5
    x = tone([(0.0, 0.8, 1.0)], 0.8)
    assert PN.nuclei(logp_of(labs), x, Net) == [round(5 * 0.02 - PN.CORR_S, 3), round(26 * 0.02 - PN.CORR_S, 3)]
    assert PN.nuclei(logp_of(labs), x, Net, corr=0.0) == [0.1, 0.52]


def test_a_level_dip_between_two_vowels_makes_the_second_a_nucleus():
    labs = ["<pad>"] * 4 + ["ɑ"] * 10 + ["ɪ"] * 10 + ["<pad>"] * 6
    # the note dips to a quarter (12 dB) for 60 ms around the vowel change at 0.28 s
    x = tone([(0.0, 0.25, 1.0), (0.25, 0.31, 0.25), (0.31, 0.6, 1.0)], 0.6)
    assert PN.nuclei(logp_of(labs), x, Net, corr=0.0) == [0.08, 0.28]


def test_a_vowel_on_silence_is_not_a_nucleus_and_two_within_min_gap_are_one():
    labs = ["<pad>"] * 2 + ["ɑ"] * 2 + ["t"] + ["ɪ"] * 6 + ["<pad>"] * 20
    x = tone([(0.0, 0.24, 1.0)], 0.62)  # the voice is on only until 0.24 s
    assert PN.nuclei(logp_of(labs), x, Net, corr=0.0) == [0.04]  # ɪ at 0.10 is within MIN_GAP of ɑ at 0.04
    quiet = ["<pad>"] * 20 + ["ɑ"] * 4 + ["<pad>"] * 7
    assert PN.nuclei(logp_of(quiet), x, Net) == []  # the vowel sits on silence


def test_flux_onsets_land_on_the_clicks_of_a_click_train():
    x = 0.0005 * np.random.default_rng(1).standard_normal(SR * 3)
    clicks = [0.5, 1.0, 1.6, 2.3]
    for c in clicks:
        k = int(c * SR)
        x[k : k + 400] += 0.5 * np.sin(2 * np.pi * 1000 * np.arange(400) / SR) * np.hanning(400)
    got = PN.flux_onsets(x)
    assert len(got) == len(clicks) and all(abs(g - c) <= 0.03 for g, c in zip(got, clicks)), got


def test_vocal_events_merge_nuclei_and_flux_min_gap_apart_and_fall_back_to_flux():
    x = tone([(0.0, 1.0, 1.0)], 1.0)
    labs = ["<pad>"] * 5 + ["ɑ"] * 10 + ["t"] + ["ʌ"] * 10 + ["<pad>"] * 24
    nuc = PN.nuclei(logp_of(labs), x, Net)
    fl = PN.flux_onsets(x)
    ev = PN.vocal_events(x, logp_of(labs), Net)
    assert set(nuc) <= set(ev) and all(b - a >= PN.MIN_GAP for a, b in zip(ev, ev[1:]))
    assert PN.vocal_events(x) == fl
    assert PN.merge([0.5, 0.52, 0.7, 0.75, 0.9]) == [0.5, 0.7, 0.9]


def test_the_arpabet_map_reaches_the_vocabulary_by_ipa_an_alternative_or_a_first_letter():
    assert PN.arpa_token(Net, "M") == Net.index["m"] and PN.arpa_token(Net, "AA1") == Net.index["ɑ"]
    assert PN.arpa_token(Net, "AH") == Net.index["ʌ"] and PN.arpa_token(Net, "AY") == Net.index["a"]  # first letter of aɪ
    assert PN.arpa_token(Net, "G") == Net.index["ɡ"] and PN.arpa_token(Net, "R") == Net.index["ɹ"]
    assert PN.arpa_token(Net, "ZH") == Net.index["<unk>"]


def test_sung_mask_and_level_follow_the_voice():
    x = tone([(0.2, 0.6, 1.0)], 1.0)
    s = PN.sung(x)
    assert not s[5] and s[30] and not s[90] and len(s) == len(PN.level_db(x))
