# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lyric_align on a voice with gaps (CPU, no models): the fallback spread closes the mouth over an unvoiced gap, a lyric
match of 0 still places the lyrics on the voice, uncovered singing gets the unheard line or vocalisations, and the
Align node warns / raises on a low lyric match; LSDub Dub passes its vocalisations mode to the warp."""

import numpy as np
import pytest

from lsdub_engine.core import lyric_align as LA


def env_of(*parts):
    """10 ms dB frames: ('v', s) voiced, ('u', s) unvoiced."""
    return np.concatenate([np.full(int(round(s * 100)), -10.0 if k == "v" else -90.0) for k, s in parts])


def covered(phones, t):
    return any(p["start"] <= t < p["end"] for p in phones)


def test_spread_closes_the_mouth_over_a_gap():
    env = env_of(("v", 0.5), ("u", 0.2), ("v", 0.5))  # a 200 ms gap at 0.5-0.7 s
    w = {"w": "away", "start": 0.0, "end": 1.2}
    ph = LA.spread_voiced(w, ["AH", "W", "EY"], env, LA.voiced_thr(env), 0.0)
    assert not any(covered(ph, t) for t in (0.52, 0.6, 0.68))  # no phone in the gap: the mouth closes
    assert covered(ph, 0.1) and covered(ph, 1.1)
    assert all(p["end"] <= 0.5 + 1e-6 or p["start"] >= 0.7 - 1e-6 for p in ph)
    old = LA.spread(w, ["AH", "W", "EY"])  # the old spread held a phone across it
    assert covered(old, 0.6)


def test_a_short_dip_does_not_close_the_mouth():
    env = env_of(("v", 0.5), ("u", 0.05), ("v", 0.5))  # 50 ms < GAP_CLOSE: bridged
    ph = LA.spread_voiced({"w": "oh", "start": 0.0, "end": 1.05}, ["OW"], env, LA.voiced_thr(env), 0.0)
    assert len(ph) == 1 and ph[0]["start"] == 0.0 and ph[0]["end"] == pytest.approx(1.05)


def test_ratio_zero_still_places_the_lyrics_on_the_voice():
    env = env_of(("u", 0.5), ("v", 1.0), ("u", 0.5), ("v", 1.0), ("u", 0.3))
    lines = [["fly", "me"], ["to", "the", "moon"]]
    W = LA.place_lyrics(lines, env, 10.0, LA.voiced_thr(env))
    assert [w["w"] for w in W] == ["fly", "me", "to", "the", "moon"]
    assert W[0]["start"] == pytest.approx(10.5) and W[-1]["end"] == pytest.approx(13.0)
    assert all(w["align"] == "voiced-place" for w in W)
    assert all(b["start"] >= a["end"] - 1e-6 for a, b in zip(W, W[1:]))  # in order, no overlap


def test_unheard_line_and_scat_fill_the_uncovered_singing():
    env = env_of(("v", 1.0), ("v", 1.0), ("u", 0.3), ("v", 1.0))
    lines = [["hello", "you"], ["why", "try"]]
    final = [{"w": "hello", "start": 0.0, "end": 0.5, "phones": []}, {"w": "you", "start": 0.5, "end": 1.0, "phones": []}]
    added = LA.fill_voiced(final, lines, env, LA.voiced_thr(env), 0.0, 3.3)
    words = [(w["w"], w["src"]) for w in added]
    assert ("why", "lyric") in words and ("try", "lyric") in words
    assert any(w == "da" and s == "vocalisation" for w, s in words)  # the other stretch: a vocalisation
    assert all(w["align"] == "voiced-fill" and w["phones"] for w in added)
    assert all(b["start"] >= a["start"] for a, b in zip(final, final[1:]))


def test_letter_phones_keep_the_closures():
    assert LA.letter_phones("pleiadians")[0] == "P" and "N" in LA.letter_phones("pleiadians")
    assert LA.letter_phones("zzz") == ["Z"]


def test_align_node_warns_and_raises_on_a_low_match(pack):
    A = pack.nodes_audio
    assert A.check_match(0.9, True) == "" and A.check_match(0.0, False) == ""
    assert "placed over the voiced" in A.check_match(0.35, True)
    with pytest.raises(RuntimeError, match="match what is sung"):
        A.check_match(0.1, True)


def test_lyric_prompt(pack, tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("[Chorus]\nFly me\n\nTo the moon!\n")
    assert pack.nodes_audio.lyric_prompt(str(p)) == "Fly me. To the moon!"


def test_dub_passes_the_vocalisations_mode(pack, tmp_path, monkeypatch):
    M, T, P = pack.nodes_mouth, pack.lsdub_types, pack.project
    seen = {}
    import importlib

    sing = importlib.import_module(pack.__name__ + ".lsdub_engine.core.sing")
    continuity = importlib.import_module(pack.__name__ + ".lsdub_engine.core.continuity")

    def fake_run(take, align, vocal, t0, out, work, pa, **kw):
        seen["pa"] = list(pa)
        open(out, "w").write("x")
        return {"pass": True, "corr": 0.9}

    monkeypatch.setattr(sing, "run", fake_run)
    monkeypatch.setattr(continuity, "run", lambda take, sung, render, report=None, **k: open(render, "w").write("x"))
    monkeypatch.setattr(M, "extract_shot", lambda src, f0, f1, out: out)
    p = P.Project(root=str(tmp_path), name="cut").ensure()
    s = {"shot": 1, "f0": 0, "f1": 9, "start": 0.0, "verdict": "DUB", "target": "a", "box": [0, 0, 10, 10]}
    st = p.shot(1, "k")
    M.LSDubDub.run_shot(
        p,
        T.Source(path="v.mp4"),
        s,
        T.Song(mix="m.wav"),
        T.Align(align="a.json"),
        T.Kit(dir=str(tmp_path), identity="a"),
        M.DEFAULT_TWEAKS,
        None,
        None,
        st,
        "mouth",
    )
    assert seen["pa"][-2:] == ["--vocalisations", "mouth"]
    sch = M.LSDubDub.define_schema()
    c = next(i for i in sch.inputs if i.id == "vocalisations")
    assert c.kw["options"] == ["skip", "rest", "mouth"] and c.kw["default"] == "skip"
    k = lambda v: M.LSDubDub._shot_key(
        p,
        T.Source(path="v.mp4", hash="h"),
        s,
        T.Song(mix="m"),
        T.Align(align="a", hash="a"),
        T.Kit(dir="d", identity="a", hash="k"),
        M.DEFAULT_TWEAKS,
        None,
        None,
        v,
    )
    assert k("mouth") != k("skip") and k("skip") == k("skip")


def _w(name, idx, a, b):
    return dict(w=name, idx=idx, start=a, end=b, phones=[dict(p="AH", start=a, end=(a + b) / 2), dict(p="T", start=(a + b) / 2, end=b)])


def test_restore_order_puts_a_swapped_pair_back_inside_its_span():
    from lsdub_engine.core.lyric_align import restore_order

    W = [_w("its", 0, 7.88, 8.41), _w("that", 2, 8.41, 8.68), _w("about", 1, 8.68, 9.19), _w("time", 3, 9.19, 9.8)]
    assert restore_order(W) == 2
    assert [w["w"] for w in W] == ["its", "about", "that", "time"]
    assert W[1]["start"] == 8.41 and abs(W[2]["end"] - 9.19) < 1e-9
    assert abs(W[1]["end"] - W[2]["start"]) < 1e-9
    assert abs(W[1]["phones"][-1]["end"] - W[1]["end"]) < 1e-9 and abs(W[2]["phones"][0]["start"] - W[2]["start"]) < 1e-9
    assert abs((W[1]["end"] - W[1]["start"]) / (W[2]["end"] - W[2]["start"]) - 0.51 / 0.27) < 1e-6  # each keeps its own length


def test_restore_order_leaves_words_in_order_untouched():
    from lsdub_engine.core.lyric_align import restore_order
    import copy

    W = [_w("a", 0, 0.0, 0.2), _w("b", 1, 0.25, 0.5), _w("c", 5, 0.5, 0.9)]
    before = copy.deepcopy(W)
    assert restore_order(W) == 0
    assert W == before


def test_restore_order_handles_a_run_of_three():
    from lsdub_engine.core.lyric_align import restore_order

    W = [_w("c", 2, 1.0, 1.2), _w("a", 0, 1.2, 1.4), _w("b", 1, 1.4, 1.6), _w("d", 3, 1.7, 1.9)]
    assert restore_order(W) == 3
    assert [w["w"] for w in W] == ["a", "b", "c", "d"]
    assert all(x["end"] <= y["start"] + 1e-9 for x, y in zip(W, W[1:]))


def test_restore_order_keeps_a_silent_gap_silent():
    from lsdub_engine.core.lyric_align import restore_order

    W = [_w("b", 1, 1.0, 1.4), _w("a", 0, 2.0, 2.2)]  # 0.6 s of silence between them
    assert restore_order(W) == 2
    assert [w["w"] for w in W] == ["a", "b"]
    assert W[1]["start"] - W[0]["end"] == pytest.approx(0.6)
    assert W[0]["start"] == 1.0 and W[1]["end"] == pytest.approx(2.2)


def test_restore_order_keeps_the_silence_where_it_was():
    from lsdub_engine.core.lyric_align import restore_order

    W = [_w("x", 2, 0.0, 1.0), _w("y", 1, 1.5, 1.6)]  # silence in [1.0, 1.5]
    assert restore_order(W) == 2
    assert [w["w"] for w in W] == ["y", "x"]
    assert (W[0]["start"], W[0]["end"]) == (0.0, pytest.approx(1.0)) and (W[1]["start"], W[1]["end"]) == (1.5, pytest.approx(1.6))
    assert not any(w["start"] < 1.5 and w["end"] > 1.0 for w in W)


# -- the ctc aligner ---------------------------------------------------------------------------------------------------------------------
def test_lyric_window_keeps_the_unheard_words_between_the_heard_ones_and_falls_back_to_the_sheet():
    lyr = ["turn", "the", "lights", "up", "here", "we", "go", "every", "heartbeat"]
    heard = [dict(w="Turn", start=0, end=0.2, p=0.9), dict(w="lights", start=0.4, end=0.7, p=0.9), dict(w="go.", start=1.7, end=2.0, p=0.9)]
    W = LA.lyric_window(heard, lyr, 0.9)
    assert [w["w"] for w in W] == ["turn", "the", "lights", "up", "here", "we", "go"]  # from the first to the last heard word
    assert [w["src"] for w in W] == ["lyric", "lyric-unheard", "lyric", "lyric-unheard", "lyric-unheard", "lyric-unheard", "lyric"]
    assert W[0]["heard"] == "Turn" and W[1]["heard"] == "" and W[-1]["p"] == 0.9 and W[1]["p"] is None
    low = LA.lyric_window(heard, lyr, 0.2)  # under LOW_MATCH: the whole sheet, nothing heard
    assert [w["w"] for w in low] == lyr and all(w["heard"] == "" for w in low)
    assert [w["w"] for w in LA.lyric_window([], lyr, None)] == lyr and LA.lyric_window(heard, [], 0.9) == []


def test_aligner_ctc_writes_ctc_words_from_the_recognisers_frames(tmp_path, monkeypatch):
    """--aligner ctc with the phone network monkeypatched: every word aligned 'ctc', boundaries on the frames the posteriors name,
    the file says aligner ctc, speech-to-text timing never read."""
    import json

    import numpy as np

    from lsdub_engine.core import phone_net as PN

    class Net:
        labels = ["<pad>", "<unk>", "m", "ɑ", "p", "a", "ɪ", "t"]
        blank, special = 0, {0, 1}
        index = {t: i for i, t in enumerate(labels)}

        def __init__(self, *a, **k):
            pass

        def release(self):
            pass

    rows = [0] * 10 + [2] * 3 + [3] * 10 + [0] * 5 + [4] * 2 + [5] * 12 + [0] * 108  # "ma" then "pie" (p a), 3 s of frames
    lp = np.full((len(rows), len(Net.labels)), np.log(0.001))
    for t, k in enumerate(rows):
        lp[t, k] = np.log(0.9)
    monkeypatch.setattr(PN, "PhoneNet", Net)
    monkeypatch.setattr(PN, "posteriors", lambda x, net: lp)
    monkeypatch.setattr(PN, "nuclei", lambda logp, x, net, corr=0.02: [])
    monkeypatch.setattr(LA, "cmu_phones", lambda w, extra: {"ma": ["M", "AA1"], "pie": ["P", "AY1"]}[w])
    sr, x = 16000, np.zeros(16000 * 3, np.float32)
    t = np.arange(len(x)) / sr
    on = ((t > 0.2) & (t < 0.46)) | ((t > 0.56) & (t < 0.84))
    x[on] = (0.3 * np.sin(2 * np.pi * 220 * t[on])).astype(np.float32)
    import wave

    wav = tmp_path / "lead.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((x * 32767).astype(np.int16).tobytes())
    (tmp_path / "lyrics.txt").write_text("ma pie\n")
    (tmp_path / "stt.json").write_text(json.dumps([dict(w="ma", start=0.0, end=0.3, p=0.9), dict(w="pie", start=0.3, end=0.9, p=0.9)]))
    out = tmp_path / "align.json"
    LA.main(["--audio", str(wav), "--lyrics", str(tmp_path / "lyrics.txt"), "--t0", "0", "--t1", "3", "--stt-json", str(tmp_path / "stt.json"),
             "--aligner", "ctc", "--out", str(out)])  # fmt: skip
    A = json.loads(out.read_text())
    words = [w for w in A["words"] if w["w"] in ("ma", "pie")]
    assert A["aligner"] == "ctc" and [w["w"] for w in words] == ["ma", "pie"] and all(w["align"] == "ctc" for w in words)
    assert words[0]["phones"][0]["start"] == pytest.approx(10 * 0.02 - 0.02) and words[0]["phones"][1]["p"] == "AA"
    assert words[1]["phones"][0]["start"] == pytest.approx(28 * 0.02 - 0.02) and words[1]["phones"][1]["p"] == "AY"
    assert A["phrases"][0]["method"] == "ctc" and A["phrases"][0]["words"] == 2
    # the voice runs on after "pie" to 0.84 s (its last frames are blank to the recogniser): a held note, the word stretched to it
    # with no cap, and no 'da' fill placed over it
    assert words[1]["end"] >= 0.80 and [w["w"] for w in A["words"]] == ["ma", "pie"]  # no 'da' fill over the held tail
