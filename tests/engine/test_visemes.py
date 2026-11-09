# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
import pytest

from lsdub_engine import viseme_track as vt


def _align():
    # "map" sung from 0.10 s to 0.70 s
    return {
        "words": [
            {
                "w": "map",
                "start": 0.10,
                "end": 0.70,
                "phones": [
                    {"p": "M", "start": 0.10, "end": 0.18},
                    {"p": "AE1", "start": 0.18, "end": 0.60},
                    {"p": "P", "start": 0.60, "end": 0.70},
                ],
            }
        ]
    }


def test_track_shape_and_closures():
    fr = vt.build(_align(), 0.0, 25, 25)
    assert len(fr) == 25
    vis = [f["vis"] for f in fr]
    assert "PP" in vis  # M/B/P closures are kept
    assert vis[0] == "sil" and vis[-1] in ("sil", "PP")
    assert all(0 <= f["open"] <= 1 for f in fr)


def test_min_hold():
    fr = vt.build(_align(), 0.0, 25, 25, min_hold=3)
    runs, cur = [], None
    for f in fr:
        if f["vis"] != cur:
            runs.append(0)
            cur = f["vis"]
        runs[-1] += 1
    # every run except the first/last is at least min_hold frames (closures are padded up to it)
    assert all(r >= 3 for r in runs[1:-1])


def test_open_curve_off_leaves_the_class_opening_alone(tmp_path, monkeypatch):
    """--open-curve off: the track's open is the class opening with no envelope under it; the envelope is never read."""
    import json

    A = _align()
    A["words"][0]["phones"][1]["p"] = "AE"  # as the aligner writes it (stress digits stripped)
    al = tmp_path / "a.json"
    al.write_text(json.dumps(A))
    monkeypatch.setattr(vt, "envelope_open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the envelope was read")))
    out = tmp_path / "t.json"
    vt.main(
        [
            "--align",
            str(al),
            "--t0",
            "0",
            "--frames",
            "25",
            "--fps",
            "25",
            "--audio",
            "v.wav",
            "--open-curve",
            "off",
            "--out",
            str(out),
            "--no-words",
        ]
    )
    fr = json.loads(out.read_text())["frames"]
    aa = [f for f in fr if f["vis"] == "aa"]
    assert aa and all(
        f["open"]
        == pytest.approx(0.5 * vt.OPEN["aa"] * (1 - f["mix"]) + 0.5 * vt.OPEN[f["vis2"]] * f["mix"] + 0.5 * vt.OPEN["aa"], abs=1e-3)
        for f in aa
    )


def test_word_close_vocal_presses_only_where_the_level_drops(tmp_path, monkeypatch):
    """--word-close-vocal: the between-word sil press applies across a boundary where the stem falls >= 6 dB, not across a legato one."""
    import json

    import numpy as np

    A = {
        "words": [
            {
                "w": "la",
                "start": 0.1,
                "end": 0.5,
                "phones": [{"p": "L", "start": 0.1, "end": 0.14}, {"p": "AA", "start": 0.14, "end": 0.5}],
            },
            {
                "w": "la",
                "start": 0.6,
                "end": 1.0,
                "phones": [{"p": "L", "start": 0.6, "end": 0.64}, {"p": "AA", "start": 0.64, "end": 1.0}],
            },
            {
                "w": "la",
                "start": 1.1,
                "end": 1.5,
                "phones": [{"p": "L", "start": 1.1, "end": 1.14}, {"p": "AA", "start": 1.14, "end": 1.5}],
            },
        ]
    }
    db = np.full(160, -20.0)
    db[53:58] = -40.0  # a 50 ms trough at the first boundary (0.55 s); the second (1.05 s) stays level
    monkeypatch.setattr(vt, "level_db_10ms", lambda audio, t0, dur: db)
    monkeypatch.setattr(vt, "envelope_open", lambda *a, **k: None)
    al = tmp_path / "a.json"
    al.write_text(json.dumps(A))
    out = tmp_path / "t.json"
    base = [
        "--align",
        str(al),
        "--t0",
        "0",
        "--frames",
        "80",
        "--fps",
        "50",
        "--audio",
        "v.wav",
        "--word-close",
        "0.06",
        "--out",
        str(out),
        "--no-words",
    ]
    vt.main(base)
    every = [f["vis"] for f in json.loads(out.read_text())["frames"]]
    vt.main([*base, "--word-close-vocal"])
    vocal = [f["vis"] for f in json.loads(out.read_text())["frames"]]
    assert every[27] == "sil" and every[52] == "sil"  # the plain press at both boundaries (0.54 s, 1.04 s)
    assert vocal[27] == "sil" and vocal[52] == "aa"  # with the vocal read: only where it dropped
    assert vt.boundary_drops(db, [0.55, 1.05], 0.0) == [pytest.approx(20.0), pytest.approx(0.0)]
