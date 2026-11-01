# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""QA step 0: masked corr, N/A, the plate sharpness mode, boil at weight 0, the ruler row."""

import json, os

import cv2, numpy as np
import pytest

from lsdub_engine.audits import mouth_ruler as R, qa, scorecard as SC, snap as SN, sync_qa as Q
from lsdub_engine.core import medium as MED, sing, vocalisations as V


def series(n=90, seed=0):
    rng = np.random.default_rng(seed)
    t = np.convolve(rng.normal(size=n + 8), np.ones(5) / 5, "same")[:n]
    return t, t.copy()


def warp(path, skip, m=1, vocal=True):
    d = {"frames": [{"i": i, "vocal_skip": bool(s)} for i, s in enumerate(np.repeat(skip, m))]}
    if vocal:
        d["vocal"] = {"mode": "skip", "spans": []}
    json.dump(d, open(path, "w"))
    return str(path)


def third(n=90):
    return np.array([n // 3 <= i < 2 * n // 3 for i in range(n)])


def test_no_spans_reads_every_frame(tmp_path):
    assert V.scored_frames(None, 10) == (None, 0.0)
    assert V.scored_frames(str(tmp_path / "gone.json"), 10) == (None, 0.0)
    assert V.scored_frames(warp(tmp_path / "a.json", np.zeros(30, bool), vocal=False), 30) == (None, 0.0)  # `mouth` writes no block
    assert V.scored_frames(warp(tmp_path / "b.json", np.zeros(30, bool)), 30) == (None, 0.0)


def test_spans_at_the_internal_rate_map_to_output_frames(tmp_path):
    keep, frac = V.scored_frames(warp(tmp_path / "a.json", third(), m=2), 90)
    assert frac == pytest.approx(1 / 3, abs=0.01) and (keep == ~third()).all()


def test_too_few_kept_frames_score_the_whole_shot(tmp_path):
    keep, frac = V.scored_frames(warp(tmp_path / "a.json", np.ones(30, bool)), 30)
    assert keep is None and frac == 1.0


def test_keep_scores_the_kept_frames():
    t, op = series()
    k = third()
    op[k] = series(seed=9)[0][k]
    assert Q.tracking(op, t, 25.0)[0] < 0.8
    assert Q.tracking(op, t, 25.0, keep=~k)[0] > 0.99


def stub_gate(monkeypatch, tmp_path, op):
    monkeypatch.setattr(MED, "read", lambda v, n=None: ([np.zeros((64, 64, 3), np.uint8)] * len(op), 25.0))
    monkeypatch.setattr(Q, "measured_opening", lambda B, roi, fps: (op, 1.0))
    monkeypatch.setattr(SN, "snap_metrics", lambda *a, **k: dict(open_in_silence=0.1, rise_ms=None, closure_hit=None, closures=0, onsets=0))
    monkeypatch.setattr(sing, "vocal_clip", lambda *a: "v.wav")
    al = tmp_path / "align.json"
    al.write_text('{"words": []}')
    return [str(al), {"per_frame": [{"centre": [32, 32]}], "face_h_px": 40.0}]


def test_gate_is_identical_without_spans(monkeypatch, tmp_path):
    t, op = series()
    op = op + np.random.default_rng(1).normal(0, 0.3, len(op))
    al, Rr = stub_gate(monkeypatch, tmp_path, op)
    plain = sing.gate("v", Rr, t, "v", al, 0.0, 25.0, work=str(tmp_path))
    none = sing.gate("v", Rr, t, "v", al, 0.0, 25.0, work=str(tmp_path), warp_json=warp(tmp_path / "w.json", np.zeros(90, bool)))
    assert plain == none and plain["vocal_frac"] == 0.0 and "corr_all" not in plain


def test_gate_passes_a_skipped_span_that_failed_it(monkeypatch, tmp_path):
    t, op = series()
    k = third()
    op[k] = -t[k] * 3
    al, Rr = stub_gate(monkeypatch, tmp_path, op)
    whole = sing.gate("v", Rr, t, "v", al, 0.0, 25.0, work=str(tmp_path))
    masked = sing.gate("v", Rr, t, "v", al, 0.0, 25.0, work=str(tmp_path), warp_json=warp(tmp_path / "w.json", k))
    assert not whole["pass"] and masked["pass"] and masked["corr_all"] == whole["corr"]
    assert masked["vocal_frac"] == pytest.approx(1 / 3, abs=0.01)


def measure(monkeypatch, tmp_path, op, tg, **s):
    F = [np.zeros((64, 64, 3), np.uint8)] * len(op)
    monkeypatch.setattr(Q, "read_span", lambda v, a, b: F)
    monkeypatch.setattr(Q, "singer_roi", lambda *a: [0, 64, 0, 64])
    monkeypatch.setattr(Q, "measured_opening", lambda B, roi, fps: (op, 1.0))
    monkeypatch.setattr(SN, "envelope", lambda v, n, fps: tg)
    monkeypatch.setattr(SN, "snap_metrics", lambda *a, **k: dict(open_in_silence=0.1))
    monkeypatch.setattr(qa.ff, "run", lambda *a, **k: None)
    shot = dict(shot=0, f0=0, f1=len(op) - 1, start=0.0, **s)
    return qa.measure_shot("v", shot, 25.0, 0.0, "v", np.ones(len(op), bool), None, [], lambda f: [], str(tmp_path), Q, SN)


def test_measure_shot_is_identical_without_spans_and_masks_with_them(monkeypatch, tmp_path):
    t, op = series()
    k = third()
    op[k] = series(seed=9)[0][k]
    plain = measure(monkeypatch, tmp_path, op, t)
    none = measure(monkeypatch, tmp_path, op, t, warp_json=warp(tmp_path / "n.json", np.zeros(90, bool)))
    masked = measure(monkeypatch, tmp_path, op, t, warp_json=warp(tmp_path / "w.json", k))
    assert plain == none and plain["vocal_frac"] == 0.0
    assert masked["corr"] > 0.99 and masked["corr_all"] == plain["corr"] and masked["vocal_frac"] > 0.3


def row(**k):
    return dict(sung=True, motion=1.0, pops=0, corr=0.7, corr_pm1=0.7, sharpness=100.0, **k)


def test_floor_is_the_default_and_plate_is_symmetric():
    assert SC.components(row(), 200.0)["sharpness"] == SC.components(row(), 200.0, "floor")["sharpness"] == 0.5
    lo, hi = SC.components(row(), 200.0, "plate")["sharpness"], SC.components(row(), 50.0, "plate")["sharpness"]
    assert lo == pytest.approx(hi) and lo == pytest.approx(0.0)
    assert SC.components(row(), 100.0, "plate")["sharpness"] == 1.0
    assert SC.components(row(), 50.0)["sharpness"] == 1.0  # the floor cannot see over-sharp


def test_boil_is_present_and_moves_no_total():
    a, b = SC.finalize(row(), 100.0), SC.finalize(row(boil=6.0), 100.0)
    assert "boil" not in a["components"] and b["components"]["boil"] == 0.5 and a["score"] == b["score"] and a["reason"] == b["reason"]


def test_left_as_source_is_na_and_keeps_the_film_median(monkeypatch, tmp_path):
    p = tmp_path / "shots.json"
    base = dict(start=0.0, end=2.0, verdict="DUB", f0=0, f1=49)
    p.write_text(json.dumps({"fps": 25.0, "shots": [{**base, "shot": 1}, {**base, "shot": 2, "used": "source", "left_reason": "gate"}]}))
    m = {1: 200.0, 2: 100.0}
    monkeypatch.setattr(
        qa,
        "measure_shot",
        lambda v, s, *a: dict(
            voiced=True,
            corr=0.8,
            corr_pm1=0.8,
            mouth_visible=1.0,
            owner_px=400.0,
            mouth_range=0.3,
            motion=1.0,
            pops=0,
            sharpness=m[s["shot"]],
        ),
    )
    monkeypatch.setattr(qa, "vad", lambda *a: (np.ones(50, bool), None))
    monkeypatch.setattr(qa.cv2, "VideoCapture", lambda v: type("C", (), {"get": lambda s, k: 50, "release": lambda s: None})())
    from lsdub_engine.core import cast as C

    monkeypatch.setattr(C, "Faces", lambda: None)
    rows = qa.verify("v", "a.wav", str(p), work=str(tmp_path))[1]
    assert rows[1]["band"] == "N/A" and rows[1]["score"] is None and rows[1]["reason"] == "left as source: gate"
    assert rows[0]["components"]["sharpness"] == 1.0  # median 150 holds the left shot's reading
    p.write_text(p.read_text().replace('"used": "source"', '"x": 1'))
    again = qa.verify("v", "a.wav", str(p), work=str(tmp_path))[1]
    assert again[0]["score"] == rows[0]["score"] and again[0]["band"] == rows[0]["band"]
    out = str(tmp_path / "v")
    monkeypatch.setattr(qa, "sheet", lambda *a: None)
    qa.write_verify({"video": "v"}, rows, out)
    assert "**N/A**" in open(out + ".md").read()


def test_mouth_identity_reads_the_lower_face_and_scores_nothing():
    assert qa.lower_face([0, 0, 10, 100]) == pytest.approx([0, 55.0, 10, 100])
    r = qa.score_row(
        dict(voiced=True, verdict="DUB", corr=0.8, corr_pm1=0.8, mouth_visible=1.0, owner_px=400.0, mouth_range=0.3, motion=1.0, pops=0),
        None,
        SC,
    )
    q = qa.score_row(
        dict(
            voiced=True,
            verdict="DUB",
            corr=0.8,
            corr_pm1=0.8,
            mouth_visible=1.0,
            owner_px=400.0,
            mouth_range=0.3,
            motion=1.0,
            pops=0,
            mouth_identity_cos=0.1,
        ),
        None,
        SC,
    )
    assert (r["band"], r["score"], r["reason"]) == (q["band"], q["score"], q["reason"])


def ellipse(cav):
    p = np.zeros((478, 2), np.float32)
    for L, a, b in ((R.LIPO, 60, 43), (R.LIPI, 52, 25)):
        t = np.linspace(np.pi, -np.pi, len(L), endpoint=False)
        for i, k in enumerate(L):
            p[k] = (300 + a * np.cos(t[i]), 200 - b * np.sin(t[i]))
    p[61], p[291], p[13], p[14] = (240, 200), (360, 200), (300, 175), (300, 225)
    p[234], p[454] = (60, 180), (540, 180)
    f = np.full((400, 600, 3), (120, 150, 200), np.uint8)
    cv2.fillPoly(f, [np.int32(p[R.LIPO])], (70, 60, 170))
    cv2.fillPoly(f, [np.int32(p[R.LIPI])], cav)
    return f, p


def test_shot_row_on_the_synthetic_ellipse(monkeypatch, tmp_path):
    a, p = ellipse((150, 130, 225))
    b, _ = ellipse((3, 3, 3))
    for name, f in (("t.mp4", a), ("r.mp4", b)):
        w = cv2.VideoWriter(str(tmp_path / name), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (600, 400))
        for _ in range(5):
            w.write(f)
        w.release()
    monkeypatch.setattr(R, "cached_track", lambda F, key, work=None: np.array([p] * len(F)))
    r = R.shot_row(str(tmp_path / "t.mp4"), str(tmp_path / "r.mp4"), [("take", str(tmp_path / "t.mp4"), 0, None, None)])
    assert r["pink_frac"][0] > r["pink_frac"][1] and r["cav_L50"][1] < 10 and "cavity void" in r["notes"]
    assert r["stages"][0]["stage"] == "take" and "→" in R.rows_md([{"shot": 1, **r}])
    assert "no frames" in R.rows_md([{"shot": 2, **R.shot_row(str(tmp_path / "gone.mp4"), str(tmp_path / "gone.mp4"))}])


def test_ruler_uses_only_stats_it_is_handed(monkeypatch):
    f, p = ellipse((150, 130, 225))
    monkeypatch.setattr(R, "frame_stats", lambda fr, q: dict(w=120.0, ap=50.0, pink_frac=0.25))
    s = R.stats([f] * 3, np.array([p] * 3))
    assert s["take"]["pink_frac"] == 0.25


@pytest.mark.skipif(not os.environ.get("LIPS_FACE_MODEL"), reason="needs the face model")
def test_boil_of_a_faceless_clip_is_none(tmp_path):
    w = cv2.VideoWriter(str(tmp_path / "c.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (128, 128))
    for _ in range(6):
        w.write(np.zeros((128, 128, 3), np.uint8))
    w.release()
    assert SC.boil_of(str(tmp_path / "c.mp4"), 0, 6) is None


def test_kept_tracking_with_too_few_pairs_reads_zero():
    x = np.arange(10.0)
    assert Q.tracking(x, x, 25.0, keep=np.zeros(10, bool)) == (0.0, 0.0, 0.0)


def test_scorecard_plate_mode_needs_a_plate():
    with pytest.raises(SystemExit):
        SC.main(["--video", "v", "--cuts", "c", "--vocal", "w", "--out", "o", "--sharpness-mode", "plate"])


GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "fixtures", "demo_crop.npz")


@pytest.mark.skipif(not os.environ.get("LIPS_FACE_MODEL") or not os.path.exists(GOLDEN), reason="needs the face model and the golden crop")
def test_shot_row_tracks_the_golden_crop(tmp_path):
    F = np.load(GOLDEN)["frames"]
    p = str(tmp_path / "take.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, F.shape[2:0:-1])
    for f in F:
        w.write(f)
    w.release()
    r = R.shot_row(p, p, [("take", p, 0, None, None)], str(tmp_path / "work"))
    assert 0 < r["pink_frac"][0] == r["pink_frac"][1] and r["teeth_L"][0] == r["teeth_L"][1]
    assert r["notes"] == [] and r["stages"][0]["tracked"] == len(F)
