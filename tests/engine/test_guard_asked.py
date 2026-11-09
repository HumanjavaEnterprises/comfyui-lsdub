# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sing's travel guard reads the reference Verify reads (0.2.1): the take's travel scaled by the plan's asked_travel (at most 1), so the
guard and Verify call the same mouth a held gape. Synthetic, no models."""

import json, os

import numpy as np

from lsdub_engine.audits import qa, scorecard as SC
from lsdub_engine.core import sing as S


def _guard(monkeypatch, tmp_path, travels, asks, ref="aligner"):
    it, ai = iter(travels), iter(asks)
    monkeypatch.setattr(S, "mouth_open", lambda v, R, n: np.array([0.0, next(it)]))
    monkeypatch.setattr(S, "travel", lambda op: None if op is None else float(op[1]))
    monkeypatch.setattr(S, "asked_of", lambda wj: next(ai))
    calls = []

    def fake_run(take, align, vocal, t0, out, work, **k):
        calls.append(out)
        os.makedirs(work, exist_ok=True)
        open(out, "wb").write(b"aligner")
        return dict(frames=4, truth=0.40, warp_json="alt.json", **{"pass": True})

    monkeypatch.setattr(S, "run", fake_run)
    out = tmp_path / "sung.mp4"
    out.write_bytes(b"retimed")
    info = dict(frames=4, truth=0.6, retime=dict(used=True), warp_json="w.json", **{"pass": True})
    got = S.travel_guard(info, "take.mp4", "a.json", "v.wav", 0.0, str(out), str(tmp_path), {}, dict(paint_args=()), ref=ref)
    return got, out.read_bytes(), calls


def test_a_held_line_the_plan_asked_for_is_not_a_held_gape(monkeypatch, tmp_path):
    """take 0.0902, retimed 0.0673: under 0.75 of the take (held by the take alone), but the plan asked 0.6 of the take's travel and the
    mouth delivers 0.0673 >= 0.75 x 0.6 x 0.0902: Verify passes it, and now the guard keeps it without a second render."""
    got, body, calls = _guard(monkeypatch, tmp_path, (0.0902, 0.0673), (0.6,))
    g = got["retime"]["guard"]
    assert body == b"retimed" and calls == [] and g["held"] is False and g["asked_travel"] == 0.6
    assert g["held"] == SC.held_gape(dict(mouth_range=0.0673, before=dict(mouth_range=0.0902), asked_travel=0.6))


def test_with_no_ask_the_guard_reads_the_take_as_before(monkeypatch, tmp_path):
    got, body, calls = _guard(monkeypatch, tmp_path, (0.1333, 0.0717, 0.1025), (None, None))
    g = got["retime"]["guard"]
    assert g["held"] is True and g["asked_travel"] is None and body == b"aligner" and len(calls) == 1
    assert g["aligner_asked_travel"] is None


def test_an_ask_over_one_does_not_raise_the_reference(monkeypatch, tmp_path):
    got, _, calls = _guard(monkeypatch, tmp_path, (0.108, 0.095), (1.6,))
    assert got["retime"]["guard"]["held"] is False and calls == []


def test_the_aligners_own_ask_decides_its_swap(monkeypatch, tmp_path):
    """the aligner's render is swapped in only when it is not a held gape against its OWN plan's ask."""
    got, body, _ = _guard(monkeypatch, tmp_path, (0.1333, 0.0717, 0.1025), (1.0, 1.0))
    assert body == b"aligner" and got["retime"]["guard"]["aligner_asked_travel"] == 1.0


def test_asked_of_is_verifys_reading(tmp_path):
    wj = tmp_path / "work" / "crop_painted_warp.json"
    wj.parent.mkdir()
    g = np.linspace(0.0, 1.0, 20)
    wj.write_text(json.dumps(dict(frames=[dict(gap=float(a), tgap=float(0.5 * a)) for a in g])))
    assert S.asked_of(str(wj)) == qa.asked_travel({"warp_json": str(wj)})["asked_travel"] == 0.5
    assert S.asked_of(None) is None and S.asked_of(str(tmp_path / "none.json")) is None
