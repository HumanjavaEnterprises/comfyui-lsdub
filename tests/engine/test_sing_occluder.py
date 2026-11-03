# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sing with an occluder mask: the crop paint gets work/occ.npz as it is, the native paint (auto) gets it mapped back to the take
with its grow in take px, and a mask of any other shape stops the shot before a paint runs. Medium and the gate are stubbed."""

import os

import cv2, numpy as np
import pytest

from lsdub_engine.core import medium as MED, sing as S, steps
from lsdub_engine.core.medium import crop_matrix

H, W, N, MM = 96, 160, 6, 2
CS, Z = 40.0, 2.5
O = int(CS * Z)
CENTRES = [(70.0 + 2 * k, 50.0 + k) for k in range(N)]
F = [np.zeros((H, W, 3), np.uint8)] * N
GATE = {
    "corr": 0.6,
    "corr_all": 0.6,
    "best": 0.6,
    "lag_ms": 0.0,
    "corr_pm1": 0.6,
    "open_in_silence": 0.1,
    "rise_ms": 40,
    "closure_hit": None,
    "pass": True,
}


def crop_masks():
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (70, 60), 8, 255, -1)
    return np.stack([cv2.warpAffine(m, crop_matrix(*CENTRES[j // MM], CS, Z), (O, O)) for j in range(N * MM)])


@pytest.fixture
def run(monkeypatch, tmp_path):
    seen = dict(medium=[], paint=[])

    def call(step, argv, **k):
        if step == "paint":
            seen["paint"].append([str(a) for a in argv])

    def med_run(take, trk, out, pa, *a, **k):
        seen["medium"].append([str(x) for x in pa])
        open(out, "wb").write(b"x")
        return dict(
            per_frame=[{"centre": list(c)} for c in CENTRES], crop_src_px=CS, crop_px=O, zoom=Z, fps_mult=MM, face_h_px=100.0,
            outside_max_preencode=0, not_pasted=[], picked=list(range(0, N * MM, MM)), timings={},
        )  # fmt: skip

    monkeypatch.setattr(steps, "call", call)
    monkeypatch.setattr(MED, "run", med_run)
    monkeypatch.setattr(MED, "read", lambda v, n=0, *a: (F, 25.0))
    monkeypatch.setattr(S, "gate", lambda *a, **k: dict(GATE))
    monkeypatch.setattr(S, "mic_near_lips", lambda *a, **k: (0.3, True))
    monkeypatch.setattr(S, "moved_target", lambda *a, **k: np.zeros(N))
    work = str(tmp_path / "work")
    os.makedirs(work)

    def go(masks=None, auto=True, paint_args=("--occluder-plate", "--occluder-grow", "18")):
        if masks is not None:
            np.savez_compressed(os.path.join(work, "occ.npz"), masks=masks)
        info = S.run("t.mp4", "a.json", "v.wav", 0.0, str(tmp_path / "o.mp4"), work, list(paint_args), box=[0, 0, 9, 9], auto=auto)
        return info, seen

    go.work = work
    return go


def after(argv, flag):
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


def test_both_paints_get_the_mask(run):
    info, seen = run(crop_masks())
    (crop,), (native,) = seen["medium"], seen["paint"]
    occ, nat = os.path.join(run.work, "occ.npz"), os.path.join(run.work, "occ_native.npz")
    assert after(crop, "--occluder-mask") == occ and after(native, "--occluder-mask") == nat
    with np.load(occ) as a, np.load(nat) as b:
        assert a["masks"].shape == (N * MM, O, O) and b["masks"].shape == (N, H, W)
    g_n = str(round(18 / Z))
    assert after(native, "--occluder-grow") == g_n
    assert native.index("--occluder-grow") < len(native) - 1 - native[::-1].index("--occluder-grow")  # the user's comes first
    assert info["occluder"] == {"space": "crop", "frames": N * MM, "native": {"mask": nat, "grow": int(g_n)}}


def test_take_shaped_mask_stops_before_medium(run):
    with pytest.raises(ValueError) as e:
        run(np.zeros((N, H, W), np.uint8))
    assert str((N, H, W)) in str(e.value) and f"{N * MM} square frames" in str(e.value) and "crop.mp4" in str(e.value)


def test_take_shaped_mask_runs_no_paint(run, monkeypatch):
    called = []
    monkeypatch.setattr(MED, "run", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError):
        run(np.zeros((N, H, W), np.uint8))
    assert called == []


def test_side_not_the_crop_stops_before_the_gate(run, monkeypatch):
    gated = []
    monkeypatch.setattr(S, "gate", lambda *a, **k: gated.append(1) or dict(GATE))
    with pytest.raises(ValueError):
        run(np.zeros((N * MM, O + 4, O + 4), np.uint8))
    assert gated == []


def test_no_mask_no_flags(run):
    info, seen = run(None)
    (crop,), (native,) = seen["medium"], seen["paint"]
    assert "--occluder-mask" not in crop and "--occluder-mask" not in native
    assert native.count("--occluder-grow") == 1  # the user's only
    assert "occluder" not in info


def test_auto_off_writes_no_native_mask(run):
    info, seen = run(crop_masks(), auto=False)
    assert seen["paint"] == [] and not os.path.exists(os.path.join(run.work, "occ_native.npz"))
    assert info["occluder"] == {"space": "crop", "frames": N * MM}
