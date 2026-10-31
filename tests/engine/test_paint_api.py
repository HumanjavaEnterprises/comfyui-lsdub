# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The one paint call: layers, their precedence, --recipe."""

import pytest

from lsdub_engine.core import mouth_paint as MP, paint_api as A, presets as P, recipes as R


def last(argv, flag):
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


def test_plain_flags_are_the_layers_in_order():
    got = A.flags(lib="k", mode_mods=["--warp-track-mode", "chain"], user_overrides=["--vocalisations", "mouth"], operational=["--x", 1])
    assert got == ["--lib", "k", "--warp-track-mode", "chain", "--vocalisations", "mouth", "--x", "1"]


def test_a_later_layer_wins():
    a = A.flags("layers_v16_1", mode_mods=["--warp-erase", "m"], face_overrides=["--warp-erase", "f"], user_overrides=["--warp-erase", "u"])
    assert last(a, "--warp-erase") == "u" and "m" not in a and "closures" not in a
    a = A.flags("layers_v16_1", face_overrides=["--warp-cavity", "off"])
    assert a.count("--warp-cavity") == 1 and last(a, "--warp-cavity") == "off"
    a = A.flags(user_overrides=["--warp-temporal", "0.3"], operational=["--warp-temporal", "0.6"])
    assert MP._parser().parse_args(["--video", "-", "--track", "-", "--out", "-", *a]).warp_temporal == 0.6


def test_face_overrides_keep_the_landing_slot():
    assert tuple(A.flags("layers_v16_1", face_overrides=P.LANDING.face_overrides["red_nova"])) == P.LANDING.flags_for("red_nova")


def test_argv_modes_and_the_operational_tail():
    assert A.argv("v", "t", 4, out="o", warp_json="w") == [
        "--mode", "warp", "--video", "v", "--track", "t", "--frames", "4", "--out", "o", "--warp-json", "w", "--no-watermark",
    ]  # fmt: skip
    assert A.argv("v", "t", 4, "morph", out="o")[:2] == ["--mode", "morph"]


def test_coart_dyn_only_on_a_coart_target():
    assert A.coart_dyn(["--lib", "k"], "d") == ()
    assert A.coart_dyn(["--warp-target", "own"], "d") == ()
    assert A.coart_dyn(R.LAYERS_V16_1, "d")[:2] == ("--coart-dyn", "d")
    assert A.coart_dyn(["--warp-target", "coart"], "d")  # a user's own target still gets it
    assert A.coart_dyn(["--recipe", "layers_v16_1"], "d")
    assert A.coart_dyn([*R.LAYERS_V16_1, "--warp-target=v9"], "d") == ()


def test_recipe_expands_in_place():
    got = A.expand(["--lib", "k", "--recipe", "layers_v16_1", "--warp-erase", "x"])
    assert got == ["--lib", "k", "--mode", "warp", *R.LAYERS_V16_1, "--warp-erase", "x"]
    assert A.expand(["--recipe=plain"]) == ["--mode", "warp"] and A.expand(["--recipe", "morph"]) == ["--mode", "morph"]
    with pytest.raises(SystemExit, match="unknown recipe"):
        A.expand(["--recipe", "nope"])
    assert MP.paint_warnings(["--recipe", "layers_v16_1"]) is not None


def test_paint_runs_through_steps(monkeypatch):
    seen = []
    monkeypatch.setattr(A.steps, "call", lambda step, argv: seen.append((step, argv)))
    A.paint("v", "t", 4, out="o")
    assert seen[0][0] == "paint" and seen[0][1][:2] == ["--mode", "warp"]
