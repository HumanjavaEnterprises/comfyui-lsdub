# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sync 0.4: which fixes are the default. On (the 190-shot ledger held on every row): the lyric-confirmed nuclei (step 6) and the
plan-aware repair (step 1). Off (a ledger row regressed): the contact seal (2), the rise floor (3), the rise onset (4), the event
pick on thin shots (5). `classic` on the stage flags is still the 0.2.x paint."""

from lsdub_engine.core import mouth_paint as MP
from lsdub_engine.core import mouth_warp as MW
from lsdub_engine.core import sing as S

OLD = ("--warp-events", "--warp-hold", "--warp-closures", "--warp-aperture", "--warp-realise")  # what a 0.3.0 caller names for classic
CLASSIC_030 = [x for f in OLD for x in (f, "classic")]
BASE = ["--video", "v", "--track", "t", "--out", "o"]


def _value(flags, name):
    return flags[flags.index(name) + 1] if name in flags else None


def test_accepted_defaults_are_on():
    snap = S.snap_paint([], 2)
    assert _value(snap, "--warp-nuclei") == "lyric"
    assert S.stage_value([], snap, "--warp-nuclei") == "lyric"
    assert S.REPAIR_PLAN == "warp" and S.repair_plan([], True) == "warp"
    assert vars(MP._parser().parse_args([*BASE, *snap]))["warp_nuclei"] == "lyric"


def test_rejected_steps_stay_off():
    a = vars(MP._parser().parse_args([*BASE, *S.snap_paint([], 2)]))
    assert a["warp_pbm"] == "off" and a["warp_rise"] == "off" and a["warp_onset"] == "apex"
    assert MW.SEAL_GAP < 0  # the seal target is the take's own shut unless --warp-pbm contact asks
    assert S.PICK_THIN == "envelope"


def test_a_caller_names_the_nuclei():
    for v in ("all", "classic"):
        snap = S.snap_paint(["--warp-nuclei", v], 2)
        assert "--warp-nuclei" not in snap  # the caller's value wins
        assert S.stage_value(["--warp-nuclei", v], snap, "--warp-nuclei") == "all"
    assert vars(MP._parser().parse_args([*BASE, "--warp-nuclei", "classic"]))["warp_nuclei"] == "all"


def test_classic_as_030_named_it_is_still_the_02x_paint():
    """a caller that names only 0.3.0's five stages (the pack's classic switch) gets no 0.4 flag and no plan-aware repair."""
    for args in (CLASSIC_030, ["--warp-events", "off"], ["--warp-events=classic"]):
        snap = S.snap_paint(args, 2)
        assert "--warp-nuclei" not in snap, snap  # it reads the event stage, so it leaves with it
        assert S.repair_plan(args, True) == "off"
    assert not any(f in S.snap_paint(CLASSIC_030, 2) for f in S.STAGES)
    old = vars(MP._parser().parse_args([*BASE, *CLASSIC_030, *S.snap_paint(CLASSIC_030, 2)]))
    assert old["warp_nuclei"] == "all" and old["warp_events"] == "off"
    assert S.repair_plan([], False) == "off"  # snap off


def test_coart_faces_keep_their_own_planner():
    snap = S.snap_paint(["--warp-target", "coart"], 2)
    assert "--warp-nuclei" not in snap
