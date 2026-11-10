# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""0.3.0: on a coart face (the landing preset's layer stack) the vocal-driven mouth rides the plain target, classic keeps coart, and the
auto pick's `by` names the rule that decided."""

from lsdub_engine.core import paint_api, recipes, sing as S

CLASSIC = [x for f in S.STAGES for x in (f, "classic")]
LAYERS = paint_api.flags("layers_v16_1", lib="K", mode_mods=["--warp-track-mode", "chain"], user_overrides=["--vocalisations", "skip"])


def test_the_landing_stack_is_coart():
    assert paint_api.target(LAYERS) == "coart"
    assert recipes.get("layers_v16_1").expand(()).count("coart") >= 1


def test_coart_at_the_defaults_routes_to_the_plain_target_and_the_stages_ride():
    a = S.routed(LAYERS)
    assert a[: len(LAYERS)] == LAYERS and paint_api.target(a) == "v9"
    snap = S.snap_paint(a, 2)
    for f in S.STAGES:
        assert f in snap
    assert S.stage_value(a, snap, "--warp-events") == "vocal"
    assert S.warp_audio(a, snap, "v.wav") == ("--warp-audio", "v.wav")
    assert paint_api.coart_dyn(a, "d.csv") == ()


def test_classic_snap_off_and_other_targets_are_unchanged():
    for args in ([*LAYERS, *CLASSIC], [*LAYERS, "--warp-events", "off"], [*LAYERS, "--warp-events=classic"], ["--warp-target", "own"], []):
        assert S.routed(args) == list(args)
    assert S.routed(LAYERS, snap=False) == list(LAYERS)
    assert not any(f in S.snap_paint([*LAYERS, *CLASSIC], 2) for f in S.STAGES)


def test_routing_is_idempotent():
    assert S.routed(S.routed(LAYERS)) == S.routed(LAYERS)


def test_one_stage_on_classic_still_routes_while_events_are_vocal():
    a = S.routed([*LAYERS, "--warp-hold", "classic"])
    assert paint_api.target(a) == "v9" and S.stage_value(a, S.snap_paint(a, 2), "--warp-hold") == "off"


G = dict(truth=0.5, truth_lag_ms=0.0, corr=0.5, best=0.5, open_in_silence=0.1)


def test_by_says_events_only_when_the_event_band_decided():
    s = dict(G, **{"pass": False, "truth": 0.25, "ev_band": True})
    n = dict(G, **{"pass": True, "ev_band": False})
    assert S.pick_rule(s, n) == ("sing", "events")


def test_by_names_the_0_2_rule_when_it_decided():
    s = dict(G, **{"pass": False, "truth": 0.25, "ev_band": True})
    n = dict(G, **{"pass": True, "ev_band": False})
    assert S.pick_rule(dict(s, truth=0.1), n) == ("native", "envelope (the sing inside the event band is under the envelope floor)")
    assert S.pick_rule(dict(s, ev_band=False), n) == ("native", "envelope (neither inside the event band)")
    assert S.pick_rule(dict(s, ev_band=True), dict(n, ev_band=True)) == ("native", "envelope (both inside the event band)")
    c = {"ev_read": "classic: the envelope gate picks"}
    assert S.pick_rule(dict(s, ev_band=None, **c), dict(n, ev_band=None, **c)) == ("native", "envelope (classic: the envelope gate picks)")
    for a, b in ((s, n), (dict(s, ev_band=None), n), (dict(s, truth=0.1), n)):
        assert S.pick_rule(a, b)[0] == S.pick_paint(a, b)
