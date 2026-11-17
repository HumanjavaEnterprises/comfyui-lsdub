# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sync 0.4 step 5: the auto pick counts events where Verify's event band cannot split the two candidates (sing.pick_rule thin)."""

import pytest

from lsdub_engine.core import sing as S


def g(truth, hit, n, band=False, silence=0.1, ok=True, **kw):
    return dict(
        truth=truth, corr=0.6, open_in_silence=silence, **{"pass": ok}, ev_read="phones", ev_band=band, ev_onset_hit=hit, ev_onset_n=n, **kw
    )


def test_event_count_is_onsets_plus_closures_less_the_strays_over_the_take():
    assert S.event_count(g(0.4, 0.75, 4)) == 3
    assert S.event_count(g(0.4, 0.5, 8, ev_closure_hit=0.5, ev_closure_n=2, ev_spurious=3, ev_spurious_take=1)) == 3
    assert S.event_count(g(0.4, 0.5, 8, ev_spurious=1, ev_spurious_take=4)) == 4  # fewer strays than the take: no credit
    assert S.event_count({"ev_read": "classic: the envelope gate picks"}) is None
    assert S.event_count(dict(g(0.4, None, 0))) is None


def test_thin_events_picks_the_candidate_with_more_events_ahead_of_pass_beats_fail():
    s, n = g(0.30, 0.75, 4, silence=0.39, ok=False), g(0.35, 0.25, 4)  # L003's shape: sing fails the silence line, 3 events v 1
    assert S.pick_rule(s, n, thin="envelope") == ("native", "envelope (neither inside the event band)")
    assert S.pick_rule(s, n, thin="events") == ("sing", "event count (neither inside the event band)")
    assert S.pick_rule(n, s, thin="events")[0] == "native"
    both = S.pick_rule(dict(s, ev_band=True), dict(n, ev_band=True), thin="events")
    assert both == ("sing", "event count (both inside the event band)")


def test_thin_events_leaves_a_lead_inside_the_reading_noise_to_the_envelope():
    s, n = g(0.30, 0.75, 4, ok=False), g(0.35, 0.5, 4)  # 3 v 2: under PICK_MARGIN
    assert S.PICK_MARGIN == 2
    assert S.pick_rule(s, n, thin="events") == S.pick_rule(s, n, thin="envelope") == ("native", "envelope (neither inside the event band)")


def test_thin_events_never_carries_a_candidate_under_the_envelope_floor():
    s, n = g(0.05, 1.0, 4), g(0.35, 0.25, 4)
    assert S.pick_rule(s, n, thin="events")[0] == "native"


def test_thin_events_changes_nothing_where_the_band_decides_or_does_not_read():
    s, n = g(0.4, 0.8, 10, band=True), g(0.5, 1.0, 10, band=False)
    assert S.pick_rule(s, n, thin="events") == ("sing", "events")
    c = {"ev_read": "classic: the envelope gate picks", "ev_band": None}
    a, b = dict(g(0.3, 1.0, 4), **c), dict(g(0.5, 0.0, 4), **c)
    assert S.pick_rule(a, b, thin="events") == S.pick_rule(a, b, thin="envelope")
    assert S.pick_rule(a, b, thin="events")[1] == "envelope (classic: the envelope gate picks)"


def test_thin_mode_is_named_and_checked():
    assert S.PICK_THINS == ("envelope", "events") and S.PICK_THIN in S.PICK_THINS
    with pytest.raises(ValueError):
        S.pick_rule(g(0.4, 1.0, 4), g(0.4, 0.5, 4), thin="loudest")
    assert "ev_spurious_take" in S.AUTO_EV_KEYS  # the count's anchor is recorded with the pick
