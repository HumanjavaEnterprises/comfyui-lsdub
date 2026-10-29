from lsdub_engine import cone


def test_zones():
    assert cone.zone(0, 0, 0) == "green"
    assert cone.zone(25, 0, 0) == "amber"
    assert cone.zone(0, -17.5, 0) == "amber"   # head tilted back past the green limit
    assert cone.zone(45, 0, 0) == "red"
    assert cone.zone(None, None, None) == "lost"


def test_report_advice():
    frames = [{"yaw": 0, "pitch": 0, "roll": 0}] * 20
    assert cone.report({"frames": frames})["advice"] == "shared library"
    frames = [{"yaw": 30, "pitch": 0, "roll": 0}] * 20
    assert cone.report({"frames": frames})["advice"].startswith("per-shot")
    frames = [{"yaw": 60, "pitch": 0, "roll": 0}] * 20
    assert cone.report({"frames": frames})["advice"].startswith("re-stage")


def test_lost_frames_do_not_count():
    frames = [{"yaw": 0, "pitch": 0, "roll": 0}] * 18 + [{"yaw": None, "pitch": None, "roll": None}] * 2
    r = cone.report({"frames": frames})
    assert r["fractions"]["green"] == 1.0 and r["fractions"]["lost"] == 0.1
