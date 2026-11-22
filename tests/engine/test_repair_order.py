# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the continuity repair of a shot does not depend on what the process repaired before it: the same shot gives the same frames and
the same report first, after a small shot, and in either order with another shot. (One optical-flow object kept for the life of the
process carried a setting from a small image into every later shot; cv2's DIS lowers its finest scale on an image whose longer side
is 45 px or less and keeps it lowered.) The same for the slip audit's dense flow."""

import hashlib
import json

import cv2
import numpy as np

from lsdub_engine.audits import slip_audit as SA
from lsdub_engine.core import continuity as C

FPS = 25.0


def _texture(seed, size):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur((rng.random((size + 80, size + 80, 3)) * 255).astype(np.uint8), (0, 0), 2.5)
    return cv2.normalize(t, None, 20, 235, cv2.NORM_MINMAX)


def _shot(seed, size, n):
    """a take that drifts (so the take's flow is not zero) and a render with a painted mouth on it: a steady blob that moves with the
    take, a one-frame glitch and a three-frame odd run."""
    tex = _texture(seed, size)
    a, b = size * 3 // 8, size * 5 // 8
    T, R = [], []
    for i in range(n):
        dx, dy = 40 - i, 40 - (i // 2)
        t = np.ascontiguousarray(tex[dy : dy + size, dx : dx + size])
        r = t.copy()
        r[a:b, a - size // 16 : b + size // 16] = (60, 60, 160)
        r[a + 2 : b - 2 : 3, a:b] = t[a + 2 : b - 2 : 3, a:b] // 2 + 40  # some texture in the paint, for the flow to read
        T.append(t)
        R.append(r)
    R[n // 3][a:b, a:b] = (250, 250, 250)
    for i in range(n * 5 // 8, min(n, n * 5 // 8 + 3)):
        R[i][a:b:2, a:b] = (235, 240, 245)
    return T, R


def _repair(monkeypatch, shot, tmp_path):
    """continuity.run on an in-memory shot (the reader and the writer are the only parts replaced) -> (frames digest, report)."""
    T, R = shot
    got = {}
    monkeypatch.setattr(C, "_read", lambda p: ([f.copy() for f in (T if p == "take" else R)], FPS))
    monkeypatch.setattr(C, "write", lambda frames, fps, out, src: got.update(frames=[f.copy() for f in frames]))
    rep = C.run("take", "render", "out", report=str(tmp_path / "r.json"))
    h = hashlib.sha256()
    for f in got["frames"]:
        h.update(np.ascontiguousarray(f).tobytes())
    return h.hexdigest(), json.dumps(rep, sort_keys=True)


A = lambda: _shot(1, 96, 20)
B = lambda: _shot(2, 112, 16)
SMALL = lambda: _shot(3, 40, 16)  # every image the repair hands the flow from this shot has a longer side of 45 px or less


def test_the_repair_changes_frames_on_the_test_shots(monkeypatch, tmp_path):
    """the test below would pass on a repair that does nothing: these shots are ones it re-makes frames on."""
    for shot in (A(), B()):
        _, rep = _repair(monkeypatch, shot, tmp_path)
        rep = json.loads(rep)
        assert rep["fixed"] or rep["coherence"]


def test_a_repair_is_the_same_before_and_after_a_small_shot(monkeypatch, tmp_path):
    first = _repair(monkeypatch, A(), tmp_path)
    _repair(monkeypatch, SMALL(), tmp_path)
    again = _repair(monkeypatch, A(), tmp_path)
    assert first == again


def test_repairs_are_the_same_in_either_order(monkeypatch, tmp_path):
    shots = dict(a=A, small=SMALL, b=B)
    one = {k: _repair(monkeypatch, shots[k](), tmp_path) for k in ("a", "small", "b")}
    two = {k: _repair(monkeypatch, shots[k](), tmp_path) for k in ("small", "b", "a")}
    three = {k: _repair(monkeypatch, shots[k](), tmp_path) for k in ("b", "a", "small")}
    assert one == two == three


def test_the_repair_keeps_no_flow_object_between_calls():
    assert not [k for k, v in vars(C).items() if isinstance(v, cv2.DISOpticalFlow)]
    C._carry(np.zeros((40, 40), np.float32), np.zeros((40, 40), np.uint8), np.zeros((40, 40), np.uint8))
    assert not [k for k, v in vars(C).items() if isinstance(v, cv2.DISOpticalFlow)]


def test_the_slip_audit_flow_is_the_same_before_and_after_a_small_video():
    big, small = _shot(4, 128, 6)[0], _shot(5, 40, 4)[0]
    first = SA.flows(big)
    SA.flows(small)
    again = SA.flows(big)
    assert all(np.array_equal(x, y) for x, y in zip(first, again))


def test_a_small_crop_is_carried_the_same_every_time():
    """every flow runs on a new object, so a crop small enough to lower the setting (41 to 45 px paint boxes occur part-way through
    real shots) is measured the same on the first frame and on every later one, and a large crop the same before and after it."""
    g = [cv2.cvtColor(_texture(s, 0)[:40, :40], cv2.COLOR_BGR2GRAY) for s in (11, 12)]
    big = [cv2.cvtColor(_texture(s, 40), cv2.COLOR_BGR2GRAY) for s in (13, 14)]
    first, wide = C._carry(g[0], g[0], g[1]), C._carry(big[0], big[0], big[1])
    for _ in range(3):
        assert np.array_equal(C._carry(g[0], g[0], g[1]), first)
    assert np.array_equal(C._carry(big[0], big[0], big[1]), wide)
