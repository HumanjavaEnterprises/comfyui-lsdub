# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the landmarker factory: shared IMAGE instances, a fresh VIDEO one (timestamps are per instance), CPU delegate, one warning."""

import logging, sys

import pytest

from lsdub_engine.core import landmarker as L


@pytest.fixture
def fake(monkeypatch):
    made = []

    class Inst:
        closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(L, "_make", lambda *a: made.append(a) or Inst())
    monkeypatch.setattr(L, "_shared", {})
    return made


def test_image_is_shared_per_key(fake):
    assert L.get(conf=0.2) is L.get(conf=0.2)
    assert L.get(conf=0.3) is not L.get(conf=0.2)
    assert len(fake) == 2


def test_video_is_new_each_time(fake):
    assert L.get("video") is not L.get("video")


def test_close_all(fake):
    a = L.get()
    L.close_all()
    assert a.closed and L.get() is not a


def test_bad_mode(fake):
    with pytest.raises(ValueError):
        L.get("stream")


def test_warns_once_when_mediapipe_came_first(monkeypatch, caplog):
    monkeypatch.setattr(L, "_checked", False)
    monkeypatch.setitem(sys.modules, "mediapipe", object())
    with caplog.at_level(logging.WARNING, "lsdub"):
        L.mp()
        L.mp()
    assert sum("before the landmarker factory" in r.message for r in caplog.records) == 1
