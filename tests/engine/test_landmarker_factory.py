# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""landmarker.py without a model: one warning at most, the shared table under its lock, VIDEO instances closed by their callers."""

import logging, sys, threading, types

from lsdub_engine.core import landmarker as L


class Fake:
    closed = 0

    def close(self):
        type(self).closed += 1


def test_mp_warns_only_when_someone_else_imported_it(monkeypatch, caplog):
    monkeypatch.setattr(L, "_checked", False)
    monkeypatch.setitem(sys.modules, "mediapipe", types.ModuleType("mediapipe"))
    monkeypatch.setattr(L, "guard", lambda: None)
    with caplog.at_level(logging.WARNING, "lsdub"):
        L.mp()
        L.mp()
    assert len(caplog.records) == 1


def test_mp_stays_silent_after_its_own_import(monkeypatch, caplog):
    monkeypatch.setattr(L, "_checked", False)
    monkeypatch.delitem(sys.modules, "mediapipe", raising=False)
    monkeypatch.setattr(L, "guard", lambda: None)
    monkeypatch.setitem(sys.modules, "mediapipe", types.ModuleType("mediapipe"))
    monkeypatch.setattr(L, "_checked", True)  # the state after a first call
    with caplog.at_level(logging.WARNING, "lsdub"):
        L.mp()
    assert not caplog.records


def test_shared_instance_is_made_once_across_threads(monkeypatch):
    made = []

    def make(*a):
        made.append(a)
        return Fake()

    monkeypatch.setattr(L, "_make", make)
    monkeypatch.setattr(L, "_shared", {})
    ts = [threading.Thread(target=L.get, kwargs={"model": "m"}) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(made) == 1
    L.close_all()
    assert Fake.closed == 1 and not L._shared
