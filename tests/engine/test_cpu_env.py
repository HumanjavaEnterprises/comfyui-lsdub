# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the environment of a step: child mode carries cpu_env(), in-process mode touches nothing, one process boundary."""

import os, pathlib, re
import pytest
import lsdub_engine
from lsdub_engine.core import cpu_env as CE, steps
from lsdub_engine.io import ff

ROOT = pathlib.Path(lsdub_engine.__file__).resolve().parent


def test_no_process_start_outside_io_ff():
    bad = []
    for p in ROOT.rglob("*.py"):
        rel = str(p.relative_to(ROOT))
        if rel == os.path.join("io", "ff.py") or rel.startswith(("legacy", "backends")):
            continue
        if re.search(r"\bsubprocess\b|sys\.executable|os\.system|os\.popen", p.read_text(encoding="utf-8")):
            bad.append(rel)
    assert not bad, bad


def test_cpu_env_values():
    e = CE.cpu_env()
    assert e["CUDA_VISIBLE_DEVICES"] == "" and e["LIBGL_ALWAYS_SOFTWARE"] == "1" and e["MESA_LOADER_DRIVER_OVERRIDE"] == "llvmpipe"


def test_guard_keeps_caller_values(monkeypatch):
    monkeypatch.setenv("MESA_LOADER_DRIVER_OVERRIDE", "mine")
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
    CE.guard()
    assert os.environ["MESA_LOADER_DRIVER_OVERRIDE"] == "mine" and os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1"


def test_guard_never_touches_the_hosts_cuda_or_threads(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    CE.guard()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ and "OMP_NUM_THREADS" not in os.environ


def test_thread_report_observes_and_sets_nothing():
    import os
    from lsdub_engine.core.cpu_env import thread_report

    before = dict(os.environ)
    line = thread_report()
    assert os.environ == before
    for k in ("cpu_count=", "cv2=", "cv2_threads=", "cv2_file=", "torch_threads=", "OMP_NUM_THREADS="):
        assert k in line


def test_child_mode_carries_both_var_sets(monkeypatch):
    seen = {}

    class R:
        returncode = 0

    monkeypatch.setenv("LSDUB_STEP_MODE", "child")
    monkeypatch.setattr(ff, "python", lambda exe, mod, args, **k: seen.update(exe=exe, mod=mod, args=args, **k) or R)
    steps.call("paint", ["--mode", "warp", 3], quiet=True)
    assert seen["exe"] is None and seen["mod"] == "lsdub_engine.cli" and seen["args"] == ["paint", "--mode", "warp", "3"]
    assert seen["stdout"] == ff.DEVNULL
    assert {**CE.GL_VARS, **CE.CHILD_VARS}.items() <= seen["env"].items()


def test_child_failure_raises(monkeypatch):
    class R:
        returncode = 3

    monkeypatch.setenv("LSDUB_STEP_MODE", "child")
    monkeypatch.setattr(ff, "python", lambda *a, **k: R)
    with pytest.raises(RuntimeError, match="sung_dyn exited 3"):
        steps.call("sung_dyn", [])


def test_inprocess_touches_no_env_and_keeps_argv(monkeypatch):
    monkeypatch.delenv("LSDUB_STEP_MODE", raising=False)
    from lsdub_engine.core import viseme_track as VT

    got = []
    monkeypatch.setattr(VT, "main", lambda argv: got.append(argv) or 0)
    before = dict(os.environ)
    steps.call("visemes", ["--t0", 1.5], quiet=True)
    assert got == [["--t0", "1.5"]] and dict(os.environ) == before


def test_each_step_has_a_module_with_main():
    import importlib

    for mod, _, _ in steps.STEPS.values():
        assert callable(importlib.import_module(f"lsdub_engine.{mod}").main)


def test_landmarkers_only_from_the_factory():
    for p in ROOT.rglob("*.py"):
        rel = str(p.relative_to(ROOT))
        if rel == os.path.join("core", "landmarker.py") or rel.startswith("legacy"):
            continue
        s = p.read_text(encoding="utf-8")
        assert "create_from_options" not in s and not re.search(r"^\s*(import|from) mediapipe|mediapipe as ", s, re.M), rel


def test_guard_runs_only_inside_the_factory():
    hits = [p.name for p in ROOT.rglob("*.py") if re.search(r"\bguard\(\)", p.read_text(encoding="utf-8"))]
    assert sorted(hits) == ["cpu_env.py", "landmarker.py"]
