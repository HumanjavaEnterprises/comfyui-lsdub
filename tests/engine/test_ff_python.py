# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""ff.python: engine modules only, named by the package as imported or bare; the child resolves them top-level. These tests vendor
with the engine, so they also run on the renamed tree. A host's package around it is the pack's own test."""

import os, sys

import pytest

from lsdub_engine.core import steps
from lsdub_engine.io import ff


def test_a_foreign_module_is_refused():
    for m in ("json.tool", "os", "lsdubx.cli"):
        with pytest.raises(ValueError):
            ff.python(None, m, [])


def test_a_host_package_prefix_is_dropped(monkeypatch):
    monkeypatch.setattr(ff, "ENGINE", f"host.{ff._TOP}")
    assert ff._engine_module(f"host.{ff._TOP}.cli") == f"{ff._TOP}.cli"
    assert ff._engine_module(f"{ff._TOP}.cli") == f"{ff._TOP}.cli"


def test_child_resolves_from_anywhere_without_pythonpath(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    r = ff.python(None, "lsdub_engine.core.cpu_env", [], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_second_interpreter_path_resolves(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    r = ff.python(sys.executable, "lsdub_engine.cli", ["--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_child_env_keeps_the_callers_entries():
    env = ff._engine_env({"PYTHONPATH": "/x", "A": "1"})
    assert env["PYTHONPATH"].split(os.pathsep) == [ff._HOME, "/x"] and env["A"] == "1"


def test_step_mode_child_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LSDUB_STEP_MODE", "child")
    assert steps.call("visemes", ["--help"], quiet=True) == 0
    assert steps.call("sung_dyn", ["--help"], quiet=True) == 0


def test_mouth_ruler_fan_out_module_resolves(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    r = ff.python(None, "lsdub_engine.audits.mouth_ruler", ["shot", "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_whisper_child_module_resolves(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    r = ff.python(sys.executable, "lsdub_engine.core.lyric_align", ["--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--stt-only" in r.stdout, r.stderr
