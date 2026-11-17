# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""scripts/install_path_check.py always stops the ComfyUI it started (by its own Popen handle), also when a step fails
early or the job is cancelled with SIGTERM."""

import importlib.util
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "install_path_check.py")
spec = importlib.util.spec_from_file_location("install_path_check", SCRIPT)
ipc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ipc)

SLEEPER = [sys.executable, "-c", "import time; time.sleep(600)"]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_stop_none_and_exited_are_noops():
    ipc.stop_comfyui(None)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    ipc.stop_comfyui(p)


def test_stop_terminates_the_child():
    p = subprocess.Popen(SLEEPER)
    ipc.stop_comfyui(p, grace=10)
    assert p.returncode is not None


def test_stop_kills_a_child_that_ignores_sigterm():
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            'import signal, time, sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready", flush=True); time.sleep(600)',
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert p.stdout.readline().strip() == "ready"
    ipc.stop_comfyui(p, grace=1)
    assert p.returncode == -signal.SIGKILL


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name != "posix", reason="posix signals")
def test_sigterm_still_stops_the_child():
    """the same try/finally main() uses: SIGTERM to the script raises, the finally stops the child it started."""
    code = textwrap.dedent(f"""
        import importlib.util, subprocess, sys, time
        spec = importlib.util.spec_from_file_location('ipc', {SCRIPT!r})
        ipc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ipc)
        proc = None
        old = ipc.stop_on_signals()
        try:
            proc = subprocess.Popen({SLEEPER!r})
            print(proc.pid, flush=True)
            time.sleep(600)
        finally:
            ipc.stop_comfyui(proc, grace=10)
    """)
    runner = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = int(runner.stdout.readline())
    assert alive(child)
    runner.send_signal(signal.SIGTERM)
    _, err = runner.communicate(timeout=30)
    assert "INSTALL-PATH STOPPED" in err
    t0 = time.time()
    while alive(child) and time.time() - t0 < 10:
        time.sleep(0.1)
    assert not alive(child)
