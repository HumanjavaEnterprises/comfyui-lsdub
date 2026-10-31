# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The engine loaded the way ComfyUI loads it, inside the pack's package: its children still resolve, with no PYTHONPATH help."""
import importlib
import json
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def host(pack, tmp_path, monkeypatch):
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.chdir(tmp_path)
    return pack.__name__


def test_vendored_ff_takes_the_host_prefixed_name(host):
    ff = importlib.import_module(f'{host}.lsdub_engine.io.ff')
    assert ff.ENGINE == f'{host}.lsdub_engine' and ff._TOP == 'lsdub_engine'
    r = ff.python(None, f'{host}.lsdub_engine.cli', ['--help'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_step_mode_child_under_the_host_package(host, monkeypatch):
    steps = importlib.import_module(f'{host}.lsdub_engine.core.steps')
    monkeypatch.setenv('LSDUB_STEP_MODE', 'child')
    assert steps.call('visemes', ['--help'], quiet=True) == 0


def test_whisper_python_child_runs_stt_end_to_end(host, tmp_path):
    """`align --whisper-python`: a second interpreter starts lyric_align --stt-only; the speech model is a stand-in."""
    stub = tmp_path / 'stub'
    stub.mkdir()
    (stub / 'faster_whisper.py').write_text(textwrap.dedent('''\
        import types
        class WhisperModel:
            def __init__(self, *a, **k): pass
            def transcribe(self, *a, **k):
                w = types.SimpleNamespace(word=' la', start=0.1, end=0.2, probability=0.9)
                return [types.SimpleNamespace(words=[w], text='la', start=0.1, end=0.2)], None
        '''))
    ff = importlib.import_module(f'{host}.lsdub_engine.io.ff')
    wav = tmp_path / 'a.wav'
    ff.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', '-t', '0.5', str(wav)], check=True)
    out = tmp_path / 'a.json'
    env = dict(os.environ, PYTHONPATH=str(stub))
    lyrics = tmp_path / 'l.txt'
    lyrics.write_text('la\n')
    sj = tmp_path / 's.json'
    r = ff.python(sys.executable, 'lsdub_engine.core.lyric_align',
                  ['--stt-only', '--audio', wav, '--lyrics', lyrics, '--t0', 0, '--t1', 0.5, '--out', out, '--stt-json', sj],
                  env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.load(open(sj))


def test_mouth_ruler_fan_out_under_the_host_package(host):
    ruler = importlib.import_module(f'{host}.lsdub_engine.audits.mouth_ruler')
    r = ruler.ff.python(None, 'lsdub_engine.audits.mouth_ruler', ['shot', '--help'], capture_output=True, text=True)
    assert r.returncode == 0 and '--manifest' in r.stdout, r.stderr


def test_packs_own_modules_are_not_needed_by_the_child(host):
    code = 'import sys, lsdub_engine.cli; assert not [m for m in sys.modules if m in ("media", "project", "graphs", "generate")]'
    assert subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}).returncode == 0
