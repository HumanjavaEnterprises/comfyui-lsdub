# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Clean-room + registry gates, run as tests: the scrub grep is empty (code AND tests), ruff S102/S307/E702/F is clean,
no eval/exec, no runtime pip, no downloads at import, and no subprocess outside the one ffmpeg helper."""
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANNED = re.compile('|'.join(['latent' + 'sync', 'wav2' + 'lip', 'sad' + 'talker', 'muse' + 'talk', 'insight' + 'face',
                              'arc' + 'face', 'ante' + 'lope', 'code' + 'former', 'gfp' + 'gan', '/ho' + 'me/', 'gpu' + 'host']), re.I)
SKIP_DIRS = {'.git', '__pycache__', '.ruff_cache', '.pytest_cache'}
# every external process goes through io/ff.py (ffmpeg/ffprobe, the opt-in aligners, a validated interpreter)
SUBPROCESS_OK = {os.path.join('lsdub_engine', 'io', 'ff.py')}


def _files(exts=('.py', '.md', '.json', '.txt', '.toml', '.cff', '.yaml', '.yml', '')):
    for d, ds, fs in os.walk(ROOT):
        ds[:] = [x for x in ds if x not in SKIP_DIRS]
        for f in fs:
            if f in SKIP_DIRS: continue                                    # a worktree's .git is a FILE
            if os.path.splitext(f)[1] in exts: yield os.path.join(d, f)


def test_scrub_grep_is_empty():
    hits = []
    for p in _files():
        try: txt = open(p, encoding='utf-8').read()
        except (UnicodeDecodeError, OSError): continue
        hits += [f'{os.path.relpath(p, ROOT)}:{i + 1}' for i, ln in enumerate(txt.splitlines()) if BANNED.search(ln)]
    assert not hits, hits


def _ruff():
    exe = shutil.which('ruff')
    if exe: return [exe]
    try:
        import ruff  # noqa: F401
        return [sys.executable, '-m', 'ruff']
    except ImportError:
        return None


def test_ruff_registry_rules():
    r = _ruff()
    if r is None: pytest.skip('ruff not installed')
    out = subprocess.run([*r, 'check', '--select', 'S102,S307,E702,F', '--target-version', 'py310', ROOT],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:]


def test_no_subprocess_outside_ff_helper():
    bad = []
    for p in _files(('.py',)):
        rel = os.path.relpath(p, ROOT)
        if not rel.startswith('lsdub_engine') and not rel.startswith('nodes') and rel not in ('__init__.py', 'project.py', 'lsdub_types.py'):
            continue
        txt = open(p).read()
        if rel not in SUBPROCESS_OK and re.search(r'\bsubprocess\b|os\.system|os\.popen|sys\.executable', txt): bad.append(rel)
    assert not bad, bad


def test_no_eval_exec_pip():
    bad = []
    for p in _files(('.py',)):
        rel = os.path.relpath(p, ROOT)
        if rel.startswith('tests'): continue
        for i, ln in enumerate(open(p).read().splitlines()):
            if re.search(r'(?<![.\w])(eval|exec)\(|pip install|ensurepip|pip\.main', ln) and 'help=' not in ln and '(pip install sam2)' not in ln:
                bad.append(f'{rel}:{i + 1}')
    assert not bad, bad


def test_import_is_cheap_and_offline(pack):
    """importing the pack + the engine pulls in no heavy library and opens no network connection."""
    code = ('import sys, socket\n'
            'def _no(*a, **k): raise RuntimeError("network at import")\n'
            'socket.socket.connect = _no\n'
            f'sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n'
            'import conftest\n'
            'conftest.load_pack()\n'
            'import lsdub_engine, lsdub_engine.cli\n'
            'heavy = [m for m in ("torch", "cv2", "mediapipe", "scipy", "transformers", "onnx") if m in sys.modules]\n'
            'print("HEAVY", heavy)\n')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert 'HEAVY []' in r.stdout, r.stdout
