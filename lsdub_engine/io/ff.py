# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""ff.py: the ONE place the engine starts an external process, always as an argv list, never through a shell.

  run / popen   ffmpeg and ffprobe (every node path uses only these)
  tool          the opt-in CLI aligners: rhubarb (lip cues) and mfa (Montreal Forced Aligner); never on a node path
  python        a second Python interpreter for speech-to-text (`lsdub align --whisper-python`); CLI only, validated

Every other engine step runs in-process."""
import os
import shutil
import subprocess

PIPE, DEVNULL, CalledProcessError = subprocess.PIPE, subprocess.DEVNULL, subprocess.CalledProcessError
ALLOWED = ('ffmpeg', 'ffprobe')
TOOLS = ('rhubarb', 'mfa')


def _resolve(exe, allowed):
    name = os.path.basename(exe)
    if name not in allowed:
        raise ValueError(f'ff: only {allowed} may be started, got {exe!r}')
    path = exe if os.path.isabs(exe) else shutil.which(exe)
    if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise FileNotFoundError(f'ff: {exe} is not installed (or not executable)')
    return path


def _argv(argv, allowed=ALLOWED):
    argv = [str(a) for a in argv]
    if not argv:
        raise ValueError('ff: empty argv')
    return [_resolve(argv[0], allowed), *argv[1:]]


def run(argv, **kw):
    """subprocess.run for ffmpeg / ffprobe (argv list, shell=False)."""
    kw.pop('shell', None)
    return subprocess.run(_argv(argv), shell=False, **kw)  # noqa: S603


def popen(argv, **kw):
    """subprocess.Popen for ffmpeg / ffprobe (argv list, shell=False), e.g. a raw-frame pipe into the encoder."""
    kw.pop('shell', None)
    return subprocess.Popen(_argv(argv), shell=False, **kw)  # noqa: S603


def tool(argv, **kw):
    """subprocess.run for an opt-in aligner (rhubarb, mfa): argv[0] is the tool name or an absolute path to it."""
    kw.pop('shell', None)
    return subprocess.run(_argv(argv, TOOLS), shell=False, **kw)  # noqa: S603


def python(interpreter, module, args, **kw):
    """run `<interpreter> -m <module> args` where interpreter is an existing executable named python*; module must be
    one of the engine's own (lsdub_engine.*)."""
    kw.pop('shell', None)
    if not module.startswith('lsdub_engine.'):
        raise ValueError(f'ff: only engine modules may be run, got {module!r}')
    exe = os.path.realpath(interpreter)
    if not os.path.basename(exe).startswith('python') or not os.path.isfile(exe) or not os.access(exe, os.X_OK):
        raise ValueError(f'ff: {interpreter!r} is not a Python interpreter')
    return subprocess.run([exe, '-m', module, *[str(a) for a in args]], shell=False, **kw)  # noqa: S603
