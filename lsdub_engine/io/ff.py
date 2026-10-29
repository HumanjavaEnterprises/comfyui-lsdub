# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""ff.py: the ONE place the engine starts an external process. Only ffmpeg and ffprobe, always as an argv list,
never through a shell. Every other step runs in-process."""
import shutil
import subprocess

PIPE, DEVNULL, CalledProcessError = subprocess.PIPE, subprocess.DEVNULL, subprocess.CalledProcessError
ALLOWED = ('ffmpeg', 'ffprobe')


def _argv(argv):
    argv = [str(a) for a in argv]
    if not argv or argv[0] not in ALLOWED:
        raise ValueError(f'ff: only {ALLOWED} may be started, got {argv[:1]}')
    exe = shutil.which(argv[0])
    if exe is None:
        raise FileNotFoundError(f'ff: {argv[0]} is not on PATH (install ffmpeg)')
    return [exe, *argv[1:]]


def run(argv, **kw):
    """subprocess.run for ffmpeg / ffprobe (argv list, shell=False)."""
    kw.pop('shell', None)
    return subprocess.run(_argv(argv), shell=False, **kw)  # noqa: S603


def popen(argv, **kw):
    """subprocess.Popen for ffmpeg / ffprobe (argv list, shell=False), e.g. a raw-frame pipe into the encoder."""
    kw.pop('shell', None)
    return subprocess.Popen(_argv(argv), shell=False, **kw)  # noqa: S603
