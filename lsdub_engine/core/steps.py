# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""steps.py: one engine step run from another. In-process by default (`main(argv)` here); LSDUB_STEP_MODE=child runs it as
`python -m ...` with cpu_env(), through io/ff.py. Either way the step gets the same argv."""

import contextlib, importlib, io, os, sys

from ..io import ff
from .cpu_env import cpu_env

PKG = __name__.rpartition(".core")[0]
# step -> (module holding main, what `-m` runs, its leading argv)
STEPS = {
    "paint": ("core.mouth_paint", "cli", ("paint",)),
    "visemes": ("core.viseme_track", "cli", ("visemes",)),
    "sung_dyn": ("core.sung_dyn", "core.sung_dyn", ()),
}


def mode():
    return os.environ.get("LSDUB_STEP_MODE") or "inprocess"


def _failed(what, ret, argv):
    print(f"{what} failed ({ret!r}): {' '.join(argv)}", file=sys.stderr)
    raise RuntimeError(f"{what} exited {ret!r}")


def _inprocess(main, argv, quiet, what):
    sink = io.StringIO() if quiet else None
    try:
        with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
            ret = main(argv)
    except SystemExit as e:
        ret = e.code
    if ret not in (None, 0):
        _failed(what, ret, argv)
    return ret


def _child(step, argv, quiet, what):
    _, module, lead = STEPS[step]
    r = ff.python(None, f"{PKG}.{module}", [*lead, *argv], env=cpu_env(), stdout=ff.DEVNULL if quiet else None)
    if r.returncode != 0:
        _failed(what, r.returncode, argv)
    return 0


def call(step, argv, quiet=False, name=None):
    """run a step ('paint', 'visemes', 'sung_dyn', or in-process only a main function). quiet drops its stdout, stderr stays.
    A failure raises RuntimeError: a failed step stops the caller."""
    argv = [str(a) for a in argv]
    if callable(step):
        return _inprocess(step, argv, quiet, name or getattr(step, "__module__", "step"))
    what = name or step
    if mode() == "child":
        return _child(step, argv, quiet, what)
    main = importlib.import_module(f"{PKG}.{STEPS[step][0]}").main
    return _inprocess(main, [a for a in argv if a != "--no-watermark"], quiet, what)  # the watermark is the CLI's
