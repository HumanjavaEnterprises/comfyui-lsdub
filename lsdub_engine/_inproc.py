# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""_inproc.py: run an engine step's `main(argv)` in this process (what used to be `python -m lsdub_engine.cli <step>`)."""
import contextlib
import io
import sys


def call(main, argv, quiet=False, name=None):
    """main(argv) in-process. quiet drops its stdout (stderr stays). A non-zero return / SystemExit raises RuntimeError,
    like the old subprocess check=True."""
    argv = [str(a) for a in argv]
    sink = io.StringIO() if quiet else None
    try:
        with (contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()):
            ret = main(argv)
    except SystemExit as e:
        ret = e.code
    if ret not in (None, 0):
        what = name or getattr(main, '__module__', 'step')
        print(f'{what} failed ({ret!r}): {" ".join(argv)}', file=sys.stderr)
        raise RuntimeError(f'{what} exited {ret!r}')
    return ret
