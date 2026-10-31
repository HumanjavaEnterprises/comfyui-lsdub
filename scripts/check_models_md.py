#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Every model file named in graphs.py's tables appears in MODELS.md (offline; exit 1 with the missing names)."""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = ('SONG_MODELS', 'STILL_MODELS', 'VIDEO_MODELS')
VARIANT = re.compile(r'(\.safetensors|-V1\.0|[-_](bf16|int8_convrot))+$')           # MODELS.md names a file without its variant suffix


def model_names():
    out = []
    for n in ast.parse(open(os.path.join(ROOT, 'graphs.py'), encoding='utf-8').read()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in TABLES:
            out += [k.value.value for k in n.value.keywords]
    return out


def missing(names=None, text=None):
    text = open(os.path.join(ROOT, 'MODELS.md'), encoding='utf-8').read() if text is None else text
    stems = lambda n: (n, VARIANT.sub('', n))
    return [n for n in (model_names() if names is None else names) if not any(s in text for s in stems(n))]


if __name__ == '__main__':
    gone = missing()
    if gone: print('not in MODELS.md:', *gone)
    sys.exit(1 if gone else 0)
