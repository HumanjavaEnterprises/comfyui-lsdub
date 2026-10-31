# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub_engine/ is build output: VERSION names the lsdub commit and hashes the tree it wrote. CI's vendor job rebuilds it."""
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = os.path.join(ROOT, 'lsdub_engine')


def _version():
    return dict(ln.split(': ', 1) for ln in open(os.path.join(ENG, 'VERSION'), encoding='utf-8').read().splitlines())


def test_version_names_a_clean_commit():
    v = _version()
    assert re.fullmatch(r'[0-9a-f]{40}', v['source_head']) and v['uncommitted'] == '0' and v['pack_version']


def test_tree_hash_matches():
    h = hashlib.sha256()
    for d, ds, fs in sorted(os.walk(ENG)):
        ds[:] = sorted(x for x in ds if x != '__pycache__')
        for f in sorted(fs):
            if f != 'VERSION':
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, ENG).encode() + b'\0')
                h.update(open(p, 'rb').read())
    assert h.hexdigest() == _version()['tree_sha256']
