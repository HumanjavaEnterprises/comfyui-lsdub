# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub_engine/ is build output: VERSION hashes the tree that was written, and any hand edit breaks the hash."""
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = os.path.join(ROOT, 'lsdub_engine')


def _version():
    return dict(ln.split(': ', 1) for ln in open(os.path.join(ENG, 'VERSION'), encoding='utf-8').read().splitlines())


def test_version_is_clean_and_names_no_commit():
    v = _version()
    assert 'source_head' not in v and v['uncommitted'] == '0' and v['pack_version']
    assert re.fullmatch(r'[0-9a-f]{64}', v['tree_sha256'])


def test_tree_hash_matches():
    h = hashlib.sha256()
    for d, ds, fs in sorted(os.walk(ENG)):
        for f in sorted(fs):
            if f != 'VERSION' and '__pycache__' not in d:
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, ENG).encode() + b'\0')
                h.update(open(p, 'rb').read())
    assert h.hexdigest() == _version()['tree_sha256']
