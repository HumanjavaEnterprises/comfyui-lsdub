# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""a new demo zip takes a new version, tag and hash; the released one is left alone."""
import importlib.util
import os

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_stamp_moves_version_tag_and_hash_and_the_url_follows():
    sp = importlib.util.spec_from_file_location('stamp_demo', os.path.join(PACK, 'scripts', 'stamp_demo.py'))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    with open(os.path.join(PACK, 'scripts', 'fetch_demo.py'), encoding='utf-8') as f:
        s = m.stamp(f.read(), '9.9.9', 'demo-x', 'ab' * 32)
    g = {}
    head = s[s.index('VERSION ='):s.index('def guess_comfyui')]
    exec(compile(head, 'head', 'exec'), g)  # noqa: S102
    assert g['VERSION'] == '9.9.9' and g['SHA256'] == 'ab' * 32
    assert g['URL'].endswith('/download/demo-x/lsdub-demo-9.9.9.zip')
