# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The two unpackers (the Lip Kit zip in media.py, the demo zip in scripts/fetch_demo.py) write each member to its checked
path and call no extractall: the Registry's scanner flagged the pack for path traversal."""
import ast
import importlib.util
import os
import zipfile

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fetch():
    spec = importlib.util.spec_from_file_location('fetch_demo', os.path.join(PACK, 'scripts', 'fetch_demo.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('rel', ['media.py', os.path.join('scripts', 'fetch_demo.py')])
def test_no_extractall(rel):
    tree = ast.parse(open(os.path.join(PACK, rel), encoding='utf-8').read())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'extractall' not in calls and 'extract' not in calls


def test_fetch_demo_unpacks_a_good_zip(tmp_path):
    z = tmp_path / 'ok.zip'
    with zipfile.ZipFile(z, 'w') as zf:
        zf.writestr('lsdub-demo-1/a.txt', 'a')
        zf.writestr('lsdub-demo-1/kits/k/b.txt', 'b')
    out = tmp_path / 'out'
    with zipfile.ZipFile(z) as zf: _fetch().unpack(zf, str(out))
    assert (out / 'lsdub-demo-1' / 'kits' / 'k' / 'b.txt').read_text() == 'b'


@pytest.mark.parametrize('name', ['../escape.txt', 'a/../../escape.txt', '/abs.txt', 'C:/x.txt', 'a\\..\\x.txt'])
def test_fetch_demo_refuses_an_escaping_member(tmp_path, name):
    z = tmp_path / 'bad.zip'
    with zipfile.ZipFile(z, 'w') as zf: zf.writestr(name, 'x')
    with zipfile.ZipFile(z) as zf, pytest.raises(SystemExit): _fetch().unpack(zf, str(tmp_path / 'out'))
    assert not (tmp_path / 'escape.txt').exists()
