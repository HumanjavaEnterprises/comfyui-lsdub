# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""a new demo zip takes a new version, tag and hash; the released one is left alone."""
import importlib.util
import os
import zipfile

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(PACK, 'scripts', f'{name}.py'))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def demo_dir(tmp_path):
    d = tmp_path / 'demo'
    (d / 'kits' / 'a').mkdir(parents=True)
    (d / 'b.txt').write_text('b')
    (d / 'kits' / 'a' / 'c.txt').write_text('c')
    return str(d)


def test_the_build_is_the_same_bytes_each_time_under_the_versions_root(tmp_path):
    B = load('build_demo_zip')
    a, b = str(tmp_path / 'a.zip'), str(tmp_path / 'b.zip')
    src = demo_dir(tmp_path)
    B.build(src, '1.2.3', a)
    os.utime(os.path.join(src, 'b.txt'), (1, 1))
    B.build(src, '1.2.3', b)
    assert open(a, 'rb').read() == open(b, 'rb').read()
    assert zipfile.ZipFile(a).namelist() == ['lsdub-demo-1.2.3/b.txt', 'lsdub-demo-1.2.3/kits/a/c.txt']


def test_the_build_refuses_a_bad_version_or_folder(tmp_path):
    B = load('build_demo_zip')
    with pytest.raises(SystemExit):
        B.build(demo_dir(tmp_path), 'one', str(tmp_path / 'a.zip'))
    with pytest.raises(SystemExit):
        B.build(str(tmp_path / 'none'), '1.2.3', str(tmp_path / 'a.zip'))


def test_stamp_checks_the_root_the_version_and_the_tag(tmp_path):
    B, m = load('build_demo_zip'), load('stamp_demo')
    z = str(tmp_path / 'a.zip')
    B.build(demo_dir(tmp_path), '1.2.3', z)
    m.check(z, '1.2.3', 'v1.2.3')
    for a in ((z, '1.2.4', 'v1'), (z, 'x', 'v1'), (z, '1.2.3', 'bad tag'), (str(tmp_path / 'none.zip'), '1.2.3', 'v1')):
        with pytest.raises(SystemExit):
            m.check(*a)
    with pytest.raises(SystemExit):
        m.main(['only', 'two'])


def test_stamp_wants_the_kits(tmp_path):
    B, m = load('build_demo_zip'), load('stamp_demo')
    d = tmp_path / 'demo'
    d.mkdir()
    (d / 'b.txt').write_text('b')
    z = str(tmp_path / 'a.zip')
    B.build(str(d), '1.2.3', z)
    with pytest.raises(SystemExit, match='kits'):
        m.check(z, '1.2.3', 'v1')


def test_stamp_rewrites_every_name_in_the_readme():
    m = load('stamp_demo')
    t = 'Get `lsdub-demo-0.1.0.zip`, then `--zip lsdub-demo-0.1.0.zip`; lsdub-demo stays.'
    assert m.readme(t, '9.9.9') == 'Get `lsdub-demo-9.9.9.zip`, then `--zip lsdub-demo-9.9.9.zip`; lsdub-demo stays.'


def test_stamp_moves_version_tag_and_hash_and_the_url_follows():
    m = load('stamp_demo')
    with open(os.path.join(PACK, 'scripts', 'fetch_demo.py'), encoding='utf-8') as f:
        s = m.stamp(f.read(), '9.9.9', 'demo-x', 'ab' * 32)
    g = {}
    head = s[s.index('VERSION ='):s.index('def guess_comfyui')]
    exec(compile(head, 'head', 'exec'), g)  # noqa: S102
    assert g['VERSION'] == '9.9.9' and g['SHA256'] == 'ab' * 32
    assert g['URL'].endswith('/download/demo-x/lsdub-demo-9.9.9.zip')
