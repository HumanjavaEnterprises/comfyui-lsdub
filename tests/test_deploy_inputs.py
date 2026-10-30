# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Inputs that must travel with `comfy deploy run`: LSDub Align's transcript_json (a file) and LSDub Lip Kit's kit_dir
(a folder never travels; a .zip of the kit does). The deploy scanner uploads any string input with a file extension
that names a file under an asset root and the worker rewrites it to that file's name in its input folder; the nodes
must find the file by that name."""
import os
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    d = tmp_path / 'input'
    d.mkdir()
    monkeypatch.setattr(sys.modules['folder_paths'], 'get_input_directory', lambda: str(d), raising=False)
    return d


@pytest.fixture
def media_mod(pack):
    import importlib
    return importlib.import_module(pack.__name__ + '.media')


def _kit(root):
    k = root / 'red_nova'
    (k / 'cards').mkdir(parents=True)
    (k / 'library.json').write_text('{}')
    (k / 'base.png').write_bytes(b'png')
    (k / 'cards' / 'aa.png').write_bytes(b'aa')
    return k


def _zip(src, z, top=True):
    with zipfile.ZipFile(z, 'w') as zf:
        for p in sorted(src.rglob('*')):
            if p.is_file(): zf.write(p, os.path.relpath(p, src.parent if top else src))
    return z


def test_input_file_resolves_like_load_video(inputs, media_mod):
    (inputs / 'sub').mkdir()
    (inputs / 'a_stt.json').write_text('[]')
    (inputs / 'sub' / 'b_stt.json').write_text('[]')
    f = media_mod.input_file
    assert f('a_stt.json') == str(inputs / 'a_stt.json')
    assert f('a_stt.json [input]') == str(inputs / 'a_stt.json')
    assert f('sub/b_stt.json') == str(inputs / 'sub' / 'b_stt.json')       # the worker's `subfolder/name` form
    assert f('elsewhere/a_stt.json') == str(inputs / 'a_stt.json')         # uploaded by basename
    with pytest.raises(PermissionError, match='escapes'):
        f('../a_stt.json')
    assert f(str(inputs / 'a_stt.json')) == str(inputs / 'a_stt.json')
    assert f('') == ''
    with pytest.raises(FileNotFoundError, match='nope.json'):
        f('nope.json', 'LSDub Align: transcript_json')


def test_align_reads_the_transcript_from_the_input_folder(inputs, pack):
    import importlib
    na = importlib.import_module(pack.__name__ + '.nodes_audio')
    (inputs / 'lsdub_demo_landing_stt.json').write_text('[]')
    assert na._input_path('lsdub_demo_landing_stt.json') == str(inputs / 'lsdub_demo_landing_stt.json')
    with pytest.raises(FileNotFoundError, match='transcript_json'):
        na._input_path('missing_stt.json')


def test_kit_zip_unpacks_once_and_finds_the_kit(inputs, media_mod, tmp_path):
    k = _kit(tmp_path / 'src')
    _zip(k, inputs / 'red_nova.zip')
    d1 = media_mod.kit_folder('red_nova.zip')
    assert os.path.isfile(os.path.join(d1, 'library.json')) and os.path.isfile(os.path.join(d1, 'cards', 'aa.png'))
    assert d1.startswith(str(inputs / 'lsdub_kits_unpacked'))
    m = os.path.getmtime(os.path.join(d1, 'library.json'))
    assert media_mod.kit_folder('red_nova.zip') == d1 and os.path.getmtime(os.path.join(d1, 'library.json')) == m
    _zip(k, inputs / 'flat.zip', top=False)                                 # a zip of the kit's contents, no top folder
    assert os.path.isfile(os.path.join(media_mod.kit_folder('flat.zip'), 'library.json'))
    assert media_mod.kit_folder('lsdub_demo_kits/red_nova') == str(inputs / 'lsdub_demo_kits' / 'red_nova')   # folders unchanged


def test_kit_zip_refuses_escaping_members(inputs, media_mod):
    with zipfile.ZipFile(inputs / 'evil.zip', 'w') as zf:
        zf.writestr('../escape.txt', 'x')
    with pytest.raises(ValueError, match='escapes'):
        media_mod.kit_folder('evil.zip')
    assert not (inputs / 'escape.txt').exists()


def test_lip_kit_node_takes_a_zip(inputs, pack, tmp_path):
    import importlib
    nm = importlib.import_module(pack.__name__ + '.nodes_mouth')
    _zip(_kit(tmp_path / 'src'), inputs / 'kit.zip')
    proj = type('P', (), {})()
    fp = nm.LSDubBuildLipKit.fingerprint_inputs(proj, 'red', 'kit.zip')
    assert fp and fp != 'none'


def test_deploy_scanner_uploads_transcript_and_kit_zip(inputs, media_mod, tmp_path):
    """comfy-cli's own scan (what `comfy deploy run --asset-root` runs) on Align + Lip Kit inputs, then the worker's
    rewrite (core/ASSET -> the file's name in its input folder), then the nodes' lookup."""
    import comfy_cli.command.deploy_workflow as dw    # CI installs comfy-cli: this must run, not skip
    assets = tmp_path / 'assets'
    (assets / 'lsdub_demo_kits').mkdir(parents=True)
    (assets / 'lsdub_demo_landing_stt.json').write_text('[]')
    _zip(_kit(tmp_path / 'src'), assets / 'lsdub_demo_kits' / 'red_nova.zip')
    _kit(assets / 'lsdub_demo_kits')
    wf = {'9': {'class_type': 'LSDubAlign', 'inputs': {'stt_model': 'large-v3', 'transcript_json': 'lsdub_demo_landing_stt.json',
                                                       'pronunciations': '', 'project': ['1', 0], 'song': ['6', 0]}},
          '14': {'class_type': 'LSDubBuildLipKit', 'inputs': {'identity': 'red_nova', 'kit_dir': 'lsdub_demo_kits/red_nova.zip',
                                                              'project': ['1', 0]}},
          '15': {'class_type': 'LSDubBuildLipKit', 'inputs': {'identity': 'red_nova', 'kit_dir': 'lsdub_demo_kits/red_nova',
                                                              'project': ['1', 0]}}}
    plan = dw.scan_workflow_inputs(wf, asset_roots=(Path(assets).resolve(),))
    names = sorted(a.file_path for a in plan.assets)
    assert names == ['lsdub_demo_landing_stt.json', 'red_nova.zip']
    assert plan.workflow['15']['inputs']['kit_dir'] == 'lsdub_demo_kits/red_nova'        # a folder is never uploaded
    for a in plan.assets:                                                                  # the worker: upload into input/
        (inputs / a.file_path).write_bytes(a.local_path.read_bytes())
    t = plan.workflow['9']['inputs']['transcript_json']
    k = plan.workflow['14']['inputs']['kit_dir']
    assert t['__type'] == 'core/ASSET' and k['__type'] == 'core/ASSET'
    assert media_mod.input_file(t['info']['file_path']) == str(inputs / 'lsdub_demo_landing_stt.json')
    assert os.path.isfile(os.path.join(media_mod.kit_folder(k['info']['file_path']), 'library.json'))


def test_absolute_inputs_refused_inside_comfyui_unless_allowed(inputs, media_mod, tmp_path, monkeypatch):
    outside = tmp_path / 'secret.json'
    outside.write_text('[]')
    monkeypatch.delenv('LSDUB_ALLOW_ABS_INPUTS', raising=False)
    with pytest.raises(PermissionError, match='absolute'):
        media_mod.input_file(str(outside))
    with pytest.raises(PermissionError):
        media_mod.kit_folder(str(tmp_path))
    with pytest.raises(PermissionError, match='escapes'):
        media_mod.kit_folder('../somewhere')
    monkeypatch.setenv('LSDUB_ALLOW_ABS_INPUTS', '1')
    assert media_mod.input_file(str(outside)) == str(outside)


def test_absolute_inputs_ok_outside_comfyui(media_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules['folder_paths'], 'get_input_directory', lambda: None, raising=False)
    f = tmp_path / 'x.json'
    f.write_text('[]')
    assert media_mod.input_file(str(f)) == str(f)


def test_kit_zip_bomb_caps(inputs, media_mod, monkeypatch):
    with zipfile.ZipFile(inputs / 'bomb.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('library.json', b'\0' * (8 << 20))                    # 8 MB of zeros: ~1000:1
    with pytest.raises(ValueError, match='compresses'):
        media_mod.kit_folder('bomb.zip')
    with zipfile.ZipFile(inputs / 'many.zip', 'w') as zf:
        for i in range(30): zf.writestr(f'f{i}', 'x')
    monkeypatch.setattr(media_mod, 'ZIP_MAX_MEMBERS', 20)
    with pytest.raises(ValueError, match='members'):
        media_mod.kit_folder('many.zip')
    with zipfile.ZipFile(inputs / 'big.zip', 'w') as zf:
        zf.writestr('library.json', b'x' * 4096)
    monkeypatch.setattr(media_mod, 'ZIP_MAX_BYTES', 1024)
    with pytest.raises(ValueError, match='unpacks'):
        media_mod.kit_folder('big.zip')
    with zipfile.ZipFile(inputs / 'win.zip', 'w') as zf:
        zf.writestr('..\\evil.txt', 'x')
    with pytest.raises(ValueError, match='escapes'):
        media_mod.kit_folder('win.zip')


def test_vocal_split_reads_models_audio_separation(pack):
    """the spec places MelBandRoformer_kim_vocals.ckpt under models/audio_separation; the Vocal Split node's model
    list and path come from that folder."""
    import importlib
    import folder_paths
    importlib.import_module(pack.__name__ + '.nodes_vocal')
    assert os.path.join(folder_paths.models_dir, 'audio_separation') in folder_paths._extra['audio_separation']
    nv = sys.modules[pack.__name__ + '.nodes_vocal']
    assert "folder_paths.get_full_path('audio_separation', model)" in open(nv.__file__).read()
    assert "get_filename_list('audio_separation')" in open(nv.__file__).read()
