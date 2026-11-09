# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""What the docs generate or check: README and bug form from the schemas, MODELS.md, the model table against the build yaml,
the .comfyignore list, and the records the docs lean on."""
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_readme_and_bug_form_match_the_schemas():
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'make_readme_nodes.py'), '--check'], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr[-500:]


def test_models_md_names_every_graph_model():
    m = script('check_models_md')
    assert len(m.model_names()) == 12 and m.missing() == []
    assert m.missing(['gone.safetensors'], 'nothing') == ['gone.safetensors']


def test_models_known_agrees_with_the_build_yaml(pack):
    M = importlib.import_module(pack.__name__ + '.lsdub_engine._models')
    text = open(os.path.join(ROOT, 'comfy-build.yaml'), encoding='utf-8').read()
    seen = 0
    for name, sha, repo, rev in re.findall(r'filename: (\S+)\n\s+sha256: (\w+)\n\s+sourceUri: https://huggingface.co/([^/]+/[^/]+)/resolve/(\w+)/', text):
        if repo in M.KNOWN and rev != 'main':
            assert M.KNOWN[repo]['revision'] == rev and M.KNOWN[repo]['files'][name] == sha, name
            seen += 1
    assert seen == 11
    assert M.FACE_SHA256 in text and M.YUNET_SHA256 in text


def test_comfyignore_drops_the_dev_scripts():
    ig = open(os.path.join(ROOT, '.comfyignore')).read().split()
    assert {'scripts/make_readme_nodes.py', 'scripts/check_models_md.py', 'comfy-build.yaml'} <= set(ig)
    assert 'scripts/fetch_demo.py' not in ig and 'scripts/' not in ig


def test_records_round_trip_and_old_ones_load(pack):
    T = pack.lsdub_types
    for cls, d in ((T.Song, {'mix': 'm'}), (T.Align, {'align': 'a'}), (T.Cast, {}), (T.Report, {}), (T.Render, {})):
        rec = cls(**d)
        assert cls.from_json(json.loads(json.dumps(rec.to_json()))) == rec
    assert T.Song.from_json({'mix': 'm', 'vocal': 'v', 'inst': 'i', 'lead_activity': 'c'}) == T.Song(mix='m')
    assert T.Align.from_json({'align': 'a', 'visemes': 'x'}) == T.Align(align='a')
    assert T.Cast.from_json({'plan': 'p'}) == T.Cast()
    assert T.Report.from_json({'fix': [1]}) == T.Report()
    r = T.Render.from_json({'cut': 'c'})
    assert (r.audio, r.on_fail, r.song) == ('source', 'source', {})


def test_continuity_keeps_the_song(pack, tmp_path, monkeypatch):
    m, T = pack.nodes_mouth, pack.lsdub_types
    p = pack.project.Project(root=str(tmp_path), name='c').ensure()
    src = T.Source(path='src.mp4', hash='s')
    seen = []
    monkeypatch.setattr(m, 'assemble', lambda s, r, cut, fps, q, audio, song, off: seen.append((audio, song)) or open(cut, 'w').close() or [])
    monkeypatch.setattr(m.LSDubDub, '_video', staticmethod(lambda p: p))
    song = T.Song(mix='mix.wav', hash='h')
    rnd, _ = m.cut_render(p, src, {}, 18, 'song', song)
    rnd.on_fail = 'render'
    out = m.LSDubContinuity.execute(p, src, rnd).args[0]
    assert seen[-1] == ('song', 'mix.wav') and (out.audio, out.on_fail) == ('song', 'render')
    old = T.Render(shots={}, cut='c', hash='old')                                     # a record from before: the source's audio
    assert m.LSDubContinuity.execute(p, src, old).args[0].audio == 'source' and seen[-1] == ('source', None)


def test_video_output_needs_a_real_comfyui(pack, monkeypatch):
    import builtins
    import pytest
    real = builtins.__import__

    def no_latest(name, *a, **k):
        if name == 'comfy_api.latest': raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', no_latest)
    with pytest.raises(RuntimeError, match='0.3.48'):
        pack.nodes_mouth.LSDubDub._video('x.mp4')
