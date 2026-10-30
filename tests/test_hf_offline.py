# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""DINOv2 and Whisper load by local PATH when their files are in models/lsdub, so a worker with no network (or no
refs/main, which a build that places only snapshots/<rev>/ files never writes) starts without a hub call.
The real-weights check runs when LSDUB_HF_MODELS names a models/lsdub folder laid out as the build spec places it."""
import os
import subprocess
import sys
import textwrap

import pytest

from lsdub_engine._models import hf_snapshot
from lsdub_engine.core.lyric_align import whisper_source

DINO, WHISPER = 'facebook/dinov2-base', 'Systran/faster-whisper-large-v3'
REV = 'f9e44c814b77203eaa57a6bdbbd535f21ede1415'


def _files(d, names):
    os.makedirs(d, exist_ok=True)
    for n in names:
        with open(os.path.join(d, n), 'w') as f: f.write('{}')
    return str(d)


def test_snapshot_without_refs_main(tmp_path):
    s = _files(tmp_path / 'models--facebook--dinov2-base' / 'snapshots' / REV, ['config.json', 'model.safetensors'])
    assert hf_snapshot(str(tmp_path), DINO, ('config.json', 'model.safetensors')) == s
    assert hf_snapshot(str(tmp_path), DINO, ('config.json', 'missing.bin')) is None
    assert hf_snapshot(None, DINO) is None and hf_snapshot(str(tmp_path / 'nope'), DINO) is None


def test_refs_main_wins_and_flat_layout(tmp_path):
    root = tmp_path / 'models--facebook--dinov2-base'
    a = _files(root / 'snapshots' / 'aaaa', ['config.json'])
    _files(root / 'snapshots' / 'bbbb', ['config.json'])
    os.makedirs(root / 'refs')
    (root / 'refs' / 'main').write_text('aaaa\n')
    assert hf_snapshot(str(tmp_path), DINO) == a
    flat = _files(tmp_path / 'other' / 'dinov2-base', ['config.json'])
    assert hf_snapshot(str(tmp_path / 'other'), DINO) == flat


def test_whisper_name_maps_to_the_local_snapshot(tmp_path):
    s = _files(tmp_path / 'models--Systran--faster-whisper-large-v3' / 'snapshots' / 'edaa',
               ['model.bin', 'config.json', 'tokenizer.json'])
    assert whisper_source('large-v3', str(tmp_path)) == s
    assert whisper_source('large-v3', None) == 'large-v3'
    assert whisper_source('medium', str(tmp_path)) == 'medium'                      # not there: the name (downloads)


def test_embedder_loads_by_path(tmp_path, monkeypatch):
    s = _files(tmp_path / 'models--facebook--dinov2-base' / 'snapshots' / REV, ['config.json', 'model.safetensors'])
    import transformers as tf                         # declared in requirements.txt: must run, not skip
    seen = {}

    class _M:
        def eval(self): return self
        def to(self, d): return self
    monkeypatch.setattr(tf.AutoModel, 'from_pretrained', lambda src, **kw: seen.setdefault('src', src) and _M())
    from lsdub_engine.core import cast
    cast.TorchEmbedder('cpu', cache_dir=str(tmp_path))
    assert seen['src'] == s


MODELS = os.environ.get('LSDUB_HF_MODELS')


@pytest.mark.skipif(not MODELS, reason='set LSDUB_HF_MODELS to a models/lsdub folder laid out as the build spec places it')
def test_real_weights_load_offline():
    """a fresh interpreter, HF_HUB_OFFLINE=1 and an unreachable hub endpoint: both models load from models/lsdub."""
    code = textwrap.dedent(f'''
        import os
        from lsdub_engine.core import cast
        from lsdub_engine.core.lyric_align import whisper_source
        e = cast.TorchEmbedder('cpu', cache_dir=os.path.join({MODELS!r}, 'dinov2'))
        import numpy as np
        v = e([np.zeros((224, 224, 3), np.uint8)])
        assert v.shape == (1, 1536), v.shape
        from faster_whisper import WhisperModel
        src = whisper_source('large-v3', os.path.join({MODELS!r}, 'whisper'))
        assert os.path.isdir(src), src
        WhisperModel(src, device='cpu', compute_type='int8')
        print('OFFLINE-OK')
    ''')
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_ENDPOINT='http://127.0.0.1:9',
               HF_HOME=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'lsdub-empty-hf-home'))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=900,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert 'OFFLINE-OK' in r.stdout, r.stderr[-3000:]
