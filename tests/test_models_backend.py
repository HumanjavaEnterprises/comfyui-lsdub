# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The face-model fetcher's failure messages (no network used), the local ComfyUI graph submit with the HTTP calls
stubbed, and the in-process step runner."""
import hashlib
import json
import os
import urllib.error

import pytest

from lsdub_engine import _inproc, _models
from lsdub_engine.backends import comfy as CF, comfy_backend as CB


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(_models, 'CACHE', str(tmp_path))
    monkeypatch.delenv('LIPS_YUNET', raising=False)
    monkeypatch.delenv('LIPS_FACE_MODEL', raising=False)
    return tmp_path


def test_models_no_network_says_where_to_place_the_file(cache, monkeypatch):
    def fail(url, dst): raise urllib.error.URLError('no route to host')
    monkeypatch.setattr(_models.urllib.request, 'urlretrieve', fail)
    with pytest.raises(RuntimeError) as e:
        _models.fetch_yunet()
    msg = str(e.value)
    assert 'could not download face_detection_yunet_2023mar.onnx' in msg and _models.YUNET_URL in msg
    assert str(cache / 'face_detection_yunet_2023mar.onnx') in msg and not os.listdir(cache)


def test_models_checksum_mismatch_is_refused(cache, monkeypatch):
    monkeypatch.setattr(_models.urllib.request, 'urlretrieve', lambda url, dst: open(dst, 'wb').write(b'not the model'))
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        _models.fetch_face_landmarker()
    assert not os.listdir(cache)                                                     # nothing half-written is kept


def test_models_good_download_and_reuse(cache, monkeypatch):
    body = b'model bytes'
    monkeypatch.setattr(_models, 'YUNET_SHA256', hashlib.sha256(body).hexdigest())
    n = []
    monkeypatch.setattr(_models.urllib.request, 'urlretrieve', lambda url, dst: (n.append(url), open(dst, 'wb').write(body)))
    p = _models.fetch_yunet()
    assert open(p, 'rb').read() == body and _models.fetch_yunet() == p and len(n) == 1   # the second call reuses it
    monkeypatch.setenv('LIPS_YUNET', '/elsewhere/y.onnx')
    assert _models.yunet_path() == '/elsewhere/y.onnx'


def test_local_graph_submit(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(CF, 'upload', lambda url, path: 'srv_' + os.path.basename(path))

    def queue(url, graph, client_id='lsdub'):
        seen['graph'] = graph
        return 'pid1'
    monkeypatch.setattr(CF, 'queue', queue)
    monkeypatch.setattr(CF, 'wait', lambda url, pid, timeout=900: {'outputs': {'9': {'images': [
        {'filename': 'a.png', 'subfolder': 'lsdub', 'type': 'output'}], 'text': ['not a file']}}})
    monkeypatch.setattr(CF, 'fetch', lambda url, row: b'png:' + row['filename'].encode())
    graph = {'1': {'class_type': 'LoadImage', 'inputs': {'image': '{UPLOAD_0}'}},
             '2': {'class_type': 'X', 'inputs': {'text': 'keep {OTHER}', 'n': 3}}}
    paths = CB._local_run('http://x', graph, '9', str(tmp_path), [str(tmp_path / 'face.png')])
    assert seen['graph']['1']['inputs']['image'] == 'srv_face.png' and seen['graph']['2']['inputs']['text'] == 'keep {OTHER}'
    assert paths == [str(tmp_path / 'a.png')] and open(paths[0], 'rb').read() == b'png:a.png'


def test_wait_raises_the_execution_error(monkeypatch):
    hist = {'p': {'status': {'status_str': 'error', 'messages': [['execution_error', {
        'node_type': 'KSampler', 'exception_type': 'OOM', 'exception_message': 'out of memory'}]]}}}
    monkeypatch.setattr(CF, '_call', lambda url, path, data=None, ctype=None, timeout=60: json.dumps(hist).encode())
    with pytest.raises(RuntimeError, match='KSampler OOM: out of memory'):
        CF.wait('http://x', 'p', timeout=5, poll=0)
    assert CB.resolve_backend(None) in ('local', os.environ.get('LSDUB_BACKEND', 'local').lower())
    assert CB._substitute({'a': ['{seed}', 1]}, {'SEED': 7}) == {'a': ['7', 1]}


def test_inproc_call():
    out = []

    def ok(argv):
        out.append(argv)
        print('noise')
    assert _inproc.call(ok, [1, 'x'], quiet=True) is None and out == [['1', 'x']]   # args become strings
    with pytest.raises(RuntimeError, match='step exited 3'):
        _inproc.call(lambda a: 3, [], name='step')
    with pytest.raises(RuntimeError, match='exited 2'):
        _inproc.call(lambda a: (_ for _ in ()).throw(SystemExit(2)), [])
    assert _inproc.call(lambda a: (_ for _ in ()).throw(SystemExit(0)), []) == 0
