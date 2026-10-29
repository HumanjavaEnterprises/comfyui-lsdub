# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Generator slots and backends, without a GPU or the cloud: the graphs are well-formed, the slots call the generator
with the project's backend and cache the result, and the Comfy Cloud path drives a mocked SDK client (no real calls)."""
import asyncio
import os

import numpy as np
import pytest


def _proj(pack, tmp_path, **kw):
    return pack.project.Project(root=str(tmp_path), name='cut', **kw).ensure()


def _links_ok(g):
    for nid, n in g.items():
        for v in n['inputs'].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, (nid, v)
    assert 'out' in g


def test_graphs_are_wired(pack):
    G = pack.graphs
    for g in (G.song('pop', 'la la', 30, 1), G.still('@img', 'smile', 1, ref='@ref'), G.video('@img', 'walks', '', 5, 1)):
        _links_ok(g)
    assert G.video_frames(5, 25) == 121 and (G.video_frames(5, 25) - 1) % 8 == 0
    assert G.song('pop', 'x', 30, 7)['pos']['inputs']['seed'] == 7


def _fake_run(calls, kind):
    async def run(graph, backend='local', uploads=None, out_dir='.', api_key=None, out_node='out'):
        calls.append(dict(graph=graph, backend=backend, uploads=uploads, api_key=api_key))
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, 'gen.' + kind)
        if kind == 'flac':
            pack_ff = __import__('subprocess')
            pack_ff.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=f=330:d=1', p], check=True)
        elif kind == 'png':
            import cv2
            cv2.imwrite(p, np.full((64, 96, 3), 128, np.uint8))
        return [p]
    return run


def test_song_slot_generates_once_per_setting(pack, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pack.generate, 'run', _fake_run(calls, 'flac'))
    p = _proj(pack, tmp_path, seed=5, backend='comfy_cloud')
    A = pack.nodes_audio.LSDubSongSlot
    s1, audio = asyncio.run(A.execute(p, None, tags='synthpop, female vocal', seconds=10.0)).args
    assert os.path.exists(s1.mix) and calls[0]['backend'] == 'comfy_cloud'
    assert calls[0]['graph']['pos']['inputs']['seed'] == 5 and '[Instrumental]' in calls[0]['graph']['pos']['inputs']['lyrics']
    s2 = asyncio.run(A.execute(p, None, tags='synthpop, female vocal', seconds=10.0)).args[0]
    assert s2.mix == s1.mix and len(calls) == 1                                       # cached
    asyncio.run(A.execute(p, None, tags='synthpop, male vocal', seconds=10.0))
    assert len(calls) == 2


def test_still_slot_passthrough_and_edit(pack, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pack.generate, 'run', _fake_run(calls, 'png'))
    p = _proj(pack, tmp_path)
    img = np.zeros((1, 32, 48, 3), np.float32)
    S = pack.nodes_video.LSDubStillSlot
    assert asyncio.run(S.execute(p, img, '')).args[0].shape == (1, 32, 48, 3) and not calls
    out = asyncio.run(S.execute(p, img, 'add a microphone')).args[0]
    assert out.shape[1:3] == (64, 96) and set(calls[0]['uploads']) == {'img'}
    assert calls[0]['graph']['pos']['inputs']['prompt'] == 'add a microphone'


def test_video_slot_needs_a_still_to_generate(pack, tmp_path):
    p = _proj(pack, tmp_path)
    with pytest.raises(ValueError, match='connect a still'):
        asyncio.run(pack.nodes_video.LSDubVideoSlot.execute(p, None, None, prompt='a singer on a stage'))


def test_cloud_backend_uses_the_sdk_and_never_logs_the_key(pack, tmp_path, capsys):
    G = pack.generate
    log = []

    class Out:
        name = 'song_00001.flac'
        def to_file(self, path): open(path, 'wb').write(b'x')

    class Job:
        def get_outputs(self, nid):
            log.append(('outputs', nid))
            return [Out()]

    class WF:
        def __init__(self, g): self.g = g
        def set_input(self, nid, field, v): log.append(('set', nid, field, v))

    class Client:
        def __init__(self, api_key): log.append(('key', api_key == 'sk-secret'))
        workflows = type('W', (), {'from_json': staticmethod(lambda g: WF(g))})()
        assets = type('A', (), {'from_file': staticmethod(lambda p: f'asset:{os.path.basename(p)}')})()
        def run(self, wf):
            log.append(('run', sorted(wf.g)))
            return Job()
        def close(self): log.append(('close',))

    img = tmp_path / 'in.png'
    img.write_bytes(b'png')
    paths = G.run_cloud(pack.graphs.still('@img', 'smile', 1), 'out', {'img': str(img)}, str(tmp_path / 'o'), 'sk-secret',
                        client_factory=Client)
    assert os.path.exists(paths[0]) and ('key', True) in log and ('set', 'load', 'image', 'asset:in.png') in log
    assert log[-1] == ('close',)
    assert 'sk-secret' not in capsys.readouterr().out


def test_cloud_backend_without_a_key(pack, tmp_path, monkeypatch):
    monkeypatch.delenv('COMFY_API_KEY', raising=False)
    with pytest.raises(pack.generate.GenerationError, match='comfy.org API key'):
        pack.generate.run_cloud({'out': {'class_type': 'X', 'inputs': {}}}, client_factory=object)


def test_project_backend_setting(pack, tmp_path):
    N = pack.nodes_project.LSDubProjectNode
    kw = dict(name='cut', root=str(tmp_path), mode='run', fps=25.0, song_offset=0.0, preset='none', seed=0, force_shots='')
    p = N.execute(**kw, backend='comfy_cloud').args[0]
    assert p.backend == 'comfy_cloud' and N.execute(**kw).args[0].backend == 'local'
    assert N.fingerprint_inputs(**kw, backend='comfy_cloud') != N.fingerprint_inputs(**kw, backend='local')
    assert p.cache_key() == N.execute(**kw).args[0].cache_key()                     # analysis is shared across backends


def test_cloud_free_tier_refusal_in_plain_words(pack, tmp_path):
    G = pack.generate

    class Forbidden(Exception):
        pass

    class Client:
        def __init__(self, api_key): pass
        workflows = type('W', (), {'from_json': staticmethod(lambda g: type('WF', (), {'set_input': lambda *a: None})())})()

        def run(self, wf): raise Forbidden('403: API key authentication is not available for free tier accounts')

        def close(self): pass
    with pytest.raises(G.GenerationError, match='paid Comfy Cloud plan') as e:
        G.run_cloud({'out': {'class_type': 'X', 'inputs': {}}}, 'out', None, str(tmp_path), 'sk-secret', client_factory=Client)
    assert 'sk-secret' not in str(e.value)


def test_performance_covers_the_song(pack, tmp_path, monkeypatch, clip):
    import shutil
    import subprocess
    calls = []

    async def run(graph, backend='local', uploads=None, out_dir='.', api_key=None, out_node='out'):
        calls.append((graph['noise']['inputs']['noise_seed'], uploads['img']))
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, 'gen.mp4')
        shutil.copyfile(clip, p)                                         # a 2 s, 50-frame stand-in clip
        return [p]
    monkeypatch.setattr(pack.generate, 'run', run)
    wav = str(tmp_path / 'song.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=f=220:d=4.5', wav], check=True)
    p = pack.project.Project(root=str(tmp_path), name='cut', fps=25.0).ensure()
    song = pack.lsdub_types.Song(mix=wav, hash='s')
    img = np.zeros((1, 64, 96, 3), np.float32)
    P = pack.nodes_video.LSDubPerformance
    src = asyncio.run(P.execute(p, img, song, 'sings', '', 2.0, 0.0, 'same_still')).args[0]
    assert len(calls) == 3 and len({c[0] for c in calls}) == 3 and len({c[1] for c in calls}) == 1   # 3 shots, 3 seeds, one still
    assert abs(src.n_frames - 112) <= 1                                   # 4.5 s at 25 fps
    calls.clear()
    asyncio.run(P.execute(p, img, song, 'sings', '', 2.0, 0.0, 'last_frame'))
    assert len({c[1] for c in calls}) == 3                                # each shot starts from the previous one's end


def test_cloud_model_names(pack):
    G = pack.graphs
    g = G.still('@img', 'x', 1)
    assert G.for_backend(g, 'local') is g
    c = G.for_backend(g, 'comfy_cloud')
    assert c['lora']['inputs']['lora_name'].endswith('-fp32.safetensors') and g['lora']['inputs']['lora_name'].endswith('-bf16.safetensors')


def _perf_setup(pack, tmp_path, monkeypatch, song_s, clip_frames=121):
    import shutil
    import subprocess
    clip = str(tmp_path / f'clip{clip_frames}.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'testsrc=size=96x64:rate=25', '-frames:v', str(clip_frames),
                    '-pix_fmt', 'yuv420p', clip], check=True)
    calls = []

    async def run(graph, backend='local', uploads=None, out_dir='.', api_key=None, out_node='out'):
        calls.append(graph['noise']['inputs']['noise_seed'])
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, 'gen.mp4')
        shutil.copyfile(clip, p)
        return [p]
    monkeypatch.setattr(pack.generate, 'run', run)
    wav = str(tmp_path / f'song{song_s}.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'sine=f=220:d={song_s}', wav], check=True)
    return calls, pack.lsdub_types.Song(mix=wav, hash=f's{song_s}'), np.zeros((1, 64, 96, 3), np.float32)


def test_performance_keeps_generating_until_the_song_is_covered(pack, tmp_path, monkeypatch):
    calls, song, img = _perf_setup(pack, tmp_path, monkeypatch, 10)
    p = _proj(pack, tmp_path, fps=25.0)
    src = asyncio.run(pack.nodes_video.LSDubPerformance.execute(p, img, song, 'sings', '', 5.0, 0.0, 'same_still')).args[0]
    assert len(calls) == 3                                                # 2 x 121 frames < 250: a third shot, no frozen tail
    assert abs(src.n_frames - 250) <= 1
    F, _ = pack.lsdub_engine.core.medium.read(src.path)
    tail = [float(np.abs(F[k].astype(int) - F[k - 1].astype(int)).mean()) for k in range(len(F) - 8, len(F))]
    assert min(tail) > 0.5                                                # the last frames still move


def test_performance_song_shorter_than_one_shot(pack, tmp_path, monkeypatch):
    calls, song, img = _perf_setup(pack, tmp_path, monkeypatch, 3)
    p = _proj(pack, tmp_path, fps=25.0)
    src = asyncio.run(pack.nodes_video.LSDubPerformance.execute(p, img, song, 'sings', '', 5.0, 0.0, 'same_still')).args[0]
    assert len(calls) == 1 and abs(src.n_frames - 75) <= 1


def test_performance_song_longer_than_the_clips_and_max_seconds(pack, tmp_path, monkeypatch):
    calls, song, img = _perf_setup(pack, tmp_path, monkeypatch, 12, clip_frames=50)
    p = _proj(pack, tmp_path, fps=25.0)
    src = asyncio.run(pack.nodes_video.LSDubPerformance.execute(p, img, song, 'sings', '', 5.0, 0.0, 'same_still')).args[0]
    assert len(calls) == 6 and abs(src.n_frames - 300) <= 1                # 50-frame clips: 6 shots for 300 frames
    calls.clear()
    src = asyncio.run(pack.nodes_video.LSDubPerformance.execute(p, img, song, 'sings', '', 5.0, 4.0, 'same_still')).args[0]
    assert abs(src.n_frames - 100) <= 1                                   # max_seconds trims the song


def test_performance_refuses_a_song_offset(pack, tmp_path, monkeypatch):
    calls, song, img = _perf_setup(pack, tmp_path, monkeypatch, 3)
    p = _proj(pack, tmp_path, fps=25.0, song_offset=1.5)
    with pytest.raises(ValueError, match='song_offset'):
        asyncio.run(pack.nodes_video.LSDubPerformance.execute(p, img, song, 'sings', '', 5.0, 0.0, 'same_still'))
    assert calls == []


def test_performance_fingerprint_follows_the_project(pack, tmp_path):
    P = pack.nodes_video.LSDubPerformance
    song = pack.lsdub_types.Song(mix='m.wav', hash='s')
    img = np.zeros((1, 8, 8, 3), np.float32)
    a = P.fingerprint_inputs(_proj(pack, tmp_path, fps=25.0), img, song, 'x', '')
    b = P.fingerprint_inputs(pack.project.Project(root=str(tmp_path), name='other', fps=25.0).ensure(), img, song, 'x', '')
    assert a != b
