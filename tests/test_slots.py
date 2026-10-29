# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Slots: an asset is stored and keyed by its content (no generator call); prompt-only says generation comes later."""
import os

import numpy as np
import pytest


def _proj(pack, tmp_path):
    return pack.project.Project(root=str(tmp_path), name='cut').ensure()


def _audio(f=440.0, sr=16000):
    t = np.arange(sr // 2) / sr
    return {'waveform': np.sin(2 * np.pi * f * t).astype(np.float32)[None, None].repeat(2, 1), 'sample_rate': sr}


def test_song_slot_asset_is_stored_and_cached(pack, tmp_path):
    A = pack.nodes_audio.LSDubSongSlot
    p = _proj(pack, tmp_path)
    s1, a1 = A.execute(p, _audio()).args
    assert os.path.exists(s1.mix) and s1.offset == p.song_offset and a1['sample_rate'] == 16000
    s2 = A.execute(p, _audio()).args[0]
    assert s2.mix == s1.mix and s2.hash == s1.hash
    s3 = A.execute(p, _audio(f=220.0)).args[0]
    assert s3.mix != s1.mix and os.path.exists(s1.mix)
    assert A.fingerprint_inputs(p, _audio()) == A.fingerprint_inputs(p, _audio())


def test_prompt_only_slots_say_generation_comes_later(pack, tmp_path):
    p = _proj(pack, tmp_path)
    with pytest.raises(RuntimeError, match='later LSDub release'):
        pack.nodes_audio.LSDubSongSlot.execute(p, None, tags='disco, female vocal')
    with pytest.raises(RuntimeError, match='later LSDub release'):
        pack.nodes_video.LSDubVideoSlot.execute(p, None, None, prompt='a singer on a stage')
    with pytest.raises(ValueError):
        pack.nodes_audio.LSDubSongSlot.execute(p, None)
    with pytest.raises(ValueError):
        pack.nodes_video.LSDubVideoSlot.execute(p)


def test_video_slot_asset(pack, tmp_path, clip):
    V = pack.nodes_video.LSDubVideoSlot
    p = _proj(pack, tmp_path)

    class Vid:
        def get_stream_source(self): return clip
    src, out = V.execute(p, Vid()).args
    assert os.path.exists(src.path) and src.n_frames == 50 and src.wh == (640, 360) and abs(src.fps - 25) < 1e-6
    assert V.execute(p, Vid()).args[0].path == src.path


def test_shot_split_on_a_clip(pack, tmp_path, clip):
    V = pack.nodes_video
    p = _proj(pack, tmp_path)

    class Vid:
        def get_stream_source(self): return clip
    src = V.LSDubVideoSlot.execute(p, Vid()).args[0]
    shots = V.LSDubShotSplit.execute(p, src).args[0]
    assert shots.shots and shots.shots[0]['f0'] == 0 and os.path.exists(shots.features)
    assert V.load_detected(shots)[1] == pytest.approx(25.0)
