# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Socket payloads round-trip through JSON and hash stably."""
import json
from dataclasses import fields

import pytest


def _sample(cls):
    v = {'str': 'x', 'float': 1.5, 'int': 3, 'bool': True, 'tuple': [1, 2], 'dict': {'a': 1}, 'list': [{'id': 1}]}
    out = {}
    for f in fields(cls):
        t = f.type if isinstance(f.type, str) else getattr(f.type, '__name__', 'str')
        out[f.name] = v.get(t.split('[')[0].split('|')[0].strip(), 'x')
    return out


def test_payload_round_trip(pack):
    T = pack.lsdub_types
    for name, cls in T.PAYLOADS.items():
        if name == 'LSDUB_PROJECT': continue
        obj = cls(**_sample(cls))
        back = cls.from_json(json.loads(json.dumps(obj.to_json())))
        assert back.to_json() == json.loads(json.dumps(obj.to_json())), name
        assert obj.cache_key() == back.cache_key()


def test_project_round_trip_and_key(pack, tmp_path):
    P = pack.project.Project
    p = P(root=str(tmp_path), name='cut', mode='test_learn', fps=24.0, song_offset=12.0, seed=7, preset='landing', force_shots=(3, 21))
    q = P.from_json(json.loads(json.dumps(p.to_json())))
    assert q == p
    assert p.cache_key() == P(root='/elsewhere', name='cut', mode='test_learn', fps=24.0, song_offset=12.0, seed=7,
                              preset='landing').cache_key()        # force_shots and root do not move keys
    assert p.cache_key() != P(root=str(tmp_path), name='cut', fps=24.0).cache_key()


def test_project_rejects_bad_input(pack, tmp_path):
    P = pack.project.Project
    with pytest.raises(ValueError): P(root=str(tmp_path), name='x', mode='nope')
    with pytest.raises(ValueError): P(root=str(tmp_path), name='../up')


def test_custom_socket_types(pack):
    T = pack.lsdub_types
    assert set(T.PAYLOADS) == {f'LSDUB_{k}' for k in ('PROJECT', 'SOURCE', 'SONG', 'LYRICS', 'ALIGN', 'SHOTS', 'CAST', 'KIT',
                                                    'MASKS', 'TWEAKS', 'RENDER', 'REPORT')}
    assert T.LSDubProject.io_type == 'LSDUB_PROJECT'
