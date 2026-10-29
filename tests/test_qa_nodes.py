# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Verify and Scorecard nodes on stubbed engine calls: inputs are passed through, reports parsed, results cached."""
import importlib
import json
import os


def _setup(pack, tmp_path):
    T = pack.lsdub_types
    p = pack.project.Project(root=str(tmp_path), name='cut', song_offset=12.0, preset='landing').ensure()
    src = T.Source(path=str(tmp_path / 'src.mp4'), hash='s1', fps=25.0)
    shots = T.Shots(shots=[{'shot': 3, 'f0': 0, 'f1': 49, 'start': 0.0, 'end': 2.0, 'verdict': 'DUB', 'reason': 'ok'}], hash='h1')
    song = T.Song(mix='mix.wav', lead='lead.wav', hash='sg')
    al = T.Align(align='align.json', hash='al')
    rnd = T.Render(shots={}, cut=str(tmp_path / 'cut.mp4'), hash='r1')
    return p, src, shots, song, al, rnd


def test_verify_node(pack, tmp_path, monkeypatch):
    calls = []

    def verify(video, vocal, sj, before=None, align=None, model=None, work=None, emb=None):
        calls.append((video, vocal, before, align, json.load(open(sj))['offset']))
        return {'video': video}, [{'shot': 3, 'band': 'REVIEW', 'score': 60.0, 'reason': 'mouth open in the silences'}]

    def write_verify(meta, rows, out):
        json.dump({**meta, 'shots': rows}, open(out + '.json', 'w'))
        open(out + '.md', 'w').write('# verify\n')
    mod = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.qa')
    monkeypatch.setattr(mod, 'verify', verify)
    monkeypatch.setattr(mod, 'write_verify', write_verify)
    p, src, shots, song, al, rnd = _setup(pack, tmp_path)
    V = pack.nodes_qa.LSDubVerify
    rep, md = V.execute(p, src, rnd, shots, song, al).args
    assert calls == [(rnd.cut, 'lead.wav', src.path, 'align.json', 12.0)]
    assert rep.verdicts == {'3': 'REVIEW'} and rep.fix == [3] and md == '# verify\n'
    V.execute(p, src, rnd, shots, song, al)
    assert len(calls) == 1                                                            # cached
    fmd, force = pack.nodes_qa.LSDubFixIt.execute(p, shots, rep, None, 0.5).args
    assert force == '3' and 'word_close' in fmd


def test_scorecard_node(pack, tmp_path, monkeypatch):
    calls = []
    mod = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.scorecard')

    def score(video, cuts, vocal, offset=12.0, align=None, script=None, cast=None, work=None, preset=None):
        calls.append((video, cuts, vocal, offset, preset))
        return [{'cut': 0, 'shot': 3, 'class': 'SUNG', 'score': 80}], {}

    def write(rows, thumbs, out, video, compared=False):
        json.dump({'cuts': rows}, open(out + '.json', 'w'))
        open(out + '.md', 'w').write('# scorecard\n')
    monkeypatch.setattr(mod, 'score', score)
    monkeypatch.setattr(mod, 'write', write)
    p, src, shots, song, al, rnd = _setup(pack, tmp_path)
    S = pack.nodes_qa.LSDubScorecard
    rep, md = S.execute(p, src, shots, song, al, None).args                           # test_learn style: the source
    assert calls[0][0] == src.path and calls[0][1] == {'cut': [{'start': 0.0, 'dur': 2.0, 'shot': 3}]}
    assert calls[0][3] == 12.0 and calls[0][4] == 'landing' and rep.verdicts == {'3': 'SUNG'} and md.startswith('# score')
    S.execute(p, src, shots, song, al, rnd)
    assert calls[1][0] == rnd.cut and len(calls) == 2 and os.path.exists(rep.json)
