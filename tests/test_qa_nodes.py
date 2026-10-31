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
    assert rep.verdicts == {'3': 'REVIEW'} and md == '# verify\n'
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


# ------------------------------------------------------------------------------- QA step 0
def _rnd(pack, shots, cut='cut.mp4'):
    return pack.lsdub_types.Render(shots=shots, cut=cut, hash='r2')


def _capture_verify(pack, monkeypatch):
    mod = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.qa')
    seen = {}

    def verify(video, vocal, sj, before=None, align=None, model=None, work=None, emb=None):
        seen['shots'] = json.load(open(sj))['shots']
        return {}, [{'shot': 3, 'band': 'N/A', 'score': None, 'reason': 'left as source: sync gate failed: measured corr 0.1'}]

    def write_verify(meta, rows, out):
        json.dump({**meta, 'shots': rows}, open(out + '.json', 'w'))
        open(out + '.md', 'w').write('# verify\n')
    monkeypatch.setattr(mod, 'verify', verify)
    monkeypatch.setattr(mod, 'write_verify', write_verify)
    return seen


def test_verify_passes_the_left_shot_and_fixit_is_informational(pack, tmp_path, monkeypatch):
    seen = _capture_verify(pack, monkeypatch)
    p, src, shots, song, al, _ = _setup(pack, tmp_path)
    rnd = _rnd(pack, {'3': {'used': 'source', 'reason': 'sync gate failed: measured corr 0.1'}})
    rep, _ = pack.nodes_qa.LSDubVerify.execute(p, src, rnd, shots, song, al).args
    assert seen['shots'][0]['used'] == 'source' and 'sync gate' in seen['shots'][0]['left_reason']
    assert rep.verdicts == {'3': 'N/A'}
    md, force = pack.nodes_qa.LSDubFixIt.execute(p, shots, rep, None, 0.5).args
    assert force == '' and 'verify N/A' in md


def test_verify_passes_the_winning_warp_log(pack, tmp_path, monkeypatch):
    seen = _capture_verify(pack, monkeypatch)
    g = tmp_path / 'gate.json'
    g.write_text(json.dumps({'warp_json': '/w/native_warp.json'}))
    p, src, shots, song, al, _ = _setup(pack, tmp_path)
    pack.nodes_qa.LSDubVerify.execute(p, src, _rnd(pack, {'3': {'used': 'render', 'gate': str(g)}}), shots, song, al)
    assert seen['shots'][0]['warp_json'] == '/w/native_warp.json' and 'used' not in seen['shots'][0]


def test_scorecard_plate_mode_reaches_the_engine_and_the_key(pack, tmp_path, monkeypatch):
    mod = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.scorecard')
    calls = []

    def score(video, cuts, vocal, offset=12.0, align=None, script=None, cast=None, work=None, preset=None, **kw):
        calls.append(kw)
        return [{'cut': 0, 'shot': 3, 'class': 'PERF', 'score': 80}], {}

    def write(rows, thumbs, out, video, compared=False):
        json.dump({'cuts': rows}, open(out + '.json', 'w'))
        open(out + '.md', 'w').write('# scorecard\n')
    monkeypatch.setattr(mod, 'score', score)
    monkeypatch.setattr(mod, 'write', write)
    p, src, shots, song, al, rnd = _setup(pack, tmp_path)
    S = pack.nodes_qa.LSDubScorecard
    S.execute(p, src, shots, song, al, rnd)
    S.execute(p, src, shots, song, al, rnd, 'plate')
    assert calls == [{}, {'sharpness_mode': 'plate', 'plate': src.path}]
    assert S._key(p, src, shots, song, al, rnd) == S._key(p, src, shots, song, al, rnd, 'floor') != S._key(p, src, shots, song, al, rnd, 'plate')


def test_fixit_schema_and_execute_end_on_scorecard(pack, tmp_path):
    import inspect
    F = pack.nodes_qa.LSDubFixIt
    assert [i.id for i in F.define_schema().inputs][-2:] == ['scorecard', 'ruler']
    assert list(inspect.signature(F.execute).parameters)[-2:] == ['scorecard', 'ruler']
    T = pack.lsdub_types
    j = tmp_path / 's.json'
    j.write_text(json.dumps({'cuts': [{'shot': 3, 'score': 40, 'reason': 'static / near-still (diff 0.01) [0.10]'},
                                      {'shot': 4, 'score': 90, 'reason': 'soft image'}]}))
    p, _, shots, *_ = _setup(pack, tmp_path)
    md, force = F.execute(p, shots, scorecard=T.Report(json=str(j))).args
    assert 'a still playing as video' in md and '| 4 |' not in md and force == ''


def _reasons(pack):
    qa = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.qa')
    SC = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.scorecard')
    near = dict(frames=50, faces=1, px1080=400.0, present=1.0, yaw=0.0, mouth_visible=1.0)
    verdicts = [dict(frames=3), dict(frames=50), dict(frames=50, faces=1, largest={'who': 'x', 'px1080': 90}),
                {**near, 'present': 0.2}, {**near, 'yaw': 60.0}, {**near, 'mouth_visible': 0.1}, {**near, 'px1080': 60.0}]
    out = [qa.verdict(v, True)[1] for v in verdicts]
    assert all(qa.verdict(v, True)[0] == 'NO' for v in verdicts)
    out += list(SC.REASON.values())
    out += [SC.coverage_reason(dict(owner_px=0, tier='none')), SC.coverage_reason(dict(owner_px=40, tier='wide')),
            SC.coverage_reason(dict(owner_px=200, tier='medium', mouth_visible=0.1))]
    out += [qa.score_row(dict(voiced=False, verdict='DUB', motion=1.0, pops=0), None, SC)['reason']]
    r = qa.score_row(dict(voiced=True, verdict='DUB', corr=0.1, corr_pm1=0.1, owner_px=400.0, mouth_visible=1.0, motion=1.0, pops=0,
                          mouth_range=0.3, vocal_frac=0.5), None, SC)
    assert r['reason'].startswith('vocalisation spans cover 50%')
    out += [r['reason'], 'identity drift (cos 0.70); ', 'no kit for target', 'sync gate failed: measured corr 0.1', 'dub error: x']
    return out


def test_fixit_lists_ruler_notes_without_a_redo(pack, tmp_path):
    R = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.mouth_ruler')
    T = pack.lsdub_types
    default = pack.nodes_qa.fix_for('')
    names = [n for n, _ in R.NOTES] + ['grain lost']
    assert all(pack.nodes_qa.fix_for(n) != default for n in names)
    j = tmp_path / 'r.json'
    j.write_text(json.dumps({'shots': [{'shot': 3, 'notes': names}, {'shot': 4, 'note': 'left as source'}]}))
    p, _, shots, *_ = _setup(pack, tmp_path)
    md, force = pack.nodes_qa.LSDubFixIt.execute(p, shots, ruler=T.Report(json=str(j))).args
    assert force == '' and md.count('mouth ruler:') == len(names) and 'The inner mouth went black' in md and '| 4 |' not in md


def test_every_emitted_reason_has_a_fix(pack):
    default = pack.nodes_qa.fix_for('')
    missing = [x for x in _reasons(pack) if pack.nodes_qa.fix_for(x) == default]
    assert not missing


def test_the_vocalisation_fix_beats_the_sync_levers(pack):
    fix = pack.nodes_qa.fix_for('vocalisation spans cover 40% of the shot; sync gate failed: measured corr 0.2 (open in silence 0.1)')
    assert 'vocalisations to mouth' in fix


def test_mouth_ruler_node(pack, tmp_path, monkeypatch):
    R = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.mouth_ruler')
    calls = []

    def shot_row(take, render, stages=(), work=None, frames=None):
        calls.append((take, render, [s[0] for s in stages]))
        return {'pink_frac': [0.5, 0.1], 'cav_L50': [40, 8], 'teeth_L': [80, 60], 'edge_inner': [5, 9], 'boil_loose': [1, 2],
                'grain_gap': 0.5, 'stages': [], 'notes': ['grain lost']}
    monkeypatch.setattr(R, 'shot_row', shot_row)
    d = tmp_path / 'shot'
    (d / 'work').mkdir(parents=True)
    for f in ('take.mp4', 'sung.mp4', 'gate.json', 'render.mp4', 'work/crop.mp4'): (d / f).write_text('x')
    p, src, *_ = _setup(pack, tmp_path)
    rnd = _rnd(pack, {'3': {'used': 'render', 'render': str(d / 'render.mp4'), 'gate': str(d / 'gate.json'), 'f0': 0, 'f1': 9},
                      '4': {'used': 'source', 'render': '', 'f0': 10, 'f1': 19}})
    M = pack.nodes_qa.LSDubMouthRuler
    rep, md = M.execute(p, rnd, src).args
    assert calls == [(str(d / 'take.mp4'), str(d / 'render.mp4'), ['take', 'crop', 'sung', 'render'])]
    assert rep.verdicts == {'3': 'grain lost', '4': 'left as source'} and '0.50 → 0.10' in md
    M.execute(p, rnd, src)
    assert len(calls) == 1                                                            # cached


def _syncqa(pack, tmp_path, monkeypatch, rows):
    Q = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.sync_qa')
    monkeypatch.setattr(Q, 'score_film', lambda *a, **k: rows)
    monkeypatch.setattr(Q, 'film_md', lambda r: '# sync\n')
    mouth = pack.nodes_mouth
    cards = []
    monkeypatch.setattr(mouth, 'extract_shot', lambda v, f0, f1, dst: cards.append((f0, f1)) or dst)
    steps = importlib.import_module(pack.__name__ + '.lsdub_engine.core.steps')
    monkeypatch.setattr(steps, 'call', lambda *a, **k: None)
    monkeypatch.setattr(pack.media, 'placeholder_video', lambda path, text: path + ':' + text)
    monkeypatch.setattr(mouth.LSDubDub, '_video', staticmethod(lambda p: p))
    p, src, shots, song, al, rnd = _setup(pack, tmp_path)
    shots.shots.append({'shot': 4, 'f0': 50, 'f1': 99, 'start': 2.0, 'end': 4.0, 'verdict': 'DUB', 'reason': 'ok'})
    return pack.nodes_qa.LSDubSyncQA.execute(p, src, shots, song, al, rnd).args, cards


def test_sync_qa_with_no_rows_shows_the_placeholder(pack, tmp_path, monkeypatch):
    args, cards = _syncqa(pack, tmp_path, monkeypatch, [])
    assert cards == [] and args[1].endswith('no sung shot to show')


def test_sync_qa_cards_only_the_first_measured_shot(pack, tmp_path, monkeypatch):
    args, cards = _syncqa(pack, tmp_path, monkeypatch, [{'shot': 3, 'corr': None}, {'shot': 4, 'corr': 0.7}])
    assert cards == [(50, 99)] and args[0].verdicts == {'3': None, '4': 0.7}
