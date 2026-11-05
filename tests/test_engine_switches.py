# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The three dub switches on LSDub Mouth Tweaks (warp_shape, travel_guard, coherence_charge) and coherence_charge on LSDub
Continuity: each reaches its engine call; at their defaults the Tweaks hash is the one before the switches; set to their
second choice they make the engine calls and flags the 0.1.9 pack made; and a project dubbed by an older engine never
reuses its shots."""
import importlib
import json
import os
import types

import pytest

BASE = (0.06, 0.06, 512.0, True, True, True, '')
OLD = ('off', 'take', 'all')                       # the 0.1.9 behaviour
SNAP_019 = ['--warp-sigma', '0.8', '--warp-temporal', '0.6', '--warp-close-depth', '0.05', '--warp-follow', '0.15']


def _engine(pack, name):
    return importlib.import_module(f'{pack.__name__}.lsdub_engine.core.{name}')


def _tweaks(pack, *switches):
    return pack.nodes_mouth.LSDubTweaks.execute(*BASE, '', 0.0, False, *switches).args[0]


def test_the_switches_are_combo_inputs_with_the_new_defaults(pack):
    from test_example_workflows import schemas
    S = schemas(pack)
    tw = {p.id: p for p in S['LSDubTweaks'].inputs}
    for name, options in (('warp_shape', ['asym', 'off']), ('travel_guard', ['aligner', 'take']), ('coherence_charge', ['remade', 'all'])):
        p = tw[name]
        assert p.io_type == 'COMBO' and p.optional and p.kw['options'] == options and p.kw['default'] == options[0]
        assert '0.1.9' in p.tooltip and '\n' not in p.tooltip
    p = {p.id: p for p in S['LSDubContinuity'].inputs}['coherence_charge']
    assert p.io_type == 'COMBO' and p.optional and p.kw['options'] == ['remade', 'all'] and p.kw['default'] == 'remade'


def test_the_defaults_keep_the_tweaks_hash_and_a_change_moves_it(pack):
    N = pack.nodes_mouth.LSDubTweaks
    before = N.execute(*BASE).args[0]
    assert _tweaks(pack, 'asym', 'aligner', 'remade').hash == before.hash
    assert _tweaks(pack, 'asym', 'aligner', 'remade').flags == before.flags
    hashes = {_tweaks(pack, *s).hash for s in (('off', 'aligner', 'remade'), ('asym', 'take', 'remade'), ('asym', 'aligner', 'all'), OLD)}
    assert len(hashes) == 4 and before.hash not in hashes
    with pytest.raises(ValueError, match='warp_shape'):
        _tweaks(pack, 'closures', 'aligner', 'remade')


def _run_shot(pack, tmp_path, monkeypatch, tw):
    """LSDubDub.run_shot with the engine's sing stubbed: what it hands sing.run and sing.repair."""
    M, SING = pack.nodes_mouth, _engine(pack, 'sing')
    calls = {}

    def run(take, align, vocal, t0, out, work, paint, **kw):
        calls['paint'], calls['run'] = list(paint), kw
        return {'pass': True}

    def repair(take, sung, out, info, work, **kw):
        calls['repair'] = kw
        return {}
    monkeypatch.setattr(SING, 'run', run)
    monkeypatch.setattr(SING, 'repair', repair)
    monkeypatch.setattr(M, 'extract_shot', lambda src, f0, f1, dst: dst)
    ns = types.SimpleNamespace
    st = ns(path=lambda n: str(tmp_path / n))
    project = pack.project.Project(root=str(tmp_path), name='cut')
    s = {'shot': 1, 'f0': 0, 'f1': 9, 'start': 0.0, 'end': 0.4, 'verdict': 'DUB', 'target': 'a', 'box': None}
    M.LSDubDub.run_shot(project, ns(path='src.mp4'), s, ns(lead='l.wav', mix='m.wav'), ns(align='a.json'),
                        ns(dir=str(tmp_path), identity='a'), tw, None, None, st)
    return calls


def test_the_defaults_reach_the_engine(pack, tmp_path, monkeypatch):
    SING = _engine(pack, 'sing')
    c = _run_shot(pack, tmp_path, monkeypatch, _tweaks(pack, 'asym', 'aligner', 'remade'))
    assert c['run']['travel_ref'] == 'aligner' and c['repair']['charge'] == 'remade'
    assert '--warp-shape' not in c['paint']                                     # sing's own asym stands
    assert SING.snap_paint(c['paint'], 2) == [*SNAP_019, '--warp-shape', 'asym']
    assert _run_shot(pack, tmp_path, monkeypatch, pack.nodes_mouth.DEFAULT_TWEAKS) == c   # no Tweaks node: the same


def test_the_0_1_9_values_make_the_0_1_9_calls(pack, tmp_path, monkeypatch):
    SING, MP = _engine(pack, 'sing'), _engine(pack, 'mouth_paint')
    new = _run_shot(pack, tmp_path, monkeypatch, _tweaks(pack, 'asym', 'aligner', 'remade'))
    old = _run_shot(pack, tmp_path, monkeypatch, _tweaks(pack, *OLD))
    assert old['run'] == {**new['run'], 'travel_ref': 'take'} and old['repair'] == {**new['repair'], 'charge': 'all'}
    i = old['paint'].index('--warp-shape')
    assert old['paint'][:i] + old['paint'][i + 2:] == new['paint'] and old['paint'][i + 1] == 'off'
    assert MP.paint_warnings(['--mode', 'warp']) == MP.paint_warnings(['--mode', 'warp', '--warp-shape', 'off'])  # off is paint's default
    assert SING.snap_paint(old['paint'], 2) == SNAP_019                          # sing's paint flags as 0.1.9 had them


def test_the_engines_own_defaults_are_the_new_values(pack):
    import inspect
    SING = _engine(pack, 'sing')
    assert inspect.signature(SING.run).parameters['travel_ref'].default == 'aligner'
    assert inspect.signature(SING.repair).parameters['charge'].default == 'remade'
    assert inspect.signature(_engine(pack, 'continuity').run).parameters['charge'].default == 'remade'


def test_a_warp_shape_typed_in_settings_wins(pack, tmp_path, monkeypatch):
    tw = _tweaks(pack, 'off', 'aligner', 'remade')
    tw.settings = '--warp-shape closures'
    p = _run_shot(pack, tmp_path, monkeypatch, tw)['paint']
    assert p[len(p) - 1 - p[::-1].index('--warp-shape') + 1] == 'closures'


def test_continuity_takes_the_charge_and_keys_on_it_only_off_its_default(pack, tmp_path, monkeypatch):
    M, T, CONT = pack.nodes_mouth, pack.lsdub_types, _engine(pack, 'continuity')
    K = pack.project.stage_key
    seen = []

    def run(take, sung, out, **kw):
        seen.append(kw['charge'])
        open(out, 'w').write('r')
    monkeypatch.setattr(CONT, 'run', run)
    monkeypatch.setattr(M, 'cut_render', lambda project, source, shots, quality, audio, song: (T.Render(shots=shots, cut='c.mp4'), []))
    project = pack.project.Project(root=str(tmp_path), name='cut').ensure()
    (tmp_path / 'r.mp4').write_text('x')
    render = T.Render(shots={'1': {'render': str(tmp_path / 'r.mp4'), 'key': 'k', 'result': 'h'}}, cut='c.mp4', hash='rh')
    monkeypatch.setattr(M.LSDubDub, '_video', staticmethod(lambda p: p))
    for charge in ('remade', 'all'):
        M.LSDubContinuity.execute(project, None, render, coherence_charge=charge)
    assert seen == ['remade', 'all']
    fp = M.LSDubContinuity.fingerprint_inputs
    assert fp(project, None, render) == K('LSDubContinuity', {'ids': '', 'z': 3.5, 'q': 18}, ['rh'])   # the key before the option
    assert fp(project, None, render, coherence_charge='all') != fp(project, None, render)


def test_a_project_dubbed_by_an_older_engine_re_dubs(pack, monkeypatch):
    """the switches are absent from the Tweaks hash at their defaults, so the default tweaks of 0.1.9 and of this release hash the
    same: what keeps a 0.1.9 shot from standing is the dub's key, which carries the whole vendored engine's version."""
    P = pack.project
    assert P.ENGINE_VERSION.split('+')[1] != '0ac7dc5e0512'                      # 0.1.9's vendored tree
    ns = types.SimpleNamespace
    ck = ns(cache_key=lambda: 'ck')
    tw = _tweaks(pack, 'asym', 'aligner', 'remade')

    def key():
        return pack.nodes_mouth.LSDubDub._shot_key(P.Project(root='/r', name='cut'), ns(hash='src'),
                                                   dict(shot=1, f0=0, f1=9, start=0.0, target='a', box=None), ck, ns(hash='al'),
                                                   ns(hash='kh', identity='a'), tw, None, None)
    now = key()
    monkeypatch.setattr(P, 'ENGINE_VERSION', '0.1.9+0ac7dc5e0512')
    assert key() != now
    for node in ('LSDubDub/shot', 'LSDubContinuity', 'LSDubVerify'):            # the dub and after: the whole engine
        assert node.split('/')[0] not in pack.engine_keys.STAGES
    monkeypatch.undo()
    assert key() == now


def test_verify_says_what_it_reads_for_a_held_line(pack):
    from test_example_workflows import schemas
    s = schemas(pack)['LSDubVerify']
    assert 'travel the song asks for' in s.description
    md = next(o for o in s.outputs if o.display_name == 'md')
    assert 'asks for' in md.tooltip and 'share' in md.tooltip
    assert 'asked travel' in pack.nodes_qa.fix_for('mouth barely moves (held gape / mumble): delivered 0.59 of the travel the song asked')


def test_the_readme_names_the_switches():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, 'README.md'), encoding='utf-8').read()
    for name in ('warp_shape', 'travel_guard', 'coherence_charge'):
        assert f'`{name}`' in text
    wf = json.load(open(os.path.join(root, 'example_workflows', '2-LSDub-BYO-Clip-and-Song.json')))
    n = next(x for x in wf['nodes'] if x['type'] == 'LSDubTweaks')
    assert n['widgets_values'][-3:] == ['asym', 'aligner', 'remade']
