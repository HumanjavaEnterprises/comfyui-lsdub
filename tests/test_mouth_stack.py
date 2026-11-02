# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_stack and the one re-key release: plain shots keep the key they had, layers shots get their own, old projects load."""
import json
import os
import shutil
import sys
import types

from test_example_workflows import WIDGET, has_control, schemas

OLD_PROJECT = '''{
 "root": "/tmp/p", "name": "cut", "mode": "run", "fps": 25.0, "song_offset": 0.0, "vocalisations": "skip",
 "seed": 0, "preset": "none", "backend": "local", "force_shots": [21], "engine_version": "0.1.0"
}'''


def _proj(pack, **kw):
    return pack.project.Project(root='/r', name='cut', **kw)


def _key(pack, project, kit_id='a', vocalisations='skip'):
    ns = types.SimpleNamespace
    ck = ns(cache_key=lambda: 'ck')
    return pack.nodes_mouth.LSDubDub._shot_key(project, ns(hash='src'), dict(shot=1, f0=0, f1=9, start=0.0, target='a', box=None), ck,
                                               ns(hash='al'), ns(hash='kh', identity=kit_id), ck, None, None, vocalisations)


def test_widget_order(pack):
    s = schemas(pack)['LSDubProject']
    ordered = [p for p in s.inputs if not p.optional] + [p for p in s.inputs if p.optional]
    names = [k for p in ordered if p.io_type in WIDGET for k in ((p.id, 'fixed') if has_control(p) else (p.id,))]
    assert names == ['name', 'root', 'mode', 'fps', 'song_offset', 'preset', 'seed', 'fixed', 'force_shots', 'backend', 'mouth_stack']
    p = ordered[-1]
    assert p.kw['options'] == ['plain', 'layers', 'layers_interior', 'layers_shape'] and p.kw['default'] == 'plain' and p.optional


def test_project_key_snapshot(pack):
    assert _proj(pack).cache_key() == _proj(pack, mouth_stack='layers').cache_key()
    assert _proj(pack).cache_key() == pack.project.text_hash(pack.project.canonical(
        {'name': 'cut', 'fps': 25.0, 'song_offset': 0.0, 'preset': 'none'}))


def test_plain_shot_key_is_the_formula_without_the_option(pack):
    P = pack.project
    p = _proj(pack)
    want = P.stage_key('LSDubDub/shot', {'f': [0, 9], 'start': 0.0, 'target': 'a', 'box': None, 'occ': ''},
                       [p.cache_key(), 'src', 'ck', 'al', 'kh', 'ck', ''])
    assert _key(pack, p) == want
    assert _key(pack, p, kit_id='renamed') == want                                  # A1: identity is not in a plain key


def test_layers_shot_key_folds_stack_identity_recipe(pack):
    lay = _proj(pack, mouth_stack='layers')
    assert _key(pack, lay) != _key(pack, _proj(pack))
    assert _key(pack, lay) != _key(pack, lay, kit_id='b')


def test_literal_old_project_loads(pack, tmp_path):
    f = tmp_path / 'project.json'
    f.write_text(OLD_PROJECT)
    p = pack.project.Project.from_json(json.load(open(f)))
    assert p.force_shots == (21,) and p.mouth_stack == 'plain' and not hasattr(p, 'vocalisations')


def test_kit_hash_is_bytes_not_mtime(pack, tmp_path):
    a = tmp_path / 'a'
    a.mkdir()
    (a / 'x.npy').write_bytes(b'one')
    b = tmp_path / 'b'
    shutil.copytree(a, b)
    os.utime(b / 'x.npy', (1, 1))
    H = pack.nodes_mouth.LSDubBuildLipKit._hash
    assert H(str(a)) == H(str(b))
    (b / 'x.npy').write_bytes(b'two')
    assert H(str(a)) != H(str(b))


def test_engine_version_carries_the_tree(pack):
    tree = open(os.path.join(os.path.dirname(pack.project.__file__), 'lsdub_engine', 'VERSION')).read().split('tree_sha256: ')[1][:12]
    assert pack.project.ENGINE_VERSION.endswith('+' + tree)


def test_unload_closes_landmarkers(pack):
    import comfy.model_management as mm
    calls = []
    mm.unload_all_models = lambda: calls.append('unload')
    pack.devices.hook_unload()
    pack.devices.hook_unload()                                                       # once
    name = pack.__name__ + '.lsdub_engine.core.landmarker'
    real = sys.modules.get(name)
    sys.modules[name] = types.SimpleNamespace(close_all=lambda: calls.append('close'))
    try:
        mm.unload_all_models()
    finally:
        sys.modules.pop(name)
        if real: sys.modules[name] = real
    assert calls == ['unload', 'close']


def test_node_paint_warnings(pack, capfd):
    w = pack.nodes_mouth._paint_warnings(['--lib', 'k', '--mode', 'morph', '--warp-temporal', '0.3'])
    assert w and '--warp-' in w[0]
    assert pack.nodes_mouth._paint_warnings(['--not-a-flag']) == []


def test_unload_hook_without_the_attribute(pack):
    import comfy.model_management as mm
    keep = mm.__dict__.pop('unload_all_models', None)
    try:
        pack.devices.hook_unload()
        assert not hasattr(mm, 'unload_all_models')
    finally:
        if keep: mm.unload_all_models = keep


def _kit(identity='singer'):
    return types.SimpleNamespace(dir='KIT', identity=identity)


def _tw(settings=''):
    return types.SimpleNamespace(settings=settings)


def _flags(pack, stack='plain', preset='none', identity='singer', settings='', v='skip'):
    return pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack=stack, preset=preset), _kit(identity), _tw(settings), v)


def _landing(pack):
    return list(pack.nodes_mouth.presets.LANDING.flags_for('red_nova'))


def test_argv_snapshot_plain_and_landing(pack):
    plain = ['--lib', 'KIT', '--warp-track-mode', 'chain', '--vocalisations', 'skip']
    assert _flags(pack) == plain
    assert _flags(pack, preset='landing') == plain                                    # a face the preset does not list
    assert _flags(pack, preset='landing', identity='red_nova') == ['--lib', 'KIT', *_landing(pack), *plain[2:]]


def test_layers_on_any_kit(pack):
    R = pack.nodes_mouth.recipes.LAYERS_V16_1
    full = ['--lib', 'KIT', *R, '--warp-track-mode', 'chain', '--vocalisations', 'mouth']
    assert _flags(pack, 'layers', v='mouth') == full
    assert _flags(pack, 'layers', 'landing', v='mouth') == full
    assert '--warp-compose' in full and '--warp-lip-lock' not in _flags(pack, 'layers', 'landing', 'red_nova')


def test_settings_are_shell_quoted_and_can_change_vocalisations(pack):
    a = _flags(pack, settings='--vocalisations rest --warp-erase "a b"')
    assert a[-5:] == ['skip', '--vocalisations', 'rest', '--warp-erase', 'a b'] and a[-6] == '--vocalisations'


def test_two_kits_no_cast_names_each_dropped_shot(pack, capfd):
    D = pack.nodes_mouth.LSDubDub
    shots = types.SimpleNamespace(shots=[dict(shot=4, verdict='DUB', target='face', f0=0, f1=9, start=0.0)])
    kits = {'a': object(), 'b': object()}
    assert D._plan(None, shots, kits, []) == []
    msg = D._dropped(shots, kits, [])[4][1]
    assert msg == "shot 4: target 'face' matches no kit (kits: a, b); connect LSDub Cast or use one kit"
    assert D._dropped(shots, {'a': object()}, []) == {}
    assert 'connect LSDub Cast' not in D._dropped(shots, kits, [], cast=object())[4][1]


def test_a_dub_gets_no_false_morph_warning(pack):
    warned = pack.nodes_mouth._paint_warnings(_flags(pack, 'layers', settings='--warp-erase closures'))
    assert not [w for w in warned if 'morph' in w]


def test_settings_keep_backslashes_and_quotes(pack):
    a = _flags(pack, settings=r'''--warp-erase C:\Users\me\a.csv --warp-lip-rim "a b" --x "it's"''')
    assert r'C:\Users\me\a.csv' in a and 'a b' in a and "it's" in a


def test_a_lone_apostrophe_stops_at_the_tweaks_node_with_a_clear_message(pack):
    import pytest
    with pytest.raises(ValueError, match='LSDub Mouth Tweaks: settings: unmatched quote'):
        pack.nodes_mouth.LSDubTweaks.execute(0.06, 0.06, 512.0, True, True, True, "--warp-erase it's")


def test_a_failed_gate_names_the_vocalisation(pack):
    note = pack.nodes_mouth._vocal_note
    assert note({'vocal_frac': 0.4}).startswith('vocalisation spans cover 40%') and note({'vocal_frac': 0.1}) == '' and note({}) == ''


def test_sync_advance_reaches_the_paint_flags_and_settings_win(pack):
    tw = types.SimpleNamespace(settings='', flags=dict(sync_advance_ms=60.0))
    a = pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers'), _kit('a'), tw, 'skip')
    assert a[a.index('--coart-advance-ms') + 1] == '60.0'
    tw.settings = '--coart-advance-ms 30'
    a = pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers'), _kit('a'), tw, 'skip')
    assert a[len(a) - 1 - a[::-1].index('--coart-advance-ms') + 1] == '30'
    tw.flags = {}
    tw.settings = ''
    assert '--coart-advance-ms' not in pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers'), _kit('a'), tw, 'skip')


def test_sync_advance_at_zero_keeps_the_tweaks_hash(pack):
    N = pack.nodes_mouth.LSDubTweaks
    base = (0.06, 0.06, 512.0, True, True, True, '')
    assert N.execute(*base).args[0].hash == N.execute(*base, '', 0.0).args[0].hash
    assert N.execute(*base).args[0].hash != N.execute(*base, '', 60.0).args[0].hash


def test_layers_shape_reaches_its_flags_and_leaves_interior_alone(pack):
    a = pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers_shape'), _kit('a'), _tw(''), 'skip')
    b = pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers_interior'), _kit('a'), _tw(''), 'skip')
    for f in ('--coart-gap-anchor', '--warp-erase-close', '--warp-interior-open-floor', '--coart-phones', '--coart-function-gate', '--warp-round', '--warp-contact'):
        assert f in a and f not in b


def test_layers_interior_reaches_the_take_interior_and_not_the_teeth_fit(pack):
    a = pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers_interior'), _kit('a'), _tw(''), 'skip')
    assert a[a.index('--warp-interior-source') + 1] == 'take' and '--warp-target' in a
    assert '--warp-teeth-fit' not in a
    assert '--warp-interior-source' not in pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers'), _kit('a'), _tw(''), 'skip')


def test_sync_phrases_reaches_the_paint_flags_with_the_vocal(pack):
    tw = types.SimpleNamespace(settings='', flags=dict(sync_phrases=True))
    a = pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers'), _kit('a'), tw, 'skip', '/v.wav')
    assert a[a.index('--coart-sync-vocal') + 1] == '/v.wav'
    tw = types.SimpleNamespace(settings='', flags={})
    assert '--coart-sync-vocal' not in pack.nodes_mouth._paint_flags(_proj(pack, mouth_stack='layers'), _kit('a'), tw, 'skip', '/v.wav')


def test_sync_phrases_off_keeps_the_tweaks_hash(pack):
    N = pack.nodes_mouth.LSDubTweaks
    base = (0.06, 0.06, 512.0, True, True, True, '')
    assert N.execute(*base).args[0].hash == N.execute(*base, '', 0.0, False).args[0].hash
    assert N.execute(*base).args[0].hash != N.execute(*base, '', 0.0, True).args[0].hash


def test_workflow_four_uses_the_interior_stack_and_old_workflows_keep_plain():
    import glob
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in sorted(glob.glob(os.path.join(root, 'example_workflows', '*.json'))):
        n = next(x for x in json.load(open(f))['nodes'] if x['type'] == 'LSDubProject')
        assert ('layers_interior' in n['widgets_values']) == os.path.basename(f).startswith('4-')
