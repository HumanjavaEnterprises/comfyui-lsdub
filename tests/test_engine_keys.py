# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""a change to the paint keeps the take, the kit and the alignment; a change a stage can reach does not."""
import ast
import glob
import os
import re
import shutil

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAINT = {'core.mouth_warp', 'core.mouth_layers', 'core.mouth_paint', 'core.teeth_layer', 'core.interior_take', 'core.coart'}


def test_no_stage_before_the_dub_reaches_the_paint(pack):
    for stage, e in pack.engine_keys.STAGES.items():
        mods, _ = pack.engine_keys.reached(e)
        assert not (mods & (PAINT - ({'core.coart'} if stage in ('LSDubShotSplit', 'LSDubCast', 'LSDubTriage') else set()))), stage


def copy_engine(tmp_path):
    d = tmp_path / 'lsdub_engine'
    shutil.copytree(os.path.join(PACK, 'lsdub_engine'), d, ignore=shutil.ignore_patterns('__pycache__'))
    return str(d)


def keys(pack, root):
    pack.engine_keys._cache.clear()
    return {s: pack.engine_keys.part(s, '1', root) for s in pack.engine_keys.STAGES}


def edit(root, rel, text='\n# a change\nX_CHANGED = 1\n'):
    with open(os.path.join(root, rel), 'a') as f:
        f.write(text)


def test_a_paint_change_moves_no_stage_and_the_whole_engine_version_would_have(pack, tmp_path):
    root = copy_engine(tmp_path)
    a = keys(pack, root)
    for f in ('core/mouth_warp.py', 'core/mouth_layers.py', 'core/mouth_paint.py', 'core/teeth_layer.py', 'core/interior_take.py'):
        edit(root, f)
    assert keys(pack, root) == a
    pack.engine_keys._cache.clear()


@pytest.mark.parametrize('f,moved', [('core/medium.py', 'LSDubPerformance'), ('core/mouth_library.py', 'LSDubBuildLipKit'), ('core/lyric_align.py', 'LSDubAlign'),
                                     ('core/vocal_retime.py', 'LSDubAlign'), ('core/lead_sheet.py', 'LSDubAlign'), ('audits/qa.py', 'LSDubShotSplit'), ('core/cast.py', 'LSDubTriage')])
def test_a_change_a_stage_can_reach_moves_its_key(pack, tmp_path, f, moved):
    root = copy_engine(tmp_path)
    a = keys(pack, root)
    edit(root, f)
    assert keys(pack, root)[moved] != a[moved]
    pack.engine_keys._cache.clear()


def test_a_dependency_two_steps_off_moves_the_key_through_a_local_import(tmp_path):
    from importlib import util
    sp = util.spec_from_file_location('ek', os.path.join(PACK, 'engine_keys.py'))
    k = util.module_from_spec(sp)
    sp.loader.exec_module(k)
    root = tmp_path / 'e'
    (root / 'core').mkdir(parents=True)
    (root / '__init__.py').write_text('')
    (root / 'core' / '__init__.py').write_text('')
    (root / 'core' / 'steps.py').write_text('STEPS = {"paint": ("core.p", "cli", ())}\ndef call(*a): pass\n')
    (root / 'core' / 'p.py').write_text('def main(): pass\n')
    (root / 'core' / 'b.py').write_text('def g():\n    from . import c\n    return c.h()\n')
    (root / 'core' / 'c.py').write_text('def h(): return 1\n')
    (root / 'core' / 'a.py').write_text('from .b import g\nfrom . import steps\ndef f(): return g()\ndef s(): steps.call("paint")\ndef unused(): pass\n')
    (root / 'core' / 'z.py').write_text('def nothing(): pass\n')
    assert k.reached(['core.a:f'], str(root))[0] == {'core.a', 'core.b', 'core.c'}
    assert 'core.p' in k.reached(['core.a:s'], str(root))[0]  # by name, through STEPS


def test_the_table_lists_every_engine_function_each_node_calls(pack):
    """every `alias.fn` a node's class uses on an engine module is in its stage's entries, so a new call cannot slip past the key."""
    E = pack.engine_keys
    node_of = {'LSDubSongSlot': 'nodes_audio.py', 'LSDubAlign': 'nodes_audio.py', 'LSDubBuildLipKit': 'nodes_mouth.py', 'LSDubStillSlot': 'nodes_video.py', 'LSDubVideoSlot': 'nodes_video.py',
               'LSDubPerformance': 'nodes_video.py', 'LSDubShotSplit': 'nodes_video.py', 'LSDubCast': 'nodes_video.py', 'LSDubTriage': 'nodes_video.py', 'LSDubLyricsSlot': 'nodes_project.py'}
    assert set(node_of) == set(E.STAGES)
    for stage, f in node_of.items():
        tree = ast.parse(open(os.path.join(PACK, f)).read())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == stage)
        helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        roots, q = [cls], [cls]  # the file's own helpers the class calls count as the class
        while q:
            for n in ast.walk(q.pop()):
                if isinstance(n, ast.Name) and n.id in helpers and helpers[n.id] not in roots:
                    roots.append(helpers[n.id])
                    q.append(helpers[n.id])
        al = {}
        for top in [tree, *roots]:
            for n in ast.walk(top):
                if isinstance(n, ast.ImportFrom) and n.module and 'lsdub_engine' in n.module and (top is not tree or n in tree.body):
                    base = n.module.split('lsdub_engine', 1)[1].strip('.')
                    for a in n.names:
                        al[a.asname or a.name] = (base + '.' + a.name).strip('.')
        used = set()
        for r in roots:
            for n in ast.walk(r):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in al:
                    used.add(f'{al[n.value.id]}:{n.attr}')
        have = set(E.STAGES[stage])
        pure = {'core.presets', 'core.recipes', 'core.paint_api', 'io.ff'}
        missing = {u for u in used if u not in have and u.split(':')[0] not in pure}
        assert not missing, (stage, missing)


def test_nothing_in_the_engine_dispatches_by_name_but_the_steps_table():
    hits = []
    for p in glob.glob(os.path.join(PACK, 'lsdub_engine', '**', '*.py'), recursive=True):
        src = open(p).read()
        if re.search(r'importlib\.import_module|__import__\(|(?<![\w.])exec\(|(?<![\w.])eval\(', src):
            hits.append(os.path.relpath(p, os.path.join(PACK, 'lsdub_engine')))
    assert set(hits) <= {'core/steps.py', 'cli.py', '__init__.py'}, hits  # steps runs STEPS; cli and __init__ no stage reaches


def test_a_stale_take_cannot_stand_when_the_stage_upstream_really_changed(pack):
    """the dub keys on the hash of what came from upstream, whatever the upstream's own key does."""
    K = pack.project.stage_key
    assert K('LSDubDub/shot', {}, ['src-a']) != K('LSDubDub/shot', {}, ['src-b'])
    assert K('LSDubPerformance', {'p': 1}) != K('LSDubPerformance', {'p': 2})  # take inputs move it
    assert K('LSDubPerformance/shot', {'i': 1}) != K('LSDubPerformance/shot', {'i': 2})


def test_the_dub_keeps_the_whole_engine(pack):
    K = pack.project.stage_key
    assert K('LSDubDub/shot', {}, []) == K('LSDubDub/shot', {}, [], engine_version=pack.project.ENGINE_VERSION)
    assert K('LSDubDub/shot', {}, []) != K('LSDubDub/shot', {}, [], engine_version='other')
