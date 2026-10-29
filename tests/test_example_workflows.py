# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The example workflows match the node schemas: every LSDub node exists, its socket inputs are the schema's (names,
types, order), required inputs are linked, the widget values are one per widget input and valid (combo options, number
ranges, types), and every link joins existing slots of the same type."""
import glob
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = sorted(glob.glob(os.path.join(ROOT, 'example_workflows', '*.json')))
WIDGET = {'STRING', 'INT', 'FLOAT', 'BOOLEAN', 'COMBO'}


def schemas(pack):
    out = {}
    for m in ('nodes_project', 'nodes_audio', 'nodes_vocal', 'nodes_video', 'nodes_mouth', 'nodes_qa'):
        for cls in getattr(pack, m).NODES:
            s = cls.define_schema()
            out[s.node_id] = s
    return out


def check_widget(p, v):
    t, kw = p.io_type, p.kw
    if t == 'COMBO':
        opts = list(kw.get('options') or [])
        if opts: assert v in opts, f'{p.id}: {v!r} not in {opts}'
        else: assert isinstance(v, str) and v, f'{p.id}: empty combo value'          # a model list read at runtime
    elif t == 'INT':
        assert isinstance(v, int) and not isinstance(v, bool), f'{p.id}: {v!r} is not an int'
    elif t == 'FLOAT':
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f'{p.id}: {v!r} is not a number'
    elif t == 'BOOLEAN':
        assert isinstance(v, bool), f'{p.id}: {v!r} is not a boolean'
    elif t == 'STRING':
        assert isinstance(v, str), f'{p.id}: {v!r} is not a string'
    if t in ('INT', 'FLOAT'):
        if 'min' in kw: assert v >= kw['min'], f'{p.id}: {v} < min {kw["min"]}'
        if 'max' in kw: assert v <= kw['max'], f'{p.id}: {v} > max {kw["max"]}'


def check_workflow(wf, S):
    nodes = {n['id']: n for n in wf['nodes']}
    links = {l[0]: l for l in wf['links']}
    for lid, src, so, dst, di, typ in wf['links']:                                      # links join real, typed slots
        assert src in nodes and dst in nodes, lid
        assert nodes[src]['outputs'][so]['type'] == typ and lid in nodes[src]['outputs'][so]['links'], lid
        assert nodes[dst]['inputs'][di]['link'] == lid and nodes[dst]['inputs'][di]['type'] in (typ, '*'), lid
    n_lsdub = 0
    for n in wf['nodes']:
        if not n['type'].startswith('LSDub'): continue
        n_lsdub += 1
        assert n['type'] in S, f"unknown node type {n['type']}"
        s = S[n['type']]
        sockets = [p for p in s.inputs if p.io_type not in WIDGET]
        widgets = [p for p in s.inputs if p.io_type in WIDGET]
        assert [(i['name'], i['type']) for i in n['inputs']] == [(p.id, p.io_type) for p in sockets], n['type']
        for i, p in zip(n['inputs'], sockets):
            if not p.optional: assert i['link'] is not None and i['link'] in links, f"{n['type']}.{p.id} is not linked"
        assert [(o['name'], o['type']) for o in n['outputs']] == [(o.display_name, o.io_type) for o in s.outputs], n['type']
        wv = n['widgets_values']
        assert len(wv) == len(widgets), f"{n['type']}: {len(wv)} widget values for {len(widgets)} widgets"
        for p, v in zip(widgets, wv):
            check_widget(p, v)
    assert n_lsdub >= 5


@pytest.mark.parametrize('path', WORKFLOWS, ids=[os.path.basename(p) for p in WORKFLOWS])
def test_example_workflow_matches_the_schemas(pack, path):
    check_workflow(json.load(open(path)), schemas(pack))


def test_the_check_catches_a_bad_widget_value(pack):
    S = schemas(pack)
    wf = json.load(open([p for p in WORKFLOWS if '4-' in os.path.basename(p)][0]))
    dub = next(n for n in wf['nodes'] if n['type'] == 'LSDubDub')
    dub['widgets_values'][-1] = 'sing'                                                  # not a vocalisations option
    with pytest.raises(AssertionError, match='not in'):
        check_workflow(wf, S)
    dub['widgets_values'] = dub['widgets_values'][:-1]                                  # one widget value short
    with pytest.raises(AssertionError, match='widget values'):
        check_workflow(wf, S)


def test_workflow_four_mouths_the_vocalisations(pack):
    wf = json.load(open([p for p in WORKFLOWS if '4-' in os.path.basename(p)][0]))
    S = schemas(pack)
    dub = next(n for n in wf['nodes'] if n['type'] == 'LSDubDub')
    names = [p.id for p in S['LSDubDub'].inputs if p.io_type in WIDGET]
    assert dict(zip(names, dub['widgets_values']))['vocalisations'] == 'mouth'
    assert len(WORKFLOWS) == 4
