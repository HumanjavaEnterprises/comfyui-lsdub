# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The example workflows match the node schemas: every LSDub node exists, its socket inputs are the schema's (names,
types, order), required inputs are linked, the widget values are one per widget input and valid (combo options, number
ranges, types), and every link joins existing slots of the same type."""
import glob
import json
import math
import os
import re

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


PENDING = {'LSDubProject': ('mouth_stack',), 'LSDubScorecard': ('sharpness_mode',), 'LSDubTweaks': ('sync_phrases',), 'LSDubAlign': ('retime_to_vocal', 'lead_sheet', 'lead_overrides')}  # appended trailing optionals
PENDING_SOCKETS = {'LSDubFixIt': ('scorecard', 'ruler')}                                    # saved before these; regenerated with the rest


def has_control(p):
    """the frontend rule: an INT/COMBO named seed or noise_seed, or declared with control_after_generate, gets a
    'control after generate' widget right after it, so one extra value in widgets_values."""
    return p.io_type in ('INT', 'COMBO') and (p.id in ('seed', 'noise_seed') or bool(p.kw.get('control_after_generate')))


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
        ordered = [p for p in s.inputs if not p.optional] + [p for p in s.inputs if p.optional]   # the frontend's input_order
        widgets = [p for p in ordered if p.io_type in WIDGET]
        sockets = [p for p in ordered if p.io_type not in WIDGET]
        if len(n['inputs']) < len(sockets):
            sockets = [p for p in sockets if p.id not in PENDING_SOCKETS.get(n['type'], ())]
        assert [(i['name'], i['type']) for i in n['inputs']] == [(p.id, p.io_type) for p in sockets], n['type']
        for i, p in zip(n['inputs'], sockets):
            if not p.optional: assert i['link'] is not None and i['link'] in links, f"{n['type']}.{p.id} is not linked"
        assert [(o['name'], o['type']) for o in n['outputs']] == [(o.display_name, o.io_type) for o in s.outputs], n['type']
        wv = list(n['widgets_values'])
        pending = PENDING.get(n['type'], ())                                            # saved before these: they take their default
        if len(wv) == len(widgets) + sum(has_control(p) for p in widgets) - len(pending):
            widgets = [p for p in widgets if p.id not in pending]
        n_values = len(widgets) + sum(has_control(p) for p in widgets)
        assert len(wv) == n_values, f"{n['type']}: {len(wv)} widget values for {n_values} widget slots"
        for p in widgets:                                                               # read them the way the frontend does
            check_widget(p, wv.pop(0))
            if has_control(p):
                c = wv.pop(0)
                assert c in ('fixed', 'increment', 'decrement', 'randomize'), f"{n['type']}.{p.id}: control value {c!r}"
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


def test_workflow_four_vocalisations_are_the_recorded_choice(pack):
    wf = json.load(open([p for p in WORKFLOWS if '4-' in os.path.basename(p)][0]))
    S = schemas(pack)
    dub = next(n for n in wf['nodes'] if n['type'] == 'LSDubDub')
    ins = S['LSDubDub'].inputs
    names = [p.id for p in [q for q in ins if not q.optional] + [q for q in ins if q.optional] if p.io_type in WIDGET]
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'make_workflows.py')).read()
    choice = re.search(r"^WF4_VOCALISATIONS = '(\w+)'", src, re.M).group(1)
    assert choice in ('skip', 'mouth')                                                   # `rest` is not on workflow 4's menu
    assert dict(zip(names, dub['widgets_values']))['vocalisations'] == choice
    assert len(WORKFLOWS) == 4


def test_workflow_four_generates_no_wasted_shot(pack):
    wf = json.load(open([p for p in WORKFLOWS if '4-' in os.path.basename(p)][0]))
    perf = next(n for n in wf['nodes'] if n['type'] == 'LSDubPerformance')
    s = next(v for v in perf['widgets_values'] if isinstance(v, float) and v > 0)
    need = int(round(15.3 * 25))                                                         # the demo song
    assert math.ceil(15.3 / s) == 3 and 3 * pack.graphs.video_frames(s, 25) >= need


def test_project_widgets_in_ui_order(pack):
    """the frontend fills widgets_values required-then-optional: backend must read 'local', force_shots ''."""
    for f in WORKFLOWS:
        wf = json.load(open(f))
        n = next(x for x in wf['nodes'] if x['type'] == 'LSDubProject')
        s = schemas(pack)['LSDubProject']
        ordered = [p for p in s.inputs if not p.optional] + [p for p in s.inputs if p.optional]
        names = [k for p in ordered if p.io_type in WIDGET
                 for k in ((p.id, 'control_after_generate') if has_control(p) else (p.id,)) if k != 'mouth_stack']
        vals = dict(zip(names, n['widgets_values']))
        extra = n['widgets_values'][len(names):]                                         # an appended optional a workflow may set
        assert len(extra) <= 1 and all(x in schemas(pack)['LSDubProject'].inputs[-1].kw['options'] for x in extra), (f, n['widgets_values'])
        assert vals['seed'] == 0 and vals['control_after_generate'] == 'fixed', (f, vals)
        assert vals['backend'] == 'local' and vals['force_shots'] == '', (f, vals)


def test_the_check_catches_a_missing_seed_control(pack):
    """the v0.1.1 bug: no control value after seed shifts every later value in the frontend."""
    wf = json.load(open(WORKFLOWS[0]))
    n = next(x for x in wf['nodes'] if x['type'] == 'LSDubProject')
    n['widgets_values'].remove('fixed')
    with pytest.raises(AssertionError):
        check_workflow(wf, schemas(pack))
