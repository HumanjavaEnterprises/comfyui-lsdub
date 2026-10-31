# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""No licence word or third-party product name outside SPDX headers and scripts/cleanroom_allowlist.txt, over every tracked file;
no running time in the README or the workflow notes."""
import json
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = re.compile('(?i:licen[cs]|non-?commercial|copyleft|proprietary|research[- ]only)|'
                   r'\b(MIT|BSD|[AL]?GPL|CC[- ]BY|Apache)\b|' + 'Audio' + '2Face|' + 'NVI' + 'DIA')
SPDX = re.compile(r'SPDX-License-Identifier')
TIME = re.compile(r'\b\d+(?:\.\d+)?(?:\s*(?:to|-)\s*\d+)?\s*(?:min|mins|minutes?|hours?|hrs?)\b|\*\*Time:\*\*|\bin \d+ s\b', re.I)


def tracked():
    if not shutil.which('git') or not os.path.exists(os.path.join(ROOT, '.git')): pytest.skip('not a git checkout')
    return subprocess.run(['git', '-C', ROOT, 'ls-files'], capture_output=True, text=True, check=True).stdout.split('\n')[:-1]


def allowlist():
    whole, lines, after = set(), [], []
    for ln in open(os.path.join(ROOT, 'scripts', 'cleanroom_allowlist.txt'), encoding='utf-8'):
        ln = ln.strip()
        if not ln or ln.startswith('#'): continue
        if ' after:: ' in ln: after.append(tuple(ln.split(' after:: ', 1)))
        elif ' :: ' in ln: lines.append(tuple(ln.split(' :: ', 1)))
        else: whole.add(ln)
    return whole, lines, after


def hits(path, text, allowed=None):
    whole, lines, after = allowed or allowlist()
    if path in whole: return []
    out = []
    for i, ln in enumerate(text.split('\n')):
        if any(path == p and t in ln for p, t in after): break
        if WORDS.search(ln) and not SPDX.search(ln) and not any(path == p and t in ln for p, t in lines): out.append(f'{path}:{i + 1}')
    return out


def test_no_licence_or_product_word_in_the_tree():
    bad = []
    for p in tracked():
        try: bad += hits(p, open(os.path.join(ROOT, p), encoding='utf-8').read())
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError): pass
    assert not bad, bad


def test_the_grep_sees_a_planted_word_and_keeps_to_the_allowlist():
    allowed = ({'LICENSE'}, [('a.py', 'keep')], [('R.md', '## Credits')])
    assert hits('b.py', 'x\nunder the MIT terms', allowed) == ['b.py:2']
    assert hits('a.py', 'keep MIT\nGPL', allowed) == ['a.py:2']
    assert hits('R.md', 'ok\n## Credits\nMIT', allowed) == []
    assert hits('LICENSE', 'MIT', allowed) == []


def test_allowlist_entries_exist():
    whole, lines, after = allowlist()
    have = set(tracked())
    assert not (whole | {p for p, _ in lines + after}) - have


def test_no_running_time_in_readme_or_notes():
    txt = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    assert not TIME.findall(txt) and '## Runtimes' not in txt
    d = os.path.join(ROOT, 'example_workflows')
    for f in os.listdir(d):
        if f.endswith('.json'):
            notes = [n for n in json.load(open(os.path.join(d, f)))['nodes'] if n['type'] == 'MarkdownNote']
            assert notes and not TIME.findall(' '.join(notes[0]['widgets_values'])), f
    src = open(os.path.join(ROOT, 'scripts', 'make_workflows.py'), encoding='utf-8').read()
    assert 'RUNTIMES[4]' in src and '**Time:**' not in src
