#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The README's Project inputs and the bug form's node list and version line, emitted from the node schemas.

  python scripts/make_readme_nodes.py          rewrite README.md and .github/ISSUE_TEMPLATE/bug_report.yml
  python scripts/make_readme_nodes.py --check  exit 1 when either differs"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import conftest  # noqa: E402  (installs the ComfyUI stubs, loads the pack)

README, BUG = os.path.join(ROOT, 'README.md'), os.path.join(ROOT, '.github', 'ISSUE_TEMPLATE', 'bug_report.yml')
BLOCK = re.compile(r'(<!-- project-inputs:begin -->\n).*?(<!-- project-inputs:end -->)', re.S)


def schemas(pack):
    return [c.define_schema() for m in ('nodes_project', 'nodes_video', 'nodes_vocal', 'nodes_audio', 'nodes_mouth', 'nodes_qa')
            for c in getattr(pack, m).NODES]


def readme(text, pack):
    p = pack.nodes_project.NODES[0].define_schema()
    lines = ''.join(f"  - `{i.id}`: {' '.join(i.tooltip.split())}\n" for i in p.inputs)
    return BLOCK.sub(lambda m: m.group(1) + lines + m.group(2), text)


def bug(text, pack):
    names = [s.display_name.removeprefix('LSDub ').split(' (')[0] for s in schemas(pack)]
    text = re.sub(r'(label: Node that failed \(if any\)\n\s+options: )\[.*\]', lambda m: f"{m.group(1)}[{', '.join([*names, 'Not sure'])}]", text)
    version = re.search(r'^version = "(.+)"', open(os.path.join(ROOT, 'pyproject.toml')).read(), re.M).group(1)
    return re.sub(r'placeholder: "LSDub [^,]+,', f'placeholder: "LSDub {version},', text)


def main(argv=None):
    check = '--check' in (sys.argv[1:] if argv is None else argv)
    pack, bad = conftest.load_pack(), []
    for path, fn in ((README, readme), (BUG, bug)):
        old = open(path, encoding='utf-8').read()
        new = fn(old, pack)
        if new == old: continue
        bad.append(os.path.relpath(path, ROOT))
        if not check:
            with open(path, 'w', encoding='utf-8') as f: f.write(new)
    if check and bad:
        print('out of date:', *bad)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
