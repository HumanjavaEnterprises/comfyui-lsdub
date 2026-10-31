# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""constraints-hosted.txt mirrors the Comfy Build yaml's pipDependencies, and every pin sits inside the
pack's own dependency ranges (pyproject.toml and requirements.txt agree too)."""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSTRAINTS = os.path.join(ROOT, 'constraints-hosted.txt')
YAML_DEFAULT = os.path.join(os.path.dirname(ROOT), 'lsdub-livetry-build', 'comfy-build.yaml')


def _lines(text):
    return [ln.split('#', 1)[0].strip() for ln in text.splitlines() if ln.split('#', 1)[0].strip()]


def _header(name):
    with open(CONSTRAINTS, encoding='utf-8') as f:
        m = re.search(rf'^# {name}: (\S+)$', f.read(), re.M)
    return m.group(1) if m else None


def _constraints():
    with open(CONSTRAINTS, encoding='utf-8') as f:
        return _lines(f.read())


def _yaml_pip(text):
    """The pipDependencies block scalar, without a YAML dependency."""
    out, inside, indent = [], False, None
    for ln in text.splitlines():
        if not inside:
            if re.match(r'^\s*pipDependencies:\s*\|', ln):
                inside = True
            continue
        if not ln.strip():
            continue
        lead = len(ln) - len(ln.lstrip())
        if indent is None:
            indent = lead
        if lead < indent:
            break
        out.append(ln.strip())
    return _lines('\n'.join(out))


def _name(req):
    return re.split(r'[<>=!~;\[ ]', req, 1)[0].lower().replace('_', '-')


def _pyproject_deps():
    with open(os.path.join(ROOT, 'pyproject.toml'), encoding='utf-8') as f:
        block = re.search(r'^dependencies = \[(.*?)^\]', f.read(), re.M | re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def test_constraints_match_build_yaml():
    path = os.environ.get('LSDUB_BUILD_YAML', YAML_DEFAULT)
    if not os.path.isfile(path):
        pytest.skip(f'build yaml not present ({path}); set LSDUB_BUILD_YAML to check')
    with open(path, encoding='utf-8') as f:
        text = f.read()
    assert _constraints() == _yaml_pip(text)
    assert _header('build-id') == re.search(r'^id: (\S+)', text, re.M).group(1)
    assert _header('syncedRevision') == re.search(r"^syncedRevision: '?([^'\s]+)'?", text, re.M).group(1)


def test_constraints_inside_pack_ranges():
    from packaging.requirements import Requirement
    ranges = {_name(r): Requirement(r).specifier for r in _pyproject_deps()}
    for line in _constraints():
        req = Requirement(line)
        spec = ranges.get(_name(line))
        if spec is None:
            continue                                          # transitive pin (matplotlib, contourpy, contrib cv2)
        pins = [s.version for s in req.specifier if s.operator == '==']
        for v in pins:
            assert v in spec, f'{line} falls outside the pack range {req.name}{spec}'


def test_requirements_txt_matches_pyproject():
    with open(os.path.join(ROOT, 'requirements.txt'), encoding='utf-8') as f:
        assert sorted(_lines(f.read())) == sorted(_pyproject_deps())
