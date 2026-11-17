# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The publish workflow's release-notes rule, run on the repo's CHANGELOG.md so CI fails before a tag does."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts" / "release_notes.py")
rn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rn)


def _key(v):
    return tuple(int(x) for x in v.split("."))


def test_repo_changelog_passes():
    """The top section is the released version, or the next one: release.py wants that section written before it bumps
    pyproject. Every other rule holds either way; the publish job itself stays strict (the heading names pyproject)."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    top = re.search(r"\d+\.\d+\.\d+", rn.top_section(text)[0]).group(0)
    assert _key(top) >= _key(rn.pyproject_version())
    notes = rn.check(text, top)
    assert notes.startswith(f"LSDub {top}")
    assert "Unreleased" not in notes.splitlines()[0]


def _cl(head="1.2.3 (2026-01-01)", n=2, link=True):
    body = "".join(f"- item {i}\n" for i in range(n)) + (f"\n{rn.LINK}\n" if link else "")
    return f"# Changelog\n\n## Unreleased\n\n- not yet\n\n## {head}\n\n{body}\n## 1.2.2\n\n- old\n"


def test_skips_unreleased():
    assert rn.check(_cl(), "1.2.3").startswith("LSDub 1.2.3 (2026-01-01)\n")


@pytest.mark.parametrize(
    "text,ver,why",
    [
        (_cl(), "1.2.4", "does not name"),
        (_cl(head="1.2.30"), "1.2.3", "does not name"),
        (_cl(n=7), "1.2.3", "7 bullets"),
        (_cl(n=0), "1.2.3", "no bullets"),
        (_cl(link=False), "1.2.3", "must end with"),
    ],
)
def test_rules_fail(text, ver, why):
    with pytest.raises(ValueError, match=why):
        rn.check(text, ver)
