#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Release notes for the Comfy Registry: the first CHANGELOG.md section not headed "Unreleased".

The section must name the pyproject version in its heading, carry at most MAX_BULLETS top-level bullets and end
with the full-changelog link. `python scripts/release_notes.py [--out FILE]` prints (or writes) the notes, headed
"LSDub <version> (<date>)", and exits 1 with the reason when a rule fails. The publish workflow and
tests/test_release_notes.py both run it."""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BULLETS = 6
LINK = "Full changelog: https://lsdub.sndbnch.app/changelog"


def pyproject_version(root=ROOT):
    m = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    if not m:
        raise ValueError("no version in pyproject.toml")
    return m.group(1)


def top_section(text):
    """(heading, body lines) of the first `## ` section whose heading is not Unreleased."""
    sections, cur = [], None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = [line[3:].strip(), []]
            sections.append(cur)
        elif cur is not None:
            cur[1].append(line)
    for head, body in sections:
        if head.strip("[] ").lower() != "unreleased":
            return head, body
    raise ValueError("no released section in CHANGELOG.md")


def check(text, version):
    """Return the notes for `version`; raise ValueError naming the rule that fails."""
    head, body = top_section(text)
    if not re.search(rf"(?<![\d.]){re.escape(version)}(?![\d.])", head):
        raise ValueError(f"top section '{head}' does not name pyproject version {version}")
    bullets = [ln for ln in body if ln.startswith("- ")]
    if not bullets:
        raise ValueError(f"section {head} has no bullets")
    if len(bullets) > MAX_BULLETS:
        raise ValueError(f"section {head} has {len(bullets)} bullets (at most {MAX_BULLETS})")
    rest = [ln.strip() for ln in body if ln.strip()]
    if rest[-1] != LINK:
        raise ValueError(f"section {head} must end with the line: {LINK}")
    return "\n".join([f"LSDub {head}", *body]).strip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write the notes here instead of stdout")
    a = ap.parse_args(argv)
    try:
        notes = check((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), pyproject_version())
    except ValueError as e:
        print(f"release notes: {e}", file=sys.stderr)
        return 1
    if a.out:
        Path(a.out).write_text(notes, encoding="utf-8")
    else:
        sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
