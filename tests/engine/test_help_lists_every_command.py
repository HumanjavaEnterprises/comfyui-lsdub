# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""`lsdub --help` names every command the build registers, and nothing it does not: SHIP in the shipped list, DEV in the
dev list (empty in a vendored build). A command added to the table without a help line fails here."""

import os
import re

import pytest

from lsdub_engine import cli

LINE = re.compile(r"^\s+lsdub (\S+)", re.M)
BUILTIN = {"dry", "fetch-models", "verify-watermark"}  # handled in cli.main itself, no module
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def listed(text):
    return set(LINE.findall(text))


def test_every_shipped_command_is_in_the_shipped_help():
    assert listed(cli.HELP_SHIP) == set(cli.SHIP) | BUILTIN


def test_every_dev_command_is_in_the_dev_help():
    assert listed(cli.HELP_DEV) == set(cli.DEV)


def test_help_prints_every_command(capsys):
    assert cli.main(["--help"]) == 0
    assert listed(capsys.readouterr().out) == set(cli.CMDS) | BUILTIN


def test_the_alias_the_help_names_is_installed():
    py = os.path.join(ROOT, "pyproject.toml")
    text = open(py, encoding="utf-8").read() if os.path.exists(py) else ""
    m = re.search(r"^\[project\.scripts\]\n(.*?)(?:^\[|\Z)", text, re.M | re.S)
    if not m:
        pytest.skip("no console scripts here (a vendored build)")
    scripts = set(re.findall(r"^(\S+)\s*=", m.group(1), re.M))
    alias = re.search(r"`(\S+)` is a short alias", cli.HELP_SHIP)
    assert alias and alias.group(1) in scripts
