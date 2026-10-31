# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Every tracked Python file carries the SPDX header."""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_python_file_has_the_header():
    if not shutil.which('git') or not os.path.exists(os.path.join(ROOT, '.git')): pytest.skip('not a git checkout')
    files = subprocess.run(['git', '-C', ROOT, 'ls-files', '*.py'], capture_output=True, text=True, check=True).stdout.split()
    bad = [f for f in files if 'SPDX-License-Identifier: Apache-2.0' not in ''.join(open(os.path.join(ROOT, f), encoding='utf-8').readlines()[:3])]
    assert not bad, bad
