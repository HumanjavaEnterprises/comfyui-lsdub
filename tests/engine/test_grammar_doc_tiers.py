# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The shot-grammar tiers the module docstring states are the ones the code uses (CU_PX, MED_PX, WIDE_PX)."""

import re

from lsdub_engine.core import grammar as G


def test_docstring_tiers_match_the_constants():
    m = re.search(r"close-up >= (\d+).*?medium (\d+)-(\d+), wide (\d+)-(\d+), none \(< (\d+)", G.__doc__)
    assert m, "the tiers line is missing from the grammar docstring"
    cu, med_lo, med_hi, wide_lo, wide_hi, none_below = map(float, m.groups())
    assert cu == med_hi == G.CU_PX
    assert med_lo == wide_hi == G.MED_PX
    assert wide_lo == none_below == G.WIDE_PX
