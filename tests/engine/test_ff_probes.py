# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""every ffprobe call in the tree is one the emulation in io/ff.py was checked against: -show_entries / -of pairs, in either quote
style, including the conditional shape in sync_strip."""

import glob, os, re

import lsdub_engine

ROOT = os.path.dirname(lsdub_engine.__file__)
PROBES = {
    ("stream=width,height,r_frame_rate", "json"),
    ("format_tags=comment", "default=nw=1:nk=1"),
    ("format=duration", "csv=p=0"),
    ("stream=index", "csv=p=0"),
    ("entries", "csv=p=0"),  # sync_strip._probe: `entries` is a parameter, its callers are pinned below
}
STRIP_ENTRIES = {
    "stream=width,height,r_frame_rate",
    "format=duration",
    "stream=r_frame_rate",
    "stream=width,height",
    "stream=nb_read_packets",
}
LIT = re.compile(r"""["']ffprobe["']""")
Q = r"""["']"""
ENT = re.compile(Q + r"-show_entries" + Q + r"\s*,\s*(?:" + Q + r"([^\"']+)" + Q + r"|(\w+))")
OF = re.compile(Q + r"-of" + Q + r"\s*,\s*" + Q + r"([^\"']+)" + Q)


def pairs(src):
    out = []
    for m in LIT.finditer(src):
        win = src[m.end() : m.end() + 400]
        e, o = ENT.search(win), OF.search(win)
        out.append(((e.group(1) or e.group(2)) if e else None, o.group(1) if o else None))
    return out


def sources():
    for p in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        if os.sep + "legacy" + os.sep not in p and not p.endswith(os.path.join("io", "ff.py")):
            yield p, open(p, encoding="utf-8").read()


def test_every_probe_is_a_known_pair():
    seen = []
    for p, s in sources():
        for pair in pairs(s):
            assert pair in PROBES, (p, pair)
            seen.append(pair)
    assert seen


def test_the_scan_reads_both_quote_styles():
    assert pairs("""['ffprobe', '-show_entries', 'format=duration', '-of', 'csv=p=0']""") == [("format=duration", "csv=p=0")]
    assert pairs("""["ffprobe", "-show_entries", 'format=duration', "-of", 'csv=p=0']""") == [("format=duration", "csv=p=0")]


def test_sync_strip_conditional_shape_and_its_callers():
    src = open(os.path.join(ROOT, "audits", "sync_strip.py"), encoding="utf-8").read()
    assert '(["-select_streams", "v:0"] if stream else [])' in src or "(['-select_streams', 'v:0'] if stream else [])" in src
    used = re.findall(r"""_probe\([^,]+,\s*["']([^"']+)["']""", src)
    assert used and set(used) <= STRIP_ENTRIES, used
