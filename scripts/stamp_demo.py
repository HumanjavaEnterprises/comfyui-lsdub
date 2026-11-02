#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""stamp_demo.py: point fetch_demo.py and the README at a new demo zip, at release: a released asset is never replaced, a new zip takes a new version and tag.

  python scripts/stamp_demo.py lsdub-demo-0.1.4.zip 0.1.4 v0.1.4"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def stamp(text, version, tag, sha):
    text = re.sub(r"^VERSION = '.*'$", f"VERSION = '{version}'", text, flags=re.M)
    text = re.sub(r"^TAG = .*$", f"TAG = '{tag}'", text, flags=re.M)
    return re.sub(r"^SHA256 = '.*'$", f"SHA256 = '{sha}'", text, flags=re.M)


def main(argv=None):
    z, version, tag = (sys.argv[1:] if argv is None else argv)
    with open(z, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    p = os.path.join(HERE, 'fetch_demo.py')
    with open(p, encoding='utf-8') as f:
        s = stamp(f.read(), version, tag, sha)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(s)
    r = os.path.join(HERE, '..', 'README.md')
    with open(r, encoding='utf-8') as f:
        s = re.sub(r'lsdub-demo-\d+\.\d+\.\d+\.zip', f'lsdub-demo-{version}.zip', f.read())
    with open(r, 'w', encoding='utf-8') as f:
        f.write(s)
    print(sha)


if __name__ == '__main__':
    main()
