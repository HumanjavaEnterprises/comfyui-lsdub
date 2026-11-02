#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""build_demo_zip.py: the demo folder as lsdub-demo-<version>.zip, the same bytes each time: sorted names, one fixed time, the root folder named for the version.

  python scripts/build_demo_zip.py path/to/demo/folder 0.1.4 [out.zip]"""
import os
import re
import sys
import zipfile

STAMP = (1980, 1, 1, 0, 0, 0)


def build(src, version, out):
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise SystemExit(f'build_demo_zip: {version!r} is not a version')
    if not os.path.isdir(src):
        raise SystemExit(f'build_demo_zip: no folder {src}')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d, dirs, files in sorted(os.walk(src)):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(d, f)
                name = '/'.join((f'lsdub-demo-{version}', *os.path.relpath(p, src).split(os.sep)))
                info = zipfile.ZipInfo(name, STAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(p, 'rb') as fh:
                    z.writestr(info, fh.read())
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) not in (2, 3):
        raise SystemExit(__doc__)
    print(build(a[0], a[1], a[2] if len(a) == 3 else f'lsdub-demo-{a[1]}.zip'))
