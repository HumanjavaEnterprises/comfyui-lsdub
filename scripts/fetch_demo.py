#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Fetch the LSDub demo assets (a 7.5 s clip, its song and dry lead vocal, the lyrics, a 12 s chorus, a start still and a lip kit;
~29 MB, Apache-2.0, original synthetic characters and song) into a ComfyUI install, for the example workflows.

  python scripts/fetch_demo.py [--comfyui PATH/TO/ComfyUI] [--zip lsdub-demo-<VERSION>.zip]

Media go to ComfyUI/input/lsdub_demo_<name> (where Load Video / Load Audio / Load Image list them), the kit to
ComfyUI/input/lsdub_demo_kits/red_nova. The download is checked against its SHA-256. Run it once, by hand: the node
pack itself never downloads anything at startup.

A released zip is never replaced. A new one takes a new version: scripts/build_demo_zip.py makes it, scripts/stamp_demo.py sets VERSION, TAG and
SHA256 here and the name in the README, the release attaches the zip to TAG."""
import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile

VERSION = '0.1.1'
TAG = 'demo-0.1.1'
URL = f'https://github.com/HumanjavaEnterprises/comfyui-lsdub/releases/download/{TAG}/lsdub-demo-{VERSION}.zip'
SHA256 = '0b15dfcf069b6dc2cac25bbe312137eda435e4ac350cce6b61436419fd2b518c'


def unpack(zf, dst):
    """each member written to its own path under dst, after checking it stays there (no symlink, no `..`, no drive or
    backslash form); the checksum already pins the zip, this keeps the unpack safe on its own."""
    root = os.path.realpath(dst)
    for m in zf.infolist():
        n = m.filename
        out = os.path.realpath(os.path.join(dst, n))
        if ((m.external_attr >> 16) & 0o170000 == 0o120000 or '\\' in n or (len(n) > 1 and n[1] == ':')
                or (out != root and not out.startswith(root + os.sep))):
            raise SystemExit(f'zip member {n!r} escapes the folder; not installing')
        if m.is_dir():
            os.makedirs(out, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with zf.open(m) as src, open(out, 'wb') as f: shutil.copyfileobj(src, f)


def guess_comfyui():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # custom_nodes/comfyui-lsdub
    up = os.path.dirname(os.path.dirname(here))
    return up if os.path.isdir(os.path.join(up, 'custom_nodes')) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comfyui', default=guess_comfyui(), help='the ComfyUI folder (has input/ and custom_nodes/)')
    ap.add_argument('--zip', help='use this local copy of the demo zip instead of downloading it')
    a = ap.parse_args()
    if not a.comfyui:
        ap.error('run this from custom_nodes/comfyui-lsdub, or pass --comfyui')
    inp = os.path.join(a.comfyui, 'input')
    os.makedirs(inp, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        z = a.zip
        if not z:
            z = os.path.join(d, 'demo.zip')
            print('downloading', URL)
            urllib.request.urlretrieve(URL, z)  # noqa: S310 (fixed https URL, checked below)
        h = hashlib.sha256(open(z, 'rb').read()).hexdigest()
        if h != SHA256: raise SystemExit(f'checksum mismatch ({h}); not installing')
        with zipfile.ZipFile(z) as zf: unpack(zf, d)
        root = os.path.join(d, f'lsdub-demo-{VERSION}')
        for f in sorted(os.listdir(root)):
            p = os.path.join(root, f)
            if os.path.isfile(p) and f != 'README.txt':
                shutil.copyfile(p, os.path.join(inp, f'lsdub_demo_{f}'))
                print('->', os.path.join(inp, f'lsdub_demo_{f}'))
        kits = os.path.join(inp, 'lsdub_demo_kits')
        shutil.rmtree(kits, ignore_errors=True)
        shutil.copytree(os.path.join(root, 'kits'), kits)
        print('->', kits)
        for k in sorted(os.listdir(kits)):                    # each kit also as a .zip: the form `comfy deploy run` uploads
            if os.path.isdir(os.path.join(kits, k)):
                print('->', shutil.make_archive(os.path.join(kits, k), 'zip', kits, k))


if __name__ == '__main__':
    main()
