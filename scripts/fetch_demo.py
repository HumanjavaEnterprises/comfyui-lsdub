#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Fetch the LSDub demo assets (a 7.5 s clip, its song and dry lead vocal, the lyrics, a 12 s chorus, a start still and a lip kit;
~29 MB, Apache-2.0, original synthetic characters and song) into a ComfyUI install, for the example workflows.

  python scripts/fetch_demo.py [--comfyui ~/ComfyUI] [--zip lsdub-demo-0.1.0.zip]

Media go to ComfyUI/input/lsdub_demo_<name> (where Load Video / Load Audio / Load Image list them), the kit to
ComfyUI/input/lsdub_demo_kits/red_nova. The download is checked against its SHA-256. Run it once, by hand: the node
pack itself never downloads anything at startup."""
import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile

VERSION = '0.1.0'
URL = f'https://github.com/HumanjavaEnterprises/comfyui-lsdub/releases/download/v{VERSION}/lsdub-demo-{VERSION}.zip'
SHA256 = '7981fef8f453b0bf29e47b3ffc00b3adcba1d22d8dbe259504f66c2beb00de17'


def guess_comfyui():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # custom_nodes/comfyui-lsdub
    up = os.path.dirname(os.path.dirname(here))
    return up if os.path.isdir(os.path.join(up, 'custom_nodes')) else os.path.expanduser('~/ComfyUI')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comfyui', default=guess_comfyui(), help='the ComfyUI folder (has input/ and custom_nodes/)')
    ap.add_argument('--zip', help='use this local copy of the demo zip instead of downloading it')
    a = ap.parse_args()
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
        with zipfile.ZipFile(z) as zf: zf.extractall(d)
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
