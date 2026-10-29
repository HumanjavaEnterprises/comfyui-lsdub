# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""qwen_graph.py — masked paste-back helper for Qwen-Image-Edit-2511 output."""
from PIL import Image, ImageFilter


def paste_back(plate_path, gen_path, mask_path, out_path):
    """Paste generated lips back onto a plate using a Gaussian-blurred mask."""
    plate = Image.open(plate_path).convert("RGB")
    gen = Image.open(gen_path).convert("RGB").resize(plate.size, Image.LANCZOS)
    m = Image.open(mask_path).convert("L").resize(plate.size).filter(ImageFilter.GaussianBlur(6))
    Image.composite(gen, plate, m).save(out_path)
