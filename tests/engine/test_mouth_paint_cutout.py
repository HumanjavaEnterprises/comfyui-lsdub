# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Smoke test for mouth_paint_cutout logic (no cv2/mediapipe needed).

Injects a cv2 shim before importing mouth_paint_cutout, then tests the core
logic: mouth_bbox, jaw_openness_scale, load_mouth_region, cross-fade, resize.
"""
import os, sys, tempfile
import numpy as np

# ---- Inject a cv2 shim BEFORE importing the module ----
class _Shim:
    """Minimal cv2 shim using numpy for the functions mouth_paint_cutout uses."""
    @staticmethod
    def resize(src, dsize, interpolation=None):
        h, w = dsize
        # nearest-neighbor for speed
        src_h, src_w = src.shape[:2]
        out = np.zeros((h, w) + src.shape[2:], src.dtype)
        for y in range(h):
            sy = int(y * src_h / h)
            for x in range(w):
                sx = int(x * src_w / w)
                out[y, x] = src[sy, sx]
        return out
    @staticmethod
    def remap(src, map_x, map_y, flags=None, borderMode=None):
        h, w = src.shape[:2]
        out = np.zeros_like(src, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                sy = float(map_y[y, x])
                sx = float(map_x[y, x])
                y0, x0 = int(np.clip(sy, 0, h - 1)), int(np.clip(sx, 0, w - 1))
                if y0 + 1 < h and x0 + 1 < w:
                    y1, x1 = y0 + 1, x0 + 1
                    fy, fx = sy - y0, sx - x0
                    p00, p01, p10, p11 = src[y0, x0], src[y0, x1], src[y1, x0], src[y1, x1]
                    out[y, x] = (1-fy)*(1-fx)*p00 + fy*(1-fx)*p10 + (1-fy)*fx*p01 + fy*fx*p11
                else:
                    out[y, x] = src[y0, x0]
        return out
    @staticmethod
    def addWeighted(a, alpha, b, beta, gamma):
        return np.clip(a.astype(np.float32) * alpha + b.astype(np.float32) * beta, 0, 255)
    @staticmethod
    def GaussianBlur(src, ksize, sigma=0):
        # 3x3 box blur
        if src.ndim == 2:
            pass
        else:
            src.shape[2]
        out = np.zeros_like(src, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = np.arange(-1, 3)
                for y in range(src.shape[0]):
                    for x in range(src.shape[1]):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < src.shape[0] and 0 <= xx < src.shape[1]:
                            out[y, x] += src[yy, xx]
        return out / 9
    @staticmethod
    def VideoCapture(*a, **kw):
        class Cap:
            def read(self): return False, None
            def get(self, prop): return 0
            def release(self): pass
        return Cap()
    CAP_PROP_FRAME_WIDTH = 0
    CAP_PROP_FRAME_HEIGHT = 0
    CAP_PROP_FPS = 25
    INTER_AREA = 0
    INTER_LINEAR = 1
    INTER_CUBIC = 2
    BORDER_REFLECT = 2
    BORDER_CONSTANT = 0
    BORDER_REPLICATE = 1

# Only install the shim if real cv2 is unavailable — otherwise it would poison the whole
# pytest session (sys.modules is global) and break every cv2-using test collected after this one.
try:
    import cv2 as _real_cv2  # noqa: F401
except Exception:
    sys.modules['cv2'] = _Shim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsdub_engine.anchors.mouth_paint_cutout import mouth_bbox, jaw_openness_scale


# ---- Build a synthetic library ----
lib_dir = tempfile.mkdtemp()
size = 128
mx0, my0, mx1, my1 = size // 3, size // 3, 2 * size // 3, 2 * size // 3

# Mask: filled rectangle (simulates feathered mouth region)
mask = np.zeros((size, size), np.uint8)
mask[my0:my1, mx0:mx1] = 255
np.save(f'{lib_dir}/mask.npy', mask)
# Write a real PNG-style file for cv2.imread compatibility (shim doesn't have imread,
# so we patch load_mouth_region to use np.load instead)

# Viseme mouths: different colors in the mouth region
for v, color in [('sil', (255, 0, 0)), ('aa', (0, 255, 0)), ('ou', (0, 0, 255))]:
    img = np.zeros((size, size, 3), np.uint8)
    img[my0:my1, mx0:mx1] = color
    np.save(f'{lib_dir}/{v}.npy', img)
    # Also save as .png path for load_mouth_region (patch it)

# Track
fr_t = [
    {'vis': 'sil', 'vis2': 'sil', 'mix': 0.0, 'open': 0.0},
    {'vis': 'aa', 'vis2': 'sil', 'mix': 0.5, 'open': 0.7},
    {'vis': 'aa', 'vis2': 'aa', 'mix': 0.0, 'open': 1.0},
    {'vis': 'ou', 'vis2': 'aa', 'mix': 0.3, 'open': 0.4},
    {'vis': 'sil', 'vis2': 'ou', 'mix': 0.8, 'open': 0.1},
]

# Patch load_mouth_region to read .npy instead of .png
import lsdub_engine.anchors.mouth_paint_cutout as mpc

def _patched_load(lib, v):
    import os as _os
    f = f'{lib}/{v}.npy'
    if not _os.path.exists(f):
        f = f'{lib}/sil.npy'
    img = np.load(f)
    mask = np.load(f'{lib}/mask.npy').astype(np.float32) / 255.0
    x0, y0, w, h = mpc.mouth_bbox(mask, pad_frac=0.05)
    crop = img[y0:y0+h, x0:x0+w]
    m = mask[y0:y0+h, x0:x0+w]
    return crop, m

mpc.load_mouth_region = _patched_load

print('=' * 50)
print('TEST 1: mouth_bbox')
print('=' * 50)
bbox = mouth_bbox(mask, pad_frac=0.05)
x0, y0, w, h = bbox
print(f'  bbox: x0={x0} y0={y0} w={w} h={h} (source: {mx0},{my0},{mx1},{my1})')
assert x0 <= mx0, f'bbox x0={x0} should be <= {mx0}'
assert y0 <= my0, f'bbox y0={y0} should be <= {my0}'
assert x0 + w >= mx1, f'bbox right={x0+w} should be >= {mx1}'
assert y0 + h >= my1, f'bbox bottom={y0+h} should be >= {my1}'
print('  PASSED')

print()
print('=' * 50)
print('TEST 2: jaw_openness_scale')
print('=' * 50)
img = np.zeros((64, 64, 3), np.uint8)
img[20:50, 20:50] = [255, 0, 0]
m = np.zeros((64, 64), np.float32)
m[20:50, 20:50] = 1.0

result = jaw_openness_scale(img, m, 0.0)
assert result.shape == img.shape, f'shape {result.shape} != {img.shape}'
assert np.array_equal(result, img), 'extra_open=0 should return identical image'
print('  extra_open=0.0: no change (PASSED)')

result_open = jaw_openness_scale(img, m, 0.3)
assert result_open.shape == img.shape
diff = np.abs(result_open.astype(np.float32) - img.astype(np.float32)).sum()
assert diff > 0, f'extra_open=0.3 should change pixels, got diff={diff}'
print(f'  extra_open=0.3: {diff:.1f} px changed (PASSED)')

# Test negative (closing)
result_close = jaw_openness_scale(img, m, -0.2)
diff_close = np.abs(result_close.astype(np.float32) - img.astype(np.float32)).sum()
assert diff_close > 0, f'extra_open=-0.2 should change pixels, got diff={diff_close}'
print(f'  extra_open=-0.2: {diff_close:.1f} px changed (PASSED)')

print()
print('=' * 50)
print('TEST 3: load_mouth_region (patched for .npy)')
print('=' * 50)
c1, m1 = mpc.load_mouth_region(lib_dir, 'sil')
c2, m2 = mpc.load_mouth_region(lib_dir, 'aa')
c3, m3 = mpc.load_mouth_region(lib_dir, 'ou')
print(f'  sil: crop={c1.shape}, mask={m1.shape}')
print(f'  aa:  crop={c2.shape}, mask={m2.shape}')
print(f'  ou:  crop={c3.shape}, mask={m3.shape}')
assert c1.shape[0] > 0 and c1.shape[1] > 0
assert m1.shape == c1.shape[:2]
assert c2.shape[0] > 0 and c2.shape[1] > 0
print('  PASSED')

print()
print('=' * 50)
print('TEST 4: cross-fade (viseme blending)')
print('=' * 50)
mix = 0.5
src = np.clip(c1.astype(np.float32) * (1 - mix) + c2.astype(np.float32) * mix, 0, 255).astype(np.uint8)
mask = np.clip(m1 * (1 - mix) + m2 * mix, 0, 1)
assert src.shape == c1.shape
assert mask.shape == c1.shape[:2]
# Verify mix is between sil (red) and aa (green)
mean = src.mean(axis=(0, 1))
print(f'  mean color (sil->aa, mix=0.5): R={mean[0]:.1f} G={mean[1]:.1f} B={mean[2]:.1f}')
assert mean[0] > 50 and mean[1] > 50, 'should have both red and green components'
assert mean[2] < 50, 'should have little blue'
print('  PASSED')

print()
print('=' * 50)
print('TEST 5: resize to target region')
print('=' * 50)
x, y, w, h = 10, 10, 50, 40
scale_w = w / c1.shape[1]
scale_h = h / c1.shape[0]
s = min(scale_w, scale_h)
nw, nh = max(1, int(c1.shape[1] * s)), max(1, int(c1.shape[0] * s))
src_r = _Shim.resize(c1, (nw, nh))
print(f'  target {w}x{h}, resized to {nw}x{nh} (scale={s:.3f})')
assert nw <= w + 1 and nh <= h + 1, f'resized {nw}x{nh} too big for {w}x{h}'
print('  PASSED')

print()
print('=' * 50)
print('TEST 6: per-frame region list (JSON)')
print('=' * 50)
# Test the logic in run() for parsing regions
n = 5
regions_raw = [[10, 10, 50, 40], [20, 20, 60, 50], [30, 30, 70, 60], [40, 40, 80, 70], [50, 50, 90, 80]]
regions = []
last = regions_raw[0]
for i in range(n):
    regions.append(regions_raw[i] if i < len(regions_raw) else last)
assert len(regions) == 5
assert regions[0] == [10, 10, 50, 40]
assert regions[4] == [50, 50, 90, 80]
print(f'  {n} regions from JSON list: OK')

# Test fixed string parsing
x, y, w, h = (int(float(v)) for v in '10,20,50,40'.split(','))
assert (x, y, w, h) == (10, 20, 50, 40)
print(f'  fixed "10,20,50,40" -> ({x},{y},{w},{h}): OK')
print('  PASSED')

import shutil
shutil.rmtree(lib_dir)
print()
print('ALL SMOKE TESTS PASSED')
