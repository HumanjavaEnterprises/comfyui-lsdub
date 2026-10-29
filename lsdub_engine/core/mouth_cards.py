#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_cards.py — build a DETAILED, FACE-SPECIFIC card representation of every viseme from a mouth kit.

The card glyph in viseme_overlay.glyph() is a generic, procedural lip shape (same for every face). This builds
the real thing: for each viseme in a kit (its captured 478-pt face-mesh landmarks), it draws THAT face's actual
outer+inner lip contour in the card's style, at 2-3x the glyph resolution, and writes:
  <kit>/cards/<v>.png          one high-res card per viseme (the detailed mouth shape of this face)
  <kit>/cards/atlas.png        all visemes on one labelled sheet (the face's mouth-shape alphabet)
  <kit>/cards/cards.json       per-viseme metrics (open/width px, normalized) for the overlay to key off

This is a BuildLipKit step: while we harvest the face's mouth library (real pixels + landmarks), we also emit a
detailed card atlas of every mouth shape. The overlay can then draw the face's own shape instead of the generic glyph.

  python -m lsdub_engine.core.mouth_cards <kit-dir> [--tile 512] [--order sil,PP,...]
"""
import os, json, argparse
import numpy as np, cv2

SHAPES = ['sil', 'PP', 'FF', 'TH', 'DD', 'SS', 'CH', 'RR', 'aa', 'E', 'ih', 'oh', 'ou']
# MediaPipe FaceMesh canonical lip loops (ordered, so fillPoly makes a clean polygon)
OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
# colours (BGR) — the card palette
BG = (24, 22, 28)
LIP = (104, 92, 178)
CAV = (28, 16, 30)
TEETH = (210, 214, 220)
INK = (235, 235, 240)


def _lips(lm):
    return lm[OUTER].astype(np.float64), lm[INNER].astype(np.float64)


def card(lm, tile, scale, centre, label=None, ss=3):
    """one viseme card: the real outer+inner lip contour of `lm`, card-styled, at `tile` px (supersampled x ss)."""
    T = tile * ss
    img = np.full((T, T, 3), BG, np.uint8)
    out, inn = _lips(lm)
    P = lambda q: np.round((np.asarray(q, float) - centre) * scale * ss + T / 2).astype(np.int32)
    cv2.fillPoly(img, [P(out)], LIP, cv2.LINE_AA)
    # inner cavity + a teeth hint across the top of the opening when it is open enough
    innP = P(inn)
    cv2.fillPoly(img, [innP], CAV, cv2.LINE_AA)
    y0, y1 = innP[:, 1].min(), innP[:, 1].max()
    x0, x1 = innP[:, 0].min(), innP[:, 0].max()
    openpx = (inn[:, 1].max() - inn[:, 1].min())
    if openpx * scale > tile * 0.06:                       # show upper teeth band when the mouth is open
        band = np.zeros((T, T), np.uint8)
        cv2.fillPoly(band, [innP], 255)
        th = band.copy()
        th[int(y0 + 0.42 * (y1 - y0)):] = 0
        img[th > 0] = TEETH
        for k in np.linspace(x0, x1, 6)[1:-1]:             # tooth gaps
            xx = int(k)
            img[(th[:, xx] > 0), max(0, xx - ss):xx + ss] = (150, 150, 158)
    cv2.polylines(img, [P(out)], True, INK, max(1, int(0.010 * T)), cv2.LINE_AA)
    cv2.polylines(img, [P(inn)], True, INK, max(1, int(0.006 * T)), cv2.LINE_AA)
    img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
    # note: the saved tile is LABEL-FREE (so the overlay can drop it straight into the glyph slot);
    # labels are drawn only on the atlas composite below.
    return img, float(openpx)


def build(kit, tile=512, order=None, ss=3):
    order = order or SHAPES
    lms = {v: np.load(os.path.join(kit, f'{v}_lm.npy')) for v in order if os.path.exists(os.path.join(kit, f'{v}_lm.npy'))}
    if not lms: raise FileNotFoundError(f'no <viseme>_lm.npy in {kit}')
    # ONE global scale + mouth centre, so the cards are comparable (you SEE aa/oh open, sil/PP closed)
    allpts = np.concatenate([np.concatenate(_lips(l)) for l in lms.values()])
    cen = allpts.mean(0)
    span = np.percentile(np.abs(allpts - cen), 99, axis=0).max() * 2
    scale = (tile * 0.80) / max(span, 1e-6)
    od = os.path.join(kit, 'cards')
    os.makedirs(od, exist_ok=True)
    metrics = {}
    tiles = []
    labels = []
    for v in order:
        if v not in lms: continue
        im, opx = card(lms[v], tile, scale, cen, ss=ss)       # label-free tile
        cv2.imwrite(os.path.join(od, f'{v}.png'), im)
        metrics[v] = {'open_px': round(opx, 1), 'open_norm': round(opx * scale / tile, 4)}
        tiles.append(im)
        labels.append(v)
    # atlas: grid (labels drawn here, not on the saved tiles)
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    pad = tile // 32
    sheet = np.full((rows * tile + (rows + 1) * pad, cols * tile + (cols + 1) * pad, 3), BG, np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (tile + pad)
        x = pad + c * (tile + pad)
        im = im.copy()
        cv2.putText(im, labels[i], (int(tile * 0.04), int(tile * 0.13)), cv2.FONT_HERSHEY_SIMPLEX,
                                    tile / 560, INK, max(1, tile // 300), cv2.LINE_AA)
        sheet[y:y + tile, x:x + tile] = im
    cv2.imwrite(os.path.join(od, 'atlas.png'), sheet)
    json.dump({'kit': os.path.basename(kit), 'tile': tile, 'scale': round(scale, 5),
               'order': [v for v in order if v in lms], 'metrics': metrics}, open(os.path.join(od, 'cards.json'), 'w'), indent=1)
    return od, len(tiles)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('kit')
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--order', default=None)
    ap.add_argument('--ss', type=int, default=3)
    a = ap.parse_args(argv)
    od, n = build(a.kit, a.tile, a.order.split(',') if a.order else None, a.ss)
    print(f'built {n} detailed viseme cards @ {a.tile}px -> {od}')


if __name__ == '__main__':
    main()
