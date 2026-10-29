#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_library.py — a per-character viseme MOUTH LIBRARY from one frame, with Qwen-Image-Edit-2511 (Apache-2.0).

  prep:   pick the most-closed-mouth frame of a take (MediaPipe FaceLandmarker), crop a square around the face, 1024 px
          -> D/base.png, D/base_lm.npy, D/mask.png (lips+jaw region, the ONLY area Qwen may change), D/prep.json
    lsdub mouths prep --video body.mp4 --out-dir D
  gen:    one Qwen edit per library viseme (Lightning 8 steps, seeds), original pasted back outside the feathered mask
          (pixels outside the mouth/jaw are locked to the source frame), then each take is measured (mouth open / width
          vs the closed source) and the best seed per viseme kept -> D/<vis>.png, D/library.json, D/sheet.png
    COMFY_URL=... lsdub mouths gen --out-dir D [--seeds 11 22 33]
  sheet:  re-measure / re-pick / rebuild the contact sheet without the GPU. D/picks_override.json {vis: take file} pins an
          eyeballed pick (the metric cannot see e.g. a stray painted letter).
Visemes: lsdub visemes (Oculus 15; kk/nn share the DD mouth).
"""
import argparse, json, os, sys
import numpy as np, cv2

from .._models import face_landmarker_path
MODEL = face_landmarker_path()
LIP_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
JAW = [205, 425, 98, 327, 2, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 61, 291]
SHAPES = {
    'sil': 'the lips gently closed and relaxed, neutral resting mouth',
    'PP': 'the lips closed and pressed flat together in a straight line, mouth shut, NOT puckered, not pouting',
    'FF': 'the mouth slightly open with the upper front teeth gently biting the lower lip, the lower lip tucked under the upper teeth',
    'TH': 'the mouth slightly open with the tip of the tongue just visible between the upper and lower teeth',
    'DD': 'the mouth slightly open, teeth a little apart and visible, relaxed lips, as when saying D or N',
    'SS': 'the teeth together and visible through parted, slightly spread lips, as when saying S',
    'CH': 'the lips pushed forward and slightly parted, teeth close together, as when saying SH',
    'RR': 'the lips slightly rounded and a little open, relaxed',
    'aa': 'the mouth open in a relaxed singing ah, jaw dropped moderately, upper teeth visible, natural singing, not shouting, not screaming',
    'E': 'the mouth half open with lips spread wide, singing EH, teeth visible',
    'ih': 'the mouth slightly open in a wide grin, singing EE, teeth visible',
    'oh': 'the lips rounded into an open O shape, jaw dropped, singing OH',
    'ou': 'the lips pursed into a small tight round O, puckered forward, singing OO',
}
# expected shape vs the closed source: (inner-lip gap / face height, mouth width ratio)
TARGET = {'sil': (0.0, 1.0), 'PP': (0.0, 1.0), 'FF': (0.015, 1.0), 'TH': (0.03, 1.0), 'DD': (0.03, 1.0), 'SS': (0.015, 1.06),
          'CH': (0.025, 0.88), 'RR': (0.03, 0.9), 'aa': (0.08, 1.0), 'E': (0.06, 1.08), 'ih': (0.035, 1.12), 'oh': (0.08, 0.84),
          'ou': (0.025, 0.74)}


def landmarker(video=False):
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    return mp, vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), num_faces=1,
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))


def landmarks(img, lm=None):
    mp, l = (None, lm) if lm else landmarker()
    import mediapipe as mpp
    r = l.detect(mpp.Image(image_format=mpp.ImageFormat.SRGB, data=np.ascontiguousarray(img[:, :, ::-1])))
    if not r.face_landmarks: return None
    H, W = img.shape[:2]
    return np.array([[q.x * W, q.y * H] for q in r.face_landmarks[0]])


RIGID = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 33, 133, 263, 362, 127, 356, 234, 454]   # forehead, nose bridge, eyes, temples


def register(base, raw, l=None):
    """Align a Qwen edit back onto the base crop by the RIGID landmarks (the editor sometimes re-frames the whole face,
    which pastes the mouth tens of px off). Returns (aligned, pre-shift px, residual px); aligned is None if the face is lost."""
    raw = cv2.resize(raw, base.shape[1::-1])
    pb = landmarks(base, l)
    pr = landmarks(raw, l)
    if pb is None or pr is None: return None, 1e9, 1e9
    pre = float(np.median(np.linalg.norm(pr[RIGID] - pb[RIGID], axis=1)))
    M, _ = cv2.estimateAffinePartial2D(pr[RIGID].astype(np.float32), pb[RIGID].astype(np.float32))
    al = cv2.warpAffine(raw, M, base.shape[1::-1], flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    pa = pr[RIGID] @ M[:, :2].T + M[:, 2]
    return al, pre, float(np.median(np.linalg.norm(pa - pb[RIGID], axis=1)))


def mouth_metrics(p):
    fh = np.linalg.norm(p[10] - p[152])
    fw = np.linalg.norm(p[234] - p[454])
    return {'open': float(np.linalg.norm(p[13] - p[14]) / fh), 'width': float(np.linalg.norm(p[61] - p[291]) / fw)}


def region_mask(p, shape, grow=0.12, blur=0.04):
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    pts = p[JAW + LIP_OUT]
    c = pts.mean(0)
    pts = c + (pts - c) * (1 + grow)
    cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 255)
    k = int(W * blur) | 1
    return cv2.GaussianBlur(m, (k, k), 0)


def prep(a):
    mp, l = landmarker(video=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    best = None
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        r = l.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr[:, :, ::-1])), int(i * 1000 / fps))
        if r.face_landmarks:
            H, W = fr.shape[:2]
            p = np.array([[q.x * W, q.y * H] for q in r.face_landmarks[0]])
            m = mouth_metrics(p)
            if a.frame is None or a.frame == i:
                if best is None or m['open'] < best[0]: best = (m['open'], i, fr.copy(), p)
        i += 1
    _, idx, fr, p = best
    fh = np.linalg.norm(p[10] - p[152])
    c = (p[10] + p[152]) / 2
    side = int(fh * a.margin)
    x0, y0 = int(c[0] - side / 2), int(c[1] - side / 2 + 0.08 * side)
    crop = fr[max(0, y0):y0 + side, max(0, x0):x0 + side]
    base = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_LANCZOS4)
    os.makedirs(a.out_dir, exist_ok=True)
    cv2.imwrite(f'{a.out_dir}/base.png', base)
    pb = landmarks(base)
    np.save(f'{a.out_dir}/base_lm.npy', pb)
    cv2.imwrite(f'{a.out_dir}/mask.png', region_mask(pb, base.shape, grow=0.25, blur=0.03))
    json.dump({'video': a.video, 'frame': idx, 'crop': [x0, y0, side], 'size': a.size, 'face_h_px': float(fh),
               'base_metrics': mouth_metrics(pb)}, open(f'{a.out_dir}/prep.json', 'w'), indent=1)
    print(f'frame {idx} (open {best[0]:.3f}) face {fh:.0f}px -> crop {side}px -> {a.size}px; {a.out_dir}')


def gen(a):
    from ..backends import comfy_backend as cb
    run, label = cb.make_client()
    D = a.out_dir
    os.makedirs(f'{D}/takes', exist_ok=True)
    for v in (a.only or SHAPES):
        (f'Change ONLY the mouth and jaw: {SHAPES[v]}. Keep everything else exactly the same as image 1: the same '
                  f'{a.keep}, '
                  'the same lighting, colours, camera angle, framing and sharpness. Photoreal, natural lips. No text, no letters, no captions.')
        for s in a.seeds:
            dst = f'{D}/takes/{v}_s{s}.png'
            if os.path.exists(dst): continue
            paths = run('qwen-mouth-viseme.json', '60', f'{D}/takes',
                        upload_paths=[f'{D}/base.png'],
                        VISEME=SHAPES[v], KEEP=a.keep, SEED=s)
            src = None
            for p in paths:
                if p.endswith('.png'):
                    src = p
                    break
            if src is None:
                continue
            if not a.no_register:   # register by the rigid landmarks before pasting (drift guard)
                al, pre, res = register(cv2.imread(f'{D}/base.png'), cv2.imread(src))
                if al is not None:
                    reg_dst = src.replace('.png', '_reg.png')
                    cv2.imwrite(reg_dst, al)
                    src = reg_dst
                json.dump({'pre_shift_px': pre, 'residual_px': res}, open(src.replace('.png', '_reg.json'), 'w'))
            from ..backends import qwen_graph as qe
            qe.paste_back(f'{D}/base.png', src, f'{D}/mask.png', src.replace('.png', '_paste.png'))
            print('gen', v, s, label, flush=True)
    sheet(a)


def sheet(a):
    D = a.out_dir
    base = cv2.imread(f'{D}/base.png')
    _, l = landmarker()
    pb = np.load(f'{D}/base_lm.npy')
    m0 = mouth_metrics(pb)
    mask = cv2.imread(f'{D}/mask.png', 0) / 255.0
    outside = mask < 0.02
    lib = {}
    ov = json.load(open(f'{D}/picks_override.json')) if os.path.exists(f'{D}/picks_override.json') else {}
    for v in SHAPES:
        cands = []
        for fn in sorted(os.listdir(f'{D}/takes')):   # an override may borrow ANOTHER viseme's take (e.g. DD -> a clean CH take)
            if not ((fn.startswith(v + '_s') and fn.endswith('_paste.png')) or fn == ov.get(v)): continue
            img = cv2.imread(f'{D}/takes/{fn}')
            rf = f'{D}/takes/{fn.replace("_paste", "_reg")}'
            raw = cv2.imread(rf if os.path.exists(rf) else f'{D}/takes/{fn.replace("_paste", "")}')
            rj = f'{D}/takes/{fn.replace("_paste.png", "_reg.json")}'
            res = json.load(open(rj))['residual_px'] if os.path.exists(rj) else 0.0
            raw = cv2.resize(raw, base.shape[1::-1])
            p = landmarks(img, l)
            if p is None: continue
            m = mouth_metrics(p)
            to, tw = TARGET[v]
            err = abs(m['open'] - to) / 0.03 + abs(m['width'] / m0['width'] - tw) / 0.08
            # identity/scene drift of the RAW edit outside the mouth region (paste hides it, but big drift = a bad edit)
            drift = float(np.abs(raw.astype(float) - base.astype(float))[outside].mean())
            if res > getattr(a, 'max_drift', 1e9) and fn != ov.get(v): continue   # face moved under the mouth: reject
            cap = getattr(a, 'open_cap', 0.0)
            if cap and m['open'] > cap and fn != ov.get(v): continue            # wider than the face-constraints limit
            cands.append({'file': fn, 'open': round(m['open'], 4), 'width_ratio': round(m['width'] / m0['width'], 3),
                          'err': round(err + drift / 20, 3), 'raw_drift': round(drift, 2), 'reg_residual_px': round(res, 2)})
        if not cands: continue
        best = next((c for c in cands if c['file'] == ov.get(v)), None) or min(cands, key=lambda c: c['err'])
        lib[v] = {'pick': best, 'takes': cands, 'override': v in ov}
        img = cv2.imread(f'{D}/takes/{best["file"]}')
        cv2.imwrite(f'{D}/{v}.png', img)
        np.save(f'{D}/{v}_lm.npy', landmarks(img, l))
    json.dump({'base_metrics': m0, 'library': lib}, open(f'{D}/library.json', 'w'), indent=1)
    # BuildLipKit step: a DETAILED, face-specific card atlas of every mouth shape (from this face's own
    # captured landmarks), at 2-3x the overlay glyph's resolution -> D/cards/{<v>.png, atlas.png, cards.json}
    try:
        from . import mouth_cards
        od, nc = mouth_cards.build(D, tile=getattr(a, 'card_tile', 512))
        print(f'detailed viseme cards: {nc} @ {getattr(a, "card_tile", 512)}px -> {od}')
    except Exception as e:
        print(f'viseme cards: skipped ({type(e).__name__}: {e})', file=sys.stderr)
    # contact sheet: mouth close-ups (lower face) with labels
    tiles = []
    for v in ['base'] + list(SHAPES):
        f = f'{D}/base.png' if v == 'base' else f'{D}/{v}.png'
        if not os.path.exists(f): continue
        im = cv2.imread(f)
        H = im.shape[0]
        t = cv2.resize(im[int(H * 0.30):int(H * 0.95), int(H * 0.15):int(H * 0.80)], (360, 360))
        lab = v if v == 'base' else f"{v} o={lib[v]['pick']['open']:.3f} w={lib[v]['pick']['width_ratio']:.2f}"
        cv2.rectangle(t, (0, 0), (360, 30), (0, 0, 0), -1)
        cv2.putText(t, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % 7: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 7]) for i in range(0, len(tiles), 7)]
    cv2.imwrite(f'{D}/sheet.png', np.vstack(rows))
    # all takes sheet for eyeballing seed choice
    allt = []
    for v in SHAPES:
        for c in lib.get(v, {}).get('takes', []):
            im = cv2.imread(f"{D}/takes/{c['file']}")
            H = im.shape[0]
            t = cv2.resize(im[int(H * 0.30):int(H * 0.95), int(H * 0.15):int(H * 0.80)], (240, 240))
            mark = '*' if c['file'] == lib[v]['pick']['file'] else ''
            cv2.putText(t, f"{mark}{c['file'][:-10]} e{c['err']:.1f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            allt.append(t)
    if allt:
        n = max(len(lib[v]['takes']) for v in lib)
        while len(allt) % n: allt.append(np.zeros_like(allt[0]))
        cv2.imwrite(f'{D}/sheet_takes.png', np.vstack([np.hstack(allt[i:i + n]) for i in range(0, len(allt), n)]))
    for v, e in lib.items(): print(v, e['pick'])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('prep')
    p.add_argument('--video', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--frame', type=int)
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--margin', type=float, default=1.5)
    g = sp.add_parser('gen')
    g.add_argument('--out-dir', required=True)
    g.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    g.add_argument('--only', nargs='+')
    g.add_argument('--no-register', action='store_true')
    g.add_argument('--keep', default='person, same face shape, skin tone, hair, eyewear and makeup',
                   help='who to keep, after "the same ": describe the character (face, skin, hair, eyewear)')
    s = sp.add_parser('sheet')
    s.add_argument('--out-dir', required=True)
    for q in (g, s):
        q.add_argument('--max-drift', type=float, default=12.0, help='reject takes whose rigid-landmark residual (px at 1024) exceeds this')
        q.add_argument('--open-cap', type=float, default=0.0, help='reject takes opening wider than this (inner gap / face h; 0 = off)')
    a = ap.parse_args(argv)
    {'prep': prep, 'gen': gen, 'sheet': sheet}[a.cmd](a)


if __name__ == '__main__':
    main()
