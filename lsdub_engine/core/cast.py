#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""cast.py: `lsdub cast`, find the known characters (the singers) in every shot, so the dub knows WHO to animate WHERE.

  enroll  reference images (canon, head turns, sheets) -> a cast db: per character, DINOv2 head-crop embeddings
          + a colour signature of the head and costume
  scan    a video or a folder of clips: MediaPipe faces per frame -> IoU tracks -> embed every --stride frames ->
          one identity per TRACK (character + confidence, or 'unknown'), with box, face px, yaw, mouth-occluded and
          dub_safe per frame; a contact sheet with labelled boxes (--sheet) and labelled review videos (--review)
  plan    cast scan x the word timing (lsdub align json) x who sings each cut (the shooting script) -> the dub plan:
          (cut, character, frame span, words, dub_safe) = exactly where the singing gets animated
  hook    DRY RUN: print the lsdub visemes / paint / continuity commands for every dub-safe plan span (runs nothing)

  lsdub cast enroll --root characters --char singer_a --char singer_b --char extra_1 \\
                    --out cast_db.npz [--add singer_a=shot_frame.png]
  lsdub cast scan   --db cast_db.npz --clips clips --out cast.json --sheet cast_sheet.jpg [--review dir]
  lsdub cast plan   --cast cast.json --cut cut.json --align align.json \\
                    --script SCRIPT.md --song-offset 12 --out dub_plan.json --md dub_plan.md
  lsdub cast hook   --plan dub_plan.json --clips clips --audio lead.wav --work castdub

Models (all commercially permissive): MediaPipe Face Landmarker (Apache-2.0) for faces, landmarks and head pose;
DINOv2 ViT-B/14 (facebook/dinov2-base, Apache-2.0 code AND weights) as the embedder, run on CPU through OpenCV's DNN
from an ONNX export (`lsdub cast export-onnx`, needs torch + transformers + onnx once; LIPS_CAST_MODEL or --model).
No face-recognition weights: identity comes from the general-purpose embedder plus colour, which also holds up on
stylised characters.
"""
import argparse, glob, json, os, re, sys
import cv2, numpy as np

from .face_scan import euler
from .._models import face_landmarker_path, CACHE
from .. import presets as P

MODEL = os.environ.get('LIPS_CAST_MODEL', os.path.join(CACHE, 'dinov2_base_cast.onnx'))   # the CLI's ONNX embedder
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
CHEEKS = [50, 280, 205, 425, 118, 347]
# dub-safe rule (docs/dub-safe-closeups.md, FACE_CONSTRAINTS): face >= 300 px tall at 1080p, |yaw| <= 16, mouth clear
SAFE_PX, SAFE_YAW = 300.0, 16.0
NEAR_PX, NEAR_YAW = 200.0, 25.0     # 'near': a reframe / zoom-crop (lsdub zoom) away from dub-safe
VARIANTS = re.compile(r'_(egirl|office)$')


# ---------------------------------------------------------------------------------------------------- faces
YUNET = None   # resolved when a detector is made (_models.yunet_path), so the pack can point the cache at models/lsdub


class Faces:
    """Two stages. YuNet (OpenCV Zoo, MIT) finds faces at any scale in the full frame (MediaPipe's short-range
    detector misses anything under ~1/5 of the frame); then the MediaPipe Face Landmarker (Apache-2.0) runs on a
    square crop around each face for 478 landmarks + head pose. A face the landmarker cannot read keeps its YuNet box
    and a 5-point yaw estimate, with lm=None (so it can never be dub_safe)."""
    def __init__(self, num=6, upscale=1.0, conf=0.3, det_thr=0.6, min_box=18):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self.mp, self.min_box, self.det_thr = mp, min_box, det_thr
        from .._models import yunet_path
        self.det = cv2.FaceDetectorYN.create(YUNET or yunet_path(), '', (320, 320), det_thr, 0.3, 500)
        self.l = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_landmarker_path()), running_mode=vision.RunningMode.IMAGE,
            num_faces=2, output_facial_transformation_matrixes=True,
            min_face_detection_confidence=conf, min_face_presence_confidence=conf))

    def _landmarks(self, bgr, box):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        s = 2.4 * max(w, h)
        S = 384
        crop = _square(bgr, cx, cy, s, S)
        r = self.l.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop[:, :, ::-1])))
        best = None
        for lm, M in zip(r.face_landmarks or [], r.facial_transformation_matrixes or []):
            p = np.array([[q.x * S, q.y * S] for q in lm], np.float32)
            d = np.linalg.norm(p.mean(0) - S / 2)
            if d < 0.25 * S and (best is None or d < best[0]):
                best = (d, p, np.array(M))
        if best is None:
            return None, None
        p = best[1] * (s / S) + np.float32([cx - s / 2, cy - s / 2])
        return p, best[2]

    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        self.det.setInputSize((W, H))
        _, dets = self.det.detect(bgr)
        out = []
        for d in (dets if dets is not None else []):
            x, y, w, h = map(float, d[:4])
            if h < self.min_box:
                continue
            p, M = self._landmarks(bgr, (x, y, w, h))
            if p is not None:
                f = face_record(p, M, H)
            else:
                k = d[4:14].reshape(5, 2)                                   # eyes, nose, mouth corners
                eye = k[:2].mean(0)
                iod = np.linalg.norm(k[0] - k[1]) + 1e-6
                yaw = float(np.degrees(np.arcsin(np.clip((k[2, 0] - eye[0]) / iod, -1, 1))) * 1.2)
                fh = 1.1 * h
                f = {'box': [x, y, x + w, y + h], 'lm': None, 'yaw': yaw, 'pitch': 0.0, 'roll': 0.0,
                     'face_h': fh, 'face_px1080': fh * 1080.0 / H}
            f['det'] = round(float(d[14]), 3)
            f['det_box'] = [x, y, x + w, y + h]
            out.append(f)
        return out


def face_record(p, M, H):
    yaw, pitch, roll = euler(M) if M is not None else (0.0, 0.0, 0.0)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    fh = float(np.linalg.norm(p[10] - p[152]))
    return {'box': [float(x0), float(y0), float(x1), float(y1)], 'lm': p, 'yaw': yaw, 'pitch': pitch, 'roll': roll,
            'face_h': fh, 'face_px1080': fh * 1080.0 / H}


def head_crop(img, box, size=224):
    """Square crop around the face box, grown to take in the hair/headwear (the strongest cue on these characters)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.15 * h
    s = 1.9 * max(w, h)
    return _square(img, cx, cy, s, size)


def face_crop(img, box, size=224):
    """Tight face crop (no hair, no neighbours): the second embedding view, so a big neighbour's hair in the head
    crop cannot carry the vote on its own."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return _square(img, (x0 + x1) / 2, (y0 + y1) / 2, 1.15 * max(w, h), size)


def body_crop(img, box, size=96):
    """The costume just below the chin: collar, jacket, vest."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return _square(img, (x0 + x1) / 2, y1 + 0.9 * h, 1.6 * max(w, h), size)


def _square(img, cx, cy, s, size):
    H, W = img.shape[:2]
    M = np.float32([[size / s, 0, size / 2 - cx * size / s], [0, size / s, size / 2 - cy * size / s]])
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))


# ---------------------------------------------------------------------------------------------------- embed + colour
class Embedder:
    def __init__(self, model=MODEL):
        if not os.path.exists(model):
            raise SystemExit(f'cast: no embedder at {model}; run `lsdub cast export-onnx --out {model}` '
                             '(needs torch + transformers + onnx once) or pass --model')
        self.net = cv2.dnn.readNetFromONNX(model)

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, 1536), np.float32)
        x = np.stack([((c[:, :, ::-1].astype(np.float32) / 255.0) - MEAN) / STD for c in crops]).transpose(0, 3, 1, 2)
        out = []
        for i in range(0, len(x), 16):
            self.net.setInput(np.ascontiguousarray(x[i:i + 16]))
            out.append(self.net.forward())
        e = np.concatenate(out).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


class TorchEmbedder:
    """the same DINOv2 embedding ([CLS, mean patch], 1536-d, same preprocessing) run by torch on a chosen device: the
    ComfyUI nodes pass ComfyUI's device. Weights load lazily (Hugging Face, into cache_dir) on first use; offload()
    moves them back to the CPU so the card is free for the render."""

    def __init__(self, device, hf='facebook/dinov2-base', cache_dir=None, offload_device='cpu'):
        import torch
        from transformers import AutoModel
        self.torch, self.device, self.offload_device = torch, device, offload_device
        self.m = AutoModel.from_pretrained(hf, cache_dir=cache_dir).eval().to(device)

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, 1536), np.float32)
        torch = self.torch
        if next(self.m.parameters()).device != torch.device(self.device): self.m.to(self.device)
        x = np.stack([((c[:, :, ::-1].astype(np.float32) / 255.0) - MEAN) / STD for c in crops]).transpose(0, 3, 1, 2)
        out = []
        with torch.inference_mode():
            for i in range(0, len(x), 16):
                h = self.m(pixel_values=torch.from_numpy(np.ascontiguousarray(x[i:i + 16])).to(self.device)).last_hidden_state
                out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
        e = np.concatenate(out).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

    def offload(self):
        self.m.to(self.offload_device)


def make_embedder(model=None, device=None, cache_dir=None, offload_device='cpu'):
    """device None / 'cpu' -> the ONNX embedder on OpenCV (the CLI default); any other device -> TorchEmbedder."""
    if device is None or str(device) == 'cpu':
        return Embedder(model or MODEL)
    return TorchEmbedder(device, cache_dir=cache_dir, offload_device=offload_device)


def colour_sig(crop):
    """Hue x saturation histogram of the saturated pixels + a share for near-black / near-white (L1 = 1)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    col = (s > 70) & (v > 50)
    hist = np.histogram2d(h[col], s[col], bins=[18, 2], range=[[0, 180], [70, 256]])[0].ravel()
    extra = np.array([((v <= 50)).sum(), ((s <= 40) & (v > 200)).sum()], np.float64)
    sig = np.concatenate([hist, 0.5 * extra]).astype(np.float32)
    return sig / (sig.sum() + 1e-8)


def colour_sim(a, b):
    """Bhattacharyya coefficient of two signatures (1 = same palette)."""
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))


# ---------------------------------------------------------------------------------------------------- enroll
def ref_images(root, char):
    d = os.path.join(root, char)
    pats = ['canon.png', '*_canon_face.png', '*_sheet.png', 'card.png', 'hero_s*.png', 'heads/*.png', 'sheet/*.png',
            'shutter/*.png', 'clear/*.png', 'clean/*.png']
    seen, out = set(), []
    for p in pats:
        for f in sorted(glob.glob(os.path.join(d, p))):
            if f not in seen and 'back' not in os.path.basename(f):
                seen.add(f)
                out.append(f)
    return out


def identity_of(char):
    return VARIANTS.sub('', char)


def enroll(images, faces, emb, max_faces=6):
    """images: [(identity, path)] -> dict of arrays: name, src, emb, head colour, body colour, yaw."""
    rows = {'name': [], 'src': [], 'emb': [], 'emb_f': [], 'col_h': [], 'col_b': [], 'yaw': []}
    for name, path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        if max(img.shape[:2]) > 1600:
            img = cv2.resize(img, None, fx=1600 / max(img.shape[:2]), fy=1600 / max(img.shape[:2]), interpolation=cv2.INTER_AREA)
        fs = sorted(faces(img), key=lambda f: -f['face_h'])[:max_faces]
        fs = [f for f in fs if f['face_h'] >= 24]
        if not fs:
            print(f'  enroll {name}: no face in {path}', file=sys.stderr)
            continue
        crops = [head_crop(img, f['det_box']) for f in fs]
        ef = emb([face_crop(img, f['det_box']) for f in fs])
        for f, c, e, e2 in zip(fs, crops, emb(crops), ef):
            rows['name'].append(name)
            rows['src'].append(path)
            rows['emb'].append(e)
            rows['emb_f'].append(e2)
            rows['col_h'].append(colour_sig(c))
            rows['col_b'].append(colour_sig(body_crop(img, f['det_box'])))
            rows['yaw'].append(f['yaw'])
    return {k: np.array(v) for k, v in rows.items()}


# ---------------------------------------------------------------------------------------------------- identify
def score_track(db, E, CH, CB, w_col=0.3, topk=3, EF=None):
    """Per identity: mean of the top-k cosine sims (over track samples x refs) blended with the colour vote.
    Returns {name: score} (higher = better)."""
    names = sorted(set(db['name'].tolist()))
    sims = E @ db['emb'].T                                          # (samples, refs)
    simf = EF @ db['emb_f'].T if EF is not None and 'emb_f' in db else None
    out = {}
    for n in names:
        m = db['name'] == n
        s = np.sort(sims[:, m].max(0))[::-1][:topk].mean()         # best refs, each at its best sample
        if simf is not None:
            s = 0.5 * s + 0.5 * np.sort(simf[:, m].max(0))[::-1][:topk].mean()
        ch = np.mean([max(colour_sim(c, r) for r in db['col_h'][m]) for c in CH])
        cb = np.mean([max(colour_sim(c, r) for r in db['col_b'][m]) for c in CB])
        out[n] = float((1 - w_col) * s + w_col * (0.6 * ch + 0.4 * cb))
    return out


def family(name, preset=None):
    """report name: the preset's family for near-identical extras, else the name without a _N suffix."""
    return P.get(preset).families.get(name) or re.sub(r'_\d+$', '', name)


def decide(scores, thr=0.45, margin=0.03, min_conf=0.5, preset=None):
    """-> (character | 'unknown', confidence 0..1, runner-up). The margin is taken against the best identity OUTSIDE
    the winner's family: near-identical extras (a preset's families) are alike by design, so a close
    within-family runner-up does not make the face unknown; the family is reported alongside."""
    r = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = r[0]
    other = [kv for kv in r[1:] if family(kv[0], preset) != family(best, preset)]
    s2 = other[0][1] if other else 0.0
    conf = float(np.clip((s1 - thr) / 0.15, 0, 1) * np.clip((s1 - s2) / (2 * margin), 0, 1))
    if s1 < thr or s1 - s2 < margin or conf < min_conf:
        return 'unknown', round(conf, 3), best
    return best, round(conf, 3), other[0][0] if other else None


# ---------------------------------------------------------------------------------------------------- track
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def link(per_frame, min_iou=0.3, max_gap=3):
    """Greedy IoU tracker. per_frame: [[face, ...] per frame] -> [[(frame, face), ...] per track]."""
    tracks, live = [], []
    for i, fs in enumerate(per_frame):
        used = set()
        for t in sorted(live, key=lambda t: -t[-1][1]['face_h']):
            last = t[-1]
            if i - last[0] > max_gap + 1:
                continue
            cand = [(iou(last[1].get('det_box', last[1]['box']), f.get('det_box', f['box'])), j) for j, f in enumerate(fs) if j not in used]
            cand = [c for c in cand if c[0] >= min_iou]
            if cand:
                _, j = max(cand)
                used.add(j)
                t.append((i, fs[j]))
        for j, f in enumerate(fs):
            if j not in used:
                t = [(i, f)]
                tracks.append(t)
                live.append(t)
        live = [t for t in live if i - t[-1][0] <= max_gap]
    return tracks


# ---------------------------------------------------------------------------------------------------- mouth occlusion
def mouth_occluded(img, lm, thr=0.35):
    """Heuristic: share of the lip region that is neither skin-like, lip-like nor mouth-dark (a mic, a hand edge, a
    prop). The landmarker always hallucinates a mouth, so we look at the pixels. -> (flag, share)."""
    H, W = img.shape[:2]
    hull = cv2.convexHull(lm[LIPS].astype(np.int32))
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    if m.sum() == 0:
        return False, 0.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ck = [lab[int(np.clip(lm[i, 1], 0, H - 1)), int(np.clip(lm[i, 0], 0, W - 1))] for i in CHEEKS]
    skin = np.median(np.array(ck), 0)
    px = lab[m > 0]
    d_skin = np.linalg.norm(px - skin, axis=1)
    lipish = (px[:, 1] > skin[1] + 4)                                   # redder than the cheeks
    dark = px[:, 0] < 0.55 * skin[0]                                    # mouth interior
    foreign = (d_skin > 28) & ~lipish & ~dark
    share = float(foreign.mean())
    return share > thr, round(share, 3)


# ---------------------------------------------------------------------------------------------------- scan
def read_frames(video):
    cap = cv2.VideoCapture(video)
    F = []
    while True:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F, cap.get(cv2.CAP_PROP_FPS) or 25.0


def scan_video(video, db, model=MODEL, stride=2, min_len=4, min_px=40, thr=0.45, margin=0.03, w_col=0.3, preset=None):
    F, fps = read_frames(video)
    faces, emb = Faces(), Embedder(model)
    per = [[f for f in faces(fr) if f['face_h'] >= min_px * fr.shape[0] / 1080.0] for fr in F]
    out = []
    for t in link(per):
        if len(t) < min_len:
            continue
        samp = t[::stride] if len(t) > stride else t
        samp = sorted(samp, key=lambda x: -x[1]['face_h'])[:24]
        crops = [head_crop(F[i], f['det_box']) for i, f in samp]
        E = emb(crops)
        EF = emb([face_crop(F[i], f['det_box']) for i, f in samp])
        CH = [colour_sig(c) for c in crops]
        CB = [colour_sig(body_crop(F[i], f['det_box'])) for i, f in samp]
        sc = score_track(db, E, CH, CB, w_col=w_col, EF=EF)
        who, conf, runner = decide(sc, thr, margin, preset=preset)
        frames = []
        for i, f in t:
            occ, share = mouth_occluded(F[i], f['lm']) if f['lm'] is not None else (None, None)
            safe = f['lm'] is not None and f['face_px1080'] >= SAFE_PX and abs(f['yaw']) <= SAFE_YAW and not occ
            frames.append({'f': i, 'box': [round(v, 1) for v in f['box']], 'face_px': round(f['face_px1080'], 1),
                           'yaw': round(f['yaw'], 1), 'pitch': round(f['pitch'], 1), 'mouth_occluded': occ,
                           'occ_share': share, 'landmarks': f['lm'] is not None, 'dub_safe': bool(safe)})
        out.append({'character': who, 'family': family(who, preset) if who != 'unknown' else None, 'confidence': conf, 'runner_up': runner,
                    'scores': {k: round(v, 3) for k, v in sorted(sc.items(), key=lambda kv: -kv[1])},
                    'f0': t[0][0], 'f1': t[-1][0], 'n': len(t),
                    'face_px_med': round(float(np.median([x['face_px'] for x in frames])), 1),
                    'yaw_med': round(float(np.median([x['yaw'] for x in frames])), 1),
                    'dub_safe_frac': round(float(np.mean([x['dub_safe'] for x in frames])), 3),
                    'dub_safe_runs': runs([x['f'] for x in frames if x['dub_safe']]),
                    'frames': frames})
    out.sort(key=lambda t: (t['f0'], -t['face_px_med']))
    return {'video': video, 'fps': fps, 'n_frames': len(F), 'size': [F[0].shape[1], F[0].shape[0]] if F else None,
            'tracks': out}


def runs(fs, gap=1):
    """sorted frame list -> [[a, b], ...] inclusive runs (a gap of <= gap missing frames is bridged)."""
    out = []
    for f in fs:
        if out and f - out[-1][1] <= gap + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def _scan_one(args):
    video, db_path, model, kw = args
    cv2.setNumThreads(2)
    return scan_video(video, load_db(db_path), model, **kw)


def load_db(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------------------- review
COLOURS = {'unknown': (160, 160, 160)}


def _col(name, preset=None):
    c = {**COLOURS, **P.get(preset).colours}
    if name in c: return c[name]
    h = abs(hash(name)) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw(img, tracks, i):
    for t in tracks:
        fr = next((x for x in t['frames'] if x['f'] == i), None)
        if fr is None: continue
        c = _col(t['character'])
        x0, y0, x1, y1 = map(int, fr['box'])
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 3 if fr['dub_safe'] else 1)
        lab = f"{t['character']} {t['confidence']:.2f}{' SAFE' if fr['dub_safe'] else ''}{' OCC' if fr['mouth_occluded'] else ''}"
        cv2.putText(img, lab, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, lab, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv2.LINE_AA)
    return img


def contact_sheet(results, path, cols=6, tw=448):
    tiles = []
    for k, r in enumerate(results):
        F, _ = read_frames(r['video'])
        if not F: continue
        # the frame with the most / biggest labelled faces, else the middle
        best = max(range(len(F)), key=lambda i: sum(next((x['face_px'] for x in t['frames'] if x['f'] == i), 0)
                                                    for t in r['tracks'])) if r['tracks'] else len(F) // 2
        img = draw(F[best].copy(), r['tracks'], best)
        th = int(tw * img.shape[0] / img.shape[1])
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        lab = f"{os.path.basename(r['video'])} f{best}  " + ', '.join(sorted({t['character'] for t in r['tracks']}))
        cv2.rectangle(img, (0, 0), (tw, 18), (0, 0, 0), -1)
        cv2.putText(img, lab[:60], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles: return
    th = tiles[0].shape[0]
    while len(tiles) % cols: tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def track_sheet(results, path, cols=10, ts=160, min_px=60):
    """One tile per track (its biggest frame's head crop): the by-eye ID check. Label: clip, frames, who, conf, px."""
    tiles = []
    for r in results:
        F = None
        for k, t in enumerate(r['tracks']):
            if t['character'] == 'unknown' and t['face_px_med'] < min_px: continue
            if F is None: F, _ = read_frames(r['video'])
            fr = max(t['frames'], key=lambda x: x['face_px'])
            b = fr['box']
            img = head_crop(F[fr['f']], b, ts)
            c = _col(t['character'])
            cv2.rectangle(img, (0, 0), (ts - 1, ts - 1), c, 4 if t['dub_safe_frac'] > 0 else 2)
            cv2.rectangle(img, (0, ts - 34), (ts, ts), (0, 0, 0), -1)
            l1 = f"{os.path.splitext(os.path.basename(r['video']))[0]}#{k} f{t['f0']}-{t['f1']} {t['face_px_med']:.0f}px"
            l2 = f"{t['character'][:9]} {t['confidence']:.2f}" + (f" S{t['dub_safe_frac']:.0%}" if t['dub_safe_frac'] else '')
            for j, l in enumerate((l1, l2)):
                cv2.putText(img, l, (3, ts - 21 + 14 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
    if not tiles: return
    while len(tiles) % cols: tiles.append(np.zeros((ts, ts, 3), np.uint8))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def review_video(r, path):
    F, fps = read_frames(r['video'])
    if not F: return
    H, W = F[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for i, f in enumerate(F): vw.write(draw(f.copy(), r['tracks'], i))
    vw.release()


# ---------------------------------------------------------------------------------------------------- plan
def sing_re(names):
    """'<Name> sings' / 'The duo sing' in a beat cell, for the preset's script names."""
    alt = '|'.join(re.escape(k) for k in sorted(names, key=len, reverse=True))
    return re.compile(r'\b(' + alt + r'|duo)\b[^|.;]{0,40}?\bsing', re.I)


def script_singers(text, preset=None):
    """Shooting-script beat tables -> {cut index: set of singers} from the 'Story beat' cell ('Alice sings', 'Bob leans
    across the podium singing', 'The duo sing'). Cuts without a sung mention are absent."""
    pre = P.get(preset)
    names = {k.lower(): v for k, v in pre.names.items()}
    if not names: return {}
    rx, allv = sing_re(names), set(pre.singers or names.values())
    out, beat_col = {}, None
    for line in text.splitlines():
        if line.startswith('#'):
            beat_col = None
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')] if line.startswith('|') else []
        if cells and cells[0] == '#' and 'Story beat' in cells and 'Shot' in cells:   # a beat-table header
            beat_col = cells.index('Story beat')
            continue
        m = re.match(r'\|\s*(\d+)\s*\|', line)
        if not m or beat_col is None or len(cells) <= beat_col: continue
        beat = cells[beat_col]
        who = set()
        for g in rx.findall(beat):
            g = g.lower()
            who |= allv if g == 'duo' else {names[g]}
        if who: out[int(m.group(1))] = who
    return out


def words_in(words, t0, t1):
    return [w for w in words if w['end'] > t0 and w['start'] < t1 and re.search(r'[a-z]', w['w'].lower())]


def plan(cast, cut, words, singers, song_offset=12.0, fps=25.0, speed=None, chars=None, min_words=1, preset=None):
    """One entry per (cut, track of a singer) whose frames overlap sung words. A clip's frame k plays at video time
    cut.start + k / fps (the delivery cut is already speed-adjusted). Words are in song time: video = song + offset."""
    by_clip = {int(os.path.splitext(os.path.basename(r['video']))[0]): r for r in cast['results']
               if os.path.splitext(os.path.basename(r['video']))[0].isdigit()}
    chars = set(chars) if chars else P.singer_set(preset, cast)
    out = []
    for ci, c in enumerate(cut['cut']):
        r = by_clip.get(ci)
        if r is None: continue
        sing = singers.get(ci, set())
        for t in r['tracks']:
            if t['character'] not in chars: continue
            v0, v1 = c['start'] + t['f0'] / fps, c['start'] + (t['f1'] + 1) / fps
            ws = words_in(words, v0 - song_offset, v1 - song_offset)
            if len(ws) < min_words: continue
            safe = {x['f'] for x in t['frames'] if x['dub_safe']}
            near = {x['f'] for x in t['frames'] if x.get('landmarks') and x['face_px'] >= NEAR_PX and abs(x['yaw']) <= NEAR_YAW
                    and not x['mouth_occluded']}
            wl = []
            for w in ws:
                a = max(t['f0'], int(np.floor((w['start'] + song_offset - c['start']) * fps)))
                b = min(t['f1'], int(np.ceil((w['end'] + song_offset - c['start']) * fps)) - 1)
                n = max(1, b - a + 1)
                wl.append({'w': w['w'], 'song_t': [round(w['start'], 3), round(w['end'], 3)], 'frames': [a, b],
                           'dub_safe': sum(f in safe for f in range(a, b + 1)) / n >= 0.8})
            sung_frames = sorted({f for x in wl for f in range(x['frames'][0], x['frames'][1] + 1)})
            sf = float(np.mean([f in safe for f in sung_frames])) if sung_frames else 0.0
            nf = float(np.mean([f in near for f in sung_frames])) if sung_frames else 0.0
            role = 'singer' if t['character'] in sing else ('not_singing' if sing else 'unassigned')
            out.append({'cut': ci, 'shot': c['shot'], 'clip': r['video'], 'character': t['character'],
                        'confidence': t['confidence'], 'role': role, 'frames': [t['f0'], t['f1']],
                        'video_t': [round(v0, 3), round(v1, 3)], 'face_px_med': t['face_px_med'], 'yaw_med': t['yaw_med'],
                        'words': ' '.join(w['w'] for w in wl), 'word_spans': wl,
                        'sung_frames': [sung_frames[0], sung_frames[-1]] if sung_frames else None,
                        'dub_safe_frac': round(sf, 3), 'dub_safe': sf >= 0.8 and role == 'singer',
                        'near_frac': round(nf, 3), 'near_safe': nf >= 0.8 and role == 'singer',
                        'safe_runs': [rr for rr in t['dub_safe_runs'] if any(rr[0] <= f <= rr[1] for f in sung_frames)]})
    return out


def plan_md(P, summary):
    L = ['# Dub plan', '', '| cut | shot | who | role | conf | frames | face px | yaw | safe | words |', '|---|---|---|---|---|---|---|---|---|---|']
    for p in P:
        L.append(f"| {p['cut']} | {p['shot']} | {p['character']} | {p['role']} | {p['confidence']:.2f} | {p['frames'][0]}-{p['frames'][1]} | "
                 f"{p['face_px_med']:.0f} | {p['yaw_med']:+.0f} | {'**YES**' if p['dub_safe'] else ('near' if p['near_safe'] else ('no (' + format(p['dub_safe_frac'], '.0%') + ')'))} | {p['words']} |")
    L += ['', '## Summary', '', '```', json.dumps(summary, indent=1), '```']
    return '\n'.join(L) + '\n'


def summarise(P):
    s = {}
    for p in P:
        d = s.setdefault(p['character'], {'spans': 0, 'singer_spans': 0, 'dub_safe': 0, 'near_safe': 0, 'unassigned': 0, 'not_singing': 0})
        d['spans'] += 1
        d['dub_safe'] += p['dub_safe']
        d['near_safe'] += p['near_safe']
        d[{'singer': 'singer_spans', 'unassigned': 'unassigned', 'not_singing': 'not_singing'}[p['role']]] += 1
    return s


# ---------------------------------------------------------------------------------------------------- hook
def hook(P, clips, audio, work, align, fps=25.0, song_offset=12.0, cut=None, tier='safe'):
    """Dry-run commands for each dub-safe span. Nothing is run here."""
    cmds = []
    for p in P:
        ok = p['dub_safe'] or (tier == 'near' and p.get('near_safe'))
        if not ok or not p['sung_frames']: continue
        runs_ = (p['safe_runs'] if p['dub_safe'] else None) or [p['sung_frames']]
        a, b = runs_[0][0], runs_[-1][1]
        n = b - a + 1
        d = os.path.join(work, f"{p['cut']:03d}_{p['character']}")
        c0 = cut['cut'][p['cut']]['start'] if cut else p['video_t'][0] - p['frames'][0] / fps
        t0 = round(c0 + a / fps - song_offset, 3)
        clip = p['clip']
        seg = f'{d}/take.mp4'
        note = '' if p['dub_safe'] else '  [NEAR: face under 300 px@1080 -> lsdub zoom (crop/upscale) or re-frame first]'
        cmds.append([f'# cut {p["cut"]} {p["shot"]} {p["character"]} frames {a}-{b} "{p["words"]}"{note}',
                     f'mkdir -p {d}',
                     f"ffmpeg -y -i {clip} -vf \"select='between(n\\,{a}\\,{b})',setpts=N/FRAME_RATE/TB\" -an {seg}",
                     f'lsdub visemes --align {align} --t0 {t0} --frames {n} --audio {audio} --out {d}/visemes.json',
                     f'lsdub paint --video {seg} --track {d}/visemes.json --mode warp --out {d}/painted.mp4   # CPU; paint follows ONE face: check it locks onto this character',
                     f'lsdub continuity --take {seg} --render {d}/painted.mp4 --out {d}/cont.mp4 --json {d}/cont.json'])
    return cmds


# ---------------------------------------------------------------------------------------------------- cli
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog='lsdub cast')
    sp = ap.add_subparsers(dest='cmd', required=True)
    e = sp.add_parser('enroll')
    e.add_argument('--root', required=True, help='characters/ dir: <root>/<char>/canon.png ...')
    e.add_argument('--char', action='append', default=[])
    e.add_argument('--add', action='append', default=[],
                   help='extra reference: identity=image.png (e.g. a frame from a shot)')
    e.add_argument('--out', required=True)
    e.add_argument('--model', default=MODEL)
    r_ = sp.add_parser('sheet', help='re-draw the sheets from a scan json')
    r_.add_argument('--cast', required=True)
    r_.add_argument('--sheet')
    r_.add_argument('--tracks-sheet')
    s = sp.add_parser('scan')
    s.add_argument('--preset', help='cast preset (presets.py): family names for near-identical extras')
    s.add_argument('--db', required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--video', action='append')
    g.add_argument('--clips', help='folder of NNN.mp4 clips')
    s.add_argument('--out', required=True)
    s.add_argument('--model', default=MODEL)
    s.add_argument('--sheet')
    s.add_argument('--tracks-sheet', help='jpg: one tile per track, for the by-eye ID check')
    s.add_argument('--review', help='dir for labelled review videos')
    s.add_argument('--stride', type=int, default=2)
    s.add_argument('--thr', type=float, default=0.45)
    s.add_argument('--margin', type=float, default=0.03)
    s.add_argument('--w-col', type=float, default=0.3, help='weight of the colour/costume vote')
    s.add_argument('--jobs', type=int, default=8)
    p = sp.add_parser('plan')
    p.add_argument('--preset', help='cast preset (presets.py), e.g. landing')
    p.add_argument('--cast', required=True)
    p.add_argument('--cut', required=True)
    p.add_argument('--align', required=True)
    p.add_argument('--script')
    p.add_argument('--song-offset', type=float, default=12.0)
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--out', required=True)
    p.add_argument('--md')
    h = sp.add_parser('hook')
    h.add_argument('--plan', required=True)
    h.add_argument('--audio', required=True)
    h.add_argument('--work', required=True)
    h.add_argument('--tier', choices=['safe', 'near'], default='safe', help='near: also the spans a zoom/re-frame away from dub-safe')
    h.add_argument('--out', help='write the commands as a shell script (not run)')
    x = sp.add_parser('export-onnx')
    x.add_argument('--out', default=MODEL)
    x.add_argument('--hf', default='facebook/dinov2-base')
    a = ap.parse_args(argv)

    if a.cmd == 'export-onnx':
        return export_onnx(a.hf, a.out)
    if a.cmd == 'sheet':
        res = json.load(open(a.cast))['results']
        if a.sheet: contact_sheet(res, a.sheet)
        if a.tracks_sheet: track_sheet(res, a.tracks_sheet)
        return 0
    if a.cmd == 'enroll':
        imgs = [(identity_of(c), f) for c in a.char for f in ref_images(a.root, c)]
        imgs += [tuple(x.split('=', 1)) for x in a.add]
        db = enroll(imgs, Faces(), Embedder(a.model))
        np.savez_compressed(a.out, **db)
        for n in sorted(set(db['name'].tolist())):
            print(f"  {n}: {int((db['name'] == n).sum())} reference faces")
        return 0
    if a.cmd == 'scan':
        vids = a.video or sorted(glob.glob(os.path.join(a.clips, '[0-9]*.mp4')))
        kw = {'stride': a.stride, 'thr': a.thr, 'margin': a.margin, 'w_col': a.w_col, 'preset': a.preset}
        jobs = [(v, a.db, a.model, kw) for v in vids]
        if a.jobs > 1 and len(jobs) > 1:
            import multiprocessing as mp
            with mp.get_context('spawn').Pool(a.jobs) as pool:
                res = pool.map(_scan_one, jobs)
        else:
            res = [_scan_one(j) for j in jobs]
        json.dump({'db': a.db, 'rule': {'face_px_1080': SAFE_PX, 'yaw': SAFE_YAW}, 'results': res}, open(a.out, 'w'), indent=1)
        for r in res:
            print(f"{os.path.basename(r['video'])}: " + (', '.join(f"{t['character']}({t['confidence']:.2f}, f{t['f0']}-{t['f1']}, "
                  f"{t['face_px_med']:.0f}px, safe {t['dub_safe_frac']:.0%})" for t in r['tracks']) or '-'))
        if a.sheet: contact_sheet(res, a.sheet)
        if a.tracks_sheet: track_sheet(res, a.tracks_sheet)
        if a.review:
            os.makedirs(a.review, exist_ok=True)
            for r in res: review_video(r, os.path.join(a.review, os.path.basename(r['video'])))
        return 0
    if a.cmd == 'plan':
        cast, cut = json.load(open(a.cast)), json.load(open(a.cut))
        words = json.load(open(a.align))['words']
        singers = script_singers(open(a.script).read(), a.preset) if a.script else {}
        PL = plan(cast, cut, words, singers, a.song_offset, a.fps, preset=a.preset)
        S = summarise(PL)
        json.dump({'cast': a.cast, 'align': a.align, 'song_offset': a.song_offset, 'singers': {k: sorted(v) for k, v in singers.items()},
                   'summary': S, 'plan': PL}, open(a.out, 'w'), indent=1)
        if a.md: open(a.md, 'w').write(plan_md(PL, S))
        print(json.dumps(S, indent=1))
        return 0
    if a.cmd == 'hook':
        D = json.load(open(a.plan))
        C = hook(D['plan'], None, a.audio, a.work, D['align'], song_offset=D.get('song_offset', 12.0), tier=a.tier)
        txt = '#!/usr/bin/env bash\n# DRY RUN from lsdub cast hook: review, then run by hand. Nothing was executed.\nset -e\n' + \
              '\n\n'.join('\n'.join(c) for c in C) + '\n'
        if a.out: open(a.out, 'w').write(txt)
        print(txt)
        return 0


def export_onnx(hf, out):
    """One-off: DINOv2 -> ONNX ([CLS, mean patch] = 1536-d). Needs torch + transformers + onnx (any venv; CPU is fine)."""
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained(hf).eval()

    class W(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m
        def forward(s, x):
            h = s.m(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.onnx.export(W(m), torch.randn(1, 3, 224, 224), out, input_names=['x'], output_names=['emb'],
                      dynamic_axes={'x': {0: 'n'}, 'emb': {0: 'n'}}, opset_version=17, dynamo=False)
    print('->', out)
    return 0


if __name__ == '__main__':
    main()
