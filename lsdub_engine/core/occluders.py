#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""occluders.py: `lsdub occluders`, masks of what sits IN FRONT of the mouth (a mic, a hand, hair). GPU.

The painter must never paint over an object that crosses the face: its pixels come from the take, untouched. This
finds the objects from plain words and follows them through the shot:
  find     Florence-2 (MIT) phrase grounding on a few seed frames: one box per phrase (the tightest it returns)
  follow   SAM 2.1 (Apache-2.0) video propagation from that box, forward and backward over the whole shot
  out      an npz: masks (n, H, W) uint8 0/255, the union of all phrases, plus per-phrase coverage

  lsdub occluders --video plate.mp4 --out occ.npz [--phrase microphone --phrase hand] [--review occ.mp4]

Then `lsdub paint ... --occluder-mask occ.npz`: the painted frame takes the take's own pixels under the mask.
In ComfyUI the LSDub Occluders node runs this on the dubbed shots' face crops.

SAM 2 code: the sam2 package (Meta, Apache-2.0; the optional `occluders` extra). Weights: --model-dir (the node uses
models/lsdub/occluders).
"""
import argparse, os
from ..io import ff
import cv2, numpy as np

FLORENCE = 'florence-community/Florence-2-large'
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def _read(video):
    cap = cv2.VideoCapture(video)
    F = []
    while True:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F, cap.get(cv2.CAP_PROP_FPS) or 24.0


def ground(F, phrases, dev='cuda', seeds=3, model_dir=None):
    """-> {phrase index: (frame, box xyxy)}: the first seed frame where Florence-2 finds the phrase."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(FLORENCE, cache_dir=model_dir)
    fl = Florence2ForConditionalGeneration.from_pretrained(FLORENCE, torch_dtype=torch.float16, cache_dir=model_dir).to(dev)
    H, W = F[0].shape[:2]
    task = '<CAPTION_TO_PHRASE_GROUNDING>'
    out = {}
    for oi, ph in enumerate(phrases):
        for k in np.linspace(0, len(F) - 1, seeds + 1)[:-1].astype(int):
            inp = proc(text=task + ph, images=Image.fromarray(F[k][..., ::-1]), return_tensors='pt').to(dev, torch.float16)
            ids = fl.generate(**inp, max_new_tokens=256, num_beams=3)
            bb = proc.post_process_generation(proc.batch_decode(ids, skip_special_tokens=False)[0], task=task, image_size=(W, H))[task]['bboxes']
            if bb:
                out[oi] = (int(k), min(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
                break
    del fl, proc
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


SAM2_ID, SAM2_CFG, SAM2_FILE = 'facebook/sam2.1-hiera-base-plus', 'configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'


def sam2_checkpoint(ckpt=None, model_dir=None):
    """the SAM 2.1 base+ checkpoint: a given file, else downloaded once (Hugging Face) into model_dir."""
    if ckpt and os.path.exists(ckpt): return ckpt
    from huggingface_hub import hf_hub_download
    return hf_hub_download(SAM2_ID, SAM2_FILE, cache_dir=model_dir)


def follow(F, seeds, dev='cuda', ckpt=None, model_dir=None):
    """-> (n_phrases, n, H, W) bool. The sam2 package (Meta, Apache-2.0) video predictor, seeded with the boxes."""
    import tempfile
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    H, W = F[0].shape[:2]
    M = np.zeros((max(seeds) + 1 if seeds else 0, len(F), H, W), bool)
    if not seeds: return M
    model = build_sam2_video_predictor(SAM2_CFG, sam2_checkpoint(ckpt, model_dir), device=dev)
    try:
        with tempfile.TemporaryDirectory() as d:
            for i, f in enumerate(F): cv2.imwrite(os.path.join(d, f'{i:05d}.jpg'), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with torch.inference_mode(), torch.autocast(str(dev).split(':')[0], dtype=torch.bfloat16, enabled=str(dev) != 'cpu'):
                st = model.init_state(video_path=d)
                for oi, (k, bb) in seeds.items():
                    model.add_new_points_or_box(st, frame_idx=k, obj_id=oi, box=np.array(bb, np.float32))
                for rev in (False, True):
                    for fi, ids, lg in model.propagate_in_video(st, reverse=rev):
                        for j, oid in enumerate(ids): M[oid, fi] = (lg[j, 0] > 0).cpu().numpy()
                del st
    finally:
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return M


def load(path, n, shape):
    """the occluder masks for a render: (n, H, W) float 0..1, resized to the frame; missing frames = none."""
    m = np.load(path)['masks']
    out = np.zeros((n,) + tuple(shape), np.float32)
    for i in range(min(n, len(m))):
        out[i] = cv2.resize(m[i], (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return out


def restore(outs, takes, occ, grow=3, feather=1.5):
    """outs[i] = the take's pixels under the (grown, feathered) occluder mask. In place; -> covered px per frame."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)) if grow > 0 else None
    cov = []
    for i in range(min(len(outs), len(occ))):
        m = occ[i]
        if m.max() <= 0:
            cov.append(0)
            continue
        if k is not None: m = cv2.dilate(m, k)
        if feather > 0: m = cv2.GaussianBlur(m, (0, 0), feather)
        m = np.clip(m, 0, 1)[..., None]
        outs[i] = (outs[i].astype(np.float32) * (1 - m) + takes[i].astype(np.float32) * m + 0.5).astype(np.uint8)
        cov.append(int((m > 0.5).sum()))
    return cov


def review(F, M, phrases, fps, out):
    H, W = F[0].shape[:2]
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                          '-c:v', 'libx264', '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', out],
                         stdin=ff.PIPE)
    for i, f in enumerate(F):
        g = f.copy()
        for oi in range(len(M)):
            m = M[oi, i]
            c = cols[oi % 4]
            g[m] = (0.5 * g[m] + 0.5 * np.array(c)).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(g, cs, -1, c, 2)
            cv2.putText(g, phrases[oi], (10, 30 + 28 * oi), 0, 0.8, c, 2)
        cv2.putText(g, f'f{i}', (W - 80, 30), 0, 0.8, (255, 255, 255), 2)
        p.stdin.write(g.tobytes())
    p.stdin.close()
    p.wait()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True, help='masks npz')
    ap.add_argument('--phrase', action='append', help='what is in front of the face (repeatable; default "microphone")')
    ap.add_argument('--review', help='also write an overlay video of the masks')
    ap.add_argument('--sam2-ckpt', help='SAM 2.1 base+ checkpoint (.pt); default: downloaded into --model-dir')
    ap.add_argument('--model-dir', help='where Florence-2 / SAM 2 weights are cached (default: the Hugging Face cache)')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    phrases = a.phrase or ['microphone']
    F, fps = _read(a.video)
    seeds = ground(F, phrases, a.device, model_dir=a.model_dir)
    M = follow(F, seeds, a.device, a.sam2_ckpt, a.model_dir)
    if len(M) < len(phrases): M = np.concatenate([M, np.zeros((len(phrases) - len(M),) + M.shape[1:], bool)])
    cov = M.reshape(len(phrases), len(F), -1).mean(-1)
    np.savez_compressed(a.out, masks=(M.any(0) * 255).astype(np.uint8), phrases=np.array(phrases), coverage=cov.astype(np.float32))
    for oi, ph in enumerate(phrases):
        s = seeds.get(oi)
        print(f'occluders "{ph}": ' + (f'seed frame {s[0]} box {[round(v) for v in s[1]]}, on {(cov[oi] > 0).sum()}/{len(F)} frames, '
                                       f'mean area {cov[oi].mean() * 100:.2f}%' if s else 'NOT FOUND'))
    if a.review: review(F, M, phrases, fps, a.review)
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
