#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_paint_cutout.py — paint viseme mouths onto a non-human cutout face (rocks / objects / muppets).

No MediaPipe face tracking. The mouth lives at a FIXED region in the frame (a single x,y,w,h, or a per-frame
JSON list of such regions if the head bobs). Each frame: pick the viseme from the track, cross-fade with the
runner-up, scale the jaw drop to match the track's openness, and paste the library mouth into the fixed region
with a feathered mask.

  lsdub paint --video body.mp4 --track vis_001.json --lib mouthlib \
      --mode cutout --mouth "x,y,w,h" --out painted.mp4 [--frames 200] [--temporal 0.25]

--mouth is one of:
  x,y,w,h           (pixels in the output frame)
  mouth_region.json  (per-frame list of [x,y,w,h]; frames not in the list reuse the last known region)
"""
import argparse, json, os
from ..io import ff
import numpy as np, cv2

from ..core import viseme_track as vt
from . import stylized as st
OPEN = vt.OPEN

# LIBMAP is defined in mouth_paint; lazy-import to avoid circular import
def _libmap():
    from ..core import mouth_paint
    return mouth_paint.LIBMAP


def mouth_bbox(mask, pad_frac=0.08):
    """Bounding box of a mask image (HxW), with a small pad. Returns (x0, y0, w, h)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, y0 = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    H, W = mask.shape[:2]
    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(W, int(xs.max()) + pad_x + 1)
    y1 = min(H, int(ys.max()) + pad_y + 1)
    return x0, y0, x1 - x0, y1 - y0


def load_mouth_region(lib, v):
    """Load library viseme v, return (mouth_crop HxWx3, mask HxW) cropped to the mouth bbox."""
    f = f'{lib}/{v}.png'
    if not os.path.exists(f):
        f = f'{lib}/sil.png'
    img = cv2.imread(f)
    mask = cv2.imread(f'{lib}/mask.png', 0).astype(np.float32) / 255.0
    x0, y0, w, h = mouth_bbox(mask, pad_frac=0.05)
    crop = img[y0:y0 + h, x0:x0 + w]
    m = mask[y0:y0 + h, x0:x0 + w]
    return crop, m


def jaw_openness_scale(crop, mask, extra_open):
    """Scale the crop vertically to simulate extra jaw drop. extra_open = -0.5..+1.0
    (negative = close jaw, positive = open more). The upper portion stays fixed; the lower jaw drops."""
    if abs(extra_open) < 0.02:
        return crop
    H, W = crop.shape[:2]
    col_sums = mask.sum(axis=1)
    center_y = int(np.argmax(col_sums))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dy = np.clip((yy - center_y) * extra_open, -H + 1, H - 1)
    yy_src = np.clip(yy + dy, 0, H - 1).astype(np.float32)
    return cv2.remap(crop, xx, yy_src, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def puppet_jaw(frame, hinge, max_deg, open_amt, interior=(28, 18, 40), jaw_w_frac=0.60):
    """Drop the lower jaw straight down to open a felt-puppet mouth (classic Muppet hinge, read in 2D as
    the lower jaw sliding down). The gap between the fixed upper lip and the dropped lower jaw is filled
    with a dark mouth interior + tongue. frame is HxWx3 BGR; hinge = (hx, hy) = the mouth/hinge line.
    max_deg is the max open angle; the drop in px = open_amt * sin(max_deg) * (H - hy)."""
    H, W = frame.shape[:2]
    hx, hy = int(hinge[0]), int(hinge[1])
    drop = int(max(0.0, open_amt) * np.sin(np.deg2rad(max_deg)) * (H - hy))
    if drop < 2:
        return frame
    out = frame.copy()
    jw = int(W * jaw_w_frac)
    x0, x1 = max(0, hx - jw // 2), min(W, hx + jw // 2)
    # dark mouth cavity revealed between the upper lip (hy) and where the jaw now sits (hy+drop)
    cv2.rectangle(out, (x0, hy), (x1, min(H, hy + drop)), interior, -1)
    cv2.ellipse(out, (hx, hy + drop), (jw // 2, max(4, drop // 2)), 0, 180, 360, (70, 70, 150), -1)  # tongue
    # move the lower jaw strip straight down by `drop`, only within the jaw width
    lower = frame[hy:H, x0:x1].copy()
    lh = lower.shape[0]
    paste_y0 = hy + drop
    paste_y1 = min(H, paste_y0 + lh)
    src_h = paste_y1 - paste_y0
    if src_h > 0:
        out[paste_y0:paste_y1, x0:x1] = lower[:src_h]
    return out


def run(a):
    T = json.load(open(a.track))
    fr_t = T['frames']
    n = a.frames or len(fr_t)
    mode = getattr(a, 'mode', 'cutout')
    if mode == 'puppet':
        return run_puppet(a, fr_t, n)

    fps0 = float(T.get('fps', 25.0))
    # Stylized (classical-animation) snap path: reduce to anchors, SNAP (no morph), on-twos, lead,
    # closure-hold, accent, singing holds + volume-conserving squash. Default for cutout.
    preset = getattr(a, 'preset', 'classic') or 'classic'
    snap = getattr(a, 'transition', 'snap') != 'fade'
    styl = None
    if snap:
        styl = st.build_stylized_track(fr_t[:n], fps=fps0, preset=preset,
                                       singing=getattr(a, 'singing', False),
                                       cartoon=getattr(a, 'cartoon', 1.0))

    # Load mouth region(s)
    if a.mouth.endswith('.json'):
        regions_raw = json.load(open(a.mouth))
        if isinstance(regions_raw[0], (int, float)):
            regions_raw = [regions_raw] * max(n, 1)
        regions = []
        last = regions_raw[0]
        for i in range(n):
            regions.append(regions_raw[i] if i < len(regions_raw) else last)
    else:
        x, y, w, h = (int(float(v)) for v in a.mouth.split(','))
        regions = [(x, y, w, h)] * n

    lm = _libmap()
    # Pre-load all needed library mouths
    if styl is not None:
        needed_vis = set(s['rep'] for s in styl)
    else:
        needed_vis = set(lm.get(f['vis'], f['vis']) for f in fr_t) | \
                     set(lm.get(f['vis2'], f['vis2']) for f in fr_t)
    lib_mouths = {}
    for v in needed_vis:
        lib_mouths[v] = load_mouth_region(a.lib, v)
    lib_mouths['sil'] = lib_mouths.get('sil', lib_mouths.get(next(iter(needed_vis)), (None, None)))

    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Temporal blend uses per-frame painted mouth (after openness scaling),
    # so we compute all frames before writing.
    frames_out = []
    painted = []   # list of (resized_mouth_bgr, feather_mask)
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        f = fr_t[i]
        lm = _libmap()
        sq_w = sq_h = 1.0
        if styl is not None:
            s = styl[i]
            v1 = v2 = s['rep']
            mix = 0.0                       # SNAP: no cross-fade between shapes
            open_amt = float(s['open'])
            sq_w, sq_h = float(s['sq_w']), float(s['sq_h'])
        else:
            v1 = lm.get(f['vis'], f['vis'])
            v2 = lm.get(f['vis2'], f['vis2'])
            mix = f['mix'] if v2 != v1 else 0.0
            open_amt = float(f['open'])

        c1, m1 = lib_mouths.get(v1, (None, None))
        c2, m2 = lib_mouths.get(v2, (None, None))
        if c1 is None or c1.size == 0:
            c1, m1 = (None, None)
            frames_out.append(fr)
            painted.append((None, None))
            continue

        if c2 is not None and c2.size > 0 and mix > 0.02:
            src = cv2.addWeighted(c1, 1 - mix, c2, mix, 0)
            mask = (m1 * (1 - mix) + m2 * mix)
        else:
            src = c1
            mask = m1

        # openness: scale the mouth vertically
        nom = OPEN.get(v1, 0.0)
        if nom > 0.02:
            scale = float(np.clip(open_amt / nom, 0.2, 2.0))
            extra = scale - 1.0
            src = jaw_openness_scale(src, mask, extra)

        # resize to target (re-resize after openness scaling each frame)
        x, y, w, h = regions[i]
        scale_w = w / src.shape[1]
        scale_h = h / src.shape[0]
        s = min(scale_w, scale_h)
        # volume-conserving squash/stretch (stylized): widen/narrow and shorten/lengthen the mouth
        nw = max(1, int(src.shape[1] * s * sq_w))
        nh = max(1, int(src.shape[0] * s * sq_h))
        src_r = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        m_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # feather
        m_feather = cv2.GaussianBlur(m_r.astype(np.float32) * 255.0,
                                     (max(3, int(nw * 0.03) | 1), max(3, int(nw * 0.03) | 1)), 0)
        m_feather = m_feather.astype(np.float32) / 255.0

        frames_out.append(fr)
        painted.append((src_r, m_feather))

    cap.release()

    # Temporal blend: blend painted mouth with neighbours that share the same viseme
    N = len(frames_out)
    for i in range(N):
        reg, m = painted[i]
        if styl is None and a.temporal > 0 and 0 < i < N - 1 and reg is not None:
            same = [j for j in (i - 1, i + 1) if painted[j][0] is not None
                    and fr_t[j]['vis'] == fr_t[i]['vis']]
            if same:
                avg = np.stack([painted[j][0].astype(np.float32) for j in same]).mean(axis=0)
                reg = reg.astype(np.float32) * (1 - a.temporal) + avg * a.temporal

    # Write to ffmpeg
    out_p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                              '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                              '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', a.out],
                             stdin=ff.PIPE)
    for i in range(N):
        fr = frames_out[i]
        reg, m = painted[i]
        if reg is None or m is None or reg.size == 0:
            out = fr
        else:
            x, y, w, h = regions[i]
            nw, nh = reg.shape[1], reg.shape[0]
            off_x = (w - nw) // 2
            off_x = max(0, off_x)
            off_y = (h - nh) // 2
            off_y = max(0, off_y)
            # idle breath wobble on rest so the mouth isn't dead-still on a frozen subject
            if styl is not None and getattr(a, 'idle_wobble', True) and styl[i]['anchor'] == 'rest':
                off_y += int(round(1.2 * np.sin(i * 0.5)))
            fx, fy = x + off_x, y + off_y
            # clamp to frame
            cx0, cy0 = max(0, fx), max(0, fy)
            cx1, cy1 = min(W, fx + nw), min(H, fy + nh)
            sw = cx1 - cx0
            sh = cy1 - cy0
            if sw > 0 and sh > 0:
                out = fr.astype(np.float32).copy()
                src_crop = reg.astype(np.float32)[cy0 - fy:cy0 - fy + sh, cx0 - fx:cx0 - fx + sw]
                m_crop = m[cy0 - fy:cy0 - fy + sh, cx0 - fx:cx0 - fx + sw]
                # contact shadow / ambient occlusion at the seam: the #1 "reads-alive" cue for a mouth
                # pasted on an object. Darken a soft band just inside the mask edge, stronger when open.
                if styl is not None and getattr(a, 'contact_shadow', True) and sh > 2 and sw > 2:
                    ao_amt = 0.25 + 0.35 * float(styl[i]['open'])
                    edge = m_crop - cv2.erode(m_crop, np.ones((3, 3), np.float32),
                                              iterations=max(1, int(min(sw, sh) * 0.10)))
                    edge = cv2.GaussianBlur(np.clip(edge, 0, 1), (0, 0), max(1.0, min(sw, sh) * 0.05))
                    shade = 1.0 - ao_amt * np.clip(edge, 0, 1)
                    src_crop = src_crop * shade[..., None]
                out[cy0:cy1, cx0:cx1] = (src_crop * m_crop[..., None] +
                                          out[cy0:cy1, cx0:cx1] * (1 - m_crop[..., None]))
                out = np.clip(out, 0, 255).astype(np.uint8)
        if a.debug_dir and i % 25 == 0:
            os.makedirs(a.debug_dir, exist_ok=True)
            cv2.imwrite(f'{a.debug_dir}/{i:04d}.png', out)
        out_p.stdin.write(out.tobytes())
    out_p.stdin.close()
    out_p.wait()
    print(f'{N} frames -> {a.out} (cutout mode)')


def run_puppet(a, fr_t, n):
    """Puppet mode: the subject's lower jaw hinges open on the track's openness. No library swap needed
    (felt-mouth puppets have no viseme shapes), but if --lib is given a cutout mouth is also painted."""
    hx, hy = (float(v) for v in a.jaw_hinge.split(','))
    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # felt-mouth hinge: per-anchor gape, one-open-per-syllable, Henson accent, fast ease-out rise.
    opens = np.array(st.puppet_gape_track(fr_t[:n], fps=fps, accent=getattr(a, 'accent', True),
                                          singing=getattr(a, 'singing', False)), np.float32)
    if opens.size < n:
        opens = np.pad(opens, (0, n - opens.size), mode='edge')
    out_p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                              '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                              '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', a.out],
                             stdin=ff.PIPE)
    i = 0
    while i < n:
        ok, fr = cap.read()
        if not ok:
            break
        out = puppet_jaw(fr, (hx, hy), a.jaw_max_deg, float(opens[i]))
        out_p.stdin.write(out.tobytes())
        i += 1
    cap.release()
    out_p.stdin.close()
    out_p.wait()
    print(f'{i} frames -> {a.out} (puppet mode, hinge {hx:.0f},{hy:.0f}, max {a.jaw_max_deg} deg)')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--track', required=True)
    ap.add_argument('--lib', default='', help='mouth library (required for cutout mode; optional for puppet)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--mode', choices=['cutout', 'puppet'], default='cutout')
    ap.add_argument('--mouth', default='', help='cutout: fixed mouth region x,y,w,h OR per-frame JSON list')
    ap.add_argument('--jaw-hinge', dest='jaw_hinge', default='', help='puppet: hinge pivot "hx,hy" in pixels')
    ap.add_argument('--jaw-max-deg', dest='jaw_max_deg', type=float, default=22.0, help='puppet: max jaw-open angle')
    ap.add_argument('--temporal', type=float, default=0.25, help='temporal blend weight (fade mode only; 0 = off)')
    ap.add_argument('--preset', choices=['classic', 'crude', 'southpark'], default='classic',
                    help='stylized shape set: classic(10) | crude(6, Rhubarb) | southpark(4)')
    ap.add_argument('--transition', choices=['snap', 'fade'], default='snap',
                    help='snap = hard cut between shapes (classic cutout, default); fade = cross-dissolve')
    ap.add_argument('--cartoon', type=float, default=1.0, help='squash/stretch strength (0=off, 1=normal, >1=broad)')
    ap.add_argument('--singing', action='store_true', help='hold vowels across notes; clip sung consonants')
    ap.add_argument('--no-accent', dest='accent', action='store_false', help='disable animate-to-vowels accent')
    ap.add_argument('--no-contact-shadow', dest='contact_shadow', action='store_false', help='disable seam AO')
    ap.add_argument('--no-idle-wobble', dest='idle_wobble', action='store_false', help='disable rest breath wobble')
    ap.add_argument('--debug-dir')
    a = ap.parse_args(argv)
    if a.mode == 'cutout' and (not a.lib or not a.mouth):
        ap.error('cutout mode needs --lib and --mouth')
    if a.mode == 'puppet' and not a.jaw_hinge:
        ap.error('puppet mode needs --jaw-hinge "hx,hy"')
    run(a)


if __name__ == '__main__':
    main()
