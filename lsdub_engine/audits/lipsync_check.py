#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lipsync_check.py — automatic lip-sync checker: measured mouth opening in a video vs Audio2Face-3D jawOpen.

Per video frame: MediaPipe FaceLandmarker (Apache-2.0 code + model) -> inner-lip gap (lm 13-14) / face height (lm 10-152).
Compared against A2F curves (60 fps, song time) resampled at the video frame times, with a lag search. Vocal activity
from the vocal stem's RMS flags "mouth open during silence" and "mouth closed while singing".

  lsdub check --video take.mp4 --t0 8.04 \
      [--curves a2f_blendshapes.csv ...] [--audio vocal.wav] [--json out.json] [--plot out.png]
--t0 = song (stem) time of video frame 0. For a v8 cut segment: cut.start - 12.0 (OPEN).
--robust: retry missed frames (tracked box + upscale + low confidence + CLAHE); same metric.
--visemes vis.json: viseme agreement (lsdub visemes) — does the mouth SHAPE match the expected viseme + per-word M/B/P closures and OO rounding.
--crop x,y,w,h: only look in this pixel box (helps when several faces are in frame; default = largest face).
Positive lag = the video mouth moves LATER than the A2F curve.
"""
import argparse, csv, json, os
import numpy as np
from ..io import ff

from .._models import face_landmarker_path
MODEL = face_landmarker_path()
STEM = None


def refill(video, mouth, fh, boxes_pf):
    """--robust: re-detect missed frames (IMAGE mode, low confidence) on a tight upscaled crop around the face box
    interpolated from neighbouring hits, with a CLAHE-enhanced retry. Same mouth metric, so scores stay comparable."""
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.1, min_face_presence_confidence=0.1))
    hit = np.array([b is not None for b in boxes_pf])
    n = len(mouth)
    if hit.sum() == 0: return mouth, fh
    B = np.array([b if b is not None else [np.nan] * 4 for b in boxes_pf], float)
    ii = np.arange(n)
    for k in range(4): B[:, k] = np.interp(ii, ii[hit], B[hit, k])
    cap = cv2.VideoCapture(video)
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if not np.isfinite(mouth[i]):
            b = B[i]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            s = max(b[2] - b[0], b[3] - b[1]) * 1.0
            for pad in (1.0, 1.6):
                x0, y0 = int(max(0, cx - s * pad)), int(max(0, cy - s * pad))
                x1, y1 = int(min(W, cx + s * pad)), int(min(H, cy + s * pad))
                c = fr[y0:y1, x0:x1]
                sc = 768 / max(1, min(c.shape[:2]))
                c = cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                got = None
                for var in (c, cv2.cvtColor(clahe.apply(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)):
                    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(var[:, :, ::-1])))
                    if r.face_landmarks:
                        hh, ww = var.shape[:2]
                        p = np.array([[q.x * ww, q.y * hh] for q in r.face_landmarks[0]])
                        f_h = np.linalg.norm(p[10] - p[152])
                        got = (np.linalg.norm(p[13] - p[14]) / f_h, f_h / sc)
                        break
                if got:
                    mouth[i], fh[i] = got
                    break
        i += 1
    lm.close()
    cap.release()
    return mouth, fh


def mouth_series(video, crop=None, scale=1.0, boxes=None, boxes_pf=None, widths=None):
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO, num_faces=4,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, min_tracking_confidence=0.3))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        x0 = y0 = 0
        if crop:
            x0, y0, w, h = crop
            fr = fr[y0:y0 + h, x0:x0 + w]
        if scale != 1.0: fr = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        H, W = fr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr[:, :, ::-1]))
        r = lm.detect_for_video(img, int(i * 1000 / fps))
        best = None
        for f in r.face_landmarks:
            p = np.array([[q.x * W, q.y * H] for q in f]) / scale + [x0, y0]
            fh = np.linalg.norm(p[10] - p[152])
            if best is None or fh > best[1]: best = (p, fh)
        if boxes_pf is not None: boxes_pf.append(None if best is None else [*best[0].min(0), *best[0].max(0)])
        if widths is not None:   # mouth-corner distance / face width (61-291 vs 234-454)
            widths.append(np.nan if best is None else np.linalg.norm(best[0][61] - best[0][291]) / np.linalg.norm(best[0][234] - best[0][454]))
        if best is None: out.append((np.nan, np.nan))
        else:
            p, fh = best
            if boxes is not None: boxes.append([*p.min(0), *p.max(0)])
            out.append((np.linalg.norm(p[13] - p[14]) / fh, fh))
        i += 1
    lm.close()
    cap.release()
    a = np.array(out, float)
    return a[:, 0], a[:, 1], fps


def load_curves(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        rows = np.array([[float(x) for x in row] for row in r])
    return hdr, rows


def a2f_signal(hdr, rows):
    c = {n: rows[:, hdr.index(n)] for n in ('jawOpen', 'mouthClose')}
    t = rows[:, hdr.index('time_s')]
    return t, c['jawOpen'], np.clip(c['jawOpen'] - c['mouthClose'], 0, None)


def vocal_rms(audio, t0, n, fps):
    raw = ff.run(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-t', f'{n / fps + 0.1:.3f}', '-i', audio,
                          '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    hop = 16000 / fps
    win = int(16000 * 0.04)
    rms = []
    for i in range(n):
        c = int(i * hop + hop / 2)
        s = x[max(0, c - win // 2): c + win // 2]
        rms.append(np.sqrt(np.mean(s ** 2)) if len(s) else 0.0)
    return 20 * np.log10(np.array(rms) + 1e-9)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9: return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])


ROUND = {'ou', 'oh', 'CH', 'RR'}
SPREAD = {'ih', 'E', 'SS'}
OPENV = {'aa', 'oh', 'E'}
CLOSED = {'PP'}


def viseme_agreement(mouth, width, vis_path, t0, fps):
    """Does the measured mouth SHAPE agree with the expected viseme? Per frame: PP -> lips shut (gap < 0.25 x the take's
    voiced-median gap); open vowels (aa/oh/E) -> gap above median; rounded (ou/oh/CH/RR) -> width below median;
    spread (ih/E/SS) -> width above median. Per word: every M/B/P must reach a closure within +-1 frame of its phone,
    every OO/OW/W must narrow the mouth (width < median) within the phone."""
    V = json.load(open(vis_path))
    fr = V['frames']
    vt = np.array([f['t'] for f in fr])
    n = len(mouth)
    tv = t0 + np.arange(n) / fps
    idx = np.clip(np.round((tv - vt[0]) * V['fps']).astype(int), 0, len(fr) - 1)
    vis = [fr[i]['vis'] for i in idx]
    ok = np.isfinite(mouth) & np.isfinite(width)
    sing = np.array([v != 'sil' for v in vis]) & ok
    if sing.sum() < 5: return {'error': 'too few frames'}
    gm, wm = np.nanmedian(mouth[sing]), np.nanmedian(width[sing])
    tests = {'closed': [], 'open': [], 'round': [], 'spread': []}
    for i, v in enumerate(vis):
        if not ok[i]: continue
        if v in CLOSED: tests['closed'].append(mouth[i] < 0.25 * gm)
        if v in OPENV: tests['open'].append(mouth[i] > gm)
        if v in ROUND: tests['round'].append(width[i] < wm)
        if v in SPREAD: tests['spread'].append(width[i] > wm)
    per = {k: (round(float(np.mean(x)), 3) if x else None) for k, x in tests.items()}
    allt = sum(tests.values(), [])
    score = float(np.mean(allt)) if allt else float('nan')
    # per-word checks from the phone timings embedded in the track
    words = []
    al = json.load(open(V['align'])) if V.get('align') and os.path.exists(V['align']) else None
    if al:
        for w in al['words']:
            for ph in w['phones']:
                kind = 'closure' if ph['p'] in ('M', 'B', 'P') else 'round' if ph['p'] in ('UW', 'OW', 'W', 'UH') else None
                if not kind: continue
                a, b = int(np.floor((ph['start'] - t0) * fps)) - 1, int(np.ceil((ph['end'] - t0) * fps)) + 1
                if b < 0 or a >= n: continue
                a, b = max(0, a), min(n, b)
                seg_m, seg_w = mouth[a:b], width[a:b]
                if not np.isfinite(seg_m).any(): continue
                hit = bool(np.nanmin(seg_m) < 0.25 * gm) if kind == 'closure' else bool(np.nanmin(seg_w) < wm)
                words.append({'word': w['w'], 'phone': ph['p'], 't': ph['start'], 'kind': kind, 'hit': hit,
                              'min_gap': round(float(np.nanmin(seg_m)), 4), 'min_width': round(float(np.nanmin(seg_w)), 4)})
    return {'score': round(score, 3), 'per_class': per, 'gap_median': round(float(gm), 4), 'width_median': round(float(wm), 4),
            'words': words, 'word_hits': f"{sum(w['hit'] for w in words)}/{len(words)}"}


def check(video, t0, curves_list, audio, crop=None, robust=False, max_lag=0.3, sil_db=-40.0, open_thr=0.06, closed_thr=0.02, visemes=None):
    if crop is None:   # pass 1 finds the face, pass 2 re-detects on an upscaled padded crop (sunglasses/mic hurt)
        import cv2
        boxes = []
        mouth_series(video, None, 1.0, boxes)
        if boxes:
            b = np.median(np.array(boxes), 0)
            cw, ch = b[2] - b[0], b[3] - b[1]
            cap = cv2.VideoCapture(video)
            W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            x0, y0 = int(max(0, b[0] - cw)), int(max(0, b[1] - ch))
            x1, y1 = int(min(W, b[2] + cw)), int(min(H, b[3] + ch))
            crop = (x0, y0, x1 - x0, y1 - y0)
    sc = max(1.0, 640 / min(crop[2], crop[3])) if crop else 1.0
    bpf = []
    wid = []
    mouth, fh, fps = mouth_series(video, crop, sc, boxes_pf=bpf, widths=wid)
    face_frac_pass2 = float(np.isfinite(mouth).mean())
    if robust: mouth, fh = refill(video, mouth, fh, bpf)
    n = len(mouth)
    tv = t0 + np.arange(n) / fps
    db = vocal_rms(audio, t0, n, fps)
    voiced = db > max(sil_db, db.max() - 30)
    rep = {'video': video, 't0_song': t0, 'crop': crop, 'frames': n, 'fps': fps,
           'face_found_frac': float(np.isfinite(mouth).mean()), 'face_found_frac_strict': face_frac_pass2, 'robust': robust, 'face_height_px_median': float(np.nanmedian(fh)) if np.isfinite(fh).any() else None,
           'mouth_open_mean': float(np.nanmean(mouth)) if np.isfinite(mouth).any() else None,
           'voiced_frames': int(voiced.sum()), 'curves': {}}
    # silence/singing flags are curve-independent
    rep['open_during_silence'] = [int(i) for i in np.where(~voiced & (mouth > open_thr))[0]]
    rep['closed_while_singing'] = [int(i) for i in np.where(voiced & (mouth < closed_thr))[0]]
    rep['per_frame'] = {'mouth': [None if not np.isfinite(v) else round(float(v), 4) for v in mouth],
                        'vocal_db': [round(float(v), 1) for v in db]}
    if visemes: rep['visemes'] = viseme_agreement(mouth, np.array(wid, float), visemes, t0, fps)
    for cp in curves_list:
        hdr, rows = load_curves(cp)
        ta, jaw, jo = a2f_signal(hdr, rows)
        best = None
        lags = np.arange(-max_lag, max_lag + 1e-9, 0.01)
        res = []
        for L in lags:
            j = np.interp(tv - L, ta, jaw)
            c1 = corr(mouth, j)
            c2 = corr(mouth, np.interp(tv - L, ta, jo))
            res.append((L, c1, c2))
            if best is None or (np.isfinite(c1) and c1 > best[1]): best = (L, c1, c2)
        j0 = np.interp(tv, ta, jaw)
        e = {'r_jawOpen_lag0': corr(mouth, j0), 'r_jawOpen_minus_close_lag0': corr(mouth, np.interp(tv, ta, jo)),
             'best_lag_s': round(float(best[0]), 3), 'r_jawOpen_best': best[1], 'r_jawOpen_minus_close_best': best[2],
             # A2F says the mouth should be shut/open but the video disagrees
             'a2f_closed_video_open': [int(i) for i in np.where((j0 < 0.05) & (mouth > open_thr))[0]],
             'a2f_open_video_closed': [int(i) for i in np.where((j0 > 0.25) & (mouth < closed_thr))[0]],
             'a2f_jawOpen': [round(float(v), 4) for v in j0]}
        rep['curves'][os.path.basename(cp)] = e
    return rep


def plot(rep, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = np.array([np.nan if v is None else v for v in rep['per_frame']['mouth']])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = np.arange(len(m)) / rep['fps'] + rep['t0_song']
    ax.plot(t, m / (np.nanmax(m) or 1), 'k', lw=2, label='video mouth (norm)')
    for k, e in rep['curves'].items():
        ax.plot(t, e['a2f_jawOpen'], lw=1, label=f"{k} jawOpen r={e['r_jawOpen_lag0']:.2f}")
    db = np.array(rep['per_frame']['vocal_db'])
    ax.fill_between(t, 0, (db - db.min()) / (np.ptp(db) or 1) * 0.3, color='0.85', label='vocal level')
    ax.set_xlabel('song time (s)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title(os.path.basename(rep['video']))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--t0', type=float, required=True)
    ap.add_argument('--curves', nargs='+', default=[], help='optional ARKit blendshape CSVs (e.g. Audio2Face) to correlate against')
    ap.add_argument('--audio', required=True, help='the (dry) vocal, song time')
    ap.add_argument('--crop')
    ap.add_argument('--robust', action='store_true', help='re-detect missed frames (tracked box, upscale, low conf, CLAHE)')
    ap.add_argument('--visemes', help='lsdub visemes JSON: adds the viseme-agreement metric (mouth shape vs expected viseme)')
    ap.add_argument('--json')
    ap.add_argument('--plot')
    a = ap.parse_args(argv)
    rep = check(a.video, a.t0, a.curves, a.audio, tuple(int(v) for v in a.crop.split(',')) if a.crop else None, a.robust, visemes=a.visemes)
    if a.json: json.dump(rep, open(a.json, 'w'), indent=1)
    if a.plot: plot(rep, a.plot)
    print(f"{rep['video']}: frames={rep['frames']} face={rep['face_found_frac']:.2f} faceH={rep['face_height_px_median']} voiced={rep['voiced_frames']}")
    for k, e in rep['curves'].items():
        print(f"  {k}: r(jawOpen)={e['r_jawOpen_lag0']:.3f}  r(jaw-close)={e['r_jawOpen_minus_close_lag0']:.3f}  "
              f"best lag {e['best_lag_s']:+.2f}s r(jawOpen)={e['r_jawOpen_best']:.3f}  "
              f"a2f-shut/video-open={e['a2f_closed_video_open']} a2f-open/video-shut={e['a2f_open_video_closed']}")
    if 'visemes' in rep and 'score' in rep['visemes']:
        v = rep['visemes']
        print(f"  viseme agreement {v['score']:.3f} {v['per_class']}  words {v['word_hits']}: "
                                  + ' '.join(f"{w['word']}/{w['phone']}{'+' if w['hit'] else '-'}" for w in v['words']))
    print(f"  open during silence: {rep['open_during_silence']}\n  closed while singing: {rep['closed_while_singing']}")


if __name__ == '__main__':
    main()
