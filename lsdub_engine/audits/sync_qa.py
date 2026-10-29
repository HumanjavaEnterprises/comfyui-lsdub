#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sync_qa.py: `lsdub sync-qa`, the mouth sync QA video (founder: "this card / QA tool is amazing, bake it in"). CPU.

One video to judge timing by eye AND by numbers, per frame:
  top      the mouth, zoomed: the take (LTX original) | our render, frame number + clip time
  panel    a scrolling window (+-1.4 s, the white line = now): the waveform, the opening the AUDIO asks for (red, the
           --coart-dyn curve), OUR rendered opening (green, warp.json egap), the take's own opening (grey, warp.json gap)
  card     the mouth-shape card (lsdub annotate) in a side column: the word, the viseme, its openness, the next 1 s
  numbers  corr(MEASURED rendered opening, audio target) and the best lag (primary); plan corr (warp egap) secondary, printed and written on the first frame's panel

  lsdub sync-qa --video film.mp4 --cuts cut.json --vocal lead_dry.wav --offset 12 [--align align.json] [--cast cast.json]
                [--json r.json] [--md r.md]      DELIVERED-FILM mode: every sung cut's singer mouth vs the vocal (measured)
  lsdub sync-qa --take take.mp4 --render out.mp4 --warp-json out_warp.json --dyn dyn.csv --t0 6.17 --audio out_av.mp4
                [--track track.json] [--out qa.mp4] [--half] [--roi y0,y1,x0,x1] [--fps 36]
"""
import argparse, csv, json, os, sys, tempfile
from ..io import ff
from .._inproc import call
from ..core import sung_dyn as SD
import cv2, numpy as np

SR = 8000
PX = 400
PW = 1120
PH = 300
CARD_W = 360


def _read(v):
    cap = cv2.VideoCapture(v)
    L = []
    while True:
        ok, f = cap.read()
        if not ok: break
        L.append(f)
    return L, cap.get(cv2.CAP_PROP_FPS) or 36.0


def _norm(x):
    x = np.asarray(x, float)
    return (x - np.nanmin(x)) / (np.ptp(x) + 1e-9)


def tracking(ours, target, fps, max_lag=10):
    """-> (corr at 0 lag, best corr, best lag ms; + = ours LATER than the audio, the mouth trails).
    NOTE (reco14 / round 4): before this fix the returned sign was inverted (+ meant the mouth LEADS); corr unchanged."""
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    c = {L: float(np.mean(z(ours[L:]) * z(target[:len(target) - L]))) if L >= 0 else
         float(np.mean(z(ours[:L]) * z(target[-L:]))) for L in range(-max_lag, max_lag + 1)}
    L = max(c, key=c.get)
    return c[0], c[L], L * 1000.0 / fps     # L>0 pairs ours[i+L] with target[i]: ours trails


DISAGREE = 0.3


def disagree(plan_corr, measured_corr, margin=DISAGREE):
    """True when the warp's PLAN tracks the audio by more than `margin` better than the delivered pixels do (or the
    pixels could not be measured): the plan is not reaching the face, so the plan corr must not be read as sync."""
    return measured_corr is None or not np.isfinite(measured_corr) or plan_corr - measured_corr > margin


def face_roi(roi, shape, k=2.6):
    """The display ROI is a MOUTH crop; the landmarker needs the whole face. -> a square k x the ROI's larger side,
    centred a little above the mouth, clipped to the frame."""
    y0, y1, x0, x1 = roi
    H, W = shape[:2]
    s = k * max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) / 2 - 0.25 * s, (x0 + x1) / 2
    return [int(max(0, cy - s / 2)), int(min(H, cy + s / 2)), int(max(0, cx - s / 2)), int(min(W, cx + s / 2))]


def measured_opening(frames, roi, fps=25.0):
    """The RENDERED mouth opening, per frame: MediaPipe inner-lip gap (13-14) / face height (10-152) on the ROI crop,
    upscaled 2x. NaN where no face; gaps filled by interpolation. This is what the viewer sees -- the plan (egap) is not."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    from .._models import face_landmarker_path
    det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_landmarker_path()), running_mode=vision.RunningMode.VIDEO,
        num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2, min_tracking_confidence=0.2))
    y0, y1, x0, x1 = roi
    op = np.full(len(frames), np.nan)
    for i, f in enumerate(frames):
        c = cv2.resize(f[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=np.ascontiguousarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))), int(i * 1000 / fps))
        if r.face_landmarks:
            L = r.face_landmarks[0]
            p = lambda j: np.array([L[j].x * c.shape[1], L[j].y * c.shape[0]])
            op[i] = np.linalg.norm(p(13) - p(14)) / (np.linalg.norm(p(10) - p(152)) + 1e-9)
    det.close()
    ok = ~np.isnan(op)
    return (np.interp(np.arange(len(op)), np.where(ok)[0], op[ok]) if ok.sum() >= 5 else None), float(ok.mean())


def auto_roi(A, B):
    D = np.abs(np.asarray(B[: len(A)], np.float32) - np.asarray(A[: len(B)], np.float32)).max(0).max(-1)
    ys, xs = np.where(D > 40)
    if len(ys) < 50:
        h, w = A[0].shape[:2]
        return [h // 3, h * 2 // 3, w // 4, w * 3 // 4]
    y0, y1, x0, x1 = np.percentile(ys, 2), np.percentile(ys, 98), np.percentile(xs, 2), np.percentile(xs, 98)
    cy, cx, hh, hw = (y0 + y1) / 2, (x0 + x1) / 2, (y1 - y0) * 0.5, (x1 - x0) * 0.5
    hw = max(hw, hh * 1.4 / 2 * 2 / 2)
    hh = hw / 1.4                 # 1.4:1 box (560 x 400 tiles)
    H, W = A[0].shape[:2]
    return [int(max(0, cy - hh)), int(min(H, cy + hh)), int(max(0, cx - hw)), int(min(W, cx + hw))]


def render(take, rend, warp_json, dyn, t0, audio, out, track=None, roi=None, half=False):
    A, fps = _read(take)
    B, _ = _read(rend)
    n = min(len(A), len(B))
    roi = roi or auto_roi(A[:n], B[:n])
    y0, y1, x0, x1 = roi
    au = np.frombuffer(ff.run(['ffmpeg', '-v', 'error', '-i', audio, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                                      capture_output=True, check=True).stdout, np.float32)
    W = json.load(open(warp_json))['frames'][:n]
    ours = np.array([f['egap'] for f in W], float)
    orig = np.array([f['gap'] for f in W], float)
    R = list(csv.DictReader(open(dyn)))
    dt = np.array([float(r['time_s']) for r in R]) - t0
    dg = np.array([float(r['geo_gap_cm']) for r in R])
    ft = np.arange(n) / fps
    target = np.interp(ft, dt, dg)
    meas, cover = measured_opening(B[:n], face_roi(roi, B[0].shape), fps)                    # PRIMARY: the pixels we deliver
    if meas is None:
        m0 = mb = mlag = float('nan')
        stat = f"MEASURED: no face ({cover:.0%})"
    else:
        m0, mb, mlag = tracking(meas, target, fps)
        stat = f"MEASURED corr {m0:.2f} (best {mb:.2f} at {mlag:+.0f} ms, face {cover:.0%})"
    c0, cb, lag = tracking(ours, target, fps)                      # secondary: the warp's PLANNED opening
    stat += f"  | plan {c0:.2f}"
    if disagree(c0, m0): stat += "  PLAN/PIXELS DISAGREE"
    ew = np.array([f['ewidth'] for f in W], float)
    tw = np.array([f['twidth'] for f in W], float)
    wc = float(np.corrcoef(ew, tw)[0, 1]) if ew.std() > 0 and tw.std() > 0 else float('nan')
    stat += f"  | shape width vs plan {wc:.2f}"                     # < ~0.8: the width is not reaching the face (a cap)
    on, gn, dn = _norm(orig), _norm(ours), _norm(dg)
    mn = _norm(meas) if meas is not None else np.full(n, np.nan)
    tmp = tempfile.mkdtemp(prefix='syncqa_')
    raw = os.path.join(tmp, 'qa.mp4')
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{PW + CARD_W}x{PH + 400}',
                          '-r', str(fps), '-i', '-', '-i', audio, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-crf', '14',
                          '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', raw], stdin=ff.PIPE)
    win = PW / PX / 2
    xs = np.arange(PW)
    for i in range(n):
        t = i / fps
        a = cv2.resize(A[i][y0:y1, x0:x1], (560, 400))
        b = cv2.resize(B[i][y0:y1, x0:x1], (560, 400))
        cv2.putText(a, 'ORIGINAL (take)', (10, 30), 0, 0.9, (0, 255, 255), 2)
        cv2.putText(b, 'OURS', (10, 30), 0, 0.9, (0, 255, 255), 2)
        cv2.putText(b, f'f{i}  {t:.2f}s', (10, 385), 0, 0.7, (255, 255, 255), 2)
        pan = np.zeros((PH, PW, 3), np.uint8)
        ts = t - win + xs / PX
        for x in xs:
            s = int(ts[x] * SR)
            e = int((ts[x] + 1 / PX) * SR)
            if 0 <= s < len(au) and e > s:
                h = int(min(1, np.abs(au[s:e]).max() * 1.6) * 70)
                cv2.line(pan, (x, 75 - h), (x, 75 + h), (255, 200, 60), 1)
        for tt, v, col, lab, yl in ((dt, dn, (120, 120, 255), 'audio target opening', 175),
                                    (ft, mn, (80, 255, 80), 'OUR mouth opening (measured)', 200),
                                    (ft, gn, (60, 140, 60), 'planned opening (warp egap)', 225),
                                    (ft, on, (150, 150, 150), 'take (original) opening', 250)):
            yy = np.interp(ts, tt, v, left=np.nan, right=np.nan)
            pts = [(x, int(PH - 15 - yy[x] * (PH - 170))) for x in xs if not np.isnan(yy[x])]
            if len(pts) > 1: cv2.polylines(pan, [np.array(pts, np.int32)], False, col, 2)
            cv2.putText(pan, lab, (10, yl), 0, 0.55, col, 2)
        cv2.putText(pan, stat, (PW - 640, 160), 0, 0.55, (255, 255, 255), 1)
        cv2.line(pan, (PW // 2, 0), (PW // 2, PH), (255, 255, 255), 2)
        fr = np.zeros((PH + 400, PW + CARD_W, 3), np.uint8)
        fr[:, :PW] = np.vstack([np.hstack([a, b]), pan])
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    src = raw
    if track:                                                         # the mouth card, lower right (the side column)
        from . import viseme_overlay
        carded = os.path.join(tmp, 'card.mp4')
        viseme_overlay.main(['--video', raw, '--track', track, '--out', carded, '--no-mediapipe', '--corner', 'lower-right',
                             '--scale', '1.0'])
        src = carded
    ff.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-c', 'copy', '-movflags', '+faststart', out], check=True)
    if half:
        ff.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-filter_complex', '[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]',
                        '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart',
                        out.replace('.mp4', '_half.mp4')], check=True)
    print(f"sync-qa {out}: {stat}; roi {roi}")
    return dict(corr=m0, best=mb, lag_ms=mlag, measured_corr=m0, face_cover=cover, plan_corr=c0, plan_best=cb,
                plan_lag_ms=lag, disagree=disagree(c0, m0), lag_sign='+ = mouth trails the audio', width_corr=wc, roi=roi)


# ---------------------------------------------------------------------------------------------------- delivered film
def read_span(video, a, b):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    F = []
    for _ in range(a, b):
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F


def singer_roi(F, cast_frames=None, faces=None, k=2.4):
    """A square face box for the cut: the median singer box (cast frames {frame: box} if given, else the largest
    YuNet / MediaPipe face on a few sampled frames), grown k x around its centre. -> [y0, y1, x0, x1] or None."""
    B = list(cast_frames.values()) if cast_frames else []
    if not B:
        from ..core.cast import Faces
        faces = faces or Faces()
        for i in np.linspace(0, len(F) - 1, min(5, len(F))).astype(int):
            fs = faces(F[i])
            if fs: B.append(max(fs, key=lambda f: f['face_h'])['box'])
    if not B: return None
    x0, y0, x1, y1 = np.median(np.array(B, float), 0)
    H, W = F[0].shape[:2]
    s = k * max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [int(max(0, cy - s / 2)), int(min(H, cy + s / 2)), int(max(0, cx - s / 2)), int(min(W, cx + s / 2))]


def cast_singer_frames(cast, g0, g1, chars=None, preset=None):
    """{video frame: box} of the largest singer face per frame in [g0, g1) from a whole-video cast scan. chars: the
    singers (default: the preset's, or any named character)."""
    from .. import presets as P
    chars = set(chars) if chars else P.singer_set(preset, cast)
    out = {}
    for r in cast['results']:
        for t in r['tracks']:
            if t['character'] not in chars: continue
            for x in t['frames']:
                if g0 <= x['f'] < g1 and (x['f'] not in out or x['face_px'] > out[x['f']][1]): out[x['f']] = (x['box'], x['face_px'])
    return {f: b for f, (b, _) in out.items()}


def score_film(video, cuts, vocal, offset=12.0, align=None, cast=None, work=None, min_words=2, preset=None):
    """Delivered-film mode: per SUNG cut, the singer's measured mouth opening vs the vocal. Target = the aligned sung
    opening (sung_dyn geo_gap) when --align is given, else the vocal RMS envelope. -> list of per-cut dicts."""
    from . import snap as SN
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    work = work or tempfile.mkdtemp(prefix='syncqa_film_')
    os.makedirs(work, exist_ok=True)
    words = json.load(open(align))['words'] if align else []
    rows = []
    for ci, c in enumerate(cuts['cut']):
        g0 = int(round(c['start'] * fps))
        g1 = int(round((c['start'] + c['dur']) * fps))
        n = g1 - g0
        t0 = c['start'] - offset
        ws = [w for w in words if w['end'] > t0 and w['start'] < t0 + c['dur']] if words else []
        vox = os.path.join(work, f'vox_{ci:03d}.wav')
        if t0 < 0: continue
        ff.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t0:.3f}', '-t', f'{c["dur"]:.3f}', '-i', vocal, vox], check=True)
        env = SN.envelope(vox, n, fps)
        sung = len(ws) >= min_words if words else (env > SN.SIL_RMS).mean() >= 0.3
        if not sung: continue
        F = read_span(video, g0, g1)
        roi = singer_roi(F, cast_singer_frames(cast, g0, g1, preset=preset) if cast else None)
        row = {'cut': ci, 'shot': c.get('shot'), 'video_t': [round(c['start'], 3), round(c['start'] + c['dur'], 3)], 'frames': len(F),
               'words': ' '.join(w['w'] for w in ws), 'roi': roi}
        if roi is None:
            rows.append({**row, 'corr': None, 'note': 'no face'})
            continue
        op, cover = measured_opening(F, roi, fps)
        if op is None:
            rows.append({**row, 'corr': None, 'cover': round(cover, 3), 'note': 'no landmarks'})
            continue
        if align:
            dyn = os.path.join(work, f'dyn_{ci:03d}.csv')
            call(SD.main, ['--align', align, '--audio', vocal, '--t0', f'{t0:.3f}',
                            '--dur', f'{c["dur"] + 0.5:.3f}', '--out', dyn], quiet=True, name='sung_dyn')
            R = list(csv.DictReader(open(dyn)))
            tg = np.interp(np.arange(len(op)) / fps, np.array([float(r['time_s']) for r in R]) - t0, np.array([float(r['geo_gap_cm']) for r in R]))
        else:
            tg = env[:len(op)]
        c0, cb, lag = tracking(op, tg, fps)
        e0 = tracking(op, env[:len(op)], fps)[0]
        sn = SN.snap_metrics(op, vox, words, t0, fps)
        c1 = max(c0, tracking(op, tg, fps, max_lag=1)[1])
        ok = c1 >= 0.5 and (sn['open_in_silence'] is None or sn['open_in_silence'] <= 0.35)
        rows.append({**row, 'mouth_range': round(float(np.percentile(op, 95) - np.percentile(op, 5)), 4), 'corr': round(c0, 3), 'corr_pm1': round(c1, 3), 'best': round(cb, 3), 'lag_ms': lag, 'corr_env': round(e0, 3), 'cover': round(cover, 3),
                     'target': 'sung_dyn' if align else 'rms', **sn, 'pass': bool(ok)})
        print(f"cut {ci:2d} {str(c.get('shot')):<18} corr {c0:+.2f} (best {cb:+.2f} at {lag:+.0f} ms) sil {sn['open_in_silence']} "
              f"rise {sn['rise_ms']} closures {sn['closure_hit']} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return rows


def film_md(rows):
    L = ['| cut | shot | corr (best @lag ms) | open in silence | rise ms | closure hit (n) | gate |', '|---|---|---|---|---|---|---|']
    for r in rows:
        if r.get('corr') is None:
            L.append(f"| {r['cut']} | {r['shot']} | - | - | - | - | {r.get('note')} |")
            continue
        L.append(f"| {r['cut']} | {r['shot']} | {r['corr']:+.2f} ({r['best']:+.2f} @{r['lag_ms']:+.0f}) | {r['open_in_silence']} | {r['rise_ms']} | "
                 f"{r['closure_hit']} ({r['closures']}) | {'PASS' if r['pass'] else '**FAIL**'} |")
    return '\n'.join(L) + '\n'


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--video' in argv:                                             # delivered-film mode
        ap = argparse.ArgumentParser(prog='lsdub sync-qa --video')
        ap.add_argument('--video', required=True)
        ap.add_argument('--cuts', required=True)
        ap.add_argument('--vocal', required=True)
        ap.add_argument('--offset', type=float, default=12.0, help='video time = song time + offset')
        ap.add_argument('--align', help='lyric alignment: target = the aligned sung opening, sung cuts = cuts with words, '
                        'and the p/b/m closure metric (without it: vocal RMS target, no closures)')
        ap.add_argument('--cast', help='whole-video cast scan: the singer box per frame (else the largest face)')
        ap.add_argument('--preset', help='cast preset (presets.py), e.g. landing')
        ap.add_argument('--json')
        ap.add_argument('--md')
        ap.add_argument('--work')
        a = ap.parse_args(argv)
        rows = score_film(a.video, json.load(open(a.cuts)), a.vocal, a.offset, a.align, json.load(open(a.cast)) if a.cast else None, a.work, preset=a.preset)
        if a.json: json.dump({'video': a.video, 'offset': a.offset, 'cuts': rows}, open(a.json, 'w'), indent=1)
        if a.md: open(a.md, 'w').write(film_md(rows))
        ok = [r for r in rows if r.get('corr') is not None]
        print(f"sync-qa film {a.video}: {sum(r['pass'] for r in ok)}/{len(ok)} sung cuts PASS (corr >= 0.5, open-in-silence <= 0.35)")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', required=True)
    ap.add_argument('--render', required=True)
    ap.add_argument('--warp-json', required=True)
    ap.add_argument('--dyn', required=True, help='the --coart-dyn curve (song time)')
    ap.add_argument('--t0', type=float, required=True, help='song time of the clip start')
    ap.add_argument('--audio', required=True, help='a file with the clip audio from clip time 0 (the _av.mp4)')
    ap.add_argument('--track', help='viseme track: adds the mouth-shape card')
    ap.add_argument('--out')
    ap.add_argument('--roi', help='y0,y1,x0,x1 (default: where the render differs from the take)')
    ap.add_argument('--half', action='store_true', help='also write a half-speed copy (_half.mp4)')
    a = ap.parse_args(argv)
    render(a.take, a.render, a.warp_json, a.dyn, a.t0, a.audio, a.out or a.render.replace('.mp4', '_syncqa.mp4'), a.track,
           [int(v) for v in a.roi.split(',')] if a.roi else None, a.half)


if __name__ == '__main__':
    main()
