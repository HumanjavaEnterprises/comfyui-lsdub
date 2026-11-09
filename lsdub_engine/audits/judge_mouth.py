# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""judge_mouth.py: the event judge's own reading of the mouth on the delivered pixels. CPU, MediaPipe.

A frozen copy of the measurement the dub's gate makes (inner-lip gap 13-14 over the inter-ocular distance 33-263, on a face crop
upscaled 2x, the face landmarker in video mode at confidence 0.2), kept here so the judge's readings never move when the driver
changes how it measures. The only thing taken from core/ is the landmarker FACTORY (core/landmarker.get: the repo allows one place
that makes a MediaPipe landmarker, for the GL guard); the measurement is this file's own. tests/test_judge_frozen.py pins the series
this yields on a fixture clip.

  frames(video, n=0)                  the frames of a video (BGR) and its fps
  face_box(frames)                    the largest face on a few sampled frames -> [x0, y0, x1, y1] or None
  roi_of(frames, boxes=None)          a square ROI K x the face box around its centre -> [y0, y1, x0, x1]
  opening(frames, roi, fps)           (gap in eye distances, gap over face height, cover) per frame, gaps filled by interpolation
  opening_of(video, boxes=None)       the three on a video file, plus fps

Change nothing here to improve a dub.
"""

import numpy as np

from .._models import face_landmarker_path  # noqa: F401  (the model the factory loads; named here for the tests)

CONF = 0.2
UP = 2  # the crop is upscaled this much before the landmarker reads it
K = 2.4  # the ROI side, in face-box sides
SAMPLES = 5  # frames read to find the face box
GAP = (13, 14)
EYES = (33, 263)
FACE = (10, 152)


def _mp():
    """the mediapipe module, through the one factory the repo allows (core/landmarker: the process-level GL guard lives there; the
    factory is the single piece shared with the driver, the measurement below is the judge's own)."""
    from ..core import landmarker

    return landmarker.mp()


def landmarker(video_mode, num_faces=1, conf=CONF):
    from ..core import landmarker as LM

    return LM.get("video" if video_mode else "image", conf=conf, num_faces=num_faces)


def frames(video, n=0):
    import cv2

    cap = cv2.VideoCapture(video)
    F = []
    while not n or len(F) < n:
        ok, f = cap.read()
        if not ok:
            break
        F.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return F, fps


def _image(mp, bgr):
    import cv2

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def face_box(F):
    """the largest face's box [x0, y0, x1, y1] (median over SAMPLES frames), by the landmarks' extent; None when no face is found."""
    mp = _mp()
    det = landmarker(False, num_faces=4)
    B = []
    for i in np.linspace(0, len(F) - 1, min(SAMPLES, len(F))).astype(int):
        r = det.detect(_image(mp, F[i]))
        best = None
        for L in r.face_landmarks:
            xs, ys = [p.x for p in L], [p.y for p in L]
            h, w = F[i].shape[:2]
            box = [min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]
            if best is None or (box[3] - box[1]) > (best[3] - best[1]):
                best = box
        if best:
            B.append(best)
    return list(np.median(np.array(B, float), 0)) if B else None  # an image-mode instance is shared: not closed here


def roi_of(F, boxes=None):
    """[y0, y1, x0, x1]: a square K x the face box (the median of `boxes` {frame: [x0, y0, x1, y1]} when given, else face_box), clipped;
    the whole frame when no face is found."""
    H, W = F[0].shape[:2]
    B = list(boxes.values()) if boxes else []
    box = list(np.median(np.array(B, float), 0)) if B else face_box(F)
    if box is None:
        return [0, H, 0, W]
    x0, y0, x1, y1 = box
    s = K * max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [int(max(0, cy - s / 2)), int(min(H, cy + s / 2)), int(max(0, cx - s / 2)), int(min(W, cx + s / 2))]


def opening(F, roi, fps=25.0):
    """(gap / inter-ocular, gap / face height, cover) per frame on the ROI crop upscaled UP x; NaN frames filled by interpolation
    when at least five frames read; None series when fewer did."""
    import cv2

    mp = _mp()
    det = landmarker(True)
    y0, y1, x0, x1 = roi
    giod, gface = np.full(len(F), np.nan), np.full(len(F), np.nan)
    for i, f in enumerate(F):
        c = cv2.resize(f[y0:y1, x0:x1], None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
        r = det.detect_for_video(_image(mp, c), int(i * 1000 / fps))
        if r.face_landmarks:
            L = r.face_landmarks[0]
            p = lambda j: np.array([L[j].x * c.shape[1], L[j].y * c.shape[0]])
            gap = np.linalg.norm(p(GAP[0]) - p(GAP[1]))
            giod[i] = gap / (np.linalg.norm(p(EYES[0]) - p(EYES[1])) + 1e-9)
            gface[i] = gap / (np.linalg.norm(p(FACE[0]) - p(FACE[1])) + 1e-9)
    det.close()
    ok = np.isfinite(giod)
    fill = lambda a: np.interp(np.arange(len(a)), np.where(ok)[0], a[ok]) if ok.sum() >= 5 else None
    return fill(giod), fill(gface), float(ok.mean()) if len(F) else 0.0


def opening_of(video, boxes=None, n=0):
    """(gap / inter-ocular, gap / face height, cover, fps) of a video file."""
    F, fps = frames(video, n)
    if not F:
        return None, None, 0.0, fps
    giod, gface, cover = opening(F, roi_of(F, boxes), fps)
    return giod, gface, cover, fps
