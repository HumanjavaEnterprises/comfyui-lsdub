#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""qa.py: `lsdub qa`, a general LSDub QA tool for a contributed long video with many shots. CPU (own shot
detector; YuNet + MediaPipe faces; DINOv2 identity, all already in LSDub). Two modes:

  triage  BEFORE the dub: which shots are fit to redub.
          shots     own detector: per frame the grey frame difference and an HSV histogram distance at 160x90; a cut is
                    a SPIKE (>= --cut-px grey levels, >= --cut-ratio x the local median of the +-6 frames around it, with
                    either a histogram jump or a very large change). A gradual fade never spikes; a flash (the frame
                    after the spike returns to the frame before it) is not a cut. Shots shorter than 3 frames merge.
          per shot  up to --samples frames: every face (YuNet + MediaPipe landmarks), its identity against the given
                    references (cast db / --refs dir / --face name=img), the largest TARGET face: px @1080 and native,
                    yaw, mouth visible (landmarks, no occlusion, no chrome mic at the lips), motion, blur, length;
                    with --vocal, voice activity (energy VAD on the stem) over the shot.
          verdict   DUB         target >= 300 px @1080, |yaw| <= 16, mouth clear  -> the close-up warp (lsdub paint / sing)
                    DUB-MEDIUM  target 120-300 px @1080 (or yaw 16-35)             -> the tier-2 crop (lsdub sing / medium)
                    NO          too short / no face / wrong person / target rarely on screen / profile / mouth hidden / too
                                small, each with its number
          out       <out>.json (THE SHOT LIST the dub step takes), .md / .csv ranked, _sheet.jpg coloured by verdict
  verify  AFTER the dub: how likely each dubbed shot is to have succeeded.
          per DUB / DUB-MEDIUM shot of the triage json: the measured mouth opening on the dubbed video vs the vocal
          (sync-qa's measured opening; target = the aligned sung opening with --align, else the vocal envelope),
          open-in-silence, snap, mouth travel, mouth visible, the tier; classes PERF / COVERAGE / INSERT, the travel cap
          and the reason priority of `lsdub scorecard`; a 0-100 LIKELY-SUCCESS score and a band PASS (>= 75) /
          REVIEW (>= 55) / FAIL; identity kept (DINOv2 cosine of the target's head, dubbed vs --before, < 0.85 =
          identity drift -> REVIEW); with --before, the same score of the original and the delta. A shot the dub left as
          source (`used: source` in the json) is measured with the rest and banded N/A. A shot carrying `warp_json` leaves its
          vocalisation frames out of the corr and reports vocal_frac. mouth_identity_cos is the same cosine on the lower
          face: reported, it moves no band. `path` is the paint delivered (sing, native, source) and `lag_frames` how many of a sing
          shot's frames were picked from before their own time, so the skin under the paint is a frame old: read from the shot's
          gate.json, reported, moving no band; blank with no gate.

  lsdub qa triage --video film.mp4 [--vocal stem.wav --offset 0] [--cast-db db.npz | --refs dir | --face name=ref.png ...]
                  [--target name ...] --out dir/name
  lsdub qa verify --video dubbed.mp4 --vocal stem.wav --shots name.json [--before orig.mp4] [--align align.json] --out dir/name
"""

from ..io import ff
import argparse, csv, glob, json, os, sys, tempfile
import cv2, numpy as np
from ..core import steps
from ..core.vocalisations import scored_frames


def dyn_curve(R, n, t0, fps, moved=None):
    """the opening built from the alignment (sung_dyn's rows) on n frames from t0; moved: coart.logged_offsets."""
    t = np.arange(n) / fps + (0.0 if moved is None else moved)
    return np.interp(t, np.array([float(x["time_s"]) for x in R]) - t0, np.array([float(x["geo_gap_cm"]) for x in R]))


DUB_PX, MED_PX, DUB_YAW, PROFILE_YAW = 300.0, 120.0, 16.0, 35.0
MIN_SHOT, MIN_PRESENT, MIN_VISIBLE = 12, 0.5, 0.5
VAD_REL, VAD_FLOOR, VOICED_FRAC = 0.12, 0.01, 0.3
PASS_BAND, REVIEW_BAND, ID_DRIFT = 75.0, 55.0, 0.85
VOCAL_NOTE = 0.25  # a shot this much vocalisation says so when it does not pass
VERDICTS = ("DUB", "DUB-MEDIUM", "NO")


# ---------------------------------------------------------------------------------------------------- shots
def frame_features(video, size=(160, 90)):
    """-> (hist distance, grey diff) per frame (index 0 = 0), fps, (W, H), n."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dh, dp, ph, pg = [], [], None, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        h = cv2.calcHist([cv2.cvtColor(s, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        h = (h / (h.sum() + 1e-9)).astype(np.float32)
        g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dh.append(0.0 if ph is None else float(cv2.compareHist(ph, h, cv2.HISTCMP_BHATTACHARYYA)))
        dp.append(0.0 if pg is None else float(np.abs(g - pg).mean()))
        ph, pg = h, g
    return np.array(dh), np.array(dp), fps, (W, H), len(dp), None


def detect_cuts(dh, dp, cut_px=25.0, ratio=3.0, hist_min=0.10, big=45.0, flash=None, min_gap=3):
    """frame indices where a new shot starts. flash: optional (n,) grey diff of frame i+k vs i-1 for k in 1..3 (min);
    a spike whose later frames return to the frame before it is a flash, not a cut."""
    n = len(dp)
    out = []
    for i in range(1, n):
        w = np.r_[dp[max(1, i - 6) : i], dp[i + 1 : i + 7]]
        loc = max(2.0, float(np.median(w))) if len(w) else 2.0
        if not (dp[i] >= cut_px and dp[i] >= ratio * loc and (dh[i] >= hist_min or dp[i] >= big)):
            continue
        if flash is not None and flash[i] < 0.35 * dp[i]:
            continue  # the picture comes back: a flash
        if out and i - out[-1] < min_gap:
            if dp[i] > dp[out[-1]]:
                out[-1] = i
            continue
        out.append(i)
    return out


def flash_returns(video, cuts, size=(160, 90)):
    """for candidate spike frames i: min grey diff between frame i-1 and frames i+1..i+3 (a flash returns)."""
    cap = cv2.VideoCapture(video)
    G = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        G.append(cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32))
    out = np.full(len(G), np.inf)
    for i in cuts:
        if 1 <= i < len(G) - 1:
            out[i] = min(float(np.abs(G[j] - G[i - 1]).mean()) for j in range(i + 1, min(len(G), i + 4)))
    return out


def shots_from_cuts(cuts, n, fps):
    b = [0] + list(cuts) + [n]
    return [
        {
            "shot": k,
            "f0": b[k],
            "f1": b[k + 1] - 1,
            "start": round(b[k] / fps, 3),
            "end": round(b[k + 1] / fps, 3),
            "frames": b[k + 1] - b[k],
        }
        for k in range(len(b) - 1)
    ]


def detect_shots(video):
    dh, dp, fps, size, n, _ = frame_features(video)
    cand = detect_cuts(dh, dp)
    fl = flash_returns(video, cand)
    cuts = detect_cuts(dh, dp, flash=fl)
    return shots_from_cuts(cuts, n, fps), fps, size, (dh, dp)


def match_cuts(detected, truth, tol=2):
    """share of true boundaries with a detected boundary within +-tol frames, and the unmatched detections."""
    hit = [g for g in truth if any(abs(g - d) <= tol for d in detected)]
    fp = [d for d in detected if not any(abs(g - d) <= tol for g in truth)]
    return (len(hit) / len(truth) if truth else 1.0), [g for g in truth if g not in hit], fp


# ---------------------------------------------------------------------------------------------------- identity
def build_db(cast_db=None, refs=None, faces_kv=(), model=None, emb=None):
    """the identity db: a cast npz, or enrolled from a refs dir (<dir>/<name>/*.png|jpg) and/or name=image pairs."""
    from ..core import cast as C

    if cast_db:
        return C.load_db(cast_db)
    imgs = []
    if refs:
        for d in sorted(glob.glob(os.path.join(refs, "*"))):
            if os.path.isdir(d):
                fs = sorted(
                    f
                    for e in ("png", "jpg", "jpeg", "webp")
                    for f in glob.glob(os.path.join(d, "**", f"*.{e}"), recursive=True)
                    if "back" not in os.path.basename(f).lower()
                )
                imgs += [(os.path.basename(d), f) for f in fs[:16]]
    imgs += [tuple(kv.split("=", 1)) for kv in faces_kv]
    if not imgs:
        return None
    return C.enroll(imgs, C.Faces(), emb or C.Embedder(model or C.MODEL))


def identify(db, emb, img, f, preset=None):
    """one face -> (name | 'unknown', confidence, best score) from a single head + face crop."""
    from ..core import cast as C

    hc = C.head_crop(img, f["det_box"])
    E = emb([hc])
    EF = emb([C.face_crop(img, f["det_box"])])
    sc = C.score_track(db, E, [C.colour_sig(hc)], [C.colour_sig(C.body_crop(img, f["det_box"]))], EF=EF)
    who, conf, _ = C.decide(sc, preset=preset)
    return who, conf, float(max(sc.values())), sc


def head_embedding(emb, img, box):
    from ..core import cast as C

    return emb([C.head_crop(img, box)])[0]


# ---------------------------------------------------------------------------------------------------- measures
def read_frames(video, idx):
    cap = cv2.VideoCapture(video)
    out = {}
    for i in sorted(set(int(j) for j in idx)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            out[i] = f
    return out


def vad(vocal, n, fps, offset=0.0):
    """per video frame: voice active (energy VAD on the stem; video time = audio time + offset)."""
    sr = 16000
    a = np.frombuffer(
        ff.run(
            ["ffmpeg", "-v", "error", "-i", vocal, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True
        ).stdout,
        np.float32,
    )
    hop = sr / fps
    e = np.zeros(n)
    for i in range(n):
        s = int((i / fps - offset) * sr)
        if 0 <= s < len(a):
            e[i] = np.sqrt(np.mean(a[s : s + int(hop)] ** 2))
    thr = max(VAD_FLOOR, VAD_REL * float(np.percentile(e[e > 0], 95)) if (e > 0).any() else VAD_FLOOR)
    act = e > thr
    return np.convolve(act.astype(float), np.ones(3), "same") >= 2, e


def blur(img, box):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    c = img[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)]
    return float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if c.size > 64 else 0.0


def mouth_clear(img, f):
    """landmarks read, cast's lip-region occlusion test passes, no chrome mic at the lips."""
    from ..core import cast as C
    from ..core.grammar import mic_share, MIC_SHARE

    if f.get("lm") is None:
        return False, None, None
    occ, share = C.mouth_occluded(img, f["lm"])
    mic = mic_share(img, f["box"])
    return (not occ and mic <= MIC_SHARE), share, round(mic, 3)


# ---------------------------------------------------------------------------------------------------- triage
def triage(video, vocal=None, offset=0.0, db=None, targets=(), samples=6, model=None, detected=None, preset=None, emb=None):
    """detected: a detect_shots() result to reuse (shots, fps, (W, H), (dh, dp)); preset: cast family names."""
    from ..core import cast as C

    shots, fps, (W, H), (dh, dp) = detected if detected is not None else detect_shots(video)
    shots = [dict(s) for s in shots]
    n = shots[-1]["f1"] + 1 if shots else 0
    faces = C.Faces()
    emb = (emb or C.Embedder(model or C.MODEL)) if db is not None else None
    act = vad(vocal, n, fps, offset)[0] if vocal else None
    want = set(targets) if targets else (set(db["name"].tolist()) if db is not None else set())
    for s in shots:
        L = s["frames"]
        s["motion"] = round(float(np.median(dp[s["f0"] + 1 : s["f1"] + 1])) if L > 1 else 0.0, 3)
        if act is not None:
            fr = float(act[s["f0"] : s["f1"] + 1].mean())
            s.update(voice_frac=round(fr, 3), voiced=fr >= VOICED_FRAC)
        idx = np.unique(np.linspace(s["f0"] + min(2, L // 4), s["f1"] - min(2, L // 4), min(samples, L)).astype(int))
        F = read_frames(video, idx)
        per = []
        for i, img in F.items():
            fs = faces(img)
            tgt = None
            others = []
            for f in fs:
                who, conf, best, _ = identify(db, emb, img, f, preset) if db is not None else ("face", 1.0, 1.0, None)
                f["who"] = who
                f["conf"] = conf
                if (not want or who in want) and (tgt is None or f["face_h"] > tgt["face_h"]):
                    tgt = f
                else:
                    others.append(f)
            rec = {"f": int(i), "faces": len(fs), "largest": max(fs, key=lambda f: f["face_h"]) if fs else None, "target": tgt}
            if tgt is not None:
                ok, share, mic = mouth_clear(img, tgt)
                rec.update(clear=ok, occ_share=share, mic=mic, blur=blur(img, tgt["box"]))
            per.append(rec)
        T = [p for p in per if p["target"] is not None]
        s["faces"] = int(np.median([p["faces"] for p in per])) if per else 0
        lg = [p["largest"] for p in per if p["largest"] is not None]
        if lg:
            big = max(lg, key=lambda f: f["face_h"])
            s["largest"] = {"who": big.get("who"), "px1080": round(big["face_px1080"], 1)}
        s["present"] = round(len(T) / len(per), 3) if per else 0.0
        if T:
            tg = [p["target"] for p in T]
            s.update(
                target=max(sorted(set(f["who"] for f in tg)), key=[f["who"] for f in tg].count),  # a tie -> the first name, every run
                px1080=round(float(np.median([f["face_px1080"] for f in tg])), 1),
                px=round(float(np.median([f["face_h"] for f in tg])), 1),
                yaw=round(float(np.median([f["yaw"] for f in tg])), 1),
                mouth_visible=round(float(np.mean([p["clear"] for p in T])), 3),
                mic=max([p["mic"] for p in T if p["mic"] is not None], default=None),
                blur=round(float(np.median([p["blur"] for p in T])), 1),
                box=[round(float(np.median([f["box"][j] for f in tg])), 1) for j in range(4)],
                boxes={str(p["f"]): [round(v, 1) for v in p["target"]["box"]] for p in T},
            )
        s["verdict"], s["reason"] = verdict(s, db is not None)
    meta = {
        "video": video,
        "fps": fps,
        "size": [W, H],
        "frames": n,
        "targets": sorted(want),
        "vocal": vocal,
        "offset": offset,
        "rules": {
            "dub_px1080": DUB_PX,
            "medium_px1080": MED_PX,
            "dub_yaw": DUB_YAW,
            "profile_yaw": PROFILE_YAW,
            "min_shot_frames": MIN_SHOT,
            "min_present": MIN_PRESENT,
            "min_mouth_visible": MIN_VISIBLE,
        },
    }
    return meta, shots


def verdict(s, has_refs):
    """-> (DUB | DUB-MEDIUM | NO, the exact reason with its number). Checks in this order."""
    if s["frames"] < MIN_SHOT:
        return "NO", f"too short ({s['frames']} frames < {MIN_SHOT})"
    if not s.get("faces") and "px1080" not in s:
        return "NO", "no face"
    if "px1080" not in s:
        lg = s.get("largest") or {}
        return "NO", (f"wrong person (largest face = {lg.get('who')} {lg.get('px1080')} px)" if has_refs else "no face")
    if s["present"] < MIN_PRESENT:
        return "NO", f"target on screen only {s['present']:.0%} of the shot"
    if abs(s["yaw"]) > PROFILE_YAW:
        return "NO", f"profile (yaw {s['yaw']:+.0f} deg > {PROFILE_YAW:.0f})"
    if s["mouth_visible"] < MIN_VISIBLE:
        why = "mic at the lips" if (s.get("mic") or 0) > 0.1 else "mouth occluded / unread"
        return "NO", f"mouth hidden ({why}; visible {s['mouth_visible']:.0%})"
    if s["px1080"] < MED_PX:
        return "NO", f"too small ({s['px1080']:.0f} px @1080 < {MED_PX:.0f})"
    if s["px1080"] >= DUB_PX and abs(s["yaw"]) <= DUB_YAW:
        return "DUB", f"close-up {s['px1080']:.0f} px @1080, yaw {s['yaw']:+.0f}"
    if s["px1080"] >= DUB_PX:
        return "DUB-MEDIUM", f"turned (yaw {s['yaw']:+.0f} deg > {DUB_YAW:.0f}): crop path"
    return "DUB-MEDIUM", f"medium {s['px1080']:.0f} px @1080: crop path"


# ---------------------------------------------------------------------------------------------------- verify
def verify(video, vocal, shots_json, before=None, align=None, model=None, work=None, emb=None):
    from . import sync_qa as Q, snap as SN, scorecard as SC
    from ..core import cast as C

    J = json.load(open(shots_json))
    fps = J["fps"]
    offset = J.get("offset", 0.0)
    work = work or tempfile.mkdtemp(prefix="lsdub_qa_")
    os.makedirs(work, exist_ok=True)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    act, env = vad(vocal, n, fps, offset)
    words = json.load(open(align))["words"] if align else []
    vocal_words = heard = None
    if align:
        vocal_words, heard = vocal_walk(json.load(open(align)), vocal)
    faces = C.Faces()
    emb = (emb or C.Embedder(model or C.MODEL)) if before else None
    rows = []
    for s in J["shots"]:
        if s["verdict"] not in ("DUB", "DUB-MEDIUM"):
            continue
        r = {k: s.get(k) for k in ("shot", "f0", "f1", "start", "end", "verdict", "target", "px1080")}
        r.update(measure_shot(video, s, fps, offset, vocal, act, align, words, faces, work, Q, SN))
        r.update(vocal_reading(words, vocal_words, s, offset, heard))
        r.update(path_reading(s))
        if before:
            b = measure_shot(before, s, fps, offset, vocal, act, align, words, faces, os.path.join(work, "before"), Q, SN)
            r["before"] = {k: b.get(k) for k in ("corr", "mouth_range", "open_in_silence", "mouth_visible")}
            r["identity_cos"] = identity_cos(video, before, s, emb)
            r["mouth_identity_cos"] = identity_cos(video, before, s, emb, lower_face)
        rows.append(r)
    sh = [r["sharpness"] for r in rows if r.get("sharpness")]
    smed = float(np.median(sh)) if sh else None
    for r in rows:
        score_row(r, smed, SC)
        if "before" in r:
            b = dict(r, **r["before"])
            b["corr_pm1"] = b.get("corr")
            score_row(b, smed, SC)
            r["before"]["score"] = b["score"]
            r["delta"] = None if r["score"] is None or b["score"] is None else round(r["score"] - b["score"], 1)
    for r in rows:
        s = next(x for x in J["shots"] if x.get("shot") == r["shot"])
        if s.get("used") == "source":  # after the scoring: a shot left as source still counts in the film's medians
            r.update(band="N/A", score=None, delta=None, reason=f"left as source: {s.get('left_reason') or 'no render used'}")
    meta = {"video": video, "shots_json": shots_json, "before": before, "bands": {"PASS": PASS_BAND, "REVIEW": REVIEW_BAND}}
    meta["closures_lost"] = SN.lost_closures(words, n / fps - offset)
    return meta, rows


def shape_readings(shape, warp_json, words, t0, fps, SN):
    """the delivered mouth's shape in eye distances, reported, moving no band: width travel, real closures, height contrast by class."""
    if shape is None:
        return {}
    from ..core.coart import logged_classes
    from .viseme_overlay import GLYPH

    gap, width, *corner = shape
    out = {"width_range": round(float(np.nanpercentile(width, 95) - np.nanpercentile(width, 5)), 4)}
    fr = SN.closures(words, t0, len(gap), fps)
    out["closure_real"], out["closure_real_n"] = SN.closure_real(gap, fr)
    if corner:
        out["closure_corner"] = SN.closure_corner(gap, corner[0], fr)
    cl = logged_classes(warp_json, len(gap), fps)
    if cl:
        opens = {k: v["oi"] for k, v in GLYPH.items() if k not in ("sil",)}
        out["height_contrast"], out["height_r"], out["class_gap"] = SN.class_contrast(gap, cl, opens)
    return out


def vocal_walk(align, vocal):
    """the lyric alone over the vocal, no time of the plan's going in, and the vocal's samples and features for the run: -> (words, (samples, features))."""
    from ..core.vocal_retime import features, read_mono, retime

    x = read_mono(vocal)
    return retime(align, vocal, x=x, free=True)["words"], (x, features(x))


def vocal_reading(words, vocal_words, s, offset, vocal=None):
    """the plan's word times against the vocal itself on this shot: reported, moving no band. The plan's own words, as the dub moved them.
    vocal: the vocal's (samples, features), once for the whole run."""
    if not vocal_words or not s.get("warp_json"):
        return {}
    from ..core.coart import logged_words
    from ..core.vocal_retime import plan_hits, word_offsets

    done = logged_words(words, s["warp_json"])
    a, b = s["start"] - offset, s["end"] - offset
    return (
        dict(word_offsets(done, vocal_words, a, b), **plan_hits(done, vocal[0], a, b, vocal[1]))
        if vocal
        else word_offsets(done, vocal_words, a, b)
    )


def lag_keys(picked, m):
    """the output frames whose pick sits before their own: each frame is painted m times, so a pick under k * m is an older frame's skin."""
    return [k for k, j in enumerate(picked) if int(j) < k * m]


def gate_reading(gate):
    """(path, lag frames) off a shot's gate.json: (None, None) with none. The picks are the sing paint's, written whatever won: a native
    shot delivers none of them. lag frames is None when a sing gate cannot say."""
    if not gate or not os.path.exists(gate):
        return None, None
    try:
        G = json.load(open(gate))
        path = (G.get("auto") or {}).get("winner") or ("sing" if G.get("medium") else None)
        if path is not None and not isinstance(path, str):
            return None, None
        if path != "sing":
            return path, [] if path else None
        picked, m = (G.get("medium") or {}).get("picked"), G.get("fps_mult")
        return path, lag_keys(picked, int(m)) if picked and m else None
    except (OSError, ValueError, TypeError, AttributeError, OverflowError):  # a reading only: a gate it cannot read leaves it blank
        return None, None


def delivered(s):
    """gate_reading of a shot of a shots.json, its cell the folder above its warp log. A shot left as source delivers no paint."""
    if s.get("used") == "source":
        return "source", []
    wj = s.get("warp_json")
    return gate_reading(os.path.join(os.path.dirname(os.path.dirname(wj)), "gate.json") if wj else None)


def path_reading(s):
    """the two report-only columns: path, lag_frames."""
    path, lag = delivered(s)
    return dict(path=path, lag_frames=None if lag is None else len(lag))


def nose_of(F, s):
    """the nose on these frames against the shot's own take; {} without one."""
    from . import mouth_ruler as MR

    take = s.get("warp_json") and os.path.join(s["warp_json"].split(os.sep + "work" + os.sep)[0], "take.mp4")
    if not take or not os.path.exists(take):
        return {}
    A = MR.read_clip(take, 0, len(F))
    n = min(len(A), len(F))
    if not n:
        return {}
    return MR.nose_reading(A[:n], MR.track(A[:n]), F[:n], MR.track(F[:n])) or {}


def measure_shot(video, s, fps, offset, vocal, act, align, words, faces, work, Q, SN):
    from ..core.coart import logged_dyn_flags, logged_offsets, logged_words

    os.makedirs(work, exist_ok=True)
    F = Q.read_span(video, s["f0"], s["f1"] + 1)
    out = {"voiced": bool(act[s["f0"] : s["f1"] + 1].mean() >= VOICED_FRAC) if len(F) else False}
    boxes = {int(k): v for k, v in (s.get("boxes") or {}).items()} or ({s["f0"]: s["box"]} if s.get("box") else {})
    roi = Q.singer_roi(F, boxes or None, faces)
    t0 = s["start"] - offset
    vox = os.path.join(work, f"vox_{s['shot']:04d}.wav")
    ff.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t0):.3f}", "-t", f"{len(F) / fps:.3f}", "-i", vocal, vox], check=True)
    env = SN.envelope(vox, len(F), fps)
    if roi is not None and F:
        op, cover = Q.measured_opening(F, roi, fps)
        if op is not None:
            moved = logged_offsets(s.get("warp_json"), len(op), fps)  # the plan the dub moved: the same reading moved with it
            if align:
                dyn = os.path.join(work, f"dyn_{s['shot']:04d}.csv")
                steps.call(
                    "sung_dyn",
                    [
                        "--align",
                        align,
                        "--audio",
                        vocal,
                        "--t0",
                        f"{t0:.3f}",
                        "--dur",
                        f"{len(F) / fps + 0.5:.3f}",
                        "--out",
                        dyn,
                        *logged_dyn_flags(s.get("warp_json")),
                    ],
                    quiet=True,
                )
                R = list(csv.DictReader(open(dyn)))
                tg = dyn_curve(R, len(op), t0, fps, moved)
            else:
                tg = env[: len(op)]
            keep, vocal_frac = scored_frames(s.get("warp_json"), len(op))
            c0, cb, lag = Q.tracking(op, tg, fps, keep=keep)
            c1 = max(c0, Q.tracking(op, tg, fps, max_lag=1, keep=keep)[1])
            out["vocal_frac"] = vocal_frac
            ec, eb, el = Q.tracking(op, np.asarray(env[: len(op)], float), fps, keep=keep)  # no alignment in this reading
            out.update(env_corr=round(ec, 3), env_best=round(eb, 3), env_lag_ms=el)
            if keep is not None:
                out["corr_all"] = round(Q.tracking(op, tg, fps)[0], 3)
            out.update(
                corr=round(c0, 3),
                corr_pm1=round(c1, 3),
                best=round(cb, 3),
                lag_ms=lag,
                cover=round(cover, 3),
                mouth_range=round(float(np.percentile(op, 95) - np.percentile(op, 5)), 4),
                **SN.snap_metrics(
                    op,
                    vox,
                    logged_words(words, s.get("warp_json")),
                    t0,
                    fps,
                    dt=moved,
                    keep=keep if any(w.get("vocal") == "pass-over" for w in words) else None,
                ),
            )
            out.update(shape_readings(Q.shape_of(F), s.get("warp_json"), logged_words(words, s.get("warp_json")), t0, fps, SN))
            out.update(nose_of(F, s))
    # mouth visible + tier on the DUBBED frames (a few samples)
    idx = np.linspace(0, len(F) - 1, min(5, len(F))).astype(int) if F else []
    vis, px = [], []
    for i in idx:
        fs = faces(F[i])
        if not fs:
            continue
        f = max(fs, key=lambda f: f["face_h"])
        px.append(f["face_px1080"])
        vis.append(mouth_clear(F[i], f)[0])
        out.setdefault("sharpness", blur(F[i], f["box"]))
    out["mouth_visible"] = round(float(np.mean(vis)), 3) if vis else 0.0
    out["owner_px"] = round(float(np.median(px)), 1) if px else 0.0
    from .scorecard import motion_flicker, SMALL

    g = [
        cv2.cvtColor(cv2.resize(x, (x.shape[1] // SMALL, x.shape[0] // SMALL), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
        for x in F
    ]
    d = [float(np.abs(g[i] - g[i - 1]).mean()) for i in range(1, len(g))] if len(g) > 1 else [0.0]
    med, pops = motion_flicker(d)
    out.update(motion=round(med, 3), pops=pops)
    return out


def lower_face(box):
    """the mouth end of a face box: its lower 45 %."""
    x0, y0, x1, y1 = box
    return [x0, y0 + 0.55 * (y1 - y0), x1, y1]


def identity_cos(video, before, s, emb, region=None):
    """DINOv2 cosine of the target's head crop, dubbed vs original, median over the triage sample frames. region: box -> the
    box to embed instead of the whole one."""
    boxes = {int(k): v for k, v in (s.get("boxes") or {}).items()}
    if not boxes:
        return None
    A, B = read_frames(video, boxes), read_frames(before, boxes)
    c = []
    for f, b in boxes.items():
        if f in A and f in B:
            q = region(b) if region else b
            c.append(float(np.dot(head_embedding(emb, A[f], q), head_embedding(emb, B[f], q))))
    return round(float(np.median(c)), 4) if c else None


def tier_of(px1080):
    return "cu" if px1080 >= DUB_PX else "medium" if px1080 >= MED_PX else "wide" if px1080 >= 25 else "none"


def score_row(r, smed, SC):
    """the scorecard's classes / components / caps on a verify row -> score + PASS / REVIEW / FAIL."""
    r["sung"] = bool(r.get("voiced"))
    r["need"] = "cu" if r.get("verdict") == "DUB" else "medium"
    r["tier"] = tier_of(r.get("owner_px") or 0)
    SC.finalize(r, smed)
    if r["class"] == "INSERT":
        r["band"] = "n/a"
        r["reason"] = "no voice in this shot: nothing to judge"
        r["score"] = None
    elif r["class"] == "COVERAGE":
        r["band"] = "REVIEW"
    else:
        r["band"] = "PASS" if r["score"] >= PASS_BAND else "REVIEW" if r["score"] >= REVIEW_BAND else "FAIL"
    if (r.get("vocal_frac") or 0) >= VOCAL_NOTE and r["band"] in ("REVIEW", "FAIL"):
        r["reason"] = f"vocalisation spans cover {r['vocal_frac']:.0%} of the shot; " + r["reason"]
    if r.get("identity_cos") is not None and r["identity_cos"] < ID_DRIFT:
        r["band"] = "FAIL" if r["band"] == "FAIL" else "REVIEW"
        r["reason"] = f"identity drift (cos {r['identity_cos']:.2f}); " + r["reason"]
    return r


# ---------------------------------------------------------------------------------------------------- outputs
def write_triage(meta, shots, out):
    order = sorted(shots, key=lambda s: (VERDICTS.index(s["verdict"]), -(s.get("px1080") or 0)))
    json.dump({**meta, "shots": shots}, open(out + ".json", "w"), indent=1)
    cols = [
        "shot",
        "start",
        "end",
        "frames",
        "verdict",
        "reason",
        "target",
        "px1080",
        "px",
        "yaw",
        "mouth_visible",
        "mic",
        "present",
        "faces",
        "motion",
        "blur",
        "voiced",
        "voice_frac",
    ]
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in order:
            w.writerow([s.get(k) for k in cols])
    cnt = {v: sum(s["verdict"] == v for s in shots) for v in VERDICTS}
    L = [
        f"# LSDub QA triage: {os.path.basename(meta['video'])}",
        "",
        f"{len(shots)} shots auto-detected; DUB {cnt['DUB']}, DUB-MEDIUM {cnt['DUB-MEDIUM']}, NO {cnt['NO']}. Targets: "
        f"{', '.join(meta['targets']) or 'the largest face'}. Rules: DUB >= {DUB_PX:.0f} px @1080 and |yaw| <= {DUB_YAW:.0f}, "
        f"medium >= {MED_PX:.0f}, mouth visible >= {MIN_VISIBLE:.0%}, shot >= {MIN_SHOT} frames.",
        "",
        "| shot | t (s) | frames | verdict | reason | target | px @1080 (native) | yaw | mouth | voiced |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in order:
        L.append(
            f"| {s['shot']} | {s['start']:.2f}-{s['end']:.2f} | {s['frames']} | **{s['verdict']}** | {s['reason']} | {s.get('target', '')} | "
            f"{'' if 'px1080' not in s else (format(s['px1080'], '.0f') + ' (' + format(s['px'], '.0f') + ')')} | {'' if 'yaw' not in s else (format(s['yaw'], '+.0f'))} | "
            f"{'' if 'mouth_visible' not in s else (format(s['mouth_visible'], '.0%'))} | {'' if 'voiced' not in s else ('yes' if s['voiced'] else 'no')} |"
        )
    open(out + ".md", "w").write("\n".join(L) + "\n")
    sheet(
        meta["video"],
        shots,
        out + "_sheet.jpg",
        lambda s: {"DUB": (60, 200, 60), "DUB-MEDIUM": (0, 170, 255), "NO": (60, 60, 210)}[s["verdict"]],
        lambda s: f"{s['verdict']} {'' if 'px1080' not in s else int(s['px1080'])}",
    )


def _audio_cell(r):
    return "" if r.get("env_best") is None else f"{r['env_best']:+.2f} @ {r['env_lag_ms']:+.0f}"


def _shape_cell(r):
    if r.get("width_range") is None:
        return ""
    h = "-" if r.get("height_contrast") is None else f"{r['height_contrast']:.2f}"
    c = "-" if r.get("closure_real") is None else f"{round(r['closure_real'] * r['closure_real_n'])}/{r['closure_real_n']}"
    k = "" if r.get("closure_corner") is None else f", corner {r['closure_corner']:.2f}"
    return f"{h}, shut {c}, w {r['width_range']:.2f}{k}"


def _path_cell(r):
    if r.get("path") is None:
        return ""
    return r["path"] if r.get("lag_frames") is None else f"{r['path']}, lag {r['lag_frames']}"


def write_verify(meta, rows, out):
    order = sorted(
        rows,
        key=lambda r: ({"FAIL": 0, "REVIEW": 1, "PASS": 2, "N/A": 3, "n/a": 3}[r["band"]], r["score"] if r["score"] is not None else 0),
    )
    json.dump(
        {**meta, "shots": order}, open(out + ".json", "w"), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()
    )
    cols = [
        "shot",
        "start",
        "end",
        "verdict",
        "target",
        "class",
        "band",
        "score",
        "delta",
        "corr",
        "env_corr",
        "env_best",
        "env_lag_ms",
        "mouth_range",
        "open_in_silence",
        "mouth_visible",
        "owner_px",
        "identity_cos",
        "mouth_identity_cos",
        "vocal_frac",
        "width_range",
        "closure_real",
        "closure_real_n",
        "closure_corner",
        "height_contrast",
        "nose_off",
        "nose_step",
        "nose_ok",
        "path",
        "lag_frames",
        "vocal_lost",
        "vocal_off_ms",
        "vocal_off_n",
        "vocal_words",
        "vocal_worst",
        "reason",
    ]
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in order:
            w.writerow([r.get(k) for k in cols])
    L = [
        f"# LSDub QA verify: {os.path.basename(meta['video'])}",
        "",
        f"Likely-success score 0-100 per dubbed shot: PASS >= {PASS_BAND:.0f}, REVIEW >= {REVIEW_BAND:.0f}, else FAIL. "
        f"COVERAGE (no readable target) = REVIEW, n/a; no voice = n/a; N/A = left as source. Identity drift = DINOv2 cosine vs --before < {ID_DRIFT}.",
        "`sync corr` reads the mouth against the opening built from the alignment, the curve the dub itself was painted from: a high "
        "number says the render followed its plan, and cannot see a wrong alignment. `audio` reads it against the vocal's "
        "loudness with no alignment, best lag in ms (+ = the mouth trails): reported, it moves no band.",
        "`shape` is reported and moves no band: height contrast (1 = the delivered heights follow the card's, 0 = every class the same "
        "height), then p / b / m phones with the lips really shut (gap under a share of the eye distance), then the width travel, then "
        "the darkest reading at the lips' corners on closures (no calibrated threshold: compare to the take's).",
        "`vocal` is reported and moves no band, and takes no time or length from the plan or the aligner, so a retimed plan is not marked against itself; "
        "a lead sheet's passed spans stay closed to it: the lyric alone walked over the vocal, with the plan's word starts against that walk (median offset in ms, words past 100 ms of those in "
        "the shot, the worst word; a free walk, so a word it loses reads as off, and `lost` when a shot's words sit more than 0.6 s off, where the walk has run away on a long song: no reading), then the plan's phones read directly on the vocal's frames: "
        "hiss on hiss, closures on a dip or nasal, vowels on open sound, 0..1. Passed-over words (a lead sheet) are not read.",
        "`path` is reported and moves no band: the paint delivered, then the frames of it picked from before their own time, whose skin is a frame old.",
        "",
        "| shot | t (s) | band | score | Δ vs before | class | path | sync corr | audio | shape | vocal | travel | silence | mouth | identity | reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in order:
        L.append(
            f"| {r['shot']} | {r['start']:.2f}-{r['end']:.2f} | **{r['band']}** | {'-' if r['score'] is None else (format(r['score'], '.0f'))} | "
            f"{'' if r.get('delta') is None else (format(r['delta'], '+.0f'))} | {r['class']} | {_path_cell(r)} | {'' if r.get('corr') is None else (format(r['corr'], '+.2f'))} | "
            f"{_audio_cell(r)} | {_shape_cell(r)} | {_vocal_cell(r)} | {r.get('mouth_range', '')} | {r.get('open_in_silence') if r.get('open_in_silence') is not None else ''} | "
            f"{r['mouth_visible']:.0%} | {'' if r.get('identity_cos') is None else r['identity_cos']} | {r['reason']} |"
        )
    if meta.get("closures_lost"):
        L += [
            "",
            "**Closures the mouth cannot show** (past the clip's end, or left no time by a retime): "
            + ", ".join(f"{w} {p} at {t:.2f} s" for w, p, t in meta["closures_lost"]),
        ]
    open(out + ".md", "w").write("\n".join(L) + "\n")
    sheet(
        meta["video"],
        rows,
        out + "_sheet.jpg",
        lambda r: {"PASS": (60, 200, 60), "REVIEW": (0, 170, 255), "FAIL": (60, 60, 210), "N/A": (150, 150, 150), "n/a": (150, 150, 150)}[
            r["band"]
        ],
        lambda r: f"{r['band']} {'' if r['score'] is None else int(r['score'])}",
    )


def _vocal_cell(r):
    hits = " ".join(f"{k} {r[f'vocal_{k}']:.2f}" for k in ("hiss", "shut", "open") if r.get(f"vocal_{k}") is not None)
    if r.get("vocal_lost"):
        return f"lost; {hits}" if hits else "lost"
    if r.get("vocal_off_ms") is None:
        return ""
    return f"{r['vocal_off_ms']} ms, {r['vocal_off_n']}/{r['vocal_words']}, {r['vocal_worst']}" + (f"; {hits}" if hits else "")


def sheet(video, shots, path, colour, label, cols=8, tw=256):
    F = read_frames(video, [(s["f0"] + s["f1"]) // 2 for s in shots])
    tiles = []
    for s in shots:
        img = F.get((s["f0"] + s["f1"]) // 2)
        if img is None:
            continue
        t = cv2.resize(img, (tw, int(tw * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        t = cv2.copyMakeBorder(t, 6, 30, 6, 6, cv2.BORDER_CONSTANT, value=colour(s))
        cv2.putText(t, f"#{s['shot']} {s['start']:.1f}s", (8, t.shape[0] - 18), 0, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(t, label(s), (8, t.shape[0] - 4), 0, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return
    h = max(x.shape[0] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="lsdub qa")
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("triage", help="which shots are fit to redub")
    t.add_argument("--preset", help="cast preset (presets.py): family names for near-identical extras")
    t.add_argument("--device", default="cpu", help="identity embedder device (cpu = the ONNX model; cuda / cuda:N = torch)")
    t.add_argument("--video", required=True)
    t.add_argument("--vocal", help="the vocal / dialogue stem: voice-activity per shot")
    t.add_argument("--offset", type=float, default=0.0, help="video time = audio time + offset (s)")
    t.add_argument("--cast-db", help="a `lsdub cast enroll` npz")
    t.add_argument("--refs", help="dir of <name>/ reference images")
    t.add_argument("--face", action="append", default=[], help="name=reference.png (repeatable)")
    t.add_argument("--target", action="append", default=[], help="who to dub (default: every reference identity)")
    t.add_argument("--samples", type=int, default=6)
    t.add_argument("--model", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--truth-cuts", help="a cut.json to score the auto-detected boundaries against (+-2 frames)")
    v = sp.add_parser("verify", help="how likely each dubbed shot is to have succeeded")
    v.add_argument("--video", required=True)
    v.add_argument("--vocal", required=True)
    v.add_argument("--shots", required=True)
    v.add_argument("--before")
    v.add_argument("--align")
    v.add_argument("--model", default=None)
    v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.cmd == "triage":
        from ..core import cast as C

        db = build_db(a.cast_db, a.refs, a.face, a.model)
        meta, shots = triage(
            a.video,
            a.vocal,
            a.offset,
            db,
            a.target,
            a.samples,
            a.model,
            preset=a.preset,
            emb=C.make_embedder(a.model, a.device) if a.device != "cpu" else None,
        )
        if a.truth_cuts:
            tc = json.load(open(a.truth_cuts))
            tr = [int(round(c["start"] * meta["fps"])) for c in tc["cut"]][1:]
            share, missed, fp = match_cuts([s["f0"] for s in shots][1:], tr)
            meta["cut_match"] = {
                "within_2_frames": round(share, 4),
                "truth": len(tr),
                "detected": len(shots) - 1,
                "missed": missed,
                "extra": fp,
            }
            print(f"cuts vs {a.truth_cuts}: {share:.1%} of {len(tr)} within +-2 frames; missed {missed}; extra {fp}")
        write_triage(meta, shots, a.out)
        print(
            f"qa triage {a.video}: {len(shots)} shots; "
            + ", ".join(f"{k} {sum(s['verdict'] == k for s in shots)}" for k in VERDICTS)
            + f" -> {a.out}.json"
        )
        return 0
    meta, rows = verify(a.video, a.vocal, a.shots, a.before, a.align, a.model, a.out + "_work")
    write_verify(meta, rows, a.out)
    print(
        f"qa verify {a.video}: "
        + ", ".join(f"{k} {sum(r['band'] == k for r in rows)}" for k in ("PASS", "REVIEW", "FAIL", "N/A", "n/a"))
        + f" -> {a.out}.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
