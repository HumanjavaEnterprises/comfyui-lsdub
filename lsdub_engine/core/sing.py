#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sing.py: `lsdub sing`, the ONE standard path for every sung cut (the per-shot face / lip deep dive is a standard
step, for EVERY sung shot). CPU.

  1 singer    `lsdub cast` (YuNet + MediaPipe tracks, DINOv2 + colour identity) on the take -> the --char track's box
  2 crop      smoothed track, one fixed crop per shot, Lanczos up so the face is --face-target (512) px. Close-ups too:
              they were warped at their native 250-300 px.
  3 deep dive the warp builds its per-shot landmarks, mouth kit and teeth atlas from the take it is given, so on the
              crop they are built at 512, not native. A mic near the lips (chrome in the lower face of the crop) writes
              the occluder command for the GPU agent (`lsdub occluders`, Florence-2 + SAM 2) to work/gpu_occluders.sh;
              an existing work/occ.npz is used (`--occluder-mask`).
  4 warp      SNAP: the viseme track and the crop at --fps-mult (2) x the frame rate, with between-word closures
              (--word-close), a shorter vocal attack (sung_dyn --attack-ms) and less smoothing in frame units; then
              one internal frame per output frame, keeping the extremes: at a closure the most-closed, at a peak the
              most-open, else the on-time frame (pick_extremes)
  5 composite the tier-2 paste (lsdub medium): area down, Lab match, grain, feathered nose-to-chin mask, the rest of the
              frame bit-exact; then the MEASURED gate: rendered-mouth corr vs the audio opening >= 0.5 and
              open-in-silence <= 0.35, plus snap numbers (rise_ms, closure_hit) -> --json

  lsdub sing --take take.mp4 --align align.json --vocal lead.wav --t0 64.3 --out sung.mp4 --work dir
             [--char singer_a --cast-db cast_db.npz | --box x0,y0,x1,y1] [--face-target 512] [--fps-mult 2]
             [--no-snap] [--a2f-csv a2f.csv] [--json gate.json] -- <lsdub paint flags (the face's registry flags, --lib)>
"""

from ..io import ff
import argparse, csv, json, os, sys, tempfile, time
from . import paint_api, steps
from .cpu_env import thread_report
import cv2, numpy as np

from . import medium as MED

GATE_CORR, GATE_SILENCE = 0.5, 0.35
GATE_KEYS = ("corr", "corr_all", "best", "lag_ms", "corr_pm1", "open_in_silence", "rise_ms", "closure_hit", "pass", "vocal_frac")
SNAP = dict(
    lead=0.06, word_close=0.06, attack_ms=15.0, paint=["--warp-sigma", "0.8", "--warp-temporal", "0.6", "--warp-close-depth", "0.05"]
)


def snap_paint(paint_args, m):
    """SNAP's paint flags. A --warp-temporal in the caller's flags, in output frames, holds here at the internal rate (x m); it
    never goes under SNAP's own. --warp-temporal-width, in output frames too, the same, and --warp-temporal-spare eases over one output
    frame. Named by nobody: SNAP's flags unchanged."""
    p = list(SNAP["paint"])
    t = paint_api.temporal(paint_args)
    if t is not None:
        i = p.index("--warp-temporal") + 1
        p[i] = str(max(float(p[i]), float(t) * m))
    w = paint_api.temporal(paint_args, "--warp-temporal-width")
    if w is not None:
        p += ["--warp-temporal-width", str(float(w) * m)]
    if any(str(x) == "--warp-temporal-spare" for x in paint_args):
        p += ["--warp-temporal-spare-ease", str(float(m))]
    return p


# ---------------------------------------------------------------------------------------------------- pick
def pick_extremes(egap_hi, target, m):
    """egap_hi: the warp's planned opening per INTERNAL frame (n * m); target: the audio opening per OUTPUT frame (n).
    -> internal index per output frame: around k*m (+-m//2), the most-closed where the audio is at a local minimum,
    the most-open at a local maximum, else k*m. Keeps the hard closures and peaks a plain decimation smears."""
    e = np.asarray(egap_hi, float)
    t = np.asarray(target, float)
    n = len(t)
    out = []
    for k in range(n):
        c = [j for j in range(k * m - m // 2, k * m + m // 2 + 1) if 0 <= j < len(e)] or [min(len(e) - 1, k * m)]
        a, b = t[max(0, k - 1)], t[min(n - 1, k + 1)]
        lo = t[k] <= a and t[k] < b  # the bottom of a closure (where it starts to rise)
        hi = t[k] >= a and t[k] > b  # the top of an opening (where it starts to fall)
        if lo and not hi:
            out.append(min(c, key=lambda j: e[j]))
        elif hi and not lo:
            out.append(max(c, key=lambda j: e[j]))
        else:
            out.append(min(len(e) - 1, k * m))
    return out


def dyn_target(dyn_csv, n, t0, fps, dt=0.0):
    R = list(csv.DictReader(open(dyn_csv)))
    return np.interp(np.arange(n) / fps + dt, np.array([float(r["time_s"]) for r in R]) - t0, np.array([float(r["geo_gap_cm"]) for r in R]))


def plan_t0(t0, paint_args, rate):
    """where --coart-advance-ms puts the plan, on the track's whole frames as the paint moves it."""
    from .coart import whole_frames

    return t0 + whole_frames(paint_api.advance_s(paint_args), rate)


def moved_target(dyn_csv, n, t0, fps, warp_json, m=1):
    """dyn_target as the paint behind warp_json left it: --coart-sync-vocal logs each frame's offset beside its --warp-json, at
    the track's rate (m x fps), and the gate reads the same moved curve. No log: dyn_target."""
    log = os.path.splitext(warp_json)[0] + "_coart.json"
    dt = (json.load(open(log)).get("sync_dt") if os.path.exists(log) else None) or None
    return dyn_target(dyn_csv, n, t0, fps, 0.0 if dt is None else np.resize(np.asarray(dt, float)[::m], n))


# ---------------------------------------------------------------------------------------------------- singer
def singer_box(take, char, db_path, model=None, preset=None):
    """`lsdub cast` on the take: the --char track with the largest median face -> its median det box (x0,y0,x1,y1)."""
    from . import cast as C

    r = C.scan_video(take, C.load_db(db_path), model or C.MODEL, preset=preset)
    T = [t for t in r["tracks"] if t["character"] == char]
    if not T:
        raise RuntimeError(f"sing: no '{char}' track in {take} (found {sorted({t['character'] for t in r['tracks']})})")
    t = max(T, key=lambda t: t["face_px_med"] * t["n"])
    return [float(np.median([f["box"][j] for f in t["frames"]])) for j in range(4)], t


def mic_near_lips(crop, face_px, thr=0.08):
    """chrome / grey share in the crop's lower face (below the mouth to the chin): a mic the warp can drag."""
    O = crop.shape[0]
    c = O / 2
    z = crop[int(c + 0.15 * face_px) : int(c + 0.65 * face_px), int(c - 0.3 * face_px) : int(c + 0.3 * face_px)]
    if z.size == 0:
        return 0.0, False
    hsv = cv2.cvtColor(z, cv2.COLOR_BGR2HSV)
    s = float(((hsv[..., 1] < 50) & (hsv[..., 2] > 70)).mean())
    return round(s, 3), s > thr


# ---------------------------------------------------------------------------------------------------- run
def run(
    take,
    align,
    vocal,
    t0,
    out,
    work,
    paint_args=(),
    char=None,
    db=None,
    box=None,
    face_target=512.0,
    fps_mult=2,
    snap=True,
    a2f=None,
    report=None,
    lead=None,
    word_close=None,
    auto=False,
    preset=None,
    lossless=False,
):
    os.makedirs(work, exist_ok=True)
    if not a2f:
        wt = paint_api.target(paint_args)
        by = (
            "the coart plan's sung_dyn curve"
            if wt == "coart"
            else "the take's own mouth"
            if wt == "own"
            else "the viseme track's audio opening"
        )
        print(f"sing: no --a2f-csv: {by} drives the jaw; a jawOpen csv gives better silences", file=sys.stderr)
    lead = (SNAP["lead"] if snap else 0.1) if lead is None else lead
    word_close = (SNAP["word_close"] if snap else 0.0) if word_close is None else word_close
    threads = thread_report()
    print(threads, file=sys.stderr)
    F, fps = MED.read(take)
    n = len(F)
    m = max(1, int(fps_mult)) if snap else 1
    info = {"take": take, "t0": t0, "frames": n, "fps": fps, "face_target": face_target, "fps_mult": m, "snap": snap, "threads": threads}
    tm, t_ = {}, time.perf_counter()  # seconds per stage, into `info`; nothing reads them back
    if char and db:
        box, t = singer_box(take, char, db, preset=preset)
        info["cast"] = {"char": char, "confidence": t["confidence"], "face_px_med": t["face_px_med"], "box": box}
    tm["track"], t_ = round(time.perf_counter() - t_, 3), time.perf_counter()
    # 4a. tracks: the audio opening (sung_dyn) and the viseme track at the internal rate
    dyn = os.path.join(work, "dyn.csv")
    trk = os.path.join(work, "track.json")
    sd = ["--align", align, "--audio", vocal, "--t0", str(t0), "--dur", str(n / fps + 0.5), "--out", dyn]
    vs = [
        "--align",
        align,
        "--t0",
        str(t0),
        "--frames",
        str(n * m),
        "--fps",
        str(fps * m),
        "--audio",
        vocal,
        "--lead",
        str(lead),
        "--out",
        trk,
    ]
    if a2f:
        sd += ["--a2f-csv", a2f]
        vs += ["--a2f-csv", a2f]
    if snap:
        sd += ["--attack-ms", str(SNAP["attack_ms"]), "--lead", str(lead)]
    sd += paint_api.dyn_flags(paint_args)
    if word_close:
        vs += ["--word-close", str(word_close)]
    steps.call("sung_dyn", sd, quiet=True)
    steps.call("visemes", vs, quiet=True)
    tm["plan"] = round(time.perf_counter() - t_, 3)
    # 3. occluders: a mic near the lips -> the GPU command (written, not run); an existing mask is used
    occ = os.path.join(work, "occ.npz")
    pa = paint_api.flags(
        user_overrides=paint_args,
        operational=[
            *paint_api.coart_dyn(paint_args, dyn),
            *(snap_paint(paint_args, m) if snap else ()),
            *(("--occluder-mask", occ) if os.path.exists(occ) else ()),
        ],
    )
    t_plan = plan_t0(t0, paint_args, fps * m)  # the plan moved: the gate reads the same curve
    pick = (
        (lambda wj, n_, m_: pick_extremes([f["egap"] for f in json.load(open(wj))["frames"]], moved_target(dyn, n, t_plan, fps, wj, m), m_))
        if m > 1
        else None
    )
    R = MED.run(
        take,
        trk,
        out,
        pa,
        box,
        face_target,
        work=work,
        fps_mult=m,
        pick=pick,
        zoom_note=False,
        report=os.path.join(work, "medium.json"),
        **({"lossless": True} if lossless else {}),
    )
    tm.update(R.get("timings", {}))
    crop0 = MED.read(os.path.join(work, "crop.mp4"), 1)[0][0]
    share, near = mic_near_lips(crop0, face_target)
    info["mic_share"] = share
    if near and not os.path.exists(occ):
        crop = os.path.join(work, "crop.mp4")
        open(os.path.join(work, "gpu_occluders.sh"), "w").write(
            "#!/usr/bin/env bash\n# lsdub sing: a mic sits near the lips on this crop. Occluder masks need a GPU (Florence-2 + SAM 2).\n"
            "# Run this on a GPU, then re-run the same `sing` (it picks up occ.npz).\n"
            f"python -m lsdub_engine.cli occluders --video {crop} --out {occ} --phrase microphone --review {occ[:-4]}_review.mp4\n"
        )
        info["gpu_todo"] = os.path.join(work, "gpu_occluders.sh")
    # 5. the measured gate on the delivered pixels
    t_ = time.perf_counter()
    sing_json = os.path.join(work, "crop_painted_warp.json")
    target = moved_target(dyn, n, t_plan, fps, sing_json, m)
    info.update(gate(out, R, target, vocal, align, t0, fps, work=work, warp_json=sing_json), warp_json=sing_json)
    tm["gate"], t_ = round(time.perf_counter() - t_, 3), time.perf_counter()
    if auto:
        info.update(auto_select(take, align, vocal, t0, out, work, paint_args, dyn, R, target, fps, n, a2f, info, t_plan=t_plan))
        tm["native"] = round(time.perf_counter() - t_, 3)
    info["timings"] = tm
    info["medium"] = {k: R[k] for k in ("face_h_px", "zoom", "crop_px", "outside_max_preencode", "not_pasted", "picked")}
    if report:
        json.dump(info, open(report, "w"), indent=1)
    g = info
    print(
        f"sing {out}: face {R['face_h_px']:.0f}px x{R['zoom']:.2f} @{m}x fps; MEASURED corr {g['corr']} (best {g['best']} at "
        f"{g['lag_ms']} ms), open-in-silence {g['open_in_silence']}, rise {g['rise_ms']} ms, closures {g['closure_hit']} "
        f"-> {'PASS' if g['pass'] else 'FAIL'}"
        + (
            f"; auto -> {g['auto']['winner']} (sing {g['auto']['sing']['corr']}, native {g['auto']['native']['corr']})"
            if g.get("auto")
            else ""
        )
        + (f"; GPU todo {g['gpu_todo']}" if g.get("gpu_todo") else "")
    )
    return info


def pick_paint(s, n):
    """the rate that wins, by their gates: one that passes beats one that fails, else the better sync, within 0.02 the quieter silences. Unmeasured silence is the worst."""
    key = lambda g: (g["corr"] if g["corr"] is not None else -9, -(g["open_in_silence"] if g.get("open_in_silence") is not None else 9))
    if bool(s.get("pass")) != bool(n.get("pass")):
        return "native" if n.get("pass") else "sing"
    s_c, n_c = key(s), key(n)
    return "native" if (n_c[0] > s_c[0] + 0.02) or (abs(n_c[0] - s_c[0]) <= 0.02 and n_c[1] > s_c[1]) else "sing"


def auto_select(take, align, vocal, t0, out, work, paint_args, dyn, R, target, fps, n, a2f, sing_info, t_plan=None):
    """Per-shot auto-selection: also run the native warp (lsdub paint on the full take at the frame rate, the pre-sing
    recipe) and keep whichever scores the higher MEASURED corr (lag 0); open-in-silence breaks a tie (within 0.02).
    Both are kept: <out>_sing.mp4 and <out>_native.mp4; <out> is the winner."""
    import shutil

    trk1 = os.path.join(work, "track_1x.json")
    nat = out[:-4] + "_native.mp4"
    sng = out[:-4] + "_sing.mp4"
    vs = [
        "--align",
        align,
        "--t0",
        str(t0),
        "--frames",
        str(n),
        "--fps",
        str(fps),
        "--audio",
        vocal,
        "--lead",
        "0.1",
        "--out",
        trk1,
    ] + (["--a2f-csv", a2f] if a2f else [])
    steps.call("visemes", vs, quiet=True)
    paint_api.paint(
        take,
        trk1,
        n,
        user_overrides=paint_args,
        operational=paint_api.coart_dyn(paint_args, dyn),
        out=nat,
        warp_json=os.path.join(work, "native_warp.json"),
    )
    shutil.copyfile(out, sng)
    nat_json = os.path.join(work, "native_warp.json")
    target = target if t_plan is None else moved_target(dyn, n, t_plan, fps, nat_json)  # its own paint, its own offsets
    g_n = gate(nat, R, target, vocal, align, t0, fps, work=os.path.join(work, "native_gate"), warp_json=nat_json)

    win = pick_paint(sing_info, g_n)
    if win == "native":
        shutil.copyfile(nat, out)
    return {
        "auto": {
            "winner": win,
            "sing": {k: sing_info.get(k) for k in ("corr", "best", "open_in_silence", "rise_ms", "closure_hit", "pass")},
            "native": {k: g_n.get(k) for k in ("corr", "best", "open_in_silence", "rise_ms", "closure_hit", "pass")},
        },
        **(
            {
                **{k: g_n[k] for k in GATE_KEYS if k in g_n},
                "warp_json": nat_json,
            }
            if win == "native"
            else {}
        ),
    }


def gate(video, R, target, vocal, align, t0, fps, roi_pad=1.0, work=None, warp_json=None):
    work = work or tempfile.mkdtemp(prefix="lsdub_gate_")
    """measured corr (sync-qa's measured_opening on a face box around the composite) + snap; pass = corr >= GATE_CORR
    (best within +-1 frame, as the round-4 gate) and open-in-silence <= GATE_SILENCE. warp_json: the frames a vocalisation span
    took out of the audio's hands (--vocalisations skip | rest) are left out of the corr; vocal_frac reports their share."""
    from ..audits import sync_qa as Q, snap as SN
    from . import vocalisations as VOC

    os.makedirs(work, exist_ok=True)
    B, _ = MED.read(video, len(target))
    cx = np.median([f["centre"][0] for f in R["per_frame"]])
    cy = np.median([f["centre"][1] for f in R["per_frame"]])
    s = R["face_h_px"] * 1.6 * roi_pad
    H, W = B[0].shape[:2]
    roi = [int(max(0, cy - s / 2)), int(min(H, cy + s / 2)), int(max(0, cx - s / 2)), int(min(W, cx + s / 2))]
    op, cover = Q.measured_opening(B, roi, fps)
    if op is None:
        return {
            "corr": None,
            "best": None,
            "lag_ms": None,
            "corr_pm1": None,
            "cover": cover,
            "open_in_silence": None,
            "rise_ms": None,
            "closure_hit": None,
            "pass": False,
            "vocal_frac": 0.0,
        }  # same keys as a measured gate (run() prints them)
    keep, vocal_frac = VOC.scored_frames(warp_json, len(op))
    c0, cb, lag = Q.tracking(op, target, fps, keep=keep)
    c1 = max(Q.tracking(op, target, fps, max_lag=1, keep=keep)[1], c0)
    extra = {"vocal_frac": vocal_frac} | ({} if keep is None else {"corr_all": round(Q.tracking(op, target, fps)[0], 3)})
    from .coart import logged_offsets, logged_words

    A = json.load(open(align))
    words = logged_words(A["words"], warp_json)
    sn = SN.snap_metrics(
        op,
        vocal_clip(vocal, t0, len(target) / fps, work),
        words,
        t0,
        fps,
        dt=logged_offsets(warp_json, len(op), fps),
        keep=keep if A.get("pass_over") else None,
    )
    np.save(os.path.join(work, "measured_open.npy"), op)
    ok = c1 >= GATE_CORR and (sn["open_in_silence"] is None or sn["open_in_silence"] <= GATE_SILENCE)
    return dict(
        corr=round(c0, 3), best=round(cb, 3), lag_ms=lag, corr_pm1=round(c1, 3), cover=round(cover, 3), **sn, **extra, **{"pass": bool(ok)}
    )


def vocal_clip(vocal, t0, dur, work):
    p = os.path.join(work, "vox.wav")
    ff.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", vocal, p], check=True)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(prog="lsdub sing")
    ap.add_argument("--take", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--vocal", required=True)
    ap.add_argument("--t0", type=float, required=True, help="song time of the take's first frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--preset", help="cast preset (presets.py), e.g. landing")
    ap.add_argument("--char")
    ap.add_argument("--cast-db")
    ap.add_argument("--box")
    ap.add_argument("--face-target", type=float, default=512.0)
    ap.add_argument("--fps-mult", type=int, default=2)
    ap.add_argument("--no-snap", dest="snap", action="store_false")
    ap.add_argument("--a2f-csv", help="jaw-curve csv (jawOpen): part of the standard recipe")
    ap.add_argument("--json")
    ap.add_argument("--lossless-intermediates", action="store_true", help="crop and painted crop lossless; changes the output's pixels")
    ap.add_argument(
        "--lead",
        type=float,
        help="viseme / dyn lead, s (default 0.06 with snap (lead sweep on 21: 0.20 -> 0.33, 0.14 -> 0.47, 0.06 -> 0.59, 0.03 -> 0.67, 0.0 -> 0.69; 19 prefers 0.14), 0.10 without)",
    )
    ap.add_argument("--word-close", type=float, help="between-word closure, s (default 0.06 with snap)")
    ap.add_argument(
        "--auto",
        action="store_true",
        help="also run the native warp (full frame, frame rate, --lead 0.1, no word-close: the pre-sing recipe) and keep "
        "the higher measured corr; open-in-silence breaks a tie within 0.02 (an unmeasured silence counts as the worst)",
    )
    a = ap.parse_args(argv)
    info = run(
        a.take,
        a.align,
        a.vocal,
        a.t0,
        a.out,
        a.work,
        extra,
        a.char,
        a.cast_db,
        [float(v) for v in a.box.split(",")] if a.box else None,
        a.face_target,
        a.fps_mult,
        a.snap,
        a.a2f_csv,
        a.json,
        lead=a.lead,
        word_close=a.word_close,
        auto=a.auto,
        preset=a.preset,
        lossless=a.lossless_intermediates,
    )
    return 0 if info["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
