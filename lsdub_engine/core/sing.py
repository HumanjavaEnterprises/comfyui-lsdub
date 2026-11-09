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
  4 plan      the alignment's words and phones re-read off the vocal (vocal_retime, anchored to the aligner's times) ->
              work/align_vocal.json, which every step below plans from (--no-vocal-retime: the alignment as it is)
    warp      SNAP: the viseme track and the crop at --fps-mult (2) x the frame rate, with between-word closures
              (--word-close), a shorter vocal attack (sung_dyn --attack-ms), less smoothing in frame units and closing
              edges 20 ms early (--warp-shape asym); then
              one internal frame per output frame, keeping the extremes: at a closure the most-closed, at a peak the
              most-open, else the on-time frame (pick_extremes)
  5 composite the tier-2 paste (lsdub medium): area down, Lab match, grain, feathered nose-to-chin mask, the rest of the
              frame bit-exact; then the MEASURED gate on the delivered mouth: against the VOCAL itself (sync_truth: audio
              sync >= 0.30 at lag 0, the mouth no more than ~1 frame late) and open-in-silence <= 0.35; the corr against the
              alignment-built opening (>= 0.5 before) is reported as a diagnostic, and decides only when the vocal has nothing
              to read; plus snap numbers (rise_ms, closure_hit) -> --json

  lsdub sing --take take.mp4 --align align.json --vocal lead.wav --t0 64.3 --out sung.mp4 --work dir
             [--char singer_a --cast-db cast_db.npz | --box x0,y0,x1,y1] [--face-target 512] [--fps-mult 2]
             [--no-snap] [--a2f-csv a2f.csv] [--json gate.json] -- <lsdub paint flags (the face's registry flags, --lib)>
"""

from ..io import ff
import argparse, csv, json, os, sys, tempfile, time
from . import paint_api, spaces as SP, steps
from .cpu_env import thread_report
import cv2, numpy as np

from . import medium as MED

GATE_CORR, GATE_SILENCE = 0.5, 0.35
GATE_KEYS = (
    "corr", "corr_all", "best", "lag_ms", "corr_pm1", "open_in_silence", "rise_ms", "closure_hit", "pass", "vocal_frac",
    "truth", "truth_lag_ms", "truth_best", "truth_level", "truth_events", "truth_pass", "truth_kept", "truth_kept_lag_ms",
)  # fmt: skip
AUTO_KEYS = ("truth", "truth_lag_ms", "corr", "best", "open_in_silence", "rise_ms", "closure_hit", "pass")
SNAP = dict(
    lead=0.06,
    word_close=0.06,
    attack_ms=15.0,
    paint=["--warp-sigma", "0.8", "--warp-temporal", "0.6", "--warp-close-depth", "0.05", "--warp-follow", "0.15", "--warp-shape", "asym"],
)  # --warp-follow: the plain target holds the opening near the plan, so a take's own mouth timing does not ride through.
# --warp-shape asym: closing edges 20 ms early, openings on time (core/drive_shape.py); a --warp-shape in the caller's flags wins


def snap_paint(paint_args, m):
    """SNAP's paint flags. A --warp-temporal in the caller's flags, in output frames, holds here at the internal rate (x m); it
    never goes under SNAP's own. --warp-temporal-width, in output frames too, the same, and --warp-temporal-spare eases over one output
    frame. A --warp-shape in the caller's flags replaces SNAP's (--warp-shape off: the drive as it was before asym). Named by nobody:
    SNAP's flags unchanged."""
    p = list(SNAP["paint"])
    if any(str(x) == "--warp-shape" or str(x).startswith("--warp-shape=") for x in paint_args):
        i = p.index("--warp-shape")
        del p[i : i + 2]
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
def has_flag(flags, name):
    """name is among flags, as `--x v` or `--x=v`."""
    return any(str(f) == name or str(f).startswith(name + "=") for f in flags)


def flag_value(flags, name):
    """the value after name in flags (`--x v` or `--x=v`); None when absent."""
    fl = [str(f) for f in flags]
    for i, f in enumerate(fl):
        if f == name and i + 1 < len(fl):
            return fl[i + 1]
        if f.startswith(name + "="):
            return f.split("=", 1)[1]
    return None


def track_target(warp_json, n, m):
    """the paint's own planned opening (its per-frame target gap, after every drive stage) as the pick's target per OUTPUT frame: the
    mean of the internal frames of each output frame. The hold stage's apex and closures are then picked where the plan put them,
    not where a low-passed envelope peaks."""
    tg = np.array([float(f.get("tgap", f.get("egap", 0.0))) for f in json.load(open(warp_json))["frames"]], float)
    if not len(tg):
        return np.zeros(n)
    out = np.array([tg[k * m : (k + 1) * m].mean() if k * m < len(tg) else tg[-1] for k in range(n)])
    return out


def warp_audio(paint_args, snap_flags, vocal):
    """the stem the event stage reads (--warp-audio), set when --warp-events is among the paint's flags (the caller's or SNAP's) and
    the caller did not name one; () otherwise, so a paint without the stage gets the argv it always did."""
    flags = [str(x) for x in (*paint_args, *snap_flags)]
    if "--warp-events" not in flags and not any(f.startswith("--warp-events=") for f in flags):
        return ()
    if "--warp-audio" in flags or any(f.startswith("--warp-audio=") for f in flags):
        return ()
    return ("--warp-audio", str(vocal))


def pick_extremes(egap_hi, target, m):
    """egap_hi: the warp's planned opening per INTERNAL frame (n * m); target: the audio opening per OUTPUT frame (n).
    -> internal index per output frame: from k*m to k*m+m//2, the most-closed where the audio is at a local minimum,
    the most-open at a local maximum, else k*m. Keeps the hard closures and peaks a plain decimation smears. Never below
    k*m: the internal frames there hold the source frame before, and pasting one puts old skin and an old nose in this frame."""
    e = np.asarray(egap_hi, float)
    t = np.asarray(target, float)
    n = len(t)
    out = []
    for k in range(n):
        c = [j for j in range(k * m, k * m + m // 2 + 1) if 0 <= j < len(e)] or [min(len(e) - 1, k * m)]
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


RETIME_CAP = 0.25  # s: no word moves further than this off the aligner (and no ad-lib moves at all)
RETIME_MARGIN = 0.02  # the retimed plan's audio sync must beat the aligner's by this much: a smaller gain is not a reading


def retimed_align(align, vocal, work, t0=None, n=None, fps=25.0, lead=0.06, word_close=0.06):
    """the alignment with its words and phones re-read off the vocal (vocal_retime, the walk anchored to the aligner's times and held to
    +-RETIME_CAP; ad-libs keep their times, so the spans handed back to the take do not move), kept only when it is better on this
    shot: the viseme plan's opening (viseme_track at fps over the shot's n frames from t0) read against the vocal (sync_truth) must
    rise by RETIME_MARGIN. -> (the alignment's path, a note {used, plan_truth: [aligned, retimed]}); the retimed one is work/align_vocal.json. No
    alignment or vocal file to read: the alignment as given (the plan's own steps then say what is missing)."""
    from ..audits import sync_truth as ST
    from . import viseme_track as VT, vocal_retime as VR

    if not (os.path.exists(align) and os.path.exists(vocal)):
        return align, {"used": False, "why": "nothing to read"}
    A = json.load(open(align))
    if A.get("aligner") == "ctc":  # read off the vocal's own frames already; the capped walk and the envelope judge are not a better reader
        return align, {"used": False, "why": "ctc alignment"}
    x = VR.read_mono(vocal)
    R = VR.retime(A, vocal, x=x, cap=RETIME_CAP)
    note = {"used": False, "cap_s": RETIME_CAP}
    if t0 is not None and n:
        oc = VT.envelope_open(vocal, t0, n, fps)
        clip = x[int(round(max(0.0, t0) * VR.SR)) :]
        plan = lambda al: np.array(
            [f["open"] for f in VT.build(al, t0, n, fps, lead=lead, min_hold=3, open_curve=oc, word_close=word_close)]
        )
        t = [ST.read(plan(al), clip, fps)["truth"] for al in (A, R)]
        note["plan_truth"] = t
        if t[0] is not None and t[1] is not None and t[1] < t[0] + RETIME_MARGIN:
            return align, note  # the vocal's times do not make this shot's plan clearly better: the aligner's stand
    out = os.path.join(work, "align_vocal.json")
    json.dump(R, open(out, "w"), indent=0)
    note["used"] = True
    return out, note


def compressed_vocal(vocal, work):
    """the lead stem levelled for the drive (ffmpeg acompressor, then a peak normaliser) -> work/vocal_drive.wav."""
    out = os.path.join(work, "vocal_drive.wav")
    af = "acompressor=threshold=-24dB:ratio=4:attack=5:release=80:makeup=2,alimiter=limit=0.9"
    ff.run(["ffmpeg", "-v", "error", "-y", "-i", vocal, "-af", af, out], check=True)
    return out


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


def occluders_cmd(work):
    """the line that makes work/occ.npz on the crop sing paints (GPU)."""
    crop, occ = os.path.join(work, "crop.mp4"), os.path.join(work, "occ.npz")
    return f"python -m lsdub_engine.cli occluders --video {crop} --out {occ} --phrase microphone --review {occ[:-4]}_review.mp4"


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
    vocal_retime=True,
    pump=None,
    drive_comp=False,
    travel_ref="aligner",
):
    os.makedirs(work, exist_ok=True)
    again = dict(paint_args=paint_args, char=char, db=db, box=box, face_target=face_target, fps_mult=fps_mult, snap=snap, a2f=a2f,
                 lead=lead, word_close=word_close, auto=auto, preset=preset, lossless=lossless, pump=pump, drive_comp=drive_comp)  # fmt: skip
    align0 = align
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
    if vocal_retime:
        align, info["retime"] = retimed_align(align, vocal, work, t0, n, fps, lead, word_close)
    info["align"] = align
    tm, t_ = {}, time.perf_counter()  # seconds per stage, into `info`; nothing reads them back
    if char and db:
        box, t = singer_box(take, char, db, preset=preset)
        info["cast"] = {"char": char, "confidence": t["confidence"], "face_px_med": t["face_px_med"], "box": box}
    tm["track"], t_ = round(time.perf_counter() - t_, 3), time.perf_counter()
    # 4a. tracks: the audio opening (sung_dyn) and the viseme track at the internal rate
    dyn = os.path.join(work, "dyn.csv")
    trk = os.path.join(work, "track.json")
    drive = compressed_vocal(vocal, work) if drive_comp else vocal  # the stem the plan opens to; the gate always reads `vocal`
    sd = ["--align", align, "--audio", drive, "--t0", str(t0), "--dur", str(n / fps + 0.5), "--out", dyn]
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
        drive,
        "--lead",
        str(lead),
        "--out",
        trk,
    ]
    if pump:  # (gate, ratio): the drive's gate + compressor (sung_dyn.pump) on both curves
        pf = ["--pump-gate", str(pump[0]), "--pump-ratio", str(pump[1])]
        sd += pf
        vs += pf
    if a2f:
        sd += ["--a2f-csv", a2f]
        vs += ["--a2f-csv", a2f]
    if snap:
        sd += ["--attack-ms", str(SNAP["attack_ms"]), "--lead", str(lead)]
    sd += paint_api.dyn_flags(paint_args)
    if word_close:
        vs += ["--word-close", str(word_close)]
    hold = has_flag(paint_args, "--warp-hold") or has_flag(SNAP["paint"] if snap else (), "--warp-hold")
    if hold:  # the hold stage writes the opening: the track carries the class opening alone, no envelope under it
        vs += ["--open-curve", "off"]
    if word_close and (flag_value(paint_args, "--warp-closures") or flag_value(SNAP["paint"] if snap else (), "--warp-closures")) in (
        "aligned",
        "aligned-press",
    ):
        vs += ["--word-close-vocal"]  # the between-word press only where the vocal drops
    steps.call("sung_dyn", sd, quiet=True)
    steps.call("visemes", vs, quiet=True)
    tm["plan"] = round(time.perf_counter() - t_, 3)
    # 3. occluders: a mic near the lips -> the GPU command (written, not run); an existing mask is used. It must be made on the
    # crop this paints (n * m square frames); any other shape stops here, before a paint runs
    occ = os.path.join(work, "occ.npz")
    mask = SP.Occluder.open(occ, n * m, how=occluders_cmd(work)) if os.path.exists(occ) else None
    pa = paint_api.flags(
        user_overrides=paint_args,
        operational=[
            *paint_api.coart_dyn(paint_args, dyn),
            *(snap_paint(paint_args, m) if snap else ()),
            *(("--occluder-mask", occ) if mask else ()),
            *warp_audio(paint_args, snap_paint(paint_args, m) if snap else (), drive),
        ],
    )
    t_plan = plan_t0(t0, paint_args, fps * m)  # the plan moved: the gate reads the same curve
    pick = (
        (lambda wj, n_, m_: pick_extremes([f["egap"] for f in json.load(open(wj))["frames"]], track_target(wj, n, m) if hold else moved_target(dyn, n, t_plan, fps, wj, m), m_))
        if m > 1
        else None
    )  # fmt: skip
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
    native_occ = ()
    if mask:
        mask = mask.bind(SP.CropSpace.from_report(R, m))  # the side must be the crop's
        info["occluder"] = {"space": "crop", "frames": n * m}
        if auto:  # the native paint gets the same mask, mapped back to the take, with the grow in take px
            native_occ = SP.native_occluder(mask, n, F[0].shape[:2], paint_args, os.path.join(work, "occ_native.npz"))
            info["occluder"]["native"] = {"mask": native_occ[1], "grow": int(native_occ[3])}
    crop0 = MED.read(os.path.join(work, "crop.mp4"), 1)[0][0]
    share, near = mic_near_lips(crop0, face_target)
    info["mic_share"] = share
    if near and not os.path.exists(occ):
        open(os.path.join(work, "gpu_occluders.sh"), "w").write(
            "#!/usr/bin/env bash\n# lsdub sing: a mic sits near the lips on this crop. Occluder masks need a GPU (Florence-2 + SAM 2).\n"
            "# Run this on a GPU, then re-run the same `sing` (it picks up occ.npz).\n"
            f"{occluders_cmd(work)}\n"
        )
        info["gpu_todo"] = os.path.join(work, "gpu_occluders.sh")
    # 5. the measured gate on the delivered pixels
    t_ = time.perf_counter()
    sing_json = os.path.join(work, "crop_painted_warp.json")
    target = moved_target(dyn, n, t_plan, fps, sing_json, m)
    info.update(gate(out, R, target, vocal, align, t0, fps, work=work, warp_json=sing_json), warp_json=sing_json)
    tm["gate"], t_ = round(time.perf_counter() - t_, 3), time.perf_counter()
    if auto:
        info.update(
            auto_select(
                take, align, vocal, t0, out, work, paint_args, dyn, R, target, fps, n, a2f, info, t_plan=t_plan, native_occ=native_occ
            )
        )
        tm["native"] = round(time.perf_counter() - t_, 3)
    info["timings"] = tm
    info["medium"] = {k: R[k] for k in ("face_h_px", "zoom", "crop_px", "outside_max_preencode", "not_pasted", "picked")}
    if (info.get("retime") or {}).get("used"):
        info = travel_guard(info, take, align0, vocal, t0, out, work, R, again, ref=travel_ref)
    if report:
        json.dump(info, open(report, "w"), indent=1)
    g = info
    print(
        f"sing {out}: face {R['face_h_px']:.0f}px x{R['zoom']:.2f} @{m}x fps; MEASURED audio sync {g.get('truth')} (mouth "
        f"{g.get('truth_lag_ms')} ms vs the vocal), plan corr {g['corr']} (best {g['best']} at "
        f"{g['lag_ms']} ms), open-in-silence {g['open_in_silence']}, rise {g['rise_ms']} ms, closures {g['closure_hit']} "
        f"-> {'PASS' if g['pass'] else 'FAIL'}"
        + (
            f"; auto -> {g['auto']['winner']} (sing {g['auto']['sing']['truth']}, native {g['auto']['native']['truth']})"
            if g.get("auto")
            else ""
        )
        + (f"; GPU todo {g['gpu_todo']}" if g.get("gpu_todo") else "")
    )
    return info


def pick_paint(s, n):
    """the rate that wins, by their gates: one that passes beats one that fails, else the better sync, within 0.02 the quieter silences. Unmeasured silence is the worst.
    The sync is the audio sync (truth) when both gates read it, else the plan corr."""
    k = "truth" if s.get("truth") is not None and n.get("truth") is not None else "corr"
    key = lambda g: (g[k] if g[k] is not None else -9, -(g["open_in_silence"] if g.get("open_in_silence") is not None else 9))
    if bool(s.get("pass")) != bool(n.get("pass")):
        return "native" if n.get("pass") else "sing"
    s_c, n_c = key(s), key(n)
    return "native" if (n_c[0] > s_c[0] + 0.02) or (abs(n_c[0] - s_c[0]) <= 0.02 and n_c[1] > s_c[1]) else "sing"


def auto_select(take, align, vocal, t0, out, work, paint_args, dyn, R, target, fps, n, a2f, sing_info, t_plan=None, native_occ=()):
    """Per-shot auto-selection: also run the native warp (lsdub paint on the full take at the frame rate, the pre-sing
    recipe) and keep pick_paint's choice of the two.
    Both are kept: <out>_sing.mp4 and <out>_native.mp4; <out> is the winner. native_occ: the native paint's occluder flags
    (spaces.native_occluder), passed through as they are."""
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
        operational=[*paint_api.coart_dyn(paint_args, dyn), *native_occ],
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
            "sing": {k: sing_info.get(k) for k in AUTO_KEYS},
            "native": {k: g_n.get(k) for k in AUTO_KEYS},
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
    """the mouth measured on the delivered pixels (sync-qa's measured_opening on a face box around the composite), read against
    the vocal itself (sync_truth: truth, truth_lag_ms) and against the alignment-built opening (corr, best within +-1 frame as
    the round-4 gate: a diagnostic) + snap. pass = sync_truth passes and open-in-silence <= GATE_SILENCE; a vocal sync_truth
    cannot read falls back to corr >= GATE_CORR. warp_json: the frames a vocalisation span took out of the audio's hands
    (--vocalisations skip | rest) are left out of both readings; vocal_frac reports their share."""
    from ..audits import sync_qa as Q, snap as SN, sync_truth as ST
    from . import vocalisations as VOC

    os.makedirs(work, exist_ok=True)
    B, _ = MED.read(video, len(target))
    op, cover = Q.measured_opening(B, face_roi(R, B[0].shape, roi_pad), fps)
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
            **dict.fromkeys(ST.KEYS),
        }  # same keys as a measured gate (run() prints them)
    keep, vocal_frac = VOC.scored_frames(warp_json, len(op))
    c0, cb, lag = Q.tracking(op, target, fps, keep=keep)
    c1 = max(Q.tracking(op, target, fps, max_lag=1, keep=keep)[1], c0)
    extra = {"vocal_frac": vocal_frac} | ({} if keep is None else {"corr_all": round(Q.tracking(op, target, fps)[0], 3)})
    from .coart import logged_offsets, logged_words

    A = json.load(open(align))
    words = logged_words(A["words"], warp_json)
    vox = vocal_clip(vocal, t0, len(target) / fps, work)
    sn = SN.snap_metrics(
        op,
        vox,
        words,
        t0,
        fps,
        dt=logged_offsets(warp_json, len(op), fps),
        keep=keep if A.get("pass_over") else None,
    )
    np.save(os.path.join(work, "measured_open.npy"), op)
    tr = ST.read_wav(op, vox, fps, keep)
    synced = tr["truth_pass"] if tr["truth_pass"] is not None else c1 >= GATE_CORR
    ok = synced and (sn["open_in_silence"] is None or sn["open_in_silence"] <= GATE_SILENCE)
    return dict(
        corr=round(c0, 3),
        best=round(cb, 3),
        lag_ms=lag,
        corr_pm1=round(c1, 3),
        cover=round(cover, 3),
        **sn,
        **extra,
        **tr,
        **{"pass": bool(ok)},
    )


def face_roi(R, shape, roi_pad=1.0):
    """the gate's box on the composite: the paste's median centre, 1.6 face heights a side."""
    cx = np.median([f["centre"][0] for f in R["per_frame"]])
    cy = np.median([f["centre"][1] for f in R["per_frame"]])
    s = R["face_h_px"] * 1.6 * roi_pad
    H, W = shape[:2]
    return [int(max(0, cy - s / 2)), int(min(H, cy + s / 2)), int(max(0, cx - s / 2)), int(min(W, cx + s / 2))]


REPAIR_DROP = 0.03  # a continuity repair that costs the audio sync more than this is not delivered


def mouth_open(video, R, n):
    """the mouth opening of a shot video on the gate's box (sync_qa.measured_opening), n frames."""
    from ..audits import sync_qa as Q

    B, fps = MED.read(video, n)
    return Q.measured_opening(B, face_roi(R, B[0].shape), fps)[0]


def travel(op):
    return None if op is None else round(float(np.percentile(op, 95) - np.percentile(op, 5)), 4)


TRAVEL_REFS = ("take", "aligner")


def travel_guard(info, take, align, vocal, t0, out, work, R, again, ref="take"):
    """a retimed shot whose delivered mouth is a held gape against its own take (scorecard.held_gape: under 0.10 of travel and under
    TRAVEL_KEEP of the take's, or under the floor) is rendered again from the aligner's times (work/aligner); when that one passes and
    is not a held gape, it is delivered instead. The retime can shorten the plan's openings in a way the plan's own reading does not
    show. -> the info of what is delivered, with the readings under retime.guard.

    ref: what the retimed mouth's travel is read against before it is swapped. "take": the take's alone. "aligner": also the
    aligner's render, the same shot painted without the retime; a retimed mouth that keeps TRAVEL_KEEP of that render's travel was
    not shortened by the retime (its low travel is the shot's own, e.g. a take that rests half open), so it stays."""
    import shutil

    from ..audits import scorecard as SC

    if ref not in TRAVEL_REFS:
        raise ValueError(f"travel ref {ref!r}")
    n = info["frames"]
    t_take, t_out = travel(mouth_open(take, R, n)), travel(mouth_open(out, R, n))
    held = lambda t: t is not None and t_take is not None and SC.held_gape(dict(mouth_range=t, before=dict(mouth_range=t_take)))
    guard = {"take_travel": t_take, "retimed_travel": t_out, "held": held(t_out)}
    if not guard["held"]:
        info["retime"]["guard"] = guard
        return info
    alt_work = os.path.join(work, "aligner")
    alt_out = os.path.join(alt_work, os.path.basename(out))
    os.makedirs(alt_work, exist_ok=True)
    if os.path.exists(os.path.join(work, "occ.npz")):  # the occluder mask (a mic at the lips) goes with the shot
        shutil.copyfile(os.path.join(work, "occ.npz"), os.path.join(alt_work, "occ.npz"))
    alt = run(take, align, vocal, t0, alt_out, alt_work, vocal_retime=False, report=None, **again)
    guard["aligner_travel"] = travel(mouth_open(alt_out, R, n))
    guard["aligner_pass"] = bool(alt.get("pass"))
    if ref == "aligner" and t_out is not None and guard["aligner_travel"] is not None:
        guard["shortened"] = bool(SC.held_gape(dict(mouth_range=t_out, before=dict(mouth_range=guard["aligner_travel"]))))
        if not guard["shortened"]:
            info["retime"]["guard"] = dict(guard, delivered="retimed")
            return info
    if alt.get("pass") and not held(guard["aligner_travel"]):
        shutil.copyfile(alt_out, out)
        alt["retime"] = dict(info["retime"], used=False, guard=dict(guard, delivered="aligner"))
        return alt
    info["retime"]["guard"] = dict(guard, delivered="retimed")
    return info


def delivered_truth(video, info, work):
    """sync_truth of a finished shot video on the gate's box, against the gate's vocal clip (work/vox.wav) and frames."""
    from ..audits import sync_qa as Q, sync_truth as ST
    from . import vocalisations as VOC

    R = json.load(open(os.path.join(work, "medium.json")))
    B, fps = MED.read(video, info["frames"])
    op, _ = Q.measured_opening(B, face_roi(R, B[0].shape), fps)
    if op is None:
        return dict.fromkeys(ST.KEYS)
    return ST.read_wav(op, os.path.join(work, "vox.wav"), fps, VOC.scored_frames(info.get("warp_json"), len(op))[0])


def repaired_silence(video, info, work):
    """the gate's open-in-silence (snap) on a finished shot video: its mouth on the gate's box, the gate's vocal clip, the plan's moves."""
    from ..audits import snap as SN
    from .coart import logged_offsets

    R = json.load(open(os.path.join(work, "medium.json")))
    op = mouth_open(video, R, info["frames"])
    vox = os.path.join(work, "vox.wav")
    if op is None or not os.path.exists(vox):
        return None
    dt = logged_offsets(info.get("warp_json"), len(op), info["fps"])
    return SN.snap_metrics(op, vox, [], info["t0"], info["fps"], dt=dt)["open_in_silence"]


def repair(take, sung, out, info, work, report=None, charge="remade"):
    """the continuity repair (single-frame pops against the take) on a sung shot, delivered only when the repaired mouth still follows
    the vocal: it puts take frames back, and a take's mouth is not on the song. A repair that costs more than REPAIR_DROP of the audio
    sync, turns a passing shot into a failing one, or puts its open-in-silence over GATE_SILENCE, is undone (out = the sung shot).
    charge: what the continuity pass's coherence share is charged with (continuity.run's charge: "remade", the default, or "all").
    -> the reading, into the gate's info."""
    import shutil

    from . import continuity as CONT

    CONT.run(take, sung, out, report=report, charge=charge)
    if info.get("truth") is None or not os.path.exists(os.path.join(work, "medium.json")):
        return {"repair": "kept"}
    t = delivered_truth(out, info, work)
    sil = repaired_silence(out, info, work)
    over = sil is not None and sil > GATE_SILENCE and (info.get("open_in_silence") is None or info["open_in_silence"] <= GATE_SILENCE)
    if over or (
        t.get("truth") is not None and (t["truth"] < info["truth"] - REPAIR_DROP or (info.get("truth_pass") and not t["truth_pass"]))
    ):
        shutil.copyfile(sung, out)
        if report and os.path.exists(report):  # the report says what was delivered: the sung shot, its repairs not applied
            rep = json.load(open(report))
            rep.update(
                undone=True,
                delivered="sung",
                undone_why=f"audio sync {info['truth']} -> {t['truth']}, open in silence {info.get('open_in_silence')} -> {sil} with the repairs",
            )
            json.dump(rep, open(report, "w"), indent=1)
        return {"repair": "undone", "repair_truth": t["truth"], "repair_silence": sil}
    return {"repair": "kept", "repair_truth": t.get("truth"), "repair_silence": sil}


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
        help="viseme / dyn lead, s (default 0.06 with snap, 0.10 without)",
    )
    ap.add_argument("--word-close", type=float, help="between-word closure, s (default 0.06 with snap)")
    ap.add_argument(
        "--drive-pump",
        type=float,
        help="the drive's gate + 4:1 compressor (sung_dyn.pump) at this gate, on the viseme envelope and the sung_dyn curve; off by "
        "default",
    )
    ap.add_argument("--drive-comp", action="store_true", help="the drive opens to a level-compressed copy of the lead stem; off by default")
    ap.add_argument(
        "--no-vocal-retime",
        dest="vocal_retime",
        action="store_false",
        help="plan from the alignment's own times; by default its words and phones are first re-read off the vocal (vocal_retime)",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="also run the native warp (full frame, frame rate, --lead 0.1, no word-close: the pre-sing recipe) and keep "
        "the one whose gate passes, else the higher audio sync (the plan corr when the vocal reads nothing); open-in-silence "
        "breaks a tie within 0.02 (an unmeasured silence counts as the worst)",
    )
    ap.add_argument(
        "--travel-guard",
        choices=TRAVEL_REFS,
        default="aligner",
        help="a retimed mouth read as a held gape is swapped for the aligner's render; 'aligner' (default): only when the retime "
        "shortened it against that render too; 'take': on the take's reading alone",
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
        vocal_retime=a.vocal_retime,
        pump=(a.drive_pump, 4.0) if a.drive_pump else None,
        drive_comp=a.drive_comp,
        travel_ref=a.travel_guard,
    )
    return 0 if info["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
