# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub — photoreal sung lip-sync from words: forced alignment -> visemes -> a lip kit -> warp and paste -> QA.

`lsdub --help` lists the commands. SHIP is what every build carries, DEV is the workbench (a vendored build has none of it)."""

import importlib
import os
import sys

DOCS_URL = "https://github.com/HumanjavaEnterprises/comfyui-lsdub/blob/main"
HELP_SHIP = """
  lsdub align     lyrics + vocal -> word/phone timings (sphinx | mfa | best)
  lsdub visemes   timings -> per-frame viseme track (min-hold 3)
  lsdub rhubarb   Rhubarb Lip Sync cues (cross-check / fallback)
  lsdub scan      per-frame face mesh + head pose, pose clusters, keyframes
  lsdub cone      dub-safe cone report for a scan (shared vs per-shot library vs re-stage)
  lsdub mouths    mouth library from one frame (prep | gen | sheet; gen needs ComfyUI + Qwen-Image-Edit-2511)
  lsdub animate   animator pass: phones + vocal -> eased, coarticulated track (jaw/lip split, G gain)
  lsdub paint     paint the viseme track onto a take (morph, tame, open cap 0.85; --recipe NAME) [watermarked]
  lsdub interior  one steady mouth interior per shot (upper teeth hang from the upper lip)     [watermarked]
  lsdub calm      flow-guided temporal calm inside the mouth (kills per-frame texture boil)  [watermarked]
  lsdub motion-audit  in-motion metrics: boil, ghost, slide, step; frame-by-frame mouth crops
  lsdub pin       pin the painted lower face to the take's REAL skin motion (dense flow), so the mouth never swims  [watermarked]
  lsdub slip-audit  does the painted mouth stay pinned? philtrum / lip slip vs the nose, in px, and its lag
  lsdub frame-gate   per-frame checker: every frame vs its neighbours and the take (PASS / SOFT / FAIL + causes, slosh windows)
  lsdub track     skip = mark a viseme track's vocalisations (uh uh, ah, da da, mm, hey) for the warp; scan = list a song's spans
  lsdub annotate  per-frame card in a corner: the sung word, its mouth shape (a line glyph), openness, the next second of visemes
  lsdub strip     sync strip: the vocal waveform scrolling under the video past a fixed playhead, to judge lips vs voice by eye
  lsdub sync-qa   mouth sync QA: measured mouth vs the opening the aligned vocal asks for (corr, best lag) + a review video
  lsdub sync-truth  the mouth read against the vocal alone (loudness + open-voiced curves, levels and events), no alignment in between
  lsdub cadence   motion cadence checker: held / skipped frames (a generator stall) measured on the shot's own motion
  lsdub retime    motion cadence maker: re-spaces the frames around each hitch the cadence checker flags, so the motion is even
  lsdub occluders masks of what sits in front of the mouth (a mic, a hand) from plain words, followed through the shot (GPU)
  lsdub continuity  repairs single-frame pops in a painted mouth against the take, checked against the frames before it
  lsdub skull-lock  put the mouth back on the skull: one midline, mouth line parallel to the pupils, rigid upper lip  [watermarked]
  lsdub constraints anatomy-rule violations per video vs the character's own take
  lsdub skull-audit midline / axis / teeth-plane alignment metrics
  lsdub plane-fit  is the mouth on the face's 3D plane? upper-lip arc vs the take's own lip rotated into each frame's pose
  lsdub posemix   blend paints from several per-pose libraries by head pose                  [watermarked]
  lsdub check     lip-sync checker: viseme agreement, word hits, jaw correlation
  lsdub jerk      temporal metrics: mouth jerk p95, seam
  lsdub naturalness  ruler: a render's mouth MOTION vs the take's own, statistically (kinematics, spectrum, shape manifold, dynamics)
  lsdub layer-reg    ruler: inter-layer registration -- teeth / cavity / tongue vs the lips over time in the skull frame (slosh, jitter, jaw residual)
  lsdub tooth-scale  ruler: are the teeth the face's size? visible tooth span vs the take (IPD), teeth counted, plate central widths
  lsdub cast      who is in the shot: enroll characters, scan clips (track + identify + dub_safe), plan the dub from lyrics
  lsdub grammar   shot-grammar gate: every sung word on a big-enough face of its OWN singer (CU hooks, medium verses, wides on held notes)
  lsdub medium    tier-2: track + crop + upscale a 120-300 px singer face, warp it, paste ONLY the feathered lower face back  [watermarked]
  lsdub sing      THE standard path for every sung cut: cast -> 512 px crop -> snappy 2x-fps warp -> lower-face paste -> measured gate  [watermarked]
  lsdub qa        general QA for a contributed video: `qa triage` (auto shots -> DUB / DUB-MEDIUM / NO + reason) and `qa verify` (likely success per dubbed shot)
  lsdub scorecard score a delivered film SHOT BY SHOT (0-100, worst first): sync, silence, snap, tier, mouth, canon, motion, pops, sharpness  [--compare old.mp4]
  lsdub mouth-ruler  natural-mouth baseline: take vs render stats, per-stage sharpness, null test, mouth vs cheek spectrum and grain
  lsdub snap      snap metrics of a mouth-opening series: rise time vs vocal onsets, p/b/m closure hit-rate, open-in-silence
  lsdub reel      before/after reel                                                           [watermarked]
  lsdub cutout    NON-HUMAN mode: dub a rock / object / muppet via a fixed 2D mouth anchor (--mode cutout|puppet), no face tracking  [watermarked]
  lsdub watermark apply | verify
  lsdub verify-watermark VIDEO
  lsdub fetch-models
  lsdub dry       how to get a dry vocal

Output of the [watermarked] commands carries an invisible watermark and a metadata tag. --no-watermark turns it off
and prints a responsible-use warning. `lsd` is a short alias.
"""
HELP_DEV = ""
HELP = HELP_SHIP + HELP_DEV

SHIP = {
    "naturalness": "audits.naturalness",
    "align": "core.lyric_align",
    "visemes": "core.viseme_track",
    "rhubarb": "core.rhubarb_cues",
    "scan": "core.face_scan",
    "cone": "core.cone",
    "mouths": "core.mouth_library",
    "paint": "core.mouth_paint",
    "posemix": "core.pose_mix",
    "check": "audits.lipsync_check",
    "jerk": "audits.lipsync_temporal",
    "reel": "audits.reel",
    "watermark": "watermark",
    "animate": "core.animator_pass",
    "interior": "core.mouth_interior",
    "calm": "core.temporal_stab",
    "motion-audit": "audits.temporal_audit",
    "skull-lock": "core.skull_lock",
    "constraints": "core.face_constraints",
    "skull-audit": "audits.skull_audit",
    "plane-fit": "audits.plane_fit",
    "pin": "core.mouth_pin",
    "slip-audit": "audits.slip_audit",
    "cutout": "anchors.mouth_paint_cutout",
    "frame-gate": "audits.frame_gate",
    "track": "core.vocalisations",
    "annotate": "audits.viseme_overlay",
    "strip": "audits.sync_strip",
    "cadence": "audits.cadence",
    "sync-qa": "audits.sync_qa",
    "sync-truth": "audits.sync_truth",
    "retime": "core.retime",
    "occluders": "core.occluders",
    "continuity": "core.continuity",
    "cast": "core.cast",
    "grammar": "core.grammar",
    "medium": "core.medium",
    "sing": "core.sing",
    "snap": "audits.snap",
    "scorecard": "audits.scorecard",
    "qa": "audits.qa",
    "mouth-ruler": "audits.mouth_ruler",
    "layer-reg": "audits.layer_reg",
    "tooth-scale": "audits.tooth_scale",
}
DEV = {}
CMDS = {**SHIP, **DEV}
# command -> how to find the file it wrote (None = no final video output)
WATERMARKED_SHIP = {
    "pin": "--out",
    "skull-lock": "--out",
    "paint": "--out",
    "interior": "--out",
    "calm": "--out",
    "posemix": "--out",
    "reel": "--out",
    "cutout": "--out",
    "medium": "--out",
    "sing": "--out",
}
WATERMARKED_DEV = {}
WATERMARKED = {**WATERMARKED_SHIP, **WATERMARKED_DEV}

DRY = """A dry (dereverbed, lead-only) vocal makes alignment and the jaw curve far better. Any tool works; what we used:
  1. stem split (e.g. Demucs htdemucs_ft) to get the vocal stem;
  2. dereverb + lead/backing split (we used our own WPE + late-reverb subtraction + stereo-centre DSP).
Then pass the dry lead as --audio to `lsdub align` and `lsdub check`."""


def _out_path(cmd, argv):
    spec = WATERMARKED.get(cmd)
    if spec is None:
        return None
    if isinstance(spec, tuple):
        sub, flag = spec
        if not argv or argv[0] != sub:
            return None
    else:
        flag = spec
    if flag in argv and argv.index(flag) + 1 < len(argv):
        return argv[argv.index(flag) + 1]
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        return 0
    cmd, rest = argv[0], argv[1:]
    no_wm = "--no-watermark" in rest
    rest = [a for a in rest if a != "--no-watermark"]
    if cmd == "dry":
        print(DRY)
        return 0
    if cmd == "fetch-models":
        from ._models import fetch_face_landmarker, fetch_yunet

        print("face landmarker ->", fetch_face_landmarker())
        print("face detector ->", fetch_yunet())
        print(f"Rhubarb / MFA / ComfyUI models are installed separately: see {DOCS_URL}/MODELS.md")
        return 0
    if cmd == "verify-watermark":
        from . import watermark

        return watermark.main(["verify", *rest])
    if cmd not in CMDS:
        print(f"unknown command {cmd!r}\n", file=sys.stderr)
        print(HELP, file=sys.stderr)
        return 2
    mod = importlib.import_module(f".{CMDS[cmd]}", __package__)
    if cmd == "run" and no_wm:
        rest = rest + ["--no-watermark"]
    ret = mod.main(rest)
    out = _out_path(cmd, rest)
    if out and os.path.exists(out):
        from . import watermark

        if no_wm:
            print(watermark.NO_WM_WARNING, file=sys.stderr)
        elif not watermark.verify(out, frames=8)["watermark"]:
            watermark.apply_inplace(out)
            print(f"watermarked {out}", file=sys.stderr)
    if cmd == "paint" and out and "--annotate" in rest and not no_wm:  # paint --annotate's card video, marked as the clean one
        from .core.mouth_card import annotated_path
        from . import watermark

        ann = annotated_path(out)
        if os.path.exists(ann) and not watermark.verify(ann, frames=8)["watermark"]:
            watermark.apply_inplace(ann)
            print(f"watermarked {ann}", file=sys.stderr)
    return ret if isinstance(ret, int) else 0


def entry():
    """Console entry: exit hard after flushing, so MediaPipe's native destructors don't print shutdown noise."""
    ret = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(ret if isinstance(ret, int) else 0)


if __name__ == "__main__":
    entry()
