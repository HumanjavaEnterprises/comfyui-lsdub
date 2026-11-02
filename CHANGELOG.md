# Changelog

## Unreleased

- Stage keys: a stage before the dub carries the release plus the hash of the engine files it can reach, so a change to the paint keeps the take, the kit and the alignment; the dub and after carry the whole engine. A change a stage can reach re-runs it, and every folder keyed by the older scheme re-runs once.
- `lead_sheet`: a span read only as outside the clip's pitch range is a proposal: it stays in the sheet and passes no word until a `pass` line says so. A number like `1.2.3` in `lead_overrides` stops with the usage line.
- LSDub Verify's `vocal` takes no vowel length from the plan and reads `lost`, no number, on a shot where the free walk has run away, as it can past about 40 s of a long song. Verify reads the vocal once for the run, and clips under a window no longer stop it. The retime's silence follows a noise floor on a clip that has gaps.
- `layers_shape`: carries its temporal smoothing and a 40 ms plan advance, so no `settings` are needed. Reduced vowels read smaller than full ones, diphthongs move through both their shapes, a function word loses opening only when it is quick, lips round on rounded sounds, and F, V and TH show their contact: lower lip on the upper teeth, tongue tip under them. p, b, m still shut. Other stacks are unchanged.
- LSDub Align gains optional `lead_sheet` (default off, experimental) and `lead_overrides`: the song written out with what the singer on screen sings plain and backing stutters and answer lines in braces, timed, with a confidence and a reason, as `lead_sheet.txt` and `.json` beside the alignment and in `lyric_check`. What is passed over keeps the take's mouth, drives no alignment or retime and is not read by Verify; correct a call with `pass 10.04 11.93` or `lead 10.04 10.49`.
- `layers_shape`: each closure takes its erase reference near it in face size and time. Two closures close together leave the vowel between them room to open.
- LSDub Verify's `vocal` takes no time from the plan or the aligner, so a retimed plan is not marked against itself. `retime_to_vocal`: a vocal with silence before or after it no longer pins or stretches the words at its ends; nasals, glides and frame times read truer.
- LSDub Align gains optional `retime_to_vocal` (default off, experimental): phone times read off the vocal itself, for held notes the aligner spreads across the line. The lyric stays as it is.
- `layers_shape`: closures are reached at the take's own speed; no blue streak or pale notch on a thin opening; closed lips keep the take's upper-to-lower lip tone. LSDub Verify reports the plan's word times against the vocal, moving no band.
- New mouth_stack option `layers_interior`, the inside of the mouth from the take; workflow 4 uses it. LSDub Mouth Tweaks gains optional `sync_phrases` (default off), each phrase landing by its vocal. Mouth Ruler gains `boil_all`.
- `scripts/make_workflows.py` builds the committed example workflows byte for byte again; `--check` compares without writing, and thumbnails are left alone unless a picture is given.
- LSDub Mouth Tweaks gains optional `sync_advance_ms` (default 0): the whole mouth plan lands that much earlier against the sound, shapes unchanged. At 0 the value is left out of the tweaks hash, so 0 and an older saved Tweaks node key the same; wiring a Tweaks node into a workflow that had none still moves that workflow's shot keys.
- LSDub Verify reports `env_corr`, `env_best`, `env_lag_ms`: the mouth against the vocal's loudness with no alignment, next to the sync score, which reads the curve the dub was painted from. Reported, they move no band.
- New mouth_stack option `layers_shape`: `layers_interior` with mouth heights that follow the sounds rather than the take's own range, and closures that really shut. Off unless you pick it.
- LSDub Verify reports the delivered mouth's shape, moving no band: height contrast by sound, p/b/m closures with the lips really shut, width travel. The mouth card is labelled PLAN and reads the moved timing.
- A `--warp-temporal` in Tweaks `settings` now holds on the sing paint as well as the native one: larger values calm the mouth and cost sync. Left out, nothing changes.
- `--warp-interior-source take`: the pink sink holds from the first frame it is needed, lands on its target and lets go over a few frames; the take's pink share is read clear of the lip rim, with the fill stopping at 0.94 of it; a bin of that curve reading zero counts as no reading, the curve is smoothed across bins, and no reading anywhere sets no cap (this changes that flag's render); softening 0.012 to 0.02 of the box's side (`SOFT`).
- New mouth-warp flags through `settings`: `--warp-teeth-fit take`, `--occluder-plate`, `--coart-advance-ms`. Mouth Ruler gains `pink_core`.
- Align: words a phrase split left out of lyric order go back in order inside the span they cover.
- Lip-lock is gone from the pack: the landing preset carries no lip-colour geometry and no argv changes.
- The vendored `lsdub_engine/workflows/*.json` are gone; `graphs.py` is the single source of the generator graphs.
- The lyric sheet's scat markup, `( ... )` and `[skip]`, now reaches the Dub: Align reads it, leaves it out of the speech-to-text prompt, and flags those words in `align.json`.
- Workflow 4 uses `shot_seconds` 5.5, so the demo song takes 3 generated shots, not 4.
- `lsdub_engine/` and `tests/engine/` are now build output, never edited here; `lsdub_engine/VERSION` carries a tree hash that CI checks; stage keys use it only from the dub on (see Stage keys above).
  Presets move to `lsdub_engine/core/presets.py`, recipes arrive beside them.
- One cache reset: every stage key moves once in this release, so the generator slots (LTX, Qwen) re-run too, along with analysis, kits and shots. Earlier folders stay on disk. It groups four changes: the engine version now includes the vendored tree hash, kits hash by bytes instead of modification time, the unused project field `vocalisations` is gone, and `mouth_stack` arrives. A `project.json` from 0.1.3 still loads.
- LSDub Project gains `mouth_stack` (`plain`, `layers`). `layers` renders the full layer stack on any kit, in its own cache folders. The default, `plain`, keeps every kit's render as it was.
- LSDub Mouth Tweaks `settings` can now change `vocalisations`. Quoting rule: whitespace separates, single or double quotes group a value, backslashes stay as typed (so `C:\\Users\\me\\a.csv` works). An apostrophe goes inside double quotes; an unmatched quote stops at the node with a clear message.
- Records: `Song.vocal`, `Song.inst`, `Song.lead_activity`, `Align.visemes`, `Cast.plan` and `Report.fix` are gone (nothing read them); older saved records still load. `Render` now carries `audio`, `on_fail` and the song, so LSDub Continuity keeps a song-audio cut's sound. The `lead_activity_csv` output of LSDub Lead Vocal stays.
- The engine CLI ships its pack commands only; the development commands stay in the lsdub repo. A Video output now fails with a clear message on a ComfyUI older than the floor, instead of passing a bare path along.
- Model downloads are pinned: revisions and checksums in `lsdub_engine/_models.py`, checked once per snapshot, agreed against `comfy-build.yaml`. Occluders downloads at the pinned revisions.
- README: running times removed; Project inputs and the bug form's node list are generated from the schemas.
- New node LSDub Mouth Ruler: take against render per dubbed shot, reported only.
- LSDub Verify shows `N/A` for a shot left as source, leaves vocalisation frames out of the sync score (`vocal_frac` says how many; shots without them read the same) and reports a lower-face identity beside the head one. The sing gate scores the same frames.
- LSDub Scorecard gains optional `sharpness_mode` (`floor`, `plate`) and a `boil` reading at weight 0. LSDub Fix It gains optional `scorecard` and `ruler` inputs and more reasons, among them `vocalisation` and `no kit for target`.
- LSDub Dub names each shot no kit serves (warning, and `no kit for target` in the results) instead of dropping it silently.
- The Dub logs paint flags that have no effect on a run and records them in the shot's `done.json`. Model unload also closes the shared face landmarkers.

## 0.1.3 (2026-10-30)

- Speech-to-text falls back to the CPU when the GPU runtime is missing (for example CUDA 12 cuBLAS on a CUDA 13 image), so Align works on hosted GPUs.

## 0.1.2 (2026-10-30)

- Example workflows load correctly in the ComfyUI frontend (seed control widget). The frontend adds a "control after
  generate" value right after LSDub Project's seed; the shipped workflows now include it, so no value is shifted.
- Tests model the frontend rule, and the workflow generator writes the control value.
- Issue forms for bugs and suggestions, CONTRIBUTING and SECURITY.

## 0.1.1 (2026-10-29)

Docs, CI and hosted-run fixes.

- Runs anywhere ComfyUI runs: falls back to a bundled ffmpeg (imageio-ffmpeg) when none is installed, and loads
  DINOv2 and Whisper straight from models/lsdub with no network call.
- Lip Kit accepts a zipped kit (so it travels with a job), with safe unpacking and size limits.
- Transcript and kit inputs resolve inside ComfyUI's input folder only.
- Save Cut, Verify, Scorecard, Sync QA and Triage report their files to ComfyUI, so the cut and reports come back
  with every job (queue history and hosted runs).

- README rewritten around the tool: what it is and why, the three lanes (VIDEO, AUDIO, MOUTH, then QA), start here,
  install, the four workflows, Test & Learn vs Run, Comfy Cloud, runtimes and troubleshooting.
- Model details (sizes, licences, folders, which workflow needs what, disk, first-use downloads) moved to MODELS.md.
- CI: installs the GLES/EGL libraries MediaPipe needs.
- Tests: the registration-ruler gate test skips when the face model is not present (clean CI runners).

## 0.1.0 (2026-10-29)

The whole rack: lip-sync for music videos in ComfyUI.

- **Project** with a folder that keeps everything that renders: every node saves its output keyed by its inputs, a
  re-run reuses what is unchanged, `force_shots` redoes chosen shots once. Modes: `run` and `test_learn`.
- **VIDEO lane:** Video slot (bring a clip, or animate a still with LTX-2.5), Still slot (Qwen-Image-Edit-2511),
  Shot Split, Cast (who is who), Triage (which shots can be dubbed, and why not).
- **AUDIO lane:** Song slot (bring a song, or generate one with ACE-Step 1.5 XL), Lyrics, Vocal Split
  (Mel-Band RoFormer), Lead Vocal, Align (speech-to-text prompted with your lyrics + forced alignment; a low lyric match places the
  lyrics on the voice and is flagged, or stops the run when the lyrics do not belong to the song).
- **MOUTH lane:** Lip Kit (bring one or build one from a shot), Mouth Tweaks (incl. an optional jaw-curve CSV),
  Dub (per shot, resumable, failed shots reported and kept as source by default;
  vocalisations skipped, rested or mouthed), Occluders (keep a mic in front),
  Continuity.
- **QA:** Verify, Scorecard, Sync QA (with a review card), Fix It (problem -> what to try -> force_shots), Save Cut
  (invisible watermark).
- Generators run locally in ComfyUI by default; **Comfy Cloud** is an opt-in project setting.
- **Performance:** a character image + a song -> a singing-length LTX-2.5 performance video.
- Four example workflows (incl. Make Your Character Sing, experimental) and a small demo asset pack (`scripts/fetch_demo.py`).
- V3 nodes with a tooltip on every input and output.

## 0.0.3
- Attribution files (NOTICE, CITATION.cff, SPDX headers).

## 0.0.2
- Vocal split + lead vocal nodes.
