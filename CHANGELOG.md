# Changelog

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
