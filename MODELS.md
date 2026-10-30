# LSDub models

Everything LSDub loads, where it goes, and which workflow needs it. Back to the [README](README.md).

## The short version

Nothing is downloaded when ComfyUI starts.

- **Placed by you** (LSDub does not download these): the generator models (ACE-Step, Qwen-Image-Edit, LTX-2.5) and the
  Mel-Band RoFormer vocal checkpoint. Put them in the ComfyUI model folders below. They are standard ComfyUI files: if
  you already run these models in ComfyUI, LSDub uses your copies.
- **Downloaded on first use:** Whisper, DINOv2, the face models and the Occluders models, into `models/lsdub`.
- **Disk:** about 120 GB for Prompt Everything (all the generators). About 3.5 GB for the other workflows (Whisper,
  DINOv2, the face models), plus 0.9 GB for the RoFormer in workflow 4.

## Which workflow needs what

| workflow | you place | downloaded on first use |
|---|---|---|
| 1. Prompt Everything | ACE-Step, Qwen-Image-Edit, LTX-2.5 | Whisper, DINOv2, face models |
| 2. BYO Clip and Song | nothing | Whisper, DINOv2, face models |
| 3. Test and Learn | nothing | Whisper, DINOv2, face models |
| 4. Make Your Character Sing | Qwen-Image-Edit, LTX-2.5, `MelBandRoformer_kim_vocals.ckpt` | Whisper, DINOv2, face models |

LSDub Occluders is optional in every workflow. It downloads Florence-2 and SAM 2.1 the first time it runs.

## The table

| used by | needed by workflows | file | size | licence | folder |
|---|---|---|---|---|---|
| Song slot | 1 | [acestep_v1.5_xl_turbo_bf16.safetensors](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files) | 9.3 GB | MIT ([ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5)) | `diffusion_models` |
| Song slot | 1 | ace_1.5_vae, qwen_0.6b_ace15, qwen_4b_ace15 (same repo) | 0.3 + 1.2 + 7.9 GB | MIT (ACE-Step 1.5) | `vae`, `text_encoders` |
| Still slot, Lip Kit | 1, 4 | [qwen_image_edit_2511 (int8 or bf16)](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) | 20 GB | Apache-2.0 | `diffusion_models` |
| Still slot, Lip Kit | 1, 4 | qwen_2.5_vl_7b, qwen_image_vae, Qwen-Image-Edit-2511-Lightning-8steps lora | 16 + 0.2 + 0.8 GB | Apache-2.0 | `text_encoders`, `vae`, `loras` |
| Video slot, Performance | 1, 4 | [ltx-2.5-22b-distilled-transformer-bf16](https://huggingface.co/Lightricks/LTX-2.5) | 40 GB | LTX-2.x Community License Agreement (see its card) | `diffusion_models` |
| Video slot, Performance | 1, 4 | gemma4-12b-with-proj-ltx-2.5 (text encoder), ltx-2.5-video-vae, ltx-2.5-audio-vae | 25 + 1.4 + 0.3 GB | Apache-2.0 ([Gemma 4](https://huggingface.co/google/gemma-4-12B-it)); LTX-2.x Community License (VAEs) | `text_encoders`, `vae` |
| Vocal Split | 4 | MelBandRoformer_kim_vocals.ckpt: download `MelBandRoformer.ckpt` from [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer) and save it under this name | 0.9 GB | MIT | `audio_separation` |
| Align | all | Whisper large-v3 (faster-whisper / CTranslate2), downloaded on first use | 3 GB | MIT | `lsdub/whisper` |
| Cast, Triage, Verify | all | DINOv2 base (facebook/dinov2-base), downloaded on first use | 0.3 GB | Apache-2.0 | `lsdub/dinov2` |
| every face node | all | MediaPipe Face Landmarker + YuNet, downloaded on first use (checksummed) | 4 MB | Apache-2.0 / MIT | `lsdub` |
| Occluders | optional | Florence-2-large + SAM 2.1 base+, downloaded on first use | 1.6 GB | MIT / Apache-2.0 | `lsdub/occluders` |

## First-use downloads

The first Align run downloads Whisper large-v3 (about 3 GB) with no progress bar in the UI. Let it finish once.
Workflows 2 and 3 ship the demo transcript (`lsdub_demo_landing_stt.json`), so their alignment is the same on every run.

## GPU memory

Rough needs: ACE-Step about 12 GB, Qwen-Image-Edit (int8) about 24 GB, LTX-2.5 (bf16) about 70 GB.
ComfyUI offloads to system RAM on smaller cards, which is slower. Or run the generators on Comfy Cloud
(see the README).
