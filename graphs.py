# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""graphs.py: the stock open-weights generator graphs, in ComfyUI API format, built from parameters.

  song   ACE-Step 1.5 XL turbo (lyrics + tags -> a full song)          SaveAudio      node 'out'
  still  Qwen-Image-Edit-2511 + Lightning 8-step (image + edit prompt)  SaveImage      node 'out'
  video  LTX-2.5 distilled image-to-video (a still + prompt -> clip)     SaveVideo      node 'out'

Uploaded files are referenced by name; `generate.run` maps local files to them. Model file names are parameters
with the stock names as defaults (the README's MODELS table lists them)."""
SONG_MODELS = dict(unet='acestep_v1.5_xl_turbo_bf16.safetensors', vae='ace_1.5_vae.safetensors',
                   clip1='qwen_0.6b_ace15.safetensors', clip2='qwen_4b_ace15.safetensors')
STILL_MODELS = dict(unet='qwen_image_edit_2511_int8_convrot.safetensors', clip='qwen_2.5_vl_7b.safetensors',
                    vae='qwen_image_vae.safetensors', lora='Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors')
VIDEO_MODELS = dict(unet='ltx-2.5-22b-distilled-transformer-bf16.safetensors', clip='gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',
                    vae='ltx-2.5-video-vae-bf16.safetensors', audio_vae='ltx-2.5-audio-vae-bf16.safetensors')
# model files whose name differs on Comfy Cloud (checked against cloud.comfy.org object_info, 2026-10-29)
CLOUD_NAMES = {'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors': 'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors'}


def for_backend(graph, backend):
    """the graph with each model file named as that backend has it."""
    if backend != 'comfy_cloud': return graph
    return {k: {**n, 'inputs': {f: CLOUD_NAMES.get(v, v) if isinstance(v, str) else v for f, v in n['inputs'].items()}}
            for k, n in graph.items()}


DISTILLED_8 = '1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0'
KEYS = ['C major', 'C# major', 'Db major', 'D major', 'D# major', 'Eb major', 'E major', 'F major', 'F# major', 'Gb major',
        'G major', 'G# major', 'Ab major', 'A major', 'A# major', 'Bb major', 'B major', 'C minor', 'C# minor', 'Db minor',
        'D minor', 'D# minor', 'Eb minor', 'E minor', 'F minor', 'F# minor', 'Gb minor', 'G minor', 'G# minor', 'Ab minor',
        'A minor', 'A# minor', 'Bb minor', 'B minor']


def song(tags, lyrics, seconds, seed, bpm=120, keyscale='C major', prefix='lsdub/song', models=None):
    m = {**SONG_MODELS, **(models or {})}
    return {
        'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': m['unet'], 'weight_dtype': 'default'}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': m['vae']}},
        'clip': {'class_type': 'DualCLIPLoader', 'inputs': {'clip_name1': m['clip1'], 'clip_name2': m['clip2'], 'type': 'ace',
                                                            'device': 'default'}},
        'shift': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['unet', 0], 'shift': 3.0}},
        'pos': {'class_type': 'TextEncodeAceStepAudio1.5', 'inputs': {
            'clip': ['clip', 0], 'tags': tags, 'lyrics': lyrics, 'seed': int(seed), 'bpm': int(bpm), 'duration': float(seconds),
            'timesignature': '4', 'language': 'en', 'keyscale': keyscale, 'generate_audio_codes': True, 'cfg_scale': 2.0,
            'temperature': 0.85, 'top_p': 0.9, 'top_k': 0, 'min_p': 0.0}},
        'neg': {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['pos', 0]}},
        'latent': {'class_type': 'EmptyAceStep1.5LatentAudio', 'inputs': {'seconds': float(seconds), 'batch_size': 1}},
        'sample': {'class_type': 'KSampler', 'inputs': {
            'model': ['shift', 0], 'positive': ['pos', 0], 'negative': ['neg', 0], 'latent_image': ['latent', 0],
            'seed': int(seed), 'steps': 8, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        'decode': {'class_type': 'VAEDecodeAudio', 'inputs': {'samples': ['sample', 0], 'vae': ['vae', 0]}},
        'out': {'class_type': 'SaveAudio', 'inputs': {'audio': ['decode', 0], 'filename_prefix': prefix}},
    }


def still(image, prompt, seed, ref=None, megapixels=1.0, prefix='lsdub/still', models=None,
          negative='cartoon, painting, blurry, distorted, deformed, text, watermark'):
    """Qwen-Image-Edit-2511: edit `image` (an uploaded name) by `prompt`; `ref` = an optional second reference image."""
    m = {**STILL_MODELS, **(models or {})}
    g = {
        'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': m['unet'], 'weight_dtype': 'default'}},
        'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': m['clip'], 'type': 'qwen_image'}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': m['vae']}},
        'lora': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['unet', 0], 'lora_name': m['lora'], 'strength_model': 1.0}},
        'shift': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['lora', 0], 'shift': 3.1}},
        'norm': {'class_type': 'CFGNorm', 'inputs': {'model': ['shift', 0], 'strength': 1.0}},
        'load': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        'scale': {'class_type': 'ImageScaleToTotalPixels', 'inputs': {'image': ['load', 0], 'upscale_method': 'lanczos',
                                                                     'megapixels': float(megapixels), 'resolution_steps': 8}},
        'pos': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {'clip': ['clip', 0], 'prompt': prompt, 'vae': ['vae', 0],
                                                                        'image1': ['scale', 0]}},
        'neg': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {'clip': ['clip', 0], 'prompt': negative, 'vae': ['vae', 0],
                                                                        'image1': ['scale', 0]}},
        'pos_m': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {'conditioning': ['pos', 0],
                                                                                     'reference_latents_method': 'index_timestep_zero'}},
        'neg_m': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {'conditioning': ['neg', 0],
                                                                                     'reference_latents_method': 'index_timestep_zero'}},
        'enc': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['scale', 0], 'vae': ['vae', 0]}},
        'sample': {'class_type': 'KSampler', 'inputs': {
            'model': ['norm', 0], 'seed': int(seed), 'steps': 8, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple',
            'positive': ['pos_m', 0], 'negative': ['neg_m', 0], 'latent_image': ['enc', 0], 'denoise': 1.0}},
        'decode': {'class_type': 'VAEDecode', 'inputs': {'samples': ['sample', 0], 'vae': ['vae', 0]}},
        'out': {'class_type': 'SaveImage', 'inputs': {'images': ['decode', 0], 'filename_prefix': prefix}},
    }
    if ref:
        g['ref'] = {'class_type': 'LoadImage', 'inputs': {'image': ref}}
        g['pos']['inputs']['image2'] = ['ref', 0]
        g['neg']['inputs']['image2'] = ['ref', 0]
    return g


def video_frames(seconds, fps):
    """LTX wants 8n + 1 frames."""
    n = max(1, int(round(seconds * fps)) // 8)
    return 8 * n + 1


def video(image, prompt, negative, seconds, seed, fps=25.0, width=768, height=512, image_strength=0.7, prefix='lsdub/video',
          models=None):
    """LTX-2.5 distilled image-to-video: the still is written into the first frame; 8 distilled steps; audio branch kept
    (the model is audio-video) but the clip's own audio is not used by LSDub."""
    m = {**VIDEO_MODELS, **(models or {})}
    n = video_frames(seconds, fps)
    return {
        'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': m['unet'], 'weight_dtype': 'default'}},
        'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': m['clip'], 'type': 'ltxv', 'device': 'default'}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': m['vae']}},
        'avae': {'class_type': 'VAELoader', 'inputs': {'vae_name': m['audio_vae']}},
        'load': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        'prep': {'class_type': 'LTXVPreprocess', 'inputs': {'image': ['load', 0], 'img_compression': 18}},
        'pos': {'class_type': 'CLIPTextEncode', 'inputs': {'text': prompt, 'clip': ['clip', 0]}},
        'neg': {'class_type': 'CLIPTextEncode', 'inputs': {'text': negative or 'worst quality, inconsistent motion, blurry, jittery, distorted',
                                                           'clip': ['clip', 0]}},
        'cond': {'class_type': 'LTXVConditioning', 'inputs': {'positive': ['pos', 0], 'negative': ['neg', 0], 'frame_rate': float(fps)}},
        'empty': {'class_type': 'EmptyLTXVLatentVideo', 'inputs': {'width': int(width), 'height': int(height), 'length': n, 'batch_size': 1}},
        'i2v': {'class_type': 'LTXVImgToVideoInplace', 'inputs': {'vae': ['vae', 0], 'image': ['prep', 0], 'latent': ['empty', 0],
                                                                  'strength': float(image_strength), 'bypass': False}},
        'alat': {'class_type': 'LTXVEmptyLatentAudio', 'inputs': {'frames_number': n, 'frame_rate': int(round(fps)), 'batch_size': 1,
                                                                  'audio_vae': ['avae', 0]}},
        'av': {'class_type': 'LTXVConcatAVLatent', 'inputs': {'video_latent': ['i2v', 0], 'audio_latent': ['alat', 0]}},
        'noise': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': int(seed)}},
        'guide': {'class_type': 'CFGGuider', 'inputs': {'model': ['unet', 0], 'positive': ['cond', 0], 'negative': ['cond', 1], 'cfg': 1.0}},
        'samp': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler_ancestral'}},
        'sig': {'class_type': 'ManualSigmas', 'inputs': {'sigmas': DISTILLED_8}},
        'run': {'class_type': 'SamplerCustomAdvanced', 'inputs': {'noise': ['noise', 0], 'guider': ['guide', 0], 'sampler': ['samp', 0],
                                                                  'sigmas': ['sig', 0], 'latent_image': ['av', 0]}},
        'split': {'class_type': 'LTXVSeparateAVLatent', 'inputs': {'av_latent': ['run', 0]}},
        'decode': {'class_type': 'VAEDecodeTiled', 'inputs': {'samples': ['split', 0], 'vae': ['vae', 0], 'tile_size': 512, 'overlap': 64,
                                                              'temporal_size': 64, 'temporal_overlap': 8}},
        'mk': {'class_type': 'CreateVideo', 'inputs': {'images': ['decode', 0], 'fps': float(fps)}},
        'out': {'class_type': 'SaveVideo', 'inputs': {'video': ['mk', 0], 'filename_prefix': prefix, 'format': 'mp4', 'format.codec': 'h264'}},
    }
