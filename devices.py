# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""devices.py: ComfyUI picks the device for the torch parts (identity embedder, speech-to-text, vocal split); light CV
(face finding, landmarks, sync math) stays on the CPU. Models are offloaded when a node finishes."""
import os

import folder_paths


def torch_device():
    import comfy.model_management as mm
    try:
        return mm.get_torch_device()
    except RuntimeError:                                                # no usable GPU: the CPU
        import torch
        return torch.device('cpu')


def offload_device():
    import comfy.model_management as mm
    return mm.unet_offload_device() if hasattr(mm, 'unet_offload_device') else 'cpu'


def free():
    import comfy.model_management as mm
    if hasattr(mm, 'soft_empty_cache'): mm.soft_empty_cache()


def models_dir(*parts):
    d = os.path.join(folder_paths.models_dir, 'lsdub', *parts)
    os.makedirs(d, exist_ok=True)
    return d


def embedder():
    """the identity embedder on ComfyUI's device (torch DINOv2; the ONNX model when ComfyUI runs on the CPU)."""
    from .lsdub_engine.core import cast as C
    dev = torch_device()
    if str(dev) == 'cpu' and os.path.exists(C.MODEL): return C.Embedder(C.MODEL)      # the CLI's ONNX model, if exported
    return C.TorchEmbedder(dev, cache_dir=models_dir('dinov2'), offload_device=offload_device())


def release(emb):
    if hasattr(emb, 'offload'): emb.offload()
    free()


def whisper_device():
    """(device, index) for faster-whisper from ComfyUI's device: ('cuda', N) or ('cpu', 0)."""
    dev = torch_device()
    if getattr(dev, 'type', str(dev)).startswith('cuda'): return 'cuda', (dev.index or 0)
    return 'cpu', 0


def ensure_face_models():
    """the small face models (MediaPipe Face Landmarker, YuNet) in models/lsdub: downloaded once, checksum-verified,
    the first time a face node runs (never at startup)."""
    from .lsdub_engine import _models
    _models.fetch_face_landmarker()
    _models.fetch_yunet()
