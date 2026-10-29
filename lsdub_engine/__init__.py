# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub_engine (LSDub): photoreal sung lip-sync from words.

Pipeline: forced alignment -> visemes -> per-shot pose-native mouth library ->
mesh-mapped paint (clean plate, skull-lock, jaw-lock) -> word-lead -> zero-phase
temporal calm -> optical-flow pin. Non-human subjects (rocks, cut-outs, muppets)
go through the `anchors` cutout/puppet modes instead of face tracking.

Public API is lazy: `import lsdub_engine` stays cheap (no cv2/mediapipe/scipy/comfy
pulled in). Access a module by name and it is imported from its subpackage, e.g.
`lsdub_engine.mouth_paint`, `lsdub_engine.mouth_paint_cutout`. The canonical
import paths are the subpackages themselves (`lsdub_engine.core.mouth_paint`, ...).
"""
import importlib

__version__ = "0.1.0"

# public name -> dotted submodule path (its real home)
_API = {
    # core (the v11pin spine)
    "lyric_align": "core.lyric_align", "viseme_track": "core.viseme_track",
    "rhubarb_cues": "core.rhubarb_cues", "animator_pass": "core.animator_pass",
    "face_scan": "core.face_scan", "cone": "core.cone",
    "mouth_library": "core.mouth_library", "mouth_paint": "core.mouth_paint",
    "mouth_morph": "core.mouth_morph",
    "skull_lock": "core.skull_lock", "face_constraints": "core.face_constraints",
    "pose_mix": "core.pose_mix",
    "mouth_pin": "core.mouth_pin",
    # anchors (non-human)
    "mouth_paint_cutout": "anchors.mouth_paint_cutout", "stylized": "anchors.stylized",
    # backends
    "comfy": "backends.comfy", "comfy_backend": "backends.comfy_backend",
    "qwen_graph": "backends.qwen_graph",
    # audits
    "lipsync_check": "audits.lipsync_check", "lipsync_temporal": "audits.lipsync_temporal",
    "temporal_audit": "audits.temporal_audit", "slip_audit": "audits.slip_audit", "skull_audit": "audits.skull_audit",
    "plane_fit": "audits.plane_fit", "lip_edge": "audits.lip_edge", "reel": "audits.reel",
    # shared leaves
    "watermark": "watermark", "_models": "_models",
}

__all__ = ["__version__", *sorted(_API)]


def __getattr__(name):  # PEP 562: lazy module access, keeps `import lsdub_engine` cheap
    target = _API.get(name)
    if target is None:
        raise AttributeError(f"module 'lsdub_engine' has no attribute {name!r}")
    mod = importlib.import_module(f".{target}", __name__)
    globals()[name] = mod
    return mod


def __dir__():
    return sorted(__all__)
