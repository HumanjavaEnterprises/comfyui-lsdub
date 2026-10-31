# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""landmarker.py: the one place a MediaPipe FaceLandmarker is made. guard() runs before mediapipe loads, every instance is on the
CPU delegate. IMAGE mode is stateless: one shared instance per option set, callers do not close it. VIDEO mode keeps a timestamp
per instance, so each get() is a new one, and the caller closes it when its clip ends."""

import logging, sys

from .._models import face_landmarker_path
from .cpu_env import guard

log = logging.getLogger("lsdub")
_shared = {}
_warned = False


def mp():
    """the mediapipe module, the GL variables set first."""
    global _warned
    if "mediapipe" in sys.modules and not _warned:
        _warned = True
        log.warning("mediapipe was imported before the landmarker factory: its GL setup may already be fixed")
    guard()
    import mediapipe

    return mediapipe


def _make(video, conf, num_faces, matrices, tracking, presence, model):
    mp()
    from mediapipe.tasks.python import vision, BaseOptions

    kw = {"min_tracking_confidence": conf if tracking is None else tracking} if video else {}
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=num_faces,
            output_facial_transformation_matrixes=matrices,
            min_face_detection_confidence=conf,
            min_face_presence_confidence=conf if presence is None else presence,
            **kw,
        )
    )


def get(mode="image", conf=0.2, num_faces=1, matrices=False, tracking=None, presence=None, model=None):
    """mode 'image' | 'video'. tracking / presence: when they are not `conf`."""
    if mode not in ("image", "video"):
        raise ValueError(f"landmarker mode {mode!r}")
    model = model or face_landmarker_path()
    if mode == "video":
        return _make(True, conf, num_faces, matrices, tracking, presence, model)
    key = (conf, num_faces, matrices, presence, model)
    if key not in _shared:
        _shared[key] = _make(False, conf, num_faces, matrices, tracking, presence, model)
    return _shared[key]


def close_all():
    """close the shared instances (a host's model-unload)."""
    while _shared:
        _shared.popitem()[1].close()
