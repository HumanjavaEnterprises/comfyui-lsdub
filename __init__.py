# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""LSDub for ComfyUI: lip-sync for music videos (V3 nodes). Heavy libraries load when a node runs, not at import;
nothing is downloaded at startup."""
import os

if __package__:   # loaded as a package (ComfyUI, tests/conftest.py); pytest's bare import of this file is a no-op
    import folder_paths
    from comfy_api.v0_0_2 import ComfyExtension, io

    from . import nodes_audio, nodes_mouth, nodes_project, nodes_qa, nodes_video, nodes_vocal

    folder_paths.add_model_folder_path('lsdub', os.path.join(folder_paths.models_dir, 'lsdub'))
    from .lsdub_engine import _models
    if not os.environ.get('LIPS_CACHE'): _models.set_cache(os.path.join(folder_paths.models_dir, 'lsdub'))   # config only

    def _quiet_fingerprint(cls):
        """ComfyUI may ask for a fingerprint before the upstream records exist (inputs still None): answer 'pending'
        instead of raising, so the log stays clean; the run itself still waits for, and keys on, the real inputs."""
        f = cls.__dict__.get('fingerprint_inputs')
        if f is None or getattr(f, '_lsdub_quiet', False): return cls
        orig = f.__func__

        def fingerprint_inputs(c, *a, **kw):
            if any(v is None for k, v in kw.items() if k in ('project', 'source', 'shots', 'song', 'align', 'render', 'kit')):
                return 'lsdub:pending'
            try:
                return orig(c, *a, **kw)
            except (AttributeError, TypeError):
                return 'lsdub:pending'
        fingerprint_inputs._lsdub_quiet = True
        cls.fingerprint_inputs = classmethod(fingerprint_inputs)
        return cls

    for _n in (*nodes_project.NODES, *nodes_audio.NODES, *nodes_vocal.NODES, *nodes_video.NODES, *nodes_mouth.NODES, *nodes_qa.NODES):
        _quiet_fingerprint(_n)

    class LSDubExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [*nodes_project.NODES, *nodes_audio.NODES, *nodes_vocal.NODES, *nodes_video.NODES, *nodes_mouth.NODES, *nodes_qa.NODES]

    async def comfy_entrypoint() -> LSDubExtension:
        return LSDubExtension()
