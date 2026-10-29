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

    class LSDubExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [*nodes_project.NODES, *nodes_audio.NODES, *nodes_vocal.NODES, *nodes_video.NODES, *nodes_mouth.NODES, *nodes_qa.NODES]

    async def comfy_entrypoint() -> LSDubExtension:
        return LSDubExtension()
