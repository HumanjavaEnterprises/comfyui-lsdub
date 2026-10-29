# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Every node: a unique LSDub-prefixed id, an LSDub category, a description, a tooltip on every input and output."""
import asyncio


def _nodes(pack):
    return asyncio.run(asyncio.run(pack.comfy_entrypoint()).get_node_list())


def test_entrypoint_lists_nodes(pack):
    ids = [n.define_schema().node_id for n in _nodes(pack)]
    assert {'LSDubProject', 'LSDubLyricsSlot', 'LSDubVocalSplit', 'LSDubLeadVocal', 'LSDubSongSlot', 'LSDubAlign',
            'LSDubVideoSlot', 'LSDubShotSplit', 'LSDubCast', 'LSDubTriage'} <= set(ids)
    assert len(ids) == len(set(ids))


def test_every_port_has_a_tooltip(pack):
    for n in _nodes(pack):
        s = n.define_schema()
        assert s.node_id.startswith('LSDub') and s.category.split('/')[0] == 'LSDub' and s.description, s.node_id
        assert s.display_name and s.display_name.startswith('LSDub'), s.node_id
        for p in s.inputs + s.outputs:
            assert p.tooltip and len(p.tooltip) > 8, (s.node_id, p.id or p.display_name)


def test_models_folders_registered(pack):
    import folder_paths
    assert 'lsdub' in folder_paths._extra and 'audio_separation' in folder_paths._extra
