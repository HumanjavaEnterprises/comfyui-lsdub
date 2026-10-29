# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Test harness: stub the ComfyUI modules the pack imports (folder_paths, comfy.model_management, comfy.utils,
comfy_api.v0_0_2) and load the pack as the package `lsdub_pack`, the way ComfyUI loads a custom-node folder.
No GPU, no ComfyUI install needed."""
import importlib.util
import os
import sys
import types

import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                                   # lsdub_engine as a top-level package too
os.environ['PYTHONPATH'] = ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')   # for `python -m lsdub_engine.cli`


# ---------------------------------------------------------------------------------------------------- stubs
def _io_stub():
    io = types.ModuleType('comfy_api.v0_0_2.io')

    class _Port:
        def __init__(self, id=None, display_name=None, optional=False, tooltip=None, **kw):
            self.id, self.display_name, self.optional, self.tooltip, self.kw = id, display_name, optional, tooltip, kw

    def _type(name):
        T = type(name, (), {'io_type': name})
        T.Input = type('Input', (_Port,), {'Parent': T, 'io_type': name})
        T.Output = type('Output', (_Port,), {'Parent': T, 'io_type': name})
        return T

    for n in ('String', 'Int', 'Float', 'Boolean', 'Combo', 'Audio', 'Image', 'Video', 'Mask'):
        setattr(io, n, _type(n.upper()))
    io.Custom = lambda name: _type(name)

    class Schema:
        def __init__(self, node_id, display_name=None, category='sd', inputs=(), outputs=(), description='', **kw):
            self.node_id, self.display_name, self.category = node_id, display_name, category
            self.inputs, self.outputs, self.description, self.kw = list(inputs), list(outputs), description, kw

    class NodeOutput:
        def __init__(self, *args, ui=None, **kw):
            self.args, self.result, self.ui = args, args, ui

    class ComfyNode:
        @classmethod
        def define_schema(cls): raise NotImplementedError

    io.Schema, io.NodeOutput, io.ComfyNode = Schema, NodeOutput, ComfyNode
    io.Hidden = types.SimpleNamespace(api_key_comfy_org='API_KEY_COMFY_ORG', unique_id='UNIQUE_ID')
    return io


def install_stubs(tmp_root):
    fp = types.ModuleType('folder_paths')
    fp.models_dir = os.path.join(tmp_root, 'models')
    fp._extra = {}
    fp.add_model_folder_path = lambda kind, path, is_default=False: fp._extra.setdefault(kind, []).append(path)
    fp.get_output_directory = lambda: os.path.join(tmp_root, 'output')
    fp.get_filename_list = lambda kind: []
    fp.get_full_path = lambda kind, name: os.path.join(fp.models_dir, kind, name)
    comfy = types.ModuleType('comfy')
    mm = types.ModuleType('comfy.model_management')
    mm.get_torch_device = lambda: 'cpu'
    cu = types.ModuleType('comfy.utils')

    class ProgressBar:
        def __init__(self, total): self.total, self.n = total, 0
        def update(self, k): self.n += k
        def update_absolute(self, v, total=None, preview=None): self.n = v
    cu.ProgressBar = ProgressBar
    comfy.model_management, comfy.utils = mm, cu
    api = types.ModuleType('comfy_api')
    v = types.ModuleType('comfy_api.v0_0_2')
    v.io = _io_stub()

    class ComfyExtension:
        async def get_node_list(self): return []
    v.ComfyExtension = ComfyExtension
    api.v0_0_2 = v
    sys.modules.update({'folder_paths': fp, 'comfy': comfy, 'comfy.model_management': mm, 'comfy.utils': cu,
                        'comfy_api': api, 'comfy_api.v0_0_2': v, 'comfy_api.v0_0_2.io': v.io})
    return fp


def load_pack(name='lsdub_pack'):
    if name in sys.modules: return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, '__init__.py'), submodule_search_locations=[ROOT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_TMP = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'lsdub-tests-{os.getpid()}')
install_stubs(_TMP)


@pytest.fixture(scope='session')
def pack():
    p = load_pack()
    import importlib
    importlib.import_module(p.__name__ + '.devices').ensure_face_models = lambda: None    # no downloads in tests
    return p


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """2 s 640x360 test clip with texture + a tone (no faces, nothing copyrighted)."""
    d = tmp_path_factory.mktemp("clips")
    p = d / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=2",
                    "-f", "lavfi", "-i", "sine=f=440:d=2", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(p)], check=True)
    return str(p)
