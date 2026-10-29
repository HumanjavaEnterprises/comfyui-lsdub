# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""generate.py: run a stock generator graph (graphs.py) for an LSDub slot.

  local        (default) inside this ComfyUI: the graph runs in-process through ComfyUI's own executor (a nested
               PromptExecutor), on this machine's GPU, with the models in this ComfyUI's models/ folders.
  comfy_cloud  (opt-in) on Comfy Cloud through the Comfy Python SDK, signed in with the user's comfy.org API key
               (the node's hidden api_key_comfy_org, else COMFY_API_KEY). The key is never logged or saved.

Only the GPU generators can go to the cloud; the dub, the measurements and the QA always run locally."""
import os
import shutil
import uuid

BACKENDS = ('local', 'comfy_cloud')


class GenerationError(RuntimeError):
    pass


def _key(api_key=None):
    k = api_key or os.environ.get('COMFY_API_KEY')
    if not k:
        raise GenerationError('Comfy Cloud needs your comfy.org API key: sign in to ComfyUI with your Comfy account '
                              '(Settings > User), or set COMFY_API_KEY. Or switch the project backend to local.')
    return k


# ---------------------------------------------------------------------------------------------------- local
class _QuietServer:
    """the bits of the ComfyUI server a nested PromptExecutor talks to (no websocket messages)."""
    client_id = None
    last_node_id = None
    last_prompt_id = None

    def send_sync(self, *a, **k):
        pass

    def queue_updated(self, *a, **k):
        pass


def _stage_uploads(uploads):
    """local files -> names under ComfyUI's input/lsdub/ (LoadImage reads 'lsdub/<name>')."""
    import folder_paths
    d = os.path.join(folder_paths.get_input_directory(), 'lsdub')
    os.makedirs(d, exist_ok=True)
    names = {}
    for key, path in (uploads or {}).items():
        name = f'{uuid.uuid4().hex[:12]}_{os.path.basename(path)}'
        shutil.copyfile(path, os.path.join(d, name))
        names[key] = f'lsdub/{name}'
    return names


def _cache_args():
    """the cache settings ComfyUI's own executor gets (main.py), from its command-line args and the machine's RAM."""
    try:
        import comfy.model_management as mm
        from comfy.cli_args import args
        ram = min(10.0, max(2.0, mm.total_ram * 0.10 / 1024.0))
        inactive = min(128.0, mm.total_ram / 1024.0)
        cr = list(getattr(args, 'cache_ram', []) or [])
        if len(cr) > 0: ram = cr[0]
        if len(cr) > 1: inactive = cr[1]
        return {'lru': getattr(args, 'cache_lru', 0) or 0, 'ram': ram, 'ram_inactive': inactive}
    except Exception:
        return {'lru': 0, 'ram': 4.0, 'ram_inactive': 16.0}


async def run_local(graph, out_node='out', uploads=None, out_dir='.'):
    import execution
    import folder_paths
    names = _stage_uploads(uploads)
    graph = _fill(graph, names)
    pid = f'lsdub-{uuid.uuid4().hex}'
    valid = await execution.validate_prompt(pid, graph, None)
    if not valid[0]:
        raise GenerationError(f'generator graph rejected: {valid[1]} {valid[3]}')
    ex = execution.PromptExecutor(_QuietServer(), cache_type=execution.CacheType.NONE, cache_args=_cache_args())
    await ex.execute_async(graph, pid, {}, [out_node])
    if not ex.success:
        msgs = [m for m in ex.status_messages if m[0] == 'execution_error']
        raise GenerationError(f'generator failed: {msgs[-1][1].get("exception_message") if msgs else "unknown error"}')
    rows = []
    for v in (ex.history_result or {}).get('outputs', {}).get(out_node, {}).values():
        if isinstance(v, list): rows += [r for r in v if isinstance(r, dict) and 'filename' in r]
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for r in rows:
        base = folder_paths.get_output_directory() if r.get('type', 'output') == 'output' else folder_paths.get_temp_directory()
        src = os.path.join(base, r.get('subfolder', ''), r['filename'])
        dst = os.path.join(out_dir, r['filename'])
        shutil.copyfile(src, dst)
        paths.append(dst)
    return paths


def _fill(graph, names):
    """replace upload placeholders ('@key') in LoadImage inputs with the staged names."""
    out = {}
    for nid, n in graph.items():
        inp = {k: (names.get(v[1:], v) if isinstance(v, str) and v.startswith('@') else v) for k, v in n['inputs'].items()}
        out[nid] = {**n, 'inputs': inp}
    return out


# ---------------------------------------------------------------------------------------------------- cloud
def run_cloud(graph, out_node='out', uploads=None, out_dir='.', api_key=None, client_factory=None):
    """the same graph on Comfy Cloud. uploads {key: path} become cloud assets set on the '@key' inputs."""
    key = _key(api_key)
    if client_factory is None:
        try:
            from comfy_sdk import Comfy
        except ImportError as e:
            raise GenerationError('the Comfy Cloud backend needs the Comfy Python SDK (the comfy-sdk package)') from e
        client_factory = Comfy
    c = client_factory(api_key=key)
    try:
        return _run_cloud_job(c, graph, out_node, uploads, out_dir)
    except GenerationError:
        raise
    except Exception as e:                                              # the SDK's errors, in plain words (never the key)
        raise GenerationError(cloud_error_message(e)) from None
    finally:
        c.close()


PAID_PLAN = ('Comfy Cloud refused the run: running workflows with an API key needs a paid Comfy Cloud plan (free-tier '
             'accounts cannot use API keys). Upgrade your Comfy Cloud plan, or set the project backend to local.')


def cloud_error_message(e):
    name, msg = type(e).__name__, str(e)
    low = msg.lower()
    if 'free tier' in low or 'free-tier' in low or name == 'Forbidden': return PAID_PLAN
    if name in ('Unauthorized', 'MissingApiKey') or 'unauthorized' in low:
        return 'Comfy Cloud did not accept the comfy.org API key: sign in again in ComfyUI, or check COMFY_API_KEY.'
    if name == 'QueueFull': return 'Comfy Cloud is busy (queue full): try again in a few minutes, or use the local backend.'
    if name == 'JobFailed': return f'the generator failed on Comfy Cloud: {msg[:300]}'
    return f'Comfy Cloud error ({name}): {msg[:300]}'


def _run_cloud_job(c, graph, out_node, uploads, out_dir):
    wf = c.workflows.from_json(graph)
    for nid, n in graph.items():
        for field, v in n['inputs'].items():
            if isinstance(v, str) and v.startswith('@'):
                path = (uploads or {}).get(v[1:])
                if path is None: raise GenerationError(f'no file for {v}')
                wf.set_input(nid, field, c.assets.from_file(path))
    job = c.run(wf)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for o in job.get_outputs(out_node):
        dst = os.path.join(out_dir, os.path.basename(o.name))
        o.to_file(dst)
        paths.append(dst)
    if not paths: raise GenerationError('Comfy Cloud returned no output')
    return paths


async def run(graph, backend='local', uploads=None, out_dir='.', api_key=None, out_node='out'):
    if backend == 'comfy_cloud':
        import asyncio
        from .graphs import for_backend
        graph = for_backend(graph, backend)
        return await asyncio.to_thread(run_cloud, graph, out_node, uploads, out_dir, api_key)
    if backend != 'local': raise GenerationError(f'unknown backend {backend!r} (have {BACKENDS})')
    return await run_local(graph, out_node, uploads, out_dir)
