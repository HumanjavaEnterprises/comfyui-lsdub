# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""media.py: the edges between ComfyUI's AUDIO / VIDEO / IMAGE and files in the project folder. Heavy imports are lazy."""
import hashlib
import json
import os
import shutil

from .lsdub_engine.io import ff
from .project import file_hash


def audio_hash(audio):
    """sha1 of a ComfyUI AUDIO's samples + rate (the cache key of an audio asset)."""
    w = audio['waveform']
    a = w.detach().float().cpu().contiguous().numpy() if hasattr(w, 'detach') else w
    h = hashlib.sha1(str(int(audio['sample_rate'])).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_audio(audio, path):
    """AUDIO {waveform (B, C, N), sample_rate} -> a float32 WAV (first batch item), through ffmpeg."""
    import numpy as np
    w = audio['waveform'][0]
    x = (w.detach().float().cpu().numpy() if hasattr(w, 'detach') else np.asarray(w, np.float32)).astype(np.float32)
    c = x.shape[0]
    tmp = path + '.part.wav'
    ff.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(int(audio['sample_rate'])), '-ac', str(c), '-i', '-',
            '-c:a', 'pcm_f32le', tmp], input=np.ascontiguousarray(x.T).tobytes(), check=True)
    os.replace(tmp, path)
    return path


def duration(path):
    r = ff.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path], capture_output=True, text=True)
    return float(r.stdout.strip())


def video_source(video):
    """a ComfyUI VIDEO -> (path on disk or None, bytes or None)."""
    src = video.get_stream_source() if hasattr(video, 'get_stream_source') else video
    if isinstance(src, (str, os.PathLike)): return os.fspath(src), None
    if hasattr(src, 'getvalue'): return None, src.getvalue()
    raise TypeError(f'unsupported VIDEO source {type(src).__name__}')


def store_video(video, dst):
    """copy a VIDEO asset to dst; -> its sha1."""
    path, data = video_source(video)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if path is not None:
        if os.path.abspath(path) != os.path.abspath(dst): shutil.copyfile(path, dst)
    else:
        with open(dst, 'wb') as f: f.write(data)
    return file_hash(dst)


def probe_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return float(fps), n, wh


def image_to_png(image, path):
    """IMAGE (B, H, W, 3) float 0..1 -> the first frame as PNG."""
    import cv2
    import numpy as np
    a = image[0].detach().cpu().numpy() if hasattr(image[0], 'detach') else np.asarray(image[0])
    cv2.imwrite(path, cv2.cvtColor((np.clip(a, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
    return path


def load_image(path):
    """a picture on disk -> IMAGE (1, H, W, 3) float 0..1 (for a preview output)."""
    import cv2
    import numpy as np
    im = cv2.imread(path, cv2.IMREAD_COLOR) if path else None
    a = np.zeros((1, 64, 64, 3), np.float32) if im is None else (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)[None]
    try:
        import torch
    except ImportError:                                                  # tests without torch
        return a
    return torch.from_numpy(a)


def dump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)
    return path


def load_audio(path, sr=44100):
    """a file -> ComfyUI AUDIO (stereo float32 at sr), through ffmpeg."""
    import numpy as np
    raw = ff.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '2', '-ar', str(sr), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()
    try:
        import torch
        return {'waveform': torch.from_numpy(x)[None], 'sample_rate': sr}
    except ImportError:
        return {'waveform': x[None], 'sample_rate': sr}


def to_wav(src, dst):
    ff.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-c:a', 'pcm_f32le', dst], check=True)
    return dst


def input_dir():
    """ComfyUI's input folder (None outside ComfyUI)."""
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except (ImportError, AttributeError):
        return None


ALLOW_ABS_ENV = 'LSDUB_ALLOW_ABS_INPUTS'   # =1: absolute input paths inside ComfyUI too (local use; never on a deployment)
ZIP_MAX_BYTES, ZIP_MAX_MEMBERS, ZIP_MAX_RATIO = 2 << 30, 20000, 200      # a kit is ~10 MB, ~100 files


def _within(path, root):
    rp = os.path.realpath(path)
    return rp == root or rp.startswith(root + os.sep)


def _clean(p):
    p = p.strip()
    return p[:-len(' [input]')] if p.endswith(' [input]') else p


def _input_candidates(p, what):
    """where a string input may live. Inside ComfyUI: under its input folder only (a relative path that escapes it is
    refused; an absolute path only with LSDUB_ALLOW_ABS_INPUTS=1 or when it is inside the input folder). Outside
    ComfyUI (the CLI, tests): as given, relative to the working folder."""
    d = input_dir()
    if d is None: return [p if os.path.isabs(p) else os.path.abspath(p)]
    root = os.path.realpath(d)
    if os.path.isabs(p):
        if os.environ.get(ALLOW_ABS_ENV) == '1' or _within(p, root): return [p]
        raise PermissionError(f'{what}: absolute path {p!r} refused; name a file in ComfyUI\'s input folder '
                              f'(or set {ALLOW_ABS_ENV}=1 for local use)')
    first = os.path.join(d, p)
    if not _within(first, root): raise PermissionError(f'{what}: {p!r} escapes ComfyUI\'s input folder; refusing')
    return [first, os.path.join(d, os.path.basename(p))]


def input_file(p, what='file'):
    """a file input given as a string, the way LoadVideo/LoadAudio name theirs: relative to ComfyUI's input folder
    (`comfy deploy run` uploads every string input with a file extension that names a local file, and the worker
    rewrites it to the uploaded file's name, optionally `subfolder/name`, in its input folder). ComfyUI's ' [input]'
    annotation is accepted. '' -> ''. Raises FileNotFoundError (naming where it looked) when the file is not there,
    PermissionError when the path leaves the input folder."""
    if not p: return p
    p = _clean(p)
    tried = _input_candidates(p, what)
    for q in tried:
        if os.path.isfile(q): return q
    raise FileNotFoundError(f'{what} {p!r} not found (looked in: {", ".join(dict.fromkeys(tried))})')


def _safe_extract(zf, dst):
    """extract after checking every member: no symlinks, nothing outside dst (also Windows `..\\` / drive forms), and
    zip-bomb caps on total size, member count and compression ratio."""
    root = os.path.realpath(dst)
    ms = zf.infolist()
    if len(ms) > ZIP_MAX_MEMBERS: raise ValueError(f'zip has {len(ms)} members (max {ZIP_MAX_MEMBERS}); refusing')
    total = 0
    for m in ms:
        n = m.filename
        if (m.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f'zip member {n!r} is a symlink; refusing')
        if '\\' in n or (len(n) > 1 and n[1] == ':'):
            raise ValueError(f'zip member {n!r} escapes the folder; refusing')
        out = os.path.realpath(os.path.join(dst, n))
        if out != root and not out.startswith(root + os.sep):
            raise ValueError(f'zip member {n!r} escapes the folder; refusing')
        total += m.file_size
        if m.file_size > (1 << 20) and m.file_size / max(m.compress_size, 1) > ZIP_MAX_RATIO:
            raise ValueError(f'zip member {n!r} compresses {m.file_size / max(m.compress_size, 1):.0f}:1; refusing')
    if total > ZIP_MAX_BYTES: raise ValueError(f'zip unpacks to {total} bytes (max {ZIP_MAX_BYTES}); refusing')
    zf.extractall(dst)


def kit_folder(p):
    """a lip-kit input: a folder, or a .zip of one (a single file, so `comfy deploy run` uploads it like any LoadImage
    input; a folder never travels). Same containment as input_file. A zip is unpacked once into
    <input>/lsdub_kits_unpacked/<sha1>/ and the folder holding library.json (or base.png) inside it is returned."""
    import tempfile
    import zipfile
    q = _clean(p)
    if not q.lower().endswith('.zip'): return _input_candidates(q, 'LSDub Lip Kit: kit_dir')[0]
    z = input_file(q, 'LSDub Lip Kit: kit zip')
    d = input_dir()
    h = hashlib.sha1()
    with open(z, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    dst = os.path.join(d or os.path.dirname(z), 'lsdub_kits_unpacked', h.hexdigest()[:16])
    if not os.path.isdir(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = tempfile.mkdtemp(prefix='.unpack-', dir=os.path.dirname(dst))
        try:
            with zipfile.ZipFile(z) as zf: _safe_extract(zf, tmp)
            try: os.replace(tmp, dst)
            except OSError:                                              # another run unpacked it first
                if not os.path.isdir(dst): raise
        finally:
            if os.path.isdir(tmp): shutil.rmtree(tmp, ignore_errors=True)
    for r, _, fs in sorted(os.walk(dst)):
        if 'library.json' in fs or 'base.png' in fs: return r
    raise FileNotFoundError(f'LSDub Lip Kit: {q!r} holds no kit (no library.json or base.png inside)')


def _saved(path):
    """a file under ComfyUI's output folder -> {filename, subfolder, type: 'output'} (what /view and the job outputs
    serve); None outside it (a project folder set elsewhere) or outside ComfyUI."""
    try:
        import folder_paths
        root = os.path.realpath(folder_paths.get_output_directory())
    except (ImportError, AttributeError):
        return None
    rp = os.path.realpath(path)
    if not os.path.isfile(rp) or not rp.startswith(root + os.sep): return None
    sub = os.path.relpath(os.path.dirname(rp), root)
    return {'filename': os.path.basename(rp), 'subfolder': '' if sub == '.' else sub.replace(os.sep, '/'), 'type': 'output'}


def ui_outputs(videos=(), images=(), files=()):
    """the node's ui result, so the files show in ComfyUI's history and come back as `comfy deploy run` job outputs:
    videos as core SaveVideo reports them ('images' + animated), stills under 'images', reports (md/json) under
    'files'. A node reports either videos or stills. None when nothing is under the output folder."""
    ui = {}
    vs = [r for r in map(_saved, videos) if r]
    ims = [r for r in map(_saved, images) if r]
    fs = [r for r in map(_saved, files) if r]
    if vs: ui.update(images=vs, animated=(True,))
    elif ims: ui['images'] = ims
    if fs: ui['files'] = fs
    return ui or None


def hidden_api_key(cls):
    """the comfy.org API key ComfyUI hands a node that declares Hidden.api_key_comfy_org (None outside ComfyUI)."""
    h = getattr(cls, 'hidden', None)
    return getattr(h, 'api_key_comfy_org', None) if h is not None else None


def placeholder_video(path, text, size=(640, 360), seconds=1.0, fps=25.0):
    """a short black clip with a line of text: what a Video output returns when there is nothing to show, so
    downstream preview / save nodes never receive None."""
    import cv2
    import numpy as np
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        f = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.putText(f, text[:60], (20, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        tmp = path + '.part.mp4'
        p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
                      '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', tmp], stdin=ff.PIPE)
        try:
            for _ in range(max(1, int(seconds * fps))): p.stdin.write(f.tobytes())
            p.stdin.close()
        except BrokenPipeError:
            pass
        if p.wait() != 0: raise RuntimeError('ffmpeg failed writing a placeholder clip')
        os.replace(tmp, path)
    return path
