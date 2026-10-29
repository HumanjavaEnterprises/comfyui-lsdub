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
