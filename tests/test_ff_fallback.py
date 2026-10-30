# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""ffmpeg/ffprobe absent from PATH (a bare base image): ffmpeg falls back to the imageio-ffmpeg static binary and every
ffprobe call the pack makes is emulated over `ffmpeg -i`. The emulation must print what real ffprobe prints."""
import json
import os
import shutil
import subprocess

import pytest

from lsdub_engine.io import ff

import imageio_ffmpeg  # a declared runtime dependency: missing = a failure, not a skip
SYS_FFMPEG = shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
SYS_FFPROBE = shutil.which('ffprobe')


def _mk(cmd):
    subprocess.run([SYS_FFMPEG, '-v', 'error', '-y', *cmd], check=True)


@pytest.fixture(scope='module')
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp('ffmedia')
    av, ntsc, wav, tagged = (str(d / n) for n in ('av.mp4', 'ntsc.mp4', 'tone.wav', 'tagged.mp4'))
    _mk(['-f', 'lavfi', '-i', 'testsrc2=s=320x180:r=25:d=1', '-f', 'lavfi', '-i', 'sine=f=440:d=1', '-c:v', 'libx264',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', av])
    _mk(['-f', 'lavfi', '-i', 'testsrc2=s=200x120:r=30000/1001:d=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', ntsc])
    _mk(['-f', 'lavfi', '-i', 'sine=f=220:d=1.5', wav])
    _mk(['-i', ntsc, '-c', 'copy', '-metadata', 'comment=lsdub:wm=v1 key', tagged])
    return {'av': av, 'ntsc': ntsc, 'wav': wav, 'tagged': tagged, 'dir': str(d)}


# every ffprobe argv shape in the pack (media.py, nodes_mouth.py, watermark.py, audits/reel.py, audits/sync_strip.py)
PROBES = [
    ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0'],
    ['-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0'],
    ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0'],
    ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0'],
    ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json'],
    ['-v', 'error', '-show_entries', 'format_tags=comment', '-of', 'default=nw=1:nk=1'],
]


def _norm(of, out):
    if of == 'json': return json.loads(out)['streams']
    if of == 'csv=p=0' and out.strip() and out.strip()[0].isdigit() and '.' in out and ',' not in out:
        return round(float(out), 3)                                               # duration: to the millisecond
    return out.strip()


@pytest.mark.skipif(not SYS_FFPROBE, reason='needs a real ffprobe to compare against')
@pytest.mark.parametrize('args', PROBES, ids=lambda a: a[a.index('-show_entries') + 1])
@pytest.mark.parametrize('which', ['av', 'ntsc', 'wav', 'tagged'])
def test_emulation_matches_ffprobe(media, args, which):
    path = media[which]
    real = subprocess.run([SYS_FFPROBE, *args, path], capture_output=True, text=True)
    emu = ff.probe(['ffprobe', *args, path], capture_output=True, text=True)
    of = args[args.index('-of') + 1]
    assert emu.returncode == real.returncode
    assert _norm(of, emu.stdout) == _norm(of, real.stdout), (args, emu.stdout, real.stdout)


def test_emulation_missing_file_fails_like_ffprobe(tmp_path):
    r = ff.probe(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(tmp_path / 'nope.mp4')],
                 capture_output=True, text=True)
    assert r.returncode != 0 and r.stdout == ''
    with pytest.raises(subprocess.CalledProcessError):
        ff.probe(['ffprobe', '-show_entries', 'format=duration', str(tmp_path / 'nope.mp4')], capture_output=True, check=True)


@pytest.fixture
def no_ff_on_path(tmp_path, monkeypatch):
    """PATH with python but WITHOUT ffmpeg/ffprobe (a base image that ships neither)."""
    bare = tmp_path / 'bin'
    bare.mkdir()
    monkeypatch.setenv('PATH', str(bare))
    monkeypatch.delenv('IMAGEIO_FFMPEG_EXE', raising=False)
    assert shutil.which('ffmpeg') is None and shutil.which('ffprobe') is None


def test_ffmpeg_falls_back_to_imageio_binary(no_ff_on_path, tmp_path):
    exe = ff.which('ffmpeg')
    assert exe and os.path.isfile(exe) and 'imageio_ffmpeg' in exe
    out = str(tmp_path / 'x.mp4')
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '64x48', '-r', '25', '-i', '-',
                  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out], stdin=ff.PIPE)
    p.communicate(bytes(64 * 48 * 3 * 5))
    assert p.returncode == 0 and os.path.getsize(out) > 0
    assert ff.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', out],
                  capture_output=True, text=True, check=True).stdout.strip() == '0.200000'
    with pytest.raises(FileNotFoundError):
        ff.popen(['ffprobe', out])


def test_no_ffmpeg_at_all_raises_clearly(no_ff_on_path, monkeypatch):
    monkeypatch.setattr(imageio_ffmpeg, 'get_ffmpeg_exe', lambda: (_ for _ in ()).throw(RuntimeError('no binary')))
    with pytest.raises(FileNotFoundError, match='ffmpeg'):
        ff.run(['ffmpeg', '-version'])
    with pytest.raises(FileNotFoundError, match='ffmpeg'):
        ff.run(['ffprobe', '-show_entries', 'format=duration', 'x.mp4'])


def test_pack_probe_call_sites_without_ffprobe(no_ff_on_path, media, pack):
    """the real call sites, PATH stripped: media.duration, nodes_mouth._audio_has_stream, watermark._probe,
    watermark tag read, reel._duration/_has_audio, sync_strip._probe."""
    import importlib
    media_mod = importlib.import_module(pack.__name__ + '.media')
    mouth = importlib.import_module(pack.__name__ + '.nodes_mouth')
    from lsdub_engine import watermark
    from lsdub_engine.audits import reel, sync_strip
    assert media_mod.duration(media['wav']) == pytest.approx(1.5, abs=0.02)
    assert mouth._audio_has_stream(media['av']) and not mouth._audio_has_stream(media['ntsc'])
    w, h, fps = watermark._probe(media['ntsc'])
    assert (w, h) == (200, 120) and fps == pytest.approx(30000 / 1001)
    assert reel._duration(media['av']) == pytest.approx(1.0, abs=0.05)
    assert reel._has_audio(media['av']) and not reel._has_audio(media['wav'].replace('tone.wav', 'ntsc.mp4'))
    assert sync_strip._probe(media['av'], 'stream=width,height') == ['320', '180']
    assert sync_strip._probe(media['av'], 'stream=r_frame_rate') == ['25/1']
    tag = ff.run(['ffprobe', '-v', 'error', '-show_entries', 'format_tags=comment', '-of', 'default=nw=1:nk=1', media['tagged']],
                 capture_output=True, text=True).stdout.strip()
    assert tag == 'lsdub:wm=v1 key'


def test_audio_roundtrip_without_ffmpeg_on_path(no_ff_on_path, media, pack, tmp_path):
    """media.load_audio / save through the fallback binary (the AUDIO lane's encode + decode)."""
    import importlib
    media_mod = importlib.import_module(pack.__name__ + '.media')
    a = media_mod.load_audio(media['wav'])
    assert a['waveform'].shape[-1] == pytest.approx(1.5 * 44100, abs=2000)
    out = media_mod.write_audio(a, str(tmp_path / 'rt.wav'))
    assert media_mod.duration(out) == pytest.approx(1.5, abs=0.05)


@pytest.mark.skipif(not SYS_FFPROBE, reason='needs a real ffprobe to compare against')
def test_duration_to_the_millisecond(tmp_path):
    wav = str(tmp_path / 'odd.wav')
    _mk(['-f', 'lavfi', '-i', 'sine=f=220:d=1.234567', wav])
    real = float(subprocess.run([SYS_FFPROBE, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', wav],
                                capture_output=True, text=True).stdout)
    emu = float(ff.probe(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', wav],
                         capture_output=True, text=True).stdout)
    assert abs(emu - real) < 1e-4, (emu, real)
