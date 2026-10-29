#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""sync_strip.py — the lip-sync QA strip: the vocal's waveform scrolls under the video past a fixed playhead.

Each syllable crosses the centre line the moment it is heard, so lips vs voice can be judged by eye and ear
(a mouth that opens before its blob reaches the line is early; after, late). A toggle beside the mouth card:

  lsdub strip --video shot.mp4 --out shot_strip.mp4 [--audio vocal.wav --t0 6.17] [--px-per-s 500]
  lsdub paint ... --sync-strip [--sync-strip-audio vocal.wav]     -> <out>_strip.mp4 (on the annotated video if --annotate)

Without --audio the video's own audio track is drawn. The output carries that audio.
"""
import argparse, os, tempfile
from ..io import ff


def _probe(path, entries, stream=True):
    r = ff.run(['ffprobe', '-v', 'error'] + (['-select_streams', 'v:0'] if stream else []) +
                       ['-show_entries', entries, '-of', 'csv=p=0', path], capture_output=True, text=True, check=True)
    return r.stdout.strip().split('\n')[0].split(',')


def render(video, out, audio=None, t0=0.0, px_per_s=500, height=200, color='0x40e0ff'):
    W, H = map(int, _probe(video, 'stream=width,height')[:2])
    fps = _probe(video, 'stream=r_frame_rate')[0]
    dur = float(_probe(video, 'format=duration', stream=False)[0])
    with tempfile.TemporaryDirectory() as T:
        wav = os.path.join(T, 'a.wav')
        src = ['-ss', f'{t0:.3f}', '-t', f'{dur:.3f}', '-i', audio] if audio else ['-i', video]
        ff.run(['ffmpeg', '-v', 'error', '-y'] + src + ['-vn', '-ac', '1', '-ar', '44100', wav], check=True)
        png = os.path.join(T, 'w.png')
        pw = int((dur + 2) * px_per_s)
        ff.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-filter_complex',
                        f'adelay=1000:all=1,apad=pad_dur=1,showwavespic=s={pw}x{height}:colors={color}:scale=sqrt',
                        '-frames:v', '1', png], check=True)
        fc = (f'color=c=0x101418:s={W}x{height}:r={fps}[bg];'
              f"[bg][1:v]overlay=x='{W}/2-(t+1)*{px_per_s}':y=0:shortest=1[w];"
              f'[w]drawbox=x={W // 2 - 1}:y=0:w=3:h={height}:color=white@0.9:t=fill[wp];'
              f'[0:v][wp]vstack=shortest=1[v]')
        ff.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-loop', '1', '-i', png, '-i', wav,
                        '-filter_complex', fc, '-map', '[v]', '-map', '2:a', '-c:v', 'libx264', '-crf', '18',
                        '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
    return out


def strip_path(out):
    return os.path.splitext(out)[0] + '_strip.mp4'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--audio', help='vocal to draw (default: the video\'s own audio)')
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--px-per-s', type=int, default=500)
    ap.add_argument('--height', type=int, default=200)
    a = ap.parse_args(argv)
    print(render(a.video, a.out, a.audio, a.t0, a.px_per_s, a.height))


if __name__ == '__main__':
    main()
