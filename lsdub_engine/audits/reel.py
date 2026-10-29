# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""reel.py — before/after reels: N takes of the same shot side by side, several shots one after another, a black gap
between shots, a caption per shot. Audio from --audio (or the first column's file).

  lsdub reel --out reel.mp4 --labels OLD NEW \\
      --shot "s01|first line|old_s01.mp4|new_s01.mp4" --shot "s02|second line|old_s02.mp4|new_s02.mp4" [--gap 0.5]
"""
import argparse
from ..io import ff
import os
import tempfile


def _run(*a):
    ff.run(["ffmpeg", "-v", "error", "-y", *a], check=True)


def _esc(t):
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def build(shots, out, labels, width=1920, height=1080, gap=0.5, fps=25, audio=None):
    with tempfile.TemporaryDirectory() as d:
        parts = []
        for k, (sid, caption, clips) in enumerate(shots):
            n = len(clips)
            cw = width // n
            ch = min(height - 120, int(cw * 9 / 16) if n > 1 else height - 120)
            ins, flt = [], []
            for i, c in enumerate(clips):
                ins += ["-i", c]
                lab = f",drawtext=text='{_esc(labels[i])}':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5" if labels and i < len(labels) else ""
                flt.append(f"[{i}:v]fps={fps},scale={cw}:{ch}:force_original_aspect_ratio=decrease,pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2{lab}[v{i}]")
            stack = "".join(f"[v{i}]" for i in range(n)) + (f"hstack=inputs={n}[row]" if n > 1 else "null[row]")
            cap = _esc(f"{sid}  {caption}")
            flt.append(stack + f";[row]pad={width}:{height}:(ow-iw)/2:(oh-ih)/2-40:black,"
                       f"drawtext=text='{cap}':x=(w-tw)/2:y=h-90:fontsize=36:fontcolor=white,format=yuv420p[v]")
            seg = f"{d}/s{k}.mp4"
            dur = _duration(clips[0])
            ins += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            src = audio or (clips[0] if _has_audio(clips[0]) else None)
            if src and src != clips[0]:
                ins += ["-i", src]
                aidx = n + 1
            else:
                aidx = 0 if src else n
            flt.append(f"[{aidx}:a]aresample=48000,aformat=channel_layouts=stereo[a]")
            _run(*ins, "-filter_complex", ";".join(flt), "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
                 "-c:v", "libx264", "-crf", "16", "-c:a", "aac", seg)
            parts.append(seg)
            if gap > 0 and k < len(shots) - 1:
                g = f"{d}/g{k}.mp4"
                _run("-f", "lavfi", "-i", f"color=black:s={width}x{height}:r={fps}:d={gap}", "-f", "lavfi", "-i",
                     f"anullsrc=r=48000:cl=stereo:d={gap}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", g)
                parts.append(g)
        lst = f"{d}/list.txt"
        open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
        _run("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", out)
    return out


def _duration(p):
    r = ff.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True)
    return float(r.stdout.strip())


def _has_audio(p):
    r = ff.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", p],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lsdub reel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shot", action="append", required=True, help='"id|caption|clip1.mp4|clip2.mp4|..." (repeatable)')
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args(argv)
    shots = []
    for s in a.shot:
        sid, cap, *clips = s.split("|")
        missing = [c for c in clips if not os.path.exists(c)]
        if missing:
            ap.error(f"missing clip(s): {missing}")
        shots.append((sid, cap, clips))
    build(shots, a.out, a.labels, a.width, a.height, a.gap, a.fps)
    print("->", a.out)


if __name__ == "__main__":
    main()
