# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
import subprocess

import pytest

from lsdub_engine import watermark


def test_roundtrip(clip, tmp_path):
    out = str(tmp_path / "wm.mp4")
    assert watermark.apply(clip, out) == 50
    r = watermark.verify(out, frames=8)
    assert r["watermark"] and r["z_mean"] > 8
    assert "lsdub" in (r["metadata_tag"] or "")


def test_unmarked_is_negative(clip):
    r = watermark.verify(clip, frames=8)
    assert not r["watermark"] and abs(r["z_mean"]) < 4


def test_survives_rescale_and_reencode(clip, tmp_path):
    out = str(tmp_path / "wm.mp4")
    small = str(tmp_path / "small.mp4")
    watermark.apply(clip, out)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out, "-vf", "scale=480:-2", "-c:v", "libx264", "-crf", "26", small], check=True)
    assert watermark.verify(small, frames=8)["watermark"]


def test_wrong_key_is_negative(clip, tmp_path):
    out = str(tmp_path / "wm.mp4")
    watermark.apply(clip, out)
    assert not watermark.verify(out, key="someone-else", frames=8)["watermark"]


@pytest.mark.parametrize("audio_s", [0.8, 1.0, 1.6])
def test_every_video_frame_survives_whatever_the_audio(tmp_path, audio_s):
    """25 frames, audio short of them, level with them, past them: the output keeps 25 and the audio stops by the last one."""
    src, dst = str(tmp_path / "s.mp4"), str(tmp_path / "d.mp4")
    watermark.ff.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=25:d=1", "-f", "lavfi", "-i", f"sine=f=440:d={audio_s}"]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", src],
        check=True,
    )
    watermark.apply(src, dst)
    assert watermark._frames(dst) == 25
    probe = watermark.ff.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "csv=p=0", dst],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert float(probe.split(",")[0]) <= 1.05


def _top_atoms(path):
    """the mp4's top-level box types, in file order."""
    import struct

    out = []
    with open(path, "rb") as f:
        while True:
            hd = f.read(8)
            if len(hd) < 8:
                return out
            size, kind = struct.unpack(">I4s", hd)
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0] - 8
            out.append(kind.decode("latin-1"))
            f.seek(size - 8, 1)


def test_the_watermarked_file_is_faststart(clip, tmp_path):
    """moov ahead of mdat (the saved cut, the site's redub.mp4, was the one file with its moov at the end)."""
    out = str(tmp_path / "wm.mp4")
    watermark.apply(clip, out)
    a = _top_atoms(out)
    assert "moov" in a and "mdat" in a and a.index("moov") < a.index("mdat"), a
