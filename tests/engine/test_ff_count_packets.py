# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the ffprobe emulation counts packets (`-count_packets`, stream=nb_read_packets) as ffprobe does, so the watermark's frame
count works on a host with ffmpeg and no ffprobe (Comfy Cloud's image)."""

import shutil
import subprocess

import pytest

from lsdub_engine.io import ff

FFMPEG = ff.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
ARGS = ["-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"]


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("packets")
    out = {}
    for name, src, extra in (
        ("plain", "testsrc2=s=200x120:r=25:d=1", []),
        ("bframes", "testsrc2=s=320x180:r=30000/1001:d=3", ["-bf", "2"]),
    ):
        p = str(d / f"{name}.mp4")
        subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                src,
                "-f",
                "lavfi",
                "-i",
                "sine=d=1",
                "-c:v",
                "libx264",
                *extra,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                p,
            ],
            check=True,
        )
        out[name] = p
    return out


@pytest.mark.skipif(not FFPROBE, reason="needs a real ffprobe to compare against")
@pytest.mark.parametrize("name", ["plain", "bframes"])
def test_emulated_packet_count_matches_ffprobe(clips, name):
    real = subprocess.run([FFPROBE, *ARGS, clips[name]], capture_output=True, text=True)
    emu = ff.probe(["ffprobe", *ARGS, clips[name]], capture_output=True, text=True)
    assert emu.returncode == real.returncode == 0
    assert emu.stdout.strip() == real.stdout.strip() != ""


def test_watermark_frames_without_ffprobe(clips, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from lsdub_engine import watermark

    real_which = shutil.which
    monkeypatch.setattr(ff.shutil, "which", lambda n, *a, **k: None if n == "ffprobe" else real_which(n, *a, **k))
    assert ff._emulate_probe(["ffprobe"])
    cap = cv2.VideoCapture(clips["plain"])
    n = 0
    while cap.read()[0]:
        n += 1
    assert watermark._frames(clips["plain"]) == n == 25


def test_watermark_frames_when_ffprobe_prints_nothing(clips, monkeypatch):
    """a hosted image whose ffprobe answers -count_packets with an empty line (LiveTry on Comfy Cloud): count with ffmpeg."""
    cv2 = pytest.importorskip("cv2")
    from lsdub_engine import watermark

    real_run = ff.run

    def run(argv, **kw):
        if argv and argv[0] == "ffprobe" and "-count_packets" in argv:
            return subprocess.CompletedProcess(argv, 0, "\n", "")
        return real_run(argv, **kw)

    monkeypatch.setattr(watermark.ff, "run", run)
    for name in ("plain", "bframes"):
        cap, n = cv2.VideoCapture(clips[name]), 0
        while cap.read()[0]:
            n += 1
        assert watermark._frames(clips[name]) == n > 0


def test_watermark_frames_when_ffprobe_fails(clips, monkeypatch):
    pytest.importorskip("cv2")
    from lsdub_engine import watermark

    real_run = ff.run

    def run(argv, **kw):
        if argv and argv[0] == "ffprobe" and "-count_packets" in argv:
            raise ff.CalledProcessError(1, argv, "", "boom")
        return real_run(argv, **kw)

    monkeypatch.setattr(watermark.ff, "run", run)
    assert watermark._frames(clips["plain"]) == 25
