import subprocess

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
