import subprocess
import sys

from lsdub_engine import cli


def test_help_and_unknown(capsys):
    assert cli.main([]) == 0
    assert "lsdub align" in capsys.readouterr().out
    assert cli.main(["nope"]) == 2


def test_out_path():
    assert cli._out_path("paint", ["--video", "a", "--out", "o.mp4"]) == "o.mp4"
    assert cli._out_path("sing", ["--take", "a", "--out", "s.mp4"]) == "s.mp4"
    assert "refine" not in cli.CMDS and "hybrid" not in cli.CMDS
    assert cli._out_path("check", ["--video", "a"]) is None


def test_reel_is_watermarked(clip, tmp_path):
    out = str(tmp_path / "reel.mp4")
    r = subprocess.run([sys.executable, "-m", "lsdub_engine.cli", "reel", "--out", out, "--labels", "A", "B",
                        "--shot", f"s1|first|{clip}|{clip}", "--shot", f"s2|second|{clip}|{clip}", "--width", "960", "--height", "540"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    from lsdub_engine import watermark
    assert watermark.verify(out, frames=8)["watermark"]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                               capture_output=True, text=True).stdout)
    assert 4.3 < dur < 4.8                   # 2 + 0.5 gap + 2


def test_no_watermark_warns(clip, tmp_path):
    out = str(tmp_path / "reel.mp4")
    r = subprocess.run([sys.executable, "-m", "lsdub_engine.cli", "reel", "--out", out, "--shot", f"s1|x|{clip}", "--no-watermark",
                        "--width", "960", "--height", "540"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "WITHOUT the invisible watermark" in r.stderr
