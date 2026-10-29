# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
import subprocess

import pytest


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """2 s 640x360 test clip with texture + a tone (no faces, nothing copyrighted)."""
    d = tmp_path_factory.mktemp("clips")
    p = d / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=2",
                    "-f", "lavfi", "-i", "sine=f=440:d=2", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(p)], check=True)
    return str(p)
