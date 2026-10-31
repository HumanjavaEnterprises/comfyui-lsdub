# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the vocal checkpoint default: LIPS_VOCAL_CKPT, else LIPS_CACHE/audio_separation; never a fixed install path."""

import os, subprocess, sys

CODE = "import lsdub_engine.core.vocal_split as v; print(v.DEFAULT_CKPT)"
NAME = "MelBandRoformer_kim_vocals.ckpt"


def _default(**env):
    e = {k: v for k, v in os.environ.items() if k not in ("LIPS_VOCAL_CKPT", "LIPS_CACHE")} | env
    return subprocess.run([sys.executable, "-c", CODE], env=e, capture_output=True, text=True, check=True).stdout.strip()


def test_env_wins():
    assert _default(LIPS_VOCAL_CKPT="/x/v.ckpt", LIPS_CACHE="/c") == "/x/v.ckpt"


def test_cache_is_the_fallback():
    assert _default(LIPS_CACHE="/c") == os.path.join("/c", "audio_separation", NAME)


def test_default_is_the_user_cache():
    assert _default() == os.path.expanduser(os.path.join("~/.cache/lsdub", "audio_separation", NAME))
