# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""DINOv2 and Whisper load by local PATH when their files are in models/lsdub, so a worker with no network (or no
refs/main, which a build that places only snapshots/<rev>/ files never writes) starts without a hub call.
The real-weights check runs when LSDUB_HF_MODELS names a models/lsdub folder laid out as the build spec places it."""

import os
import subprocess
import sys
import textwrap

import pytest

from lsdub_engine._models import hf_snapshot
from lsdub_engine.core.lyric_align import whisper_source

DINO, WHISPER = "facebook/dinov2-base", "Systran/faster-whisper-large-v3"
REV = "aaaa444c814b77203eaa57a6bdbbd535f21ede14"  # an unpinned snapshot: the fake files here are not the pinned bytes


def _files(d, names):
    os.makedirs(d, exist_ok=True)
    for n in names:
        with open(os.path.join(d, n), "w") as f:
            f.write("{}")
    return str(d)


def test_snapshot_without_refs_main(tmp_path):
    s = _files(tmp_path / "models--facebook--dinov2-base" / "snapshots" / REV, ["config.json", "model.safetensors"])
    assert hf_snapshot(str(tmp_path), DINO, ("config.json", "model.safetensors")) == s
    assert hf_snapshot(str(tmp_path), DINO, ("config.json", "missing.bin")) is None
    assert hf_snapshot(None, DINO) is None and hf_snapshot(str(tmp_path / "nope"), DINO) is None


def test_refs_main_wins_and_flat_layout(tmp_path):
    root = tmp_path / "models--facebook--dinov2-base"
    a = _files(root / "snapshots" / "aaaa", ["config.json"])
    _files(root / "snapshots" / "bbbb", ["config.json"])
    os.makedirs(root / "refs")
    (root / "refs" / "main").write_text("aaaa\n")
    assert hf_snapshot(str(tmp_path), DINO) == a
    flat = _files(tmp_path / "other" / "dinov2-base", ["config.json"])
    assert hf_snapshot(str(tmp_path / "other"), DINO) == flat


def test_whisper_name_maps_to_the_local_snapshot(tmp_path):
    s = _files(tmp_path / "models--Systran--faster-whisper-large-v3" / "snapshots" / "edaa", ["model.bin", "config.json", "tokenizer.json"])
    assert whisper_source("large-v3", str(tmp_path)) == s
    assert whisper_source("large-v3", None) == "large-v3"
    assert whisper_source("medium", str(tmp_path)) == "medium"  # not there: the name (downloads)


def test_embedder_loads_by_path(tmp_path, monkeypatch):
    s = _files(tmp_path / "models--facebook--dinov2-base" / "snapshots" / REV, ["config.json", "model.safetensors"])
    import transformers as tf  # declared in requirements.txt: must run, not skip

    seen = {}

    class _M:
        def eval(self):
            return self

        def to(self, d):
            return self

    monkeypatch.setattr(tf.AutoModel, "from_pretrained", lambda src, **kw: seen.setdefault("src", src) and _M())
    from lsdub_engine.core import cast

    cast.TorchEmbedder("cpu", cache_dir=str(tmp_path))
    assert seen["src"] == s


MODELS = os.environ.get("LSDUB_HF_MODELS")


@pytest.mark.skipif(not MODELS, reason="set LSDUB_HF_MODELS to a models/lsdub folder laid out as the build spec places it")
def test_real_weights_load_offline():
    """a fresh interpreter, HF_HUB_OFFLINE=1 and an unreachable hub endpoint: both models load from models/lsdub."""
    code = textwrap.dedent(f"""
        import os
        from lsdub_engine.core import cast
        from lsdub_engine.core.lyric_align import whisper_source
        e = cast.TorchEmbedder('cpu', cache_dir=os.path.join({MODELS!r}, 'dinov2'))
        import numpy as np
        v = e([np.zeros((224, 224, 3), np.uint8)])
        assert v.shape == (1, 1536), v.shape
        from faster_whisper import WhisperModel
        src = whisper_source('large-v3', os.path.join({MODELS!r}, 'whisper'))
        assert os.path.isdir(src), src
        WhisperModel(src, device='cpu', compute_type='int8')
        print('OFFLINE-OK')
    """)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_ENDPOINT="http://127.0.0.1:9",
        HF_HOME=os.path.join(os.environ.get("TMPDIR", "/tmp"), "lsdub-empty-hf-home"),
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert "OFFLINE-OK" in r.stdout, r.stderr[-3000:]


ESPEAK = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def test_judge_phones_loads_by_path(tmp_path, monkeypatch):
    from lsdub_engine.audits import judge_phones as JP
    import transformers as tf

    s = _files(tmp_path / "models--facebook--wav2vec2-lv-60-espeak-cv-ft" / "snapshots" / REV, JP.NEED)
    seen = {}

    class _M:
        class config:
            @staticmethod
            def to_dict():
                return {}

        def eval(self):
            return self

        def to(self, d):
            return self

    monkeypatch.setattr(tf.AutoModelForCTC, "from_pretrained", lambda src, **kw: seen.setdefault("src", src) and _M())
    P = JP.Phones(cache_dir=str(tmp_path))
    assert seen["src"] == s and P.source == s and P.labels == [] and P.blank == 0
    monkeypatch.setenv("LSDUB_PHONE_MODELS", str(tmp_path / "nope"))
    assert JP.model_source() == ESPEAK  # not there: the repo id (downloads)


@pytest.mark.skipif(not MODELS, reason="set LSDUB_HF_MODELS to a models/lsdub folder laid out as the build spec places it")
def test_judge_phones_real_weights_load_offline():
    """a fresh interpreter, HF_HUB_OFFLINE=1 and an unreachable hub endpoint: the recogniser loads from models/lsdub/wav2vec2-espeak and
    reads a second of silence as no nuclei."""
    code = textwrap.dedent(f"""
        import os, numpy as np
        from lsdub_engine.audits import judge_phones as JP
        P = JP.Phones(cache_dir=os.path.join({MODELS!r}, 'wav2vec2-espeak'))
        assert os.path.isdir(P.source), P.source
        x = np.zeros(16000)
        lp = JP.logposteriors(x, P)
        assert lp.shape[1] == len(P.labels) and JP.nuclei(lp, x, P) == []
        print('OFFLINE-OK')
    """)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_ENDPOINT="http://127.0.0.1:9",
        HF_HOME=os.path.join(os.environ.get("TMPDIR", "/tmp"), "lsdub-empty-hf-home"),
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert "OFFLINE-OK" in r.stdout, r.stderr[-3000:]


def test_phone_net_loads_by_path(tmp_path, monkeypatch):
    """the driver's phone network, like the judge's: the local snapshot by path when its files are there, the repo id otherwise."""
    from lsdub_engine.core import phone_net as PN
    import transformers as tf

    s = _files(tmp_path / "models--facebook--wav2vec2-lv-60-espeak-cv-ft" / "snapshots" / REV, PN.NEED)
    seen = {}

    class _M:
        class config:
            @staticmethod
            def to_dict():
                return {}

        def eval(self):
            return self

        def to(self, d):
            return self

    monkeypatch.setattr(tf.AutoModelForCTC, "from_pretrained", lambda src, **kw: seen.setdefault("src", src) and _M())
    net = PN.PhoneNet(cache_dir=str(tmp_path))
    assert seen["src"] == s and net.source == s and net.labels == [] and net.blank == 0
    monkeypatch.setenv("LSDUB_PHONE_MODELS", str(tmp_path / "nope"))
    assert PN.model_source() == ESPEAK


@pytest.mark.skipif(not os.environ.get("LSDUB_PHONE_MODELS"), reason="set LSDUB_PHONE_MODELS to the recogniser's local hub folder")
def test_align_and_paint_stages_round_trip_offline(tmp_path):
    """the round trip the pack promises: with the recogniser's files pre-placed under a models folder and the hub unreachable
    (HF_HUB_OFFLINE=1, a dead endpoint, an empty HF_HOME), `lsdub align --aligner ctc --phone-models <dir>` aligns a short cut and
    `phone_net.PhoneNet(<dir>)` loads by path in a fresh interpreter with no environment variable naming the folder; and loading by
    repo id with a cache_dir names that folder as the download target (no hub call is made: the folder already holds the snapshot)."""
    models = os.environ["LSDUB_PHONE_MODELS"]
    code = textwrap.dedent(f"""
        import json, os, numpy as np, soundfile as sf
        os.environ.pop('LSDUB_PHONE_MODELS', None); os.environ.pop('LSDUB_HF_MODELS', None)
        from lsdub_engine.core import lyric_align as LA, phone_net as PN
        net = PN.PhoneNet({models!r})
        assert os.path.isdir(net.source), net.source
        sr = 16000; t = np.arange(sr * 2) / sr
        x = np.zeros(sr * 2, np.float32); on = (t > 0.3) & (t < 1.5); x[on] = (0.3 * np.sin(2 * np.pi * 200 * t[on])).astype(np.float32)
        d = {tmp_path.as_posix()!r}
        sf.write(d + '/lead.wav', x, sr); open(d + '/lyrics.txt', 'w').write('ma\\n')
        json.dump([dict(w='ma', start=0.3, end=1.5, p=0.9)], open(d + '/stt.json', 'w'))
        LA.main(['--audio', d + '/lead.wav', '--lyrics', d + '/lyrics.txt', '--t0', '0', '--t1', '2', '--stt-json', d + '/stt.json',
                 '--aligner', 'ctc', '--phone-models', {models!r}, '--out', d + '/align.json'])
        A = json.load(open(d + '/align.json'))
        assert A['aligner'] == 'ctc' and any(w['align'] == 'ctc' for w in A['words']), A['words']
        print('ROUNDTRIP-OK')
    """)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_ENDPOINT="http://127.0.0.1:9",
        HF_HOME=str(tmp_path / "empty-hf-home"),
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0 and "ROUNDTRIP-OK" in r.stdout, r.stderr[-2000:]
