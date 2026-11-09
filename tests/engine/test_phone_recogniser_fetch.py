# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the recogniser's first use, as a user meets it: `lsdub align --aligner ctc --phone-models <folder>`.

- An empty models folder, a hub that serves only the four files KNOWN pins (anything else is refused), the tokenizer route and the
  hub's own model download blocked: the four files land in the folder sha-checked, the network loads by path, the vocabulary comes from
  vocab.json, and the lyric aligns ctc. 0.2.0 failed here: it loaded by repo id, the download left vocab.json out, and the vocabulary
  went through the phoneme tokenizer, which needs phonemizer.
- The same folder pre-placed, HF_HUB_OFFLINE=1 and every hub call refused: it loads by path, nothing is fetched.

The weights are a tiny random Wav2Vec2 (the real 1.2 GB recogniser is not in CI) and KNOWN is pointed at its files; the posteriors are
pinned as in test_lyric_align_voiced, so what is under test is the loading, not the recogniser."""

import json
import os
import shutil
import wave

import numpy as np
import pytest

from lsdub_engine import _models as M
from lsdub_engine.core import lyric_align as LA, phone_net as PN

REV = "0" * 40
LABELS = ["<pad>", "<unk>", "m", "ɑ", "p", "a", "ɪ", "t"]


@pytest.fixture(scope="module")
def fake_hub(tmp_path_factory):
    """the four pinned file names, written by a tiny Wav2Vec2 for CTC: {name: path}."""
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("hub")
    cfg = Wav2Vec2Config(vocab_size=len(LABELS), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                         conv_dim=(8,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, pad_token_id=0)  # fmt: skip
    net = Wav2Vec2ForCTC(cfg)
    net.save_pretrained(d)  # config.json (transformers 5 writes safetensors only); the pinned weights file is the .bin
    os.remove(d / "model.safetensors")
    torch.save(net.state_dict(), d / "pytorch_model.bin")
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000).save_pretrained(d)
    (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(LABELS)}), encoding="utf-8")
    files = {n: str(d / n) for n in PN.NEED}
    assert all(os.path.isfile(p) for p in files.values()), sorted(os.listdir(d))
    return files


@pytest.fixture
def pinned(fake_hub, monkeypatch):
    """KNOWN pins the fake files; the tokenizer route and the hub's model download fail loudly; phonemizer cannot be imported."""
    import transformers

    monkeypatch.setitem(M.KNOWN, PN.MODEL, {"revision": REV, "files": {n: M.sha256_file(p) for n, p in fake_hub.items()}})

    def no_tokenizer(*a, **k):
        raise AssertionError("the vocabulary went through a tokenizer class")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", no_tokenizer)
    monkeypatch.setitem(__import__("sys").modules, "phonemizer", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # a load by repo id (the 0.2.0 path) cannot reach the hub
    monkeypatch.setenv("HF_HOME", str(fake_hub["config.json"]) + ".no-home")
    monkeypatch.delenv("LSDUB_PHONE_MODELS", raising=False)
    monkeypatch.delenv("LSDUB_HF_MODELS", raising=False)
    return fake_hub


def serve(fake_hub, asked):
    """hf_hub_download for the fake hub: the pinned revision's four files only, into the cache layout, recording each ask."""

    def download(repo_id, filename, revision=None, cache_dir=None, **k):
        asked.append((repo_id, filename, revision))
        assert repo_id == PN.MODEL and revision == REV and filename in fake_hub, (repo_id, filename, revision)
        snap = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"), "snapshots", revision)
        os.makedirs(snap, exist_ok=True)
        dst = os.path.join(snap, filename)
        shutil.copyfile(fake_hub[filename], dst)
        return dst

    return download


def align_ctc(tmp_path, models, monkeypatch):
    """`lsdub align --aligner ctc --phone-models <models>` on 'ma pie'; the posteriors pinned on the loaded network's own labels."""
    seen = {}

    def posteriors(x, net):
        seen["labels"], seen["source"] = list(net.labels), net.source
        rows = [0] * 10 + [2] * 3 + [3] * 10 + [0] * 5 + [4] * 2 + [5] * 12 + [0] * 108
        lp = np.full((len(rows), len(net.labels)), np.log(0.001))
        lp[np.arange(len(rows)), rows] = np.log(0.9)
        return lp

    monkeypatch.setattr(PN, "posteriors", posteriors)
    monkeypatch.setattr(PN, "nuclei", lambda logp, x, net, corr=0.02: [])
    monkeypatch.setattr(LA, "cmu_phones", lambda w, extra: {"ma": ["M", "AA1"], "pie": ["P", "AY1"]}[w])
    sr = 16000
    t = np.arange(sr * 3) / sr
    x = np.zeros(len(t), np.float32)
    on = ((t > 0.2) & (t < 0.46)) | ((t > 0.56) & (t < 0.84))
    x[on] = (0.3 * np.sin(2 * np.pi * 220 * t[on])).astype(np.float32)
    with wave.open(str(tmp_path / "lead.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((x * 32767).astype(np.int16).tobytes())
    (tmp_path / "lyrics.txt").write_text("ma pie\n")
    (tmp_path / "stt.json").write_text(json.dumps([dict(w="ma", start=0.0, end=0.3, p=0.9), dict(w="pie", start=0.3, end=0.9, p=0.9)]))
    out = tmp_path / "align.json"
    LA.main(["--audio", str(tmp_path / "lead.wav"), "--lyrics", str(tmp_path / "lyrics.txt"), "--t0", "0", "--t1", "3",
             "--stt-json", str(tmp_path / "stt.json"), "--aligner", "ctc", "--phone-models", str(models), "--out", str(out)])  # fmt: skip
    return json.loads(out.read_text()), seen


def test_first_use_fetches_the_four_pinned_files_and_aligns_without_a_tokenizer(tmp_path, pinned, monkeypatch):
    import huggingface_hub

    asked = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", serve(pinned, asked))
    models = tmp_path / "models" / "wav2vec2-espeak"
    models.mkdir(parents=True)
    A, seen = align_ctc(tmp_path, models, monkeypatch)
    assert sorted(f for _, f, _ in asked) == sorted(PN.NEED)  # exactly the four, nothing converted, no tokenizer files
    snap = models / ("models--" + PN.MODEL.replace("/", "--")) / "snapshots" / REV
    assert seen["source"] == str(snap) and seen["labels"] == LABELS  # loaded by path; the vocabulary is vocab.json's
    assert A["aligner"] == "ctc" and [w["align"] for w in A["words"] if w["w"] in ("ma", "pie")] == ["ctc", "ctc"]
    assert "phonemizer" not in {k for k, v in __import__("sys").modules.items() if v is not None}


def test_a_fetched_file_with_the_wrong_bytes_is_refused(tmp_path, pinned, monkeypatch):
    import huggingface_hub

    bad = dict(pinned, **{"vocab.json": str(tmp_path / "vocab.json")})
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", serve(bad, []))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        PN.PhoneNet(str(tmp_path / "models"))


def test_pre_placed_files_load_offline_with_no_hub_call(tmp_path, pinned, monkeypatch):
    import huggingface_hub

    def refuse(*a, **k):
        raise AssertionError(f"a hub call with the files in place: {a} {k}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    models = tmp_path / "models" / "wav2vec2-espeak"
    snap = models / ("models--" + PN.MODEL.replace("/", "--")) / "snapshots" / REV
    snap.mkdir(parents=True)
    for n, p in pinned.items():
        shutil.copyfile(p, snap / n)
    A, seen = align_ctc(tmp_path, models, monkeypatch)
    assert seen["source"] == str(snap) and seen["labels"] == LABELS
    assert A["aligner"] == "ctc" and [w["align"] for w in A["words"] if w["w"] in ("ma", "pie")] == ["ctc", "ctc"]
