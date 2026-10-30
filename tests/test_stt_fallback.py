"""stt(): a GPU whose CUDA runtime fails lazily (on the first encode, e.g. libcublas.so.12 missing on a CUDA 13 image)
falls back to the CPU instead of failing the Align node."""
import sys
import types

import pytest

from lsdub_engine.core import lyric_align as LA


class _W:
    def __init__(self, word, start, end, p):
        self.word, self.start, self.end, self.probability = word, start, end, p


class _Seg:
    def __init__(self, words):
        self.words = words


def _fake_whisper(monkeypatch, fail_on):
    made = []

    class WhisperModel:
        def __init__(self, model, device='cpu', **kw):
            self.device = device
            made.append(device)

        def transcribe(self, pcm, **kw):
            dev = self.device

            def gen():
                if dev in fail_on:
                    raise RuntimeError('Library libcublas.so.12 is not found or cannot be loaded')
                yield _Seg([_W(' hello', 1.5, 1.9, 0.9)])
            return gen(), None

    monkeypatch.setitem(sys.modules, 'faster_whisper', types.SimpleNamespace(WhisperModel=WhisperModel))
    monkeypatch.setattr(LA, 'whisper_source', lambda m, d: m)
    return made


@pytest.fixture
def wav(tmp_path):
    import numpy as np
    import wave
    p = tmp_path / 'a.wav'
    with wave.open(str(p), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.zeros(16000, np.int16)).tobytes())
    return str(p)


def test_lazy_cuda_failure_falls_back_to_cpu(monkeypatch, wav):
    made = _fake_whisper(monkeypatch, fail_on={'cuda'})
    out = LA.stt(wav, 0.0, 1.0, device='cuda')
    assert made == ['cuda', 'cpu']
    assert out and out[0]['w'] == 'hello'


def test_cpu_failure_is_not_swallowed(monkeypatch, wav):
    _fake_whisper(monkeypatch, fail_on={'cpu'})
    with pytest.raises(RuntimeError):
        LA.stt(wav, 0.0, 1.0, device='cpu')
