# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""execute() of the nodes the demos run, with the engine steps stubbed (no models, no GPU): Align, Triage, Lip Kit
(bring and build), Save Cut, Sync QA, Vocal Split and Lead Vocal; plus the lead_vocal engine on synthetic audio."""
import asyncio
import importlib
import json
import os
import subprocess

import numpy as np
import pytest


def _mod(pack, name):
    return importlib.import_module(f'{pack.__name__}.{name}')


def _wav(path, seconds=2.0, f=220):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'sine=f={f}:d={seconds}', '-ac', '2', str(path)], check=True)
    return str(path)


def _proj(pack, tmp_path, **kw):
    return pack.project.Project(root=str(tmp_path), name='cut', **kw).ensure()


# ------------------------------------------------------------------------------------------------------ Align --
@pytest.fixture
def align_stubs(pack, monkeypatch):
    LA, VT, VO = (_mod(pack, f'lsdub_engine.core.{m}') for m in ('lyric_align', 'viseme_track', 'vocalisations'))
    seen = {'ratio': 0.9, 'argv': []}

    def la_main(argv):
        seen['argv'].append(list(argv))
        out = argv[argv.index('--out') + 1]
        if '--stt-only' in argv:
            json.dump([{'w': 'hello', 'start': 0.1, 'end': 0.5, 'p': 0.9}], open(argv[argv.index('--stt-json') + 1], 'w'))
            return
        json.dump({'lyric_match_ratio': seen['ratio'], 'aligner': 'sphinx', 'words': [
            {'w': 'hello', 'start': 0.1, 'end': 0.5, 'src': 'lyric', 'heard': 'hello', 'p_whisper': 0.9, 'align': 'sphinx',
             'phones': [{'p': 'HH', 'start': 0.1, 'end': 0.2}, {'p': 'AH', 'start': 0.2, 'end': 0.5}]}]}, open(out, 'w'))

    def vt_main(argv):
        json.dump({'frames': []}, open(argv[argv.index('--out') + 1], 'w'))

    def vo_main(argv):
        json.dump({'frames': [], 'skips': []}, open(argv[argv.index('--out') + 1], 'w'))
    monkeypatch.setattr(LA, 'main', la_main)
    monkeypatch.setattr(VT, 'main', vt_main)
    monkeypatch.setattr(VO, 'main', vo_main)
    return seen


def test_align_execute_prompts_with_the_lyrics_and_caches(pack, tmp_path, align_stubs):
    T = pack.lsdub_types
    p = _proj(pack, tmp_path)
    lead = _wav(tmp_path / 'lead.wav')
    lp = tmp_path / 'lyrics.txt'
    lp.write_text('[Verse]\nhello world\n')
    song, lyr = T.Song(mix=lead, lead=lead, hash='s'), T.Lyrics(path=str(lp), hash='l')
    A = pack.nodes_audio.LSDubAlign
    rec, md = A.execute(p, song, lyr).args
    argv = align_stubs['argv'][-1]
    assert argv[argv.index('--prompt') + 1] == 'hello world.' and argv[argv.index('--audio') + 1] == lead
    assert os.path.exists(rec.align) and rec.visemes.endswith('visemes_skip.json') and 'lyric match 0.9' in md
    n = len(align_stubs['argv'])
    A.execute(p, song, lyr)
    assert len(align_stubs['argv']) == n                                            # cached


def test_align_execute_warns_then_raises_on_a_low_match(pack, tmp_path, align_stubs):
    T = pack.lsdub_types
    lead = _wav(tmp_path / 'lead.wav')
    lp = tmp_path / 'lyrics.txt'
    lp.write_text('hello world\n')
    song, lyr = T.Song(mix=lead, lead=lead, hash='s'), T.Lyrics(path=str(lp), hash='l')
    align_stubs['ratio'] = 0.3
    md = pack.nodes_audio.LSDubAlign.execute(_proj(pack, tmp_path / 'a'), song, lyr).args[1]
    assert '**Warning:**' in md and 'placed over the voiced' in md
    align_stubs['ratio'] = 0.1
    with pytest.raises(RuntimeError, match='match what is sung'):
        pack.nodes_audio.LSDubAlign.execute(_proj(pack, tmp_path / 'b'), song, lyr)


def test_align_execute_without_lyrics_uses_the_transcript(pack, tmp_path, align_stubs):
    T = pack.lsdub_types
    lead = _wav(tmp_path / 'lead.wav')
    align_stubs['ratio'] = 0.0                                                      # no sheet given: never an error
    rec, _ = pack.nodes_audio.LSDubAlign.execute(_proj(pack, tmp_path), T.Song(mix=lead, lead=lead, hash='s')).args
    assert '--stt-only' in align_stubs['argv'][0] and '--prompt' not in align_stubs['argv'][-1]
    assert open(os.path.join(os.path.dirname(rec.align), 'lyrics_from_transcript.txt')).read().strip() == 'hello'


# ----------------------------------------------------------------------------------------------------- Triage --
def test_triage_execute(pack, tmp_path, monkeypatch):
    import cv2
    qa = _mod(pack, 'lsdub_engine.audits.qa')
    calls = []

    def triage(video, vocal=None, offset=0.0, db=None, targets=(), samples=6, model=None, detected=None, preset=None, emb=None):
        calls.append((video, vocal, offset, samples, detected[1], preset))
        return {'video': video}, [{'shot': 0, 'f0': 0, 'f1': 49, 'verdict': 'DUB', 'reasons': []},
                                  {'shot': 1, 'f0': 50, 'f1': 99, 'verdict': 'NO', 'reasons': ['too short']}]

    def write_triage(meta, rows, out):
        json.dump({**meta, 'shots': rows}, open(out + '.json', 'w'))
        open(out + '.md', 'w').write('# LSDub QA triage: copy.mp4\nrows\n')
        cv2.imwrite(out + '_sheet.jpg', np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(qa, 'triage', triage)
    monkeypatch.setattr(qa, 'write_triage', write_triage)
    T = pack.lsdub_types
    p = _proj(pack, tmp_path, song_offset=2.0)
    feats = str(tmp_path / 'f.npz')
    np.savez(feats, fps=25.0, size=np.array([64, 48]), dh=np.zeros(3), dp=np.zeros(3))
    shots = T.Shots(shots=[{'shot': 0, 'f0': 0, 'f1': 49}, {'shot': 1, 'f0': 50, 'f1': 99}], hash='sh', features=feats)
    src = T.Source(path='v.mp4', hash='v', name='mine.mp4')
    rec, sheet, md = pack.nodes_video.LSDubTriage.execute(p, src, shots, T.Song(mix='m.wav', lead='l.wav', hash='s')).args
    assert calls == [('v.mp4', 'l.wav', 2.0, 6, 25.0, p.preset)]
    assert [s['verdict'] for s in rec.shots] == ['DUB', 'NO'] and rec.features == feats
    assert md.startswith('# LSDub QA triage: mine.mp4')                                 # the user's file name
    pack.nodes_video.LSDubTriage.execute(p, src, shots, T.Song(mix='m.wav', lead='l.wav', hash='s'))
    assert len(calls) == 1


# ---------------------------------------------------------------------------------------------------- Lip Kit --
def test_lip_kit_bring_a_folder(pack, tmp_path):
    K = pack.nodes_mouth.LSDubBuildLipKit
    d = tmp_path / 'kit'
    d.mkdir()
    (d / 'aa.png').write_bytes(b'x')
    kit = asyncio.run(K.execute(_proj(pack, tmp_path), 'nova', str(d))).args[0]
    assert kit.dir == str(d) and kit.identity == 'nova' and kit.hash
    with pytest.raises(FileNotFoundError):
        asyncio.run(K.execute(_proj(pack, tmp_path), 'nova', str(tmp_path / 'missing')))
    with pytest.raises(ValueError, match='identity'):
        asyncio.run(K.execute(_proj(pack, tmp_path), ' ', str(d)))


def test_lip_kit_build_from_a_shot(pack, tmp_path, monkeypatch, clip):
    ML = _mod(pack, 'lsdub_engine.core.mouth_library')
    M = pack.nodes_mouth
    seen = {}

    def ml_main(argv):
        seen['prep'] = list(argv)
        os.makedirs(argv[argv.index('--out-dir') + 1], exist_ok=True)

    async def run(graph, backend='local', uploads=None, out_dir='.', api_key=None, out_node='out'):
        seen.setdefault('gen', []).append(backend)
        return [os.path.join(out_dir, 'x.png')]

    def gen(args, runner):
        seen['seeds'] = args.seeds
        runner('aa', args.keep, args.seeds[0], 'base.png', args.out_dir)
        open(os.path.join(args.out_dir, 'base.png'), 'w').write('b')
    monkeypatch.setattr(ML, 'main', ml_main)
    monkeypatch.setattr(ML, 'gen', gen)
    monkeypatch.setattr(pack.generate, 'run', run)
    monkeypatch.setattr(M, 'extract_shot', lambda src, f0, f1, out: (seen.__setitem__('span', (f0, f1)), out)[1])
    T = pack.lsdub_types
    shots = T.Shots(shots=[{'shot': 2, 'f0': 5, 'f1': 30}], hash='sh')
    p = _proj(pack, tmp_path, seed=1)
    kit = asyncio.run(M.LSDubBuildLipKit.execute(p, 'nova', '', T.Source(path=clip, hash='c'), shots, 2, seeds=2)).args[0]
    assert seen['span'] == (5, 30) and seen['gen'] == ['local'] and seen['seeds'] == [12, 23]
    assert kit.identity == 'nova' and kit.base_frame.endswith('base.png')
    with pytest.raises(ValueError, match='no shot 9'):
        asyncio.run(M.LSDubBuildLipKit.execute(p, 'other', '', T.Source(path=clip, hash='c'), shots, 9))


# --------------------------------------------------------------------------------------------------- Save Cut --
def test_save_cut_execute(pack, tmp_path, monkeypatch):
    WM = _mod(pack, 'lsdub_engine.watermark')
    marked = []
    monkeypatch.setattr(WM, 'apply_inplace', lambda path, crf=18: marked.append((path, crf)))
    cut = tmp_path / 'cut.mp4'
    cut.write_bytes(b'video')
    rnd = pack.lsdub_types.Render(shots={}, cut=str(cut), hash='r1')
    p = _proj(pack, tmp_path)
    S = pack.nodes_qa.LSDubSaveCut
    _, out = S.execute(p, rnd, True, 20).args
    assert out.startswith(os.path.join(p.dir, 'final')) and open(out, 'rb').read() == b'video' and marked[0][1] == 20
    S.execute(p, rnd, True, 20)
    assert len(marked) == 1                                                          # already saved
    _, out2 = S.execute(p, rnd, False, 20).args
    assert out2 != out and len(marked) == 1                                          # no watermark asked: none applied


# ---------------------------------------------------------------------------------------------------- Sync QA --
def test_sync_qa_execute(pack, tmp_path, monkeypatch):
    Q = _mod(pack, 'lsdub_engine.audits.sync_qa')
    SS = _mod(pack, 'lsdub_engine.audits.sync_strip')
    M = pack.nodes_mouth
    calls = []

    def score_film(video, cuts, vocal, offset, align, work=None, preset=None):
        calls.append((video, cuts, vocal, offset, align, preset))
        return [{'shot': 4, 'corr': None}, {'shot': 7, 'corr': 0.61, 'lag_ms': 40}]

    def strip(argv):
        calls.append(('card', argv[argv.index('--t0') + 1]))
        open(argv[argv.index('--out') + 1], 'w').write('card')
    monkeypatch.setattr(Q, 'score_film', score_film)
    monkeypatch.setattr(Q, 'film_md', lambda rows: '# sync qa\n')
    monkeypatch.setattr(SS, 'main', strip)
    monkeypatch.setattr(M, 'extract_shot', lambda src, f0, f1, out: out)
    T = pack.lsdub_types
    p = _proj(pack, tmp_path, song_offset=1.0)
    shots = T.Shots(shots=[{'shot': 4, 'f0': 0, 'f1': 9, 'start': 0.0, 'end': 0.4, 'verdict': 'DUB'},
                           {'shot': 7, 'f0': 10, 'f1': 59, 'start': 3.0, 'end': 5.0, 'verdict': 'DUB'}], hash='sh')
    src = T.Source(path='v.mp4', hash='v')
    song, al = T.Song(mix='m.wav', lead='l.wav', hash='s'), T.Align(align='a.json', hash='a')
    rep, card, md = pack.nodes_qa.LSDubSyncQA.execute(p, src, shots, song, al).args
    assert calls[0][0] == 'v.mp4' and calls[0][2:] == ('l.wav', 1.0, 'a.json', p.preset)
    assert calls[1] == ('card', '2.000')                                              # the first measured shot: 7
    assert rep.verdicts == {'4': None, '7': 0.61} and md == '# sync qa\n'
    pack.nodes_qa.LSDubSyncQA.execute(p, src, shots, song, al)
    assert len(calls) == 2                                                            # cached


# --------------------------------------------------------------------------------------- Vocal Split / Lead ---
def test_vocal_split_execute(pack, monkeypatch):
    torch = pytest.importorskip('torch')
    vs = _mod(pack, 'lsdub_engine.core.vocal_split')
    seen = {}

    class Net:
        def to(self, d): seen['to'] = d; return self   # noqa: E702
    monkeypatch.setattr(vs, 'load', lambda path, dev, dtype: (seen.update(path=path, dtype=dtype), Net())[1])
    monkeypatch.setattr(vs, 'separate', lambda mix, m: mix * 0.25)
    audio = {'waveform': torch.ones(1, 1, vs.SR), 'sample_rate': vs.SR}
    v, inst = pack.nodes_vocal.LSDubVocalSplit.execute(audio, 'kim.ckpt', 'bf16').args
    assert seen['path'].endswith(os.path.join('audio_separation', 'kim.ckpt')) and seen['dtype'] == torch.bfloat16
    assert seen['to'] == 'cpu' and v['waveform'].shape == (1, 2, vs.SR)               # mono doubled to stereo
    assert torch.allclose(v['waveform'] + inst['waveform'], torch.ones(1, 2, vs.SR))


def test_lead_vocal_engine_on_synthetic_audio():
    pytest.importorskip('torch')
    from lsdub_engine.core import lead_vocal as lv
    sr = lv.SR
    t = np.arange(int(1.5 * sr)) / sr
    tone = 0.3 * sum(np.sin(2 * np.pi * 220 * k * t) / k for k in (1, 2, 3))          # a harmonic "voice" at 220 Hz
    x = np.r_[tone[:sr // 2], np.zeros(sr // 2), tone[:sr // 2]].astype(np.float32)   # sung, a rest, sung
    d = lv.extract(np.stack([x, x]))
    act, f0, tt = d['active'], d['f0'], d['t']
    sung = (tt < 0.45) | (tt > 1.05)
    rest = (tt > 0.6) & (tt < 0.9)
    assert act[sung].mean() > 0.5 and act[rest].max() == 0.0                          # lead in the sung parts only
    assert np.median(f0[sung & (f0 > 0)]) == pytest.approx(220, rel=0.06)
    assert d['lead'].shape == (2, len(x)) and np.allclose(d['lead'] + d['backing'], np.stack([x, x]), atol=1e-5)
    d2 = lv.extract(np.stack([x, x]), share=0.99)                                       # a stricter share keeps less
    assert (d2['active'] > 0).sum() <= (act > 0).sum()


def test_align_retries_once_without_the_prompt(pack, tmp_path, align_stubs, monkeypatch):
    T = pack.lsdub_types
    LA = _mod(pack, 'lsdub_engine.core.lyric_align')
    lead = _wav(tmp_path / 'lead.wav')
    lp = tmp_path / 'lyrics.txt'
    lp.write_text('hello world\n')
    first = LA.main
    calls = []

    def flaky(argv):                                                                # a bad transcript, then a good one
        calls.append(list(argv))
        align_stubs['ratio'] = 0.0 if len(calls) == 1 else 0.9
        first(argv)
    monkeypatch.setattr(LA, 'main', flaky)
    rec, md = pack.nodes_audio.LSDubAlign.execute(_proj(pack, tmp_path), T.Song(mix=lead, lead=lead, hash='s'),
                                                  T.Lyrics(path=str(lp), hash='l')).args
    assert len(calls) == 2 and '--prompt' in calls[0] and '--prompt' not in calls[1] and 'lyric match 0.9' in md
