# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""LSDub Dub's bookkeeping, with the per-shot dub and the cut assembly stubbed: failed shots are reported, the cut key
follows each shot's result, force_shots acts once per value, masks from another crop are refused; plus the real
frame-exact extraction and the assembly's failure / mismatch handling."""
import os

import numpy as np
import pytest


def _setup(pack, tmp_path, force=''):
    T, P = pack.lsdub_types, pack.project
    p = P.Project(root=str(tmp_path), name='cut', force_shots=P.parse_force_shots(force)).ensure()
    src = T.Source(path=str(tmp_path / 'src.mp4'), hash='src', fps=25.0)
    shots = T.Shots(shots=[{'shot': 1, 'f0': 0, 'f1': 9, 'start': 0.0, 'end': 0.4, 'verdict': 'DUB', 'target': 'a'},
                           {'shot': 2, 'f0': 10, 'f1': 19, 'start': 0.4, 'end': 0.8, 'verdict': 'DUB', 'target': 'a'},
                           {'shot': 3, 'f0': 20, 'f1': 29, 'start': 0.8, 'end': 1.2, 'verdict': 'NO', 'target': 'a'}], hash='sh')
    song = T.Song(mix='m.wav', lead='l.wav', hash='song')
    al = T.Align(align='a.json', hash='al')
    kit = T.Kit(dir=str(tmp_path), identity='a', hash='kit')
    return p, src, shots, song, al, kit


@pytest.fixture
def stubbed(pack, monkeypatch):
    M = pack.nodes_mouth
    log = {'runs': [], 'cuts': [], 'outcome': {}}

    def run_shot(cls, project, source, s, song, align, kit, tw, masks, cast, st, vocalisations='skip'):
        log['runs'].append(s['shot'])
        mode = log['outcome'].get(s['shot'], 'pass')
        if mode == 'raise': raise RuntimeError('no face found')
        r = st.path('render.mp4')
        with open(r, 'w') as f: f.write(f"{s['shot']}-{len(log['runs'])}")       # a new result every run
        return r, {'pass': mode == 'pass', 'corr': 0.9 if mode == 'pass' else 0.2}

    def assemble(source, renders, out, fps, crf=18, audio='source', song=None, offset=0.0):
        log['cuts'].append(sorted(renders))
        open(out, 'w').write('cut')
        return []
    monkeypatch.setattr(M.LSDubDub, 'run_shot', classmethod(run_shot))
    monkeypatch.setattr(M, 'assemble', assemble)
    return log


def _dub(pack, p, src, shots, song, al, kit, **kw):
    return pack.nodes_mouth.LSDubDub.execute(p, src, shots, song, al, kit, **kw).args[0]


def test_failed_and_raising_shots_are_reported_and_keep_source(pack, tmp_path, stubbed):
    p, src, shots, song, al, kit = _setup(pack, tmp_path)
    stubbed['outcome'] = {1: 'fail', 2: 'raise'}
    r = _dub(pack, p, src, shots, song, al, kit)
    assert set(r.shots) == {'1', '2'}                                                  # NO shot 3 is not dubbed
    assert r.shots['1']['pass'] is False and 'sync gate failed' in r.shots['1']['reason'] and r.shots['1']['used'] == 'source'
    assert r.shots['2']['pass'] is False and 'no face found' in r.shots['2']['reason'] and r.shots['2']['render'] == ''
    assert stubbed['cuts'][-1] == []                                                   # nothing spliced on failure
    r2 = _dub(pack, p, src, shots, song, al, kit, on_fail='render')
    assert r2.shots['1']['used'] == 'render' and stubbed['cuts'][-1] == [(0, 9)]
    md, force = pack.nodes_qa.LSDubFixIt.execute(p, shots, None, None, 0.5, r2).args
    assert 'sync gate failed' in md and 'dub error' in md and force == '2'


def test_cut_key_follows_results_and_force_acts_once(pack, tmp_path, stubbed):
    p, src, shots, song, al, kit = _setup(pack, tmp_path)
    r1 = _dub(pack, p, src, shots, song, al, kit)
    assert stubbed['runs'] == [1, 2]
    assert _dub(pack, p, src, shots, song, al, kit).hash == r1.hash and stubbed['runs'] == [1, 2]    # all cached
    pf, *_ = _setup(pack, tmp_path, force='2')
    r2 = _dub(pack, pf, src, shots, song, al, kit)
    assert stubbed['runs'] == [1, 2, 2] and r2.hash != r1.hash                         # redone shot -> new cut key
    assert r2.shots['1']['result'] == r1.shots['1']['result'] and r2.shots['2']['result'] != r1.shots['2']['result']
    _dub(pack, pf, src, shots, song, al, kit)
    assert stubbed['runs'] == [1, 2, 2]                                                # same force value: not again
    p0, *_ = _setup(pack, tmp_path, force='')
    _dub(pack, p0, src, shots, song, al, kit)
    _dub(pack, pf, src, shots, song, al, kit)
    assert stubbed['runs'] == [1, 2, 2, 2]                                             # cleared and set again: once more


def test_failed_shot_is_retried(pack, tmp_path, stubbed):
    p, src, shots, song, al, kit = _setup(pack, tmp_path)
    stubbed['outcome'] = {2: 'raise'}
    _dub(pack, p, src, shots, song, al, kit)
    stubbed['outcome'] = {}
    r = _dub(pack, p, src, shots, song, al, kit)
    assert stubbed['runs'] == [1, 2, 2] and r.shots['2']['pass']


def test_mask_from_another_crop_is_refused(pack, tmp_path):
    import json
    M, T = pack.nodes_mouth, pack.lsdub_types
    m = tmp_path / 'occ.npz'
    np.savez(m, masks=np.zeros((20, 8, 8), np.uint8))
    json.dump({'face_target': 512.0, 'fps_mult': 2, 'frames': 20}, open(tmp_path / 'occ.json', 'w'))
    masks = T.Masks(dir=str(tmp_path), shots={'1': str(m)})
    s = {'shot': 1}
    assert M.LSDubDub._mask_for(masks, s, M.DEFAULT_TWEAKS, 10)[0] == str(m)
    tw = T.Tweaks(flags={**M.DEFAULT_TWEAKS.flags, 'face_target': 384.0})
    occ, note = M.LSDubDub._mask_for(masks, s, tw, 10)
    assert occ is None and 'different crop' in note


def test_extract_shot_is_frame_exact(pack, tmp_path, clip):
    import cv2
    M = pack.nodes_mouth
    out = M.extract_shot(clip, 17, 26, str(tmp_path / 'take.mp4'))
    cap = cv2.VideoCapture(clip)
    ref = []
    while True:
        ok, f = cap.read()
        if not ok: break
        ref.append(f.astype(np.float32))
    cap = cv2.VideoCapture(out)
    got = []
    while True:
        ok, f = cap.read()
        if not ok: break
        got.append(f.astype(np.float32))
    assert len(got) == 10
    for k in (0, 9):
        d = [np.abs(got[k] - r).mean() for r in ref]
        assert int(np.argmin(d)) == 17 + k
    with pytest.raises(RuntimeError, match='only'):
        M.extract_shot(clip, 45, 60, str(tmp_path / 'late.mp4'))


def test_assemble_raises_and_warns(pack, tmp_path, clip):
    M = pack.nodes_mouth
    short = M.extract_shot(clip, 0, 4, str(tmp_path / 'short.mp4'))
    warn = M.assemble(clip, {(0, 9): short}, str(tmp_path / 'cut.mp4'), 25.0)
    assert any('render has 5 frames, expected 10' in w for w in warn) and os.path.exists(tmp_path / 'cut.mp4')
    with pytest.raises(RuntimeError, match='ffmpeg failed'):
        M.assemble(clip, {}, str(tmp_path / 'no' / 'such' / 'dir' / 'cut.mp4'), 25.0)


def test_content_hashes(pack, tmp_path):
    V = pack.nodes_video
    p = pack.project.Project(root=str(tmp_path), name='cut').ensure()

    class Mem:
        def __init__(self, b): self.b = b
        def get_stream_source(self):
            import io
            return io.BytesIO(self.b)
    assert V.LSDubVideoSlot.fingerprint_inputs(p, Mem(b'aaaa')) != V.LSDubVideoSlot.fingerprint_inputs(p, Mem(b'aaab'))
    a, b = np.zeros((1, 4, 4, 3), np.float32), np.zeros((1, 4, 4, 3), np.float32)
    b[0, 0, 0, 0], b[0, 0, 1, 0] = 0.5, -0.5                                           # same sum, different image
    assert V.LSDubVideoSlot.fingerprint_inputs(p, None, a) != V.LSDubVideoSlot.fingerprint_inputs(p, None, b)
    d = tmp_path / 'refs' / 'alice'
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'1')
    T = pack.lsdub_types
    src, sh = T.Source(path='x', hash='s'), T.Shots(hash='h')
    k1 = V.LSDubCast._key(p, src, sh, None, '', '', str(tmp_path / 'refs'))
    (d / 'b.png').write_bytes(b'22')
    assert V.LSDubCast._key(p, src, sh, None, '', '', str(tmp_path / 'refs')) != k1


def test_occluders_use_only_the_sam2_package():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, 'lsdub_engine', 'core', 'occluders.py')).read()
    assert 'sys.path' not in src and 'custom_nodes' not in src and 'segment-anything' not in src
    assert 'from sam2.build_sam import build_sam2_video_predictor' in src
