# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""LSDub Dub's mouth_card toggle: off by default and, off, the node runs and fingerprints exactly as before; on, each
dubbed shot gets the engine's mouth card beside its render (the render is only read, byte for byte), the card cut comes
out of card_video, and the clean cut and its key do not move."""
import hashlib
import json
import os
import subprocess

import numpy as np
import pytest

from test_dub import _dub, _setup, stubbed  # noqa: F401  (test_dub's fixture, used by `cards`)


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def test_the_toggle_is_off_by_default_and_has_its_own_output(pack):
    s = pack.nodes_mouth.LSDubDub.define_schema()
    card = next(i for i in s.inputs if i.id == 'mouth_card')
    assert card.io_type == 'BOOLEAN' and card.kw['default'] is False and card.optional
    assert [o.display_name for o in s.outputs] == ['render', 'video', 'card_video']


@pytest.fixture
def cards(pack, monkeypatch, stubbed):  # noqa: F811  (the imported fixture, requested by name)
    """the dub stubbed (test_dub's stubbed), plus the card step and the watermark: shot_card writes a file per shot."""
    M = pack.nodes_mouth
    log = {'cards': [], 'marked': []}

    def shot_card(cls, project, source, s, st, song, align, kit, tw, vocalisations, render):
        log['cards'].append(s['shot'])
        p = render[:-4] + '_annotated.mp4'
        open(p, 'w').write(f"card-{s['shot']}")
        return p
    monkeypatch.setattr(M.LSDubDub, 'shot_card', classmethod(shot_card))
    import importlib
    WM = importlib.import_module(pack.__name__ + '.lsdub_engine.watermark')
    monkeypatch.setattr(WM, 'apply_inplace', lambda path, **kw: log['marked'].append(path))
    log['dub'] = stubbed
    return log


def test_off_runs_and_fingerprints_as_before(pack, tmp_path, cards):
    p, src, shots, song, al, kit = _setup(pack, tmp_path)
    D = pack.nodes_mouth.LSDubDub
    out = D.execute(p, src, shots, song, al, kit).args
    assert cards['cards'] == [] and out[2] == out[1] == out[0].cut                  # no card; card_video = the cut
    assert D.fingerprint_inputs(p, src, shots, song, al, kit) == D.fingerprint_inputs(p, src, shots, song, al, kit, mouth_card=False)
    assert D.fingerprint_inputs(p, src, shots, song, al, kit, mouth_card=True) != D.fingerprint_inputs(p, src, shots, song, al, kit)


def test_on_cards_each_delivered_shot_and_keeps_the_clean_cut(pack, tmp_path, cards):
    p, src, shots, song, al, kit = _setup(pack, tmp_path)
    off = _dub(pack, p, src, shots, song, al, kit)
    cards['dub']['outcome'] = {}
    out = pack.nodes_mouth.LSDubDub.execute(p, src, shots, song, al, kit, mouth_card=True).args
    rec = out[0]
    assert rec.hash == off.hash and out[1] == rec.cut == off.cut                    # the clean cut and its key unchanged
    assert cards['dub']['runs'] == [1, 2]                                          # the toggle re-dubs nothing
    assert cards['cards'] == [1, 2] and all(rec.shots[k]['card'].endswith('_annotated.mp4') for k in ('1', '2'))
    assert out[2] != rec.cut and out[2].endswith('card_cut.mp4') and cards['marked'] == [out[2]]
    assert cards['dub']['cuts'][-1] == [(0, 9), (10, 19)]                          # the card cut: both shots from their cards


def test_a_shot_left_as_source_gets_no_card(pack, tmp_path, cards):
    p, src, shots, song, al, kit = _setup(pack, tmp_path)
    cards['dub']['outcome'] = {1: 'fail'}
    rec = _dub(pack, p, src, shots, song, al, kit, mouth_card=True)
    assert cards['cards'] == [2] and 'card' not in rec.shots['1']


def test_a_failing_card_cut_falls_back_to_the_clean_cut(pack, tmp_path, cards, monkeypatch):
    import importlib
    WM = importlib.import_module(pack.__name__ + '.lsdub_engine.watermark')

    def boom(path, **kw): raise RuntimeError('ffmpeg died')
    monkeypatch.setattr(WM, 'apply_inplace', boom)
    p, src, shots, song, al, kit = _setup(pack, tmp_path)
    out = pack.nodes_mouth.LSDubDub.execute(p, src, shots, song, al, kit, mouth_card=True).args
    assert out[2] == out[1] == out[0].cut


def _noise(path, n=30, W=320, H=240):
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', str(path)], stdin=subprocess.PIPE)
    rng = np.random.default_rng(3)
    for _ in range(n): pr.stdin.write(np.clip(rng.normal(110, 25, (H, W, 3)), 0, 255).astype(np.uint8).tobytes())
    pr.stdin.close()
    pr.wait()


def _align():
    """three sung words, the middle one whisper-timed with a low probability (a weak word)."""
    def w(word, t0, t1, ph, **kw):
        step = (t1 - t0) / len(ph)
        return dict(w=word, start=t0, end=t1, src='lyric', heard=word, p_whisper=0.99, align='mfa',
                    phones=[dict(p=x, start=t0 + i * step, end=t0 + (i + 1) * step) for i, x in enumerate(ph)], **kw)
    W = [w('ask', 1.0, 1.3, ['AE', 'S', 'K'], idx=0), w('them', 1.3, 1.5, ['DH', 'EH', 'M'], idx=1),
         w('why', 1.5, 2.0, ['W', 'AY'], idx=2)]
    W[1].update(p_whisper=0.29, align='whisper+spread')
    return dict(aligner='best', words=W, phrases=[dict(phrase='ask them why', seg=[0.8, 2.2], method='sphinx (best)')])


def test_shot_card_runs_the_engine_card_and_only_reads_the_render(pack, tmp_path, monkeypatch):
    import importlib
    T, P = pack.lsdub_types, pack.project
    VO = importlib.import_module(pack.__name__ + '.lsdub_engine.audits.viseme_overlay')
    MC = importlib.import_module(pack.__name__ + '.lsdub_engine.core.mouth_card')
    monkeypatch.setattr(VO, 'face_boxes', lambda frames, n, *a, **kw: [[] for _ in range(n)])   # no face model in tests
    p = P.Project(root=str(tmp_path), name='cut').ensure()
    st = p.shot(1, 'k')
    render = st.path('render.mp4')
    _noise(render)
    json.dump(_align(), open(tmp_path / 'a.json', 'w'))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=f=220:d=3', str(tmp_path / 'lead.wav')], check=True)
    src = T.Source(path=render, hash='src', fps=25.0)
    s = {'shot': 1, 'f0': 0, 'f1': 29, 'start': 0.9, 'end': 2.1, 'verdict': 'DUB', 'target': 'a'}
    song = T.Song(mix=str(tmp_path / 'lead.wav'), lead=str(tmp_path / 'lead.wav'), hash='song')
    tw = pack.nodes_mouth.DEFAULT_TWEAKS
    before = md5(render)
    ann = pack.nodes_mouth.LSDubDub.shot_card(p, src, s, st, song, T.Align(align=str(tmp_path / 'a.json'), hash='al'),
                                              T.Kit(dir=str(tmp_path), identity='a'), tw, 'skip', render)
    assert ann == MC.annotated_path(render) and os.path.exists(ann)
    assert md5(render) == before                                                   # the render, byte for byte
    import cv2
    c, n = cv2.VideoCapture(ann), 0
    while c.read()[0]: n += 1
    assert n == 30
    check = json.load(open(MC.side_paths(render)[1]))
    assert check['counts']['words'] >= 3 and check['counts']['weak'] >= 1          # the word facts reached the card
    calls = []
    monkeypatch.setattr(MC, 'after_paint', lambda a: calls.append(a) or {})
    assert pack.nodes_mouth.LSDubDub.shot_card(p, src, s, st, song, T.Align(align=str(tmp_path / 'a.json'), hash='al'),
                                               T.Kit(dir=str(tmp_path), identity='a'), tw, 'skip', render) == ann
    assert calls == []                                                             # newer than the render: reused
