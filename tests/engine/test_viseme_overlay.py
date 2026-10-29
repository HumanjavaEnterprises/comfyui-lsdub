# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v10a `lsdub annotate` (lsdub/audits/viseme_overlay.py), CPU only, no MediaPipe (--no-mediapipe; the face boxes come
from a keep-out JSON): every viseme class has its own glyph; the card shows the word between its neighbours, a skipped
vocalisation struck through; the FACE RULE: lower-right, else lower-left, else a band below the picture for that clip
(the whole clip); the EDL maps video frames to track frames; the card layer (--overlay-out) is where the card is."""
import json, subprocess
import numpy as np, cv2
import pytest

from lsdub_engine.audits import viseme_overlay as vo
from lsdub_engine.core import vocalisations as vz


def test_every_class_has_its_own_glyph():
    G = {v: vo.glyph(v, (96, 68))[0].astype(int) for v in vo.VIS}
    for v, g in G.items():
        assert g.shape == (68, 96, 3) and (np.abs(g - vo.BG).max(2) > 30).mean() > 0.05, v       # something is drawn
    for a in vo.VIS:
        for b in vo.VIS:
            if a < b: assert np.abs(G[a] - G[b]).mean() > 1.0, (a, b)                              # and they differ
    # dimmed (skip) is darker; rest draws the closed rest mouth whatever the class
    assert vo.glyph('aa', (96, 68), dim=True)[0].astype(int).sum() < G['aa'].sum()
    assert np.array_equal(vo.glyph('aa', (96, 68), rest=True)[0], vo.glyph('sil', (96, 68))[0])


def track(words, vis='aa'):
    return dict(fps=25, t0=0.0, frames=[dict(t=k / 25, vis=vis if w else 'sil', vis2='sil', mix=0.0, open=0.7 if w else 0.0, word=w, phone='')
                                        for k, w in enumerate(words)])


@pytest.fixture()
def tracks(tmp_path):
    w = ['never'] * 6 + ['dance'] * 6 + ['ooh'] * 8 + [''] * 4 + ['day'] * 6
    p = tmp_path / 't.json'
    json.dump(track(w), open(p, 'w'))
    q = tmp_path / 'k.json'
    assert vz.main(['skip', '--track', str(p), '--out', str(q)]) == 0
    return p, q


def test_card_words_and_states(tracks):
    p, q = tracks
    tv = vo.TrackView(str(p), 'auto')
    assert tv.mode == 'mouth' and tv.state(14) == 'ad-lib'      # unmarked: v9, tagged
    assert tv.prev_next(14) == ('dance', 'ooh', 'day') and tv.prev_next(21) == ('ooh', '', 'day')
    tk = vo.TrackView(str(q), 'auto')
    assert tk.mode == 'skip' and tk.state(14) == 'skip' and tk.state(3) == ''
    assert vo.TrackView(str(q), 'rest').state(14) == 'rest'
    c = vo.Card(1.0)
    img, al = c.render(tk, 14)
    assert img.shape == (c.H, c.W, 3) and al.shape == (c.H, c.W)
    plain, _ = c.render(tk, 8)
    def has(im, col, tol=40): return (np.abs(im.astype(int) - col).max(2) < tol).sum()
    assert has(img, vo.TAG['skip']) > 200 and has(plain, vo.TAG['skip']) == 0                    # the skip tag + strike line
    assert has(img, vo.COL['aa'], 30) < has(plain, vo.COL['aa'], 30)                             # skipped: no class colour


def video(path, n=30, W=640, H=360):
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-f', 'lavfi', '-i', 'sine=f=440:d=2', '-shortest', '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p',
                           '-c:a', 'aac', str(path)], stdin=subprocess.PIPE)
    rng = np.random.default_rng(1)
    for i in range(n): pr.stdin.write(np.clip(rng.normal(90, 20, (H, W, 3)), 0, 255).astype(np.uint8).tobytes())
    pr.stdin.close()
    pr.wait()


def run(tmp, faces, extra=(), n=30):
    v = tmp / 'v.mp4'
    if not v.exists(): video(v, n)
    json.dump(dict(frames=[faces(i) for i in range(n)]), open(tmp / 'faces.json', 'w'))
    out = tmp / 'o.mp4'
    log = tmp / 'log.json'
    lay = tmp / 'lay.mov'
    assert vo.main(['--video', str(v), '--track', str(tmp / 't.json'), '--out', str(out), '--no-mediapipe', '--faces', str(tmp / 'faces.json'),
                    '--log', str(log), '--overlay-out', str(lay)] + list(extra)) == 0
    return json.load(open(log)), out, lay


def frames_of(p, rgba=False):
    c = cv2.VideoCapture(str(p))
    F = []
    while True:
        ok, f = c.read()
        if not ok: break
        F.append(f)
    return F


def test_face_rule_lower_right_then_lower_left_then_band(tmp_path, tracks):
    s = 640 / 1280
    _cw, ch, m = round(vo.CARD[0] * s), round(vo.CARD[1] * s), round(vo.MARGIN * s)
    L, out, lay = run(tmp_path, lambda i: [])                                        # no face: lower-right
    assert L['clips'][0]['placement'] == 'lower-right' and not L['band']
    F = frames_of(out)
    assert len(F) == 30 and F[0].shape == (360, 640, 3)
    # a face in the lower right on ONE frame moves the whole clip to the lower left
    L, out, _ = run(tmp_path, lambda i: [{'b': [520, 250, 600, 330], 'src': 'mplm'}] if i == 7 else [])
    assert L['clips'][0]['placement'] == 'lower-left' and L['clips'][0]['blocked_lower_right'] == 1
    # both corners blocked (on different frames): a band below the picture, for the whole clip
    L, out, _ = run(tmp_path, lambda i: [{'b': [520, 250, 600, 330], 'src': 'mplm'}] if i == 3 else ([{'b': [40, 250, 120, 330], 'src': 'mplm'}] if i == 9 else []))
    assert L['clips'][0]['placement'] == 'band' and L['band'] and L['out_size'][1] == 360 + ch + 2 * m
    F = frames_of(out)
    assert F[0].shape[0] == 360 + ch + 2 * m
    # weak / advisory boxes (a low YuNet score, a person matte) do not move the card; --faces-strict counts them
    weak = lambda i: [{'b': [520, 250, 600, 330], 'src': 'yunet', 'score': 0.4}, {'b': [40, 250, 120, 330], 'src': 'matte'}]
    assert run(tmp_path, weak)[0]['clips'][0]['placement'] == 'lower-right'
    assert run(tmp_path, weak, ['--faces-strict'])[0]['clips'][0]['placement'] == 'band'
    assert run(tmp_path, lambda i: [], ['--corner', 'band'])[0]['clips'][0]['placement'] == 'band'


def test_card_layer_sits_in_the_corner(tmp_path, tracks):
    L, out, lay = run(tmp_path, lambda i: [])
    al = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(lay), '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'], capture_output=True).stdout
    A = np.frombuffer(al, np.uint8).reshape(-1, 360, 640, 4)[..., 3]
    assert len(A) == 30
    ys, xs = np.nonzero(A[14] > 32)
    s = 640 / 1280
    assert xs.min() >= 640 - round((vo.CARD[0] + vo.MARGIN) * s) - 1 and ys.min() >= 360 - round((vo.CARD[1] + vo.MARGIN) * s) - 1
    assert xs.max() <= 640 - round(vo.MARGIN * s) and A[14, :150].max() == 0                        # nothing elsewhere


def test_edl_maps_video_frames_to_track_frames(tmp_path, tracks):
    p, q = tracks
    video(tmp_path / 'v.mp4', 30)
    E = dict(segments=[dict(label='A', track=str(q), frames=[0, 0, 1, 1, 14, 14, 14] + list(range(23))), dict(label='B', track=None, nf=0)])
    json.dump(E, open(tmp_path / 'e.json', 'w'))
    class A_: pass
    a = A_()
    a.edl = str(tmp_path / 'e.json')
    a.vocalisations = 'auto'
    a.skip_words = None
    a.lyrics = None
    a.track = None
    segs = vo.segments_of(a, 30)
    assert segs[0][2][:7] == [0, 0, 1, 1, 14, 14, 14] and len(segs[0][2]) == 30 and segs[0][1][0].label == 'A'
    assert segs[1][1] == [] and segs[1][2] == []
    assert vo.main(['--video', str(tmp_path / 'v.mp4'), '--edl', str(tmp_path / 'e.json'), '--out', str(tmp_path / 'o.mp4'),
                    '--no-mediapipe']) == 0
    assert len(frames_of(tmp_path / 'o.mp4')) == 30


def test_annotate_default_is_skip_like_paint(tmp_path, tracks, monkeypatch):
    # v11: lsdub paint skips vocalisations by default, so the card's default describes that render: an UNMARKED track's
    # 'ooh' (found here by the same lexicon) is struck through as 'skip'; mouth keeps v9's 'ad-lib'
    p, q = tracks
    tv = vo.TrackView(str(p))
    assert tv.mode == 'skip' and tv.state(14) == 'skip' and tv.state(3) == ''
    assert vo.TrackView(str(p), 'mouth').state(14) == 'ad-lib'
    seen, real = [], vo.segments_of
    monkeypatch.setattr(vo, 'segments_of', lambda a, n: (seen.append(a.vocalisations), real(a, n))[1])
    video(tmp_path / 'v.mp4', 30)
    assert vo.main(['--video', str(tmp_path / 'v.mp4'), '--track', str(p), '--out', str(tmp_path / 'o.mp4'), '--no-mediapipe']) == 0
    assert seen == ['skip']


def test_keepout_gating_matches_the_studio_gate():
    assert vo.gating({'b': [0, 0, 1, 1], 'src': 'mplm'}) and vo.gating({'b': [0, 0, 1, 1]}) and vo.gating({'b': [0, 0, 1, 1], 'src': 'mpfd', 'score': 0.6})
    assert not vo.gating({'b': [0, 0, 1, 1], 'src': 'mpfd', 'score': 0.4}) and not vo.gating({'b': [0, 0, 1, 1], 'src': 'yunet', 'score': 0.85})
    assert not vo.gating({'b': [0, 0, 1, 1], 'src': 'matte'}) and not vo.gating({'b': [0, 0, 1, 1], 'src': 'mplm_lo'})
    # padding as the studio gate: 10 % to the sides and below, 16 % above
    assert vo.overlaps((0, 0, 10, 10), [[0, 11, 10, 21]], 0.1) and not vo.overlaps((0, 0, 10, 10), [[0, 12, 10, 22]], 0.1)
    assert vo.overlaps((0, 0, 10, 10), [[10, 0, 20, 10]], 0.1) and not vo.overlaps((0, 0, 10, 10), [[11, 0, 21, 10]], 0.1)


def test_cli_knows_annotate():
    from lsdub_engine import cli
    assert cli.CMDS['annotate'] == 'audits.viseme_overlay' and 'lsdub annotate' in cli.__doc__
