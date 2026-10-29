# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v12 the mouth card as a toggle: `lsdub paint --annotate` / mouth_paint.paint(show_mouth_card=True) writes the annotated
video + the lyric check NEXT TO the clean output and never changes it (md5 with and without the toggle, through the CLI's
watermark too); the card draws the weak mark and the heard: line from the track's word facts (never on a skipped
vocalisation); the facts also come from the alignment for a track without a words block. CPU only, no MediaPipe (the
card goes to the band or gets --no-mediapipe)."""
import hashlib, json, os, subprocess

import numpy as np, cv2
import pytest

from lsdub_engine.audits import viseme_overlay as vo
from lsdub_engine.core import mouth_card, mouth_paint, viseme_track as vt
from lsdub_engine.core import vocalisations as vz
from test_lyric_check import align


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def video(path, n=30, W=320, H=240):
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-f', 'lavfi', '-i', 'sine=f=440:d=2', '-shortest', '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p',
                           '-c:a', 'aac', str(path)], stdin=subprocess.PIPE)
    rng = np.random.default_rng(3)
    for i in range(n): pr.stdin.write(np.clip(rng.normal(110, 25, (H, W, 3)), 0, 255).astype(np.uint8).tobytes())
    pr.stdin.close()
    pr.wait()


@pytest.fixture()
def shot(tmp_path):
    """a 30-frame take, a cutout mouth library (sil + mask), and a v12 track (words block) of align()'s 'ask them why'."""
    video(tmp_path / 'take.mp4')
    lib = tmp_path / 'lib'
    lib.mkdir()
    img = np.full((120, 160, 3), 90, np.uint8)
    cv2.ellipse(img, (80, 60), (50, 22), 0, 0, 360, (60, 40, 170), -1)
    m = np.zeros((120, 160), np.uint8)
    cv2.ellipse(m, (80, 60), (54, 26), 0, 0, 360, 255, -1)
    cv2.imwrite(str(lib / 'sil.png'), img)
    cv2.imwrite(str(lib / 'mask.png'), m)
    np.save(lib / 'sil.npy', img)
    np.save(lib / 'mask.npy', m)    # test_mouth_paint_cutout.py swaps the module's loader for
                                                                    # an .npy one at collection, for the whole session
    (tmp_path / 'lyrics.txt').write_text("Don't ask 'em why\n")
    json.dump(align(), open(tmp_path / 'a.json', 'w'))
    vt.main(['--align', str(tmp_path / 'a.json'), '--t0', '0.9', '--frames', '30', '--fps', '25', '--out', str(tmp_path / 't.json')])
    return tmp_path


def paint_argv(d, out, *extra):
    return ['paint', '--mode', 'cutout', '--video', str(d / 'take.mp4'), '--track', str(d / 't.json'), '--lib', str(d / 'lib'),
            '--mouth', '120,120,80,40', '--out', str(out), *extra]


def test_annotate_leaves_the_clean_output_untouched(shot):
    from lsdub_engine import cli
    a, b = shot / 'plain.mp4', shot / 'card.mp4'
    assert cli.main(paint_argv(shot, a)) == 0
    assert cli.main(paint_argv(shot, b, '--annotate', '--annotate-corner', 'band')) == 0
    assert md5(a) == md5(b)                                                          # the clean output, byte for byte
    ann, cj, cm, sg = mouth_card.side_paths(str(b))
    assert ann.endswith('card_annotated.mp4') and all(os.path.exists(p) for p in (ann, cj, cm, sg))
    assert not os.path.exists(mouth_card.annotated_path(str(a)))                     # off by default
    from lsdub_engine import watermark
    assert watermark.verify(ann, frames=8)['watermark'] and watermark.verify(str(b), frames=8)['watermark']
    C = json.load(open(cj))
    assert C['counts']['mismatches'] == 1 and C['mismatches'][0]['heard'] == 'them'
    assert C['mismatches'][0]['lyric'] == "'em" and C['mismatches'][0]['frames'] == [10, 14]
    assert "ask [heard: them] why" in open(sg).read()
    c = cv2.VideoCapture(ann)
    n = 0
    H = None
    while True:
        ok, f = c.read()
        if not ok: break
        n += 1
        H = f.shape[0]
    assert n == 30 and H > 240                                                       # band below the 320x240 picture


def test_a_failing_card_never_costs_the_clean_output(shot, monkeypatch):
    from lsdub_engine import cli, watermark
    a, b = shot / 'plain.mp4', shot / 'card.mp4'
    assert cli.main(paint_argv(shot, a)) == 0
    stale = mouth_card.annotated_path(str(b))
    open(stale, 'w').write('last run')
    def boom(argv):                                                                  # a card that dies half-way
        open(argv[argv.index('--out') + 1], 'w').write('half a video')
        raise RuntimeError('card broke')
    monkeypatch.setattr(vo, 'main', boom)
    assert cli.main(paint_argv(shot, b, '--annotate', '--annotate-corner', 'band')) == 0
    assert md5(a) == md5(b) and watermark.verify(str(b), frames=8)['watermark']                  # watermarked as always
    assert not os.path.exists(stale) and not os.path.exists(mouth_card.side_paths(str(b))[1])    # no stale / half card
    r = mouth_paint.paint(str(shot / 'take.mp4'), str(shot / 't.json'), str(b), mode='cutout', watermark=False, lib=str(shot / 'lib'),
                          args=['--mouth', '120,120,80,40'], show_mouth_card=True, card_corner='band')
    assert r['clean'] == str(b) and r['annotated'] is None and r['lyric_check'] is None                # nothing half-made returned


def test_paint_api_returns_both_paths(shot):
    r = mouth_paint.paint(str(shot / 'take.mp4'), str(shot / 't.json'), str(shot / 'api.mp4'), mode='cutout', watermark=False,
                          lib=str(shot / 'lib'), args=['--mouth', '120,120,80,40'], show_mouth_card=True, card_corner='band')
    assert r['ret'] == 0 and r['clean'] == str(shot / 'api.mp4') and r['annotated'].endswith('api_annotated.mp4')
    assert r['lyric_check'].endswith('api_lyric_check.json') and r['lyric_check_md'] and r['lyrics_suggested']
    r2 = mouth_paint.paint(str(shot / 'take.mp4'), str(shot / 't.json'), str(shot / 'api.mp4'), mode='cutout', watermark=False,
                           lib=str(shot / 'lib'), args=['--mouth', '120,120,80,40'])
    assert r2['annotated'] is None and r2['lyric_check'] is None and r2['clean'] == str(shot / 'api.mp4')


def has(im, col, tol=36):
    return int((np.abs(im.astype(int) - col).max(2) < tol).sum())


def test_card_marks_weak_and_heard_words(shot):
    tv = vo.TrackView(str(shot / 't.json'), 'skip')
    assert tv.words_src == 'track' and tv.wi[12] == 1 and tv.info[12]['diff'] and tv.info[12]['weak']    # 'them'
    c = vo.Card(1.0, line2=True)
    them, _ = c.render(tv, 12)
    ask, _ = c.render(tv, 5)
    why, _ = c.render(tv, 20)
    assert has(them, vo.HEARD) > 150 and has(them, vo.WEAK) > 60                     # heard: line + the weak marks
    assert has(why, vo.HEARD) < 20 and has(why, vo.WEAK) < 20                        # 'why': whisper-timed only (0.7), no mark
                                                                                     # (< 20: anti-aliased glyph edges)
    assert has(ask, vo.HEARD) > 0                                                    # ... but the 1 s timeline shows 'them' ahead
    assert c.H == vo.CARD[1] + vo.LINE2 == 144 and vo.Card(1.0).H == vo.CARD[1] == 128                # v10a's card without facts
    plain, _ = vo.Card(1.0).render(tv, 12)
    assert has(plain, vo.HEARD) < 20 and has(plain, vo.WEAK) < 20
    # a skipped vocalisation never gets the marks: the same frames marked skip
    T = json.load(open(shot / 't.json'))
    q = shot / 'k.json'
    json.dump(vz.mark_track(T, dict(spans=[dict(f0=10, f1=14, t0=1.3, t1=1.5, words=['them'], reasons=['skip-words:them'])],
                                    frames={k: 'skip-words:them' for k in range(10, 15)}, align=None)), open(q, 'w'))
    tk = vo.TrackView(str(q), 'auto')
    assert tk.state(12) == 'skip' and tk.facts(12) is None
    sk, _ = c.render(tk, 12)
    assert has(sk, vo.HEARD) < 20 and has(sk, vo.WEAK) < 20


def test_facts_from_the_alignment_for_an_old_track(shot):
    vt.main(['--align', str(shot / 'a.json'), '--t0', '0.9', '--frames', '30', '--fps', '25', '--out', str(shot / 'old.json'), '--no-words'])
    T = json.load(open(shot / 'old.json'))
    T['align'] = 'elsewhere/a.json'
    json.dump(T, open(shot / 'old.json', 'w'))
    assert vo.TrackView(str(shot / 'old.json')).words_src is None                    # the alignment cannot be found: no facts
    tv = vo.TrackView(str(shot / 'old.json'), align=str(shot / 'a.json'))            # --align
    assert tv.words_src == str(shot / 'a.json') and tv.info[12]['diff'] and tv.info[12]['heard'] == 'them'
    new = vo.TrackView(str(shot / 't.json'))
    assert [x and x['conf'] for x in tv.info] == [x and x['conf'] for x in new.info]


def test_annotate_cli_corner_auto_and_lyric_outputs(shot):
    out = shot / 'o.mp4'
    assert vo.main(['--video', str(shot / 'take.mp4'), '--track', str(shot / 't.json'), '--out', str(out), '--no-mediapipe',
                    '--log', str(shot / 'log.json'), '--lyric-check', str(shot / 'c.json'), '--lyrics-suggested', str(shot / 's.txt')]) == 0
    L = json.load(open(shot / 'log.json'))
    assert L['clips'][0]['placement'] == 'lower-right' and not L['band']
    s = 240 / 720
    assert L['card'] == [round(340 * s), round(144 * s)]               # (340, 128 + 16) at 240p: facts on
    vt.main(['--align', str(shot / 'a.json'), '--t0', '0.9', '--frames', '30', '--fps', '25', '--out', str(shot / 'nw.json'), '--no-words'])
    T = json.load(open(shot / 'nw.json'))
    T['align'] = 'elsewhere/a.json'
    json.dump(T, open(shot / 'nw.json', 'w'))   # not found
    assert vo.main(['--video', str(shot / 'take.mp4'), '--track', str(shot / 'nw.json'), '--out', str(shot / 'o2.mp4'), '--no-mediapipe',
                    '--log', str(shot / 'log2.json')]) == 0
    assert json.load(open(shot / 'log2.json'))['card'] == [round(340 * s), round(128 * s)]   # no facts found: v10a's card size
    assert json.load(open(shot / 'c.json'))['counts']['mismatches'] == 1 and os.path.exists(shot / 'c.md')
    assert open(shot / 's.txt').read() == "Don't ask [heard: them] why\n"
