# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v9 per-frame gate, the MAKER + CONFIRMATION (lsdub/core/frame_gate.py) on a synthetic clip, CPU only. The clip's
input track is HELD for its first frames (no detection) while the head drifts -- the mv30 S2 / S5 / S8 starts in
miniature: the warp moves the wrong pixels there. A fake detector stands in for MediaPipe (it measures the synthetic
lips). With --warp-gate off the render is the batch's bit-for-bit; render_frame(i, {}) reproduces every batch frame;
repair re-makes the failing frames (FAIL -> not FAIL) without making their neighbours fail."""
import json, os, subprocess, sys
import numpy as np, cv2
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_frame_gate import face, draw, LIP, CAV
from lsdub_engine.core import mouth_paint, mouth_warp as mw, frame_gate as fgm


def detect(img, box=None):
    """the fake landmark detector: the lips are the red pixels, the opening the dark ones (centre column)."""
    d = np.abs(img.astype(int) - np.array(LIP)).max(2) < 40
    ys, xs = np.nonzero(d)
    if len(xs) < 50: return None
    c = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
    col = np.abs(img[:, int(c[0])].astype(int) - np.array(CAV)).max(1) < 30
    g = np.nonzero(col)[0]
    gap = float(g.max() - g.min() + 1) if len(g) else 1.0
    return face(gap, c=(float(c[0]), float(c[1])))


def make_clip(tmp, n=30, held=6, drift=2.5):
    gaps = 3 + 2 * np.sin(np.arange(n) / 5.0)                                   # a mostly shut take
    S = np.array([face(g, c=(160.0, 150.0 + drift * min(t, 10))) for t, g in enumerate(gaps)])
    frames = [draw(P) for P in S]
    raw = S.copy()
    raw[:held] = S[held]         # the input track held for the first frames
    vid = str(tmp / 'take.mp4')
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '320x300', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', vid], stdin=subprocess.PIPE)
    for f in frames: pr.stdin.write(f.tobytes())
    pr.stdin.close()
    pr.wait()
    np.savez(str(tmp / 'trk.npz'), S3=raw, fps=25.0, found=1.0, n=n, video=vid)
    vis = dict(frames=[dict(vis='aa', vis2='aa', mix=0.0, open=0.9) for _ in range(n)])       # the audio asks for an open mouth
    json.dump(vis, open(str(tmp / 'vis.json'), 'w'))
    return vid, frames, S


def args(tmp, vid, n, extra=()):
    cap = []
    run = mw.run
    mw.run = lambda a: cap.append(a)
    try:
        mouth_paint.main(['--mode', 'warp', '--video', vid, '--track', str(tmp / 'vis.json'), '--frames', str(n), '--out', str(tmp / 'o.mp4'),
                          '--warp-track-cache', str(tmp / 'trk.npz'), '--warp-json', str(tmp / 'w.json'), '--mask-out', str(tmp / 'm.npz')] + list(extra))
    finally:
        mw.run = run
    return cap[0]


def frames_of(p):
    c = cv2.VideoCapture(p)
    F = []
    while True:
        ok, f = c.read()
        if not ok: break
        F.append(f)
    return F


@pytest.fixture(scope='module')
def clip_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('gate')
    vid, frames, S = make_clip(tmp)
    return tmp, vid, frames, S


def test_render_frame_reproduces_the_batch_and_gate_off_is_v8(clip_dir):
    tmp, vid, frames, S = clip_dir
    n = len(frames)
    a = args(tmp, vid, n)
    clip = mw.WarpClip(a).run_batch()
    for i in range(n):
        out, mask, ent, P = clip.render_frame(i, {})
        assert np.array_equal(out, clip.outs[i]) and np.array_equal(mask, clip.masks[i]), i
    # --warp-gate off: the same file as without the flag (the v8 path)
    a0 = args(tmp, vid, n)
    a0.out = str(tmp / 'v8.mp4')
    mw.run(a0)
    a1 = args(tmp, vid, n, ['--warp-gate', 'off'])
    a1.out = str(tmp / 'off.mp4')
    mw.run(a1)
    F0, F1 = frames_of(a0.out), frames_of(a1.out)
    assert len(F0) == n and all(np.array_equal(x, y) for x, y in zip(F0, F1))


def test_repair_remakes_the_held_frames_without_failing_neighbours(clip_dir):
    tmp, vid, frames, S = clip_dir
    n = len(frames)
    a = args(tmp, vid, n, ['--warp-gate', 'repair', '--warp-gate-report', str(tmp / 'g.json')])
    clip = mw.WarpClip(a).run_batch()
    before = [o.copy() for o in clip.outs]
    res = fgm.gate(clip, a, lmk=detect)['gate']
    fb = {f['i']: f['verdict'] for f in json.load(open(str(tmp / 'g.json')))['before']}
    fa = {f['i']: f['verdict'] for f in json.load(open(str(tmp / 'g.json')))['frames']}
    failed = [i for i in range(n) if fb[i] == 'FAIL']
    assert failed and min(failed) == 0, fb                                # the held start fails ...
    for i in failed: assert fa[i] != 'FAIL', (i, fa[i], res['repairs'])    # ... and is re-made into a frame that does not
    assert all(fa[i] != 'FAIL' for i in range(n)), fa
    assert res['chosen'] and res['after']['counts']['FAIL'] == 0
    # the whole HELD input run is re-made by lm_hold, its non-failing frames too (no held / re-detected seam inside it)
    raw = np.load(str(tmp / 'trk.npz'))['S3']
    held_run = list(range(0, max(t for t in range(1, n) if np.array_equal(raw[t], raw[t - 1])) + 1))
    assert all(clip.log[t].get('gate_cand') == 'a:lm_hold' for t in held_run), [(t, clip.log[t].get('gate_cand')) for t in held_run]
    assert any(set(r.get('remade', [])) >= set(held_run) for r in res['repairs']), res['repairs']
    changed = [i for i in range(n) if not np.array_equal(before[i], clip.outs[i])]
    assert set(changed) <= set(held_run) | set(range(0, max(failed) + 3)), changed   # only the held / failing run (+ edge)
    # the re-made frames carry the take's real mouth: the output mouth centre is back on the take's
    for i in failed:
        P = detect(clip.outs[i])
        assert P is not None and abs(P[13, 1] - S[i][13, 1]) < 6, (i, P[13], S[i][13])
    # the final plan beside the report (the offline checker's --plan): the re-made frames' placement + input landmarks
    z = np.load(str(tmp / 'g_plan.npz'))
    assert z['P_out'].shape == (n, 478, 2) and z['S_in'].shape == (n, 478, 3) and len(z['held_fixed']) == n
    for i in held_run:
        assert np.allclose(z['P_out'][i], np.asarray(clip.P_out[i], float)[:, :2]) and np.allclose(z['S_in'][i], clip.gate_ov[i]['S'])
        assert z['held_fixed'][i], i


def test_low_order_fits_pass_through_the_anchors():
    class C: pass
    A = np.zeros((12, 2))
    A[:, 0] = np.arange(12) ** 1.5
    A[5:8, 1] = [3.0, -3.0, 3.0]      # a curve + an oscillation
    for deg in (1, 2):
        out = fgm.interp_params(C(), list(range(3, 10)), 2, 10, dict(x=A), deg)
        f = np.array([out[t]['x'] for t in range(3, 10)])
        # continuous with both anchors (the fit at the anchor frames would be the anchor values) and no oscillation left
        lo, hi = A[2], A[10]
        u = (np.arange(3, 10) - 2) / 8.0
        if deg == 1: assert np.allclose(f, (1 - u)[:, None] * lo + u[:, None] * hi)
        assert np.all(np.abs(np.diff(f[:, 1], 2)) < 1e-6) or deg == 2
        assert np.ptp(np.diff(f[:, 1])) < 3.0


def test_slosh_window_is_tried_and_only_kept_when_it_helps(tmp_path):
    n = 36
    gaps = 3 + 2 * np.sin(np.arange(n) / 5.0)
    S = np.array([face(g) for g in gaps])
    frames = [draw(P) for P in S]
    raw = S.copy()
    for t in range(12, 25): raw[t][:, 0] += 10.0 * np.sin(2 * np.pi * (t - 12) / 6.0)   # the input track wobbles (6-frame
    # period: a one-frame alternation would be cancelled by the pipeline's own k5 landmark smoothing)
    vid = str(tmp_path / 'take.mp4')
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '320x300', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', vid], stdin=subprocess.PIPE)
    for f in frames: pr.stdin.write(f.tobytes())
    pr.stdin.close()
    pr.wait()
    np.savez(str(tmp_path / 'trk.npz'), S3=raw, fps=25.0, found=1.0, n=n, video=vid)
    json.dump(dict(frames=[dict(vis='aa', vis2='aa', mix=0.0, open=0.9) for _ in range(n)]), open(str(tmp_path / 'vis.json'), 'w'))
    a = args(tmp_path, vid, n, ['--warp-gate', 'repair', '--warp-gate-report', str(tmp_path / 'g.json')])
    clip = mw.WarpClip(a).run_batch()
    res = fgm.gate(clip, a, lmk=detect)['gate']
    wins = res['windows']
    assert res['before']['slosh_windows'], res['before']                          # the sloshing window is found
    assert wins and all(w['window'][0] <= 24 and w['window'][1] >= 12 for w in wins[:1])
    for w in wins:
        assert len(w.get('tried', [])) in (0, 2)                                    # linear and quadratic both tried
        for t in w.get('tried', []): assert set(t) >= {'deg', 'energy', 'structure', 'viseme', 'no_fail'}
        if w.get('chosen'):                                                         # kept only when it helped
            assert w['energy_after'] < w['energy_before'] and any(t['structure'] and t['no_fail'] for t in w['tried'])
            z = np.load(str(tmp_path / 'g_plan.npz'))                                # the plan knows the fitted input
            for f in w['frames']: assert np.allclose(z['S_in'][f], clip.gate_ov[f]['S']), f
        elif (w.get('fallback') or {}).get('chosen'):
            assert w['fallback']['energy_after'] < w['energy_before']


def wobble_clip(tmp_path, closures=()):
    """only the MOUTH of the input track wobbles sideways (frames 12-24, 6-frame period): a residual beyond the gap /
    width the warp must follow, so the drawn layers slosh against the take's texture. closures: frames the audio
    closes (viseme PP)."""
    from lsdub_engine.core import lip_layers as ll
    n = 36
    gaps = 3 + 2 * np.sin(np.arange(n) / 5.0)
    S = np.array([face(g) for g in gaps])
    frames = [draw(P) for P in S]
    raw = S.copy()
    mouth = sorted(set(ll.LIP_OUT) | set(ll.LIP_IN) | {13, 14, 0, 17, 61, 291})
    for t in range(12, 25): raw[t][mouth, 0] += 4.0 * np.sin(2 * np.pi * (t - 12) / 6.0)
    vid = str(tmp_path / 'take.mp4')
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '320x300', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', vid], stdin=subprocess.PIPE)
    for f in frames: pr.stdin.write(f.tobytes())
    pr.stdin.close()
    pr.wait()
    np.savez(str(tmp_path / 'trk.npz'), S3=raw, fps=25.0, found=1.0, n=n, video=vid)
    vis = [dict(vis='PP', vis2='PP', mix=0.0, open=0.0) if t in closures else dict(vis='aa', vis2='aa', mix=0.0, open=0.9) for t in range(n)]
    json.dump(dict(frames=vis), open(str(tmp_path / 'vis.json'), 'w'))
    a = args(tmp_path, vid, n, ['--warp-gate', 'repair', '--warp-gate-report', str(tmp_path / 'g.json')])
    return n, a, mw.WarpClip(a).run_batch()


def test_slosh_fit_removes_a_residual_wobble_and_keeps_the_structure(tmp_path):
    """(g) re-fits the residual through the window's PASS anchors: the slosh energy collapses, the outline's gap / width
    per frame stay where they were, the plan knows the fitted input landmarks, and no frame outside the window changes."""
    n, a, clip = wobble_clip(tmp_path)
    before = [o.copy() for o in clip.outs]
    P0 = [np.asarray(p, float).copy() for p in clip.P_out]
    res = fgm.gate(clip, a, lmk=detect)['gate']
    w = next(x for x in res['windows'] if x.get('chosen'))
    assert w['chosen'] == 'g:slosh_fit' and w['energy_after'] < 0.5 * w['energy_before'], w
    assert not res['after']['slosh_windows'] and res['after']['counts']['FAIL'] == 0
    for f in w['frames']:                                                         # the structure: gap / width per frame
        for i, j in ((13, 14), (61, 291)):
            d0 = np.linalg.norm(P0[f][i, :2] - P0[f][j, :2])
            d1 = np.linalg.norm(np.asarray(clip.P_out[f])[i, :2] - np.asarray(clip.P_out[f])[j, :2])
            assert abs(d1 - d0) < 1.0, (f, i, j, d0, d1)
    z = np.load(str(tmp_path / 'g_plan.npz'))
    for f in w['frames']: assert np.allclose(z['S_in'][f], clip.gate_ov[f]['S']), f
    changed = [i for i in range(n) if not np.array_equal(before[i], clip.outs[i])]
    assert changed and set(changed) <= set(w['frames']), (changed, w['frames'])


def test_slosh_fit_never_remakes_a_closure(tmp_path):
    """a closure (viseme PP) inside the sloshing window is mouth structure: it is not re-made, it anchors the fit
    (piecewise: PASS anchor .. closure .. PASS anchor) and the window still loses its slosh."""
    n, a, clip = wobble_clip(tmp_path, closures=(17, 18))
    before = [o.copy() for o in clip.outs]
    res = fgm.gate(clip, a, lmk=detect)['gate']
    w = next(x for x in res['windows'] if x.get('chosen'))
    assert w['closures'] == [17, 18] and 17 in w['anchors'] and 18 in w['anchors'] and not {17, 18} & set(w['frames']), w
    assert w['energy_after'] < 0.5 * w['energy_before'], w
    for t in (17, 18): assert np.array_equal(before[t], clip.outs[t]) and 'gate_cand' not in clip.log[t], t


def test_warp_soft_is_never_compounded():
    """an unresolved frame keeps its best candidate and is offered the candidates again in later rounds: warp_soft
    composes with the committed override, so without a guard 0.85 of the residual would become 0.85^3 by round 3."""
    assert fgm.cand_warp_soft(None, None, 0, {}, {'soft': True, 'dFe': np.ones(2)}) is None


def test_unresolved_none_leaves_an_unfixable_frame_as_round_0(clip_dir):
    """--warp-gate-unresolved none: a failing frame no candidate fixes stays bit-identical to round 0 (with the default,
    best, the best-scoring candidate is kept when it beats the original). warp_soft alone cannot fix a held input track."""
    tmp, vid, frames, S = clip_dir
    n = len(frames)
    outs = {}
    for mode in ('best', 'none'):
        a = args(tmp, vid, n, ['--warp-gate', 'repair', '--warp-gate-cands', 'b', '--warp-gate-unresolved', mode, '--warp-gate-rounds', '1'])
        clip = mw.WarpClip(a).run_batch()
        before = [o.copy() for o in clip.outs]
        res = fgm.gate(clip, a, lmk=detect)['gate']
        assert res['unresolved'], res                                               # held input: warp_soft cannot fix it
        outs[mode] = [i for i in range(n) if not np.array_equal(before[i], clip.outs[i])]
    assert outs['none'] == [], outs


# ------------------------------------------------------------------------------- v11 --
def test_held_fix_never_feeds_a_false_face_fit_to_the_warp(clip_dir, monkeypatch):
    # mv30 v10d S3 s1234: one IMAGE-mode re-detection locked onto a false face 0.8 IOD up; held_fix substituted it and the
    # k5 smoothing spread it over 5 frames -> the gate's lm_hold drew the mouth at the nostrils. v11: it is never used.
    from test_frame_gate import _false_fit
    tmp, vid, frames, S = clip_dir
    n = len(frames)
    a = args(tmp, vid, n, ['--warp-gate', 'repair'])
    clip = mw.WarpClip(a).run_batch()
    k = 3
    def det_bad(img, box=None):
        p = detect(img, box)
        return _false_fit(p) if p is not None and np.array_equal(img, clip.frames[k]) else p
    ck = fgm.make_checker(clip, a, lmk=det_bad)
    assert [x['i'] for x in ck.redet_rejected] == [k] and np.isnan(ck.take_lm[k]).all()
    S_fix, bad, plf = fgm.held_fix(clip, ck)
    io = 120.0
    lips = lambda X: X[:, [13, 14, 61, 291], :2].mean(1)
    assert np.abs(lips(S_fix)[:k + 3] - lips(S)[:k + 3]).max() < 0.15 * io      # every frame near the real lips
    # the same with the plausibility check disabled (the v9 behaviour): the false fit reaches the repaired track
    monkeypatch.setattr(fgm.chk, 'REDET', dict(skull=99.0, spread=(0.0, 99.0), jump=99.0))
    ck2 = fgm.make_checker(clip, a, lmk=det_bad)
    S_bad, _, _ = fgm.held_fix(clip, ck2)
    assert np.abs(lips(S_bad)[k] - lips(S)[k]).max() > 0.25 * io


LAYERS = ('--warp-teeth rigid --warp-teeth-anchor skin --warp-teeth-jaw plan --warp-compose layers --warp-teeth-chin-pin arc '
          '--warp-tongue wedge --warp-cover fill --warp-layers-smooth 1.0').split()


def test_layers_off_candidate_and_the_candidate_order(clip_dir):
    tmp, vid, frames, S = clip_dir
    n = len(frames)
    assert fgm.ORDER.index('h') > fgm.ORDER.index('e') and 'f' not in fgm.ORDER and 'h' in fgm.CANDS
    a0 = args(tmp, vid, n)
    c0 = mw.WarpClip(a0).run_batch()
    assert fgm.cand_layers_off(c0, None, 5, {}, {}) is None                       # no layer stack in this render: nothing to turn off
    a = args(tmp, vid, n, LAYERS)
    clip = mw.WarpClip(a).run_batch()
    assert clip.rigid_teeth
    ov, how, extra = fgm.cand_layers_off(clip, None, 10, {}, {})
    assert ov['layers_off'] and how == 'plain warp' and fgm.cand_layers_off(clip, None, 10, {}, ov) is None
    out, mask, ent, P = clip.render_frame(10, ov)                                 # the plain own-pixel warp: no layer stack
    assert ent.get('layers_off') and 'reg' not in ent and np.array_equal(out, clip.warped[10])
    assert not np.array_equal(clip.outs[10], clip.warped[10])                     # (the batch frame had the stack)
    out2, *_ = clip.render_frame(10, {})
    assert np.array_equal(out2, clip.outs[10])
    # h is tried after a..e and before f (repair_run's order), wherever it sits in --warp-gate-cands
    keys = sorted((k for k in 'h,a,f,e'.split(',') if k in fgm.CANDS), key=fgm.ORDER.index)
    assert keys == ['a', 'e', 'h']


@pytest.mark.skipif(not os.path.exists(os.environ.get("LIPS_FACE_MODEL", os.path.expanduser("~/.cache/lsdub/face_landmarker.task"))),
                    reason="needs the MediaPipe face model (downloaded on first use)")
def test_gate_reports_the_registration_ruler(clip_dir):
    tmp, vid, frames, S = clip_dir
    n = len(frames)
    a = args(tmp, vid, n, LAYERS + ['--warp-gate', 'report', '--warp-gate-reg', '--warp-gate-report', str(tmp / 'gr.json')])
    mw.run(a)
    W = json.load(open(a.warp_json))
    json.load(open(str(tmp / 'gr.json')))
    reg = W['gate']['reg']
    assert reg['repaired'] is False and 'slosh_sum' in reg['summary'] and set(reg['offsets']) == {'up_lip', 'lo_lip', 'cav_open', 'tip_lo', 'bite'}
    assert all('reg' in f for f in W['frames']) and W['teeth']['layers']['reg_meta']['iod'] > 0


def test_h_guard_allows_h_only_near_closed_or_between_own_frames(clip_dir):
    """v12 --warp-gate-h-guard: h (layers_off) only where the planned opening is near-closed (<= the take's p5 + the closure
    band) or where both neighbours show the take's own interior; a one-frame switch of the interior's look is counted."""
    tmp, vid, frames, S = clip_dir
    n = len(frames)
    clip = mw.WarpClip(args(tmp, vid, n, LAYERS + ['--warp-gate-h-guard'])).run_batch()
    t = 10
    r = clip.model.rng
    gs = r['g_hi'] - r['g_lo']
    thr = fgm.h_closed_thr(clip)
    assert abs(thr - (np.percentile((clip.F[:, 0] - r['g_lo']) / gs, 5) + 0.12)) < 1e-9
    assert (clip.Fe[t][0] - r['g_lo']) / gs > thr and not fgm.near_closed(clip, t)       # the audio asks for 'aa': open
    assert fgm.cand_layers_off(clip, None, t, {}, {}) is None                            # open, layered neighbours: no h
    for j in (t - 1, t + 1): clip.log[j]['layers_off'] = True                            # both neighbours show the take's own
    assert fgm.cand_layers_off(clip, None, t, {}, {})[0]['layers_off']
    for j in (t - 1, t + 1): clip.log[j].pop('layers_off')
    # a run being re-made together: its members count for each other only when they qualify themselves
    assert not fgm.h_allowed(clip, t, {'run': [t - 1, t, t + 1]})
    Fe0 = clip.Fe.copy()
    clip.Fe[t, 0] = r['g_lo']                                       # a planned closure: near-closed
    assert fgm.near_closed(clip, t) and fgm.cand_layers_off(clip, None, t, {}, {}) is not None
    clip.Fe[:] = Fe0
    # the style-continuity term: a layers-off frame between two layered ones is one switch; between own ones, none
    assert fgm.style_switches(clip, {t: ({'layers_off': True}, {})}) == 1
    assert fgm.style_switches(clip, {t: ({}, {})}) == 0
    assert fgm.style_switches(clip, {t - 1: ({'layers_off': True}, {}), t: ({'layers_off': True}, {}), t + 1: ({'layers_off': True}, {})}) == 0
    # without the flag: v11's h, unchanged
    c11 = mw.WarpClip(args(tmp, vid, n, LAYERS)).run_batch()
    assert fgm.cand_layers_off(c11, None, t, {}, {})[0]['layers_off']
