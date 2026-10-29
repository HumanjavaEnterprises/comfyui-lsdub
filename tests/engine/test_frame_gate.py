"""v9 per-frame gate, the CHECKER (lsdub/audits/frame_gate.py): synthetic sequences, CPU only, no MediaPipe (the
landmarks are given). A smooth sequence passes; an injected one-frame jump, a warp fold, a ghost lip, a lip colour
flicker, a held input track and a sloshing lip texture are each flagged with their own cause."""
import numpy as np, cv2

from lsdub_engine.audits import frame_gate as fg
from lsdub_engine.core import lip_layers as ll

SKIN = (120, 155, 205)
LIP = (55, 45, 175)
CAV = (25, 18, 40)


class Pose:
    """identity rigid pose for n frames (canonical = frame px)."""

    def __init__(self, n):
        self.s = np.ones(n)
        self.R = np.tile(np.eye(3), (n, 1, 1))
        self.t = np.zeros((n, 3))
        self.par = np.zeros((n, 7))

    def rigid3(self, i, X3): return np.asarray(X3, float) * self.s[i] @ self.R[i].T + self.t[i]
    def rigid(self, i, X3): return self.rigid3(i, X3)[:, :2]
    def to_lib(self, i, Y3): return (np.asarray(Y3, float) - self.t[i]) @ self.R[i] / self.s[i]


def face(gap, c=(160.0, 150.0), hw=34.0):
    """478 landmarks (z = 0) with the lips around c, the eyes (IOD 120), nose base, chin and cheeks placed."""
    P = np.zeros((478, 3))
    rng = np.random.default_rng(5)
    P[:, :2] = c + rng.uniform(-60, 60, (478, 2))
    P[:, 1] -= 40
    P[33, :2], P[263, :2], P[168, :2], P[2, :2], P[152, :2] = (c[0] - 60, c[1] - 80), (c[0] + 60, c[1] - 80), (c[0], c[1] - 80), (c[0], c[1] - 40), (c[0], c[1] + 85)
    a = np.linspace(np.pi, -np.pi, len(ll.LIP_OUT), endpoint=False)
    P[ll.LIP_OUT, :2] = np.stack([c[0] + hw * np.cos(a), c[1] + (gap / 2 + 9) * np.sin(a)], 1)
    P[ll.LIP_IN, :2] = np.stack([c[0] + 0.8 * hw * np.cos(a), c[1] + max(0.6, gap / 2) * np.sin(a)], 1)
    P[13, :2], P[14, :2] = (c[0], c[1] - gap / 2), (c[0], c[1] + gap / 2)
    P[0, :2], P[17, :2] = (c[0], c[1] - gap / 2 - 9), (c[0], c[1] + gap / 2 + 9)
    P[61, :2], P[291, :2] = (c[0] - hw, c[1]), (c[0] + hw, c[1])
    for k, idx in enumerate(fg.gc.CHEEK):
        P[idx, :2] = (c[0] + (-1) ** k * (70 + 4 * (k // 2)), c[1] - 20 + 6 * (k // 2))
    return P


def draw(P, tex_dx=0, H=300, W=320):
    """skin (a little grain), textured red lips (vertical stripes, shifted by tex_dx px), a dark opening."""
    rng = np.random.default_rng(2)
    img = np.zeros((H, W, 3), np.float32)
    img[:] = SKIN
    img += rng.normal(0, 3, (H, W, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    lips = np.zeros((H, W), np.uint8)
    cv2.fillPoly(lips, [np.round(P[ll.LIP_OUT, :2]).astype(np.int32)], 1)
    xx = np.arange(W)[None, :] + tex_dx
    stripe = (18 * np.sin(xx / 2.3)).astype(np.float32)
    col = np.clip(np.array(LIP, np.float32)[None, None, :] + stripe[..., None], 0, 255).astype(np.uint8)
    img[lips > 0] = np.broadcast_to(col, (H, W, 3))[lips > 0]
    inner = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inner, [np.round(P[ll.LIP_IN, :2]).astype(np.int32)], 1)
    img[inner > 0] = CAV
    return img


def sequence(n=40, drift=0.0):
    """n frames, the mouth opening and closing smoothly; drift: the head moves down drift px / frame over the first 10."""
    gaps = 9 + 5 * np.sin(np.arange(n) / 4.0)
    S = np.array([face(g, c=(160.0, 150.0 + drift * min(t, 10))) for t, g in enumerate(gaps)])
    frames = [draw(P) for P in S]
    return frames, S, gaps


def checker(frames, S, outs=None, lms=None, plan_over=None, S3raw=None, S_pipe=None):
    """S: the true landmarks (the re-detection of the take and of the outputs, unless lms); S3raw / S_pipe: the
    pipeline's input track (raw / smoothed), default the truth."""
    n = len(frames)
    pl = Pose(n)
    gaps = np.linalg.norm(S[:, 13, :2] - S[:, 14, :2], axis=1)
    width = np.linalg.norm(S[:, 61, :2] - S[:, 291, :2], axis=1)
    plan = dict(P_out=[P[:, :2].copy() for P in S], fold=[1.0] * n, Fe=np.stack([gaps, width], 1), F=np.stack([gaps, width], 1),
                jaw=[0.0] * n, pin=[np.zeros(2)] * n)
    plan.update(plan_over or {})
    rng = dict(g_lo=float(gaps.min()), g_hi=float(gaps.max()), w_lo=float(width.min()) - 5, w_hi=float(width.max()) + 5)
    ck = fg.Checker(frames, S if S3raw is None else S3raw, plan, rng, S if S_pipe is None else S_pipe, pl, None, 3.0, 4, lmk=None, take_lm=S)
    ck.observe_take()
    ck.observe_all(outs if outs is not None else frames, None, list(lms if lms is not None else S))
    M = ck.metrics()
    return ck, M, ck.verdict(M)


def test_smooth_sequence_passes():
    frames, S, _ = sequence()
    ck, M, V = checker(frames, S)
    bad = [f for f in V['frames'] if f['verdict'] == 'FAIL']
    assert not bad, bad[:3]
    assert not V['windows']


def test_one_frame_jump_is_flagged_as_a_pop():
    frames, S, _ = sequence()
    outs = [f.copy() for f in frames]
    lms = [P.copy() for P in S]
    t = 20
    P = S[t].copy()
    P[:, 1] += 9
    P[:, 0] += 5           # the mouth jumps 10 px on one frame
    lms[t] = P
    outs[t] = draw(P)
    ck, M, V = checker(frames, S, outs, lms)
    f = V['frames'][t]
    assert f['verdict'] == 'FAIL', f
    assert {'pop', 'geom'} & set(f['attr']), f
    assert V['frames'][t - 3]['verdict'] == 'PASS' and V['frames'][t + 3]['verdict'] == 'PASS'


def test_fold_is_a_hard_fail():
    frames, S, _ = sequence()
    folds = [1.0] * len(frames)
    folds[12] = -0.3
    ck, M, V = checker(frames, S, plan_over=dict(fold=folds))
    assert V['frames'][12]['verdict'] == 'FAIL' and 'fold' in V['frames'][12]['attr']
    assert V['frames'][11]['verdict'] != 'FAIL'


def test_ghost_lip_is_flagged():
    frames, S, _ = sequence()
    outs = [f.copy() for f in frames]
    t = 26
    P = S[t]
    # a second lip band below the lower lip (the old lip surviving under the new one), only in the output
    y = int(P[17, 1] + 18)
    cv2.ellipse(outs[t], (int(P[14, 0]), y), (30, 5), 0, 0, 360, LIP, -1)
    ck, M, V = checker(frames, S, outs)
    f = V['frames'][t]
    assert f['verdict'] == 'FAIL' and 'ghost' in f['attr'], f


def test_lip_colour_flicker_is_flagged():
    frames, S, _ = sequence()
    outs = [f.copy() for f in frames]
    t = 15
    lips = np.zeros(frames[t].shape[:2], np.uint8)
    cv2.fillPoly(lips, [np.round(S[t][ll.LIP_OUT, :2]).astype(np.int32)], 1)
    inner = np.zeros_like(lips)
    cv2.fillPoly(inner, [np.round(S[t][ll.LIP_IN, :2]).astype(np.int32)], 1)
    m = (lips > 0) & (inner == 0)
    o = outs[t].astype(np.int16)
    o[m] = o[m] + np.array([40, 45, 10])
    outs[t] = np.clip(o, 0, 255).astype(np.uint8)
    ck, M, V = checker(frames, S, outs)
    f = V['frames'][t]
    assert f['verdict'] in ('SOFT', 'FAIL') and 'colour' in f['attr'], (f, M['colour'][t - 1:t + 2])


def test_held_input_track_is_flagged():
    frames, S, _ = sequence(drift=2.5)
    raw = S.copy()
    raw[:5] = S[5]   # the input track held (no detection) for frames 0-4
    from lsdub_engine.core import mouth_paint as mp_
    ck, M, V = checker(frames, S, S3raw=raw, S_pipe=mp_.smooth_track(raw, 'k5'))
    for t in range(0, 3):                                           # (held error 0.11 .. 0.07 IOD; frame 3: 0.05 IOD)
        f = V['frames'][t]
        assert f['verdict'] == 'FAIL' and 'lm_jump' in f['attr'], (t, f)
    assert V['frames'][12]['verdict'] == 'PASS'


def test_sloshing_lip_texture_is_a_slosh_window():
    frames, S, _ = sequence()
    outs = [f.copy() for f in frames]
    for t in range(18, 25):                                         # the lip texture slides back and forth, the outline holds
        outs[t] = draw(S[t], tex_dx=4 if t % 2 else -4)
    ck, M, V = checker(frames, S, outs)
    wins = V['windows']
    assert wins and any(w['frames'][0] <= 21 <= w['frames'][1] for w in wins), (wins, np.round(M['slosh_win'][14:30], 3))
    assert any('slosh' in V['frames'][t]['attr'] for t in range(18, 25))
    assert not any('slosh' in V['frames'][t]['attr'] for t in range(0, 8))


def test_verdict_strip_marks_each_frame():
    V = [dict(verdict=v, attr=['slosh'] if k == 3 else []) for k, v in enumerate(['PASS', 'SOFT', 'FAIL', 'PASS'])]
    img = fg.strip_image(V, 400, 12)
    assert img.shape == (12, 400, 3)
    assert tuple(img[2, 50]) == (60, 170, 60) and tuple(img[2, 150]) == (0, 175, 255) and tuple(img[2, 250]) == (40, 40, 230)
    assert tuple(img[11, 350]) == (200, 60, 160)


# ------------------------------------------------------------------------- v11: implausible re-detections --
def _false_fit(P, up=-0.8, scale=0.33, iod=120.0):
    """a false face fit (MediaPipe IMAGE mode on mv30 S3 s1234 f50): the whole mesh `up` IOD higher at `scale` of the size."""
    Q = np.array(P, float)
    c = Q[:, :2].mean(0)
    Q[:, :2] = c + (Q[:, :2] - c) * scale + [0, up * iod]
    return Q


def test_redet_check_rejects_a_false_face_fit_and_keeps_honest_offsets():
    frames, S, _ = sequence(n=20)
    R = S.copy()
    R[5, :, 1] += 10.0                          # an honest offset (0.08 IOD; the held starts reach 0.1-0.2): kept
    R[12] = _false_fit(S[12])                                 # a false fit: rejected (skull 0.8 IOD off, a third of the size)
    keep, rej = fg.redet_check(R, S)
    assert keep[5] and not keep[12] and keep.sum() == 19 and [x['i'] for x in rej] == [12] and rej[0]['skull'] > 0.4
    R2 = S.copy()
    R2[8, :, 1] += 42.0                        # a one-frame flip 0.35 IOD away (inside the skull bound): the neighbour test
    keep2, rej2 = fg.redet_check(R2, S)
    assert not keep2[8] and keep2[7] and keep2[9] and rej2[0].get('jump', 0) > fg.REDET['jump']
    # a held input track (the head moves 2.5 px / frame while the track repeats its first detection): kept
    T = S.copy()
    T[:6] = S[6]
    S3 = S.copy()
    for t in range(10):
        S3[t, :, 1] += 2.5 * t
        T[t] = S3[min(max(t, 6), 6)]
    keep3, _ = fg.redet_check(S3, T)
    assert keep3.all()


def test_landmark_cache_and_observe_do_not_use_a_false_fit(tmp_path):
    frames, S, _ = sequence(n=16)
    bad = _false_fit(S[9])
    class Det:
        calls = 0
        def __call__(self, img, box=None):
            Det.calls += 1
            k = next(j for j, f in enumerate(frames) if np.array_equal(f, img))
            return bad if k == 9 else S[k]
    p = tmp_path / 'lm.npz'
    R = fg.landmark_cache(frames, str(p), Det(), S)
    assert np.isnan(R[9]).all() and np.isfinite(R[8]).all() and np.allclose(R[3], S[3])
    assert np.allclose(np.load(p)['S3'][9], bad)               # the cache keeps the raw detection
    R2 = fg.landmark_cache(frames, str(p), Det(), S)
    assert np.isnan(R2[9]).all()     # re-filtered on load
    # the checker's own fresh detection of a frame: a false fit is no detection (det False, the plan's outline is used)
    n = len(frames)
    gaps = np.linalg.norm(S[:, 13, :2] - S[:, 14, :2], axis=1)
    width = np.linalg.norm(S[:, 61, :2] - S[:, 291, :2], axis=1)
    plan = dict(P_out=[P[:, :2].copy() for P in S], fold=[1.0] * n, Fe=np.stack([gaps, width], 1), F=np.stack([gaps, width], 1), jaw=[0.0] * n, pin=[np.zeros(2)] * n)
    rng = dict(g_lo=float(gaps.min()), g_hi=float(gaps.max()), w_lo=float(width.min()) - 5, w_hi=float(width.max()) + 5)
    ck = fg.Checker(frames, S, plan, rng, S, Pose(n), None, 3.0, 4, lmk=Det(), take_lm=R)
    c0 = Det.calls
    ck.observe_take()
    assert Det.calls == c0                         # cached (incl. the rejected one): not measured again
    assert ck.obs_k[9]['det'] is False and ck.obs_k[8]['det'] is True
    o = ck.observe(9, frames[9])
    assert o['det'] is False and o.get('redet_rejected')
    o8 = ck.observe(8, frames[8])
    assert o8['det'] is True and not o8.get('redet_rejected')


def test_leak_scene_option_ignores_dark_scene_but_not_old_mouth():
    # a dark patch (hair at a turned head's far corner) beside the mouth in the take AND the output: with --leak-scene it
    # is scene, not leak; a dark blob only in the output is still a leak
    frames, S, _ = sequence(n=12)
    outs = [f.copy() for f in frames]
    for f in frames + outs: cv2.rectangle(f, (106, 140), (118, 168), (20, 15, 25), -1)        # left of the mouth (lips: 126..194)
    cv2.circle(outs[6], (205, 160), 4, (20, 15, 40), -1)                                        # old-mouth residue, output only
    n = len(frames)
    gaps = np.linalg.norm(S[:, 13, :2] - S[:, 14, :2], axis=1)
    width = np.linalg.norm(S[:, 61, :2] - S[:, 291, :2], axis=1)
    plan = dict(P_out=[P[:, :2].copy() for P in S], fold=[1.0] * n, Fe=np.stack([gaps, width], 1), F=np.stack([gaps, width], 1), jaw=[0.0] * n, pin=[np.zeros(2)] * n)
    rng = dict(g_lo=float(gaps.min()), g_hi=float(gaps.max()), w_lo=float(width.min()) - 5, w_hi=float(width.max()) + 5)
    def leak(scene):
        ck = fg.Checker(frames, S, plan, rng, S, Pose(n), None, 3.0, 4, lmk=None, take_lm=S)
        ck.leak_scene = scene
        ck.observe_take()
        ck.observe_all(outs, None, list(S))
        return np.array([o['leak'] for o in ck.obs])
    a, b = leak(False), leak(True)
    assert a[3] > 0 and b[3] == 0                         # the dark patch: a leak for v9's ruler, scene with the option
    assert b[6] > 0                                       # the output-only residue is still a leak
