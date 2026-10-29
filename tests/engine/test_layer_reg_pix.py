"""v12 pixel registration ruler (lsdub/audits/layer_reg.py --pix): template-tracked upper / lower incisal patches and the
nose-base reference in the skull-stabilised crop, velocities in IOD / frame; jitter, the windowed wander (W = 4 / 8),
the lower arch split along / across the hinge arc. Synthetic, CPU only."""
import json
import numpy as np, cv2

from lsdub_engine.audits import layer_reg as lr
from lsdub_engine.core import lip_layers as ll

W_, H_ = 320, 260
IOD = 100.0                       # frame px (canonical == frame px: A = identity), so 0.01 IOD = 1 px
REF = (160.0, 130.0)
SKIN = np.array([120, 155, 205], np.float32)          # BGR
LIP = np.array([45, 40, 185], np.float32)
TOOTH = np.array([235, 240, 245], np.float32)
CAV = np.array([20, 15, 40], np.float32)
EYE = np.eye(2, 3)


def _tex(seed, sigma=1.5):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((H_, W_)).astype(np.float32), (0, 0), sigma)
    return (t - t.min()) / max(1e-6, t.max() - t.min())


def _loop(c, hw, hh, k=20):
    """k points in LIP_IN / LIP_OUT order: left corner, along the lower lip, right corner, back along the upper lip."""
    a = np.linspace(np.pi, -np.pi, k, endpoint=False)
    return np.stack([c[0] + hw * np.cos(a), c[1] + hh * np.sin(a)], 1)


def _lm(hh_in=15.0):
    """478 x 3 plausible landmarks (frame px): lips around REF (inner loop 40 x hh_in px half-axes), eyes 100 px apart."""
    rng = np.random.default_rng(7)
    P = np.zeros((478, 3))
    P[:, 0] = rng.uniform(110, 210, 478)
    P[:, 1] = rng.uniform(30, 200, 478)
    P[:, 2] = rng.uniform(-20, 20, 478)
    P[ll.LIP_OUT, :2] = _loop(REF, 50, hh_in + 12)
    P[ll.LIP_IN, :2] = _loop(REF, 40, hh_in)
    P[ll.LIP_IN + ll.LIP_OUT, 2] = -25.0
    P[13, :2], P[14, :2] = (REF[0], REF[1] - hh_in), (REF[0], REF[1] + hh_in)
    P[33], P[263], P[168], P[152] = (110, 50, 0), (210, 50, 0), (160, 50, -10), (160, 205, -5)
    return P


def _band(x0, x1, y0, y1):
    a = np.zeros((H_, W_), np.float32)
    a[y0:y1, x0:x1] = 1.0
    return cv2.GaussianBlur(a, (0, 0), 1.0)


def _shift(img, dx, dy):
    """an exact sub-pixel translation by (dx, dy) px (Fourier phase ramp): the content moves, its shape does not change
    (a bilinear warp would also blur it by a phase-dependent kernel, i.e. change the content it is matched against)."""
    if dx == 0 and dy == 0: return img
    ky, kx = np.fft.fftfreq(img.shape[0])[:, None], np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (kx * dx + ky * dy)))).astype(np.float32)


def _frame(P, up=None, lo=None):
    """skin (static texture), red lips, dark cavity; up / lo = (texture, alpha, dx, dy): a textured tooth band moved by
    (dx, dy) frame px (exact sub-pixel shift), seen only inside the inner lip."""
    img = SKIN[None, None] * (0.85 + 0.3 * _tex(1, 2.0))[..., None]
    out = np.zeros((H_, W_), np.uint8)
    cv2.fillPoly(out, [np.round(P[ll.LIP_OUT, :2]).astype(np.int32)], 1)
    img[out > 0] = LIP
    inn = np.zeros((H_, W_), np.uint8)
    cv2.fillPoly(inn, [np.round(P[ll.LIP_IN, :2]).astype(np.int32)], 1)
    mouth = np.broadcast_to(CAV, (H_, W_, 3)).copy()
    for layer in (up, lo):
        if layer is None: continue
        tex, al, dx, dy = layer
        t = _shift(tex, dx, dy)
        a = np.clip(_shift(al, dx, dy), 0, 1)[..., None]
        mouth = mouth * (1 - a) + (TOOTH[None, None] * (0.8 + 0.2 * t[..., None])) * a
    img[inn > 0] = mouth[inn > 0]
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


UP = (_tex(11), _band(125, 196, 100, 125))           # upper incisal edge 10 px (0.1 IOD) below the inner top (115)
LO = (_tex(12), _band(128, 193, 137, 165))           # lower incisal edge 8 px above the inner bottom (145)


def _skin_L():
    return float(cv2.cvtColor(SKIN.astype(np.uint8)[None, None], cv2.COLOR_BGR2LAB)[0, 0, 0])


def _sf(n, phi=0.25):
    """a synthetic skull frame: identity stabilisation; hinge axis u tilted by phi in the image plane, T - H = (0, 80, -60)
    -> the lower incisors' arc direction a = (-sin phi, cos phi), in-plane lever 0.6 IOD / rad."""
    Cm = _lm()
    u = np.array([np.cos(phi), np.sin(phi), 0.0])
    H = Cm[14] - np.array([0.0, 80.0, -60.0])
    return dict(A=np.tile(EYE, (n, 1, 1)), ref=np.array(REF), iod=IOD, H=H, u=u, Cm=Cm)


def _track(dys_up=None, lo_xy=None, n=None):
    n = n or len(dys_up if dys_up is not None else lo_xy)
    P = _lm()
    frames = [_frame(P, None if dys_up is None else (UP[0], UP[1], 0.0, dys_up[t]),
                     None if lo_xy is None else (LO[0], LO[1], lo_xy[t][0], lo_xy[t][1])) for t in range(n)]
    sf = _sf(n)
    lm = np.tile(P, (n, 1, 1))
    return frames, sf, lm


def test_patch_shift_recovers_a_known_subpixel_shift_with_its_sign():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random((120, 160)).astype(np.float32) * 255, (0, 0), 1.5)
    mask = np.zeros((120, 160), bool)
    mask[40:80, 50:110] = True
    for dx, dy in ((2.3, -1.4), (-0.6, 0.35), (0.0, 3.7)):
        moved = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]), (160, 120), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        d, q = lr.patch_shift(moved, base, mask, 6)                     # content moved by +(dx, dy) from t-1 (base) to t (moved)
        assert d is not None and q > 0.9 and np.allclose(d, [dx, dy], atol=0.15), (dx, dy, d)
    # beyond the search range: the argmax sits on the border -> rejected
    far = cv2.warpAffine(base, np.float32([[1, 0, 9.0], [0, 1, 0]]), (160, 120), borderMode=cv2.BORDER_REFLECT)
    assert lr.patch_shift(far, base, mask, 6)[0] is None
    # the exact NumPy fallback agrees with cv2's masked TM_CCOEFF_NORMED
    T = np.ascontiguousarray(base[40:80, 50:110])
    Tm = np.zeros(T.shape, np.uint8)
    cv2.ellipse(Tm, (30, 20), (25, 15), 0, 0, 360, 1, -1)
    S = np.ascontiguousarray(base[34:86, 44:116])
    assert np.abs(lr._ncc_np(S, T, Tm) - cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED, mask=Tm)).max() < 1e-3


def test_masks_find_the_upper_and_lower_incisal_patches():
    frames, sf, lm = _track(dys_up=[0.0], lo_xy=[(0.0, 0.0)])
    g, bgr = lr.stab_crop(frames[0], sf['A'][0], sf['ref'], IOD)
    assert g.shape == (190, 240) and bgr.shape == (190, 240, 3)
    Pc = lr.to_crop(lm[0], sf['A'][0], sf['ref'], IOD)
    assert np.allclose(Pc[13], [(0 + 0.6) * 200, (-0.15 + 0.5) * 200])   # lm 13 = ref - 0.15 IOD -> crop (120, 70)
    m = lr.arch_masks(bgr, Pc, _skin_L())
    ys_u, xs_u = np.nonzero(m['up'])
    ys_l, xs_l = np.nonzero(m['lo'])
    assert m['up'].sum() > 1000 and m['lo'].sum() > 800 and not (m['up'] & m['lo']).any()
    # upper: from the eroded opening's top to the incisal edge (crop y = (125 - 80) * 2 = 90) + 0.02 IOD (4 px)
    assert ys_u.max() >= 90 and ys_u.max() <= 96 and ys_l.min() >= 2 * (137 - 80) - 6 and ys_l.max() < 2 * (145 - 80)
    assert np.abs(xs_u - 120).max() <= 50 and np.abs(xs_l - 120).max() <= 50       # |x - cx| <= 0.25 IOD
    nm = lr.nose_mask(g.shape)
    ys, xs = np.nonzero(nm)
    assert (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1) == (16, 56, 76, 164)


def test_static_teeth_have_no_jitter_and_no_wander():
    n = 20
    frames, sf, lm = _track(dys_up=np.zeros(n))
    r = lr.pix_measure(frames, None, n, lm, _skin_L(), sf=sf)
    assert r['up']['n'] == n - 1 and r['up']['jit_p95'] < 0.002 and r['up']['w4_mean'] < 1e-3 and r['up']['w8_mean'] < 1e-3
    assert r['nose']['n'] == n - 1 and r['nose']['jit_p95'] < 0.002
    assert r['lo']['n'] == 0 and r['lo_hinge']['along_resid'] is None


def test_back_and_forth_teeth_jitter_and_wander():
    n = 20
    dy = 1.0 * (-1.0) ** np.arange(n)                           # +-0.01 IOD every frame
    frames, sf, lm = _track(dys_up=dy)
    r = lr.pix_measure(frames, None, n, lm, _skin_L(), sf=sf)
    v = np.array(r['per_frame']['up'], float)
    assert r['up']['n'] == n - 1 and abs(r['up']['jit_p95'] - 0.02) < 0.003 and abs(r['up']['jit_p50'] - 0.02) < 0.003
    assert np.all(np.sign(v[1:, 1]) == np.sign(np.diff(dy)))                # the sign: +1 px down from t-1 to t is +0.01 IOD
    assert r['up']['w4_mean'] > 0.01 and r['up']['w8_mean'] > 0.01
    assert r['nose']['jit_p95'] < 0.002                                      # the skin did not move


def test_steady_drift_is_jitter_but_not_wander():
    n = 24
    frames, sf, lm = _track(dys_up=0.3 * np.arange(n))              # 0.003 IOD / frame, one direction
    r = lr.pix_measure(frames, None, n, lm, _skin_L(), sf=sf)
    v = np.array(r['per_frame']['up'], float)
    assert r['up']['n'] == n - 1 and np.allclose(v[1:, 1], 0.003, atol=0.0006) and np.abs(v[1:, 0]).max() < 0.0006
    assert r['up']['jit_p50'] > 0.0025 and r['up']['w4_mean'] < 3e-4 and r['up']['w8_mean'] < 3e-4


def test_lower_arch_along_the_hinge_arc_vs_across():
    n = 24
    sf = _sf(n)
    a = np.array([-np.sin(0.25), np.cos(0.25)])
    e = np.array([a[1], -a[0]])
    th = 0.03 * np.sin(2 * np.pi * np.arange(n) / 10)                        # jaw angle, rad
    pos = (0.6 * th)[:, None] * a[None, :] * IOD                              # in-plane lever 0.6 IOD / rad, frame px
    frames, _, lm = _track(dys_up=np.zeros(n), lo_xy=pos)
    r = lr.pix_measure(frames, None, n, lm, _skin_L(), jaw=np.degrees(th), sf=sf)
    h = r['lo_hinge']
    assert np.allclose(h['a'], a, atol=1e-4) and abs(h['lever'] - 0.6) < 1e-9 and h['lever_3d'] > h['lever']
    true_along = np.diff(0.6 * th)
    along = np.array(r['per_frame']['along'][1:], float)
    assert r['lo']['n'] >= n - 2 and np.nanmax(np.abs(along - true_along)) < 0.0015          # along recovered
    assert h['across']['jit_p95'] < 0.0015 and h['along_resid']['jit_p95'] < 0.0015            # nothing across; jaw explains the rest
    # + a lateral wobble (+-0.005 IOD across the arc every frame): across-arc jitter, the along part untouched
    pos2 = pos + (0.5 * (-1.0) ** np.arange(n))[:, None] * e[None, :]
    frames2, _, _ = _track(dys_up=np.zeros(n), lo_xy=pos2)
    r2 = lr.pix_measure(frames2, None, n, lm, _skin_L(), jaw=np.degrees(th), sf=sf)
    h2 = r2['lo_hinge']
    assert h2['across']['jit_p95'] > 0.005 and h2['across']['w4_mean'] > 0.005 and h2['along_resid']['jit_p95'] < 0.002


def test_skipped_and_unlandmarked_frames_are_not_measured():
    n = 12
    frames, sf, lm = _track(dys_up=np.zeros(n))
    lm = lm.copy()
    lm[4] = np.nan
    skip = np.zeros(n, bool)
    skip[8] = True
    r = lr.pix_measure(frames, None, n, lm, _skin_L(), skip=skip, sf=sf)
    v = np.array([np.nan if x is None else x[0] for x in r['per_frame']['up']], float)
    assert np.isnan(v[[0, 4, 5, 8, 9]]).all() and np.isfinite(np.delete(v, [0, 4, 5, 8, 9])).all() and r['skipped'] == 1
    r2 = lr.pix_measure(frames, None, n, lm, _skin_L(), sf=dict(sf, S=np.tile(_lm(), (n, 1, 1))), take_fallback=True)
    assert r2['take_fallback'] == 1 and r2['up']['n'] == n - 1                # the take may fall back to its own track


def test_pix_stats_windows_match_a_direct_computation():
    rng = np.random.default_rng(3)
    n = 30
    v = rng.normal(0, 0.01, (n, 2))
    v[0] = np.nan
    v[[6, 7, 8, 20]] = np.nan
    v[11:16] = np.nan
    s = lr.pix_stats(v, (4, 8))
    ok = np.isfinite(v).all(1)
    j = np.linalg.norm(v[ok], axis=1)
    assert s['n'] == ok.sum() and np.isclose(s['jit_p50'], np.percentile(j, 50)) and np.isclose(s['jit_p95'], np.percentile(j, 95))
    for W in (4, 8):
        E = []
        for c in range(n):
            a0, a1 = max(1, c - W + 1), min(n - 1, c + W)
            idx = [t for t in range(a0, a1 + 1) if ok[t]]
            if a1 - a0 + 1 < max(3, W) or len(idx) < 3: continue
            seg = v[idx]
            E.append(sum(np.abs(seg[:, k]).sum() - abs(seg[:, k].sum()) for k in range(2)) / len(idx))
        assert s[f'w{W}_n'] == len(E) and np.isclose(s[f'w{W}_mean'], np.mean(E)) and np.isclose(s[f'w{W}_p95'], np.percentile(E, 95))
    # one direction only: no wander whatever the speed; 1-D series work too
    assert lr.pix_stats(np.linspace(0.001, 0.02, n), (4,))['w4_mean'] == 0.0


def test_cli_pix_without_render_and_with_a_render_log(tmp_path):
    n = 12
    P = _lm()
    dy = 1.0 * (-1.0) ** np.arange(n)
    frames = [_frame(P, (UP[0], UP[1], 0.0, dy[t])) for t in range(n)]
    vid = str(tmp_path / 'v.avi')
    wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*'MJPG'), 25, (W_, H_))
    for f in frames: wr.write(f)
    wr.release()
    S3 = np.tile(P, (n, 1, 1))
    np.savez(tmp_path / 'trk.npz', S3=S3)
    np.savez(tmp_path / 'lm.npz', S3=S3)
    out = tmp_path / 'r.json'
    rc = lr.main(['--pix', '--video', vid, '--take-trk', str(tmp_path / 'trk.npz'), '--frames', str(n), '--lm', str(tmp_path / 'lm.npz'),
                  '--json', str(out), '--label', 'synthetic'])
    assert rc == 0
    r = json.load(open(out))
    for k in ('n', 'ppi', 'iod', 'up', 'lo', 'nose', 'lo_hinge', 'per_frame'): assert k in r
    assert r['n'] == n and abs(r['iod'] - IOD) < 1e-6 and r['is_take'] and r['up']['n'] >= n - 2 and r['lo_hinge']['along_resid'] is None
    rd = tmp_path / 'render'
    rd.mkdir()
    json.dump(dict(frames=[dict(i=i, reg=dict(jaw=0.5 * i), vocal_s=(0.0 if i == 5 else None)) for i in range(n)]), open(rd / 'warp.json', 'w'))
    rc = lr.main(['--pix', '--video', vid, '--take-trk', str(tmp_path / 'trk.npz'), '--frames', str(n), '--lm', str(tmp_path / 'lm.npz'),
                  '--take', vid, '--render', str(rd), '--json', str(out)])
    assert rc == 0
    r = json.load(open(out))
    assert r['skipped'] == 1 and r['lo_hinge']['along_resid'] is not None and r['per_frame']['up'][5] == [None, None] and r['per_frame']['up'][6] == [None, None]
