"""v13 tooth-scale ruler (audits/tooth_scale.py): the visible tooth span, the both-spans filter and ratio, the upper teeth
count, the plate's interdental lines and central widths, the acceptance lines, the CLI. Synthetic, CPU only."""
import json, os, sys
import numpy as np, cv2
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from lsdub_engine import cli
from lsdub_engine.audits import tooth_scale as ts
from test_teeth_layer import _P as _P0

SKIN, LIP, CAV = (100, 140, 215), (45, 40, 185), (20, 15, 40)        # BGR (skin: Lab a ~152, as the takes' skin)
TOOTH, GAP = (235, 240, 245), (135, 135, 135)                         # a grey interdental line: darker than the keys, tooth-coloured
C = (200.0, 170.0)
OCD = 200.0
IPD = 0.69 * OCD


def _face(c=C, hw=70, gap=44.0):
    """478 landmarks: test_teeth_layer's lips around c, eye corners OCD apart, iris centres IPD apart."""
    P = _P0(c=c, hw=hw, gap=gap)
    P[33, :2], P[263, :2] = (c[0] - OCD / 2, c[1] - 110), (c[0] + OCD / 2, c[1] - 110)
    P[468, :2], P[473, :2] = (c[0] - IPD / 2, c[1] - 110), (c[0] + IPD / 2, c[1] - 110)
    P[168, :2], P[152, :2] = (c[0], c[1] - 110), (c[0], c[1] + 90)
    return P


def _frame(P, n_up=6, w=14, g=3, n_lo=0, w_lo=11, tooth_h=16, H=340, W=400):
    """skin, red lips, a dark cavity in the inner lip; n_up upper teeth (w px wide, g px grey gaps) hanging from the upper
    inner lip, centred; optionally n_lo lower teeth standing on the lower inner lip (the arches apart)."""
    from lsdub_engine.core import lip_layers as ll
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = SKIN
    cv2.fillPoly(img, [np.round(P[ll.LIP_OUT, :2]).astype(np.int32)], LIP)
    inner = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inner, [np.round(P[ll.LIP_IN, :2]).astype(np.int32)], 1)
    img[inner > 0] = CAV
    cx = int(round(P[13, 0]))
    for n, wi, y0, y1 in ((n_up, w, int(P[13, 1]) - 4, int(P[13, 1]) + tooth_h), (n_lo, w_lo, int(P[14, 1]) - tooth_h, int(P[14, 1]) + 4)):
        if not n: continue
        x = cx - (n * wi + (n - 1) * g) // 2
        for k in range(n):
            m = np.zeros((H, W), bool)
            m[y0:y1, x:x + wi] = True
            img[m & (inner > 0)] = TOOTH
            if k < n - 1:
                m = np.zeros((H, W), bool)
                m[y0:y1, x + wi:x + wi + g] = True
                img[m & (inner > 0)] = GAP
            x += wi + g
    return img


def test_units_and_the_iris_guard():
    P = np.array([_face()] * 4)
    ipd, ocd = ts.units(P)
    assert np.allclose(ipd, IPD) and np.allclose(ocd, OCD)
    P[1, 473, :2] = P[1, 468, :2] + 0.5                                  # collapsed irises (no face found: the track's fill-in)
    ipd, ocd = ts.units(P)
    g, fixed = ts.guard_ipd(ipd, ocd)
    assert fixed == [1] and np.allclose(g, IPD)
    assert ts.guard_ipd(ipd, ocd, 0)[1] == []                            # off


def test_span_of_a_synthetic_tooth_row():
    P = _face(hw=95, gap=60.0)
    img = _frame(P, n_up=6, w=14, g=3)
    sk = ts.take_skin_L([img], P[None])
    assert abs(sk - cv2.cvtColor(np.uint8([[SKIN]]), cv2.COLOR_BGR2LAB)[0, 0, 0]) < 2
    x0, y0, x1, y1 = ts.mouth_box(P[:, :2], IPD, img.shape)
    assert x0 == int(np.floor(P[61, 0] - 0.1 * IPD)) and y1 == int(np.ceil(P[17, 1] + 0.03 * IPD))
    cx = int(round(P[13, 0]))
    left = cx - (6 * 14 + 5 * 3) // 2
    cols = np.concatenate([np.arange(left + k * 17, left + k * 17 + 14) for k in range(6)]) - x0     # the teeth's columns
    for key in ('lab', 'tl'):
        s = ts.spans([img, _frame(P, n_up=0)], P[None].repeat(2, 0), key, sk)
        assert abs(s[0] - (np.percentile(cols, 98) - np.percentile(cols, 2))) < 1e-6 and s[1] == 0.0
    # a render whose teeth are 1.2x wider: the span grows by ~1.2 (in px, IPD or OCD alike)
    r = ts.spans([_frame(P, n_up=6, w=17, g=3)], P[None], 'tl', sk)[0] / ts.spans([img], P[None], 'tl', sk)[0]
    assert 1.15 < r < 1.22


def test_both_spans_filter_and_ratio():
    ipd = np.full(6, 100.0)
    st = np.array([50.0, 10.0, 60.0, 0.0, 40.0, 55.0])
    sr = np.array([60.0, 70.0, 66.0, 30.0, 11.0, 66.0])
    s = ts.summarize(st, sr, ipd, ipd / 0.69)
    ok = np.array([True, False, True, False, False, True])                # both > 0.12 IPD
    assert (s['ok'] == ok).all() and s['ipd']['n'] == 3 and s['ocd']['n'] == 3
    assert s['ipd']['ratio_med'] == pytest.approx(np.median([1.2, 1.1, 1.2])) and s['ocd']['ratio_med'] == s['ipd']['ratio_med']
    assert s['ipd']['take_p90'] == pytest.approx(np.percentile([0.5, 0.6, 0.55], 90))
    assert s['ipd']['render_p50'] == pytest.approx(0.66) and s['ocd']['render_p50'] == pytest.approx(0.66 * 0.69)
    assert s['ipd']['all']['take_p90'] == pytest.approx(np.percentile(st / 100, 90))
    z = ts.summarize(np.zeros(3), np.ones(3) * 50, np.full(3, 100.0))
    assert z['ipd']['n'] == 0 and z['ipd']['ratio_med'] is None and 'ocd' not in z


@pytest.mark.parametrize('n_up,w,n_lo', [(4, 15, 0), (6, 15, 0), (8, 13, 0), (6, 15, 9)])
def test_upper_teeth_counted_on_a_synthetic_row(n_up, w, n_lo):
    P = _face(hw=110, gap=60.0)
    img = _frame(P, n_up=n_up, w=w, g=3, n_lo=n_lo)
    sk = ts.take_skin_L([img], P[None])
    for key in ('tl', 'lab'):
        r = ts.count_teeth(img, P[:, :2], IPD, OCD, key, sk)
        assert r['n'] == n_up, (key, r['n'], r['minima_x'])
        cx = int(round(P[13, 0]))
        left = cx - (n_up * w + (n_up - 1) * 3) // 2
        gaps = left + w + (w + 3) * np.arange(n_up - 1) + 1.5                        # the interdental lines' centres
        assert np.abs(np.sort(r['minima_x']) - gaps).max() <= 2.0
        assert abs(r['row'][1] - r['row'][0] - (n_up * w + (n_up - 1) * 3)) <= 2        # the row: the teeth, not the lower arch
    still = ts.teeth_still(img, r, OCD, 'synthetic')
    assert still.ndim == 3 and still.shape[1] > 2 * (r['box'][2] - r['box'][0])
    assert ts.count_teeth(_frame(P, n_up=0), P[:, :2], IPD, OCD, 'tl', sk)['n'] == 0


def _plate(widths, upper=True, iod=100.0, texel=2.0, half=0.4, line=0.55):
    """an arch plate (BGRA, platediag geometry): teeth of the given widths (IOD) outward from a midline line at a = 0 on each
    side, a dark interproximal line (2 texels, x line) at every boundary, opaque on the teeth's side of the incisal edge."""
    Tw = int(round(2 * half * iod * texel))
    h0 = -0.3 * iod if upper else -0.05 * iod
    Th = int(round(0.35 * iod * texel))
    T = np.zeros((Th, Tw, 4), np.float32)
    T[..., :3] = TOOTH
    h = (np.arange(Th) + 0.5) / texel + h0
    T[..., 3] = ((h < 0) if upper else (h > 0))[:, None].astype(np.float32)
    a = ((np.arange(Tw) + 0.5) / texel - Tw / (2 * texel)) / iod
    b = np.concatenate([[0.0], np.cumsum(widths)])
    bounds = np.concatenate([-b[:0:-1], b])
    for x in bounds:
        if abs(x) >= half: continue
        T[:, np.argsort(np.abs(a - x))[:2], :3] *= line
    return T, dict(iod=iod, texel=texel, h0=h0)


def test_plate_teeth_central_widths_upper_and_lower(tmp_path):
    up, g = _plate([0.10, 0.08, 0.09, 0.07], upper=True)
    r = ts.plate_teeth(up, upper=True, unit=0.69, **g)
    assert r['midline'] is not None and abs(r['midline']) <= 0.01
    assert r['central'][0] == pytest.approx(0.10, abs=0.011) and r['central'][1] == pytest.approx(0.10, abs=0.011)
    assert r['central_mean_ipd'] == pytest.approx(r['central_mean'] / 0.69, abs=1e-3)
    assert r['spacing_med'] == pytest.approx(np.median([0.07, 0.09, 0.08, 0.10, 0.10, 0.08, 0.09, 0.07]), abs=0.012)
    lo, gl = _plate([0.08, 0.075, 0.085, 0.09], upper=False)
    rl = ts.plate_teeth(lo, upper=False, **gl)
    assert abs(rl['midline']) <= 0.01 and rl['central_mean'] == pytest.approx(0.08, abs=0.011) and 'central_mean_ipd' not in rl
    assert ts.plate_teeth(up, upper=False, **g)['n_lines'] == 0              # the band below the upper plate's edge is empty
    # both npz formats load to the same lines
    np.savez(tmp_path / 'diag.npz', up=up, lo=lo, up_h0=g['h0'], lo_h0=gl['h0'], texel=2.0, iod=100.0)
    Tw = up.shape[1]
    aa = ((np.arange(Tw) + 0.5) / 2.0 - Tw / 4.0) / 100.0
    np.savez(tmp_path / 'out.npz', up_tex=up, up_a=aa, up_h=((np.arange(up.shape[0]) + 0.5) / 2.0 + g['h0']) / 100.0,
             lo_tex=lo, lo_a=((np.arange(lo.shape[1]) + 0.5) / 2.0 - lo.shape[1] / 4.0) / 100.0,
             lo_h=((np.arange(lo.shape[0]) + 0.5) / 2.0 + gl['h0']) / 100.0, iod=100.0)
    A, B = ts.load_tex(tmp_path / 'diag.npz'), ts.load_tex(tmp_path / 'out.npz')
    for k, upper in (('up', True), ('lo', False)):
        ra, rb = ts.plate_teeth(upper=upper, **A[k]), ts.plate_teeth(upper=upper, **B[k])
        assert np.allclose(ra['lines'], rb['lines'], atol=1e-3) and ra['central'] == rb['central']


def test_acceptance_lines():
    S = dict(keys={'tl': {'ipd': dict(n=10, take_p90=0.60, render_p90=0.62, ratio_med=1.02)},
                   'lab': {'ipd': dict(n=10, take_p90=0.50, render_p90=0.60, ratio_med=1.20)}},
             plate={'up': dict(central_mean_ipd=0.137, central_mean=0.095, central=[0.094, 0.096])}, teeth={'render_tl': dict(n=9, frame=4)})
    a = {x['name']: x['status'] for x in ts.acceptance(S)}
    assert a == {'span_tl': 'PASS', 'span_lab': 'FAIL', 'central': 'PASS', 'teeth': 'FAIL'}
    S['keys']['tl']['ipd'].update(ratio_med=1.06)
    S['plate'] = None
    S['keys']['lab']['ipd'] = dict(n=0)
    a = {x['name']: x['status'] for x in ts.acceptance(S)}
    assert a == {'span_tl': 'FAIL', 'span_lab': 'n/a', 'central': 'n/a', 'teeth': 'FAIL'}


def test_cli_end_to_end_and_help(tmp_path, capsys):
    assert cli.CMDS['tooth-scale'] == 'audits.tooth_scale'
    with pytest.raises(SystemExit) as e: ts.main(['--help'])
    assert e.value.code == 0 and '--track' in capsys.readouterr().out
    with pytest.raises(SystemExit): cli.main(['tooth-scale', '--help'])
    assert '--tex' in capsys.readouterr().out
    P = _face(hw=110, gap=60.0)
    take, rend = str(tmp_path / 'take.avi'), str(tmp_path / 'render.avi')
    for path, w in ((take, 15), (rend, 18)):                               # the render's teeth 1.2x wider
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (400, 340))
        for _ in range(3): vw.write(_frame(P, n_up=6, w=w, g=3))
        vw.release()
    np.savez(tmp_path / 'trk.npz', S3=np.array([P] * 3))
    up, g = _plate([0.10, 0.08, 0.09, 0.07])
    lo, gl = _plate([0.08, 0.075, 0.085, 0.09], upper=False)
    np.savez(tmp_path / 'tex.npz', up=up, lo=lo, up_h0=g['h0'], lo_h0=gl['h0'], texel=2.0, iod=100.0)
    js, md = str(tmp_path / 'o.json'), str(tmp_path / 'o.md')
    assert ts.main([take, rend, '--track', str(tmp_path / 'trk.npz'), '--tex', str(tmp_path / 'tex.npz'), '--json', js, '--md', md,
                    '--label', 'syn', '--stills', str(tmp_path / 'st')]) == 0
    out = capsys.readouterr().out
    S = json.load(open(js))
    assert out.startswith('syn: tl span') and S['frames'] == 3 and S['ipd_ocd'] == pytest.approx(0.69, abs=1e-3)
    for key in ('tl', 'lab'): assert 1.12 < S['keys'][key]['ipd']['ratio_med'] < 1.3 and S['keys'][key]['ipd']['n'] == 3
    assert S['teeth']['render_tl']['n'] == 6 and S['teeth']['take_tl']['n'] == 6
    assert S['plate']['up']['central_mean_ipd'] == pytest.approx(0.10 / 0.69, abs=0.02)
    st = {a['name']: a['status'] for a in S['accept']}
    assert st['span_tl'] == 'FAIL' and st['teeth'] == 'PASS' and 'acceptance' in open(md).read()
    assert len(os.listdir(tmp_path / 'st')) == 3
