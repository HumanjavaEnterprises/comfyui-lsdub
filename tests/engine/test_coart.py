# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""v10c: the COARTICULATION planner (lsdub/core/coart.py, --warp-target coart) on synthetic phone sequences, CPU only:
the dominance blend is a partition of unity; closures reach closure; a 40 ms vowel undershoots while a 400 ms one reaches
its target; the dynamics prior lands the plan's velocity p95 near the take's own; a skipped span is free (the take's own
shape); the jaw share 1 is v9's jaw reference; and --warp-target v9 (the default) is v9 bit for bit on a synthetic clip."""
import argparse, os, sys
import numpy as np

from lsdub_engine.core import coart, mouth_warp as mw

FPS = 25.0


def args(**kw):
    a = argparse.Namespace(warp_centre=0.8, warp_open_gain=1.0, warp_width_gain=1.0, warp_close_depth=0.0, coart_dom_width=60.0,
                           coart_undershoot=0.6, coart_prior=0.5, coart_anticipate=100.0, coart_jaw_share='auto')
    for k, v in kw.items(): setattr(a, k, v)
    return a


def align_of(phones, t=0.4):
    """[(ARPA, seconds), ...] -> an alignment dict (one word) starting at t; -> (align, end time)."""
    ph = []
    for p, d in phones:
        ph.append(dict(p=p, start=round(t, 4), end=round(t + d, 4)))
        t += d
    return dict(words=[dict(w='w', start=ph[0]['start'], end=ph[-1]['end'], phones=ph)]), t


def track_of(n, align, lead=0.06):
    from lsdub_engine.core import viseme_track as vt
    return vt.build(align, 0.0, n, FPS, lead=lead)


def take(n, gap=0.1, width=0.0, wobble=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    u = np.stack([gap + wobble * np.sin(t / 3.0) + 0.0 * rng.normal(size=n), width + 0.5 * wobble * np.cos(t / 4.0)], 1)
    return u


def run(phones, n=40, u=None, **kw):
    al, _ = align_of(phones)
    fr = track_of(n, al)
    u = take(n) if u is None else u
    return coart.plan(u, fr, FPS, 0.0, None, None, args(**kw), al, 0.0, None), fr, al, u


def test_blend_is_a_partition_of_unity():
    al, _ = align_of([('D', 0.06), ('AA', 0.2), ('M', 0.08), ('UW', 0.3), ('S', 0.1)])
    n = 40
    fr = track_of(n, al)
    u = take(n, 0.3)
    segs, tc = coart.segments(fr, n, FPS, 0.0, al, 0.0)
    for s in segs:
        if not s['frames']: s['near'] = int(np.argmin(np.abs(tc - 0.5 * (s['a'] + s['b']))))
    coart.segment_targets(segs, u, fr, None, args(), 0.8)
    D, R = coart.dominance(segs, tc, 60.0, 100.0)
    B, W, w0 = coart.blend(segs, D, u, np.array([0.3, 0.0]))
    assert np.allclose(W.sum(0) + w0, 1.0) and (W >= 0).all() and (w0 > 0).all()
    # the blend sits between the targets it mixes (a convex combination)
    lo = np.minimum.reduce([coart.frame_targets(s, u) for s in segs] + [np.full_like(u, [0.3, 0.0])])
    hi = np.maximum.reduce([coart.frame_targets(s, u) for s in segs] + [np.full_like(u, [0.3, 0.0])])
    assert (B >= lo - 1e-9).all() and (B <= hi + 1e-9).all()


def wobble(n, mid=0.5, amp=0.3, wid=0.18):
    """a take that moves (its opening / width wobbling at the syllable rate, like the mv30 takes in their own units)."""
    t = np.arange(n)
    return np.stack([mid + amp * np.sin(t / 2.5), wid * np.cos(t / 3.0)], 1)


def test_closures_reach_closure():
    # "a ma ba": two bilabial closures between open vowels; they reach the PP band AND the take's own closure depth
    n = 40
    u = wobble(n, 0.55, 0.3)
    (x, pp, info), fr, al, u = run([('AA', 0.2), ('M', 0.08), ('AA', 0.2), ('B', 0.05), ('AA', 0.2)], n=n, u=u)
    shut = min(mw.V_OPEN['PP'][1], np.percentile(u[:, 0], 5))
    for p in al['words'][0]['phones']:
        if p['p'] in ('M', 'B'):
            core = [k for k in range(n) if p['start'] - 0.02 <= (k + 0.5) / FPS < p['end'] + 0.02]
            assert core and all(x[k, 0] <= shut + 1e-9 for k in core), (p, x[core, 0])
            assert all(pp[k] for k in core)
    assert x[:, 0].max() > 0.6                         # and the vowels around them still open


def test_closures_reach_the_takes_own_closure_depth():
    # a take whose shut lips sit BELOW the PP band's target: the closure goes as deep as the take's own p5
    n = 40
    u = wobble(n, 0.4, 0.45)                    # opening -0.05 .. 0.85: its p5 is below 0.018
    (x, pp, info), fr, al, u = run([('AA', 0.2), ('M', 0.1), ('AA', 0.2)], n=n, u=u)
    p5 = np.percentile(u[:, 0], 5)
    assert p5 < 0.018
    assert all(x[k, 0] <= p5 + 1e-9 for k in info['closures']) and info['closures']


def test_short_vowel_undershoots_long_vowel_reaches():
    # a shut take; an AA of 40 ms vs one of 400 ms between weak consonants (D ... T)
    T = coart.to_band_soft(0.1, *mw.V_OPEN['aa'], 0.8)
    (xs, _, _), fr_s, al_s, _ = run([('D', 0.08), ('AA', 0.04), ('T', 0.08)], n=24, u=take(24, 0.1), coart_prior=0.0)
    (xl, _, _), fr_l, al_l, _ = run([('D', 0.08), ('AA', 0.40), ('T', 0.08)], n=34, u=take(34, 0.1), coart_prior=0.0)
    _ps, pl = al_s['words'][0]['phones'][1], al_l['words'][0]['phones'][1]
    mid_l = int(round(0.5 * (pl['start'] + pl['end']) * FPS - 0.5))
    assert xs[:, 0].max() < 0.7 * T, (xs[:, 0].max(), T)                # the 40 ms vowel only gets part of the way
    assert abs(xl[mid_l, 0] - T) < 0.05 * T, (xl[mid_l, 0], T)          # the 400 ms vowel reaches its target
    # and more undershoot = less of the way
    (xs2, _, _), _, _, _ = run([('D', 0.08), ('AA', 0.04), ('T', 0.08)], n=24, u=take(24, 0.1), coart_prior=0.0, coart_undershoot=0.0)
    assert xs2[:, 0].max() > xs[:, 0].max()


def test_dynamics_prior_lands_velocity_near_the_take():
    # a natural take (a smooth wobble, ~ the mv30 takes' own velocity in their units) and a busy phone string whose
    # blended targets (closures every 5th syllable) step much faster than the take moves
    rng = np.random.default_rng(1)
    ph = []
    for k in range(14):
        ph += [('M' if k % 5 == 0 else 'D', 0.05), (['AA', 'IY', 'UW', 'EH'][k % 4], float(rng.uniform(0.06, 0.16)))]
    al, t_end = align_of(ph)
    n = int(t_end * FPS) + 6
    fr = track_of(n, al)
    t = np.arange(n)
    u = np.stack([0.4 + 0.3 * np.sin(t / 2.5), 0.18 * np.cos(t / 3.0)], 1)
    v_take = coart.p95s(u[:, 0])[0]
    ratio = {}
    for p in (0.0, 0.5, 1.0):
        x, pp, info = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=p), al, 0.0, None)
        ratio[p] = coart.p95s(x[:, 0])[0] / v_take
        assert all(x[k, 0] <= mw.V_OPEN['PP'][1] + 1e-9 for k in np.flatnonzero(pp))     # the prior never opens a closure
    assert ratio[0.0] > ratio[0.5] > ratio[1.0], ratio                  # the prior calms the plan, monotonically ...
    assert 0.75 < ratio[1.0] < 1.33, ratio                              # ... to within 25-33 % of the take's own velocity p95


def test_skipped_span_is_free():
    al, _ = align_of([('AA', 0.2), ('D', 0.08), ('AA', 0.4), ('M', 0.1), ('AA', 0.2)])
    n = 40
    fr = track_of(n, al)
    for k in range(10, 22): fr[k] = dict(fr[k], skip=True)
    u = take(n, 0.2, 0.0, wobble=0.15)
    x, pp, info = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.0), al, 0.0, None)
    segs = [s for s in info['segments'] if s['v'] == 'skip']
    assert len(segs) == 1 and not any(13 <= k <= 19 for k in info['closures'])
    assert np.abs(x[13:19] - u[13:19]).max() < 0.12                  # inside the span: the take's own shape, no viseme


def test_jaw_share_one_is_the_v9_jaw_reference():
    rng = np.random.default_rng(0)
    T, K = 60, len(mw.CTRL)
    F = np.stack([rng.uniform(2, 20, T), rng.uniform(50, 60, T)], 1)
    J = rng.normal(0, 1, (2, K, 3))
    base = rng.normal(0, 5, (K, 3))
    X = base + np.tensordot(F, J, axes=(1, 0))
    ix = {k: j for j, k in enumerate(mw.CTRL)}
    m9 = mw.MouthModel(X, F)
    m9.jaw_reference([ix[k] for k in mw.UPPER])
    m1 = mw.MouthModel(X, F)
    s, s_take = coart.jaw_split(m1, 1.0)
    assert s == 1.0 and np.allclose(m1.J, m9.J)
    m0 = mw.MouthModel(X, F)
    coart.jaw_split(m0, 0.0)
    [ix[k] for k in mw.UPPER]
    lo = [ix[k] for k in coart.LOWER_LIP]
    assert np.allclose(m0.J[0][lo].mean(0), 0.0, atol=1e-9)          # share 0: the lower lip stays, the upper lip carries it
    ma = mw.MouthModel(X, F)
    sa, st = coart.jaw_split(ma, 'auto')
    assert st <= sa <= 1.0


def test_warp_target_v9_is_v9_and_coart_plans(tmp_path):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_frame_gate_maker import make_clip, args as clip_args
    vid, frames, S = make_clip(tmp_path)
    n = len(frames)
    a0 = clip_args(tmp_path, vid, n)
    c0 = mw.WarpClip(a0).run_batch()
    a1 = clip_args(tmp_path, vid, n, ['--warp-target', 'v9'])
    c1 = mw.WarpClip(a1).run_batch()
    assert np.array_equal(c0.Ft, c1.Ft) and np.array_equal(c0.Fe, c1.Fe) and np.array_equal(c0.model.J, c1.model.J)
    assert all(np.array_equal(x, y) for x, y in zip(c0.outs, c1.outs))
    a2 = clip_args(tmp_path, vid, n, ['--warp-target', 'coart'])
    c2 = mw.WarpClip(a2)
    assert c2.Ft.shape == c0.Ft.shape and np.isfinite(c2.Ft).all() and not np.allclose(c2.Ft, c0.Ft)
    assert os.path.exists(str(tmp_path / 'w_coart.json'))


def test_range_matches_the_takes_own_distribution():
    # a busy phrase over a take that opens wide (0.0 .. 1.0): the planned openings span the take's own p5..p95, not a
    # compressed middle (and --coart-range 0 leaves the compressed blend)
    ph = [('D', 0.06), ('AA', 0.12), ('N', 0.06), ('IH', 0.1), ('S', 0.08), ('EH', 0.14), ('T', 0.05), ('UW', 0.12), ('D', 0.06), ('AA', 0.16)]
    al, t_end = align_of(ph, t=0.05)
    n = int(t_end * FPS) + 2
    fr = track_of(n, al)
    u = wobble(n, 0.5, 0.5, 0.9)
    rng = lambda X: np.ptp(np.percentile(X, [5, 95]))
    x1, _, i1 = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.3), al, 0.0, None)
    x0, _, i0 = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.3, coart_range=0.0), al, 0.0, None)
    r1, r0 = rng(x1[:, 0]) / rng(u[:, 0]), rng(x0[:, 0]) / rng(u[:, 0])
    assert r0 < 0.85 and 0.85 < r1 < 1.15, (r0, r1)
    from scipy.stats import spearmanr
    assert spearmanr(x1[:, 0], x0[:, 0])[0] > 0.8        # the same plan (timing, order of shapes), the take's range


def test_pause_eases_to_the_faces_rest_shape():
    # two words with a 400 ms pause between them and a 60 ms gap inside the first: the long one rests, the short one not
    al = dict(words=[dict(w='a', phones=[dict(p='AA', start=0.10, end=0.30), dict(p='AA', start=0.36, end=0.56)]),
                     dict(w='b', phones=[dict(p='AA', start=0.96, end=1.20)])])
    for wd in al['words']: wd['start'], wd['end'] = wd['phones'][0]['start'], wd['phones'][-1]['end']
    n = 34
    fr = track_of(n, al)
    u = wobble(n, 0.55, 0.4)
    mid = [k for k in range(n) if 0.66 <= (k + 0.5) / FPS <= 0.86]
    gap = [k for k in range(n) if 0.30 <= (k + 0.5) / FPS <= 0.36]
    for rho in (0.0, 1.0):          # the rest holds with and without the range match
        x, pp, info = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.3, coart_range=rho), al, 0.0, None)
        rest = np.array(info['rest'])
        assert rest[0] <= coart.REST_OPEN_MAX + 1e-9 and np.abs(x[mid, 0] - rest[0]).max() < 0.08, (rho, x[mid, 0], rest)
    # (this synthetic take is closed half of its active time, so the range match pulls everything between the anchors
    # down; the gap / pause contrast is checked on the plan itself)
    x, pp, info = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.3, coart_range=0.0), al, 0.0, None)
    assert x[gap, 0].min() > rest[0] + 0.25                          # a 60 ms gap inside a word does not rest
    w2 = [k for k in range(n) if 1.0 <= (k + 0.5) / FPS <= 1.16]
    assert x[w2, 0].max() > 0.75                                     # and the word after the pause opens fully
    x_off, _, _ = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.3, coart_range=0.0, coart_rest_ms=0), al, 0.0, None)
    assert x_off[mid, 0].mean() > x[mid, 0].mean() + 0.1             # --coart-rest-ms 0: the pause is not a rest


def test_dyn_source_is_followed_outside_closures_and_pauses(tmp_path):
    # a model trajectory on the song timeline (a CSV like v10b's a2f_cut_25fps.csv): with mix 1 the plan follows its
    # mapped curve; the aligner's closure stays hard; inside a pause the rest wins; mix 0 ignores it
    import csv
    t = np.arange(0, 60) / FPS
    g = 0.6 + 0.5 * np.sin(2 * np.pi * 2.2 * t)
    wd = 5.0 + 0.3 * np.cos(2 * np.pi * 1.3 * t)
    p = tmp_path / 'dyn.csv'
    with open(p, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 'time_s', 'jawOpen', 'geo_gap_cm', 'geo_width_cm'])
        for k in range(len(t)): w.writerow([k, f'{t[k]:.3f}', 0, f'{g[k]:.4f}', f'{wd[k]:.4f}'])
    ph = [('AA', 0.30), ('M', 0.08), ('AA', 0.30), ('IH', 0.25), ('EH', 0.3)]
    al, t_end = align_of(ph, t=0.04)
    n = int(t_end * FPS)
    fr = track_of(n, al)
    u = wobble(n, 0.5, 0.45, 0.6)
    dyn = coart.load_dyn(str(p), 0.0, n, FPS)
    assert dyn is not None and dyn.shape == (n, 2)
    x1, pp, i1 = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.0, coart_dyn_mix=1.0, coart_undershoot=0.0), al, 0.0, None, None, dyn)
    x0, _, _ = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.0, coart_dyn_mix=0.0, coart_undershoot=0.0), al, 0.0, None, None, dyn)
    Dm = np.array(i1['dyn'])
    cl = i1['closures']
    free = np.array([all(abs(k - c) > 2 for c in cl) for k in range(n)])       # away from the hard closure
    assert np.corrcoef(x1[free, 0], Dm[free, 0])[0, 1] > 0.85 > np.corrcoef(x0[free, 0], Dm[free, 0])[0, 1]
    assert i1['closures'] and all(x1[k, 0] <= min(mw.V_OPEN['PP'][1], np.percentile(u[:, 0], 5)) + 1e-9 for k in i1['closures'])
    assert coart.load_dyn(str(p), 5.0, n, FPS) is None                # a clip the CSV does not cover: no source


def test_isolated_closure_survives_the_warps_dead_band():
    # the take is only slightly open around a one-frame closure; mouth_warp's dead-band confidence (eased over +-1 frame)
    # would apply only part of the residual there: the pressed target makes the applied residual reach the cap
    from scipy.ndimage import gaussian_filter1d
    n = 9
    Fs = np.tile([12.0, 80.0], (n, 1))
    Ft = Fs.copy()
    Ft[4, 0] = 7.0
    scale = np.array([94.0, 20.0])
    caps = {4: 7.0}
    applied = lambda T: Fs[:, 0] + gaussian_filter1d(mw.confidence(T - Fs, scale, 0.04, 0.12), 1.0, mode='nearest') * (T[:, 0] - Fs[:, 0])
    assert applied(Ft)[4] > 9.0                                          # unpressed: the closure arrives at ~2/3
    P = coart.press_closures(Ft, Fs, [4], caps, scale, 0.04, 0.12, 0.15 * 94)
    assert abs(applied(P)[4] - 7.0) < 0.3 and P[4, 0] >= 7.0 - 0.15 * 94 - 1e-9 and np.array_equal(np.delete(P, 4, 0), np.delete(Ft, 4, 0))


def test_range_map_keeps_the_neutral_level():
    # the plan's p5 / p95 go to the take's, the neutral level stays: shut stays shut, round stays round, spread spread,
    # and one extreme class target does not flatten the others on its side (they are clipped to the take's range first)
    rng = np.random.default_rng(0)
    take = rng.normal(0.0, 0.6, 400)
    plan = np.r_[rng.normal(-0.3, 0.2, 300), np.full(20, -2.6), rng.normal(0.8, 0.2, 80)]
    f = coart.qmap_fn(plan, take, 'neutral')
    assert abs(float(f(0.0))) < 1e-9 and f(-0.3) < 0 and f(0.5) > 0
    assert abs(f(np.percentile(plan, 95)) - np.percentile(take, 95)) < 1e-6 and abs(f(np.percentile(plan, 5)) - np.percentile(take, 5)) < 1e-6
    assert np.all(np.diff(f(np.linspace(-3, 2, 200))) >= -1e-12)        # monotone


def test_vocalisation_span_follows_the_vocalisations_mode():
    # v10a marks a vocalisation span with `skip`: free by default (the take's own shape), a rest with --vocalisations rest,
    # planned as words with --vocalisations mouth
    al = dict(words=[dict(w='uh', phones=[dict(p='AH', start=0.10, end=0.90)])])
    al['words'][0]['start'], al['words'][0]['end'] = 0.10, 0.90
    n = 30
    fr = track_of(n, al)
    u = wobble(n, 0.55, 0.35)
    for k in range(5, 20): fr[k] = dict(fr[k], skip=True)
    mid = list(range(9, 16))
    x_free, _, i_free = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.0, coart_range=0.0), al, 0.0, None)
    x_rest, _, i_rest = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.0, coart_range=0.0, vocalisations='rest'), al, 0.0, None)
    x_mouth, _, i_mouth = coart.plan(u, fr, FPS, 0.0, None, None, args(coart_prior=0.0, coart_range=0.0, vocalisations='mouth'), al, 0.0, None)
    assert any(s['v'] == 'skip' for s in i_free['segments']) and np.abs(x_free[mid, 0] - u[mid, 0]).max() < 0.12
    assert np.abs(x_rest[mid, 0] - i_rest['rest'][0]).max() < 0.08
    assert not any(s['v'] in ('skip', 'rest') and s['a'] < 0.8 and s['b'] > 0.2 for s in i_mouth['segments']) and x_mouth[mid, 0].mean() > 0.7
