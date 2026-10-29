"""v10c: the NATURALNESS ruler (lsdub/audits/naturalness.py) on synthetic mouth series, CPU only. Its calibration: the
take against itself reads exactly 0, the take against its own time reversal still reads natural (a reversed natural
motion is still a natural motion: only the rise / fall asymmetry and the phone windows move), and a step-target copy
of the take (every phone held at one shape, switched in one frame: the per-frame target motion) reads unnatural."""
import numpy as np

from lsdub_engine.audits import naturalness as nt


def take_series(n=110, seed=3):
    """a natural-looking sung mouth (IOD units): syllables as smooth raised-cosine openings of random size and length,
    width moving against the opening plus a slow drift, the jaw / lips sharing the gap, small measuring noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    gap = np.full(n, 0.03)
    k = 2
    while k < n - 4:
        L = int(rng.integers(5, 11))
        A = rng.uniform(0.1, 0.35)
        idx = np.arange(k, min(n, k + L))
        gap[idx] += A * 0.5 * (1 - np.cos(2 * np.pi * (idx - k) / L))
        k += L + int(rng.integers(0, 3))
    slow = 0.03 * np.sin(t / 13.0 + 0.4)
    width = 0.55 - 0.35 * (gap - gap.mean()) + slow
    noise = lambda s: rng.normal(0, s, n)
    s = dict(gap=gap + noise(0.004), width=width + noise(0.004), up=0.2 * gap + noise(0.003), lo=0.8 * gap + noise(0.003),
             chin=0.5 * gap + noise(0.003), corner=0.4 * slow + noise(0.003), asym=0.01 + noise(0.004))
    runs, i = [], 0
    vis = ['aa', 'DD', 'ou', 'E', 'PP', 'ih', 'nn', 'oh', 'SS']
    while i < n:
        L = int(rng.integers(2, 6))
        runs.append((i, min(n, i + L) - 1, vis[len(runs) % len(vis)]))
        i += L
    return s, runs


def score(s, ref, runs):
    return nt.compare(nt.series_stats(s, runs=runs), nt.series_stats(ref, runs=runs))


def test_take_vs_itself_is_zero():
    s, runs = take_series()
    r = score(s, s, runs)
    assert r['score'] == 0.0 and all(v == 0.0 for v in r['terms'].values())


def test_time_reversed_take_still_reads_natural():
    s, runs = take_series()
    rev = score(nt.reversed_series(s), s, runs)
    # kinematics, spectrum, shape cloud, asymmetry and the jaw / lip split are exactly invariant to time reversal
    for k in ('kin', 'spec', 'manifold', 'asym', 'jawlip'): assert rev['terms'][k] < 1e-9, (k, rev['terms'])
    assert rev['score'] < 0.15, rev


def test_step_target_track_reads_unnatural():
    s, runs = take_series()
    rev = score(nt.reversed_series(s), s, runs)['score']
    step = score(nt.step_version(s, runs), s, runs)
    assert step['score'] > 0.5 and step['score'] > 4 * max(rev, 0.05), (step, rev)
    assert step['terms']['kin'] > 0.1                   # the steps' velocity / acceleration statistics are not the take's


def test_spectrum_is_reversal_symmetric_and_normalised():
    s, _ = take_series()
    f, P, sh = nt.psd(s['gap'])
    f2, P2, sh2 = nt.psd(s['gap'][::-1].copy())
    assert np.allclose(P, P2) and np.allclose(sh, sh2) and abs(P.sum() - 1) < 1e-9


def test_cloud_counts_shapes_outside_the_face():
    s, runs = take_series()
    wide = dict(s)
    wide['width'] = s['width'] + 0.6 * (s['gap'] - s['gap'].mean())   # wide AND open
    c = nt.cloud(s, wide)
    assert c['hull_out'] > c['hull_out_ref'] + 0.1 and c['kde'] < c['kde_ref']
