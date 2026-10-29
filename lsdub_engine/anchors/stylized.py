#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""stylized.py — classical-animation mouth rules for LSDub's non-human / cutout / puppet path.

Single source of the shape set, phoneme->anchor map, decomposition axes, per-anchor puppet gape, and the
track transforms (snap, on-twos, lead, closure-hold, accent, singing holds) distilled in
docs/MOUTH-ANIMATION-INSIGHTS.md. The photoreal human path does NOT use this module.

Anchors are named by APPEARANCE, not sound. Each anchor records:
  rep      - a representative name from the 15-viseme library set (so libraries keyed by that set still load)
  openness - jaw drop 0..1
  round    - lip rounding -1 (wide/spread) .. +1 (round/pursed), 0 neutral
  contact  - 'closed' | 'fv' (teeth-on-lip) | 'open'
  gape     - puppet jaw-open fraction of max (felt-mouth hinge; color-blind, openness only)
"""
from __future__ import annotations
import re

# anchor: (rep_vis, openness, round, contact, gape)
ANCHORS = {
    'rest':   ('sil', 0.05,  0.0, 'open',   0.00),
    'closed': ('PP',  0.00,  0.0, 'closed', 0.00),
    'ah':     ('aa',  1.00,  0.0, 'open',   1.00),
    'ai':     ('aa',  0.70,  0.0, 'open',   0.60),
    'ee':     ('E',   0.50, -1.0, 'open',   0.20),
    'oh':     ('oh',  0.60,  1.0, 'open',   0.60),
    'oo':     ('ou',  0.30,  1.0, 'open',   0.30),
    'fv':     ('FF',  0.12,  0.0, 'fv',     0.15),
    'L':      ('nn',  0.30,  0.0, 'open',   0.15),
    'cons':   ('DD',  0.25,  0.0, 'open',   0.15),
}
ANCHOR_REP = {k: v[0] for k, v in ANCHORS.items()}
OPENNESS   = {k: v[1] for k, v in ANCHORS.items()}
ROUND      = {k: v[2] for k, v in ANCHORS.items()}
CONTACT    = {k: v[3] for k, v in ANCHORS.items()}
GAPE       = {k: v[4] for k, v in ANCHORS.items()}

CLOSURES = {'closed', 'fv'}                      # held-then-popped, minimal lead, never blended
VOWELS   = {'ah', 'ai', 'ee', 'oh', 'oo'}        # carry the motion; accents land here

# ARPAbet (stress digits stripped) -> anchor. Thin lookup; library is indexed by axes, not phoneme.
ARPA2ANCHOR = {
    'P': 'closed', 'B': 'closed', 'M': 'closed',
    'F': 'fv', 'V': 'fv', 'PH': 'fv',
    'L': 'L',
    'AA': 'ah', 'AE': 'ah', 'AY': 'ah', 'AW': 'ah',
    'AH': 'ai', 'AO': 'ai',
    'IY': 'ee', 'IH': 'ee', 'EY': 'ee', 'EH': 'ee',
    'OW': 'oh',
    'UW': 'oo', 'UH': 'oo', 'W': 'oo', 'Q': 'oo',
    'SIL': 'rest', 'SP': 'rest', 'SPN': 'rest', '': 'rest',
}
for _c in ['C', 'D', 'G', 'K', 'N', 'R', 'S', 'T', 'Z', 'Y', 'JH', 'J', 'SH', 'CH', 'TH', 'DH', 'HH', 'NG']:
    ARPA2ANCHOR[_c] = 'cons'

# the 15-viseme library set (viseme_track.VIS) -> anchor
VIS2ANCHOR = {
    'sil': 'rest', 'PP': 'closed', 'FF': 'fv', 'TH': 'L', 'DD': 'cons', 'kk': 'cons',
    'CH': 'cons', 'SS': 'cons', 'nn': 'cons', 'RR': 'cons',
    'aa': 'ah', 'E': 'ee', 'ih': 'ee', 'oh': 'oh', 'ou': 'oo',
}

# presets: the allowed anchor set (others remap to nearest). crude = Rhubarb A-F; southpark = 4 + rest.
PRESETS = {
    'classic':   set(ANCHORS),
    'crude':     {'closed', 'cons', 'ee', 'ah', 'oh', 'oo'},
    'southpark': {'rest', 'closed', 'ah', 'ee', 'oo'},
}


def phoneme_to_anchor(p: str) -> str:
    """ARPAbet phoneme (stress digits ok) -> anchor; unknowns route to nearest open consonant."""
    p = re.sub(r'\d+$', '', (p or '').strip().upper())
    return ARPA2ANCHOR.get(p, 'cons')


def _nearest_in(anchor: str, allowed: set) -> str:
    if anchor in allowed:
        return anchor
    # nearest by (openness, round, contact-match) distance
    o0, r0, c0 = OPENNESS[anchor], ROUND[anchor], CONTACT[anchor]
    best, bd = 'rest' if 'rest' in allowed else next(iter(allowed)), 1e9
    for a in allowed:
        d = (OPENNESS[a] - o0) ** 2 + 0.5 * (ROUND[a] - r0) ** 2 + (0.0 if CONTACT[a] == c0 else 0.4)
        if d < bd:
            bd, best = d, a
    return best


def _scale(n_at24: int, fps: float) -> int:
    return max(1, round(n_at24 * fps / 24.0))


def build_stylized_track(frames, fps=25.0, preset='classic', lead_vowel=2, lead_closed=1,
                         min_hold=2, on_twos=True, singing=False, accent=True, cartoon=1.0):
    """Turn a viseme track (list of per-frame {'vis','vis2','mix','open',...}) into a stylized, snap-ready
    per-frame list: [{'anchor','rep','open','sq_w','sq_h','accent'}].

    Applies: VIS->anchor, preset remap, lead (vowels>closures), min-hold / on-twos, closure-hold,
    rest-on-pause, accent (animate-to-vowels), singing vowel holds + vowel modification, and
    volume-conserving squash/stretch (sq_w width mult, sq_h height mult).
    """
    allowed = PRESETS.get(preset, PRESETS['classic'])
    n = len(frames)
    if n == 0:
        return []
    lv, lc = _scale(lead_vowel, fps), _scale(lead_closed, fps)
    mh = max(1, _scale(min_hold, fps))

    def _runs(seq):
        out, s = [], 0
        for i in range(1, len(seq) + 1):
            if i == len(seq) or seq[i] != seq[s]:
                out.append([s, i, seq[s]])
                s = i
        return out

    base = [_nearest_in(VIS2ANCHOR.get(f.get('vis', 'sil'), 'rest'), allowed) for f in frames]
    opn0 = [float(f.get('open', OPENNESS[base[i]])) for i, f in enumerate(frames)]

    # 1) lead: shift each shape's window EARLIER (picture ahead of sound). Vowels lead by lv, closures by
    #    lc. Paint runs in time order so an upcoming shape overwrites the tail of the previous = anticipation.
    #    Never later than its sound (we only shift earlier). src[k] tracks which source frame k came from.
    canvas = [None] * n
    src = [None] * n
    for s, e, a in _runs(base):
        lead = lc if a in CLOSURES else lv
        for k in range(max(0, s - lead), max(0, e - lead)):
            canvas[k] = a
            src[k] = min(n - 1, k + lead)
    last_a, last_s = base[0], 0
    for k in range(n):
        if canvas[k] is None:
            canvas[k], src[k] = last_a, last_s
        else:
            last_a, last_s = canvas[k], src[k]
    base = canvas
    opn = [opn0[src[k]] for k in range(n)]

    # 2) singing: bridge short rest/cons gaps between identical held vowels; modify high vowels.
    if singing:
        i = 0
        while i < n:
            if base[i] in VOWELS:
                j = i + 1
                while j < n and base[j] not in VOWELS and base[j] not in CLOSURES:
                    j += 1
                if j < n and base[j] == base[i] and (j - i) <= _scale(6, fps):
                    for k in range(i, j):
                        base[k] = base[i]
                i = j
            else:
                i += 1
        for i in range(n):
            if base[i] in VOWELS and opn[i] > 0.7:
                base[i] = {'oo': 'oh', 'ee': 'ah', 'ai': 'ah'}.get(base[i], base[i])
                base[i] = _nearest_in(base[i], allowed)

    # 3) runs: enforce min-hold (drop too-short non-closure runs into a neighbour) + on-twos; closures >=2.
    rr = _runs(base)
    for idx, (s, e, a) in enumerate(rr):
        L = e - s
        need = 2 if a in CLOSURES else mh
        if L < need:
            # closures: try to extend to 2 by borrowing from the longer neighbour; else fold into neighbour
            if a in CLOSURES and L >= 1:
                continue  # keep it; the on-twos pass below will extend to 2
            nb = None
            if idx > 0:
                nb = rr[idx - 1][2]
            if idx + 1 < len(rr) and (nb is None):
                nb = rr[idx + 1][2]
            if nb is not None:
                for k in range(s, e):
                    base[k] = nb

    # closure held >= 2 frames then popped open (never blended); min hold >= mh for everything (= on twos).
    seq = list(base)
    i = 1
    while i < n:
        if seq[i] != seq[i - 1]:
            hold = 2 if seq[i] in CLOSURES else mh
            # extend this new shape to its minimum hold by overwriting the following too-short frames,
            # but never overwrite an upcoming closure (a closure must read on time).
            j = i + 1
            while j < n and (j - i) < hold and seq[j] not in CLOSURES:
                seq[j] = seq[i]
                j += 1
            i = j
        else:
            i += 1
    base = seq

    # 4) rest on pause (only genuine silence; never demote a vowel or closure)
    for i in range(n):
        if opn[i] < 0.03 and base[i] not in CLOSURES and base[i] not in VOWELS:
            base[i] = _nearest_in('rest', allowed)

    # 5) accent: vowel onset after a closure/rest -> +40% openness (×1.4 peak), decaying over the run.
    acc = [False] * n
    if accent:
        for i in range(n):
            if base[i] in VOWELS and (i == 0 or base[i - 1] in CLOSURES or base[i - 1] == 'rest'):
                acc[i] = True

    out = []
    for i in range(n):
        a = base[i]
        o = max(opn[i], OPENNESS[a] * 0.5)
        if acc[i]:
            o = min(1.0, o * 1.4)
        # volume-conserving squash/stretch: more open -> narrower; round anchors narrower; closed -> wider.
        stretch = 1.0 + cartoon * 0.4 * (o - OPENNESS['rest'])        # 1.0..~1.4 height on a wide open
        sq_h = stretch
        sq_w = 1.0 / (stretch ** 0.5)
        if ROUND[a] > 0:
            sq_w *= (1.0 - 0.18 * cartoon * ROUND[a])                 # pucker narrows
        elif ROUND[a] < 0:
            sq_w *= (1.0 + 0.12 * cartoon * (-ROUND[a]))              # spread widens
        if a in CLOSURES:
            sq_h, sq_w = 1.0, 1.0 + 0.06 * cartoon
        out.append({'anchor': a, 'rep': ANCHOR_REP[a], 'open': round(o, 3),
                    'sq_w': round(sq_w, 3), 'sq_h': round(sq_h, 3), 'accent': acc[i]})
    return out


def puppet_gape_track(frames, fps=25.0, accent=True, singing=False):
    """Per-frame jaw-open fraction (0..1) for puppet mode: per-anchor GAPE, one-open-per-syllable
    (drop consonants between vowels), Henson accent (stressed ×1.4 snappy / unstressed ×0.65), and a
    fast ease-out rise (~2f accent / 3f normal), slower close. Rest/closed held at 0."""
    n = len(frames)
    if n == 0:
        return []
    anc = [VIS2ANCHOR.get(f.get('vis', 'sil'), 'rest') for f in frames]
    g = [GAPE[a] for a in anc]

    # one open per syllable: a short consonant run flanked by vowels is pulled down to the lower vowel floor
    i = 0
    while i < n:
        if anc[i] in VOWELS:
            j = i + 1
            while j < n and anc[j] == 'cons':
                j += 1
            if j < n and anc[j] in VOWELS and (j - i) <= _scale(4, fps):
                lo = min(g[i], g[j])
                for k in range(i + 1, j):
                    g[k] = min(g[k], lo)      # don't let an intervocalic consonant re-open the jaw
            i = j
        else:
            i += 1

    # accent: vowel onset after a closure/rest gets the Henson punch
    g = list(g)
    for i in range(n):
        if anc[i] in VOWELS and (i == 0 or anc[i - 1] in CLOSURES or anc[i - 1] == 'rest'):
            g[i] = min(1.0, g[i] * 1.4) if accent else g[i]
        elif anc[i] in VOWELS and accent and g[i] < 0.5:
            g[i] *= 0.9

    # ease: fast rise (asymmetric), slower close; work roughly on twos via a short causal+symmetric smooth
    rise = _scale(2, fps)
    out = [0.0] * n
    cur = 0.0
    for i in range(n):
        tgt = g[i]
        if tgt >= cur:
            cur += (tgt - cur) / max(1, rise)          # fast ease-out rise
        else:
            cur += (tgt - cur) / max(1, rise + 1)      # slightly slower close
        out[i] = round(max(0.0, cur), 3)
    return out
