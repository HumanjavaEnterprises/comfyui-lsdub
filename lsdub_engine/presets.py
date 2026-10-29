# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""presets.py: per-project cast settings. The engine's defaults are NEUTRAL (any named character can sing, no hook
lines, no family names); a project picks a preset by name (`--preset landing`, or LSDubProject in ComfyUI).

  singers      characters that sing (None = any named, non-'unknown' character in the cast scan)
  hooks        hook phrases that need a close-up
  names        shooting-script word -> character ('Red sings' -> red_nova); 'duo' = all singers
  families     near-identical extras named by family in reports
  colours      review-sheet colours (BGR)
  paint_flags  a face kit's warp flags (opt-in per face, never a default)
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    singers: tuple | None = None
    hooks: tuple = ()
    names: dict = field(default_factory=dict)
    families: dict = field(default_factory=dict)
    colours: dict = field(default_factory=dict)
    paint_flags: dict = field(default_factory=dict)


NEUTRAL = Preset()

# landing: the Pleiadians Don't Dance demo (original characters and song) -- two singers (Red, Sky), three hook
# lines, Pleiadian delegates and greys as extras, and Red's mouth-kit settings. The example workflows use it.
LANDING = Preset(
    singers=('red_nova', 'sky_lux'),
    hooks=("pleiadians don't dance", "don't they try", "why don't they try"),
    names={'red': 'red_nova', 'sky': 'sky_lux'},
    families={'alcyone': 'pleiadian', 'merope': 'pleiadian', 'maia': 'pleiadian', 'zeta': 'grey', 'mantid': 'mantis'},
    colours={'red_nova': (40, 40, 230), 'sky_lux': (200, 80, 255)},
    paint_flags={'red_nova': (
        '--warp-size-cap', '1.0', '--warp-erase', 'closures', '--warp-lip-lock', '107,169.7,157.6', '--warp-teeth', 'rigid',
        '--warp-teeth-anchor', 'skin', '--warp-teeth-jaw', 'plan', '--warp-teeth-tongue-drop', '0.04', '--warp-teeth-plates', '3',
        '--warp-teeth-throat', '0.6', '--warp-teeth-mirror', '--warp-compose', 'layers', '--warp-teeth-chin-pin', 'arc', '--warp-chin-pin-split',
        '3.0', '--warp-chin-pin-gain', '0.3', '--warp-tongue', 'wedge', '--warp-tongue-lit', '0.55', '--warp-arch-harvest-lower',
        '1', '--warp-cover', 'fill', '--warp-arch-harvest', '3', '--warp-arch-synth', '--warp-arch-desat', '0.4',
        '--warp-arch-shade', '0.6', '--warp-arch-facing', '0.6', '--warp-lip-rim', '0.12,0.008', '--warp-lip-wet', '0.2',
        '--warp-layers-smooth', '1.0', '--warp-gate', 'repair', '--warp-gate-cands', 'a,b,c,d,e,f,g,h', '--warp-interior-clip', '--warp-cover-corners',
        '1.0', '--warp-gate-reg', '--warp-reg', 'single', '--warp-reg-pin', 'nose', '--warp-arch-build', 'v12',
        '--warp-arch-layout', 'anthro', '--warp-arch-corridor', '0.24', '--warp-arch-texel', '4', '--warp-arch-mipmap', '--warp-arch-enamel',
        '0.6', '--warp-arch-enamel-mix', '1.0,0.5,1.2,0.9', '--warp-arch-gums', '0.85', '--warp-tongue-cover', '0.9', '--warp-lower-real',
        '0.9', '--warp-target', 'coart', '--coart-undershoot', '0.0', '--coart-prior', '0.0', '--coart-dom-width',
        '30', '--coart-hull-grow', '3.0', '--coart-vmax-scale', '3.0', '--warp-cavity', 'fill', '--warp-lower-ratio',
        'auto')},
)

PRESETS = {'none': NEUTRAL, 'landing': LANDING}


def get(p=None):
    """None / '' / 'none' -> NEUTRAL; a name -> that preset; a Preset passes through."""
    if isinstance(p, Preset): return p
    if not p: return NEUTRAL
    if p not in PRESETS: raise ValueError(f'unknown preset {p!r} (have {sorted(PRESETS)})')
    return PRESETS[p]


def singer_set(p, cast=None):
    """the singers of a preset, or (NEUTRAL) every named character in a cast scan."""
    s = get(p).singers
    if s is not None: return set(s)
    if cast is None: return set()
    return {t['character'] for r in cast.get('results', []) for t in r.get('tracks', []) if t.get('character') not in (None, 'unknown')}
