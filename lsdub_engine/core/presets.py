# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""presets.py: per-project cast settings. The engine's defaults are NEUTRAL (any named character can sing, no hook lines,
no family names); a project picks a preset by name.

  singers         characters that sing (None = any named, non-'unknown' character in the cast scan)
  hooks           hook phrases that need a close-up
  names           shooting-script word -> character ('Red sings' -> red_nova); 'duo' = all singers
  families        near-identical extras named by family in reports
  colours         review-sheet colours (BGR)
  recipe          the paint recipe (recipes.py) of this preset's kit faces
  face_overrides  identity -> that face's own argv. A face listed here gets the recipe, opt-in per face, never a default;
                  any other face paints plain. Lip-lock geometry lives here: it is one take's data, not the recipe's
"""

from dataclasses import dataclass, field

from . import recipes


@dataclass(frozen=True)
class Preset:
    singers: tuple | None = None
    hooks: tuple = ()
    names: dict = field(default_factory=dict)
    families: dict = field(default_factory=dict)
    colours: dict = field(default_factory=dict)
    recipe: str = "plain"
    face_overrides: dict = field(default_factory=dict)

    def flags_for(self, identity):
        """the paint argv of one face."""
        if identity not in self.face_overrides:
            return ()
        return recipes.get(self.recipe).expand(self.face_overrides[identity])

    @property
    def paint_flags(self):
        """identity -> paint argv, for every face with a recipe."""
        return {k: self.flags_for(k) for k in self.face_overrides}


NEUTRAL = Preset()

# landing: the Pleiadians Don't Dance demo (original characters and song): two singers (Red, Sky), three hook lines,
# Pleiadian delegates and greys as extras, and Red's mouth kit. The example workflows use it.
LANDING = Preset(
    singers=("red_nova", "sky_lux"),
    hooks=("pleiadians don't dance", "don't they try", "why don't they try"),
    names={"red": "red_nova", "sky": "sky_lux"},
    families={"alcyone": "pleiadian", "merope": "pleiadian", "maia": "pleiadian", "zeta": "grey", "mantid": "mantis"},
    colours={"red_nova": (40, 40, 230), "sky_lux": (200, 80, 255)},
    recipe="layers_v16_1",
    face_overrides={"red_nova": ("--warp-lip-lock", "107,169.7,157.6")},
)

PRESETS = {"none": NEUTRAL, "landing": LANDING}


def get(p=None):
    """None / '' / 'none' -> NEUTRAL; a name -> that preset; a Preset passes through."""
    if isinstance(p, Preset):
        return p
    if not p:
        return NEUTRAL
    if p not in PRESETS:
        raise ValueError(f"unknown preset {p!r} (have {sorted(PRESETS)})")
    return PRESETS[p]


def singer_set(p, cast=None):
    """the singers of a preset, or (NEUTRAL) every named character in a cast scan."""
    s = get(p).singers
    if s is not None:
        return set(s)
    if cast is None:
        return set()
    return {t["character"] for r in cast.get("results", []) for t in r.get("tracks", []) if t.get("character") not in (None, "unknown")}
