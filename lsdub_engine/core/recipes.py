# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""recipes.py: named paint recipes, the argv a face's render runs with. A recipe says how a mouth is painted; a Preset
(presets.py) says who is in the cast and which recipe they get.

  plain             no face flags: what a face without a preset entry gets
  layers_v16_1      the landing preset's stack, the one that ships. A face's own argv, if it has one, comes from the Preset
                    and slots in at `overrides_at`, so the expanded argv is token for token the preset's flags
  layers_v16_1_interior  layers_v16_1 with the mouth interior filmed from the take (--warp-interior-source take, its pink cap
                    included); teeth fit stays off
  layers_v16_1_shape  layers_v16_1_interior with the plan's gap anchored at the top (--coart-gap-anchor top) and closures that shut
                    (--warp-erase-close), interior donors that stay open (--warp-interior-open-floor), reduced vowels, diphthongs and the
                    function-word gate (--coart-phones, --coart-function-gate), rounding (--warp-round), F / V / TH contact (--warp-contact), the
                    temporal smoothing that keeps closures (--warp-temporal 1.5, spare, width 2) and the plan 40 ms ahead of the vocal
  layers_v16_1_min  the same minus the flags listed in `inert_under`. A candidate, nothing ships it
  morph             the `lsdub pipeline` path: `paint --mode morph` (the paint default), no warp flags

`inert_under` maps a flag to the condition under which it moves no pixel. Two kinds of row:
  a guard read in the code: the condition holds wherever the recipe's other flags hold
  OBSERVED_ONLY: 0 frames moved on the golden crops (two 24-frame windows), no guard behind it. The per-flag zero-frame evidence
    is committed for the local universe only; the hosted universe has only the joint `_min` comparison
Each row is a Rule; with RULES (flags outside any recipe) it drives mouth_paint.validate_paint_args. A Rule says the flag is inert
only when it carries inert=True, and tests/test_flag_interactions.py holds those to the sensitivity block of the golden hashes; a
Rule with inert=False is a note about what the flag no longer does, on a flag that still moves pixels.

What the `_min` proof covers: test_presets.py renders it and the full recipe side by side on the golden crops (synthetic + demo)
and compares the frames. The crops replace the original cut-51 substrate, a music-video take barred from tests/golden; the harness's
`coverage` shows both crops enter the layer stack. Not proven: the OBSERVED_ONLY flags on any longer take (--warp-gate-cands
drops candidate h, which simply never won here), lip-lock under `_min` (no golden case runs lip-lock), and how any of this
looks. layers_v16_1 is also one kit's tuning applied to a performance-built kit, and no ruler or hash covers the performance
frames before an eye pass.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recipe:
    name: str
    argv: tuple = ()
    inert_under: dict = field(default_factory=dict)
    overrides_at: int | None = None  # where per-face argv goes; None = the end

    def expand(self, overrides=()):
        """the recipe's argv with a face's overrides in their slot."""
        at = len(self.argv) if self.overrides_at is None else self.overrides_at
        named = {t for t in overrides if t.startswith("--")}  # a flag the face sets is the face's
        return (*without(self.argv[:at], named), *overrides, *without(self.argv[at:], named))


def _rigid(a):
    return a.warp_teeth == "rigid"


def _layered(a):
    return _rigid(a) and a.warp_compose == "layers"


def _single(a):
    return _layered(a) and a.warp_reg == "single"


def fill_active(a):
    return _layered(a) and a.warp_cavity == "fill"


@dataclass(frozen=True)
class Rule:
    """what `when(parsed args)` means for these flags. inert: they move no pixel; not inert: a note, they still do.
    when=None: no warning, the row only records why the flag is inert. prefix: the flags are name prefixes."""

    flags: tuple
    why: str
    when: object = None
    prefix: bool = False
    inert: bool = True

    def hits(self, names):
        """the flags in `names` this rule is about."""
        return [n for n in names if (n.startswith(self.flags) if self.prefix else n in self.flags)]


SINGLE = "--warp-reg single"
PLATES = "a plates feature, the layer stack never draws it"
_inert = lambda flag, why, when=None: (flag, Rule((flag,), why, when))
INERT = dict(
    (
        # guards read in the code
        _inert(
            "--warp-teeth-anchor",
            f"{SINGLE}: mouth_warp reads it only when not single, or under a global pin",
            lambda a: _single(a) and a.warp_reg_pin != "global",
        ),
        _inert("--warp-teeth-chin-pin", f"{SINGLE}: set_chin_pin runs only when not single", _single),
        _inert("--warp-chin-pin-split", f"{SINGLE}: read only by set_chin_pin", _single),
        _inert("--warp-chin-pin-gain", f"{SINGLE}: read only by set_chin_pin", _single),
        _inert("--warp-gate-reg", "always: the gate reports the ruler and repairs nothing with it"),
        _inert(
            "--warp-lower-ratio",
            "value auto: the default. Any other value is read only under --warp-lower-real > 0 or --warp-cavity fill",
            lambda a: a.warp_lower_real <= 0 and not fill_active(a) and _layered(a),
        ),
        # no guard
        _inert(
            "--warp-arch-harvest-lower",
            "observed on the golden crops: its atlas is the lower arch in L3, which --warp-cavity fill mostly hides inside the opening",
            fill_active,
        ),
        _inert("--warp-gate-cands", "observed on the golden crops: candidate h never won"),
        _inert(
            "--warp-layers-smooth",
            "observed on the golden crops: only prepare() reads it and --warp-reg single's plan_single replaces its output",
            _single,
        ),
        _inert("--warp-teeth-mirror", f"observed on the golden crops: {PLATES}", _layered),
        _inert("--warp-teeth-plates", f"observed on the golden crops: {PLATES}", _layered),
        _inert("--warp-teeth-throat", f"observed on the golden crops: {PLATES}", _layered),
        _inert("--warp-teeth-tongue-drop", f"observed on the golden crops: {PLATES}", _layered),
    )
)
OBSERVED_ONLY = tuple(f for f, r in INERT.items() if r.why.startswith("observed"))

LAYERS_V16_1 = (
    "--warp-size-cap", "1.0", "--warp-erase", "closures", "--warp-teeth", "rigid", "--warp-teeth-anchor", "skin",
    "--warp-teeth-jaw", "plan", "--warp-teeth-tongue-drop", "0.04", "--warp-teeth-plates", "3", "--warp-teeth-throat", "0.6",
    "--warp-teeth-mirror", "--warp-compose", "layers", "--warp-teeth-chin-pin", "arc", "--warp-chin-pin-split", "3.0",
    "--warp-chin-pin-gain", "0.3", "--warp-tongue", "wedge", "--warp-tongue-lit", "0.55", "--warp-arch-harvest-lower", "1",
    "--warp-cover", "fill", "--warp-arch-harvest", "3", "--warp-arch-synth", "--warp-arch-desat", "0.4", "--warp-arch-shade", "0.6",
    "--warp-arch-facing", "0.6", "--warp-lip-rim", "0.12,0.008", "--warp-lip-wet", "0.2", "--warp-layers-smooth", "1.0",
    "--warp-gate", "repair", "--warp-gate-cands", "a,b,c,d,e,f,g,h", "--warp-interior-clip", "--warp-cover-corners", "1.0",
    "--warp-gate-reg", "--warp-reg", "single", "--warp-reg-pin", "nose", "--warp-arch-build", "v12", "--warp-arch-layout", "anthro",
    "--warp-arch-corridor", "0.24", "--warp-arch-texel", "4", "--warp-arch-mipmap", "--warp-arch-enamel", "0.6",
    "--warp-arch-enamel-mix", "1.0,0.5,1.2,0.9", "--warp-arch-gums", "0.85", "--warp-tongue-cover", "0.9", "--warp-lower-real", "0.9",
    "--warp-target", "coart", "--coart-undershoot", "0.0", "--coart-prior", "0.0", "--coart-dom-width", "30",
    "--coart-hull-grow", "3.0", "--coart-vmax-scale", "3.0", "--warp-cavity", "fill", "--warp-lower-ratio", "auto",
)  # fmt: skip


LAYER_ONLY = (
    "--warp-contact",
    "--warp-arch-",
    "--warp-reg",
    "--warp-cavity",
    "--warp-fill-l3",
    "--warp-interior-source",
    "--warp-teeth-fit",
    "--warp-tongue",
    "--warp-cover",
    "--warp-lower-",
    "--warp-lip-rim",
    "--warp-lip-wet",
    "--warp-layers-",
    "--warp-interior-clip",
    "--warp-teeth-chin-pin",
    "--warp-chin-pin-",
)
L3_FLAGS = ("--warp-tongue", "--warp-tongue-lit", "--warp-tongue-cover", "--warp-lower-real", "--warp-lower-mix")
RULES = (
    Rule(
        ("--coart-",),
        "read only by the coart planner, no --warp-target coart here",
        lambda a: a.warp_target != "coart",
        prefix=True,
    ),
    Rule(
        ("--warp-round", "--warp-contact"),
        "read only with the coart planner's phones, no --warp-target coart here",
        lambda a: a.warp_target != "coart",
    ),
    Rule(("--warp-sigma",), "read only by the plain targets, not under this --warp-target", lambda a: a.warp_target in ("coart", "own")),
    Rule(
        ("--warp-tongue",),
        "--warp-reg single zeroes the vowel part of the tongue's pose (rise, tip, retraction: plan_single, tg = 0); the tongue is still "
        "drawn and still moves with the planned opening, smoothed by --warp-reg-tongue-sigma",
        lambda a: _single(a) and a.warp_tongue == "wedge" and a.warp_tongue_pose == "zero",
        inert=False,
    ),
    Rule(
        ("--warp-tongue-pose", "--warp-tongue-pose-sigma"),
        "read only under --warp-reg single",
        lambda a: not _single(a),
    ),
    Rule(("--warp-fill-l3",), "read only under --warp-cavity fill", lambda a: not fill_active(a)),
    Rule(
        L3_FLAGS,
        "--warp-cavity fill mostly hides L3 (lower teeth + tongue) inside the opening: its alpha is 0 there, grown by 0.03 IOD. These "
        "flags still move pixels where L3 shows past that edge, and --warp-lower-real > 0 also sets the fill's lower-teeth ratio and "
        "shades the cavity floor",
        fill_active,
        inert=False,
    ),
    Rule(
        ("--warp-arch-synth",),
        "--warp-arch-layout anthro takes the slots beyond the harvested run (synth_slots, not synth_teeth); --warp-arch-synth still "
        "re-measures the tooth period and highlight and recedes the gum colour",
        lambda a: _layered(a) and a.warp_arch_layout == "anthro",
        inert=False,
    ),
    Rule(
        LAYER_ONLY, "read only by the layer stack: needs --warp-teeth rigid --warp-compose layers", lambda a: not _layered(a), prefix=True
    ),
    Rule(
        ("--warp-teeth-", "--warp-compose", "--warp-profile-plain"),
        "read only by the rigid teeth layer: needs --warp-teeth rigid",
        lambda a: not _rigid(a),
        prefix=True,
    ),
    Rule(("--warp-gate-",), "read only by the frame gate: needs --warp-gate", lambda a: a.warp_gate in (None, "off"), prefix=True),
    Rule(("--warp-lip-lock-",), "read only by the lip lock: needs --warp-lip-lock", lambda a: not a.warp_lip_lock, prefix=True),
)


def fired(names, ns):
    """(rule, flags it is about) for each rule that applies to these set flags; a flag gets its first rule only."""
    rows = {r.flags: r for rec in RECIPES.values() for r in rec.inert_under.values()}
    seen = set()
    for r in (*rows.values(), *RULES):
        hit = [n for n in r.hits(names) if n not in seen]
        if hit and r.when is not None and r.when(ns):
            seen.update(hit)
            yield r, hit


def without(argv, flags):
    """argv minus these flags and their value tokens."""
    out, drop = [], False
    for t in argv:
        if t.startswith("--"):
            drop = t in flags
        if not drop:
            out.append(t)
    return tuple(out)


RECIPES = {
    r.name: r
    for r in (
        Recipe("plain"),
        Recipe("layers_v16_1", LAYERS_V16_1, INERT, overrides_at=4),
        Recipe("layers_v16_1_interior", (*LAYERS_V16_1, "--warp-interior-source", "take"), INERT, overrides_at=4),
        Recipe(
            "layers_v16_1_shape",
            (
                *LAYERS_V16_1,
                "--warp-interior-source",
                "take",
                "--coart-gap-anchor",
                "top",
                "--warp-erase-close",
                "--warp-erase-tone",
                "--warp-erase-near",
                "--warp-interior-open-floor",
                "0.5",
                "--warp-interior-mend",
                "--coart-closure-ramp",
                "1.0",
                "--coart-phones",
                "--coart-function-gate",
                "--warp-round",
                "0.6",
                "--warp-contact",
                "--warp-contact-reach",
                "--warp-nose-lock",
                "--warp-temporal",
                "1.5",
                "--warp-temporal-spare",
                "--warp-temporal-width",
                "2",
                "--coart-advance-ms",
                "40",
            ),
            INERT,
            overrides_at=4,
        ),
        Recipe("layers_v16_1_min", without(LAYERS_V16_1, INERT), INERT),
        Recipe("morph"),
    )
}


MOUTH_STACKS = {
    "plain": "plain",
    "layers": "layers_v16_1",
    "layers_interior": "layers_v16_1_interior",
    "layers_shape": "layers_v16_1_shape",
}  # the pack's mouth_stack option -> the recipe it resolves to


def get(name):
    if name not in RECIPES:
        raise ValueError(f"unknown recipe {name!r} (have {sorted(RECIPES)})")
    return RECIPES[name]
