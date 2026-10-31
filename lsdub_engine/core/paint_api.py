# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""paint_api.py: the one paint call. A recipe is built up in layers and run through steps.call; a later layer wins, because
paint reads the last value of a repeated flag:

  recipe < mode_mods (the track mode) < face_overrides < user_overrides < operational (--occluder-mask, --out, --warp-json)

A face override takes its recipe slot (Recipe.overrides_at) and removes the same flag from the recipe and the mode mods, so it
wins from there. Every entry point goes through here: sing (both candidates), medium, the node, pipeline as recipe='morph'.

The snap paint flags sit in `operational` for now: they have always come after the user's flags, so they win over them."""

from . import recipes, steps


def _named(argv):
    return {t.split("=", 1)[0] for t in argv if t.startswith("--")}


def flags(recipe="plain", *, lib=None, mode_mods=(), face_overrides=(), user_overrides=(), operational=()):
    """the flags of one paint, without its inputs and outputs. All strings: argv material."""
    face = tuple(map(str, face_overrides))
    named = _named(face)
    return [
        *(("--lib", str(lib)) if lib else ()),
        *recipes.get(recipe).expand(face),
        *recipes.without(tuple(map(str, mode_mods)), named),
        *map(str, user_overrides),
        *map(str, operational),
    ]


def argv(video, track, frames, recipe="plain", *, out, warp_json=None, **layers):
    """the whole argv: morph paints with `--mode morph`, every other recipe with `--mode warp`."""
    mode = "morph" if recipe == "morph" else "warp"
    tail = ["--out", str(out), *(("--warp-json", str(warp_json)) if warp_json else ()), "--no-watermark"]
    return ["--mode", mode, "--video", str(video), "--track", str(track), "--frames", str(frames), *flags(recipe, **layers), *tail]


def call(args):
    """run an argv from argv()."""
    return steps.call("paint", args)


def paint(video, track, frames, recipe="plain", **kw):
    return call(argv(video, track, frames, recipe, **kw))


def expand(args):
    """`--recipe NAME` in a paint argv -> that recipe's flags, in place, so flags after it override it."""
    out, it = [], iter(map(str, args))
    for t in it:
        name, eq, val = t.partition("=")
        if name != "--recipe":
            out.append(t)
            continue
        val = val if eq else next(it, None)
        if val not in recipes.RECIPES:
            raise SystemExit(f"paint: unknown recipe {val!r} (have {sorted(recipes.RECIPES)})")
        out += ["--mode", "morph" if val == "morph" else "warp", *recipes.RECIPES[val].expand()]
    return out


def target(args):
    """the --warp-target these flags end up with, or None."""
    a, found = expand(args), None
    for i, t in enumerate(a):
        if t == "--warp-target" and i + 1 < len(a):
            found = a[i + 1]
        elif t.startswith("--warp-target="):
            found = t.split("=", 1)[1]
    return found


def coart_dyn(args, dyn):
    """--coart-dyn belongs to the coart planner: nothing else reads it."""
    return ("--coart-dyn", dyn, "--coart-dyn-mix", "1.0") if target(args) == "coart" else ()
