# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""cone.py — the dub-safe cone: which head poses the mouth paint holds up on (measured on the paper's close-ups).

  green: shared mouth library works as-is (>= 90% of tracked frames)     |yaw| <= 16, -15 <= pitch <= 17, |roll| <= 20
  amber: needs a per-shot (pose-native) library |yaw| <= 37, |pitch| <= 32,       |roll| <= 28
  red:   re-stage the shot (or re-shoot)
Pitch sign follows `lsdub scan` (negative = head tilted back / up). Also wants the face >= 300 px tall at 1080p.
"""
import argparse
import json

GREEN = {"yaw": (-16, 16), "pitch": (-15, 17), "roll": (-20, 20)}
AMBER = {"yaw": (-37, 37), "pitch": (-32, 32), "roll": (-28, 28)}


def zone(yaw, pitch, roll):
    if yaw is None:
        return "lost"
    for name, c in (("green", GREEN), ("amber", AMBER)):
        if all(c[k][0] <= v <= c[k][1] for k, v in (("yaw", yaw), ("pitch", pitch), ("roll", roll))):
            return name
    return "red"


def report(scan):
    rows = scan["frames"]
    z = [zone(r.get("yaw"), r.get("pitch"), r.get("roll")) for r in rows]
    found = max(1, len(z) - z.count("lost"))   # frames the tracker lost don't count for or against the shot
    frac = {k: round(z.count(k) / found, 3) for k in ("green", "amber", "red")}
    frac["lost"] = round(z.count("lost") / max(1, len(z)), 3)
    if frac["green"] >= 0.9:
        advice = "shared library"
    elif frac["green"] + frac["amber"] >= 0.9:
        advice = "per-shot library (lsdub mouths --frame <cluster keyframe>), paint with --pose3d"
    else:
        advice = "re-stage: the face leaves the dub-safe cone (see blender-addon/)"
    return {"fractions": frac, "advice": advice, "zones": z}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lsdub cone")
    ap.add_argument("scan", help="JSON from `lsdub scan`")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    r = report(json.load(open(a.scan)))
    print(json.dumps({k: r[k] for k in ("fractions", "advice")}))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    return r


if __name__ == "__main__":
    main()
