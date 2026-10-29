# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
import json, subprocess


from lsdub_engine import animator_pass as ap


def _align():
    # "map" sung from 0.20 s to 0.90 s
    return {"words": [{"w": "map", "start": 0.20, "end": 0.90, "phones": [
        {"p": "M", "start": 0.20, "end": 0.30}, {"p": "AE1", "start": 0.30, "end": 0.80}, {"p": "P", "start": 0.80, "end": 0.90}]}]}


def _tone(path, dur=2.0):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=220:duration={dur}",
                    "-af", "volume=0.5", "-ar", "48000", str(path)], check=True)


def test_animator_closers_held_and_eased(tmp_path):
    al = tmp_path / "a.json"
    al.write_text(json.dumps(_align()))
    wav = tmp_path / "v.wav"
    _tone(wav)
    out = tmp_path / "t.json"
    ap.main(["--align", str(al), "--audio", str(wav), "--t0", "0", "--frames", "30", "--out", str(out)])
    fr = json.loads(out.read_text())["frames"]
    assert len(fr) == 30
    closer = [i for i, f in enumerate(fr) if f["cls"] == "closer" and f["apex"]]
    assert len(closer) >= 2                                   # rule 4: closures held >= 2 frames
    assert all(fr[i]["jaw"] <= 0.25 + 1e-6 for i in closer)  # jaw narrows over the closure apex
    assert max(f["open"] for f in fr) <= 0.75 + 1e-6         # close-up open cap
    vow = [f["alpha"] for f in fr if f["vis"] == "aa"]
    assert vow and max(vow) <= 0.6 + 1e-6                     # close-up articulation gain G = 0.6 on vowels


