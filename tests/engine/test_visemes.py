from lsdub_engine import viseme_track as vt


def _align():
    # "map" sung from 0.10 s to 0.70 s
    return {"words": [{"w": "map", "start": 0.10, "end": 0.70, "phones": [
        {"p": "M", "start": 0.10, "end": 0.18}, {"p": "AE1", "start": 0.18, "end": 0.60}, {"p": "P", "start": 0.60, "end": 0.70}]}]}


def test_track_shape_and_closures():
    fr = vt.build(_align(), 0.0, 25, 25)
    assert len(fr) == 25
    vis = [f["vis"] for f in fr]
    assert "PP" in vis                      # M/B/P closures are kept
    assert vis[0] == "sil" and vis[-1] in ("sil", "PP")
    assert all(0 <= f["open"] <= 1 for f in fr)


def test_min_hold():
    fr = vt.build(_align(), 0.0, 25, 25, min_hold=3)
    runs, cur = [], None
    for f in fr:
        if f["vis"] != cur:
            runs.append(0)
            cur = f["vis"]
        runs[-1] += 1
    # every run except the first/last is at least min_hold frames (closures are padded up to it)
    assert all(r >= 3 for r in runs[1:-1])
