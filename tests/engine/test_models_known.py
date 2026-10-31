# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""the pinned model table: a pinned snapshot is checked once, a bad file refuses, an export carries its hash."""

import hashlib

import pytest

from lsdub_engine import _models as M


def snapshot(tmp_path, repo, rev, files):
    d = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    d.mkdir(parents=True)
    for k, v in files.items():
        (d / k).write_bytes(v)
    return d


@pytest.fixture
def known(monkeypatch):
    sha = hashlib.sha256(b"ok").hexdigest()
    monkeypatch.setitem(M.KNOWN, "a/b", {"revision": "r1", "files": {"m.bin": sha}})


def test_pinned_snapshot_wins_and_is_stamped(tmp_path, known):
    snapshot(tmp_path, "a/b", "r2", {"m.bin": b"other"})
    d = snapshot(tmp_path, "a/b", "r1", {"m.bin": b"ok"})
    assert M.hf_snapshot(str(tmp_path), "a/b", ("m.bin",)) == str(d) and (d / ".verified").exists()


def test_a_stamped_snapshot_is_not_read_again(tmp_path, known, monkeypatch):
    snapshot(tmp_path, "a/b", "r1", {"m.bin": b"ok"})
    M.hf_snapshot(str(tmp_path), "a/b", ("m.bin",))
    monkeypatch.setattr(M, "check_file", lambda *a: pytest.fail("read twice"))
    M.hf_snapshot(str(tmp_path), "a/b", ("m.bin",))


def test_a_bad_pinned_file_refuses(tmp_path, known):
    snapshot(tmp_path, "a/b", "r1", {"m.bin": b"bad"})
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        M.hf_snapshot(str(tmp_path), "a/b", ("m.bin",))


def test_another_revision_still_loads_unchecked(tmp_path, known):
    d = snapshot(tmp_path, "a/b", "r2", {"m.bin": b"other"})
    assert M.hf_snapshot(str(tmp_path), "a/b", ("m.bin",)) == str(d) and not (d / ".verified").exists()


def test_export_sidecar(tmp_path):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"x")
    assert M.check_export(str(p)) == str(p)  # an older export: no sidecar
    M.stamp_export(str(p))
    assert M.check_export(str(p)) == str(p)
    p.write_bytes(b"y")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        M.check_export(str(p))


def test_every_pin_is_a_full_revision_and_sha():
    for k in M.KNOWN.values():
        assert len(k["revision"]) == 40 and k["files"] and all(len(s) == 64 for s in k["files"].values())
    assert M.CAST_ONNX["source"] in M.KNOWN
