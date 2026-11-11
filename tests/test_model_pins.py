"""The sidecars load a hub model only at a pinned revision and only once its files match the
sha256 in model_pins.json; a marker spares later launches the hashing; a hand-placed local
file (a symlink out of the cache) is left alone; a bad file is fetched once more, then fails."""

import hashlib
import json
import os

import pytest

from sidecar import model_pins
from sidecar.model_pins import expected_files, load_pins, verified_snapshot
from sidecar.polish_server import resolve_model_dir

REPO = "Org/model"
REV = "b" * 40
FILES = {"config.json": b'{"a": 1}', "model.safetensors": b"weights" * 100}


def _pins(files=FILES):
    return {
        REPO: {
            "revision": REV,
            "files": {
                name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()
            },
        }
    }


def _snapshot(root, files=FILES):
    """A Hugging Face cache layout: snapshot symlinks into blobs named by their hash."""
    repo = root / "models--Org--model"
    snap = repo / "snapshots" / REV
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    for name, data in files.items():
        blob = hashlib.sha256(data).hexdigest()
        (repo / "blobs" / blob).write_bytes(data)
        os.symlink(f"../../blobs/{blob}", snap / name)
    return snap


def _xet_blob(cache, data):
    """Where huggingface_hub 2.x keeps a Xet download: <cache>/blobs/<xx>/<hash>. The name is
    Xet's own hash, not the sha256, so it is derived differently here on purpose."""
    key = hashlib.sha1(data).hexdigest()
    shared = cache / "blobs" / key[:2] / key
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_bytes(data)
    return shared


def _xet_snapshot(cache, files=FILES, large=("model.safetensors",)):
    """The two-level layout of a Xet download: snapshot -> ../../blobs/<sha256> (a link in the
    repository's blobs) -> ../../blobs/<xx>/<hash> (the cache-wide content)."""
    repo = cache / "models--Org--model"
    snap = repo / "snapshots" / REV
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    for name, data in files.items():
        blob = hashlib.sha256(data).hexdigest()
        if name in large:
            shared = _xet_blob(cache, data)
            os.symlink(os.path.relpath(shared, repo / "blobs"), repo / "blobs" / blob)
        else:
            (repo / "blobs" / blob).write_bytes(data)
        os.symlink(f"../../blobs/{blob}", snap / name)
    return snap


def _counting_hash(monkeypatch):
    hashed = []
    real = model_pins.sha256_of

    def counting(path):
        hashed.append(os.path.basename(path))
        return real(path)

    monkeypatch.setattr(model_pins, "sha256_of", counting)
    return hashed


def test_the_pin_file_lists_both_shipped_models_with_hashes():
    pins = load_pins()
    for repo in ("mlx-community/parakeet-tdt-0.6b-v2", "Infatoshi/phonon-refiner-1.7b"):
        files = expected_files(repo, pins[repo]["revision"], pins)
        assert files
        for entry in files.values():
            assert len(entry["sha256"]) == 64 and entry["size"] > 0


def test_branches_and_unhashed_revisions_are_refused():
    for revision in (None, "main", "B" * 40, "c" * 40):
        with pytest.raises(ValueError):
            expected_files(REPO, revision, _pins())
    with pytest.raises(ValueError):
        expected_files("Other/model", REV, _pins())


def test_verifies_once_then_trusts_the_marker_until_a_size_changes(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path / "hub")
    marker_dir = tmp_path / "marker"
    hashed = _counting_hash(monkeypatch)

    def fetch():
        return str(snap)

    kwargs = {"label": "test", "pins": _pins(), "marker_dir": str(marker_dir)}
    assert verified_snapshot(REPO, REV, fetch, **kwargs) == str(snap)
    assert sorted(hashed) == sorted(FILES)
    marker = json.loads((marker_dir / "Org--model.json").read_text())
    assert marker["revision"] == REV and set(marker["files"]) == set(FILES)

    hashed.clear()
    assert verified_snapshot(REPO, REV, fetch, **kwargs) == str(snap)
    assert hashed == []

    # a file whose size changed no longer counts as verified: it is fetched again, then fails
    blob = os.path.realpath(snap / "config.json")
    with open(blob, "ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="config.json"):
        verified_snapshot(REPO, REV, fetch, **kwargs)
    marker = json.loads((marker_dir / "Org--model.json").read_text())
    assert set(marker["files"]) == {"model.safetensors"}
    assert not os.path.lexists(snap / "config.json")


def test_a_corrupt_hub_file_is_removed_and_fetched_again(tmp_path):
    snap = _snapshot(tmp_path / "hub")
    good = FILES["model.safetensors"]
    blob = os.path.realpath(snap / "model.safetensors")
    with open(blob, "wb") as handle:
        handle.write(b"X" * len(good))  # same size, wrong bytes
    fetches = []

    def fetch():
        fetches.append(1)
        if len(fetches) == 2:
            # what snapshot_download does for a missing file
            assert not os.path.lexists(snap / "model.safetensors")
            assert not os.path.exists(blob)
            with open(blob, "wb") as handle:
                handle.write(good)
            os.symlink(os.path.relpath(blob, snap), snap / "model.safetensors")
        return str(snap)

    out = verified_snapshot(REPO, REV, fetch, label="test", pins=_pins(), marker_dir=None)
    assert out == str(snap) and len(fetches) == 2


def test_a_file_that_stays_wrong_fails_after_one_more_fetch(tmp_path):
    snap = _snapshot(tmp_path / "hub", {**FILES, "config.json": b'{"a": 2}'})
    fetches = []

    def fetch():
        fetches.append(1)
        return str(snap)

    with pytest.raises(ValueError, match="config.json failed sha256"):
        verified_snapshot(REPO, REV, fetch, label="test", pins=_pins(), marker_dir=None)
    assert len(fetches) == 2


def test_a_symlink_to_a_local_model_is_a_deliberate_override(tmp_path, monkeypatch):
    # The owner's Mac: the snapshot's model.safetensors points at a fine-tuned model outside
    # the cache. It is reported and loaded as it is; the other files are still checked.
    snap = _snapshot(tmp_path / "hub")
    local = tmp_path / "Phonon" / "models" / "asr-local" / "model.safetensors"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"fine-tuned weights, a different size")
    os.remove(snap / "model.safetensors")
    os.symlink(local, snap / "model.safetensors")
    hashed = _counting_hash(monkeypatch)

    out = verified_snapshot(REPO, REV, lambda: str(snap), label="test", pins=_pins(), marker_dir=None)
    assert out == str(snap)
    assert hashed == ["config.json"]
    assert local.read_bytes() == b"fine-tuned weights, a different size"


def test_polish_refuses_an_unpinned_hub_model_before_any_download():
    with pytest.raises(ValueError):
        resolve_model_dir("mlx-community/gemma-4-e2b-it-4bit", "main")
    with pytest.raises(ValueError):
        resolve_model_dir("Some/model", "a" * 40)


# huggingface_hub 2.x (what uv resolves for the sidecars today) stores Xet downloads in the
# cache-wide blobs folder. Those files used to resolve "outside the repository" and were skipped
# as local overrides, so the speech model's weights were never hashed.


def test_a_xet_download_is_hashed_like_any_hub_file(tmp_path, monkeypatch):
    snap = _xet_snapshot(tmp_path / "hub")
    marker_dir = tmp_path / "marker"
    hashed = _counting_hash(monkeypatch)

    out = verified_snapshot(
        REPO, REV, lambda: str(snap), label="test", pins=_pins(), marker_dir=str(marker_dir)
    )
    assert out == str(snap)
    assert sorted(hashed) == sorted(FILES)
    assert not model_pins.is_local_override(str(snap), "model.safetensors")
    marker = json.loads((marker_dir / "Org--model.json").read_text())
    assert set(marker["files"]) == set(FILES)


def test_a_corrupt_xet_download_is_removed_and_fetched_again_once(tmp_path):
    cache = tmp_path / "hub"
    snap = _xet_snapshot(cache)
    good = FILES["model.safetensors"]
    repo_blob = cache / "models--Org--model" / "blobs" / hashlib.sha256(good).hexdigest()
    shared = os.path.realpath(snap / "model.safetensors")
    assert shared.startswith(os.path.realpath(cache / "blobs") + os.sep)
    os.chmod(shared, 0o644)
    with open(shared, "wb") as handle:
        handle.write(b"X" * len(good))  # same size, wrong bytes
    fetches = []

    def fetch():
        fetches.append(1)
        if len(fetches) == 2:
            # the whole chain went, so the hub cannot relink the bad content
            assert not os.path.lexists(snap / "model.safetensors")
            assert not os.path.lexists(repo_blob)
            assert not os.path.exists(shared)
            fresh = _xet_blob(cache, good)
            os.symlink(os.path.relpath(fresh, repo_blob.parent), repo_blob)
            os.symlink(os.path.relpath(repo_blob, snap), snap / "model.safetensors")
        return str(snap)

    out = verified_snapshot(REPO, REV, fetch, label="test", pins=_pins(), marker_dir=None)
    assert out == str(snap) and len(fetches) == 2
    assert (snap / "model.safetensors").read_bytes() == good


def _override(snap, target, data):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    os.remove(snap / "model.safetensors")
    os.symlink(target, snap / "model.safetensors")


@pytest.mark.parametrize("where", ["outside the cache", "another repository's blobs"])
def test_an_override_in_a_xet_cache_is_skipped_and_never_deleted(tmp_path, monkeypatch, where):
    cache = tmp_path / "hub"
    snap = _xet_snapshot(cache)
    if where == "outside the cache":
        target = tmp_path / "Phonon" / "models" / "asr-local" / "model.safetensors"
    else:
        target = cache / "models--Other--model" / "blobs" / ("e" * 64)
    data = b"fine-tuned weights, a different size"
    _override(snap, target, data)
    assert model_pins.is_local_override(str(snap), "model.safetensors")
    hashed = _counting_hash(monkeypatch)

    out = verified_snapshot(REPO, REV, lambda: str(snap), label="test", pins=_pins(), marker_dir=None)
    assert out == str(snap)
    assert hashed == ["config.json"]
    assert target.read_bytes() == data

    # even when asked to remove it, only the snapshot's link goes
    model_pins._remove(str(snap), "model.safetensors")
    assert not os.path.lexists(snap / "model.safetensors")
    assert target.read_bytes() == data
