"""cache_gc removes only revisions Phonon's manifest says it fetched: never the pin, never a
revision someone else downloaded, and never a blob a remaining snapshot links; one repo
only."""

import json
import os

from sidecar.cache_gc import (
    prune_phonon_revisions,
    prune_revisions,
    record_fetched,
)

OLD = "a" * 40
NEW = "b" * 40
FOREIGN = "c" * 40
REPO = "Org/model"


def _repo(root, files_by_revision, blobs):
    repo = root / "models--Org--model"
    (repo / "blobs").mkdir(parents=True)
    for name, data in blobs.items():
        (repo / "blobs" / name).write_bytes(data)
    for revision, files in files_by_revision.items():
        snap = repo / "snapshots" / revision
        snap.mkdir(parents=True)
        for filename, blob in files.items():
            os.symlink(f"../../blobs/{blob}", snap / filename)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(OLD)
    return repo


def _cache(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    return cache


def _manifest(tmp_path):
    return str(tmp_path / "Phonon" / "model-cache.json")


def _listed(manifest):
    with open(manifest) as handle:
        return sorted((e["repo"], e["revision"]) for e in json.load(handle)["fetched"])


def test_prunes_a_revision_phonon_fetched_and_its_own_blobs(tmp_path):
    cache = _cache(tmp_path)
    manifest = _manifest(tmp_path)
    repo = _repo(
        cache,
        {
            OLD: {"target.gguf": "old_target", "drafter.gguf": "drafter"},
            NEW: {"target.gguf": "new_target", "drafter.gguf": "drafter"},
        },
        {"old_target": b"x" * 100, "new_target": b"y" * 50, "drafter": b"d" * 10,
         "partial.incomplete": b"p"},
    )
    other = cache / "models--Org--other" / "blobs"
    other.mkdir(parents=True)
    (other / "keepme").write_bytes(b"k")
    record_fetched(REPO, OLD, str(cache), manifest)  # an earlier release resolved OLD

    removed, freed = prune_phonon_revisions(REPO, NEW, str(cache), manifest)

    assert (removed, freed) == ([OLD], 100)
    assert sorted(os.listdir(repo / "snapshots")) == [NEW]
    assert sorted(os.listdir(repo / "blobs")) == ["drafter", "new_target", "partial.incomplete"]
    assert (repo / "snapshots" / NEW / "drafter.gguf").read_bytes() == b"d" * 10
    assert not (repo / "refs" / "main").exists(), "a ref to the removed revision goes"
    assert (other / "keepme").exists(), "other repos are untouched"
    assert _listed(manifest) == [(REPO, NEW)], "the removed revision leaves the manifest"


def test_never_touches_a_revision_not_in_the_manifest(tmp_path):
    """Another tool's revision, or one 1.0.1/1.0.2 fetched before the manifest existed."""
    cache = _cache(tmp_path)
    manifest = _manifest(tmp_path)
    repo = _repo(
        cache,
        {OLD: {"target.gguf": "old_target"}, NEW: {"target.gguf": "new_target"}},
        {"old_target": b"x" * 100, "new_target": b"y"},
    )
    assert prune_phonon_revisions(REPO, NEW, str(cache), manifest) == ([], 0)
    assert sorted(os.listdir(repo / "snapshots")) == [OLD, NEW]
    assert sorted(os.listdir(repo / "blobs")) == ["new_target", "old_target"]
    assert (repo / "refs" / "main").read_text() == OLD
    assert _listed(manifest) == [(REPO, NEW)], "the pin is recorded for next time"


def test_keeps_blobs_a_foreign_snapshot_still_links(tmp_path):
    cache = _cache(tmp_path)
    manifest = _manifest(tmp_path)
    repo = _repo(
        cache,
        {
            OLD: {"target.gguf": "shared_target", "extra.bin": "old_only"},
            FOREIGN: {"target.gguf": "shared_target"},
            NEW: {"target.gguf": "new_target"},
        },
        {"shared_target": b"s" * 70, "old_only": b"o" * 30, "new_target": b"n"},
    )
    record_fetched(REPO, OLD, str(cache), manifest)

    removed, freed = prune_phonon_revisions(REPO, NEW, str(cache), manifest)

    assert (removed, freed) == ([OLD], 30)
    assert sorted(os.listdir(repo / "snapshots")) == [NEW, FOREIGN]
    assert sorted(os.listdir(repo / "blobs")) == ["new_target", "shared_target"]


def test_the_manifest_is_per_repo_and_per_cache(tmp_path):
    cache = _cache(tmp_path)
    manifest = _manifest(tmp_path)
    repo = _repo(
        cache,
        {OLD: {"target.gguf": "old_target"}, NEW: {"target.gguf": "new_target"}},
        {"old_target": b"x", "new_target": b"y"},
    )
    record_fetched("Org/other", OLD, str(cache), manifest)  # same hash, another repo
    record_fetched(REPO, OLD, str(tmp_path / "another-hub"), manifest)  # another cache
    assert prune_phonon_revisions(REPO, NEW, str(cache), manifest) == ([], 0)
    assert sorted(os.listdir(repo / "snapshots")) == [OLD, NEW]


def test_a_revision_already_gone_leaves_the_manifest(tmp_path):
    cache = _cache(tmp_path)
    manifest = _manifest(tmp_path)
    _repo(cache, {NEW: {"target.gguf": "new_target"}}, {"new_target": b"y"})
    record_fetched(REPO, OLD, str(cache), manifest)
    assert prune_phonon_revisions(REPO, NEW, str(cache), manifest) == ([], 0)
    assert _listed(manifest) == [(REPO, NEW)]


def test_recording_is_idempotent_and_only_takes_commit_hashes(tmp_path):
    cache = _cache(tmp_path)
    manifest = _manifest(tmp_path)
    record_fetched(REPO, NEW, str(cache), manifest)
    record_fetched(REPO, NEW, str(cache), manifest)
    record_fetched(REPO, "main", str(cache), manifest)
    assert _listed(manifest) == [(REPO, NEW)]


def test_an_unreadable_manifest_removes_nothing(tmp_path):
    cache = _cache(tmp_path)
    manifest = _manifest(tmp_path)
    repo = _repo(
        cache,
        {OLD: {"target.gguf": "old_target"}, NEW: {"target.gguf": "new_target"}},
        {"old_target": b"x", "new_target": b"y"},
    )
    os.makedirs(os.path.dirname(manifest))
    with open(manifest, "w") as handle:
        handle.write("{not json")
    assert prune_phonon_revisions(REPO, NEW, str(cache), manifest) == ([], 0)
    assert sorted(os.listdir(repo / "snapshots")) == [OLD, NEW]


def test_leaves_blobs_no_removed_snapshot_linked(tmp_path):
    """A snapshot whose weights link was repointed outside the cache orphans the stock
    blob; with no old revision to remove, nothing is deleted."""
    repo = _repo(tmp_path, {NEW: {"config.json": "config"}},
                 {"config": b"c", "stock_weights": b"w" * 100})
    os.symlink(str(tmp_path / "elsewhere.safetensors"),
               repo / "snapshots" / NEW / "model.safetensors")
    assert prune_revisions(REPO, NEW, {OLD}, str(tmp_path)) == ([], 0)
    assert sorted(os.listdir(repo / "blobs")) == ["config", "stock_weights"]


def test_does_nothing_without_the_pinned_snapshot(tmp_path):
    repo = _repo(tmp_path, {OLD: {"target.gguf": "old_target"}}, {"old_target": b"x"})
    assert prune_revisions(REPO, NEW, {OLD}, str(tmp_path)) == ([], 0)
    assert os.listdir(repo / "snapshots") == [OLD]
    assert prune_revisions(REPO, "main", {OLD}, str(tmp_path)) == ([], 0)
    assert prune_revisions("Org/missing", NEW, {OLD}, str(tmp_path)) == ([], 0)
    assert prune_revisions(REPO, NEW, {NEW}, str(tmp_path)) == ([], 0)
