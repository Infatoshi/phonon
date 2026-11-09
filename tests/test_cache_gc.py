"""cache_gc keeps the pinned snapshot and the blobs it links, and removes everything
else of that one repo only."""

import os

from sidecar.cache_gc import prune_other_revisions

OLD = "a" * 40
NEW = "b" * 40
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


def test_prunes_old_revision_and_its_own_blobs(tmp_path):
    repo = _repo(
        tmp_path,
        {
            OLD: {"target.gguf": "old_target", "drafter.gguf": "drafter"},
            NEW: {"target.gguf": "new_target", "drafter.gguf": "drafter"},
        },
        {"old_target": b"x" * 100, "new_target": b"y" * 50, "drafter": b"d" * 10,
         "partial.incomplete": b"p"},
    )
    other = tmp_path / "models--Org--other" / "blobs"
    other.mkdir(parents=True)
    (other / "keepme").write_bytes(b"k")

    removed, freed = prune_other_revisions(REPO, NEW, str(tmp_path))

    assert (removed, freed) == (1, 100)
    assert sorted(os.listdir(repo / "snapshots")) == [NEW]
    assert sorted(os.listdir(repo / "blobs")) == ["drafter", "new_target", "partial.incomplete"]
    assert (repo / "snapshots" / NEW / "drafter.gguf").read_bytes() == b"d" * 10
    assert not (repo / "refs" / "main").exists(), "a ref to the removed revision goes"
    assert (other / "keepme").exists(), "other repos are untouched"


def test_leaves_blobs_no_removed_snapshot_linked(tmp_path):
    """A snapshot whose weights link was repointed outside the cache orphans the stock
    blob; with no old revision to remove, nothing is deleted."""
    repo = _repo(tmp_path, {NEW: {"config.json": "config"}},
                 {"config": b"c", "stock_weights": b"w" * 100})
    os.symlink(str(tmp_path / "elsewhere.safetensors"),
               repo / "snapshots" / NEW / "model.safetensors")
    assert prune_other_revisions(REPO, NEW, str(tmp_path)) == (0, 0)
    assert sorted(os.listdir(repo / "blobs")) == ["config", "stock_weights"]


def test_does_nothing_without_the_pinned_snapshot(tmp_path):
    repo = _repo(tmp_path, {OLD: {"target.gguf": "old_target"}}, {"old_target": b"x"})
    assert prune_other_revisions(REPO, NEW, str(tmp_path)) == (0, 0)
    assert os.listdir(repo / "snapshots") == [OLD]
    assert prune_other_revisions(REPO, "main", str(tmp_path)) == (0, 0)
    assert prune_other_revisions("Org/missing", NEW, str(tmp_path)) == (0, 0)
