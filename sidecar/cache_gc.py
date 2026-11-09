"""Remove model versions Phonon no longer runs from the Hugging Face cache.

Each release pins one revision of each model repo (the correction model, the speech
model). The cache keeps every revision it ever downloaded, so each model update left
the old ~2 GB behind. Once the pinned revision is on disk and complete, this deletes the
repo's other snapshots and the blobs only they linked to (a file shared across
revisions, such as an unchanged drafter, stays). A blob no snapshot links to is left
alone: it may belong to a hand-edited snapshot or a download in progress. Only the one
repo is touched; other models in the cache and anything outside the cache are left
alone.
"""

from __future__ import annotations

import os
import shutil
import sys


def repo_cache_dir(repo_id: str, cache_root: str | None = None) -> str:
    if cache_root is None:
        from huggingface_hub import constants

        cache_root = constants.HF_HUB_CACHE
    return os.path.join(cache_root, "models--" + repo_id.replace("/", "--"))


def _linked_blobs(snapshot: str) -> set[str]:
    linked = set()
    for directory, _, files in os.walk(snapshot):
        for name in files:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                linked.add(os.path.basename(os.path.realpath(path)))
    return linked


def prune_other_revisions(
    repo_id: str, keep_revision: str, cache_root: str | None = None
) -> tuple[int, int]:
    """Delete every snapshot of `repo_id` except `keep_revision` and the blobs only they
    used. Returns (snapshots removed, bytes freed). Does nothing unless `keep_revision`
    is a full commit hash whose snapshot is present."""
    if len(keep_revision) != 40 or os.path.sep in keep_revision:
        return 0, 0
    repo = repo_cache_dir(repo_id, cache_root)
    snapshots = os.path.join(repo, "snapshots")
    if not os.path.isdir(os.path.join(snapshots, keep_revision)):
        return 0, 0
    removed = [name for name in os.listdir(snapshots) if name != keep_revision]
    # Only blobs a removed snapshot linked to are candidates. A blob nothing links to
    # was put there or orphaned by someone else (a hand-edited snapshot, a download in
    # progress) and is not ours to judge.
    candidates: set[str] = set()
    for name in removed:
        candidates |= _linked_blobs(os.path.join(snapshots, name))
    for name in removed:
        path = os.path.join(snapshots, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    refs = os.path.join(repo, "refs")
    if os.path.isdir(refs):
        for name in os.listdir(refs):
            path = os.path.join(refs, name)
            try:
                with open(path) as handle:
                    if handle.read().strip() in removed:
                        os.remove(path)
            except OSError:
                pass
    keep = _linked_blobs(os.path.join(snapshots, keep_revision))
    freed = 0
    blobs = os.path.join(repo, "blobs")
    if os.path.isdir(blobs):
        for name in sorted(candidates - keep):
            if os.sep in name or name.endswith(".incomplete"):
                continue
            path = os.path.join(blobs, name)
            if not os.path.isfile(path):
                continue
            try:
                freed += os.path.getsize(path)
                os.remove(path)
            except OSError:
                pass
    return len(removed), freed


def prune_in_background(repo_id: str, keep_revision: str, label: str) -> None:
    """Runs the prune on a daemon thread so startup never waits on it; any failure is
    logged and ignored."""
    import threading

    def run() -> None:
        try:
            count, freed = prune_other_revisions(repo_id, keep_revision)
            if count or freed:
                print(
                    f"{label}: removed {count} old version(s) of {repo_id}, "
                    f"freed {freed / 1e9:.1f} GB",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as error:  # noqa: BLE001 - cleanup must never break a sidecar
            print(f"{label}: cache cleanup skipped: {error}", file=sys.stderr, flush=True)

    threading.Thread(target=run, daemon=True).start()
