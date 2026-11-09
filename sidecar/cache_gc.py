"""Remove model versions Phonon downloaded and no longer runs from the Hugging Face cache.

Each release pins one revision of each model repo (the correction model, the speech
model). The cache keeps every revision it ever downloaded, so each model update left
the old ~2 GB behind. The cache is shared: other tools on the same machine may keep
other revisions of the same repos on purpose. So each time a sidecar resolves a pinned
revision it is written to a small manifest in Phonon's own data folder
(`MANIFEST_NAME`), and cleanup only ever removes revisions listed there: once the pinned
revision is on disk, the repo's other snapshots that are in the manifest go, with the
blobs only they linked to, and leave the manifest. A revision not in the manifest is never
touched (that includes ones 1.0.1 and 1.0.2 fetched before the manifest existed). A blob a
remaining snapshot links to stays (a file shared across revisions, such as an unchanged
drafter), and so does a blob no snapshot links to: it may belong to a hand-edited
snapshot or a download in progress. Only the one repo is touched; other models in the
cache and anything outside the cache are left alone.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

MANIFEST_NAME = "model-cache.json"


def phonon_data_dir() -> str | None:
    """Phonon's data folder, the one the engine and the app use
    (phonon-core `app_support_dir`). The sidecars run on macOS only."""
    if sys.platform != "darwin":
        return None
    home = os.environ.get("HOME")
    if not home:
        return None
    return os.path.join(home, "Library", "Application Support", "Phonon")


def default_manifest_path() -> str | None:
    data = phonon_data_dir()
    return os.path.join(data, MANIFEST_NAME) if data else None


def repo_cache_dir(repo_id: str, cache_root: str | None = None) -> str:
    if cache_root is None:
        from huggingface_hub import constants

        cache_root = constants.HF_HUB_CACHE
    return os.path.join(cache_root, "models--" + repo_id.replace("/", "--"))


def _is_commit(revision: str) -> bool:
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


# The manifest: {"version": 1, "fetched": [{"cache": <hub cache dir>, "repo": "Org/model",
# "revision": <40-hex commit>}, ...]}. Both sidecars write it, so every change is a locked
# read-modify-write and an atomic replace.


class _Manifest:
    def __init__(self, path: str):
        self.path = path
        self._lock = None

    def __enter__(self) -> list[dict]:
        import fcntl

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = open(self.path + ".lock", "a")  # held until __exit__
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        try:
            with open(self.path) as handle:
                data = json.load(handle)
            entries = data.get("fetched", []) if isinstance(data, dict) else []
        except (FileNotFoundError, ValueError):
            # Unreadable starts over empty: what it listed is then never removed, the safe side.
            entries = []
        self.entries = [
            e
            for e in entries
            if isinstance(e, dict)
            and isinstance(e.get("cache"), str)
            and isinstance(e.get("repo"), str)
            and isinstance(e.get("revision"), str)
        ]
        return self.entries

    def save(self) -> None:
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w") as handle:
            json.dump({"version": 1, "fetched": self.entries}, handle, indent=1)
            handle.write("\n")
        os.replace(tmp, self.path)

    def __exit__(self, *_exc) -> None:
        if self._lock is not None:
            self._lock.close()  # releases the flock


def record_fetched(
    repo_id: str, revision: str, cache_root: str, manifest_path: str
) -> None:
    """Note that Phonon resolved `revision` of `repo_id` into `cache_root`."""
    if not _is_commit(revision):
        return
    entry = {"cache": os.path.abspath(cache_root), "repo": repo_id, "revision": revision}
    manifest = _Manifest(manifest_path)
    with manifest as entries:
        if entry not in entries:
            entries.append(entry)
            manifest.save()


def _linked_blobs(snapshot: str) -> set[str]:
    linked = set()
    for directory, _, files in os.walk(snapshot):
        for name in files:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                linked.add(os.path.basename(os.path.realpath(path)))
    return linked


def prune_revisions(
    repo_id: str, keep_revision: str, remove: set[str], cache_root: str | None = None
) -> tuple[list[str], int]:
    """Delete the snapshots of `repo_id` named in `remove` (never `keep_revision`) and the
    blobs only they used. Returns (revisions removed, bytes freed). Does nothing unless
    `keep_revision` is a full commit hash whose snapshot is present."""
    if not _is_commit(keep_revision):
        return [], 0
    repo = repo_cache_dir(repo_id, cache_root)
    snapshots = os.path.join(repo, "snapshots")
    if not os.path.isdir(os.path.join(snapshots, keep_revision)):
        return [], 0
    present = os.listdir(snapshots)
    removed = sorted(
        name for name in present if name in remove and name != keep_revision and _is_commit(name)
    )
    if not removed:
        return [], 0
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
    # Every snapshot still here keeps its blobs: the pinned one, and any another tool
    # downloaded.
    keep: set[str] = set()
    for name in os.listdir(snapshots):
        keep |= _linked_blobs(os.path.join(snapshots, name))
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
    return removed, freed


def prune_phonon_revisions(
    repo_id: str, keep_revision: str, cache_root: str, manifest_path: str
) -> tuple[list[str], int]:
    """Record the pin, then remove the other revisions of `repo_id` that the manifest says
    Phonon fetched into `cache_root`, and drop them from it. Returns what
    `prune_revisions` does."""
    if not _is_commit(keep_revision):
        return [], 0
    cache = os.path.abspath(cache_root)
    record_fetched(repo_id, keep_revision, cache, manifest_path)
    manifest = _Manifest(manifest_path)
    with manifest as entries:
        ours = [e for e in entries if e["cache"] == cache and e["repo"] == repo_id]
        mine = {e["revision"] for e in ours if e["revision"] != keep_revision}
        removed, freed = prune_revisions(repo_id, keep_revision, mine, cache)
        snapshots = os.path.join(repo_cache_dir(repo_id, cache), "snapshots")
        # Gone now, or gone already (the user cleared the cache): no longer ours to track.
        gone = {r for r in mine if not os.path.exists(os.path.join(snapshots, r))}
        if gone:
            manifest.entries[:] = [
                e
                for e in entries
                if not (e["cache"] == cache and e["repo"] == repo_id and e["revision"] in gone)
            ]
            manifest.save()
    return removed, freed


def prune_in_background(repo_id: str, keep_revision: str, label: str) -> None:
    """Records the pin and runs the prune on a daemon thread so startup never waits on
    it; any failure is logged and ignored."""
    import threading

    def run() -> None:
        try:
            manifest_path = default_manifest_path()
            if manifest_path is None:
                return
            from huggingface_hub import constants

            removed, freed = prune_phonon_revisions(
                repo_id, keep_revision, constants.HF_HUB_CACHE, manifest_path
            )
            if removed or freed:
                print(
                    f"{label}: removed {len(removed)} old version(s) of {repo_id}, "
                    f"freed {freed / 1e9:.1f} GB",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as error:  # noqa: BLE001 - cleanup must never break a sidecar
            print(f"{label}: cache cleanup skipped: {error}", file=sys.stderr, flush=True)

    threading.Thread(target=run, daemon=True).start()
