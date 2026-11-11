"""Hash check of the models the sidecars download, against model_pins.json.

A hub model loads only at a revision listed in model_pins.json, and only once every file of
its snapshot matches the size and sha256 listed there. Hashing ~2.5 GB on every launch would
cost seconds, so a verified snapshot is recorded in a small marker in Phonon's data folder
(revision, and per file the size and sha256 it was checked against); a later launch hashes
only a file whose marker entry is missing or whose size changed.

A snapshot file that is a symlink pointing outside its repository's cache folder was put
there by hand (the owner's own fine-tuned speech model) and is reported, not checked; a file
the hub downloaded always resolves into the cache's blobs. A file that fails the check is
removed with its blob and the snapshot is fetched once more; failing again is an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable

PINS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pins.json")
MARKER_DIR_NAME = "model-verified"


def load_pins(path: str = PINS_FILE) -> dict:
    with open(path) as handle:
        return json.load(handle)["models"]


def pinned_revision(repo_id: str, pins: dict | None = None) -> str | None:
    entry = (load_pins() if pins is None else pins).get(repo_id)
    return entry["revision"] if entry else None


def is_commit(revision: str | None) -> bool:
    return (
        isinstance(revision, str)
        and len(revision) == 40
        and all(c in "0123456789abcdef" for c in revision)
    )


def expected_files(repo_id: str, revision: str | None, pins: dict | None = None) -> dict:
    """{file: {"size", "sha256"}} for a pinned hub download; ValueError for anything else, so
    no code path fetches a branch or a revision whose files nobody has hashed."""
    if not is_commit(revision):
        raise ValueError(
            f"{repo_id}: revision {revision!r} is not a 40-hex commit; "
            "pass a pinned revision or a local model folder"
        )
    entry = (load_pins() if pins is None else pins).get(repo_id)
    if not entry or entry["revision"] != revision:
        raise ValueError(
            f"{repo_id}@{revision} has no expected hashes in model_pins.json; "
            "use a local model folder instead"
        )
    return entry["files"]


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_marker_dir() -> str | None:
    try:
        from cache_gc import phonon_data_dir
    except ImportError:
        from sidecar.cache_gc import phonon_data_dir
    data = phonon_data_dir()
    return os.path.join(data, MARKER_DIR_NAME) if data else None


def _marker_path(marker_dir: str, repo_id: str) -> str:
    return os.path.join(marker_dir, repo_id.replace("/", "--") + ".json")


def _read_marker(path: str, snapshot: str, revision: str) -> dict:
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("snapshot") != snapshot or data.get("revision") != revision:
        return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def _write_marker(path: str, snapshot: str, revision: str, files: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as handle:
        json.dump({"snapshot": snapshot, "revision": revision, "files": files}, handle, indent=1)
        handle.write("\n")
    os.replace(tmp, path)


def is_local_override(snapshot: str, name: str) -> bool:
    """A symlink in the snapshot that resolves outside this repository's cache folder."""
    path = os.path.join(snapshot, name)
    if not os.path.islink(path):
        return False
    repo_dir = os.path.realpath(os.path.dirname(os.path.dirname(snapshot)))
    return not os.path.realpath(path).startswith(repo_dir + os.sep)


def check_snapshot(
    snapshot: str, revision: str, expected: dict, marker: str | None, log: Callable[[str], None]
) -> list[str]:
    """Names of the files that are missing or do not match `expected`. Files that pass are
    written to the marker; local overrides are logged and skipped."""
    snapshot = os.path.abspath(snapshot)
    known = _read_marker(marker, snapshot, revision) if marker else {}
    verified: dict = {}
    bad = []
    for name, want in sorted(expected.items()):
        path = os.path.join(snapshot, name)
        if is_local_override(snapshot, name):
            log(f"{name} is a local file ({os.path.realpath(path)}), not checked against the pin")
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            bad.append(name)
            continue
        if known.get(name) == want and size == want["size"]:
            verified[name] = want
            continue
        if size == want["size"] and sha256_of(path) == want["sha256"]:
            verified[name] = want
        else:
            bad.append(name)
    if marker:
        try:
            _write_marker(marker, snapshot, revision, verified)
        except OSError as error:
            log(f"could not record the verification: {error}")
    return bad


def _remove(snapshot: str, name: str) -> None:
    """Drop a failing file and, when it is a hub symlink, the blob it points to."""
    path = os.path.join(snapshot, name)
    repo_dir = os.path.realpath(os.path.dirname(os.path.dirname(snapshot)))
    target = os.path.realpath(path)
    try:
        if os.path.islink(path) and target.startswith(os.path.join(repo_dir, "blobs") + os.sep):
            os.remove(target)
        os.remove(path)
    except OSError:
        pass


def verified_snapshot(
    repo_id: str,
    revision: str | None,
    fetch: Callable[[], str],
    *,
    label: str,
    refetch: Callable[[], str] | None = None,
    pins: dict | None = None,
    marker_dir: str | None = "default",
) -> str:
    """Fetch the pinned snapshot of `repo_id` and return its folder once every file matches
    model_pins.json. `refetch` (default `fetch`) must download whatever files are missing.
    Raises ValueError for an unpinned revision or a file that still fails after one fresh
    fetch."""
    expected = expected_files(repo_id, revision, pins)
    if marker_dir == "default":
        marker_dir = default_marker_dir()
    marker = _marker_path(marker_dir, repo_id) if marker_dir else None

    def log(message: str) -> None:
        print(f"{label}: {message}", file=sys.stderr, flush=True)

    snapshot = fetch()
    bad = check_snapshot(snapshot, revision, expected, marker, log)
    if not bad:
        return snapshot
    log(f"{', '.join(bad)} failed verification; downloading again")
    for name in bad:
        _remove(snapshot, name)
    snapshot = (refetch or fetch)()
    bad = check_snapshot(snapshot, revision, expected, marker, log)
    if bad:
        for name in bad:
            _remove(snapshot, name)
        raise ValueError(f"{repo_id}@{revision[:7]}: {', '.join(bad)} failed sha256 verification")
    return snapshot
