"""Each sidecar's Python packages come from a hashed lock (sidecar/<script>.lock, `uv lock
--script`), never resolved at run time: the lock must match the script's inline dependencies and
the engine's display pin, and pin a sha256 for every file it can install."""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SIDECARS = [
    ("asr_server.py", "crates/phonon-asr/src/lib.rs", "ASR_RUNTIME_REQUIREMENT"),
    ("polish_server.py", "crates/phonon-llm/src/paths.rs", "POLISH_RUNTIME_REQUIREMENT"),
]


def inline_metadata(script: Path) -> dict:
    """The PEP 723 `script` block of a file."""
    match = re.search(
        r"^# /// script$\s(?P<body>(^#(| .*)$\s)+)^# ///$", script.read_text(), re.MULTILINE
    )
    assert match, f"{script.name}: no inline script metadata"
    body = "".join(line[2:] + "\n" for line in match.group("body").splitlines())
    return tomllib.loads(body)


def rust_const(path: str, name: str) -> str:
    match = re.search(rf'pub const {name}: &str = "([^"]+)";', (ROOT / path).read_text())
    assert match, f"{name} not in {path}"
    return match.group(1)


@pytest.mark.parametrize(("script", "rust_file", "const"), SIDECARS)
def test_lock_matches_the_script_and_the_engine_pin(script, rust_file, const):
    metadata = inline_metadata(ROOT / "sidecar" / script)
    requirement = rust_const(rust_file, const)
    assert metadata["dependencies"] == [requirement]
    assert metadata["requires-python"] == "==3.12.*"
    assert rust_const("crates/phonon-asr/src/lib.rs", "PYTHON_REQUIREMENT") == "3.12"

    lock = tomllib.loads((ROOT / "sidecar" / f"{script}.lock").read_text())
    assert lock["requires-python"] == metadata["requires-python"]
    name, version = requirement.split("==")
    assert lock["manifest"]["requirements"] == [{"name": name, "specifier": f"=={version}"}]
    versions = {package["name"]: package["version"] for package in lock["package"]}
    assert versions[name] == version


@pytest.mark.parametrize("script", [s for s, _, _ in SIDECARS])
def test_every_locked_file_is_from_pypi_with_a_sha256(script):
    lock = tomllib.loads((ROOT / "sidecar" / f"{script}.lock").read_text())
    for package in lock["package"]:
        assert package["source"] == {"registry": "https://pypi.org/simple"}, package["name"]
        files = list(package.get("wheels", []))
        if "sdist" in package:
            files.append(package["sdist"])
        assert files, package["name"]
        for item in files:
            assert item["url"].startswith("https://files.pythonhosted.org/"), item["url"]
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", item["hash"]), item["url"]
            assert item["size"] > 0, item["url"]


def test_the_app_bundle_ships_both_locks():
    package = (ROOT / "scripts/package-bar.sh").read_text()
    assert 'cp "$project_dir/sidecar/$sidecar.py.lock"' in package
    assert re.search(r"for sidecar in asr_server polish_server; do", package)
