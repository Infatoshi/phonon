import json
import os
import re
from pathlib import Path

HOME = Path.home()
PHONON_SUPPORT = HOME / "Library" / "Application Support" / "Phonon"


def out_dir() -> Path:
    return Path(os.environ.get("PHONON_MINER_OUT", "out")).expanduser().resolve()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    tmp.replace(path)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


_NORM = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    """Lowercase, strip everything that is not a letter or digit."""
    return _NORM.sub("", s.lower())


def norm_words(s: str) -> list[str]:
    return [w for w in _NORM.sub(" ", s.lower()).split() if w]
