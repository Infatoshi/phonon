import json
import re
from pathlib import Path

DICT_WORDS = Path("/usr/share/dict/words")


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


def load_dict_words(path: Path = DICT_WORDS) -> set[str]:
    with open(path) as f:
        return {w.strip().lower() for w in f if w.strip()}
