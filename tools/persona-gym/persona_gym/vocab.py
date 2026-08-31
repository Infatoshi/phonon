"""Real-vocabulary pass over a persona tree.

Used twice: at build time to feed repo vocabulary into chat lines, and at
grade time as the term-frequency part of the precision proxy.
"""

import re
from collections import Counter
from pathlib import Path

TEXT_EXTS = {
    ".c", ".h", ".cpp", ".hpp", ".cc", ".py", ".rs", ".js", ".ts", ".rb",
    ".go", ".sh", ".pl", ".lua", ".java", ".swift", ".m", ".mm", ".css",
    ".html", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".cfg", ".ini", ".mk", ".cmake", ".ino", ".s", ".asm", "",
}
MAX_READ = 256 * 1024
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,39}")
CAMEL = re.compile(r"[a-z][A-Z]")


def iter_text_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in TEXT_EXTS:
            yield p


def read_text(p: Path) -> str:
    try:
        return p.read_bytes()[:MAX_READ].decode("utf-8", errors="ignore")
    except OSError:
        return ""


def interesting(tok: str, words: set[str]) -> bool:
    """Identifier-shaped and not a plain dictionary word."""
    t = tok.strip("._-")
    if len(t) < 3 or len(t) > 40:
        return False
    low = t.lower()
    is_word = low in words or (low.endswith("s") and low[:-1] in words)
    # Plain words still count when written as identifiers.
    if is_word and not (CAMEL.search(t) or "_" in tok or "-" in tok or "." in tok):
        return False
    if any(ch.isdigit() for ch in t):
        return True
    return bool(CAMEL.search(t) or t.isupper() or "_" in tok or "-" in tok
                or "." in tok or low not in words)


def repo_vocab(root: Path, words: set[str], min_count: int = 2) -> Counter:
    """Counter of identifier-ish tokens over file names and file text."""
    counts: Counter = Counter()
    for p in iter_text_files(root):
        counts.update(t for t in TOKEN.findall(p.stem) if interesting(t, words))
        seen_lines = set()
        for line in read_text(p).splitlines():
            line = line.strip()
            if not line or line in seen_lines:
                continue
            seen_lines.add(line)
            counts.update(t for t in TOKEN.findall(line) if interesting(t, words))
    return Counter({t: c for t, c in counts.items() if c >= min_count})
