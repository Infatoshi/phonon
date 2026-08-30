"""Stage 3b: rule-based candidate terms. No model.

Classes: camel, caps, digit, ident (hyphen/underscore/dot), cap (capitalized,
not sentence-initial, count >= 2), lower (lowercase token absent from a fixed
general-English list). Bigrams and trigrams form where a candidate sits next to
a digit, acronym or capitalized token. Paths, URLs, hex, keys and tokens over 40
chars are dropped. The English list is /usr/share/dict/words, never the user's
own text.
"""

import re
import sys
import time
from collections import Counter, defaultdict

from .common import out_dir, read_json, write_json

WORDS_PATH = "/usr/share/dict/words"
MAX_LEN = 40
MIN_COUNT = 2
CAP_RATIO = 0.3  # English words: share of occurrences that are Capitalized non-initial or ALLCAPS

RE_URL = re.compile(r"(?:https?://|www\.)\S+")
RE_EMAIL = re.compile(r"\S+@\S+")
RE_PATH = re.compile(r"(?:~|\.{1,2})?/[\w.\-~+]+(?:/[\w.\-~+*]*)+|[\w.\-]+(?:/[\w.\-]+){2,}")
RE_HEX = re.compile(r"\b(?:0x[0-9a-fA-F]+|[0-9a-f]{16,}|[0-9A-F]{16,})\b")
RE_KEY = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")
RE_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[\-_.'][A-Za-z0-9]+)*(?:\+\+|#)?")
RE_SENT_END = re.compile(r"""[.!?:;]["')\]]?\s*$""")
RE_NUMBER = re.compile(r"[\d.,]+[kKmMgGbB]?")
RE_MULT = re.compile(r"[\d.]+[xX]")  # 2x, 1.5x
RE_DOTTED = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]?")  # e.g, i.e, a.k.a
RE_CAP = re.compile(r"[A-Z][a-z]+")
RE_CAPS = re.compile(r"[A-Z]{2,}")
RE_LOWER = re.compile(r"[a-z]{3,}")
RE_ALPHA = re.compile(r"[A-Za-z]+")
EXT = {"md", "py", "rs", "toml", "json", "txt", "sh", "yaml", "yml", "swift", "js", "ts", "html", "css", "png", "jpg",
       "wav", "lock", "cfg", "ini", "csv", "cu", "h", "cpp", "c", "pdf", "jsonl", "log", "plist", "mp4", "mov", "tsx",
       "jsx", "svg", "aiff", "zip", "tar", "gz", "dmg", "app", "so", "dylib", "pyc", "ipynb", "mjs", "cjs", "sql", "db"}
SUFFIX_RULES = [
    (lambda t: t[:-1], "s"), (lambda t: t[:-2], "es"), (lambda t: t[:-2], "ed"), (lambda t: t[:-1], "ed"),
    (lambda t: t[:-3], "ing"), (lambda t: t[:-3] + "e", "ing"), (lambda t: t[:-3] + "y", "ies"), (lambda t: t[:-2], "ly"),
    (lambda t: t[:-2], "er"), (lambda t: t[:-3], "ers"), (lambda t: t[:-3], "est"), (lambda t: t[:-4], "ing"),
    (lambda t: t[:-3], "ed"), (lambda t: t[:-1], "d"),
]


def load_english():
    words = set()
    with open(WORDS_PATH) as f:
        for w in f:
            words.add(w.strip().lower())
    return words


class English:
    def __init__(self):
        self.words = load_english()
        self.cache = {}

    def __contains__(self, t):
        r = self.cache.get(t)
        if r is None:
            r = self._check(t)
            self.cache[t] = r
        return r

    def _check(self, t):
        if t in self.words:
            return True
        for fn, suf in SUFFIX_RULES:
            if t.endswith(suf):
                s = fn(t)
                if len(s) >= 2 and s in self.words:
                    return True
        return False


def clean_line(line):
    line = RE_URL.sub(" ", line)
    line = RE_EMAIL.sub(" ", line)
    line = RE_PATH.sub(" ", line)
    line = RE_HEX.sub(" ", line)
    line = RE_KEY.sub(" ", line)
    return line


def strip_ext(tok):
    """Return (stem, is_file): 'AGENTS.md' -> ('AGENTS', True)."""
    if "." in tok:
        stem, _, ext = tok.rpartition(".")
        if ext.lower() in EXT and stem:
            return stem, True
    return tok, False


def classify(tok, english):
    """Return the set of candidate classes for a token, ignoring position."""
    cls = set()
    if RE_NUMBER.fullmatch(tok):
        return {"number"}
    if len(tok) > MAX_LEN or len(tok) < 2:
        return cls
    if "'" in tok or RE_MULT.fullmatch(tok) or RE_DOTTED.fullmatch(tok):
        return cls
    has_digit = any(c.isdigit() for c in tok)
    has_alpha = any(c.isalpha() for c in tok)
    if not has_alpha:
        return cls
    if has_digit:
        cls.add("digit")
    if "-" in tok or "_" in tok or "." in tok:
        cls.add("ident")
    if RE_ALPHA.fullmatch(tok):
        ups = sum(c.isupper() for c in tok)
        lows = len(tok) - ups
        if ups >= 2 and lows >= 1:
            cls.add("camel")
        elif RE_CAPS.fullmatch(tok):
            cls.add("caps")
        elif RE_CAP.fullmatch(tok):
            cls.add("cap")
        elif RE_LOWER.fullmatch(tok) and tok not in english:
            cls.add("lower")
    elif re.search(r"[a-z][A-Z]", tok):
        cls.add("camel")
    return cls


def run(sources=None, min_count=MIN_COUNT):
    english = English()
    od = out_dir()
    counts_path = od / "extract" / "counts.json"
    all_sources = [k for k in read_json(counts_path) if k != "until"]
    sources = sources or all_sources
    seeds = read_json(od / "seed" / "seeds.json") if (od / "seed" / "seeds.json").exists() else {}

    per_source = {}
    surface = defaultdict(Counter)  # key -> surface -> count (qualifying occurrences)
    classes = defaultdict(set)
    evidence = defaultdict(list)
    cap_noninit = defaultdict(Counter)  # per source: key -> non-initial capitalized count
    caps_count = defaultdict(Counter)  # per source: key -> ALLCAPS count
    allcount = defaultdict(Counter)  # per source: key -> all case-insensitive alpha occurrences
    t0 = time.time()
    total_lines = 0
    for src in sources:
        cnt = Counter()
        seen_lines = set()
        path = od / "extract" / f"{src}.txt"
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                total_lines += 1
                text = clean_line(line)
                toks = []
                prev_end = 0
                for m in RE_TOKEN.finditer(text):
                    tok, is_file = strip_ext(m.group())
                    gap = text[prev_end:m.start()]
                    initial = prev_end == 0 or bool(RE_SENT_END.search(gap))
                    prev_end = m.end()
                    cls = classify(tok, english)
                    key = tok.lower()
                    if RE_ALPHA.fullmatch(tok):
                        allcount[src][key] += 1
                    if "caps" in cls:
                        caps_count[src][key] += 1
                    if "cap" in cls:
                        if initial:
                            cls = {"cap_initial"}
                        else:
                            cap_noninit[src][key] += 1
                    toks.append((tok, key, cls))
                    if is_file and any(ch.isalpha() for ch in tok):
                        fkey = m.group().lower()
                        cnt[fkey] += 1
                        surface[fkey][m.group()] += 1
                        classes[fkey].add("file")
                line_keys = set()
                for i, (tok, key, cls) in enumerate(toks):
                    if cls - {"number", "cap_initial"}:
                        cnt[key] += 1
                        surface[key][tok] += 1
                        classes[key] |= cls
                        line_keys.add(key)
                    # n-grams: 2 or 3 consecutive special tokens, at least one a candidate.
                    for n in (2, 3):
                        if i + n > len(toks):
                            break
                        grp = toks[i:i + n]
                        if not all(g[2] for g in grp):
                            break
                        if not any(g[2] - {"number", "cap_initial"} for g in grp):
                            continue
                        if grp[0][2] == {"number"}:
                            continue
                        if any("lower" in g[2] and len(g[2]) == 1 for g in grp) and all(
                            g[2] <= {"lower", "number", "cap_initial", "cap"} for g in grp
                        ):
                            continue  # lowercase-only groups are not units
                        if all(g[2] <= {"cap", "cap_initial"} for g in grp) and n == 3:
                            pass
                        gkey = " ".join(g[1] for g in grp)
                        gsurf = " ".join(g[0] for g in grp)
                        cnt[gkey] += 1
                        surface[gkey][gsurf] += 1
                        classes[gkey].add("ngram")
                        line_keys.add(gkey)
                if len(line) <= 160:
                    for key in line_keys:
                        ev = evidence[key]
                        if len(ev) < 2 and line not in ev:
                            ev.append(line)
        per_source[src] = cnt
        print(f"[candidates] {src}: {len(cnt)} raw keys from {len(seen_lines)} unique lines", file=sys.stderr)

    # Capitalized rule: require count >= 2 non-initial and, for English words, a capitalization ratio.
    keys = set()
    for src, cnt in per_source.items():
        keys |= set(cnt)
    out = []
    dropped_cap = 0
    for key in keys:
        cls = classes[key]
        total = sum(c[key] for c in per_source.values())
        if cls <= {"cap", "caps"}:
            noninit = sum(c[key] for c in cap_noninit.values())
            capsn = sum(c[key] for c in caps_count.values())
            if "cap" in cls and "caps" not in cls and noninit < 2:
                dropped_cap += 1
                continue
            if key in english:
                allc = sum(c[key] for c in allcount.values())
                if allc and (noninit + capsn) / allc < CAP_RATIO:
                    dropped_cap += 1
                    continue
            total = noninit + capsn
        if total < min_count:
            continue
        srcs = {s: c[key] for s, c in per_source.items() if c[key]}
        term = surface[key].most_common(1)[0][0]
        out.append({
            "term": term,
            "key": key,
            "count": total,
            "sources": srcs,
            "classes": sorted(cls),
            "evidence": evidence.get(key, [])[:2],
            "seed": False,
        })
    have = {c["key"] for c in out}
    for s, why in seeds.items():
        key = s.lower()
        if key in have:
            for c in out:
                if c["key"] == key:
                    c["seed"] = True
                    c["seed_source"] = why
            continue
        total = sum(c[key] for c in per_source.values())
        srcs = {sn: c[key] for sn, c in per_source.items() if c[key]}
        out.append({"term": s, "key": key, "count": total, "sources": srcs, "classes": ["seed"],
                    "evidence": evidence.get(key, [])[:2], "seed": True, "seed_source": why})
    out.sort(key=lambda c: (-c["count"], -len(c["sources"]), c["key"]))
    by_class = Counter()
    for c in out:
        for k in c["classes"]:
            by_class[k] += 1
    write_json(od / "candidates" / "candidates_raw.json", out)
    print(f"[candidates] {len(out)} candidates (min_count={min_count}, cap dropped {dropped_cap}) "
          f"from {total_lines} lines in {time.time() - t0:.1f}s; classes {dict(by_class)}", file=sys.stderr)
    return out
