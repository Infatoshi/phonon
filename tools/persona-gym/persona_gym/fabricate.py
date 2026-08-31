"""Fake vocabulary from word-part combinators.

Every term must not be a real word (checked against /usr/share/dict/words)
and must be unique across all personas (global `taken` set of norms).
"""

import random

from .common import norm

CODE_HEADS = [
    "thorn", "brack", "fen", "mur", "crag", "dov", "brim", "sill", "tarn",
    "vex", "wren", "kel", "dray", "holt", "marl", "nock", "pell", "quist",
    "ryn", "skarn", "tull", "varn", "weld", "zell", "orm", "pyre", "nab",
    "garn", "bex", "crov",
]
CODE_TAILS = [
    "mill", "forge", "gate", "hollow", "spur", "weir", "march", "fold",
    "reach", "vane", "brook", "cairn", "loft", "shard", "wick", "moor",
    "ridge", "dun", "mere", "stead", "holm", "cote", "frith", "spar",
]
MACH_HEADS = [
    "bel", "cin", "dor", "fir", "gim", "hal", "jor", "kan", "lor", "mag",
    "nor", "pren", "quor", "rin", "sul", "tor", "ves", "wun", "yor", "zan",
]
MACH_TAILS = [
    "vik", "dun", "mar", "tos", "rek", "lin", "gar", "nod", "bax", "tur",
    "vim", "holt", "dek", "rum", "zor",
]
PERSON_HEADS = [
    "Vek", "Bru", "Sarn", "Dol", "Mir", "Osk", "Pet", "Rau", "Tam", "Ulv",
    "Wen", "Yar", "Zim", "Kor", "Lev", "Nes", "Gav", "Hult",
]
PERSON_TAILS = [
    "kers", "lov", "ova", "vek", "dahl", "rin", "sten", "mund", "vik",
    "gard", "sson", "berg", "wall", "quin",
]
MODEL_ROOTS = [
    "deneb", "qorv", "zelb", "dovak", "mirat", "sarv", "tuln", "veld",
    "wrex", "yalt", "krov", "plen", "ostr", "brun", "gavr",
]
MODEL_SUFFIX = ["ex", "ix", "on", "ar", "ia", "us"]
MODEL_TAGS = ["rl", "xt", "vl", "mini", "lite", "nano", "moe", "v2", "v3"]
MODEL_SIZES = ["350M", "1B", "1.3B", "2B", "3B", "7B", "8x2B"]
TOOL_TAILS = ["trace", "lock", "grid", "cast", "pipe", "kit", "gen", "sync", "ctl", "d"]

# 25 terms per persona by kind.
KIND_PLAN = [("project", 6), ("machine", 4), ("person", 5), ("model", 5), ("tool", 5)]


def is_fake(term: str, words: set[str]) -> bool:
    """No alpha token of the term (len >= 3) may be a dictionary word."""
    import re
    for tok in re.split(r"[^A-Za-z]+", term):
        if len(tok) < 3:
            continue
        t = tok.lower()
        if t in words or (t.endswith("s") and t[:-1] in words):
            return False
    return True


def _one(rng: random.Random, kind: str) -> str:
    if kind == "project":
        return rng.choice(CODE_HEADS) + rng.choice(CODE_TAILS)
    if kind == "machine":
        return rng.choice(MACH_HEADS) + rng.choice(MACH_TAILS)
    if kind == "person":
        name = rng.choice(PERSON_HEADS) + rng.choice(PERSON_TAILS)
        return name if rng.random() < 0.7 else name.lower()  # handle form
    if kind == "model":
        root = rng.choice(MODEL_ROOTS) + rng.choice(MODEL_SUFFIX)
        if rng.random() < 0.5:
            return f"{root}-{rng.choice(MODEL_TAGS)}"          # deneb-rl style
        return f"{root.capitalize()} {rng.choice(MODEL_SIZES)}"  # Qorvex 2B style
    if kind == "tool":
        return rng.choice(CODE_HEADS) + rng.choice(TOOL_TAILS)
    raise ValueError(kind)


def gen_terms(rng: random.Random, words: set[str], taken: set[str],
              plan=KIND_PLAN) -> list[dict]:
    """Fabricate terms for one persona. Mutates `taken` (global uniqueness)."""
    out = []
    for kind, n in plan:
        made = 0
        tries = 0
        while made < n:
            tries += 1
            if tries > 5000:
                raise RuntimeError(f"combinator pool exhausted for kind {kind}")
            term = _one(rng, kind)
            key = norm(term)
            if not key or key in taken or not is_fake(term, words):
                continue
            taken.add(key)
            out.append({"term": term, "kind": kind})
            made += 1
    return out
