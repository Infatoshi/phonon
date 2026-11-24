"""Dictionary boosting for Metal Parakeet: the user's terms as the flat trie pa's TDT decode walks (PA_TRIE layout).

A port of asr_bias_v0's tdt_decode.build_trie (case variants), Pieces and night/asr bias/make_trie.flatten with the
cb446 settings (s_start 4, s_cont 4, budget 6, no blank start). Parakeet's tokenizer is a sentencepiece BPE model
whose piece scores fall with the piece id, so the encode runs here from the model's vocabulary and the sidecar needs
no tokenizer package; it matches sentencepiece token for token on every dictionary term and case variant checked.

Flat layout (uint32 words): header [nodes, edges, s_start, s_cont, budget (float bits), blank_start, 0 x 10]; nodes
{edge offset, edge count, terminal, canonical index (~0: none)}; edges {token, child, boostable} sorted by token
within a node; root[1025] (child of the root on that first piece | 1 << 31 if boostable, ~0: none); word_start[1032];
boundary[1032].
"""

from __future__ import annotations

import math
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

S_START, S_CONT, BUDGET = 4.0, 4.0, 6.0  # cb446
NONE = 0xFFFFFFFF


def bpe_encoder(vocabulary):
    """vocabulary (piece by id) -> encode(text) -> token ids, as the sentencepiece BPE model: nmt_nfkc, a dummy
    prefix, spaces as ▁, then the adjacent merge whose piece has the lowest id first (leftmost on ties); characters
    outside the vocabulary are <unk>."""
    rank = {p: i for i, p in enumerate(vocabulary)}
    unk = rank.get("<unk>", 0)

    def encode(text):
        sym = list("▁" + unicodedata.normalize("NFKC", text).replace(" ", "▁"))
        while len(sym) > 1:
            best, at = None, -1
            for i in range(len(sym) - 1):
                r = rank.get(sym[i] + sym[i + 1])
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            sym[at : at + 2] = [sym[at] + sym[at + 1]]
        return [rank.get(p, unk) for p in sym]

    return encode, unk


def variants(term):
    """The surface forms the recognizer may write for a term: as written, lower, Capitalized, UPPER."""
    out = []
    for v in (term, term.lower(), term[:1].upper() + term[1:].lower(), term.upper()):
        if v.strip() and v not in out:
            out.append(v)
    return out


def flat_trie(vocabulary, terms, guard=True):
    """The flat trie (np.uint32) for terms (canonical spellings; a committed span's canonical index is the term's
    position here), or None when no term encodes without <unk>. guard: leave out ordinary lowercase English words
    (common_word_collision), which would otherwise hijack ordinary speech; their positions stay reserved."""
    encode, unk = bpe_encoder(vocabulary)
    children, terminal, canon, bnode = [{}], [False], [-1], [True]
    for ci, term in enumerate(terms):
        if guard and common_word_collision(term):
            continue
        for v in variants(term):
            seq = encode(v)
            if not seq or unk in seq:
                continue
            n = 0
            for tok in seq:
                if tok not in children[n]:
                    children.append({})
                    terminal.append(False)
                    canon.append(-1)
                    bnode.append(False)
                    children[n][tok] = len(children) - 1
                n = children[n][tok]
                bnode[n] = True  # every case variant is boostable
            terminal[n] = True
            if canon[n] < 0:
                canon[n] = ci
    if len(children) == 1:
        return None
    nodes, edges = [], []
    for i, ch in enumerate(children):
        ch = sorted(ch.items())
        nodes.append(
            (len(edges), len(ch), int(terminal[i]), canon[i] if canon[i] >= 0 else NONE)
        )
        edges += [(t, c, int(bnode[c])) for t, c in ch]
    root = np.full(1025, NONE, dtype=np.uint32)
    for t, c in children[0].items():
        root[t] = c | (0x80000000 if bnode[c] else 0)
    word_start = np.array(
        [p.startswith("▁") for p in vocabulary] + [False] * 8, dtype=np.uint32
    )
    boundary = np.array(
        [
            p.startswith(("▁", "'")) or not any(c.isalnum() for c in p)
            for p in vocabulary
        ]
        + [True] * 8,
        dtype=np.uint32,
    )
    if len(word_start) != 1032:
        raise ValueError(
            f"vocabulary of {len(vocabulary)} pieces; the decode expects 1024"
        )
    head = np.zeros(16, dtype=np.uint32)
    head[0], head[1] = len(children), len(edges)
    head[2:5] = np.array([S_START, S_CONT, BUDGET], dtype=np.float32).view(np.uint32)
    return np.concatenate(
        [
            head,
            np.array(nodes, dtype=np.uint32).ravel(),
            np.array(edges, dtype=np.uint32).ravel(),
            root,
            word_start,
            boundary,
        ]
    )


# ---- acoustic span rescoring (Hex-style): the greedy decode runs unboosted; afterwards, spans of the transcript that
# look like a dictionary term are rescored against the window's audio (pa request 0xFFFFFFFE) and take the term only
# when its transducer log-probability, plus a per-piece bonus, reaches the heard words'. Exact spoken_forms matches
# are replaced without the audio. Ordinary English never turns into a term on spelling alone.

ASR_TERM_BUDGET = 64
URLISH = set("./@:\\_")
INFLECTIONS = ("s", "es", "ed", "ing")
BOOST_LEVELS = {"off": 0.0, "light": 2.0, "normal": 2.5, "strong": 4.0}
_ENGLISH = None


def _english_paths():
    here = Path(__file__).resolve().parent
    # the repo (sidecar/ next to assets/) and the app bundle (Resources/sidecar next to Resources/assets)
    paths = [here.parent / "assets" / "english_words.txt"]
    root = os.environ.get("PHONON_ROOT")
    if root:
        paths.insert(0, Path(root) / "assets" / "english_words.txt")
    paths += [here / "english_words.txt", here.parent / "english_words.txt"]
    return paths


def english_words():
    """The app's English word list (assets/english_words.txt, lowercase, one per line) as a frozenset; empty when
    the file is missing (the guard then lets every term through)."""
    global _ENGLISH
    if _ENGLISH is None:
        words = frozenset()
        for p in _english_paths():
            try:
                with open(p, encoding="utf-8") as f:
                    words = frozenset(w.strip() for w in f if w.strip())
                break
            except OSError:
                continue
        _ENGLISH = words
    return _ENGLISH


def common_word_collision(term):
    """An ordinary single lowercase English word ("feed", "cursor"): never a boost or rescore target by spelling
    alone (it would rewrite ordinary speech); a capitalized or mixed-case spelling signals a name and passes."""
    t = term.strip()
    return bool(t) and t.isalpha() and t == t.lower() and t in english_words()


@dataclass(frozen=True)
class VocabEntry:
    term: str
    spoken_forms: tuple[str, ...] = ()
    temporary: bool = False  # a screen term: this dictation only


@dataclass(frozen=True)
class AlignedWord:
    """One word of the greedy decode: its text (with any trailing punctuation), the row indices of its tokens
    [t_start, t_end) without trailing punctuation, those core tokens, all its tokens, and its encoder frames."""

    text: str
    t_start: int = 0
    t_end: int = 0
    f_start: int = 0
    f_end: int = 0
    tokens: tuple[int, ...] = ()
    all_tokens: tuple[int, ...] = ()


@dataclass(frozen=True)
class Replacement:
    w_start: int
    w_end: int
    text: str


@dataclass(frozen=True)
class RescoreCandidate:
    w_start: int
    w_end: int
    t_start: int
    t_end: int
    f_start: int
    f_end: int
    heard: tuple[int, ...]
    cand: tuple[int, ...]
    text: str  # canonical replacement
    form: str  # the spelling the candidate tokens encode (the term or one of its spoken forms)


def _alnum(s):
    return "".join(c for c in s.lower() if c.isalnum())


def _core(word):
    """A word without leading / trailing punctuation."""
    a, b = 0, len(word)
    while a < b and not word[a].isalnum():
        a += 1
    while b > a and not word[b - 1].isalnum():
        b -= 1
    return word[a:b]


def _urlish(word):
    return any(c in URLISH for c in _core(word))


def _entry(e):
    if isinstance(e, VocabEntry):
        return e
    if isinstance(e, str):
        return VocabEntry(e.strip())
    if isinstance(e, dict):
        forms = e.get("spoken_forms") or ()
        if isinstance(forms, str):
            forms = (forms,)
        return VocabEntry(
            str(e.get("term") or "").strip(),
            tuple(str(f).strip() for f in forms if str(f).strip()),
            bool(e.get("temporary", False)),
        )
    raise TypeError(f"vocabulary entry {e!r}")


def merge_vocabulary(entries, screen_terms=(), max_terms=ASR_TERM_BUDGET):
    """User entries (VocabEntry, str or {"term", "spoken_forms"}) and this dictation's screen terms -> one deduplicated
    list (case-insensitive on the term; a user entry wins over a screen term and merges its duplicates' spoken
    forms) within the ASR budget: screen terms first up to half the budget, then user entries, then the remaining
    screen terms. Terms with fewer than 2 letters or digits, and screen terms that are ordinary English words,
    are left out."""
    user, order = {}, []
    for e in entries or ():
        e = _entry(e)
        if len(_alnum(e.term)) < 2:
            continue
        k = e.term.lower()
        if k in user:
            old = user[k]
            forms = tuple(dict.fromkeys(old.spoken_forms + e.spoken_forms))
            user[k] = VocabEntry(old.term, forms, old.temporary)
        else:
            user[k] = VocabEntry(e.term, e.spoken_forms, False)
            order.append(k)
    screen, seen = [], set(user)
    for t in screen_terms or ():
        t = (t.term if isinstance(t, VocabEntry) else str(t)).strip()
        k = t.lower()
        if len(_alnum(t)) < 2 or k in seen or common_word_collision(t):
            continue
        seen.add(k)
        screen.append(VocabEntry(t, (), True))
    if max_terms is None:
        return screen + [user[k] for k in order]
    half = max_terms // 2
    out = screen[:half] + [user[k] for k in order] + screen[half:]
    return out[:max_terms]


def apply_spoken_forms(words, entries):
    """Exact spoken_forms matches on the transcript's words (str or AlignedWord): the replacements, longest span
    first, then earliest; a shorter form never splits a longer match. Case and edge punctuation are ignored;
    forms with fewer than 2 letters or digits and words that look like URLs, paths or emails never match."""
    texts = [w.text if isinstance(w, AlignedWord) else str(w) for w in words]
    cores = [None if _urlish(t) else _core(t).lower() for t in texts]
    forms = []
    for e in entries or ():
        e = _entry(e)
        for sf in e.spoken_forms:
            toks = [_core(x).lower() for x in sf.split()]
            if len(_alnum(sf)) < 2 or not toks or not all(toks):
                continue
            forms.append((toks, e.term))
    hits = []
    for toks, term in forms:
        n = len(toks)
        for i in range(len(texts) - n + 1):
            if cores[i : i + n] == toks:
                hits.append((-n, i, term))
    hits.sort(key=lambda h: (h[0], h[1]))
    used, out = set(), []
    for neg, i, term in hits:
        span = range(i, i - neg)
        if any(j in used for j in span):
            continue
        used.update(span)
        out.append(Replacement(i, i - neg, term))
    return sorted(out, key=lambda r: r.w_start)


def apply_replacements(words, reps):
    """words (str) with each replacement's span as one word: the first word's leading and the last word's trailing
    punctuation around the replacement text."""
    out, prev = [], 0
    for r in sorted(reps, key=lambda r: r.w_start):
        out += words[prev : r.w_start]
        first, last = words[r.w_start], words[r.w_end - 1]
        a = next((i for i, c in enumerate(first) if c.isalnum()), len(first))
        b = next(
            (i for i in range(len(last), 0, -1) if last[i - 1].isalnum()), len(last)
        )
        lead, trail = first[:a], last[b:]
        out.append(lead + r.text + trail)
        prev = r.w_end
    return out + words[prev:]


_DIGRAPHS = (
    ("ph", "f"),
    ("gh", "g"),
    ("ck", "k"),
    ("sh", "S"),
    ("ch", "C"),
    ("th", "T"),
    ("qu", "kw"),
    ("wh", "w"),
)


def consonant_skeleton(s):
    """A rough phonetic key: lowercase letters and digits, common digraphs folded (ph f, ck k, sh, ch, th), c / q k
    (c before e, i, y: s), z s, x ks, then vowels, y, h and w dropped and repeats collapsed."""
    s = "".join(c for c in s.lower() if c.isalnum())
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    out = []
    for i, c in enumerate(s):
        if c == "c":
            c = "s" if i + 1 < len(s) and s[i + 1] in "eiy" else "k"
        elif c == "q":
            c = "k"
        elif c == "z":
            c = "s"
        elif c == "x":
            out += ["k", "s"]
            continue
        if c in "aeiouyhw":
            continue
        out.append(c)
    return "".join(c for i, c in enumerate(out) if i == 0 or c != out[i - 1])


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _match_rank(span_text, span_norm, span_skel, form, form_norm, form_skel):
    """(rank, edits): rank 0, the same letters ignoring spaces / hyphens; 1, within 2 edits (forms of 6+ letters);
    2, consonant skeletons within 1 edit (forms of 5+ consonant sounds). None: no match, a case-only difference
    or an inflection of the form."""
    if span_text.lower() == form.lower():
        return None  # case-only (or already the form): not an acoustic question
    for suf in INFLECTIONS:
        if span_norm.endswith(suf) and span_norm[: -len(suf)] == form_norm:
            return None
    if span_norm == form_norm:
        return 0, 0
    if len(form_norm) >= 6 and abs(len(span_norm) - len(form_norm)) <= 2:
        d = levenshtein(span_norm, form_norm)
        if d <= 2:
            return 1, d
    if len(form_skel) >= 5 and abs(len(span_skel) - len(form_skel)) <= 1:
        d = levenshtein(span_skel, form_skel)
        if d <= 1:
            return 2, d
    return None


_CANDIDATE_INDEX_CACHE = (None, None)


def _candidate_index(entries):
    """Build (or reuse) length-bucketed lookup tables for find_rescore_candidates."""
    global _CANDIDATE_INDEX_CACHE
    norm_entries = tuple(_entry(e) for e in (entries or ()))
    if _CANDIDATE_INDEX_CACHE[0] == norm_entries:
        return _CANDIDATE_INDEX_CACHE[1]
    forms = []
    by_norm = {}
    by_norm_len = {}
    by_skel_len = {}
    for e in norm_entries:
        if not e.spoken_forms and (
            len(_alnum(e.term)) < 5 or common_word_collision(e.term)
        ):
            continue
        term_lower = e.term.lower()
        for f in (e.term, *e.spoken_forms):
            fn = _alnum(f)
            if len(fn) < 2:
                continue
            fk = consonant_skeleton(f)
            fi = len(forms)
            fn_set = frozenset(fn)
            fk_set = frozenset(fk)
            rec = (fi, f, f.lower(), fn, fn_set, fk, fk_set, e.term, term_lower)
            forms.append(rec)
            by_norm.setdefault(fn, []).append(rec)
            if len(fn) >= 6:
                by_norm_len.setdefault(len(fn), []).append(rec)
            if len(fk) >= 5:
                by_skel_len.setdefault(len(fk), []).append(rec)
    index = (forms, by_norm, by_norm_len, by_skel_len)
    _CANDIDATE_INDEX_CACHE = (norm_entries, index)
    return index


def find_rescore_candidates(words, entries, max_candidates=16, encode=None, skip=()):
    """Spans of 1-4 words (str or AlignedWord) that sound or spell close to a term or one of its spoken forms ->
    up to max_candidates non-overlapping RescoreCandidate (best match rank, then fewest edits, then longer span,
    then earlier first). A span already spelled as the term (in any case), or holding a word that is, is never a
    candidate.
    encode (text -> token ids) gives the candidate tokens; skip: word indices already replaced. Entries that are
    ordinary English words or under 5 letters without spoken forms are not candidates."""
    words = [w if isinstance(w, AlignedWord) else AlignedWord(str(w)) for w in words]
    forms, by_norm, by_norm_len, by_skel_len = _candidate_index(entries)
    if not forms:
        return []
    skip = set(skip)
    hits = []
    for i in range(len(words)):
        for n in range(1, 5):
            if i + n > len(words):
                break
            span = words[i : i + n]
            if any(j in skip for j in range(i, i + n)) or any(
                _urlish(w.text) or not _core(w.text) for w in span
            ):
                break
            cores = [_core(w.text).lower() for w in span]
            text = " ".join(_core(w.text) for w in span)
            text_lower = text.lower()
            sn = _alnum(text)
            sn_len = len(sn)
            sn_set = frozenset(sn)
            sk = consonant_skeleton(text)
            sk_len = len(sk)
            sk_set = frozenset(sk)
            seen_fi = set()
            # Rank 0: exact _alnum match
            for rec in by_norm.get(sn, ()):
                fi, f, f_lower, fn, _fn_set, fk, _fk_set, _term, term_lower = rec
                if text_lower == term_lower or (n > 1 and term_lower in cores):
                    continue
                if text_lower == f_lower:
                    continue
                seen_fi.add(fi)
                hits.append((0, 0, -n, i, fi))
            # Rank 1: edit distance <= 2 for forms with len >= 6
            if sn_len >= 4:
                for flen in range(max(6, sn_len - 2), sn_len + 3):
                    for rec in by_norm_len.get(flen, ()):
                        fi, f, f_lower, fn, fn_set, fk, _fk_set, _term, term_lower = rec
                        if fi in seen_fi:
                            continue
                        if len(sn_set - fn_set) > 2 or len(fn_set - sn_set) > 2:
                            continue
                        if text_lower == term_lower or (n > 1 and term_lower in cores):
                            continue
                        if text_lower == f_lower:
                            continue
                        if any(
                            sn.endswith(suf) and sn[: -len(suf)] == fn
                            for suf in INFLECTIONS
                        ):
                            continue
                        d = levenshtein(sn, fn)
                        if d <= 2:
                            seen_fi.add(fi)
                            hits.append((1, d, -n, i, fi))
            # Rank 2: consonant skeleton edit distance <= 1 for forms with skeleton len >= 5
            if sk_len >= 4:
                for klen in range(max(5, sk_len - 1), sk_len + 2):
                    for rec in by_skel_len.get(klen, ()):
                        fi, f, f_lower, fn, _fn_set, fk, fk_set, _term, term_lower = rec
                        if fi in seen_fi:
                            continue
                        if len(sk_set - fk_set) > 1 or len(fk_set - sk_set) > 1:
                            continue
                        if text_lower == term_lower or (n > 1 and term_lower in cores):
                            continue
                        if text_lower == f_lower:
                            continue
                        if any(
                            sn.endswith(suf) and sn[: -len(suf)] == fn
                            for suf in INFLECTIONS
                        ):
                            continue
                        d = levenshtein(sk, fk)
                        if d <= 1:
                            seen_fi.add(fi)
                            hits.append((2, d, -n, i, fi))
    hits.sort()
    used, out = set(), []
    cache = {}
    for _r, _d, neg, i, fi in hits:
        if len(out) >= max_candidates:
            break
        span = range(i, i - neg)
        if any(j in used for j in span):
            continue
        f, term = forms[fi][1], forms[fi][7]
        first, last = words[i], words[i - neg - 1]
        heard = (
            tuple(t for w in words[i : i - neg - 1] for t in w.all_tokens) + last.tokens
        )
        if encode is not None and f not in cache:
            cache[f] = tuple(encode(f))
        used.update(span)
        out.append(
            RescoreCandidate(
                i,
                i - neg,
                first.t_start,
                last.t_end,
                first.f_start,
                last.f_end,
                heard,
                cache.get(f, ()),
                term,
                f,
            )
        )
    return sorted(out, key=lambda c: c.w_start)


def align_words(rows, vocabulary, n_frames, durations=(0, 1, 2, 3, 4), tail_frames=16):
    """The greedy decode's rows (token, duration index, frame, confidence) -> AlignedWord list: a word starts at a
    ▁ piece; its core ends before trailing punctuation pieces; its frames run from its first token to the next
    token after its core (at least one past its last core token; the window's last word: up to tail_frames past
    its last token, within n_frames)."""
    out, starts = (
        [],
        [i for i, r in enumerate(rows) if i == 0 or vocabulary[r[0]].startswith("▁")],
    )
    for k, a in enumerate(starts):
        b = starts[k + 1] if k + 1 < len(starts) else len(rows)
        c = b
        while c > a and not any(ch.isalnum() for ch in vocabulary[rows[c - 1][0]]):
            c -= 1
        text = "".join(vocabulary[r[0]] for r in rows[a:b]).replace("▁", " ").strip()
        if c > a:
            last = rows[c - 1]
            if c < len(rows):
                f_end = rows[c][2]
            else:
                f_end = min(
                    n_frames, last[2] + max(durations[last[1]], 1) + tail_frames
                )
            f_end = max(f_end, last[2] + 1)
        else:
            f_end = rows[a][2]
        out.append(
            AlignedWord(
                text,
                a,
                c,
                rows[a][2],
                f_end,
                tuple(r[0] for r in rows[a:c]),
                tuple(r[0] for r in rows[a:b]),
            )
        )
    return out


def boost_nats(level=None):
    """The per-piece bonus a candidate gets in should_replace: PHONON_VOCAB_BOOST off 0, light 2, normal 2.5
    (default), strong 4; an unknown level is normal."""
    if level is None:
        level = os.environ.get("PHONON_VOCAB_BOOST", "normal")
    return BOOST_LEVELS.get(str(level).strip().lower(), BOOST_LEVELS["normal"])


def should_replace(score_heard, score_cand, n_pieces, boost_per_piece=2.5):
    """Take the term when both scores are finite and log P(cand) + bonus per candidate piece >= log P(heard)."""
    if not (math.isfinite(score_heard) and math.isfinite(score_cand)):
        return False
    return score_cand + boost_per_piece * n_pieces >= score_heard
