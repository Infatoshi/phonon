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

import unicodedata

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


def flat_trie(vocabulary, terms):
    """The flat trie (np.uint32) for terms (canonical spellings; a committed span's canonical index is the term's
    position here), or None when no term encodes without <unk>."""
    encode, unk = bpe_encoder(vocabulary)
    children, terminal, canon, bnode = [{}], [False], [-1], [True]
    for ci, term in enumerate(terms):
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
            p.startswith("▁") or not any(c.isalnum() for c in p) or p.startswith("'")
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
