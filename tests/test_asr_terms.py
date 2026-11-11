"""asr_terms: the BPE encode and the flat trie pa's TDT decode walks. On a toy vocabulary, and, when the pinned
Parakeet snapshot is on disk, against the reference builder's output (sentencepiece + asr_bias_v0's build_trie +
make_trie.flatten, cb446 settings) for a fixed list of public terms."""

import hashlib
import json
import os

import numpy as np
import pytest

from sidecar import asr_terms
from sidecar.pa_client import SNAP


def toy_vocabulary(size=1024):
    v = ["<unk>", "▁", "▁a", "ab", "▁ab"] + list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    return v + [f"<pad{i}>" for i in range(size - len(v))]


def layout(flat, vocab=1024):
    """nodes, edges, and whether the length is the one pa checks before taking a dictionary"""
    nodes, edges = int(flat[0]), int(flat[1])
    ok = len(flat) == 16 + 4 * nodes + 3 * edges + (vocab + 1) + 2 * (vocab + 8)
    return nodes, edges, ok


def test_encode_merges_the_lowest_id_first():
    v = toy_vocabulary()
    encode, unk = asr_terms.bpe_encoder(v)
    # "▁a" (id 2) merges before "ab" (id 3); then "▁ab" (id 4)
    assert encode("ab") == [v.index("▁ab")]
    assert encode("ba") == [v.index("▁"), v.index("b"), v.index("a")]
    assert encode("a b") == [v.index("▁a"), v.index("▁"), v.index("b")]
    assert unk in encode("a!")


def test_variants_are_case_forms_without_repeats():
    assert asr_terms.variants("vLLM") == ["vLLM", "vllm", "Vllm", "VLLM"]
    assert asr_terms.variants("abc") == ["abc", "Abc", "ABC"]


def test_trie_layout_and_canonical_index():
    v = toy_vocabulary()
    flat = asr_terms.flat_trie(v, ["ab", "Cd"])
    nodes, edges, ok = layout(flat)
    assert ok and nodes == edges + 1  # a tree
    assert flat[2:5].view(np.float32).tolist() == [4.0, 4.0, 6.0]
    node_rows = flat[16 : 16 + 4 * nodes].reshape(nodes, 4)
    terminal = {int(r[3]) for r in node_rows if r[2]}
    assert terminal == {0, 1}  # every terminal names its term's position in the list
    root = flat[16 + 4 * nodes + 3 * edges : 16 + 4 * nodes + 3 * edges + 1025]
    assert root[v.index("▁ab")] & 0x80000000  # boostable first piece


def test_terms_that_only_encode_with_unk_give_no_trie():
    assert asr_terms.flat_trie(toy_vocabulary(), ["!!", "??"]) is None


@pytest.mark.parametrize("size", [1024, 8192])
def test_trie_follows_the_vocabulary_size(size):
    # pa's NTOK is the vocabulary + blank (v2: 1025, v3: 8193): root, word_start and boundary scale with it
    v = toy_vocabulary(size)
    flat = asr_terms.flat_trie(v, ["ab", "Cd"])
    nodes, edges, ok = layout(flat, size)
    assert ok


@pytest.mark.skipif(
    not os.path.isfile(f"{SNAP}/config.json"), reason="Parakeet snapshot not downloaded"
)
def test_matches_the_reference_builder_on_the_real_vocabulary():
    with open(f"{SNAP}/config.json") as f:
        v = json.load(f)["joint"]["vocabulary"]
    encode, _ = asr_terms.bpe_encoder(v)
    # sentencepiece's ids for the same text
    assert encode("FlashAttention") == [237, 829, 44, 827, 847, 821, 821, 765]
    assert encode("vLLM") == [190, 865, 865, 858]
    terms = ["cuBLAS", "FlashAttention", "vLLM", "PyTorch", "Kneron", "yt-dlp"]
    terms += ["B200", "Claude Code"]
    flat = asr_terms.flat_trie(v, terms)
    assert len(flat) == 4103 and layout(flat)[2]
    assert hashlib.md5(flat.tobytes()).hexdigest() == "379ed5a8d3163d1f50db7199bf77ce76"
