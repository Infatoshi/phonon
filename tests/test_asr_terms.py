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


def toy_vocabulary():
    v = ["<unk>", "▁", "▁a", "ab", "▁ab"] + list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    return v + [f"<pad{i}>" for i in range(1024 - len(v))]


def layout(flat):
    """nodes, edges, and whether the length is the one pa checks before taking a dictionary"""
    nodes, edges = int(flat[0]), int(flat[1])
    ok = len(flat) == 16 + 4 * nodes + 3 * edges + 1025 + 2 * 1032
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


def test_vocabulary_size_is_checked():
    with pytest.raises(ValueError):
        asr_terms.flat_trie(toy_vocabulary()[:1000], ["ab"])


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


# ---- acoustic span rescoring: the collision guard, spoken forms, the candidate finder, the replace rule


def rich_vocabulary():
    """letters, digits and punctuation as single pieces, a bare ▁ word start, the toy vocabulary's size"""
    v = ["<unk>", "▁"] + list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'-"
    )
    return v + [f"<pad{i}>" for i in range(1024 - len(v))]


def aligned(text, vocab=None):
    """the greedy rows a decode would give for text (one frame per piece), as AlignedWords"""
    v = vocab or rich_vocabulary()
    encode, _ = asr_terms.bpe_encoder(v)
    rows = [(t, 1, i, 0.9) for i, t in enumerate(encode(text))]
    return rows, asr_terms.align_words(rows, v, len(rows) + 4)


def test_english_word_list_loads():
    words = asr_terms.english_words()
    assert len(words) > 100_000 and "feed" in words and "fluidaudio" not in words


def test_common_word_collision_only_flags_ordinary_lowercase_words():
    assert asr_terms.common_word_collision("feed")
    assert asr_terms.common_word_collision("cursor")
    assert not asr_terms.common_word_collision(
        "Cursor"
    )  # a capitalized spelling names something
    assert not asr_terms.common_word_collision("FluidAudio")
    assert not asr_terms.common_word_collision("fluid audio")  # not a single word
    assert not asr_terms.common_word_collision("kneron")  # not English


def test_english_guard_falls_back_to_empty_when_the_list_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(asr_terms, "_ENGLISH", None)
    monkeypatch.setattr(asr_terms, "_english_paths", lambda: [tmp_path / "missing.txt"])
    assert asr_terms.english_words() == frozenset()
    assert not asr_terms.common_word_collision("feed")
    monkeypatch.setattr(asr_terms, "_ENGLISH", None)


def test_trie_guard_drops_common_words_but_keeps_canonical_positions():
    v = toy_vocabulary()
    flat = asr_terms.flat_trie(v, ["feed", "ab"])
    nodes, _edges, ok = layout(flat)
    node_rows = flat[16 : 16 + 4 * nodes].reshape(nodes, 4)
    assert ok and {int(r[3]) for r in node_rows if r[2]} == {
        1
    }  # "ab" keeps index 1; "feed" is not in it
    unguarded = asr_terms.flat_trie(v, ["feed", "ab"], guard=False)
    nodes, _, _ = layout(unguarded)
    rows = unguarded[16 : 16 + 4 * nodes].reshape(nodes, 4)
    assert {int(r[3]) for r in rows if r[2]} == {0, 1}
    assert asr_terms.flat_trie(v, ["feed"]) is None  # only common words: no dictionary


def test_merge_vocabulary_dedupes_and_budgets():
    E = asr_terms.VocabEntry
    entries = [
        {"term": "cuBLAS", "spoken_forms": ["koo blas"]},
        E(
            "CUBLAS", ("cue blast",)
        ),  # a case-insensitive duplicate: its forms merge into the first
        "FluidAudio",
        "x",  # under 2 letters
    ]
    out = asr_terms.merge_vocabulary(
        entries, screen_terms=["Kneron", "fluidaudio", "feed", "H100"]
    )
    assert [e.term for e in out] == ["Kneron", "H100", "cuBLAS", "FluidAudio"]
    assert out[2].spoken_forms == ("koo blas", "cue blast") and not out[2].temporary
    assert (
        out[0].temporary and out[1].temporary
    )  # screen terms; "feed" (English) and the duplicate dropped
    big = asr_terms.merge_vocabulary(
        [f"Term{i}" for i in range(100)], [f"Screen{i}" for i in range(50)]
    )
    assert len(big) == 64
    assert (
        sum(e.temporary for e in big) == 32
        and big[0].term == "Screen0"
        and big[32].term == "Term0"
    )
    few = asr_terms.merge_vocabulary(
        ["Alpha1"], [f"Screen{i}" for i in range(50)], max_terms=8
    )
    assert [e.term for e in few][:5] == [
        "Screen0",
        "Screen1",
        "Screen2",
        "Screen3",
        "Alpha1",
    ]


def test_spoken_forms_replace_longest_span_first():
    entries = [
        asr_terms.VocabEntry("FluidAudio", ("fluid audio",)),
        asr_terms.VocabEntry("FluidAudioKit", ("fluid audio kit",)),
    ]
    words = "use Fluid audio kit and fluid audio.".split()
    reps = asr_terms.apply_spoken_forms(words, entries)
    assert reps == [
        asr_terms.Replacement(1, 4, "FluidAudioKit"),
        asr_terms.Replacement(5, 7, "FluidAudio"),
    ]
    assert asr_terms.apply_replacements(words, reps) == [
        "use",
        "FluidAudioKit",
        "and",
        "FluidAudio.",
    ]


def test_spoken_forms_ties_go_to_the_earliest_and_never_split():
    entries = [
        asr_terms.VocabEntry("cuBLAS", ("cue blast",)),
        asr_terms.VocabEntry("Liftoff", ("blast off",)),
    ]
    assert asr_terms.apply_spoken_forms("cue blast off".split(), entries) == [
        asr_terms.Replacement(0, 2, "cuBLAS")
    ]
    entries.append(asr_terms.VocabEntry("CBO", ("cue blast off",)))
    assert asr_terms.apply_spoken_forms("cue blast off".split(), entries) == [
        asr_terms.Replacement(0, 3, "CBO")
    ]


def test_spoken_forms_skip_tiny_forms_and_urlish_words():
    entries = [
        asr_terms.VocabEntry("X", ("x",)),  # under 2 letters
        asr_terms.VocabEntry("Q", ("a !",)),
        asr_terms.VocabEntry("FluidAudio", ("fluid audio",)),
    ]
    assert asr_terms.apply_spoken_forms(["x", "a", "!"], entries) == []
    for w in ("fluid.audio", "fluid/audio", "me@fluid", "fluid_audio", "c:\\fluid"):
        assert asr_terms.apply_spoken_forms([w, "audio"], entries) == []
    assert asr_terms.apply_spoken_forms(["(fluid", "audio)"], entries) == [
        asr_terms.Replacement(0, 2, "FluidAudio")
    ]
    assert asr_terms.apply_replacements(
        ["(fluid", "audio)"], [asr_terms.Replacement(0, 2, "FluidAudio")]
    ) == ["(FluidAudio)"]


def test_ordinary_speech_is_never_a_candidate():
    entries = [asr_terms.VocabEntry("FluidAudio"), asr_terms.VocabEntry("feed")]
    for text in (
        "feed all of",
        "we feed all of them",
        "fluid and audio",
        "the audio is fluid",
    ):
        assert asr_terms.find_rescore_candidates(text.split(), entries) == [], text


def test_candidates_from_letters_edits_and_sounds():
    E = asr_terms.VocabEntry
    # the same letters across words
    c = asr_terms.find_rescore_candidates(
        "I use fluid audio daily".split(), [E("FluidAudio")]
    )
    assert [(x.w_start, x.w_end, x.text, x.form) for x in c] == [
        (2, 4, "FluidAudio", "FluidAudio")
    ]
    # within 2 edits of a 6+ letter term
    c = asr_terms.find_rescore_candidates("ran neuron models".split(), [E("Kneron")])
    assert [(x.w_start, x.w_end, x.text) for x in c] == [(1, 2, "Kneron")]
    # consonant skeleton within one edit of a spoken form; the tightest span wins over a longer one
    c = asr_terms.find_rescore_candidates(
        "switched to QBlast now".split(), [E("cuBLAS", ("cue blast",))]
    )
    assert [(x.w_start, x.w_end, x.text, x.form) for x in c] == [
        (2, 3, "cuBLAS", "cue blast")
    ]
    assert (
        asr_terms.consonant_skeleton("QBlast")
        == asr_terms.consonant_skeleton("cue blast")
        == "kblst"
    )


def test_candidate_skips():
    E = asr_terms.VocabEntry
    find = asr_terms.find_rescore_candidates
    assert find(["pytorch"], [E("PyTorch")]) == []  # case only
    assert find(["PyTorch"], [E("PyTorch")]) == []  # already the term
    for w in ("PyTorches", "PyTorchs", "PyTorched", "PyTorching", "PyTorch's"):
        assert find([w], [E("PyTorch")]) == [], w  # inflections
    assert find(["kooda"], [E("CUDA")]) == []  # a short term without spoken forms
    assert [x.text for x in find(["coo", "da"], [E("CUDA", ("cooda",))])] == ["CUDA"]
    assert find(["see", "fluid.audio"], [E("FluidAudio")]) == []  # URL-ish
    assert find(["fluid", "audio/x"], [E("FluidAudio")]) == []
    assert (
        find("a b c d e".split(), [E("abcde12")]) == []
    )  # 5 words: past the 4-word limit
    assert [
        (x.w_start, x.w_end) for x in find("ab cd ef g".split(), [E("abcdefg")])
    ] == [(0, 4)]
    assert find("fluid audio".split(), [E("FluidAudio")], skip={1}) == []
    assert (
        find("validate with Pydantic".split(), [E("Pydantic")]) == []
    )  # never swallow a neighbour


def test_candidates_are_capped_and_carry_tokens_and_frames():
    v = rich_vocabulary()
    encode, _ = asr_terms.bpe_encoder(v)
    rows, words = aligned("so fluid, audio. Fluid audio and fluid audio", v)
    assert [w.text for w in words] == [
        "so",
        "fluid,",
        "audio.",
        "Fluid",
        "audio",
        "and",
        "fluid",
        "audio",
    ]
    c = asr_terms.find_rescore_candidates(
        words, [asr_terms.VocabEntry("FluidAudio")], encode=encode
    )
    assert [(x.w_start, x.w_end) for x in c] == [(1, 3), (3, 5), (6, 8)]
    first = c[0]
    # heard: "▁fluid," then "▁audio" without its trailing "."; frames: the first piece to the "." after the core
    assert first.heard == tuple(r[0] for r in rows[first.t_start : first.t_end])
    assert v[rows[first.t_end][0]] == "."
    assert (first.f_start, first.f_end) == (
        rows[first.t_start][2],
        rows[first.t_end][2],
    )
    assert first.cand == tuple(encode("FluidAudio"))
    assert (
        len(
            asr_terms.find_rescore_candidates(
                words, [asr_terms.VocabEntry("FluidAudio")], max_candidates=2
            )
        )
        == 2
    )


def test_align_words_frames():
    v = rich_vocabulary()
    rows, words = aligned("ab cd.", v)
    # rows: ▁ a b ▁ c d . at frames 0..6
    assert [(w.text, w.t_start, w.t_end, w.f_start, w.f_end) for w in words] == [
        ("ab", 0, 3, 0, 3),
        ("cd.", 3, 6, 3, 6),
    ]
    # a 0-duration last piece before punctuation on the same frame still gets one frame
    rows2 = [
        (v.index("▁"), 0, 0, 1.0),
        (v.index("a"), 0, 0, 1.0),
        (v.index("."), 1, 0, 1.0),
    ]
    (w,) = asr_terms.align_words(rows2, v, 10)
    assert (w.f_start, w.f_end, w.tokens) == (0, 1, (v.index("▁"), v.index("a")))
    # the window's last word runs past its last piece, within the window
    (w,) = asr_terms.align_words(rows2[:2], v, 10)
    assert w.f_end == 10
    (w,) = asr_terms.align_words(rows2[:2], v, 100, tail_frames=4)
    assert w.f_end == 5


def test_boost_levels(monkeypatch):
    monkeypatch.delenv("PHONON_VOCAB_BOOST", raising=False)
    assert asr_terms.boost_nats() == 2.5
    for level, nats in (
        ("off", 0.0),
        ("light", 2.0),
        ("normal", 2.5),
        ("strong", 4.0),
        ("STRONG", 4.0),
        ("bogus", 2.5),
    ):
        monkeypatch.setenv("PHONON_VOCAB_BOOST", level)
        assert asr_terms.boost_nats() == nats
    assert asr_terms.boost_nats("light") == 2.0


def test_should_replace_math():
    sr = asr_terms.should_replace
    assert sr(-1.0, -10.0, 4, 2.5)  # -10 + 10 >= -1
    assert not sr(-1.0, -11.5, 4, 2.5)
    assert sr(-3.0, -3.0, 2, 0.0)  # boost off: a tie goes to the term
    assert not sr(-3.0, -3.1, 2, 0.0)
    assert sr(-5.0, -2.0, 1)
    for bad in (float("-inf"), float("inf"), float("nan")):
        assert not sr(bad, -1.0, 3) and not sr(-1.0, bad, 3)


@pytest.mark.skipif(
    not os.path.isfile(f"{SNAP}/config.json"), reason="Parakeet snapshot not downloaded"
)
def test_real_vocabulary_writes_the_candidates():
    with open(f"{SNAP}/config.json") as f:
        v = json.load(f)["joint"]["vocabulary"]
    encode, unk = asr_terms.bpe_encoder(v)
    for t in ("FluidAudio", "cuBLAS", "cue blast", "Kneron"):
        assert unk not in encode(t)


def test_merge_vocabulary_unlimited_keeps_all_deduplicated_entries():
    entries = [f"Term{i}" for i in range(120)]
    screen = [f"Screen{i}" for i in range(10)]
    merged = asr_terms.merge_vocabulary(entries, screen, max_terms=None)
    assert len(merged) == 130
    assert [e.term for e in merged[:10]] == screen
    assert merged[-1].term == "Term119"
