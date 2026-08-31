import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_gym.common import load_dict_words, norm
from persona_gym.fabricate import gen_terms, is_fake


def test_unique_and_fake_across_personas():
    words = load_dict_words()
    taken: set[str] = set()
    all_terms = []
    for i in range(20):
        terms = gen_terms(random.Random(f"42:{i}"), words, taken)
        assert len(terms) == 25
        all_terms.extend(t["term"] for t in terms)
    norms = [norm(t) for t in all_terms]
    assert len(set(norms)) == len(norms) == 500
    for t in all_terms:
        assert is_fake(t, words), f"{t!r} contains a real word"
        assert norm(t) not in words, f"{t!r} is a dictionary word"


def test_deterministic():
    words = load_dict_words()
    a = gen_terms(random.Random("7:0"), words, set())
    b = gen_terms(random.Random("7:0"), words, set())
    assert a == b


def test_kinds_plan():
    words = load_dict_words()
    terms = gen_terms(random.Random(1), words, set())
    kinds = [t["kind"] for t in terms]
    assert kinds.count("project") == 6
    assert kinds.count("machine") == 4
    assert kinds.count("person") == 5
    assert kinds.count("model") == 5
    assert kinds.count("tool") == 5


def test_is_fake_rejects_real_words():
    words = {"thorn", "mill", "hello"}
    assert not is_fake("hello", words)
    assert not is_fake("Hello 2B", words)
    assert is_fake("thornmill", words)  # whole token is not a word
    assert is_fake("qorvex-rl", words)
