import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_gym.grade import score_rollout, valid_answer

GOLD = {
    "persona": "p",
    "planted": [
        {"term": "thornmill", "kind": "project", "count": 5},
        {"term": "deneb-rl", "kind": "model", "count": 3},
        {"term": "Qorvex 2B", "kind": "model", "count": 2},
        {"term": "torvik", "kind": "machine", "count": 4},
    ],
    "manglings": [
        {"term": "deneb-rl", "mangled": "the neb rl", "diff": "phonetic", "count": 2},
    ],
}


def item(term, spoken=()):
    return {"term": term, "count": 1, "kind": "other",
            "spoken_forms": list(spoken), "evidence": "x"}


def test_normalized_matching_and_mangling_collapse():
    # "Thorn Mill" matches thornmill by norm; "the neb rl" alone recalls deneb-rl
    # via its injected mangling; "qorvex 2b" matches case/space-insensitively.
    answer = [item("Thorn Mill"), item("the neb rl"), item("qorvex 2b")]
    s = score_rollout(answer, GOLD, set(), "")
    assert s["planted_hits"] == 3
    assert s["planted_recall"] == 0.75
    assert s["mangling_recall"] == 1.0
    assert s["missed_planted"] == ["torvik"]
    assert s["precision_proxy"] == 1.0


def test_spoken_forms_collapse_onto_term():
    answer = [item("deneb-rl", spoken=["the neb rl"])]
    s = score_rollout(answer, GOLD, set(), "")
    assert s["mangling_recall"] == 1.0
    assert s["planted_hits"] == 1


def test_precision_proxy_tiers():
    answer = [item("thornmill"),        # planted
              item("kseq_read"),        # repo vocab
              item("weird_verbatim"),   # verbatim in tree
              item("hallucinated")]     # stray
    s = score_rollout(answer, GOLD, {"kseqread"}, "uses weird_verbatim here")
    assert s["precision_proxy"] == 0.75
    assert s["stray_terms"] == ["hallucinated"]


def test_pass_rule():
    good = [item(g["term"]) for g in GOLD["planted"]]
    s = score_rollout(good, GOLD, set(), "")
    assert s["pass"] and s["planted_recall"] == 1.0
    bad = [item(f"nope{i}") for i in range(10)]
    s2 = score_rollout(bad, GOLD, set(), "")
    assert not s2["pass"] and s2["precision_proxy"] == 0.0


def test_invalid_answers():
    assert not valid_answer(None)
    assert not valid_answer([])
    assert not valid_answer([{"count": 1}])
    assert not valid_answer({"term": "x"})
    s = score_rollout(None, GOLD, set(), "")
    assert not s["valid_json"] and not s["pass"]
