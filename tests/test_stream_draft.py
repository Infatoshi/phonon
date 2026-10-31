import os

from sidecar.linereader import LineReader
from sidecar.stream_draft import AdaptiveProposer, DraftProposer, trace_speculative

TARGET = list(range(100, 160))  # the model's greedy output in these tests


def test_lookup_proposer_follows_its_source_and_realigns():
    src = [1, 2, 3, 4, 5, 6, 7, 8]
    p = DraftProposer([src])
    assert p([1], 3) == [2, 3, 4]
    # accepted 2, 3 and the model's own 9: pointer realigns on the last tokens
    assert p([1, 2, 3, 9], 3) == []
    assert p([1, 2, 3, 9, 5], 2) == [6, 7]


def test_perfect_draft_is_verified_in_one_fused_pass():
    p = AdaptiveProposer([TARGET, [1, 2, 3]], primary=True, first=63)
    info = trace_speculative(12, TARGET, p, 512, 200, fuse=True)
    # prompt tail and the whole draft in one forward, then the end of sequence check
    assert info["passes"][0] == 12 + len(TARGET)
    assert info["accepted"] == len(TARGET)
    assert len(info["passes"]) == 1


def test_wrong_draft_still_reproduces_the_target():
    wrong = TARGET[:20] + [7] * 5 + TARGET[25:]
    p = AdaptiveProposer([wrong], primary=True, first=31)
    info = trace_speculative(5, TARGET, p, 512, 200, fuse=True)
    assert info["accepted"] < len(TARGET)
    assert info["accepted"] >= 20


def test_no_draft_decodes_one_token_per_pass():
    p = AdaptiveProposer([[1, 2]], primary=False)
    info = trace_speculative(30, TARGET[:10], p, 512, 200, fuse=False)
    assert info["passes"][0] == 30
    assert info["accepted"] == 0
    assert len(info["passes"]) == 1 + 10


def test_budget_cuts_the_output():
    p = AdaptiveProposer([TARGET], primary=True)
    info = trace_speculative(4, TARGET, p, 512, 10, fuse=True)
    assert info["accepted"] <= 10


def test_ladder_resets_after_a_rejection():
    p = AdaptiveProposer([list(range(1000))], primary=False, cap_lookup=63)
    assert len(p([0], 512)) == 3
    assert len(p([0, 1, 2, 3, 4], 512)) == 31  # all 3 accepted: next rung
    assert len(p([0, 1, 2, 3, 4, 999], 512)) <= 3  # rejection: back to the first rung


def test_a_backend_ladder_sets_the_draft_sizes():
    p = AdaptiveProposer(
        [list(range(1000))], primary=False, cap_lookup=63, ladder=(7, 15, 31)
    )
    assert len(p([0], 512)) == 7
    assert len(p(list(range(9)), 512)) == 15  # all 7 accepted: next rung
    assert AdaptiveProposer.LADDER[0] == 3  # the class default is untouched


def test_line_reader_coalesces_partial_lines():
    r, w = os.pipe()
    reader = LineReader(r)
    os.write(w, b'{"command":"stream_commit","inputText":"a"}\n{"command":')
    assert reader.poll(0.0) == [{"command": "stream_commit", "inputText": "a"}]
    os.write(w, b'"run"}\nnot json\n')
    assert reader.poll(0.0) == [{"command": "run"}, "not json"]
    os.close(w)
    assert reader.poll(0.0) == []
    assert reader.eof
    os.close(r)
