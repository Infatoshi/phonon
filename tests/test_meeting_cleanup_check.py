import json
from pathlib import Path

from tests.meeting_cleanup_check import check, combine, halves, parse_input

FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "meeting_cleanup_cases.json"
)


def test_the_reference_gives_the_fixture_verdicts_the_swift_checker_is_held_to():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) >= 8
    for case in cases:
        verdict, cleaned = check(parse_input(case["input"]), case["output"])
        assert verdict == case["verdict"], case["id"]
        if verdict == "ok":
            assert not any("TL;DR" in (t or "") for t in cleaned)


def test_a_kept_line_is_not_glued_onto_an_earlier_filler_line_of_the_same_speaker():
    # the same case as the Swift test of that name
    inp = parse_input(
        "Speaker 1: We should ship on Friday because the build is green and QA signed off.\n"
        "Speaker 2: Right.\n"
        "Speaker 2: I agree, Friday works for the whole team and for the release notes."
    )
    one = "Speaker 1: We should ship on Friday because the build is green and QA signed off."
    two = (
        "Speaker 2: I agree, Friday works for the whole team and for the release notes."
    )
    verdict, cleaned = check(inp, one + "\n" + two + "\nTL;DR: ship Friday.")
    assert verdict == "ok"
    assert cleaned[1] is None
    assert (
        cleaned[2]
        == "I agree, Friday works for the whole team and for the release notes."
    )
    assert check(inp, one + "\nSpeaker 2: Right.")[0] == "speaker line lost"
    assert check(inp, two + "\n" + one)[0] == "speaker order"
    twice = parse_input(
        "Speaker 1: Okay.\nSpeaker 1: Okay so the budget is fine for the next quarter."
    )
    assert check(twice, "Speaker 1: The budget is fine for the next quarter.") == (
        "ok",
        [None, "The budget is fine for the next quarter."],
    )


def test_the_second_pass_splits_at_the_line_nearest_half_the_words():
    inp = [("Speaker 1", "a b c d"), ("Speaker 2", "e f"), ("Speaker 1", "g h i j k l")]
    first, second = halves(inp)
    assert first == inp[:2] and second == inp[2:]
    # a tie takes the earlier cut
    assert halves([("Speaker 1", "a b"), ("Speaker 2", "c d")]) == (
        [("Speaker 1", "a b")],
        [("Speaker 2", "c d")],
    )
    assert halves(inp[:1]) is None
    assert combine("ok", "ok", "speaker order") == "ok"
    assert combine("ok", "content lost", "speaker order") == "mixed"
    assert combine("too long", "content lost", "speaker order") == "speaker order"
